"""
services/voice.py
-----------------
Speech-to-Text (STT) via faster-whisper
Text-to-Speech (TTS) via gTTS

Control via environment variables:
  VOICE_ENABLED=true/false   — set to false to disable all voice processing
                               (text chat is unaffected)

Dependencies:
  pip install faster-whisper gtts
  ffmpeg must be on PATH for webm/ogg/mp4 decoding (needed by faster-whisper)

To check dependency health:
  python scripts/check_voice.py
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from typing import Optional

from core.logger import get_logger

logger = get_logger("services.voice")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "frontend", "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Read VOICE_ENABLED from env (default True so existing behaviour is preserved)
_VOICE_ENABLED: bool = os.getenv("VOICE_ENABLED", "true").strip().lower() not in {
    "false", "0", "no", "off"
}

LANG_MAP = {
    "en": "en", "hi": "hi", "gu": "gu", "ta": "ta", "te": "te",
    "kn": "kn", "bn": "bn", "mr": "mr", "ml": "ml", "pa": "pa",
    "or": "or", "as": "as", "ur": "ur",
}

# ---------------------------------------------------------------------------
# Dependency checks (done once at import, never raises)
# ---------------------------------------------------------------------------
def _check_dep(name: str) -> bool:
    """Return True if package can be imported, False otherwise."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_HAS_FASTER_WHISPER = _check_dep("faster_whisper")
_HAS_GTTS = _check_dep("gtts")
_HAS_FFMPEG = shutil.which("ffmpeg") is not None

if not _HAS_FASTER_WHISPER:
    logger.warning(
        "faster-whisper is NOT installed — STT will be unavailable. "
        "Install with: pip install faster-whisper"
    )
if not _HAS_GTTS:
    logger.warning(
        "gTTS is NOT installed — TTS will be unavailable. "
        "Install with: pip install gtts"
    )
if not _HAS_FFMPEG:
    logger.warning(
        "ffmpeg is NOT found on PATH — WebM/OGG audio may fail to decode. "
        "Install ffmpeg and ensure it is on PATH."
    )
if not _VOICE_ENABLED:
    logger.info("VOICE_ENABLED=false — voice pipeline is disabled. Text chat works normally.")


# ---------------------------------------------------------------------------
# Whisper model singleton
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_load_failed_until: float = 0.0
_WHISPER_FAIL_COOLDOWN = 300.0   # 5 min — don't retry every request after a crash


def _get_whisper_model():
    global _whisper_model, _whisper_load_failed_until

    if _whisper_model is not None:
        return _whisper_model

    if not _HAS_FASTER_WHISPER:
        return None

    now = time.monotonic()
    if now < _whisper_load_failed_until:
        remaining = int(_whisper_load_failed_until - now)
        logger.debug(f"Whisper model in cooldown — retrying in {remaining}s")
        return None

    try:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper 'base' model on CPU …")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("faster-whisper model loaded successfully.")
        return _whisper_model
    except Exception as exc:
        _whisper_load_failed_until = time.monotonic() + _WHISPER_FAIL_COOLDOWN
        logger.error(
            f"Failed to load faster-whisper model: {exc}\n"
            "  STT will be unavailable until the cooldown expires.\n"
            "  Fix: ensure faster-whisper is correctly installed and model files are accessible."
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_lang(code: str) -> str:
    return LANG_MAP.get(str(code or "en").strip().lower(), "en")


# ---------------------------------------------------------------------------
# Public API  (signatures UNCHANGED)
# ---------------------------------------------------------------------------

def speech_to_text(audio_path: str, language_hint: str = "en") -> str:
    """
    Transcribe audio at *audio_path* using faster-whisper.
    Returns the transcribed text, or "" on failure.
    Never raises.
    """
    if not _VOICE_ENABLED:
        logger.info("STT skipped — VOICE_ENABLED=false")
        return ""

    logger.info(
        f"STT start | file={audio_path} | lang_hint={language_hint} "
        f"| exists={os.path.isfile(audio_path)} "
        f"| size_bytes={os.path.getsize(audio_path) if os.path.isfile(audio_path) else 'N/A'} "
        f"| ffmpeg={'yes' if _HAS_FFMPEG else 'MISSING'}"
    )

    if not os.path.isfile(audio_path):
        logger.error(f"STT failed — audio file does not exist: {audio_path}")
        return ""

    file_size = os.path.getsize(audio_path)
    if file_size < 100:
        logger.warning(f"STT skipped — audio file too small ({file_size} bytes), likely empty.")
        return ""

    model = _get_whisper_model()
    if model is None:
        logger.error("STT failed — faster-whisper model unavailable (see earlier logs).")
        return ""

    try:
        lang = _resolve_lang(language_hint)
        segments, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(
            seg.text.strip() for seg in segments
            if getattr(seg, "text", "").strip()
        )
        result = text.strip()

        if result:
            logger.info(
                f"STT success | lang={info.language} | confidence={info.language_probability:.2f} "
                f"| text_length={len(result)} | preview={result[:60]!r}"
            )
        else:
            logger.warning(
                f"STT returned empty text | lang={info.language} "
                f"| confidence={info.language_probability:.2f} — audio may be silent or unclear."
            )
        return result

    except Exception as exc:
        logger.error(
            f"STT exception | file={audio_path} | lang_hint={language_hint} | error={exc}"
        )
        return ""


def clean_stt_text(text: str, language_hint: str = "en") -> str:
    """Normalise whitespace/zero-width chars from STT output."""
    del language_hint
    text = str(text or "")
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_speech(text: str, language_code: str = "en") -> str:
    """
    Synthesize *text* to an MP3 file using gTTS.
    Returns the URL path to the audio file (e.g. /static/audio/tts_xxxx.mp3),
    or "" on failure.
    Never raises.
    The caller should still send the text response even if this returns "".
    """
    if not _VOICE_ENABLED:
        logger.info("TTS skipped — VOICE_ENABLED=false")
        return ""

    safe_text = str(text or "").strip()
    if not safe_text:
        logger.debug("TTS skipped — empty text input.")
        return ""

    if not _HAS_GTTS:
        logger.error("TTS failed — gTTS is not installed. Install with: pip install gtts")
        return ""

    safe_text = re.sub(r"\s+", " ", safe_text)[:1200]
    lang = _resolve_lang(language_code)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(AUDIO_DIR, filename)

    logger.info(
        f"TTS start | lang={lang} | text_length={len(safe_text)} "
        f"| output={filename}"
    )

    try:
        from gtts import gTTS
        tts = gTTS(text=safe_text, lang=lang, slow=False)
        tts.save(out_path)

        saved_size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        logger.info(
            f"TTS success | file={out_path} | size_bytes={saved_size} "
            f"| url=/static/audio/{filename}"
        )
        return f"/static/audio/{filename}"

    except Exception as exc:
        logger.error(
            f"TTS failed | lang={language_code} | text_preview={safe_text[:60]!r} | error={exc}\n"
            "  Text response will still be returned to the user."
        )
        # Clean up empty/partial file if it was created
        try:
            if os.path.isfile(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return ""
