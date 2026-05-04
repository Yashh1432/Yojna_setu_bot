import logging
import os
import re
import uuid

logger = logging.getLogger("services.voice")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "frontend", "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

_model = None

LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "gu": "gu",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "bn": "bn",
    "mr": "mr",
    "ml": "ml",
    "pa": "pa",
    "or": "or",
    "as": "as",
    "ur": "ur",
}


def _resolve_language_code(language_code: str) -> str:
    return LANG_MAP.get(str(language_code or "en").strip().lower(), "en")


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def speech_to_text(audio_path: str, language_hint: str = "en") -> str:
    try:
        model = _get_model()
        lang = _resolve_language_code(language_hint)

        segments, _info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "").strip())
        return text.strip()
    except Exception as exc:
        logger.warning("STT failed for %s: %s", audio_path, exc)
        return ""


def clean_stt_text(text: str, language_hint: str = "en") -> str:
    del language_hint
    text = str(text or "")
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_speech(text: str, language_code: str = "en") -> str:
    try:
        safe_text = str(text or "").strip()
        if not safe_text:
            return ""

        safe_text = re.sub(r"\s+", " ", safe_text)[:1200]
        lang = _resolve_language_code(language_code)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        out_path = os.path.join(AUDIO_DIR, filename)

        from gtts import gTTS

        tts = gTTS(text=safe_text, lang=lang, slow=False)
        tts.save(out_path)

        return f"/static/audio/{filename}"
    except Exception as exc:
        logger.error("TTS failed for language %s: %s", language_code, exc)
        return ""
