"""
scripts/check_voice.py
----------------------
Diagnostic script for STT (faster-whisper) and TTS (gTTS) dependencies.
Run from the project root:
    python scripts/check_voice.py

Optionally place a sample audio file at:
    scripts/sample_audio.wav   (or .webm / .mp3)
to test live STT transcription.
"""

import sys
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Load .env ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    print("[.env] Loaded successfully.")
except ImportError:
    print("[.env] python-dotenv not installed — using system environment.")

SEP = "=" * 62


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. VOICE_ENABLED flag ─────────────────────────────────────────────────
section("1. VOICE_ENABLED config")
voice_enabled_raw = os.getenv("VOICE_ENABLED", "true").strip().lower()
voice_enabled = voice_enabled_raw not in {"false", "0", "no", "off"}
print(f"  VOICE_ENABLED env value : {os.getenv('VOICE_ENABLED', '<not set, default=true>')}")
print(f"  Resolved                : {'ENABLED' if voice_enabled else 'DISABLED'}")
if not voice_enabled:
    print("  ⚠  Voice is disabled. Text chat works normally.")
    print("     Set VOICE_ENABLED=true in .env to enable.\n")


# ── 2. faster-whisper ─────────────────────────────────────────────────────
section("2. faster-whisper (STT)")
try:
    import faster_whisper
    print(f"  [OK]   faster-whisper is installed (version: {getattr(faster_whisper, '__version__', 'unknown')})")
    has_whisper = True
except ImportError:
    print("  [FAIL] faster-whisper is NOT installed.")
    print("         Fix: pip install faster-whisper")
    has_whisper = False


# ── 3. gTTS ───────────────────────────────────────────────────────────────
section("3. gTTS (TTS)")
try:
    import gtts
    print(f"  [OK]   gTTS is installed (version: {getattr(gtts, '__version__', 'unknown')})")
    has_gtts = True
except ImportError:
    print("  [FAIL] gTTS is NOT installed.")
    print("         Fix: pip install gtts")
    has_gtts = False


# ── 4. ffmpeg ─────────────────────────────────────────────────────────────
section("4. ffmpeg (required for WebM/OGG decoding in faster-whisper)")
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"  [OK]   ffmpeg found at: {ffmpeg_path}")
    has_ffmpeg = True
else:
    print("  [WARN] ffmpeg NOT found on PATH.")
    print("         WAV files will still work, but WebM/OGG/M4A may fail.")
    print("         Fix: install ffmpeg and add it to PATH.")
    print("         Windows: winget install ffmpeg  OR  choco install ffmpeg")
    has_ffmpeg = False


# ── 5. TTS live test ──────────────────────────────────────────────────────
section("5. TTS live test (English → MP3)")
if has_gtts:
    try:
        from gtts import gTTS
        tmp_mp3 = os.path.join(tempfile.gettempdir(), "check_voice_test.mp3")
        tts = gTTS(text="Hello, this is a test of YojnaBot voice output.", lang="en", slow=False)
        tts.save(tmp_mp3)
        size = os.path.getsize(tmp_mp3)
        print(f"  [OK]   TTS generated MP3 at: {tmp_mp3}")
        print(f"         File size: {size} bytes")
        os.remove(tmp_mp3)
        tts_ok = True
    except Exception as exc:
        print(f"  [FAIL] TTS test failed: {exc}")
        print("         Check internet connection (gTTS requires network access).")
        tts_ok = False
else:
    print("  [SKIP] gTTS not installed — TTS test skipped.")
    tts_ok = False


# ── 6. STT live test (optional — needs sample audio file) ─────────────────
section("6. STT live test (optional)")
sample_paths = [
    os.path.join(ROOT, "scripts", "sample_audio.wav"),
    os.path.join(ROOT, "scripts", "sample_audio.webm"),
    os.path.join(ROOT, "scripts", "sample_audio.mp3"),
]
sample_path = next((p for p in sample_paths if os.path.isfile(p)), None)

if not sample_path:
    print("  [SKIP] No sample audio file found.")
    print(f"         To test STT, place a file at one of:")
    for p in sample_paths:
        print(f"           {p}")
elif not has_whisper:
    print("  [SKIP] faster-whisper not installed.")
else:
    print(f"  Testing STT on: {sample_path}")
    try:
        from faster_whisper import WhisperModel
        print("  Loading faster-whisper 'base' model (this may take ~30s on first run) …")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(sample_path, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        print(f"  [OK]   STT succeeded!")
        print(f"         Detected language : {info.language} (confidence: {info.language_probability:.2f})")
        print(f"         Transcribed text  : {text[:200] or '<empty — audio may be silent>'}")
    except Exception as exc:
        print(f"  [FAIL] STT test failed: {exc}")
        if not has_ffmpeg:
            print("         Hint: ffmpeg missing may cause decode failure for non-WAV files.")


# ── 7. Summary ────────────────────────────────────────────────────────────
section("7. Summary")
print(f"  VOICE_ENABLED  : {'✅ Yes' if voice_enabled else '❌ No (disabled in .env)'}")
print(f"  faster-whisper : {'✅ Installed' if has_whisper else '❌ Missing'}")
print(f"  gTTS           : {'✅ Installed' if has_gtts else '❌ Missing'}")
print(f"  ffmpeg         : {'✅ Found' if has_ffmpeg else '⚠  Not on PATH (WAV-only)'}")
print(f"  TTS live test  : {'✅ Passed' if tts_ok else '❌ Failed / Skipped'}")

overall = voice_enabled and has_whisper and has_gtts and tts_ok
print(f"\n  Overall voice status: {'✅ READY' if overall else '⚠  PARTIAL / DEGRADED'}")
print(f"{SEP}\n")
