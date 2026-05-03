import os
import uuid
import logging
from engine.llm_router import router

logger = logging.getLogger("services.voice")

# 1. Absolute Path Management
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_AUDIO_DIR = os.path.join(_BASE_DIR, "frontend", "static", "audio")

if not os.path.exists(STATIC_AUDIO_DIR):
    os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

whisper_model = None

LANG_TO_GTTS = {
    "en": "en", "hi": "hi", "gu": "gu", "mr": "mr", "ta": "ta", "te": "te",
    "kn": "kn", "ml": "ml", "bn": "bn", "pa": "pa", "or": "or", "as": "as",
    "ur": "ur", "ne": "ne", "si": "si", "sd": "sd",
}

def clean_stt_text(text: str, language_hint: str = "en") -> str:
    """
    Uses LLM to clean verbal fillers and artifacts from transcription.
    """
    if not text or len(text.strip()) < 5:
        return text

    prompt = f"""SYSTEM ROLE: TRANSCRIPTION CLEANER
You are an expert in cleaning Indian multilingual transcriptions. 
Remove verbal fillers (um, ah, matlab, basically), stuttering, and non-meaningful sounds.
Keep the original meaning and language. Return ONLY the cleaned text.

INPUT: "{text}"
CLEANED OUTPUT:"""

    try:
        cleaned = router.generate_text(prompt, "You clean noisy transcriptions.")
        if cleaned:
            return cleaned.strip().replace('"', '')
    except Exception as e:
        logger.warning(f"Voice: LLM cleaning failed: {e}")
    
    return text

def speech_to_text(audio_path: str, language_hint: str = "en") -> str:
    global whisper_model
    try:
        import whisper
        if whisper_model is None:
            whisper_model = whisper.load_model("base")
        
        whisper_lang = language_hint if language_hint and language_hint != "en" else None
        result = whisper_model.transcribe(audio_path, language=whisper_lang)
        
        text = result.get("text", "").strip()
        if text:
            return text
            
    except Exception as e:
        logger.warning(f"Voice: Whisper failed: {e}")

    # Fallback to Google
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            bcp47_map = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN", "ta": "ta-IN"}
            text = r.recognize_google(audio_data, language=bcp47_map.get(language_hint, "en-IN"))
            return text
    except Exception as e:
        logger.error(f"STT Exception: {e}")
        return ""

def text_to_speech(text: str, language_code: str = "en", input_type: str = "voice") -> str:
    """
    TTS generation — ONLY if input_type is 'voice' (production requirement).
    """
    if input_type != "voice":
        return ""

    if not text:
        return ""

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(STATIC_AUDIO_DIR, filename)
    gtts_lang = LANG_TO_GTTS.get(language_code, "en")

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(filepath)
        return f"/static/audio/{filename}"
    except Exception as e:
        logger.error(f"TTS Exception: {e}")
        return ""
