from __future__ import annotations

import os
import time
from typing import Any

import warnings

from core.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

logger = get_logger("services.translation_service")

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForSeq2SeqLM = None  # type: ignore[assignment]


LANG_TO_TAG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
}

EN_TO_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDIC_TO_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"


class _IndicTranslator:
    def __init__(self) -> None:
        self.device = "cuda" if HAS_TRANSFORMERS and torch and torch.cuda.is_available() else "cpu"
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._cache: dict[tuple[str, str, str, str], str] = {}
        self._failed_until: dict[str, float] = {}
        self.local_files_only = os.getenv("INDICTRANS_LOCAL_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.hf_token = os.getenv("HF_TOKEN")
        self.fail_cooldown_sec = max(30, int(os.getenv("INDICTRANS_FAIL_COOLDOWN_SEC", "300")))
        self.max_cache_entries = max(128, int(os.getenv("INDICTRANS_CACHE_SIZE", "2048")))
        
        # Support for custom local paths
        self.en_indic_path = os.getenv("INDICTRANS_EN_INDIC_PATH", EN_TO_INDIC_MODEL)
        self.indic_en_path = os.getenv("INDICTRANS_INDIC_EN_PATH", INDIC_TO_EN_MODEL)

    def _load(self, model_name: str) -> bool:
        if model_name in self._models:
            return True
        now = time.monotonic()
        blocked_until = self._failed_until.get(model_name, 0.0)
        if blocked_until > now:
            return False
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available; translation will fallback to input text.")
            return False
        try:
            logger.info(f"Loading translation model {model_name} on {self.device} (local_files_only={self.local_files_only})")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                token=self.hf_token,
            ).to(self.device)
            self._tokenizers[model_name] = tokenizer
            self._models[model_name] = model
            self._failed_until.pop(model_name, None)
            return True
        except Exception as exc:
            self._failed_until[model_name] = time.monotonic() + self.fail_cooldown_sec
            error_msg = str(exc)
            
            logger.error(
                f"Failed to load translation model {model_name}\n"
                f"  local_files_only: {self.local_files_only}\n"
                f"  Error: {error_msg}"
            )
            
            # Detect JSON decode errors indicating corrupted huggingface cache
            if "Expecting value: line 1 column 1 (char 0)" in error_msg or "JSONDecodeError" in error_msg:
                logger.error(
                    f"CRITICAL: The Hugging Face cache for {model_name} appears to be corrupted or incomplete.\n"
                    "INSTRUCTIONS TO FIX CACHE CORRUPTION:\n"
                    "1. Delete the corrupted cache directory for this model.\n"
                    "   Usually located at: ~/.cache/huggingface/hub/models--ai4bharat--indictrans2-en-indic-1B (or indic-en-1B)\n"
                    "   Or on Windows: %USERPROFILE%\\.cache\\huggingface\\hub\\\n"
                    "2. Re-run the bot to download the model fresh.\n"
                    "3. Alternatively, manually download the model and set INDICTRANS_EN_INDIC_PATH and INDICTRANS_INDIC_EN_PATH to the local directory."
                )
                
            return False

    def _log_failure(self, *, source_lang: str, target_lang: str, text: str, error: str) -> None:
        preview = " ".join(str(text or "").split())[:80]
        logger.error(
            {
                "event": "translation_failed",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "preview": preview,
                "error": error[:160],
            }
        )

    def _translate(self, text: str, model_name: str, source_tag: str, target_tag: str) -> tuple[str, bool]:
        if not text:
            return "", False
        if source_tag == target_tag:
            return text, False
        cache_key = (model_name, source_tag, target_tag, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached, False
        if not self._load(model_name):
            self._cache_set(cache_key, text)
            self._log_failure(source_lang=source_tag, target_lang=target_tag, text=text, error="model_load_unavailable")
            return text, True
        try:
            tokenizer = self._tokenizers[model_name]
            model = self._models[model_name]
            tokenizer.src_lang = source_tag
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            forced_bos = tokenizer.convert_tokens_to_ids(target_tag)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=256,
                num_beams=4,
                repetition_penalty=1.2,
                early_stopping=True,
            )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            self._cache_set(cache_key, translated)
            return translated, False
        except Exception as exc:
            self._failed_until[model_name] = time.monotonic() + self.fail_cooldown_sec
            self._log_failure(source_lang=source_tag, target_lang=target_tag, text=text, error=str(exc))
            self._cache_set(cache_key, text)
            return text, True

    def _cache_set(self, key: tuple[str, str, str, str], value: str) -> None:
        self._cache[key] = value
        if len(self._cache) > self.max_cache_entries:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)

    def to_english(self, text: str, source_lang: str) -> str:
        translated, _ = self.to_english_with_meta(text, source_lang)
        return translated

    def to_english_with_meta(self, text: str, source_lang: str) -> tuple[str, dict[str, Any]]:
        source_tag = LANG_TO_TAG.get(source_lang, "eng_Latn")
        translated, failed = self._translate(text=text, model_name=self.indic_en_path, source_tag=source_tag, target_tag="eng_Latn")
        return translated, {"translation_failed": failed, "source_lang": source_lang, "target_lang": "en"}

    def from_english(self, text: str, target_lang: str) -> str:
        translated, _ = self.from_english_with_meta(text, target_lang)
        return translated

    def from_english_with_meta(self, text: str, target_lang: str) -> tuple[str, dict[str, Any]]:
        target_tag = LANG_TO_TAG.get(target_lang, "eng_Latn")
        translated, failed = self._translate(text=text, model_name=self.en_indic_path, source_tag="eng_Latn", target_tag=target_tag)
        return translated, {"translation_failed": failed, "source_lang": "en", "target_lang": target_lang}


_translator = _IndicTranslator()


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Use IndicTrans2 or configured Indic translation backend.
    Return English text.
    If source_lang == 'en', return text.
    If translation fails, return original text and log error.
    """
    if not text:
        return ""
    if source_lang == "en":
        return text
    try:
        translated, _ = _translator.to_english_with_meta(text, source_lang)
        return translated.strip() if translated and translated.strip() else text
    except Exception as exc:
        logger.error(f"translate_to_english fallback due to error: {exc}")
        return text


def translate_to_english_with_meta(text: str, source_lang: str) -> tuple[str, dict[str, Any]]:
    if not text:
        return "", {"translation_failed": False, "source_lang": source_lang, "target_lang": "en"}
    if source_lang == "en":
        return text, {"translation_failed": False, "source_lang": source_lang, "target_lang": "en"}
    try:
        translated, meta = _translator.to_english_with_meta(text, source_lang)
        clean = translated.strip() if translated and translated.strip() else text
        return clean, meta
    except Exception as exc:
        logger.error(f"translate_to_english_with_meta fallback due to error: {exc}")
        return text, {"translation_failed": True, "source_lang": source_lang, "target_lang": "en"}


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Use IndicTrans2 or configured Indic translation backend.
    Return target-language text.
    If target_lang == 'en', return text.
    If translation fails, return English text and log error.
    """
    if not text:
        return ""
    if target_lang == "en":
        return text
    try:
        translated, _ = _translator.from_english_with_meta(text, target_lang)
        return translated.strip() if translated and translated.strip() else text
    except Exception as exc:
        logger.error(f"translate_from_english fallback due to error: {exc}")
        return text


def translate_from_english_with_meta(text: str, target_lang: str) -> tuple[str, dict[str, Any]]:
    if not text:
        return "", {"translation_failed": False, "source_lang": "en", "target_lang": target_lang}
    if target_lang == "en":
        return text, {"translation_failed": False, "source_lang": "en", "target_lang": target_lang}
    try:
        translated, meta = _translator.from_english_with_meta(text, target_lang)
        clean = translated.strip() if translated and translated.strip() else text
        return clean, meta
    except Exception as exc:
        logger.error(f"translate_from_english_with_meta fallback due to error: {exc}")
        return text, {"translation_failed": True, "source_lang": "en", "target_lang": target_lang}
