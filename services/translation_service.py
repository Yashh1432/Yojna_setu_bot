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
            logger.info(f"Loading translation model {model_name} on {self.device}")
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
            logger.error(f"Failed to load translation model {model_name}: {exc}")
            return False

    def _translate(self, text: str, model_name: str, source_tag: str, target_tag: str) -> str:
        if not text:
            return ""
        if source_tag == target_tag:
            return text
        cache_key = (model_name, source_tag, target_tag, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if not self._load(model_name):
            self._cache_set(cache_key, text)
            return text
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
            return translated
        except Exception as exc:
            logger.error(
                "Translation inference failed",
                extra={"model_name": model_name, "source_tag": source_tag, "target_tag": target_tag, "error": str(exc)},
            )
            self._cache_set(cache_key, text)
            return text

    def _cache_set(self, key: tuple[str, str, str, str], value: str) -> None:
        self._cache[key] = value
        if len(self._cache) > self.max_cache_entries:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)

    def to_english(self, text: str, source_lang: str) -> str:
        source_tag = LANG_TO_TAG.get(source_lang, "eng_Latn")
        return self._translate(text=text, model_name=self.indic_en_path, source_tag=source_tag, target_tag="eng_Latn")

    def from_english(self, text: str, target_lang: str) -> str:
        target_tag = LANG_TO_TAG.get(target_lang, "eng_Latn")
        return self._translate(text=text, model_name=self.en_indic_path, source_tag="eng_Latn", target_tag=target_tag)


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
        translated = _translator.to_english(text, source_lang)
        return translated.strip() if translated and translated.strip() else text
    except Exception as exc:
        logger.error(f"translate_to_english fallback due to error: {exc}")
        return text


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
        translated = _translator.from_english(text, target_lang)
        return translated.strip() if translated and translated.strip() else text
    except Exception as exc:
        logger.error(f"translate_from_english fallback due to error: {exc}")
        return text
