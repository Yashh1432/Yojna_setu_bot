# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from core.logger import get_logger
from core.sanitizer import sanitize_text
from engine.engine import (
    classify_user_intent_llm,
    decide_next_question_llm,
    extract_profile_llm,
    infer_language_selection_llm,
    normalize_text_light,
)
from engine.eligibility import filter_schemes
from engine.orchestrator import load_scheme_dataset, recommend_schemes, summarize_benefit
from engine.validator import normalize_income_value, normalize_state_name
from models.users import user_model
from services.cache_service import (
    get_extraction_cache,
    get_response_cache,
    set_extraction_cache,
    set_response_cache,
)
from services.confidence_service import extraction_confidence
from services.translation_service import (
    translate_from_english,
    translate_from_english_with_meta,
    translate_to_english,
)

# Late import to avoid circular; used only for scheme card translation via LLM
def _llm_translate(text: str, target_lang: str) -> str:
    """Translate English text to target_lang using the LLM router.
    Falls back to the original text if the LLM router fails."""
    if not text or not text.strip() or target_lang == "en":
        return text
    from engine.llm_router import router as _router
    LANG_NAMES = {
        "hi": "Hindi", "gu": "Gujarati", "bn": "Bengali", "ta": "Tamil",
        "te": "Telugu", "kn": "Kannada", "mr": "Marathi", "ml": "Malayalam",
        "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
    }
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        f"Translate the following English government scheme text to {lang_name}. "
        f"Output ONLY the translated text, no explanation, no quotes.\n\n{text}"
    )
    sys_prompt = f"You are a government scheme translator. Translate accurately to {lang_name}. Output only the translated text."
    try:
        result = _router.generate_text(prompt, sys_prompt=sys_prompt)
        result = (result or "").strip()
        return result if result else text
    except Exception:
        return text


logger = get_logger("engine.state_manager")
# â”€â”€ Static translated UI templates (no LLM / no model needed) â”€â”€
STATIC_TRANSLATIONS: dict[str, dict[str, str]] = {
    "hi": {
        "language_set": "\u092d\u093e\u0937\u093e {lang_name} \u092e\u0947\u0902 \u0905\u092a\u0921\u0947\u091f \u0915\u0940 \u0917\u0908\u0964",
        "occupation": "\u0906\u092a\u0915\u093e \u092a\u0947\u0936\u093e \u0915\u094d\u092f\u093e \u0939\u0948?",
        "category": "\u0906\u092a\u0915\u094b \u0915\u093f\u0938 \u092a\u094d\u0930\u0915\u093e\u0930 \u0915\u0940 \u092f\u094b\u091c\u0928\u093e \u091a\u093e\u0939\u093f\u090f?",
        "state": "\u0906\u092a \u0915\u093f\u0938 \u0930\u093e\u091c\u094d\u092f \u092e\u0947\u0902 \u0930\u0939\u0924\u0947 \u0939\u0948\u0902?",
        "age": "\u0906\u092a\u0915\u0940 \u0909\u092e\u094d\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "annual_income": "\u0906\u092a\u0915\u0940 \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u092a\u093e\u0930\u093f\u0935\u093e\u0930\u093f\u0915 \u0906\u092f \u0915\u093f\u0924\u0928\u0940 \u0939\u0948?",
        "menu": "1 \u0930\u0940\u0938\u0947\u091f\n2 \u092d\u093e\u0937\u093e \u092c\u0926\u0932\u0947\u0902\n3 \u0938\u0939\u093e\u092f\u0924\u093e",
        "help": "\u0905\u092a\u0928\u0940 \u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u0938\u094d\u0935\u093e\u092d\u093e\u0935\u093f\u0915 \u0930\u0942\u092a \u0938\u0947 \u0938\u093e\u091d\u093e \u0915\u0930\u0947\u0902\u0964",
        "invalid_occupation_numeric": "\u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u093e \u092a\u0947\u0936\u093e \u0936\u092c\u094d\u0926\u094b\u0902 \u092e\u0947\u0902 \u0932\u093f\u0916\u0947\u0902, \u091c\u0948\u0938\u0947 \u0915\u093f\u0938\u093e\u0928, \u091b\u093e\u0924\u094d\u0930 \u092f\u093e \u0921\u094d\u0930\u093e\u0907\u0935\u0930\u0964",
    },
    "ta": {
        "language_set": "\u0bae\u0bca\u0bb4\u0bbf {lang_name} \u0b86\u0b95 \u0b85\u0bae\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1.",
        "occupation": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0ba4\u0bca\u0bb4\u0bbf\u0bb2\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?",
        "category": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9 \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd \u0ba4\u0bc7\u0bb5\u0bc8?",
        "state": "\u0ba8\u0bc0\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b8e\u0ba8\u0bcd\u0ba4 \u0bae\u0bbe\u0ba8\u0bbf\u0bb2\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd \u0bb5\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0bc0\u0bb0\u0bcd\u0b95\u0bb3\u0bcd?",
        "age": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0baf\u0ba4\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "annual_income": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b86\u0ba3\u0bcd\u0b9f\u0bc1 \u0b95\u0bc1\u0b9f\u0bc1\u0bae\u0bcd\u0baa \u0bb5\u0bb0\u0bc1\u0bae\u0bbe\u0ba9\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?",
        "menu": "1 \u0bb0\u0bc0\u0b9a\u0bc6\u0b9f\u0bcd\n2 \u0bae\u0bca\u0bb4\u0bbf\u0baf\u0bc8 \u0bae\u0bbe\u0bb1\u0bcd\u0bb1\u0bb5\u0bc1\u0bae\u0bcd\n3 \u0b89\u0ba4\u0bb5\u0bbf",
        "help": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0bbf\u0bb5\u0bb0\u0b99\u0bcd\u0b95\u0bb3\u0bc8 \u0b9a\u0bca\u0bb2\u0bcd\u0bb2\u0bc1\u0b99\u0bcd\u0b95\u0bb3\u0bcd.",
        "invalid_occupation_numeric": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0ba4\u0bca\u0bb4\u0bbf\u0bb2\u0bc8 \u0b8e\u0ba3\u0bcd\u0ba3\u0bbf\u0bb2\u0bcd \u0b85\u0bb2\u0bcd\u0bb2\u0bbe\u0bae\u0bb2\u0bcd \u0bb5\u0bbe\u0bb0\u0bcd\u0ba4\u0bcd\u0ba4\u0bc8\u0b95\u0bb3\u0bbf\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0bbf\u0b9f\u0bb5\u0bc1\u0bae\u0bcd, \u0b89\u0ba4\u0bbe\u0bb0\u0ba3\u0bae\u0bbe\u0b95 \u0bb5\u0bbf\u0bb5\u0b9a\u0bbe\u0baf\u0bbf, \u0bae\u0bbe\u0ba3\u0bb5\u0bb0\u0bcd \u0b85\u0bb2\u0bcd\u0bb2\u0ba4\u0bc1 \u0b93\u0b9f\u0bcd\u0b9f\u0bc1\u0ba8\u0bb0\u0bcd.",
    },
    "gu": {
        "language_set": "\u0aad\u0abe\u0ab7\u0abe {lang_name} \u0aae\u0abe\u0a82 \u0a85\u0aaa\u0aa1\u0ac7\u0a9f \u0a95\u0ab0\u0ac0.",
        "occupation": "\u0aa4\u0aae\u0abe\u0ab0\u0acb \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf \u0ab6\u0ac1\u0a82 \u0a9b\u0ac7?",
        "category": "\u0aa4\u0aae\u0aa8\u0ac7 \u0a95\u0acb\u0aa8\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe \u0a9c\u0acb\u0a88\u0a8f \u0a9b\u0ac7?",
        "state": "\u0aa4\u0aae\u0ac7 \u0a95\u0aaf\u0abe \u0ab0\u0abe\u0a9c\u0acd\u0aaf\u0aae\u0abe\u0a82 \u0ab0\u0ab9\u0acb \u0a9b\u0acb?",
        "age": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0a89\u0a82\u0aae\u0ab0 \u0a95\u0ac7\u0a9f\u0ab2\u0ac0 \u0a9b\u0ac7?",
        "annual_income": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0ab5\u0abe\u0ab0\u0acd\u0ab7\u0abf\u0a95 \u0a86\u0ab5\u0a95 \u0a95\u0ac7\u0a9f\u0ab2\u0ac0 \u0a9b\u0ac7?",
        "menu": "1 \u0ab0\u0ac0\u0ab8\u0ac7\u0a9f\n2 \u0aad\u0abe\u0ab7\u0abe \u0aac\u0aa6\u0ab2\u0acb\n3 \u0aae\u0aa6\u0aa6",
        "help": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0aae\u0abe\u0ab9\u0abf\u0aa4\u0ac0 \u0ab8\u0acd\u0ab5\u0abe\u0aad\u0abe\u0ab5\u0abf\u0a95 \u0ab0\u0ac0\u0aa4\u0ac7 \u0ab6\u0ac7\u0ab0 \u0a95\u0ab0\u0acb.",
        "invalid_occupation_numeric": "\u0a95\u0ac3\u0aaa\u0abe \u0a95\u0ab0\u0ac0\u0aa8\u0ac7 \u0aa4\u0aae\u0abe\u0ab0\u0acb \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf \u0ab6\u0aac\u0acd\u0aa6\u0acb\u0aae\u0abe\u0a82 \u0ab2\u0a96\u0acb, \u0a9c\u0ac7\u0ab5\u0abe \u0a95\u0ac7 \u0a96\u0ac7\u0aa1\u0ac2\u0aa4, \u0ab5\u0abf\u0aa6\u0acd\u0aaf\u0abe\u0ab0\u0acd\u0aa5\u0ac0 \u0a85\u0aa5\u0ab5\u0abe \u0aa1\u0acd\u0ab0\u0abe\u0a88\u0ab5\u0ab0.",
    },
    "kn": {
        "language_set": "\u0cad\u0cbe\u0cb7\u0cc6\u0caf\u0ca8\u0ccd\u0ca8\u0cc1 {lang_name} \u0c97\u0cc6 \u0cb8\u0cc6\u0c9f\u0ccd \u0cae\u0cbe\u0ca1\u0cb2\u0cbe\u0c97\u0cbf\u0ca6\u0cc6.",
        "occupation": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c89\u0ca6\u0ccd\u0caf\u0ccb\u0c97 \u0c8f\u0ca8\u0cc1?",
        "menu": "1 \u0cb0\u0cc0\u0cb8\u0cc6\u0c9f\u0ccd\n2 \u0cad\u0cbe\u0cb7\u0cc6 \u0cac\u0ca6\u0cb2\u0cbf\u0cb8\u0cbf\n3 \u0cb8\u0cb9\u0cbe\u0caf",
        "invalid_occupation_numeric": "\u0ca6\u0caf\u0cb5\u0cbf\u0c9f\u0ccd\u0c9f\u0cc1 \u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c89\u0ca6\u0ccd\u0caf\u0ccb\u0c97\u0cb5\u0ca8\u0ccd\u0ca8\u0cc1 \u0cb6\u0cac\u0ccd\u0ca6\u0c97\u0cb3\u0cb2\u0ccd\u0cb2\u0cbf \u0cb9\u0cc7\u0cb3\u0cbf, \u0c89\u0ca6\u0cbe\u0cb9\u0cb0\u0ca3\u0cc6\u0c97\u0cc6 \u0cb0\u0cc8\u0ca4, \u0cb5\u0cbf\u0ca6\u0ccd\u0caf\u0cbe\u0cb0\u0ccd\u0ca5\u0cbf \u0c85\u0ca5\u0cb5\u0cbe \u0ca1\u0ccd\u0cb0\u0cc8\u0cb5\u0cb0\u0ccd.",
    },
    "bn": {
        "language_set": "\u09ad\u09be\u09b7\u09be {lang_name} \u098f \u0986\u09aa\u09a1\u09c7\u099f \u0995\u09b0\u09be \u09b9\u09df\u09c7\u099b\u09c7\u0964",
        "occupation": "\u0986\u09aa\u09a8\u09be\u09b0 \u09aa\u09c7\u09b6\u09be \u0995\u09c0?",
        "menu": "1 \u09b0\u09bf\u09b8\u09c7\u099f\n2 \u09ad\u09be\u09b7\u09be \u09aa\u09b0\u09bf\u09ac\u09b0\u09cd\u09a4\u09a8\n3 \u09b8\u09be\u09b9\u09be\u09af\u09cd\u09af",
        "invalid_occupation_numeric": "\u09a6\u09df\u09be \u0995\u09b0\u09c7 \u09b8\u0982\u0996\u09cd\u09af\u09be\u09b0 \u09aa\u09b0\u09bf\u09ac\u09b0\u09cd\u09a4\u09c7 \u09aa\u09c7\u09b6\u09be \u09b6\u09ac\u09cd\u09a6\u09c7 \u09b2\u09bf\u0996\u09c1\u09a8, \u09af\u09c7\u09ae\u09a8 \u099a\u09be\u09b7\u09bf, \u09b6\u09bf\u0995\u09cd\u09b7\u09be\u09b0\u09cd\u09a5\u09c0 \u09ac\u09be \u09a1\u09cd\u09b0\u09be\u0987\u09ad\u09be\u09b0\u0964",
    },
    "mr": {
        "language_set": "\u092d\u093e\u0937\u093e {lang_name} \u092e\u0927\u094d\u092f\u0947 \u0905\u092a\u0921\u0947\u091f \u0915\u0947\u0932\u0940.",
        "occupation": "\u0924\u0941\u092e\u091a\u093e \u0935\u094d\u092f\u0935\u0938\u093e\u092f \u0915\u093e\u092f \u0906\u0939\u0947?",
        "menu": "1 \u0930\u093f\u0938\u0947\u091f\n2 \u092d\u093e\u0937\u093e \u092c\u0926\u0932\u093e\n3 \u092e\u0926\u0924",
        "invalid_occupation_numeric": "\u0915\u0943\u092a\u092f\u093e \u0906\u092a\u0932\u093e \u0935\u094d\u092f\u0935\u0938\u093e\u092f \u0936\u092c\u094d\u0926\u093e\u0924 \u0932\u093f\u0939\u093e, \u0909\u0926\u093e\u0939\u0930\u0923\u093e\u0930\u094d\u0925 \u0936\u0947\u0924\u0915\u0930\u0940, \u0935\u093f\u0926\u094d\u092f\u093e\u0930\u094d\u0925\u0940 \u0915\u093f\u0902\u0935\u093e \u0921\u094d\u0930\u093e\u092f\u0935\u094d\u0939\u0930.",
    },
    "ml": {
        "language_set": "\u0d2d\u0d3e\u0d37 {lang_name} \u0d06\u0d2f\u0d3f \u0d2e\u0d3e\u0d31\u0d4d\u0d31\u0d3f.",
        "occupation": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d24\u0d4a\u0d34\u0d3f\u0d7d \u0d0e\u0d28\u0d4d\u0d24\u0d3e\u0d23\u0d4d?",
        "category": "\u0d0f\u0d24\u0d4d \u0d24\u0d30\u0d02 \u0d2a\u0d26\u0d4d\u0d27\u0d24\u0d3f\u0d2f\u0d3e\u0d23\u0d4d \u0d35\u0d47\u0d23\u0d4d\u0d1f\u0d24\u0d4d?",
        "state": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d7e \u0d0f\u0d24\u0d4d \u0d38\u0d02\u0d38\u0d4d\u0d25\u0d3e\u0d28\u0d24\u0d4d\u0d24\u0d3e\u0d23\u0d4d \u0d24\u0d3e\u0d2e\u0d38\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d24\u0d4d?",
        "age": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d2a\u0d4d\u0d30\u0d3e\u0d2f\u0d02 \u0d0e\u0d24\u0d4d\u0d30?",
        "annual_income": "\u0d35\u0d3e\u0d7c\u0d37\u0d3f\u0d15 \u0d15\u0d41\u0d1f\u0d41\u0d02\u0d2c \u0d35\u0d30\u0d41\u0d2e\u0d3e\u0d28\u0d02 \u0d0e\u0d24\u0d4d\u0d30?",
        "menu": "1 \u0d31\u0d40\u0d38\u0d46\u0d31\u0d4d\u0d31\u0d4d\n2 \u0d2d\u0d3e\u0d37 \u0d2e\u0d3e\u0d31\u0d4d\u0d31\u0d41\u0d15\n3 \u0d38\u0d39\u0d3e\u0d2f\u0d02",
        "help": "\u0d24\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d35\u0d3f\u0d35\u0d30\u0d19\u0d4d\u0d19\u0d7e \u0d38\u0d4d\u0d35\u0d3e\u0d2d\u0d3e\u0d35\u0d3f\u0d15\u0d2e\u0d3e\u0d2f\u0d3f \u0d05\u0d2f\u0d15\u0d4d\u0d15\u0d42.",
        "invalid_occupation_numeric": "\u0d26\u0d2f\u0d35\u0d3e\u0d2f\u0d3f \u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d24\u0d4a\u0d34\u0d3f\u0d7d \u0d35\u0d3e\u0d15\u0d4d\u0d15\u0d41\u0d15\u0d33\u0d3f\u0d7d \u0d0e\u0d34\u0d41\u0d24\u0d41\u0d15, \u0d09\u0d26\u0d3e\u0d39\u0d30\u0d23\u0d24\u0d4d\u0d24\u0d3f\u0d28\u0d4d \u0d15\u0d7c\u0d37\u0d15\u0d7b, \u0d35\u0d3f\u0d26\u0d4d\u0d2f\u0d3e\u0d7c\u0d24\u0d4d\u0d25\u0d3f \u0d05\u0d32\u0d4d\u0d32\u0d46\u0d19\u0d4d\u0d15\u0d3f\u0d7d \u0d21\u0d4d\u0d30\u0d48\u0d35\u0d7c.",
    },
}


def _get_static(lang: str, key: str, **kwargs: str) -> str | None:
    """Look up a pre-translated static string. Returns None if not cached."""
    entry = STATIC_TRANSLATIONS.get(lang, {}).get(key)
    if entry and kwargs:
        try:
            entry = entry.format(**kwargs)
        except KeyError:
            pass
    return entry


def get_ui_text(lang: str | None, key: str, fallback: str) -> str:
    """Small wrapper for static UI text until full locale migration is safe."""
    cached = _get_static(_pick_lang(lang), key)
    return cached if cached else fallback


SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

_MENU_EN = "1 Reset\n2 Change Language\n3 Help"
MENU_TEXTS: dict[str, str] = {"en": _MENU_EN}
for _code, _cache in STATIC_TRANSLATIONS.items():
    if "menu" in _cache:
        MENU_TEXTS[_code] = _cache["menu"]
MENU_TEXT = _MENU_EN


LABELS = {
    "choose_language": (
        "Please choose your preferred language / \u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0940 \u092a\u0938\u0902\u0926\u0940\u0926\u093e \u092d\u093e\u0937\u093e \u091a\u0941\u0928\u0947\u0902:\n"
        "English, Hindi, Bengali, Gujarati, Kannada, Tamil, Telugu, Marathi, Malayalam, Punjabi, Odia, Assamese, Urdu"
    ),
    "language_set": "Language updated to {lang_name}.",
    "help": "Share your details naturally and I will help with matching schemes.",
    "no_match": "I could not find strong matches from the current dataset.",
    "response_reset": "Your profile is reset. Let us start again.",
    "invalid_menu_digit": "Please choose 1 Reset, 2 Change Language, or 3 Help - or type your scheme need in words.",
    "schemes_found": "Here are the matching schemes for you:",
}

# Localized version of schemes_found for all supported languages.
SCHEMES_FOUND_LABEL: dict[str, str] = {
    "en": "Here are the matching schemes for you:",
    "hi": "\u0906\u092a\u0915\u0947 \u0932\u093f\u090f \u092e\u093f\u0932\u0924\u0940 \u092f\u094b\u091c\u0928\u093e\u090f\u0902 \u092f\u0939\u093e\u0902 \u0939\u0948\u0902:",
    "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0abe \u0aae\u0abe\u0a9f\u0ac7 \u0aae\u0ab3\u0aa4\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93 \u0a87\u0a82 \u0a9b\u0ac7:",
    "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0baa\u0bca\u0bb0\u0bc1\u0ba4\u0bcd\u0ba4\u0bae\u0bbe\u0ba9 \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b87\u0ba4\u0bcb:",
    "te": "\u0c2e\u0c40\u0c15\u0c41 \u0c38\u0c30\u0c3f\u0c2a\u0c4b\u0c2f\u0c47 \u0c2a\u0c25\u0c15\u0c3e\u0c32\u0c41 \u0c07\u0c35\u0c3f:",
    "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u099c\u09a8\u09cd\u09af \u09ae\u09bf\u09b2\u09c7 \u09af\u09be\u0993\u09af\u09bc\u09be \u09aa\u09cd\u09b0\u0995\u09b2\u09cd\u09aa\u0997\u09c1\u09b2\u09bf:",
    "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae\u0ca8\u0ccd\u0ca8\u0cc1 \u0cb9\u0cca\u0c82\u0ca6\u0cc1\u0cb5 \u0caf\u0ccb\u0c9c\u0ca8\u0cc6\u0c97\u0cb3\u0cc1 \u0c87\u0cb2\u0ccd\u0cb2\u0cbf:\u0cb5\u0cc6:",
    "mr": "\u0924\u0941\u092e\u091a\u094d\u092f\u093e\u0938\u093e\u0920\u0940 \u091c\u0941\u0933\u0923\u093e\u0931\u094d\u092f\u093e \u092f\u094b\u091c\u0928\u093e \u0907\u0925\u0947 \u0906\u0939\u0947\u0924:",
    "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d4d\u0d15\u0d4d\u0d15\u0d4d \u0d38\u0d39\u0d3e\u0d2f\u0d15\u0d30\u0d2e\u0d3e\u0d2f \u0d2a\u0d26\u0d4d\u0d27\u0d24\u0d3f\u0d15\u0d7e \u0d07\u0d35\u0d3f\u0d1f\u0d46\u0d2f\u0d41\u0d23\u0d4d\u0d1f\u0d4d:",
    "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a47 \u0a32\u0a08 \u0a2e\u0a47\u0a32 \u0a16\u0a3e\u0a02\u0a26\u0a40\u0a06\u0a02 \u0a2f\u0a4b\u0a1c\u0a28\u0a3e\u0a35\u0a3e\u0a02 \u0a07\u0a71\u0a25\u0a47 \u0a39\u0a28:",
    "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b2a\u0b3e\u0b07\u0b01 \u0b2e\u0b3f\u0b33\u0b41\u0b25\u0b3f\u0b2c\u0b3e \u0b2f\u0b4b\u0b1c\u0b28\u0b3e\u0b17\u0b41\u0b21\u0b3c\u0b3f\u0b15 \u0b0f\u0b20\u0b3e\u0b30\u0b47 \u0b05\u0b1b\u0b28\u0b4d\u0b24\u0b3f:",
    "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u09ac\u09be\u09ac\u09c7 \u09ae\u09bf\u09b2\u09be \u0986\u0981\u099a\u09a8 \u0987\u09df\u09be\u09a4 \u0986\u099b\u09c7:",
    "ur": "\u0622\u067e \u06a9\u06d2 \u0644\u06cc\u06d2 \u0645\u0646\u0627\u0633\u0628 \u0627\u0633\u06a9\u06cc\u0645\u06cc\u06ba \u06cc\u06c1\u0627\u06ba \u06c1\u06cc\u06ba:",
}

FOLLOWUP_QUESTIONS = {
    "category": "Which type of scheme do you need (education, agriculture, health, employment, housing, finance, women)?",
    "state": "Which state do you live in?",
    "age": "What is your age?",
    "annual_income": "What is your approximate annual family income in rupees?",
    "occupation": "What is your occupation?",
    "education_level": "What is your current education level?",
    "caste_category": "What is your caste category (General, OBC, SC, ST)?",
    "gender": "What is your gender (male, female, or other)?",
    "academic_percentage": "What is your latest academic percentage?",
    "bpl_status": "Do you have a BPL card? (yes/no)",
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HARDCODED QUESTION MAP â€” NO LLM GENERATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
QUESTION_MAP = {
    "category": {
        "en": "Which type of scheme do you need? (education, health, agriculture, employment, housing, finance, women & child)",
        "hi": "\u0906\u092a\u0915\u094b \u0915\u093f\u0938 \u092a\u094d\u0930\u0915\u093e\u0930 \u0915\u0940 \u092f\u094b\u091c\u0928\u093e \u091a\u093e\u0939\u093f\u090f? (\u0936\u093f\u0915\u094d\u0937\u093e, \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f, \u0915\u0943\u0937\u093f, \u0930\u094b\u091c\u0917\u093e\u0930, \u0906\u0935\u093e\u0938, \u0935\u093f\u0924\u094d\u0924, \u092e\u0939\u093f\u0932\u093e \u090f\u0935\u0902 \u092c\u093e\u0932)",
        "gu": "\u0aa4\u0aae\u0aa8\u0ac7 \u0a95\u0aaf\u0abe \u0aaa\u0acd\u0ab0\u0a95\u0abe\u0ab0\u0aa8\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe \u0a9c\u0acb\u0a88\u0a8f \u0a9b\u0ac7? (\u0ab6\u0abf\u0a95\u0acd\u0ab7\u0aa3, \u0a86\u0ab0\u0acb\u0a97\u0acd\u0aaf, \u0a95\u0ac3\u0ab7\u0abf, \u0ab0\u0acb\u0a9c\u0a97\u0abe\u0ab0, \u0a86\u0ab5\u0abe\u0ab8, \u0aa8\u0abe\u0aa3\u0abe\u0a82, \u0aae\u0ab9\u0abf\u0ab2\u0abe \u0a85\u0aa8\u0ac7 \u0aac\u0abe\u0ab3\u0a95)",
        "kn": "\u0ca8\u0cbf\u0cae\u0c97\u0cc6 \u0caf\u0cbe\u0cb5 \u0cb0\u0cc0\u0ca4\u0cbf\u0caf \u0caf\u0ccb\u0c9c\u0ca8\u0cc6 \u0cac\u0cc7\u0c95\u0cc1? (\u0cb6\u0cbf\u0c95\u0ccd\u0cb7\u0ca3, \u0c86\u0cb0\u0ccb\u0c97\u0ccd\u0caf, \u0c95\u0cc3\u0cb7\u0cbf, \u0c89\u0ca6\u0ccd\u0caf\u0ccb\u0c97, \u0cb5\u0cb8\u0ca4\u0cbf, \u0cb9\u0ca3\u0c95\u0cbe\u0cb8\u0cc1, \u0cae\u0cb9\u0cbf\u0cb3\u0cc6 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0cae\u0c95\u0ccd\u0c95\u0cb3\u0cc1)",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b8e\u0ba8\u0bcd\u0ba4 \u0bb5\u0b95\u0bc8\u0baf\u0bbe\u0ba9 \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd? (\u0b95\u0bb2\u0bcd\u0bb5\u0bbf, \u0b9a\u0bc1\u0b95\u0bbe\u0ba4\u0bbe\u0bb0\u0bae\u0bcd, \u0bb5\u0bc7\u0bb3\u0bbe\u0ba3\u0bcd\u0bae\u0bc8, \u0bb5\u0bc7\u0bb2\u0bc8\u0bb5\u0bbe\u0baf\u0bcd\u0baa\u0bcd\u0baa\u0bc1, \u0bb5\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1\u0bb5\u0b9a\u0ba4\u0bbf, \u0ba8\u0bbf\u0ba4\u0bbf, \u0baa\u0bc6\u0ba3\u0bcd\u0b95\u0bb3\u0bcd \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0b95\u0bc1\u0bb4\u0ba8\u0bcd\u0ba4\u0bc8)",
        "te": "\u0c2e\u0c40\u0c15\u0c41 \u0c0f \u0c30\u0c15\u0c2e\u0c48\u0c28 \u0c2a\u0c25\u0c15\u0c02 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f? (\u0c35\u0c3f\u0c26\u0c4d\u0c2f, \u0c06\u0c30\u0c4b\u0c17\u0c4d\u0c2f\u0c02, \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f\u0c02, \u0c09\u0c2a\u0c3e\u0c27\u0c3f, \u0c17\u0c43\u0c39\u0c28\u0c3f\u0c30\u0c4d\u0cae\u0cbe\u0ca3, \u0c06\u0c30\u0c4d\u0c25\u0c3f\u0c15, \u0c2e\u0c39\u0c3f\u0c33\u0c3e \u0c2e\u0c30\u0c3f\u0c2f\u0c41 \u0c36\u0c3f\u0c36\u0c41)",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u0995\u09cb\u09a8 \u09a7\u09b0\u09a8\u09c7\u09b0 \u09aa\u09cd\u09b0\u0995\u09b2\u09cd\u09aa \u09a6\u09b0\u0995\u09be\u09b0? (\u09b6\u09bf\u0995\u09cd\u09b7\u09be, \u09b8\u09cd\u09ac\u09be\u09b8\u09cd\u09a5\u09cd\u09af, \u0995\u09c3\u09b7\u09bf, \u0995\u09b0\u09cd\u09ae\u09b8\u0982\u09b8\u09cd\u09a5\u09be\u09a8, \u0986\u09ac\u09be\u09b8\u09a8, \u0985\u09b0\u09cd\u09a5, \u09ae\u09b9\u09bf\u09b2\u09be \u0993 \u09b6\u09bf\u09b6\u09c1)",
        "mr": "\u0924\u0941\u092e\u094d\u0939\u093e\u0932\u093e \u0915\u094b\u0923\u0924\u094d\u092f\u093e \u092a\u094d\u0930\u0915\u093e\u0930\u091a\u0940 \u092f\u094b\u091c\u0928\u093e \u0939\u0935\u0940 \u0906\u0939\u0947? (\u0936\u093f\u0915\u094d\u0937\u0923, \u0906\u0930\u094b\u0917\u094d\u092f, \u0936\u0947\u0924\u0940, \u0930\u094b\u091c\u0917\u093e\u0930, \u0917\u0943\u0939\u0928\u093f\u0930\u094d\u092e\u093e\u0923, \u0935\u093f\u0924\u094d\u0924, \u092e\u0939\u093f\u0932\u093e \u0935 \u092c\u093e\u0932)",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d7e\u0d15\u0d4d\u0d15\u0d4d \u0d0f\u0d24\u0d4d \u0d24\u0d30\u0d24\u0d4d\u0d24\u0d3f\u0d32\u0d41\u0d33\u0d4d\u0d33 \u0d2a\u0d26\u0d4d\u0d27\u0d24\u0d3f \u0d35\u0d47\u0d23\u0d02? (\u0d35\u0d3f\u0d26\u0ccd\u0caf\u0cbe\u0d2d\u0d4d\u0d2f\u0d3e\u0d38\u0d02, \u0d06\u0d30\u0d4b\u0d17\u0d4d\u0d2f\u0d02, \u0d15\u0d43\u0d37\u0d3f, \u0d24\u0d4a\u0d34\u0d3f\u0d7d, \u0d2d\u0d35\u0d28\u0d02, \u0d27\u0d28\u0d15\u0d3e\u0d30\u0d4d\u0d2f\u0d02, \u0d35\u0d28\u0d3f\u0d24 \u0d36\u0d3f\u0d36\u0d41)",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a28\u0a42\u0a70 \u0a15\u0a3f\u0a38 \u0a15\u0a3f\u0a38\u0a2e \u0a26\u0a40 \u0a2f\u0a4b\u0a1c\u0a28\u0a3e \u0a1a\u0a3e\u0a39\u0a40\u0a26\u0a40 \u0a39\u0a48? (\u0a38\u0a3f\u0a71\u0a16\u0a3f\u0a06, \u0a38\u0a3f\u0a39\u0a24, \u0a16\u0a47\u0a24\u0a40\u0a2c\u0a3e\u0a5c\u0a40, \u0a30\u0a4b\u0a1c\u0a3c\u0a17\u0a3e\u0a30, \u0a30\u0a3f\u0a39\u0a3e\u0a07\u0a38\u0a3c, \u0a35\u0a3f\u0a71\u0a24, \u0a14\u0a30\u0a24\u0a3e\u0a02 \u0a05\u0a24\u0a47 \u0a2c\u0a71\u0a1a\u0a47)",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15\u0b41 \u0b15\u0b47\u0b09\u0b01 \u0b2a\u0b4d\u0b30\u0b15\u0b3e\u0b30\u0b30 \u0b2f\u0b4b\u0b1c\u0b28\u0b3e \u0b26\u0b30\u0b15\u0b3e\u0b30? (\u0b36\u0b3f\u0b15\u0b4d\u0b37\u0b3e, \u0b38\u0b4d\u0b71\u0b3e\u0b38\u0b4d\u0b25\u0b4d\u0b5f, \u0b15\u0b43\u0b37\u0b3f, \u0b28\u0b3f\u0b2f\u0b41\u0b15\u0b4d\u0b24\u0b3f, \u0b17\u0b43\u0b39, \u0b05\u0b30\u0b4d\u0b25, \u0b2e\u0b39\u0b3f\u0b33\u0b3e \u0b13 \u0b36\u0b3f\u0b36\u0b41)",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u0995 \u0995\u09bf \u09a7\u09f0\u09a3\u09f0 \u0986\u0981\u099a\u09a8\u09bf \u09b2\u09be\u0997\u09c7? (\u09b6\u09bf\u0995\u09cd\u09b7\u09be, \u09b8\u09cd\u09ac\u09be\u09b8\u09cd\u09a5\u09cd\u09af, \u0995\u09c3\u09b7\u09bf, \u09a8\u09bf\u09af\u09c1\u0995\u09cd\u09a4\u09bf, \u0997\u09c3\u09b9, \u09ac\u09bf\u09a4\u09cd\u09a4, \u09ae\u09b9\u09bf\u09b2\u09be \u0986\u09f0\u09c1 \u09b6\u09bf\u09b6\u09c1)",
        "ur": "\u0622\u067e \u06a9\u0648 \u06a9\u0633 \u0642\u0633\u0645 \u06a9\u06cc \u0627\u0633\u06a9\u06cc\u0645 \u0686\u0627\u06c1\u0626\u06d2\u061f (\u062a\u0639\u0644\u06cc\u0645\u060c \u0635\u062d\u062a\u060c \u0632\u0631\u0627\u0639\u062a\u060c \u0631\u0648\u0632\u06af\u0627\u0631\u060c \u0631\u06c1\u0627\u0626\u0634\u060c \u0645\u0627\u0644\u06cc\u0627\u062a\u060c \u062e\u0648\u0627\u062a\u06cc\u0646 \u0627\u0648\u0631 \u0628\u0686\u06d2)",
    },
    "state": {
        "en": "Which state do you live in?",
        "hi": "\u0906\u092a \u0915\u093f\u0938 \u0930\u093e\u091c\u094d\u092f \u092e\u0947\u0902 \u0930\u0939\u0924\u0947 \u0939\u0948\u0902?",
        "gu": "\u0aa4\u0aae\u0ac7 \u0a95\u0aaf\u0abe \u0ab0\u0abe\u0a9c\u0acd\u0aaf\u0aae\u0abe\u0a82 \u0ab0\u0ab9\u0acb \u0a9b\u0acb?",
        "kn": "\u0ca8\u0cc0\u0cb5\u0cc1 \u0caf\u0cbe\u0cb5 \u0cb0\u0cbe\u0c9c\u0ccd\u0caf\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf \u0cb5\u0cbe\u0cb8\u0cbf\u0cb8\u0cc1\u0ca4\u0ccd\u0ca4\u0cbf\u0ca6\u0ccd\u0ca6\u0cc0\u0cb0\u0cbf?",
        "ta": "\u0ba8\u0bc0\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b8e\u0ba8\u0bcd\u0ba4 \u0bae\u0bbe\u0ba8\u0bbf\u0bb2\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd \u0bb5\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0bc0\u0bb0\u0bcd\u0b95\u0bb3\u0bcd?",
        "te": "\u0c2e\u0c40\u0c30\u0c41 \u0c0f \u0c30\u0c3e\u0c37\u0c4d\u0c1f\u0c4d\u0c30\u0c02\u0c32\u0c4b \u0c28\u0c3f\u0c35\u0c38\u0c3f\u0c38\u0c4d\u0c24\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c41?",
        "bn": "\u0986\u09aa\u09a8\u09bf \u0995\u09cb\u09a8 \u09b0\u09be\u099c\u09cd\u09af\u09c7 \u09a5\u09be\u0995\u09c7\u09a8?",
        "mr": "\u0924\u0941\u092e\u094d\u0939\u0940 \u0915\u094b\u0923\u0924\u094d\u092f\u093e \u0930\u093e\u091c\u094d\u092f\u093e\u0924 \u0930\u093e\u0939\u0924\u093e?",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d7e \u0d0f\u0d24\u0d4d \u0d38\u0d02\u0d38\u0d4d\u0d25\u0d3e\u0d28\u0d24\u0d4d\u0d24\u0d3e\u0d23\u0d4d \u0d24\u0d3e\u0d2e\u0d38\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d24\u0d4d?",
        "pa": "\u0a24\u0a41\u0a38\u0a40\u0a02 \u0a15\u0a3f\u0a38 \u0a30\u0a3e\u0a1c \u0a35\u0a3f\u0a71\u0a1a \u0a30\u0a39\u0a3f\u0a70\u0a26\u0a47 \u0a39\u0a4b?",
        "or": "\u0b06\u0b2a\u0b23 \u0b15\u0b47\u0b09\u0b01 \u0b30\u0b3e\u0b1c\u0b4d\u0b5f\u0b30\u0b47 \u0b30\u0b41\u0b39\u0b28\u0b4d\u0b24\u0b3f?",
        "as": "\u0986\u09aa\u09c1\u09a8\u09bf \u0995\u09cb\u09a8 \u09f0\u09be\u099c\u09cd\u09af\u09a4 \u09a5\u09be\u0995\u09c7?",
        "ur": "\u0622\u067e \u06a9\u0633 \u0631\u06cc\u0627\u0633\u062a \u0645\u06cc\u06ba \u0631\u06c1\u062a\u06d2 \u06c1\u06cc\u06ba\u061f",
    },
    "age": {
        "en": "What is your age?",
        "hi": "\u0906\u092a\u0915\u0940 \u0909\u092e\u094d\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0a89\u0a82\u0aae\u0ab0 \u0a95\u0ac7\u0a9f\u0ab2\u0ac0 \u0a9b\u0ac7?",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb5\u0caf\u0cb8\u0ccd\u0cb8\u0cc1 \u0c8e\u0cb7\u0ccd\u0c9f\u0cc1?",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0baf\u0ba4\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9?",
        "te": "\u0c2e\u0c40 \u0c35\u0c2f\u0c38\u0c4d\u0c38\u0c41 \u0c0e\u0c02\u0c24?",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u09ac\u09df\u09b8 \u0995\u09a4?",
        "mr": "\u0924\u0941\u092e\u091a\u0947 \u0935\u092f \u0915\u093f\u0924\u0940 \u0906\u0939\u0947?",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d2a\u0d4d\u0d30\u0d3e\u0d2f\u0d02 \u0d0e\u0d24\u0d4d\u0d30\u0d2f\u0d3e\u0d23\u0d4d?",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a40 \u0a09\u0a2e\u0a30 \u0a15\u0a40 \u0a39\u0a48?",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b2c\u0b5f\u0b38 \u0b15\u0b47\u0b24\u0b47?",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u09ac\u09af\u09bc\u09b8 \u0995\u09bf\u09ae\u09be\u09a8?",
        "ur": "\u0622\u067e \u06a9\u06cc \u0639\u0645\u0631 \u06a9\u06cc\u0627 \u06c1\u06d2\u061f",
    },
    "annual_income": {
        "en": "What is your approximate annual family income in rupees?",
        "hi": "\u0906\u092a\u0915\u0940 \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u092a\u093e\u0930\u093f\u0935\u093e\u0930\u093f\u0915 \u0906\u092f \u0932\u0917\u092d\u0917 \u0915\u093f\u0924\u0928\u0940 \u0939\u0948?",
        "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0ab5\u0abe\u0ab0\u0acd\u0ab7\u0abf\u0a95 \u0aaa\u0abe\u0ab0\u0abf\u0ab5\u0abe\u0ab0\u0abf\u0a95 \u0a86\u0ab5\u0a95 \u0a86\u0ab6\u0ab0\u0ac7 \u0a95\u0ac7\u0a9f\u0ab2\u0ac0 \u0a9b\u0ac7?",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb5\u0cbe\u0cb0\u0ccd\u0cb7\u0cbf\u0c95 \u0c95\u0cc1\u0c9f\u0cc1\u0c82\u0cac \u0c86\u0ca6\u0cbe\u0caf \u0c8e\u0cb7\u0ccd\u0c9f\u0cc1?",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0bb5\u0bb0\u0bc1\u0b9f\u0bbe\u0ba8\u0bcd\u0ba4\u0bb0 \u0b95\u0bc1\u0b9f\u0bc1\u0bae\u0bcd\u0baa \u0bb5\u0bb0\u0bc1\u0bae\u0bbe\u0ba9\u0bae\u0bcd \u0b8e\u0bb5\u0bcd\u0bb5\u0bb3\u0bb5\u0bc1?",
        "te": "\u0c2e\u0c40 \u0c35\u0c3e\u0c30\u0c4d\u0cb7\u0c3f\u0c15 \u0c15\u0c41\u0c1f\u0c41\u0c02\u0c2c \u0c06\u0c26\u0c3e\u0c2f\u0c02 \u0c0e\u0c02\u0c24?",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u09ac\u09be\u09b0\u09cd\u09b7\u09bf\u0995 \u09aa\u09be\u09b0\u09bf\u09ac\u09be\u09b0\u09bf\u0995 \u0986\u09af\u09bc \u0995\u09a4?",
        "mr": "\u0924\u0941\u092e\u091a\u0947 \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u0915\u094c\u091f\u0941\u0902\u092c\u093f\u0915 \u0909\u0924\u094d\u092a\u0928\u094d\u0928 \u0915\u093f\u0924\u0940 \u0906\u0939\u0947?",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d35\u0d3e\u0d7c\u0d37\u0d3f\u0d15 \u0d15\u0d41\u0d1f\u0d41\u0d02\u0d2c \u0d35\u0d30\u0d41\u0d2e\u0d3e\u0d28\u0d02 \u0d0e\u0d24\u0d4d\u0d30\u0d2f\u0d3e\u0d23\u0d4d?",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a40 \u0a38\u0a3e\u0a32\u0a3e\u0a28\u0a3e \u0a2a\u0a30\u0a3f\u0a35\u0a3e\u0a30\u0a15 \u0a06\u0a2e\u0a26\u0a28 \u0a15\u0a3f\u0a70\u0a28\u0a40 \u0a39\u0a48?",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b2c\u0b3e\u0b30\u0b4d\u0b37\u0b3f\u0b15 \u0b2a\u0b3e\u0b30\u0b3f\u0b2c\u0b3e\u0b30\u0b3f\u0b15 \u0b06\u0b5f \u0b15\u0b47\u0b24\u0b47?",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u09ac\u09be\u09f0\u09cd\u09b7\u09bf\u0995 \u09aa\u09f0\u09bf\u09af\u09bc\u09be\u09b2\u09f0 \u0989\u09aa\u09be\u09f0\u09cd\u099c\u09a8 \u0995\u09bf\u09ae\u09be\u09a8?",
        "ur": "\u0622\u067e \u06a9\u06cc \u0633\u0627\u0644\u0627\u0646\u06c1 \u062e\u0627\u0646\u062f\u0627\u0646\u06cc \u0622\u0645\u062f\u0646\u06cc \u06a9\u062a\u0646\u06cc \u06c1\u06d2\u061f",
    },
    "gender": {
        "en": "What is your gender?",
        "hi": "\u0906\u092a\u0915\u093e \u0932\u093f\u0902\u0917 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0ac1\u0a82 \u0ab2\u0abf\u0a82\u0a97 \u0ab6\u0ac1\u0a82 \u0a9b\u0ac7?",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cb2\u0cbf\u0c82\u0c97 \u0caf\u0cbe\u0cb5\u0cc1\u0ca6\u0cc1?",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0baa\u0bbe\u0bb2\u0bbf\u0ba9\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?",
        "te": "\u0c2e\u0c40 \u0c32\u0c3f\u0c02\u0c17\u0c02 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u09b2\u09bf\u0999\u09cd\u0997 \u0995\u09c0?",
        "mr": "\u0924\u0941\u092e\u091a\u0947 \u0932\u093f\u0902\u0917 \u0915\u093e\u092f \u0906\u0939\u0947?",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d32\u0d3f\u0d02\u0d17\u0d02 \u0d0e\u0d28\u0d4d\u0d24\u0d3e\u0d23\u0d4d?",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a3e \u0a32\u0a3f\u0a70\u0a17 \u0a15\u0a40 \u0a39\u0a48?",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b32\u0b3f\u0b19\u0b4d\u0b17 \u0b15'\u0b23?",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u09b2\u09bf\u0982\u0997 \u0995\u09bf?",
        "ur": "\u0622\u067e \u06a9\u06cc \u062c\u0646\u0633 \u06a9\u06cc\u0627 \u06c1\u06d2\u061f",
    },
    "caste_category": {
        "en": "What is your caste category? (General, OBC, SC, ST)",
        "hi": "\u0906\u092a\u0915\u0940 \u091c\u093e\u0924\u093f \u0936\u094d\u0930\u0947\u0923\u0940 \u0915\u094d\u092f\u093e \u0939\u0948? (\u0938\u093e\u092e\u093e\u0928\u094d\u092f, OBC, SC, ST)",
        "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0a9c\u0abe\u0aa4\u0abf \u0ab6\u0acd\u0ab0\u0ac7\u0aa3\u0ac0 \u0ab6\u0ac1\u0a82 \u0a9b\u0ac7? (\u0ab8\u0abe\u0aae\u0abe\u0aa8\u0acd\u0aaf, OBC, SC, ST)",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c9c\u0cbe\u0ca4\u0cbf \u0cb5\u0cb0\u0ccd\u0c97 \u0caf\u0cbe\u0cb5\u0cc1\u0ca6\u0cc1? (\u0cb8\u0cbe\u0cae\u0cbe\u0ca8\u0ccd\u0caf, OBC, SC, ST)",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b9a\u0bbe\u0ba4\u0bbf \u0bb5\u0b95\u0bc8 \u0b8e\u0ba9\u0bcd\u0ba9? (\u0baa\u0bca\u0ba4\u0bc1, OBC, SC, ST)",
        "te": "\u0c2e\u0c40 \u0c15\u0c41\u0c32\u0c02 \u0c35\u0c30\u0c4d\u0c17\u0c02 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f? (\u0c1c\u0c28\u0c30\u0c32\u0c4d, OBC, SC, ST)",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u099c\u09be\u09a4\u09bf \u09ac\u09bf\u09ad\u09be\u0997 \u0995\u09c0? (\u09b8\u09be\u09a7\u09be\u09b0\u09a3, OBC, SC, ST)",
        "mr": "\u0924\u0941\u092e\u091a\u0940 \u091c\u093e\u0924 \u0936\u094d\u0930\u0947\u0923\u0940 \u0915\u093e\u092f \u0906\u0939\u0947? (\u0938\u093e\u092e\u093e\u0928\u094d\u092f, OBC, SC, ST)",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d1c\u0d3e\u0d24\u0d3f \u0d35\u0d3f\u0d2d\u0d3e\u0d17\u0d02 \u0d0e\u0d28\u0d4d\u0d24\u0d3e\u0d23\u0d4d? (\u0d1c\u0d28\u0d31\u0d7d, OBC, SC, ST)",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a40 \u0a1c\u0a3e\u0a24\u0a40 \u0a38\u0a3c\u0a4d\u0a30\u0a47\u0a23\u0a40 \u0a15\u0a40 \u0a39\u0a48? (\u0a1c\u0a28\u0a30\u0a32, OBC, SC, ST)",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b1c\u0b3e\u0b24\u0b3f \u0b36\u0b4d\u0b30\u0b47\u0b23\u0b40 \u0b15'\u0b23? (\u0b38\u0b3e\u0b27\u0b3e\u0b30\u0b23, OBC, SC, ST)",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u099c\u09be\u09a4\u09bf \u09b6\u09cd\u09f0\u09c7\u09a3\u09c0 \u0995\u09bf? (\u09b8\u09be\u09a7\u09be\u09f0\u09a3, OBC, SC, ST)",
        "ur": "\u0622\u067e \u06a9\u06cc \u0630\u0627\u062a \u06a9\u06cc \u06a9\u06cc\u0679\u06cc\u06af\u0631\u06cc \u06a9\u06cc\u0627 \u06c1\u06d2\u061f (\u062c\u0646\u0631\u0644\u060c OBC\u060c SC\u060c ST)",
    },
    "academic_percentage": {
        "en": "What is your latest academic percentage?",
        "hi": "\u0906\u092a\u0915\u093e \u0928\u0935\u0940\u0928\u0924\u092e \u0936\u0948\u0915\u094d\u0937\u0923\u093f\u0915 \u092a\u094d\u0930\u0924\u093f\u0936\u0924 \u0915\u094d\u092f\u093e \u0939\u0948?",
        "gu": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0a9b\u0ac7\u0ab2\u0acd\u0ab2\u0ac0 \u0ab6\u0ac8\u0a95\u0acd\u0ab7\u0aa3\u0abf\u0a95 \u0a9f\u0a95\u0abe\u0ab5\u0abe\u0ab0\u0ac0 \u0a95\u0ac7\u0a9f\u0ab2\u0ac0 \u0a9b\u0ac7?",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0c87\u0ca4\u0ccd\u0ca4\u0cc0\u0c9a\u0cbf\u0ca8 \u0cb6\u0cc8\u0c95\u0ccd\u0cb7\u0ca3\u0cbf\u0c95 \u0cb6\u0cc7\u0c95\u0ca1\u0cbe\u0cb5\u0cbe\u0cb0\u0cc1 \u0c8e\u0cb7\u0ccd\u0c9f\u0cc1?",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b9a\u0bae\u0bc0\u0baa\u0ba4\u0bcd\u0ba4\u0bbf\u0baf \u0b95\u0bb2\u0bcd\u0bb5\u0bbf \u0b9a\u0ba4\u0bb5\u0bc0\u0ba4\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?",
        "te": "\u0c2e\u0c40 \u0c24\u0c3e\u0c1c\u0c3e \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c36\u0c3e\u0c24\u0c02 \u0c0e\u0c02\u0c24?",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u09b8\u09be\u09ae\u09cd\u09aa\u09cd\u09b0\u09a4\u09bf\u0995 \u09b6\u09bf\u0995\u09cd\u09b7\u09be\u0997\u09a4 \u09b6\u09a4\u09be\u0982\u09b6 \u0995\u09a4?",
        "mr": "\u0924\u0941\u092e\u091a\u0940 \u0905\u0932\u0940\u0915\u0921\u091a\u0940 \u0936\u0948\u0915\u094d\u0937\u0923\u093f\u0915 \u091f\u0915\u094d\u0915\u0947\u0935\u093e\u0930\u0940 \u0915\u093f\u0924\u0940 \u0906\u0939\u0947?",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d0f\u0d31\u0d4d\u0d31\u0d35\u0d41\u0d02 \u0d2a\u0d41\u0d24\u0d3f\u0d2f \u0d05\u0d15\u0d4d\u0d15\u0d3e\u0d26\u0d2e\u0d3f\u0d15\u0d4d \u0d36\u0d24\u0d2e\u0d3e\u0d28\u0d02 \u0d0e\u0d24\u0d4d\u0d30\u0d2f\u0d3e\u0d23\u0d4d?",
        "pa": "\u0a24\u0a41\u0a39\u0a3e\u0a21\u0a3e \u0a38\u0a2d \u0a24\u0a4b\u0a02 \u0a24\u0a3e\u0a1c\u0a3c\u0a3e \u0a35\u0a3f\u0a26\u0a3f\u0a05\u0a15 \u0a2a\u0a4d\u0a30\u0a24\u0a40\u0a38\u0a3c\u0a24 \u0a15\u0a40 \u0a39\u0a48?",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b38\u0b3e\u0b2e\u0b4d\u0b2a\u0b4d\u0b30\u0b24\u0b3f\u0b15 \u0b36\u0b3f\u0b15\u0b4d\u0b37\u0b3e\u0b17\u0b24 \u0b36\u0b24\u0b15\u0b21\u0b3c\u0b3e \u0b15\u0b47\u0b24\u0b47?",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u09b6\u09c7\u09b9\u09a4\u09c0\u09af\u09bc\u09be \u09b6\u09c8\u0995\u09cd\u09b7\u09bf\u0995 \u09b6\u09a4\u09be\u0982\u09b6 \u0995\u09bf\u09ae\u09be\u09a8?",
        "ur": "\u0622\u067e \u06a9\u0627 \u062a\u0627\u0632\u06c1 \u062a\u0631\u06cc\u0646 \u062a\u0639\u0644\u06cc\u0645\u06cc \u0641\u06cc\u0635\u062f \u06a9\u06cc\u0627 \u06c1\u06d2\u061f",
    },
    "bpl_status": {
        "en": "Do you have a BPL card? (yes/no)",
        "hi": "\u0915\u094d\u092f\u093e \u0906\u092a\u0915\u0947 \u092a\u093e\u0938 BPL \u0915\u093e\u0930\u094d\u0921 \u0939\u0948? (\u0939\u093e\u0901/\u0928\u0939\u0940\u0902)",
        "gu": "\u0ab6\u0ac1\u0a82 \u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0aaa\u0abe\u0ab8\u0ac7 BPL \u0a95\u0abe\u0ab0\u0acd\u0aa1 \u0a9b\u0ac7? (\u0ab9\u0abe/\u0aa8\u0abe)",
        "kn": "\u0ca8\u0cbf\u0cae\u0ccd\u0cae \u0cac\u0cb3\u0cbf BPL \u0c95\u0cbe\u0cb0\u0ccd\u0ca1\u0ccd \u0c87\u0ca6\u0cc6\u0caf\u0cc7? (\u0cb9\u0ccc\u0ca6\u0cc1/\u0c87\u0cb2\u0ccd\u0cb2)",
        "ta": "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0b9f\u0bae\u0bcd BPL \u0b85\u0b9f\u0bcd\u0b9f\u0bc8 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bbe? (\u0b86\u0bae\u0bcd/\u0b87\u0bb2\u0bcd\u0bb2\u0bc8)",
        "te": "\u0c2e\u0c40 \u0c26\u0c17\u0c4d\u0c17\u0c30 BPL \u0c15\u0c3e\u0c30\u0c4d\u0c21\u0c4d \u0c09\u0c02\u0c26\u0c3e? (\u0c05\u0c35\u0c41\u0c28\u0c41/\u0c15\u0c3e\u0c26\u0c41)",
        "bn": "\u0986\u09aa\u09a8\u09be\u09b0 \u0995\u09bf BPL \u0995\u09be\u09b0\u09cd\u09a1 \u0986\u099b\u09c7? (\u09b9\u09cd\u09af\u09be\u0981/\u09a8\u09be)",
        "mr": "\u0924\u0941\u092e\u091a\u094d\u092f\u093e\u0915\u0921\u0947 BPL \u0915\u093e\u0930\u094d\u0921 \u0906\u0939\u0947 \u0915\u093e? (\u0939\u094b/\u0928\u093e\u0939\u0940)",
        "ml": "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d7e\u0d15\u0d4d\u0d15\u0d4d BPL \u0d15\u0d3e\u0d7c\u0d21\u0d4d \u0d09\u0d23\u0d4d\u0d1f\u0d4b? (\u0d05\u0d24\u0d46/\u0d07\u0d32\u0d4d\u0d32)",
        "pa": "\u0a15\u0a40 \u0a24\u0a41\u0a39\u0a3e\u0a21\u0a47 \u0a15\u0a4b\u0a32 BPL \u0a15\u0a3e\u0a30\u0a21 \u0a39\u0a48? (\u0a39\u0a3e\u0a02/\u0a28\u0a39\u0a40\u0a02)",
        "or": "\u0b06\u0b2a\u0b23\u0b19\u0b4d\u0b15 \u0b2a\u0b3e\u0b16\u0b30\u0b47 BPL \u0b15\u0b3e\u0b30\u0b4d\u0b21 \u0b05\u0b1b\u0b3f \u0b15\u0b3f? (\u0b39\u0b01/\u0b28\u0b3e)",
        "as": "\u0986\u09aa\u09cb\u09a8\u09be\u09f0 \u0993\u099a\u09f0\u09a4 BPL \u0995\u09be\u09b0\u09cd\u09a1 \u0986\u099b\u09c7 \u09a8\u09c7\u0995\u09bf? (\u09b9\u09af\u09bc/\u09a8\u09b9\u09af\u09bc)",
        "ur": "\u06a9\u06cc\u0627 \u0622\u067e \u06a9\u06d2 \u067e\u0627\u0633 BPL \u06a9\u0627\u0631\u0688 \u06c1\u06d2\u061f (\u06c1\u0627\u06ba/\u0646\u06c1\u06cc\u06ba)",
    },
}


PROFILE_FIELDS = [
    "age",
    "annual_income",
    "income",
    "occupation",
    "education_level",
    "state",
    "gender",
    "category",
    "academic_percentage",
    "caste_category",
    "bpl_status",
]

BOOLEAN_FIELDS = {"bpl_status"}
EXPECTING_NUMERIC_FIELDS = {"age", "income", "annual_income", "academic_percentage"}

# Category-specific slot policy. Occupation is NOT globally required.
REQUIRED_FIELDS_BY_CATEGORY: dict[str, list[str]] = {
    "education": ["state", "age", "annual_income"],
    "agriculture": ["state", "age", "annual_income"],
    "health": ["state", "age", "annual_income"],
    "employment": ["state", "age", "annual_income"],
    "finance_business": ["state", "age", "annual_income"],
    "women_child": ["state", "age", "annual_income"],
    "senior_citizen": ["state", "age", "annual_income"],
    "housing": ["state", "age", "annual_income"],
    "social_welfare": ["state", "age", "annual_income"],
    "disability": ["state", "age", "annual_income"],
}

RESET_COMMANDS = {"reset", "restart", "start over", "\u0ab0\u0ac0\u0ab8\u0ac7\u0a9f"}
LANGUAGE_CHANGE_COMMANDS = {"change language", "switch language", "\u0aad\u0abe\u0ab7\u0abe \u0aac\u0aa6\u0ab2\u0acb"}
HELP_COMMANDS = {"help", "\u0aae\u0aa6\u0aa6"}
CHECK_COMMANDS = set()
DIGIT_MENU_INTENTS = {"1": "reset", "2": "change_language", "3": "help"}
APPLY_COMMANDS = {"apply", "application", "apply scheme", "apply for", "select scheme", "choose scheme"}
MORE_COMMANDS = {"more", "give me more", "વધુ", "આગળ"}

YES_WORDS = {"yes", "y", "haan", "ha", "true", "1", "ji"}
NO_WORDS = {"no", "n", "nahi", "nahin", "na", "false", "0"}

NUMERIC_UNIT_ALIASES = {
    "lakhs": "lakh",
    "lac": "lakh",
    "lacs": "lakh",
    "\u0932\u093e\u0916": "lakh",
    "\u0ab2\u0abe\u0a96": "lakh",
    "\u09b2\u09be\u0996": "lakh",
    "\u0bb2\u0b9f\u0bcd\u0b9a\u0bae\u0bcd": "lakh",
    "\u0644\u0627\u06a9\u06be": "lakh",
    "crores": "crore",
    "cr": "crore",
    "\u0915\u0930\u094b\u0921\u093c": "crore",
    "\u0a95\u0ab0\u0acb\u0aa1": "crore",
    "\u06a9\u0631\u0648\u0691": "crore",
    "\u0939\u091c\u093e\u0930": "thousand",
    "\u0ab9\u0a9c\u0abe\u0ab0": "thousand",
    "\u06c1\u0632\u0627\u0631": "thousand",
}

PUBLIC_INTENTS = {"SEARCH_SCHEMES", "APPLY_SCHEME", "CHECK_ELIGIBILITY", "HELP", "CHANGE_LANGUAGE"}
GENERIC_REASON_TEXTS = {
    "matched to your query terms",
    "relevant based on profile",
    "relevant based on your profile",
    "matches your profile criteria",
    "relevant based on scheme description and your profile context",
}


def _pick_lang(language: str | None) -> str:
    return language if language in SUPPORTED_LANGUAGES else "en"


def _fallback_language_from_text(text: str) -> str | None:
    query = normalize_text_light(text or "")
    if not query:
        return None
    name_to_code = {normalize_text_light(name): code for code, name in SUPPORTED_LANGUAGES.items()}
    for code in SUPPORTED_LANGUAGES:
        name_to_code[code] = code
    if query in name_to_code:
        return name_to_code[query]
    candidates = list(name_to_code.keys())
    match = get_close_matches(query, candidates, n=1, cutoff=0.72)
    if not match:
        return None
    return name_to_code.get(match[0])


def _menu(language: str | None) -> str:
    lang = _pick_lang(language)
    # Static cache first
    cached = get_ui_text(lang, "menu", "")
    if cached:
        return cached
    if lang == "en":
        return MENU_TEXT
    translated = translate_from_english(MENU_TEXT, lang)
    return translated.strip() if translated and translated.strip() else MENU_TEXT


def _with_menu(text: str, language: str | None) -> str:
    body = _prepare_outgoing(text, language)
    if not body:
        body = _prepare_outgoing(LABELS["help"], language)
    return f"{body}\n\n{_menu(language)}"


def _ascii_ratio(text: str) -> float:
    body = str(text or "")
    if not body:
        return 0.0
    return sum(1 for ch in body if ord(ch) < 128) / len(body)


def _strip_existing_menu(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    menu_variants = {MENU_TEXT, *MENU_TEXTS.values()}
    for menu_text in menu_variants:
        menu_text = str(menu_text or "").strip()
        if menu_text and body.endswith(menu_text):
            body = body[: -len(menu_text)].rstrip()
    return body


def finalize_response(payload: dict, language: str | None) -> dict:
    lang = _pick_lang(language)
    safe_payload = dict(payload or {})
    original_response = str(safe_payload.get("response") or "")
    body = _strip_existing_menu(original_response)
    if not body:
        body = LABELS["help"]
    ascii_ratio = _ascii_ratio(body)
    translation_applied = False
    translation_failed = False

    if lang != "en":
        translated, translation_meta = translate_from_english_with_meta(body, lang)
        translated = translated.strip() if translated else ""
        translation_failed = bool((translation_meta or {}).get("translation_failed"))
        if translated:
            body = translated
            translation_applied = True
            if _ascii_ratio(body) > 0.75:
                logger.warning({"event": "untranslated_english_blocked", "language": lang})
        else:
            logger.warning({"event": "untranslated_english_blocked", "language": lang})

        # Hard language lock: if content still looks English, fall back to static localized text.
        # EXCEPTION: if schemes are present the body is English scheme data — that's fine;
        # the frontend renders cards; just replace the bubble text with the localized header.
        has_schemes = bool(safe_payload.get("schemes"))
        if _ascii_ratio(body) > 0.75:
            if has_schemes:
                body = SCHEMES_FOUND_LABEL.get(lang, SCHEMES_FOUND_LABEL["en"])
            else:
                matched_label_key = None
                for key, value in LABELS.items():
                    if str(value or "").strip() == str(_strip_existing_menu(original_response) or "").strip():
                        matched_label_key = key
                        break
                localized = _get_static(lang, str(matched_label_key or ""))
                if localized and localized.strip():
                    body = localized.strip()
                elif _get_static(lang, "help"):
                    body = str(_get_static(lang, "help")).strip()

    safe_payload["response"] = f"{body}\n\n{_menu(lang)}".strip()
    safe_payload["schemes"] = safe_payload.get("schemes") if isinstance(safe_payload.get("schemes"), list) else []
    safe_payload["fallback_used"] = bool(safe_payload.get("fallback_used"))
    safe_payload["errors"] = safe_payload.get("errors") if isinstance(safe_payload.get("errors"), list) else []
    intent_label = str(safe_payload.get("intent") or "HELP").strip().upper()
    safe_payload["intent"] = intent_label if intent_label in PUBLIC_INTENTS else "HELP"
    if translation_failed:
        internal_meta = dict(safe_payload.get("internal_metadata") or {})
        internal_meta["translation_failed"] = True
        safe_payload["internal_metadata"] = internal_meta

    logger.info(
        {
            "event": "response_finalized",
            "language": lang,
            "original_response": original_response[:200],
            "ascii_ratio": round(ascii_ratio, 3),
            "translation_applied": translation_applied,
            "translation_failed": translation_failed,
        }
    )
    return safe_payload


def _result(
    payload: dict[str, Any],
    state: str,
    language: str | None,
    intent: str | None = None,
) -> tuple[dict[str, Any], str]:
    wrapped = dict(payload or {})
    if intent:
        wrapped["intent"] = intent
    return finalize_response(wrapped, language), state


def _translate_or_fallback(text: str, language: str | None) -> str:
    lang = _pick_lang(language)
    if lang == "en":
        return text
    translated = translate_from_english(text, lang)
    return translated.strip() if translated and translated.strip() else text


def _prepare_outgoing(text: str, language: str | None) -> str:
    """
    Ensure outgoing body text is translated from English when needed.
    If text already appears non-English/non-ASCII, keep it unchanged.
    """
    lang = _pick_lang(language)
    body = str(text or "").strip()
    if not body or lang == "en":
        return body
    ascii_ratio = sum(1 for ch in body if ord(ch) < 128) / max(1, len(body))
    if ascii_ratio < 0.85:
        return body
    translated = translate_from_english(body, lang)
    return translated.strip() if translated and translated.strip() else body


def _translate_to_english_safe(text: str, language: str | None) -> str:
    lang = _pick_lang(language)
    raw = str(text or "")
    if lang != "en" and raw:
        latin_chars = sum(1 for ch in raw if ord(ch) < 128)
        non_latin_letters = sum(1 for ch in raw if ch.isalpha() and ord(ch) >= 128)
        if latin_chars >= max(1, int(len(raw) * 0.9)) and non_latin_letters == 0:
            return raw.strip()
    translated = translate_to_english(text or "", lang)
    return translated.strip() if translated and translated.strip() else str(text or "")


def _field_question(field: str, language: str | None) -> str:
    lang = _pick_lang(language)
    field_map = QUESTION_MAP.get(field)
    if field_map:
        q = field_map.get(lang)
        if q:
            return q
    # Ultimate fallback: English from FOLLOWUP_QUESTIONS
    return FOLLOWUP_QUESTIONS.get(field, FOLLOWUP_QUESTIONS["category"])


def _render_followup_question(field: str, language: str | None, llm_question_english: str | None = None) -> str:
    # HARDCODED ONLY Ã¢â‚¬â€ never use LLM-generated questions
    return _field_question(field, language)


def _localized_label(key: str, language: str | None, **kwargs: str) -> str:
    lang = _pick_lang(language)
    cached = _get_static(lang, key, **kwargs)
    if cached:
        return cached
    template = LABELS.get(key, "")
    if kwargs and template:
        try:
            template = template.format(**kwargs)
        except KeyError:
            pass
    return _translate_or_fallback(template, lang) if template else ""


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


def _normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if parsed.is_integer():
        return int(parsed)
    return int(round(parsed))


def _normalize_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _normalize_state_name(state: Any) -> str | None:
    normalized = normalize_state_name(state)
    return normalized.title() if normalized else None


def _normalize_domain_category(category: Any) -> str | None:
    """Normalize extracted categories into stable internal category keys."""
    if category is None:
        return None
    raw = normalize_text_light(str(category)).replace(" ", "_")
    if not raw:
        return None

    canonical_set = {
        "education", "health", "agriculture", "senior_citizen", 
        "disability", "women_child", "finance_business", 
        "housing", "employment", "social_welfare", "unknown"
    }
    
    # Legacy fallbacks
    legacy_mapping = {
        "finance": "finance_business",
        "financial_assistance": "finance_business",
        "business": "finance_business",
        "women": "women_child",
        "women_and_child": "women_child",
        "senior": "senior_citizen",
        "other": "social_welfare",
        "others": "social_welfare",
    }
    
    if raw in canonical_set:
        if raw == "unknown":
            return None
        return raw
        
    if raw in legacy_mapping:
        return legacy_mapping[raw]
        
    return None


def _infer_category(
    *,
    original_text: str | None,
    english_text: str | None,
    language: str | None,
    profile_context: dict[str, Any] | None = None,
) -> str | None:
    english = str(english_text or "").strip()
    original = str(original_text or "").strip()
    if not english and not original:
        return None

    context = dict(profile_context or {})

    # Gemini is the source of truth for category meaning extraction.
    llm_intent = classify_user_intent_llm(
        original_text=original or english,
        english_translation=english or original,
        selected_language=_pick_lang(language),
        conversation_context=context,
    )
    llm_category = _normalize_domain_category(llm_intent.get("canonical_category"))
    llm_confidence = float(llm_intent.get("confidence") or 0.0)
    if llm_category and llm_confidence >= 0.45:
        return llm_category

    return llm_category


def _normalize_number_text(message: str) -> str:
    text = str(message or "")
    for alias, canonical in NUMERIC_UNIT_ALIASES.items():
        text = text.replace(alias, f" {canonical} ")
    return " ".join(text.split())


def _is_number_like(message: str) -> bool:
    text = _normalize_number_text(message).strip().lower()
    if not text:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|crore|cr)", text):
        return True
    return False


def _parse_number_like(message: str) -> float | None:
    text = _normalize_number_text(message).strip().lower().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac|crore|cr)?", text)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"k", "thousand"}:
        value *= 1000
    elif unit in {"lakh", "lac"}:
        value *= 100000
    elif unit in {"crore", "cr"}:
        value *= 10000000
    return value


def _coerce_expected_field_value(expected_field: str | None, message: str) -> dict[str, Any]:
    if expected_field not in EXPECTING_NUMERIC_FIELDS:
        return {}
    if not _is_number_like(message):
        return {}

    parsed = _parse_number_like(message)
    if parsed is None:
        return {}

    if expected_field == "academic_percentage":
        return {"academic_percentage": float(parsed)}
    if expected_field in {"income", "annual_income"}:
        value = normalize_income_value(parsed)
        if value is None:
            return {}
        return {"annual_income": value, "income": value}
    return {expected_field: int(round(parsed))}


def _fast_extract_expected_field(expected_field: str | None, message: str) -> dict[str, Any]:
    field = str(expected_field or "").strip().lower()
    if not field:
        return {}
    if field == "income":
        field = "annual_income"

    numeric = _coerce_expected_field_value(field, message)
    if numeric:
        return numeric

    text = str(message or "").strip()
    lowered = text.lower()
    if not text:
        return {}

    if field == "bpl_status":
        if lowered in YES_WORDS:
            return {"bpl_status": True}
        if lowered in NO_WORDS:
            return {"bpl_status": False}
        return {}

    if field == "category":
        # Let extraction own category parsing so it can also capture intent,
        # answer_english, and any extra profile fields in the same utterance.
        return {}

    if field == "occupation":
        if _is_number_like(text):
            return {}
        if any(ch.isalpha() for ch in text):
            return {"occupation": lowered[:50]}
        return {}

    if field == "education_level":
        if len(text) <= 40:
            return {"education_level": text}
        return {}

    if field == "state":
        if _is_number_like(text) or len(text) > 40:
            return {}
        normalized = _normalize_state_name(text)
        return {"state": normalized} if normalized else {}

    if field == "gender":
        if lowered in {"male", "m"}:
            return {"gender": "male"}
        if lowered in {"female", "f"}:
            return {"gender": "female"}
        if lowered in {"other", "others"}:
            return {"gender": "other"}
        return {}

    if field == "caste_category":
        if len(text) <= 30 and len(text.split()) <= 3:
            return {"caste_category": text}
        return {}

    return {}


def _detect_explicit_intent(message: str) -> str:
    lowered = str(message or "").strip().lower()
    if any(phrase == lowered or phrase in lowered for phrase in RESET_COMMANDS):
        return "reset"
    if any(phrase == lowered or phrase in lowered for phrase in LANGUAGE_CHANGE_COMMANDS):
        return "change_language"
    if any(phrase == lowered or phrase in lowered for phrase in HELP_COMMANDS):
        return "help"
    if any(phrase == lowered or phrase in lowered for phrase in CHECK_COMMANDS):
        return "check_eligibility"
    return "unknown"


def _detect_digit_menu_intent(message: str) -> str:
    return DIGIT_MENU_INTENTS.get(str(message or "").strip().lower(), "unknown")


def _public_intent_from_internal(internal_intent: str, llm_data: dict[str, Any] | None = None) -> str:
    direct = str((llm_data or {}).get("intent") or "").strip().upper()
    if direct in PUBLIC_INTENTS:
        return direct

    mapping = {
        "apply_scheme": "APPLY_SCHEME",
        "scheme_search": "SEARCH_SCHEMES",
        "search_schemes": "SEARCH_SCHEMES",
        "check_eligibility": "CHECK_ELIGIBILITY",
        "help": "HELP",
        "change_language": "CHANGE_LANGUAGE",
    }
    return mapping.get(str(internal_intent or "").strip().lower(), "UNKNOWN")


def _profile_signal(profile: dict[str, Any]) -> bool:
    fields = [
        "category",
        "age",
        "annual_income",
        "occupation",
        "education_level",
        "state",
        "gender",
        "caste_category",
        "academic_percentage",
        "bpl_status",
    ]
    return any(profile.get(field) not in (None, "") for field in fields)


def _extraction_cacheable(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    if str(payload.get("intent") or "unknown").strip().lower() not in {"unknown", ""}:
        return True
    for field in PROFILE_FIELDS:
        if payload.get(field) not in (None, ""):
            return True
    if str(payload.get("answer_english") or "").strip():
        return True
    return False


def _extract_profile(
    original_text: str,
    english_text: str | None = None,
    expected_field: str | None = None,
    language: str | None = None,
    current_profile: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    source_english = english_text if english_text is not None else original_text
    lang_for_cache = _pick_lang(language)
    prof = current_profile or {}
    cache_dims = {
        "category": str(prof.get("category") or ""),
        "state": str(prof.get("state") or ""),
        "income": prof.get("annual_income") or prof.get("income"),
        "age": prof.get("age"),
    }
    llm_data = get_extraction_cache(
        source_english, lang_for_cache, expected_field, **cache_dims
    ) or {}
    if not llm_data:
        llm_data = extract_profile_llm(
            original_text,
            english_text=source_english,
            language=language,
            current_profile=prof,
            expected_field=expected_field,
        ) or {}
        if _extraction_cacheable(llm_data):
            set_extraction_cache(
                source_english, llm_data, lang_for_cache, expected_field, **cache_dims
            )

    profile = {field: None for field in PROFILE_FIELDS}
    profile["category"] = _normalize_domain_category(llm_data.get("category"))
    profile["age"] = _normalize_int(llm_data.get("age"))
    income_value = normalize_income_value(llm_data.get("income"))
    profile["annual_income"] = income_value
    profile["income"] = income_value
    profile["occupation"] = (str(llm_data.get("occupation")).strip().lower() if llm_data.get("occupation") else None)
    profile["education_level"] = (str(llm_data.get("education_level")).strip() if llm_data.get("education_level") else None)
    profile["state"] = _normalize_state_name(llm_data.get("state"))
    profile["gender"] = (str(llm_data.get("gender")).strip().lower() if llm_data.get("gender") else None)
    profile["caste_category"] = (str(llm_data.get("caste_category")).strip() if llm_data.get("caste_category") else None)
    profile["academic_percentage"] = _normalize_float(llm_data.get("academic_percentage"))
    profile["bpl_status"] = _normalize_bool(llm_data.get("bpl_status"))

    expected_override = _coerce_expected_field_value(expected_field, original_text)
    for field, value in expected_override.items():
        profile[field] = value

    if expected_field == "bpl_status":
        answer = str(original_text).strip().lower()
        if answer in YES_WORDS:
            profile["bpl_status"] = True
        elif answer in NO_WORDS:
            profile["bpl_status"] = False

    if not profile.get("category"):
        inferred_category = _infer_category(
            original_text=original_text,
            english_text=english_text,
            language=language,
            profile_context=current_profile,
        )
        if inferred_category:
            profile["category"] = inferred_category

    intent = str(llm_data.get("intent") or "unknown").strip().lower()
    if intent == "unknown" and _profile_signal(profile):
        intent = "profile_update"

    return intent, profile, llm_data


def _merge_profile(current_profile: dict[str, Any], extracted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    merged = dict(current_profile or {})
    changed: list[str] = []

    for field, value in (extracted or {}).items():
        if field in BOOLEAN_FIELDS:
            value = _normalize_bool(value)
        if value is None:
            continue
        if field == "category" and current_profile.get("category"):
            continue
        if merged.get(field) != value:
            merged[field] = value
            changed.append(field)

    if merged.get("annual_income") is not None:
        merged["income"] = merged.get("annual_income")
    elif merged.get("income") is not None:
        merged["annual_income"] = merged.get("income")

    if not merged.get("category"):
        inferred = _infer_category(
            original_text="",
            english_text="",
            language=merged.get("language"),
            profile_context=merged,
        )
        if inferred:
            merged["category"] = inferred
            if "category" not in changed:
                changed.append("category")

    return merged, changed


def _is_missing(profile: dict[str, Any], field: str) -> bool:
    if field == "annual_income":
        return profile.get("annual_income") in (None, "") and profile.get("income") in (None, "")
    return profile.get(field) in (None, "")


def _education_ready(profile: dict[str, Any]) -> bool:
    occupation = normalize_text_light(str(profile.get("occupation") or ""))
    has_student_role = occupation in {"student", "vidyarthi", "chatra", "chhatra"}
    has_education_level = str(profile.get("education_level") or "").strip() not in {"", "none"}
    return has_student_role or has_education_level


def _category_required_fields(profile: dict[str, Any]) -> list[str]:
    category = str(profile.get("category") or "").strip().lower()
    return list(REQUIRED_FIELDS_BY_CATEGORY.get(category, ["state", "age", "annual_income"]))


def _is_profile_ready_for_category(profile: dict[str, Any]) -> bool:
    category = str(profile.get("category") or "").strip().lower()
    if not category or category == "unknown":
        return False
    for field in _category_required_fields(profile):
        if _is_missing(profile, field):
            return False
    if category == "education" and not _education_ready(profile):
        return False
    return True


def _fallback_next_field(profile: dict[str, Any]) -> str | None:
    category = str(profile.get("category") or "").strip().lower()
    if not category or category == "unknown":
        return "category"

    for field in _category_required_fields(profile):
        if _is_missing(profile, field):
            return field

    # Education accepts either explicit level OR student role.
    if category == "education" and not _education_ready(profile):
        if _is_missing(profile, "education_level"):
            return "education_level"
        if _is_missing(profile, "occupation"):
            return "occupation"
    return None


def _decide_next_field(
    profile: dict[str, Any],
    user_goal: str,
    previous_question: str | None,
    last_user_message: str | None,
) -> tuple[str | None, str | None]:
    # DETERMINISTIC ONLY Ã¢â‚¬â€ no LLM call
    next_field = _fallback_next_field(profile)
    return next_field, None


def _meaningful_count(profile: dict[str, Any]) -> int:
    fields = [
        "category",
        "age",
        "annual_income",
        "occupation",
        "education_level",
        "state",
        "gender",
        "caste_category",
        "bpl_status",
        "academic_percentage",
    ]
    return sum(1 for field in fields if profile.get(field) not in (None, ""))


def _enough_for_results(profile: dict[str, Any]) -> bool:
    return _is_profile_ready_for_category(profile)


def _norm_state(value: Any) -> str:
    text = normalize_text_light(str(value or "")).strip().lower()
    if not text:
        return ""
    if text in {"all india", "national", "central", "india", "nationwide", "pan india"}:
        return "all india"
    normalized = normalize_state_name(text)
    return normalized or text


def _is_all_india(state: Any) -> bool:
    return _norm_state(state) == "all india"


def _state_allowed(user_state: str | None, scheme_state: str | None) -> bool:
    if not user_state:
        return True
    if not scheme_state:
        return False
    return _norm_state(scheme_state) == _norm_state(user_state) or _is_all_india(scheme_state)


def _final_geo_filter(
    schemes: list[dict[str, Any]],
    user_state: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Final non-bypassable geo gate before API payload is returned."""
    user_state_norm = _norm_state(user_state)
    if not user_state_norm:
        return list(schemes or []), 0

    allowed: list[dict[str, Any]] = []
    rejected = 0
    for scheme in schemes or []:
        scheme_state_raw = str(scheme.get("state") or "").strip()
        scheme_state_norm = _norm_state(scheme_state_raw)
        if scheme_state_norm == user_state_norm or _is_all_india(scheme_state_norm):
            allowed.append(scheme)
            continue
        rejected += 1
        logger.warning(
            {
                "event": "geo_rejected",
                "scheme_name": str(scheme.get("scheme_name") or "Unknown Scheme"),
                "scheme_state": scheme_state_raw or None,
                "user_state": str(user_state or "") or None,
                "reason": "geo_rejected: scheme_state != user_state and not_all_india",
            }
        )
    return allowed, rejected


def _build_retrieval_query(last_message_english: str | None, profile: dict[str, Any]) -> str:
    text = str(last_message_english or "").strip()
    if text and not _is_number_like(text):
        return text
    category = str(profile.get("category") or "").strip().replace("_", " ")
    state = str(profile.get("state") or "").strip()
    if category and state:
        return f"{category} schemes in {state}"
    if category:
        return f"{category} schemes"
    return text


def _docs_to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _text_or_fallback(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _is_generic_reason(reason: str) -> bool:
    body = normalize_text_light(reason)
    return body in GENERIC_REASON_TEXTS


def _build_concrete_reasons(
    profile: dict[str, Any],
    scheme_state: str,
    scheme_category: str,
    raw_reasons: list[str] | None,
) -> list[str]:
    state_claim_markers = (
        "all india",
        "national",
        "available at all india",
        "available in ",
    )
    output: list[str] = []
    for reason in raw_reasons or []:
        text = str(reason or "").strip()
        if text and any(marker in normalize_text_light(text) for marker in state_claim_markers):
            # Recompute geo reasons from actual scheme.state/profile.state to avoid false labels.
            continue
        if text and not _is_generic_reason(text):
            output.append(text)

    profile_category = str(profile.get("category") or "").strip().replace("_", " ").lower()
    scheme_category_norm = normalize_text_light(str(scheme_category or "")).replace("_", " ")
    if profile_category and scheme_category_norm and profile_category == scheme_category_norm:
        output.append(f"This scheme is for {scheme_category_norm} category, matching your selected category.")

    profile_state = str(profile.get("state") or "").strip()
    if profile_state and scheme_state:
        if _norm_state(profile_state) == _norm_state(scheme_state):
            output.append(f"This scheme is available in {profile_state}.")
        elif _is_all_india(scheme_state):
            output.append("This is a national/All India scheme.")

    income = profile.get("annual_income") if profile.get("annual_income") is not None else profile.get("income")
    if income is not None:
        try:
            output.append(f"Your income is Rs. {int(float(income)):,}.")
        except Exception:
            output.append(f"Your income is {income}.")

    if not output:
        output.append("Matched by strict category and state filtering.")
    return list(dict.fromkeys(output))


def _lookup_dataset_scheme(scheme: dict[str, Any]) -> dict[str, Any]:
    scheme_name = str(scheme.get("scheme_name") or "").strip().lower()
    scheme_id = str(scheme.get("scheme_id") or "").strip().lower()
    scheme_state = str(scheme.get("state") or "").strip().lower()
    for row in load_scheme_dataset():
        row_name = str(row.get("scheme_name") or "").strip().lower()
        row_id = str(row.get("scheme_id") or "").strip().lower()
        row_state = str(row.get("state") or "").strip().lower()
        if scheme_id and row_id and row_id == scheme_id:
            return row
        if scheme_name and row_name == scheme_name and (not scheme_state or not row_state or row_state == scheme_state):
            return row
    return {}


def _format_search_schemes_response(cards: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    lines: list[str] = ["Intent: SEARCH_SCHEMES", "Top matching schemes:"]
    for idx, card in enumerate(cards[:5], start=1):
        reasons = _build_concrete_reasons(
            profile,
            str(card.get("state") or ""),
            str(card.get("category") or ""),
            card.get("why_match"),
        )
        docs = _docs_to_list(card.get("documents_required"))
        lines.extend(
            [
                f"{idx}. Scheme Name: {_text_or_fallback(card.get('scheme_name'), 'Unknown Scheme')}",
                f"- Description: {_text_or_fallback(card.get('description'), 'Description not available in dataset.')}",
                f"- Eligibility: {_text_or_fallback(card.get('eligibility_text'), 'Eligibility details not available in dataset.')}",
                f"- Documents Required: {', '.join(docs) if docs else 'Not listed in dataset.'}",
                f"- Why this scheme matches: {reasons[0]}",
            ]
        )
    return "\n".join(lines)


def _format_apply_scheme_response(scheme: dict[str, Any], profile: dict[str, Any]) -> str:
    docs = _docs_to_list(scheme.get("documents_required"))
    steps = _docs_to_list(scheme.get("application_process"))
    if not steps:
        steps = ["Step-by-step application process is not available in the current dataset."]
    lines: list[str] = [
        "Intent: APPLY_SCHEME",
        f"Full Scheme Name: {_text_or_fallback(scheme.get('scheme_name'), 'Unknown Scheme')}",
        f"Description: {_text_or_fallback(scheme.get('description') or scheme.get('benefits_summary'), 'Description not available in dataset.')}",
        f"Eligibility Criteria: {_text_or_fallback(scheme.get('eligibility_text'), 'Eligibility details not available in dataset.')}",
        f"Required Documents: {', '.join(docs) if docs else 'Not listed in dataset.'}",
        "Step-by-step Application Process:",
    ]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append(f"Official Apply Link: {_text_or_fallback(scheme.get('application_link'), 'Not available in dataset.')}")
    reasons = _build_concrete_reasons(
        profile,
        str(scheme.get("state") or ""),
        str(scheme.get("category") or ""),
        scheme.get("why_match"),
    )
    lines.append(f"Why this scheme matches: {reasons[0]}")
    return "\n".join(lines)


def _format_eligibility_response(profile: dict[str, Any], schemes: list[dict[str, Any]]) -> str:
    category = str(profile.get("category") or "").strip()
    filtered = filter_schemes(profile, category=category, schemes=schemes)
    lines: list[str] = ["Intent: CHECK_ELIGIBILITY"]
    total = 0

    for label, bucket in (
        ("Eligible", filtered.get("eligible") or []),
        ("Not Eligible (Need More Data)", filtered.get("uncertain_needs_more_data") or []),
        ("Not Eligible", filtered.get("ineligible") or []),
    ):
        for scheme in bucket[:3]:
            total += 1
            reasons = scheme.get("why_match") or []
            reason_text = "; ".join(str(r).strip() for r in reasons if str(r).strip()) or "No reason available."
            lines.extend(
                [
                    f"- Scheme: {_text_or_fallback(scheme.get('scheme_name'), 'Unknown Scheme')}",
                    f"  Status: {label}",
                    f"  Reasons: {reason_text}",
                ]
            )

    if total == 0:
        lines.append("- Eligible / Not Eligible: Not determinable from current dataset.")
        lines.append("- Reasons: Please complete missing profile details and try again.")
    return "\n".join(lines)


def _is_apply_request(message: str) -> bool:
    text = normalize_text_light(message)
    if not text:
        return False
    if any(token in text for token in APPLY_COMMANDS):
        return True
    return bool(re.match(r"^(apply|select|choose)\s+\d+$", text))


def _is_more_request(message: str) -> bool:
    text = normalize_text_light(message)
    if not text:
        return False
    if text in MORE_COMMANDS:
        return True
    return text.startswith("more ") or text.startswith("give me more")


def _resolve_selected_scheme(message: str, user: dict[str, Any], scheme_name_hint: str | None = None) -> dict[str, Any] | None:
    text = normalize_text_light(message)
    last_schemes = user.get("last_schemes") if isinstance(user.get("last_schemes"), list) else []
    selected = user.get("selected_scheme") if isinstance(user.get("selected_scheme"), dict) else None

    index_match = re.search(r"\b(\d+)\b", text)
    if index_match and last_schemes:
        idx = int(index_match.group(1)) - 1
        if 0 <= idx < len(last_schemes):
            return dict(last_schemes[idx])

    if last_schemes:
        for card in last_schemes:
            name = normalize_text_light(card.get("scheme_name"))
            if name and name in text:
                return dict(card)
    hint = normalize_text_light(scheme_name_hint or "")
    if hint and last_schemes:
        for card in last_schemes:
            if normalize_text_light(card.get("scheme_name")) == hint:
                return dict(card)

    return dict(selected) if selected else None

def _normalize_scheme_cards(
    schemes: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
    max_cards: int | None = 5,
) -> list[dict[str, Any]]:
    current_profile = dict(profile or {})

    # ── STRICT CATEGORY PRE-FILTER ─────────────────────────────────────────
    # Dataset uses Title Case: "Housing", "Education", "Financial Assistance"
    # Profile stores lowercase slugs: "housing", "education", "finance_business"
    # Map them, then drop any scheme whose category doesn't match.
    CATEGORY_SLUG_TO_DATASET: dict[str, set[str]] = {
        "education":        {"education", "scholarship"},
        "health":           {"health", "healthcare", "medical"},
        "agriculture":      {"agriculture", "farming"},
        "employment":       {"employment", "skill", "job"},
        "finance_business": {"financial assistance", "finance", "business"},
        "women_child":      {"women & child", "women and child", "mahila", "child"},
        "housing":          {"housing", "awas", "shelter"},
        "senior_citizen":   {"senior citizen", "pension", "elderly"},
        "disability":       {"disability", "divyang"},
        "social_welfare":   {"social welfare", "welfare", "others"},
    }

    user_cat_slug = str(current_profile.get("category") or "").strip().lower().replace(" ", "_")
    allowed_cats = CATEGORY_SLUG_TO_DATASET.get(user_cat_slug)

    def _cat_allowed(scheme: dict) -> bool:
        if not allowed_cats:
            return True  # no category known, pass everything
        raw = str(
            scheme.get("category")
            or scheme.get("scheme_category")
            or ""
        ).strip().lower()
        return not raw or raw in allowed_cats  # blank category → pass through

    filtered_schemes = [s for s in (schemes or []) if _cat_allowed(s)]
    # If strict filter wiped everything, fall back to the full list
    if not filtered_schemes:
        filtered_schemes = list(schemes or [])

    cards: list[dict[str, Any]] = []
    candidate_schemes = filtered_schemes if max_cards is None else filtered_schemes[:max_cards]
    for scheme in candidate_schemes:
        # Quality gate: skip garbage entries
        name = str(scheme.get("scheme_name") or "").strip()
        if not name or name.lower() in {"unnamed scheme", "unknown scheme"}:
            continue
        dataset_row = _lookup_dataset_scheme(scheme)
        benefits_raw = (
            scheme.get("benefits_summary")
            or scheme.get("benefits")
            or scheme.get("description")
            or dataset_row.get("benefits")
            or dataset_row.get("description")
            or ""
        )
        if "no description available" in str(benefits_raw).lower():
            continue

        why_match = scheme.get("why_match") or scheme.get("reason") or []
        if isinstance(why_match, str):
            why_match = [why_match]
        why_match = [str(item).strip() for item in why_match if str(item).strip()]
        scheme_state = str(scheme.get("state") or dataset_row.get("state") or "Unknown").strip()
        scheme_category = str(
            scheme.get("category")
            or dataset_row.get("category")
            or current_profile.get("category")
            or "Unknown"
        ).strip()
        why_match = _build_concrete_reasons(current_profile, scheme_state, scheme_category, why_match)

        documents = _docs_to_list(scheme.get("documents_required"))
        if not documents:
            documents = _docs_to_list(dataset_row.get("documents_required"))

        # Convert eligibility dict to readable text
        raw_elig = (
            scheme.get("eligibility")
            or dataset_row.get("eligibility")
            or dataset_row.get("eligibility_criteria")
        )
        if isinstance(raw_elig, dict):
            parts: list[str] = []
            if raw_elig.get("min_age") is not None:
                parts.append(f"Minimum age: {raw_elig['min_age']}")
            if raw_elig.get("max_age") is not None:
                parts.append(f"Maximum age: {raw_elig['max_age']}")
            if raw_elig.get("max_income") is not None:
                parts.append(f"Maximum annual income: ₹{int(raw_elig['max_income']):,}")
            if raw_elig.get("gender") and str(raw_elig["gender"]).lower() not in {"any", "all", ""}:
                parts.append(f"Gender: {raw_elig['gender']}")
            if raw_elig.get("caste") and str(raw_elig["caste"]).strip():
                parts.append(f"Caste category: {raw_elig['caste']}")
            if raw_elig.get("occupation") and str(raw_elig["occupation"]).strip():
                parts.append(f"Occupation: {raw_elig['occupation']}")
            eligibility_text = ". ".join(parts) if parts else "Open to eligible applicants. Check official portal for criteria."
        else:
            eligibility_text = _text_or_fallback(
                raw_elig,
                "Eligibility details not available in dataset.",
            )

        cards.append(
            {
                "scheme_id": scheme.get("scheme_id") or name,
                "scheme_name": name,
                "state": scheme_state or "Unknown",
                "category": scheme_category,
                "description": _text_or_fallback(
                    scheme.get("description") or dataset_row.get("description") or benefits_raw,
                    "Description not available in dataset.",
                ),
                "eligibility": scheme.get("eligibility") or dataset_row.get("eligibility"),
                "eligibility_text": eligibility_text,
                "eligible": scheme.get("eligible"),
                "score": float(scheme.get("score") or 0.0),
                "benefits_summary": summarize_benefit(
                    benefits_raw,
                    max_chars=300,
                ),
                "why_match": why_match,
                "documents_required": documents,
                "application_process": scheme.get("application_process") or dataset_row.get("application_process") or [],
                "application_link": scheme.get("application_link") or dataset_row.get("application_link"),
            }
        )

    # ── TRANSLATE CARD TEXT FIELDS INTO USER LANGUAGE ──────────────────────
    lang = _pick_lang(current_profile.get("language"))
    if lang != "en":
        def _t(text: str) -> str:
            if not text or not text.strip():
                return text
            result = _llm_translate(text, lang)
            return result.strip() if result and result.strip() else text

        for card in cards:
            card["description"] = _t(card.get("description") or "")
            card["eligibility_text"] = _t(card.get("eligibility_text") or "")
            card["benefits_summary"] = _t(card.get("benefits_summary") or "")
            # Translate documents list
            raw_docs = card.get("documents_required") or []
            if isinstance(raw_docs, list) and raw_docs:
                translated_docs: list[str] = []
                for doc in raw_docs:
                    td = _t(str(doc))
                    translated_docs.append(td)
                card["documents_required"] = translated_docs
            # Translate application process steps
            raw_steps = card.get("application_process") or []
            if isinstance(raw_steps, list) and raw_steps:
                card["application_process"] = [_t(str(step)) for step in raw_steps]

    cards, geo_rejected_count = _final_geo_filter(cards, current_profile.get("state"))
    if geo_rejected_count:
        logger.info(
            {
                "event": "final_geo_filter_applied",
                "user_state": str(current_profile.get("state") or "") or None,
                "geo_rejected_count": geo_rejected_count,
                "remaining_cards": len(cards),
            }
        )
    return cards

def _build_schemes_payload(
    schemes: list[dict[str, Any]],
    language: str | None,
    fallback_used: bool = False,
    fallback_message: str | None = None,
    profile: dict[str, Any] | None = None,
    profile_changed: bool = False,
    errors: list[str] | None = None,
    visible_limit: int = 5,
) -> dict[str, Any]:
    lang = _pick_lang(language)
    response_confidence = extraction_confidence({"profile": profile or {}})
    # Ensure language is in profile so _normalize_scheme_cards can translate
    profile_with_lang = dict(profile or {})
    profile_with_lang["language"] = lang
    cards = _normalize_scheme_cards(schemes, profile=profile_with_lang, max_cards=visible_limit)
    if not cards:
        no_match = fallback_message or LABELS["no_match"]
        return {"response": no_match, "schemes": [], "fallback_used": fallback_used, "errors": errors or []}

    scheme_ids = [str(card.get("scheme_name") or "").strip() for card in cards if str(card.get("scheme_name") or "").strip()]
    # Only cache English responses; non-English cards are translated per-request
    if lang == "en" and profile and scheme_ids and not profile_changed and not fallback_used:
        cached_response = get_response_cache(profile, scheme_ids, lang)
        if cached_response:
            return {"response": cached_response, "schemes": cards, "fallback_used": fallback_used, "errors": errors or []}

    # ── Use a clean localized header instead of verbose scheme list text ──
    # Full details are in the expandable cards — bubble only needs a short label
    intro = SCHEMES_FOUND_LABEL.get(lang) or SCHEMES_FOUND_LABEL["en"]
    if fallback_message:
        fb = _translate_or_fallback(fallback_message, lang) if lang != "en" else fallback_message
        intro = f"{fb}\n\n{intro}"

    if lang == "en" and profile and scheme_ids and not profile_changed and not fallback_used and intro.strip():
        set_response_cache(profile, scheme_ids, intro, lang, confidence=response_confidence)
    return {"response": intro, "schemes": cards, "fallback_used": fallback_used, "errors": errors or []}


def _save_profile(phone_number: str, merged_profile: dict[str, Any]) -> None:
    payload = dict(merged_profile or {})
    if payload.get("annual_income") is not None:
        payload["income"] = payload.get("annual_income")
    user_model.update_profile_with_priority(phone_number, payload, source="manual")


def _derive_user_goal(intent: str, profile: dict[str, Any]) -> str:
    if intent in {"scheme_search", "check_eligibility"}:
        return "scheme_search"
    if intent == "profile_update":
        return "profile_update"
    if _profile_signal(profile):
        return "profile_update"
    return "general_query"


def _handle_command(
    intent: str,
    phone_number: str,
    language: str | None,
    profile: dict[str, Any],
    message: str,
) -> tuple[dict[str, Any], str]:
    lang = _pick_lang(language)
    profile_for_search = dict(profile or {})
    profile_for_search["language"] = lang

    if intent == "reset":
        user_model.reset_user(phone_number)
        user_model.update_user(
            phone_number,
            {
                "language": None,
                "conv_state": "awaiting_language",
                "last_question_field": None,
                "invalid_language_attempts": 0,
            },
        )
        body = f"{LABELS['response_reset']}\n\n{LABELS['choose_language']}"
        return _result({"response": body, "schemes": [], "fallback_used": False}, "awaiting_language", "en", intent="HELP")

    if intent == "change_language":
        user_model.update_user(
            phone_number,
            {
                "language": None,
                "conv_state": "awaiting_language",
                "last_question_field": None,
                "invalid_language_attempts": 0,
            },
        )
        return _result(
            {"response": LABELS["choose_language"], "schemes": [], "fallback_used": False},
            "awaiting_language",
            "en",
            intent="CHANGE_LANGUAGE",
        )

    if intent == "help":
        return _result({"response": LABELS["help"], "schemes": [], "fallback_used": False}, "active", lang, intent="HELP")

    if intent == "check_eligibility":
        if _enough_for_results(profile):
            retrieval_query = _build_retrieval_query(message, profile_for_search)
            direct_cards = _query_schemes_direct(retrieval_query, profile_for_search, limit=5)
            if direct_cards:
                cards = _normalize_scheme_cards(direct_cards, profile=profile_for_search)
                user_model.update_user(phone_number, {"last_schemes": cards, "last_schemes_cursor": len(cards), "selected_scheme": None})
                summary = _format_eligibility_response(profile_for_search, cards)
                return _result(
                    {"response": summary, "schemes": cards, "fallback_used": False, "errors": []},
                    "showing_schemes",
                    lang,
                    intent="CHECK_ELIGIBILITY",
                )
            result = recommend_schemes(profile_for_search, query=retrieval_query, top_k=5)
            cards = _normalize_scheme_cards(result.get("schemes") or [], profile=profile_for_search)
            if cards:
                user_model.update_user(phone_number, {"last_schemes": cards, "last_schemes_cursor": len(cards), "selected_scheme": None})
            summary = _format_eligibility_response(profile_for_search, cards)
            if not cards and result.get("fallback_message"):
                summary = f"{result.get('fallback_message')}\n\n{summary}"
            return _result(
                {"response": summary, "schemes": cards, "fallback_used": bool(result.get("fallback_used")), "errors": result.get("errors") or []},
                "showing_schemes",
                lang,
                intent="CHECK_ELIGIBILITY",
            )
        next_field, next_question_english = _decide_next_field(
            profile,
            user_goal="scheme_search",
            previous_question=None,
            last_user_message=message,
        )
        next_field = next_field or "category"
        user_model.update_user(phone_number, {"last_question_field": next_field, "conv_state": "collecting_profile"})
        question = _render_followup_question(next_field, lang, next_question_english)
        return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", lang, intent="CHECK_ELIGIBILITY")

    return _result({"response": LABELS["help"], "schemes": [], "fallback_used": False}, "active", lang, intent="HELP")


def _query_schemes_direct(query: str, profile: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    dataset = load_scheme_dataset()
    if not dataset:
        return []

    words = [w for w in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(w) >= 3]
    if not words:
        return []

    from engine.orchestrator import _to_taxonomy as _orch_taxonomy
    user_state   = _norm_state(profile.get("state"))
    user_category = profile.get("category", "")
    # Normalize user category to taxonomy slug
    user_tax = _orch_taxonomy(user_category) if user_category else "unknown"

    scored: list[tuple[int, dict[str, Any]]] = []

    for scheme in dataset:
        # Quality gate: skip garbage
        name = str(scheme.get("scheme_name") or "").strip()
        if not name or name.lower() in {"unnamed scheme", "unknown scheme"}:
            continue
        benefits = str(scheme.get("benefits") or scheme.get("description") or "").strip()
        if not benefits or len(benefits) < 20 or "no description available" in benefits.lower():
            continue

        # Category filter using taxonomy (e.g. 'women & child' → women_child)
        if user_tax != "unknown":
            scheme_tax = _orch_taxonomy(scheme.get("category"))
            if scheme_tax != user_tax:
                continue

        # State filter: match user state OR All India
        if user_state and not _state_allowed(user_state, scheme.get("state")):
            continue

        haystack = " ".join(str(scheme.get(k, "")) for k in ["scheme_name", "description", "benefits", "category"]).lower()
        score = sum(1 for word in words if word in haystack)
        scored.append((score, scheme))  # include even score=0 — category match is enough

    scored.sort(key=lambda item: item[0], reverse=True)
    direct = []
    for _, scheme in scored[:limit]:
        direct.append(
            {
                "scheme_id": scheme.get("scheme_id") or scheme.get("scheme_name", "Unknown Scheme"),
                "scheme_name": scheme.get("scheme_name", "Unknown Scheme"),
                "state": scheme.get("state"),
                "category": scheme.get("category"),
                "description": scheme.get("description"),
                "eligibility": scheme.get("eligibility"),
                "benefits_summary": summarize_benefit(scheme.get("benefits") or scheme.get("description"), max_chars=300),
                "documents_required": scheme.get("documents_required", []),
                "why_match": ["Category and state aligned with your provided details."],
                "application_link": scheme.get("application_link"),
            }
        )
    return direct


def _handle_apply_scheme(
    phone_number: str,
    language: str | None,
    profile: dict[str, Any],
    message: str,
    user: dict[str, Any],
    scheme_name_hint: str | None = None,
) -> tuple[dict[str, Any], str]:
    lang = _pick_lang(language)
    chosen = _resolve_selected_scheme(message, user, scheme_name_hint=scheme_name_hint)
    if not chosen:
        last_schemes = user.get("last_schemes") if isinstance(user.get("last_schemes"), list) else []
        if last_schemes:
            names = [f"{idx}. {str(card.get('scheme_name') or 'Unknown Scheme')}" for idx, card in enumerate(last_schemes[:5], start=1)]
            prompt = "Intent: APPLY_SCHEME\nPlease select one scheme by number:\n" + "\n".join(names)
            return _result(
                {"response": prompt, "schemes": last_schemes[:5], "fallback_used": False},
                "showing_schemes",
                lang,
                intent="APPLY_SCHEME",
            )
        return _result(
            {
                "response": "Intent: APPLY_SCHEME\nNo previous scheme list found. Please run scheme search first.",
                "schemes": [],
                "fallback_used": False,
            },
            "active",
            lang,
            intent="APPLY_SCHEME",
        )

    normalized = _normalize_scheme_cards([chosen], profile=profile)
    if not normalized:
        return _result(
            {
                "response": "Intent: APPLY_SCHEME\nSelected scheme details are not available in the current dataset.",
                "schemes": [],
                "fallback_used": False,
            },
            "active",
            lang,
            intent="APPLY_SCHEME",
        )

    selected_card = normalized[0]
    user_model.update_user(
        phone_number,
        {"selected_scheme": selected_card, "last_question_field": None, "conv_state": "showing_schemes"},
    )
    response_text = _format_apply_scheme_response(selected_card, profile)
    return _result(
        {"response": response_text, "schemes": [selected_card], "fallback_used": False},
        "showing_schemes",
        lang,
        intent="APPLY_SCHEME",
    )


def _handle_more_schemes(
    phone_number: str,
    language: str | None,
    profile: dict[str, Any],
    user: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    lang = _pick_lang(language)
    last_schemes = user.get("last_schemes") if isinstance(user.get("last_schemes"), list) else []
    cursor = int(user.get("last_schemes_cursor") or len(last_schemes))

    if last_schemes and cursor < len(last_schemes):
        next_batch = list(last_schemes[cursor: cursor + 5])
        user_model.update_user(phone_number, {"last_schemes_cursor": cursor + len(next_batch), "conv_state": "showing_schemes"})
        return _result(
            {"response": _translate_or_fallback("Here are more matching schemes:", lang), "schemes": next_batch, "fallback_used": False},
            "showing_schemes",
            lang,
            intent="SEARCH_SCHEMES",
        )

    if not _enough_for_results(profile):
        return _result(
            {"response": _translate_or_fallback("No more schemes found.", lang), "schemes": [], "fallback_used": False},
            user.get("conv_state") or "active",
            lang,
            intent="SEARCH_SCHEMES",
        )

    retrieval_query = _build_retrieval_query("", profile)
    result = recommend_schemes(profile, query=retrieval_query, top_k=15)
    all_cards = _normalize_scheme_cards(result.get("schemes") or [], profile=profile, max_cards=None)
    if not all_cards:
        return _result(
            {"response": _translate_or_fallback("No more schemes found.", lang), "schemes": [], "fallback_used": bool(result.get("fallback_used")), "errors": result.get("errors") or []},
            "showing_schemes",
            lang,
            intent="SEARCH_SCHEMES",
        )

    seen_names = {normalize_text_light(str(card.get("scheme_name") or "")) for card in last_schemes}
    unseen = [card for card in all_cards if normalize_text_light(str(card.get("scheme_name") or "")) not in seen_names]
    if not unseen:
        return _result(
            {"response": _translate_or_fallback("No more schemes found.", lang), "schemes": [], "fallback_used": False},
            "showing_schemes",
            lang,
            intent="SEARCH_SCHEMES",
        )

    next_batch = unseen[:5]
    merged_cards = list(last_schemes) + next_batch
    user_model.update_user(
        phone_number,
        {
            "last_schemes": merged_cards,
            "last_schemes_cursor": len(merged_cards),
            "selected_scheme": None,
            "conv_state": "showing_schemes",
        },
    )
    return _result(
        {"response": _translate_or_fallback("Here are more matching schemes:", lang), "schemes": next_batch, "fallback_used": bool(result.get("fallback_used")), "errors": result.get("errors") or []},
        "showing_schemes",
        lang,
        intent="SEARCH_SCHEMES",
    )


def handle_message(phone_number: str, message: str) -> tuple[dict[str, Any], str]:
    text = sanitize_text(message)
    user, _ = user_model.create_or_get_user(phone_number)
    profile = dict(user.get("profile") or {})
    language = user.get("language")
    conv_state = user.get("conv_state")
    expected_field = user.get("last_question_field")
    category = str(profile.get("category") or "").strip().lower()
    occupation_required = category in {"employment", "finance_business"}
    if expected_field == "occupation" and not occupation_required:
        expected_field = None
        user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "active"})
    logger.info({"event": "request_start", "phone": phone_number, "language_locked": language, "state": conv_state, "expected_field": expected_field})

    # is_collecting_field: True when bot is actively waiting for a field answer.
    # Use BOTH explicit expected_field AND conv_state as guards so that even
    # if last_question_field was not persisted, collecting state still protects
    # against digit-menu hijacking.
    is_collecting_field = (
        (bool(expected_field) or conv_state == "collecting_profile")
        and conv_state != "awaiting_language"
    )
    is_expecting_numeric = is_collecting_field and expected_field in EXPECTING_NUMERIC_FIELDS

    # ── PRIORITY 1: Explicit text commands always fire regardless of state ──
    # Only exact keyword phrases (not digits) trigger global actions here.
    explicit_intent = _detect_explicit_intent(text)
    if explicit_intent in {"reset", "change_language", "help"}:
        return _handle_command(explicit_intent, phone_number, language, profile, text)
    # ── PRIORITY 2: Numeric profile fields keep literal numeric input ──
    _single_digit = str(text or "").strip()
    _digit_has_context = is_expecting_numeric
    if _single_digit in DIGIT_MENU_INTENTS and is_expecting_numeric:
        logger.info({
            "event": "contextual_digit_resolved",
            "digit": _single_digit,
            "resolved_field": expected_field,
            "resolved_value": _single_digit,
        })

    # ── PRIORITY 3: Digit-menu actions ──
    digit_menu_intent = _detect_digit_menu_intent(text)
    if digit_menu_intent in {"reset", "change_language", "help"} and not is_expecting_numeric:
        return _handle_command(digit_menu_intent, phone_number, language, profile, text)

    # ── PRIORITY 4: Invalid digit fallback for removed option 4 / unsupported menu digits ──
    if _single_digit.isdigit() and not is_expecting_numeric and _single_digit not in DIGIT_MENU_INTENTS:
        return _result(
            {"response": LABELS["invalid_menu_digit"], "schemes": [], "fallback_used": False},
            conv_state or "active",
            language,
            intent="HELP",
        )

    has_scheme_context = bool(user.get("selected_scheme")) or bool(user.get("last_schemes"))
    if _is_more_request(text) and has_scheme_context:
        return _handle_more_schemes(phone_number, language, profile, user)
    if _is_apply_request(text) and has_scheme_context:
        return _handle_apply_scheme(phone_number, language, profile, text, user)

    bootstrap_extraction: tuple[str, dict[str, Any], dict[str, Any], str] | None = None

    if not language or conv_state == "awaiting_language":
        lowered = text.strip().lower()
        logger.info({"event": "llm_call", "reason": "language_inference", "state": conv_state, "expected_field": expected_field})
        language_pick = infer_language_selection_llm(text)
        chosen = language_pick.get("selected_language") if isinstance(language_pick, dict) else None
        if not chosen:
            chosen = _fallback_language_from_text(lowered)
        logger.info({"event": "language_detect", "input": text, "detected_language": chosen})
        if chosen in SUPPORTED_LANGUAGES:
            logger.info({"event": "language_locked", "phone": phone_number, "selected_language": chosen})
            user_model.update_user(
                phone_number,
                {
                    "language": chosen,
                    "conv_state": "collecting_profile",
                    "last_question_field": "category",
                    "invalid_language_attempts": 0,
                },
            )
            # Use static cache for language confirmation; fall back to translation
            lang_name = SUPPORTED_LANGUAGES.get(chosen, "English")
            lang_set = _get_static(chosen, "language_set", lang_name=lang_name)
            if not lang_set:
                lang_set = LABELS["language_set"].format(lang_name=lang_name)
                lang_set = _translate_or_fallback(lang_set, chosen)
            return _result({
                "response": f"{lang_set}\n\n{_field_question('category', chosen)}",
                "schemes": [],
                "fallback_used": False,
            }, "collecting_profile", chosen, intent="CHANGE_LANGUAGE")

        # Skip bootstrap for trivial greetings ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â force language selection
        _trivial = {"start", "hi", "hello", "hey", "hii", "helo", "namaste", "namaskar", "vanakkam"}
        if lowered in _trivial:
            user_model.update_user(phone_number, {"conv_state": "awaiting_language", "last_question_field": None})
            return _result(
                {"response": LABELS["choose_language"], "schemes": [], "fallback_used": False},
                "awaiting_language",
                "en",
                intent="CHANGE_LANGUAGE",
            )

        boot_english = text
        boot_intent, boot_profile, boot_llm = _extract_profile(
            original_text=text,
            english_text=boot_english,
            expected_field=expected_field,
            language=None,
            current_profile=profile,
        )
        # Only bootstrap if real profile data or explicit scheme intent (NOT bare general_query)
        has_profile = _profile_signal(boot_profile)
        has_explicit_intent = boot_intent in {"profile_update", "scheme_search", "check_eligibility"}
        if has_profile or has_explicit_intent:
            inferred_lang = _pick_lang(boot_llm.get("language") or "en")
            user_model.update_user(phone_number, {"language": inferred_lang, "conv_state": "active", "last_question_field": expected_field})
            language = inferred_lang
            bootstrap_extraction = (boot_intent, boot_profile, boot_llm, boot_english)
        else:
            invalid_attempts = int(user.get("invalid_language_attempts") or 0) + 1
            user_model.update_user(
                phone_number,
                {"conv_state": "awaiting_language", "last_question_field": None, "invalid_language_attempts": invalid_attempts},
            )
            hint = ""
            if invalid_attempts >= 2:
                hint = "\n\nPlease type full language name like Kannada, Tamil, Hindi."
            return _result({
                "response": f"{LABELS['choose_language']}{hint}",
                "schemes": [],
                "fallback_used": False,
            }, "awaiting_language", "en", intent="CHANGE_LANGUAGE")

    language = _pick_lang(language)
    logger.info({"event": "language_locked", "phone": phone_number, "selected_language": language})
    if user.get("language") and user.get("language") != language:
        user_model.update_user(phone_number, {"language": language})
    if bootstrap_extraction:
        llm_intent, extracted_profile, llm_data, english_text = bootstrap_extraction
    else:
        # If expected_field is None but we are in collecting_profile state, derive
        # the effective field from _fallback_next_field so contextual digit inputs
        # (e.g., category shortcut 1-7) resolve correctly even when last_question_field
        # was not explicitly persisted to the DB.
        _effective_field = expected_field
        if not _effective_field and conv_state == "collecting_profile":
            _effective_field = _fallback_next_field(profile)
            if _effective_field:
                logger.info({
                    "event": "effective_field_inferred",
                    "inferred_field": _effective_field,
                    "conv_state": conv_state,
                })
        fast_extract = _fast_extract_expected_field(_effective_field, text)
        if fast_extract:
            english_text = str(text or "")
            llm_intent = "profile_update"
            extracted_profile = {field: None for field in PROFILE_FIELDS}
            extracted_profile.update(fast_extract)
            llm_data = {"confidence": 1.0, "answer_english": None}
        else:
            english_text = _translate_to_english_safe(text, language)
            logger.info({"event": "translated_to_english", "source_language": language, "english_message": english_text[:80]})
            llm_intent, extracted_profile, llm_data = _extract_profile(
                original_text=text,
                english_text=english_text,
                expected_field=expected_field,
                language=language,
                current_profile=profile,
            )

    logger.info({"event": "extracted_profile", "intent": llm_intent, "profile": extracted_profile, "llm_confidence": llm_data.get("confidence")})
    scheme_name_hint = str(llm_data.get("scheme_name") or "").strip() if isinstance(llm_data, dict) else ""
    public_intent = _public_intent_from_internal(llm_intent, llm_data if isinstance(llm_data, dict) else None)

    if llm_intent in {"reset", "change_language", "help", "check_eligibility"}:
        return _handle_command(llm_intent, phone_number, language, profile, text)

    # Explicit routing requested:
    # if intent == "APPLY_SCHEME": return apply_handler()
    # elif intent == "SEARCH_SCHEMES": return search_handler()
    if public_intent == "APPLY_SCHEME":
        return _handle_apply_scheme(phone_number, language, profile, text, user, scheme_name_hint=scheme_name_hint or None)
    elif public_intent == "SEARCH_SCHEMES":
        llm_intent = "scheme_search"
    elif public_intent == "CHECK_ELIGIBILITY":
        return _handle_command("check_eligibility", phone_number, language, profile, text)
    elif public_intent == "CHANGE_LANGUAGE":
        return _handle_command("change_language", phone_number, language, profile, text)
    elif public_intent == "HELP":
        return _handle_command("help", phone_number, language, profile, text)

    merged_profile, changed_fields = _merge_profile(profile, extracted_profile)
    merged_profile["language"] = language
    _save_profile(phone_number, merged_profile)
    reset_flags: dict[str, Any] = {"conv_state": "active"}
    user_model.update_user(phone_number, reset_flags)
    logger.info({"event": "profile_updated", "changed_fields": changed_fields, "profile": merged_profile})

    logger.info(
        {
            "event": "turn",
            "phone": phone_number,
            "intent": llm_intent,
            "language": language,
            "changed_fields": changed_fields,
            "category": merged_profile.get("category"),
            "confidence": llm_data.get("confidence"),
        }
    )

    # If the expected field has been successfully filled, clear it and advance
    if expected_field and merged_profile.get(expected_field) not in (None, ""):
        expected_field = None
        user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "active"})

    goal = _derive_user_goal(llm_intent, merged_profile)
    next_field, next_question_english = _decide_next_field(
        merged_profile,
        user_goal=goal,
        previous_question=expected_field,
        last_user_message=english_text,
    )
    logger.info({"event": "next_field_selected", "next_field": next_field, "goal": goal})

    answer_english = str(llm_data.get("answer_english") or "").strip()

    if llm_intent == "general_query" and answer_english:
        # Force translation for answer + follow-up branch to prevent mixed-language leakage.
        answer_translated = _translate_or_fallback(answer_english, language)
        response_chunks = [answer_translated]
        if expected_field and _is_missing(merged_profile, expected_field):
            resume_question = _field_question(expected_field, language)
            response_chunks.append(resume_question)
            user_model.update_user(phone_number, {"last_question_field": expected_field, "conv_state": "collecting_profile"})
            return _result(
                {"response": "\n\n".join(response_chunks), "schemes": [], "fallback_used": False},
                "collecting_profile",
                language,
                intent="HELP",
            )
        if next_field:
            user_model.update_user(phone_number, {"last_question_field": next_field, "conv_state": "collecting_profile"})
            response_chunks.append(_render_followup_question(next_field, language, next_question_english))
            return _result(
                {"response": "\n\n".join(response_chunks), "schemes": [], "fallback_used": False},
                "collecting_profile",
                language,
                intent="HELP",
            )
        return _result({"response": "\n\n".join(response_chunks), "schemes": [], "fallback_used": False}, "active", language, intent="HELP")

    if goal == "general_query":
        if _enough_for_results(merged_profile):
            retrieval_query = _build_retrieval_query(english_text, merged_profile)
            direct_cards = _query_schemes_direct(retrieval_query, merged_profile, limit=3)
            if direct_cards:
                payload = _build_schemes_payload(
                    direct_cards,
                    language,
                    fallback_used=False,
                    fallback_message=None,
                    profile=merged_profile,
                    profile_changed=bool(changed_fields),
                    errors=[],
                )
                if payload.get("schemes"):
                    user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
                user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "active"})
                return _result(
                    payload,
                    "active",
                    language,
                    intent="SEARCH_SCHEMES",
                )
            result = recommend_schemes(merged_profile, query=retrieval_query, top_k=3)
            payload = _build_schemes_payload(
                result.get("schemes") or [],
                language,
                fallback_used=bool(result.get("fallback_used")),
                fallback_message=result.get("fallback_message"),
                profile=merged_profile,
                profile_changed=bool(changed_fields),
                errors=result.get("errors") or [],
            )
            if payload.get("schemes"):
                user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
            user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "active"})
            return _result(payload, "active", language, intent="SEARCH_SCHEMES")
        if next_field:
            user_model.update_user(phone_number, {"last_question_field": next_field, "conv_state": "collecting_profile"})
            question = _render_followup_question(next_field, language, next_question_english)
            return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", language, intent="HELP")
        forced_next = _fallback_next_field(merged_profile)
        if forced_next:
            user_model.update_user(phone_number, {"last_question_field": forced_next, "conv_state": "collecting_profile"})
            question = _render_followup_question(forced_next, language, None)
            return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", language, intent="SEARCH_SCHEMES")
        return _result({"response": LABELS["help"], "schemes": [], "fallback_used": False}, "active", language, intent="HELP")

    if goal == "scheme_search":
        if next_field:
            user_model.update_user(phone_number, {"last_question_field": next_field, "conv_state": "collecting_profile"})
            question = _render_followup_question(next_field, language, next_question_english)
            return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", language, intent="SEARCH_SCHEMES")
        retrieval_query = _build_retrieval_query(english_text, merged_profile)
        direct_cards = _query_schemes_direct(retrieval_query, merged_profile, limit=5)
        if direct_cards:
            payload = _build_schemes_payload(
                direct_cards,
                language,
                fallback_used=False,
                fallback_message=None,
                profile=merged_profile,
                profile_changed=bool(changed_fields),
                errors=[],
            )
            if payload.get("schemes"):
                user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
            user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "showing_schemes"})
            return _result(
                payload,
                "showing_schemes",
                language,
                intent="SEARCH_SCHEMES",
            )
        result = recommend_schemes(merged_profile, query=retrieval_query, top_k=5)
        payload = _build_schemes_payload(
            result.get("schemes") or [],
            language,
            fallback_used=bool(result.get("fallback_used")),
            fallback_message=result.get("fallback_message"),
            profile=merged_profile,
            profile_changed=bool(changed_fields),
            errors=result.get("errors") or [],
        )
        if payload.get("schemes"):
            user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
        user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "showing_schemes"})
        return _result(
            payload,
            "showing_schemes",
            language,
            intent="SEARCH_SCHEMES",
        )

    # profile_update (default path)
    if next_field:
        user_model.update_user(phone_number, {"last_question_field": next_field, "conv_state": "collecting_profile"})
        question = _render_followup_question(next_field, language, next_question_english)
        return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", language, intent="SEARCH_SCHEMES")

    if _enough_for_results(merged_profile):
        retrieval_query = _build_retrieval_query(english_text, merged_profile)
        direct_cards = _query_schemes_direct(retrieval_query, merged_profile, limit=5)
        if direct_cards:
            payload = _build_schemes_payload(
                direct_cards,
                language,
                fallback_used=False,
                fallback_message=None,
                profile=merged_profile,
                profile_changed=bool(changed_fields),
                errors=[],
            )
            if payload.get("schemes"):
                user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
            user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "showing_schemes"})
            return _result(
                payload,
                "showing_schemes",
                language,
                intent="SEARCH_SCHEMES",
            )
        result = recommend_schemes(merged_profile, query=retrieval_query, top_k=5)
        payload = _build_schemes_payload(
            result.get("schemes") or [],
            language,
            fallback_used=bool(result.get("fallback_used")),
            fallback_message=result.get("fallback_message"),
            profile=merged_profile,
            profile_changed=bool(changed_fields),
            errors=result.get("errors") or [],
        )
        if payload.get("schemes"):
            user_model.update_user(phone_number, {"last_schemes": payload.get("schemes"), "last_schemes_cursor": len(payload.get("schemes") or []), "selected_scheme": None})
        user_model.update_user(phone_number, {"last_question_field": None, "conv_state": "showing_schemes"})
        return _result(
            payload,
            "showing_schemes",
            language,
            intent="SEARCH_SCHEMES",
        )

    forced_next = _fallback_next_field(merged_profile)
    if forced_next:
        user_model.update_user(phone_number, {"last_question_field": forced_next, "conv_state": "collecting_profile"})
        question = _render_followup_question(forced_next, language, None)
        return _result({"response": question, "schemes": [], "fallback_used": False}, "collecting_profile", language, intent="SEARCH_SCHEMES")

    return _result({"response": LABELS["help"], "schemes": [], "fallback_used": False}, "active", language, intent="HELP")






