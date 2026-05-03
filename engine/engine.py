# -*- coding: utf-8 -*-
"""
LLM-controlled understanding helpers for language inference, profile extraction,
and follow-up question planning.

Prompt architecture:
  - EXTRACTION_PROMPT  : machine output (JSON only, deterministic, cacheable)
  - CONVERSATION_PROMPT: UI output (controlled formatting, backend-driven)
  - Gemini = source of truth for category/intent
  - Embeddings = fallback only
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import get_close_matches, SequenceMatcher
from typing import Any

from core.logger import get_logger
from core.sanitizer import sanitize_for_llm
from engine.llm_router import router
from services.cache_service import get_extraction_cache, set_extraction_cache
from services.embedding_service import semantic_match
from services.translation_service import translate_from_english

logger = get_logger("engine.engine")


# MASTER_RESPONSE_TEMPLATE removed — dead code (V3 split prompt architecture)


LANGUAGE_SELECTION_PROMPT = """You are a language selection interpreter for an Indian multilingual chatbot.

The user may type a language name with typo, transliteration, abbreviation, or native script.
Infer which supported language the user wants.

Supported languages:
English=en
Hindi=hi
Bengali=bn
Gujarati=gu
Kannada=kn
Tamil=ta
Telugu=te
Marathi=mr
Malayalam=ml
Punjabi=pa
Odia=or
Assamese=as
Urdu=ur

Examples:
"kannad" -> kn
"kanada" -> kn
"ಕನ್ನಡ" -> kn
"বাংলা" -> bn
"தமிழ்" -> ta
"اردو" -> ur

Return JSON only:
{
  "selected_language": "en|hi|bn|gu|kn|ta|te|mr|ml|pa|or|as|ur|null",
  "confidence": 0.0
}

Rules:
- If clearly a supported language, return code.
- If uncertain, return null.
- Do not infer profile details here.
"""


PROFILE_EXTRACTION_PROMPT = """You are YojnaSetu Meaning Extractor.

Extract user meaning into JSON only.

Return exactly:
{
  "intent": "SEARCH_SCHEMES|APPLY_SCHEME|CHECK_ELIGIBILITY|HELP|CHANGE_LANGUAGE|profile_update|scheme_search|search_schemes|apply_scheme|general_query|reset|change_language|help|check_eligibility|unknown",
  "scheme_name": string|null,
  "entities": {
    "age": number|null,
    "income": number|null,
    "gender": string|null,
    "occupation": string|null,
    "state": string|null
  },
  "language": "en|hi|bn|gu|kn|ta|te|mr|ml|pa|or|as|ur|null",
  "category": "Education|Health|Agriculture|Employment|Housing|Finance|Women & Child|Senior Citizen|Disability|Social Welfare|Unknown|null",
  "state": string|null,
  "age": number|null,
  "income": number|null,
  "occupation": string|null,
  "gender": string|null,
  "education_level": string|null,
  "caste_category": string|null,
  "academic_percentage": number|null,
  "bpl_status": true|false|null,
  "answer_english": string|null,
  "confidence": number
}

Rules:
- JSON only. No prose.
- Do not recommend schemes.
- Use semantic understanding from user text.
- Do not rely on static keyword mapping.
- Normalize misspellings like gujrat -> Gujarat, kerela -> Kerala.
- Normalize income like 4lakh -> 400000.
- If user message clearly implies a domain, set category.
- If user message clearly implies a role, set occupation.
- If uncertain, keep field null.
"""

CONVERSATION_PROMPT = """SYSTEM ROLE:
You are YojnaSetuBot \u2014 a STRICT conversational controller.

You support FREE SPEECH input.
But conversation flow is ALWAYS controlled.

FREE SPEECH RULE:
User can say ANYTHING \u2014 full sentence, partial info, mixed language, random order.
You MUST extract all possible data, update profile, continue conversation.

FOLLOW-UP ENFORCEMENT:
You MUST ALWAYS ask a follow-up question UNTIL all required fields are filled.
Required fields: category, state, age, income.
Ask ONLY ONE question at a time. Ask ONLY missing field.

LANGUAGE LOCK:
User language = {user_language}
ALL responses MUST be in this language. NO English leakage.

SMART EXTRACTION:
If user provides multiple fields at once, extract ALL and only ask remaining missing fields.

PRIORITY OF QUESTIONS:
1. category 2. state 3. age 4. income

LOOP PREVENTION:
If user gives ANY answer \u2192 ACCEPT IT \u2192 UPDATE FIELD \u2192 MOVE TO NEXT FIELD.
NEVER repeat same question if answer exists.

INTENT HANDLING:
If user goes off-topic, respond briefly then RETURN to flow.

SCHEME OUTPUT FORMAT:
\u1f539 Scheme Name: <name>
\u1f4cd Benefit: <one line>
\u2705 Why this matches: <reason>
\u1f4c4 Documents Required: <docs>

FAILSAFE:
If confusion \u2192 Ask next missing field in correct language.
NEVER stop conversation. NEVER switch language.
"""

# Legacy alias for backward compatibility
NEXT_QUESTION_PROMPT = CONVERSATION_PROMPT

INTENT_CLASSIFIER_PROMPT = """SYSTEM ROLE:
You are a category normalization classifier. Return JSON only.

CLASSIFY into exactly ONE canonical category:
education, health, agriculture, finance_business, women_child, housing,
senior_citizen, disability, employment, social_welfare, unknown

TRANSLITERATION RULES:
vruddh/vridh/pension/old age -> senior_citizen
mahila/widow/vidhwa/women -> women_child
kisan/farmer/agriculture/khet -> agriculture
student/scholarship/vidyarthi -> education
job/skill/rojgar/training -> employment
loan/finance/business/udyam -> finance_business
awas/house/housing -> housing
aarogya/medical/hospital -> health
divyang/viklang/disabled -> disability

Return JSON only:
{
  "canonical_category": "<category>",
  "subcategory": "short english phrase or null",
  "intent_keywords": ["up to 5 normalized english keywords"],
  "confidence": 0.0,
  "reason": "short reason"
}

Rules:
- Understand typos, mixed scripts, transliteration.
- DO NOT output scheme names.
- DO NOT decide eligibility.
- If uncertain, return unknown with low confidence.
- NEVER output text outside JSON.
"""


ALLOWED_PROFILE_INTENTS = {
    "search_schemes",
    "apply_scheme",
    "check_eligibility",
    "help",
    "change_language",
    "profile_update",
    "scheme_search",
    "general_query",
    "reset",
    "unknown",
}

ALLOWED_LANGUAGE_CODES = {"en", "hi", "gu", "ta", "te", "bn", "kn", "mr", "ml", "pa", "or", "as", "ur"}

ALLOWED_NEXT_FIELDS = {
    "category",
    "state",
    "age",
    "income",
    "occupation",
    "education_level",
    "academic_percentage",
    "caste_category",
    "gender",
    "bpl_status",
    "null",
}

CONTROLLED_TAXONOMY = {
    "education",
    "health",
    "agriculture",
    "finance_business",
    "women_child",
    "housing",
    "senior_citizen",
    "disability",
    "employment",
    "social_welfare",
    "unknown",
}

TAXONOMY_DESCRIPTIONS = {
    "education": (
        "scholarships students school college education learning support "
        "student scholarship \u0935\u093f\u0926\u094d\u092f\u093e\u0930\u094d\u0925\u0940 \u091b\u093e\u0924\u094d\u0930 \u091b\u093e\u0924\u094d\u0930\u0935\u0943\u0924\u094d\u0924\u093f \u0ab5\u0abf\u0aa7\u0abe\u0ab0\u0acd\u0aa5\u0ac0 \u0ab6\u0abf\u0ab7\u0acd\u0aaf\u0ab5\u0ac3\u0aa4\u0acd\u0aa4\u0abf "
        "\u0cb5\u0cbf\u0ca6\u0ccd\u0caf\u0cbe\u0cb0\u0ccd\u0ca5\u0cbf \u0cb5\u0cbf\u0ca6\u0ccd\u0caf\u0cbe\u0cb0\u0ccd\u0ca5\u0cbf\u0cb5\u0cc7\u0ca4\u0ca8 \u0cae\u0cbe\u0ca3\u0cb5\u0cb0 \u0c95\u0cb2\u0ccd\u0cb5\u0cbf \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e\u0c30\u0c4d\u0c25\u0c3f \u0c35\u0c3f\u0c26\u0c4d\u0c2f \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e\u0c30\u0c4d\u0c25\u0c3f \u0c35\u0c43\u0c24\u0c4d\u0c24\u0c3f"
    ),
    "health": (
        "health medical hospital treatment aarogya arogya healthcare insurance "
        "\u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0907\u0932\u093e\u091c \u0905\u0938\u094d\u092a\u0924\u093e\u0932 \u0a86\u0ab0\u0acb\u0a97\u0acd\u0aaf \u0ab9\u0acb\u0ab8\u0acd\u0aaa\u0abf\u0a9f\u0ab2 \u0a86\u0ab0\u0acb\u0a97\u0acd\u0aaf \u0c86\u0cb8\u0ccd\u0caa\u0ca4\u0ccd\u0cb0\u0cc6 \u0bae\u0bb0\u0bc1\u0ba4\u0bcd\u0ba4\u0bc1\u0bb5\u0bae\u0bcd \u0b9a\u0bc1\u0b95\u0bbe\u0ba4\u0bbe\u0bb0\u0bae\u0bcd "
        "\u0c06\u0c30\u0c4b\u0c17\u0c4d\u0c2f\u0c02 \u0c35\u0c48\u0c26\u0c4d\u0c2f\u0c02 \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u091a\u093f\u0915\u093f\u0924\u094d\u0938\u093e \u0635\u062d\u062a"
    ),
    "agriculture": (
        "farmer kisan khedut agriculture farming crop irrigation livestock raitha "
        "\u0915\u093f\u0938\u093e\u0928 \u0915\u0943\u0937\u093f \u0a96\u0ac7\u0aa1\u0ac2\u0aa4 \u0a96\u0ac7\u0aa4\u0ac0 \u0c30\u0c48\u0c24 \u0c95\u0cc3\u0cb7\u0cbf \u0bb5\u0bbf\u0bb5\u0b9a\u0bbe\u0baf\u0bbf \u0bb5\u0bc7\u0bb3\u0bbe\u0ba3\u0bcd\u0bae\u0bc8 \u0c30\u0c48\u0c24\u0c41 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f\u0c02 \u0995\u09c3\u09b7\u0995 \u0995\u09c3\u09b7\u09bf"
    ),
    "finance_business": (
        "loan finance subsidy business entrepreneur startup msme self employment udyami udyog "
        "\u0935\u094d\u092f\u0935\u0938\u093e\u092f \u0909\u0926\u094d\u092f\u092e\u0940 \u090b\u0923 \u0938\u094d\u0935\u0930\u094b\u091c\u0917\u093e\u0930 \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf \u0a89\u0aa6\u0acd\u0aaf\u0acb\u0a97\u0ab8\u0abe\u0ab9\u0ab8\u0abf\u0a95 \u0ab2\u0acb\u0aa8 \u0c89\u0ca6\u0ccd\u0caf\u0cae\u0cbf \u0cb5\u0ccd\u0caf\u0cbe\u0caa\u0cbe\u0cb0 \u0cb8\u0cbe\u0cb2 \u0cb8\u0ccd\u0cb5 \u0c89\u0ca6\u0ccd\u0caf\u0ccb\u0c97 "
        "\u0ba4\u0bca\u0bb4\u0bbf\u0bb2\u0bcd \u0ba4\u0bca\u0bb4\u0bbf\u0bb2\u0bae\u0bc1\u0ba9\u0bc8\u0bb5\u0bcb\u0bb0\u0bcd \u0b95\u0b9f\u0ba9\u0bcd \u0c35\u0c4d\u0c2f\u0c3e\u0c2a\u0c3e\u0c30\u0c02 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c4d\u0c25\u0c3e\u0c2a\u0c15\u0c41\u0c21\u0c41 \u0c05\u0c2a\u0c4d\u0c2a\u0c41 \u09ac\u09cd\u09af\u09ac\u09b8\u09be \u0989\u09a6\u09cd\u09af\u09cb\u0995\u09cd\u09a4\u09be \u098b\u09a3"
    ),
    "women_child": (
        "women woman widow vidhwa vidhva mahila mother girl child welfare "
        "\u092e\u0939\u093f\u0932\u093e \u0935\u093f\u0927\u0935\u093e \u0aae\u0ab9\u0abf\u0ab2\u0abe \u0c35\u0c3f\u0aa7\u0ab5\u0abe \u0cae\u0cb9\u0cbf\u0cb3\u0cc6 \u0cb5\u0cbf\u0ca7\u0cb5\u0cc6 \u0baa\u0bc6\u0ba3\u0bcd \u0bb5\u0bbf\u0ba4\u0bb5\u0bc8 \u0c2e\u0c39\u0c3f\u0c33 \u0c35\u0c3f\u0c24\u0c02\u0c24\u0c41\u0c35\u0c41 \u09ae\u09b9\u09bf\u09b2\u09be \u09ac\u09bf\u09a7\u09ac\u09be"
    ),
    "housing": (
        "housing house home awas shelter "
        "\u0906\u0935\u093e\u0938 \u0918\u0930 \u0a86\u0ab5\u0abe\u0ab8 \u0a98\u0ab0 \u0cb5\u0cb8\u0ca4\u0cbf \u0cae\u0ca8\u0cc6 \u0bb5\u0bc0\u0b9f\u0bc1 \u0bb5\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1 \u0c35\u0c38\u0c24\u0c3f \u0c07\u0c32\u0c4d\u0c32\u0c41 \u0c17\u0c43\u0c39\u0c02 \u09ac\u09be\u09b8\u09b8\u09cd\u09a5\u09be\u09a8 \u0998\u09b0"
    ),
    "senior_citizen": (
        "senior citizen old age elderly pension vridh vruddh "
        "\u0935\u0930\u093f\u0937\u094d\u0920 \u0928\u093e\u0917\u0930\u093f\u0915 \u0935\u0943\u0926\u094d\u0927 \u092a\u0947\u0902\u0936\u0928 \u0ab5\u0ac3\u0aa6\u0acd\u0aa7 \u0ab5\u0ab0\u0abf\u0ab7\u0acd\u0aa0 \u0aa8\u0abe\u0a97\u0ab0\u0abf\u0a95 \u0aaa\u0ac7\u0aa8\u0acd\u0ab6\u0aa8 \u0cb9\u0cbf\u0cb0\u0cbf\u0caf \u0ca8\u0cbe\u0c97\u0cb0\u0cbf\u0c95 \u0cb5\u0cc3\u0ca6\u0ccd\u0ca7 \u0caa\u0cbf\u0c82\u0c9a\u0ca3\u0cbf "
        "\u0bae\u0bc2\u0ba4\u0bcd\u0ba4 \u0b95\u0bc1\u0b9f\u0bbf\u0bae\u0b95\u0ba9\u0bcd \u0bae\u0bc1\u0ba4\u0bbf\u0baf\u0bcb\u0bb0\u0bcd \u0b93\u0baf\u0bcd\u0bb5\u0bc2\u0ba4\u0bbf\u0baf\u0bae\u0bcd \u0c35\u0c43\u0c26\u0c4d\u0c27\u0c3e\u0c2a\u0c4d\u0c2f \u0c2a\u0c3f\u0c02\u0c1b\u0c28\u0c41 \u09aa\u09cd\u09b0\u09ac\u09c0\u09a3 \u09aa\u09c7\u09a8\u09b6\u09a8"
    ),
    "disability": (
        "disabled disability divyang handicapped disability pension assistive devices "
        "\u0926\u093f\u0935\u094d\u092f\u093e\u0902\u0917 \u0935\u093f\u0915\u0932\u093e\u0902\u0917 \u0aa6\u0abf\u0ab5\u0acd\u0aaf\u0abe\u0a82\u0a97 \u0ab5\u0abf\u0a95\u0ab2\u0abe\u0a82\u0a97 \u0ca6\u0cbf\u0cb5\u0ccd\u0caf\u0cbe\u0c82\u0c97 \u0c85\u0c82\u0c97\u0cb5\u0cbf\u0c95\u0cb2 \u0cae\u0cbe\u0c9f\u0ccd\u0cb0\u0cc1\u0ca4\u0cbf\u0cb0\u0ca8\u0cbe\u0cb3\u0cbf \u0c26\u0c3f\u0c35\u0c4d\u0c2f\u0c3e\u0c02\u0c17\u0c41\u0c32\u0c41 \u0c35\u0c3f\u0c15\u0c32\u0c3e\u0c02\u0c17\u0c41\u0c32\u0c41 \u09aa\u09cd\u09b0\u09a4\u09bf\u09ac\u09a8\u09cd\u09a7\u09c0"
    ),
    "employment": (
        "job employment skill rojgar training unemployed livelihood "
        "\u0930\u094b\u091c\u0917\u093e\u0930 \u0928\u094c\u0915\u0930\u0940 \u0915\u094c\u0936\u0932 \u0ab0\u0acb\u0a9c\u0a97\u0abe\u0ab0 \u0aa8\u0acb\u0a95\u0ab0\u0ac0 \u0c89\u0ca6\u0ccd\u0caf\u0ccb\u0c97 \u0c95\u0cc6\u0cb2\u0cb8 \u0c95\u0ccc\u0cb6\u0cb2\u0ccd\u0caf \u0bb5\u0bc7\u0bb2\u0bc8 \u0ba4\u0bbf\u0bb1\u0ba9\u0bcd \u0c09\u0c26\u0c4d\u0c2f\u0c4b\u0c17\u0c02 \u0c28\u0c48\u0c2a\u0c41\u0c23\u0c4d\u0c2f\u0c02 \u099a\u09be\u0995\u09b0\u09bf \u0995\u09b0\u09cd\u09ae\u09b8\u0982\u09b8\u09cd\u09a5\u09be\u09a8"
    ),
    "social_welfare": (
        "social welfare poverty family assistance funeral marriage support "
        "general welfare social assistance"
    ),
    "unknown": "unclear intent",
}

LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "kn": "Kannada",
    "mr": "Marathi",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}


def _parse_indian_number(text: str) -> float | None:
    source = str(text or "").strip().lower().replace(",", "")
    if not source:
        return None

    normalized = source
    unit_aliases = {
        "lakhs": "lakh",
        "lakh": "lakh",
        "lac": "lakh",
        "lacs": "lakh",
        "\u0915\u0930\u094b\u0921\u093c": "crore",
        "\u0915\u0930\u094b\u0921": "crore",
        "\u0c15\u0c4b\u0c1f\u0c3f": "crore",
        "\u0b95\u0bcb\u0b9f\u0bbf": "crore",
        "\u09b9\u09be\u099c\u09be\u09b0": "thousand",
        "\u0939\u091c\u093e\u0930": "thousand",
        "\u09b9\u09be\u099c\u09be\u09b0": "thousand",
        "\u0b86\u0baf\u0bbf\u0bb0\u0bae\u0bcd": "thousand",
        "\u06c1\u0632\u0627\u0631": "thousand",
    }
    for alias, unit in unit_aliases.items():
        normalized = normalized.replace(alias, f" {unit} ")
    normalized = " ".join(normalized.split())

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr|k|thousand)?", normalized)
    if not match:
        return None

    value = float(match.group(1))
    unit = (match.group(2) or "").strip()
    if unit in {"k", "thousand"}:
        value *= 1000
    elif unit in {"lakh", "lac"}:
        value *= 100000
    elif unit in {"crore", "cr"}:
        value *= 10000000
    return value


def _to_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value

    text = str(value).strip()
    if not text:
        return None

    parsed_indian = _parse_indian_number(text)
    if parsed_indian is not None:
        return int(parsed_indian) if parsed_indian.is_integer() else parsed_indian

    try:
        numeric = float(text.replace(",", ""))
        return int(numeric) if numeric.is_integer() else numeric
    except Exception:
        return None


def _to_bool(value: Any) -> bool | None:
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


def _normalize_language_code(value: Any) -> str | None:
    code = str(value or "").strip().lower()
    return code if code in ALLOWED_LANGUAGE_CODES else None


def normalize_text_light(text: str) -> str:
    source = str(text or "")
    # Repair common UTF-8 -> latin1 mojibake seen in mixed Windows terminals/logs.
    # NFKC keeps Indic scripts intact while still normalizing compatibility forms.
    normalized = unicodedata.normalize("NFKC", source)

    # Strip Latin accent/diacritical marks (< U+0900) while preserving Indic
    # combining marks (matras, nuktas, etc.).
    stripped_chars: list[str] = []
    for ch in normalized:
        if unicodedata.category(ch) == "Mn" and ord(ch) < 0x0900:
            continue  # Drop Latin accents: VÄ á¹‡ijyÅ dyamÄ« → Vanijyodyami
        stripped_chars.append(ch)
    stripped = "".join(stripped_chars)

    lowered = stripped.lower()
    cleaned_chars: list[str] = []
    for ch in lowered:
        category = unicodedata.category(ch)
        if ch == "_" or ch.isalnum() or ch.isspace() or category.startswith("M"):
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_intent_category(value: Any) -> str:
    category = normalize_text_light(str(value or "")).replace(" ", "_")
    return category if category in CONTROLLED_TAXONOMY else "unknown"


def _normalize_intent_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    for item in value:
        key = normalize_text_light(str(item or ""))
        if not key:
            continue
        if key in keywords:
            continue
        keywords.append(key)
        if len(keywords) >= 5:
            break
    return keywords


def _context_signature(conversation_context: Any) -> str:
    try:
        if isinstance(conversation_context, (dict, list)):
            return json.dumps(conversation_context, sort_keys=True, ensure_ascii=False)
        return str(conversation_context or "")
    except Exception:
        return str(conversation_context or "")


def _build_intent_payload(
    *,
    category: str = "unknown",
    subcategory: str | None = None,
    intent_keywords: list[str] | None = None,
    confidence: float = 0.0,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "canonical_category": _normalize_intent_category(category),
        "subcategory": normalize_text_light(subcategory or "") or None,
        "intent_keywords": _normalize_intent_keywords(intent_keywords or []),
        "confidence": float(confidence or 0.0),
        "reason": str(reason or "").strip()[:240],
    }


def _tokenize(text: str) -> list[str]:
    return [tok for tok in normalize_text_light(text).split() if tok]


INTENT_STOPWORDS = {
    "scheme",
    "schemes",
    "support",
    "assistance",
    "benefit",
    "benefits",
    "welfare",
    "government",
    "india",
    "national",
    "general",
    "help",
    "plan",
    "program",
}


def _fuzzy_token_similarity(token: str, candidates: list[str]) -> float:
    if not token or not candidates:
        return 0.0
    ratios = [SequenceMatcher(None, token, cand).ratio() for cand in candidates if cand]
    return max(ratios) if ratios else 0.0


def _lexical_category_score(query: str, description: str) -> float:
    query_tokens = [t for t in _tokenize(query) if len(t) >= 2 and t not in INTENT_STOPWORDS]
    desc_tokens = [t for t in _tokenize(description) if len(t) >= 2 and t not in INTENT_STOPWORDS]
    if not query_tokens or not desc_tokens:
        return 0.0

    overlap = len(set(query_tokens) & set(desc_tokens)) / max(1, len(set(query_tokens)))
    fuzzy = 0.0
    for tok in query_tokens:
        if len(tok) < 4:
            continue
        fuzzy = max(fuzzy, _fuzzy_token_similarity(tok, desc_tokens))
    if fuzzy < 0.82:
        fuzzy = 0.0
    return max(overlap, fuzzy * 0.9)


def _extract_subcategory(query: str) -> str | None:
    query_norm = normalize_text_light(query)
    if not query_norm:
        return None
    subcategory_hints = [
        "pension",
        "scholarship",
        "loan",
        "subsidy",
        "entrepreneurship",
        "startup",
        "hospital",
        "insurance",
        "training",
        "widow",
        "old_age",
    ]
    compact = query_norm.replace(" ", "_")
    for hint in subcategory_hints:
        if hint in compact or hint.replace("_", " ") in query_norm:
            return hint.replace("_", " ")
    return None


def classify_user_intent_semantic(
    original_text: str,
    english_translation: str | None = None,
    selected_language: str | None = None,
    conversation_context: Any = None,
) -> dict[str, Any]:
    _ = selected_language
    _ = conversation_context
    english = english_translation if english_translation is not None else original_text
    query = normalize_text_light(english or original_text)
    if not query:
        return _build_intent_payload(reason="empty_input")

    best_category = "unknown"
    best_score = 0.0
    best_reason = "semantic_low_confidence"
    for category, desc in TAXONOMY_DESCRIPTIONS.items():
        if category == "unknown":
            continue
        try:
            semantic_score = float(semantic_match(query, normalize_text_light(desc)))
        except Exception:
            semantic_score = 0.0
        lexical_score = _lexical_category_score(query, desc)
        score = max(semantic_score, lexical_score)
        if score > best_score:
            best_score = score
            best_category = category
            best_reason = "semantic_similarity" if semantic_score >= lexical_score else "lexical_similarity"

    if best_score < 0.34:
        return _build_intent_payload(confidence=best_score, reason="semantic_low_confidence")

    keywords = [token for token in query.split() if len(token) >= 3 and token not in INTENT_STOPWORDS][:5]
    subcategory = _extract_subcategory(query)
    return _build_intent_payload(
        category=best_category,
        subcategory=subcategory,
        intent_keywords=keywords,
        confidence=best_score,
        reason=best_reason,
    )


def classify_user_intent_llm(
    original_text: str,
    english_translation: str | None = None,
    selected_language: str | None = None,
    conversation_context: Any = None,
) -> dict[str, Any]:
    english = english_translation if english_translation is not None else original_text
    lang = _normalize_language_code(selected_language) or "en"
    context_sig = _context_signature(conversation_context)
    cache_key_text = f"{normalize_text_light(english)}|{context_sig}"
    cached = get_extraction_cache(cache_key_text, lang, "intent_classifier")
    if isinstance(cached, dict):
        return _build_intent_payload(
            category=cached.get("canonical_category"),
            subcategory=cached.get("subcategory"),
            intent_keywords=cached.get("intent_keywords") or [],
            confidence=float(cached.get("confidence") or 0.0),
            reason=str(cached.get("reason") or ""),
        )

    request_payload = {
        "original_text": sanitize_for_llm(original_text or ""),
        "english_translation": sanitize_for_llm(english or ""),
        "selected_language": lang,
        "conversation_context": conversation_context or {},
    }
    parsed = router.generate_json(json.dumps(request_payload, ensure_ascii=False), INTENT_CLASSIFIER_PROMPT)
    if not isinstance(parsed, dict):
        semantic = classify_user_intent_semantic(original_text, english, lang, conversation_context)
        set_extraction_cache(cache_key_text, semantic, lang, "intent_classifier")
        return semantic

    intent = _build_intent_payload(
        category=parsed.get("canonical_category"),
        subcategory=parsed.get("subcategory"),
        intent_keywords=parsed.get("intent_keywords") or [],
        confidence=float(_to_number(parsed.get("confidence")) or 0.0),
        reason=str(parsed.get("reason") or ""),
    )

    set_extraction_cache(cache_key_text, intent, lang, "intent_classifier")
    return intent


def infer_language_selection_llm(user_text: str) -> dict[str, Any]:
    def _fallback_language_from_name(raw_text: str) -> tuple[str | None, float]:
        query = normalize_text_light(raw_text)
        if not query:
            return None, 0.0

        name_to_code = {normalize_text_light(name): code for code, name in LANGUAGE_CODE_TO_NAME.items()}
        for code in ALLOWED_LANGUAGE_CODES:
            name_to_code[code] = code

        if query in name_to_code:
            return name_to_code[query], 0.9

        names = list(name_to_code.keys())
        fuzzy = get_close_matches(query, names, n=1, cutoff=0.72)
        if fuzzy:
            best = fuzzy[0]
            ratio = SequenceMatcher(None, query, best).ratio()
            return name_to_code.get(best), float(ratio)
        return None, 0.0

    payload: dict[str, Any] = {"selected_language": None, "confidence": 0.0}
    request_payload = {"user_text": sanitize_for_llm(user_text or "")}
    parsed = router.generate_json(json.dumps(request_payload, ensure_ascii=False), LANGUAGE_SELECTION_PROMPT)
    if not isinstance(parsed, dict):
        fallback_code, fallback_conf = _fallback_language_from_name(user_text or "")
        payload["selected_language"] = fallback_code
        payload["confidence"] = fallback_conf
        return payload

    language = _normalize_language_code(parsed.get("selected_language"))
    confidence = _to_number(parsed.get("confidence"))
    if language is None:
        fallback_code, fallback_conf = _fallback_language_from_name(user_text or "")
        language = fallback_code
        if confidence is None or float(confidence) < fallback_conf:
            confidence = fallback_conf
    payload["selected_language"] = language
    payload["confidence"] = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    return payload


def extract_profile_llm(
    original_text: str,
    english_text: str | None = None,
    language: str | None = None,
    current_profile: dict[str, Any] | None = None,
    expected_field: str | None = None,
) -> dict:
    """
    Strict JSON-only extraction via Gemini (V3 split prompt architecture).
    This function uses the EXTRACTION PROMPT \u2014 no conversation, no formatting.
    """
    payload: dict[str, Any] = {
        "language": None,
        "intent": "unknown",
        "scheme_name": None,
        "entities": {"age": None, "income": None, "gender": None, "occupation": None, "state": None},
        "category": None,
        "age": None,
        "income": None,
        "occupation": None,
        "education_level": None,
        "state": None,
        "gender": None,
        "caste_category": None,
        "academic_percentage": None,
        "bpl_status": None,
        "answer_english": None,
        "confidence": 0.0,
    }

    english_text = english_text if english_text is not None else original_text
    request_payload = {
        "original_user_message": sanitize_for_llm(original_text or ""),
        "english_message": sanitize_for_llm(english_text or ""),
        "selected_language": language,
        "current_profile": current_profile or {},
        "expected_field": expected_field,
    }

    parsed = router.generate_json(json.dumps(request_payload, ensure_ascii=False), PROFILE_EXTRACTION_PROMPT)
    if not isinstance(parsed, dict):
        return payload

    entities = parsed.get("entities") if isinstance(parsed.get("entities"), dict) else {}
    payload["language"] = _normalize_language_code(parsed.get("language"))
    scheme_name = str(parsed.get("scheme_name") or "").strip()
    if scheme_name:
        payload["scheme_name"] = scheme_name

    intent = str(parsed.get("intent") or "").strip().lower()
    public_to_internal = {
        "search_schemes": "scheme_search",
        "apply_scheme": "apply_scheme",
        "check_eligibility": "check_eligibility",
        "help": "help",
        "change_language": "change_language",
    }
    intent = public_to_internal.get(intent, intent)
    if intent == "search_schemes":
        intent = "scheme_search"
    if intent in ALLOWED_PROFILE_INTENTS:
        payload["intent"] = intent

    for key in ["category", "occupation", "education_level", "state", "gender", "caste_category", "answer_english"]:
        value = parsed.get(key)
        if value is None and key in {"occupation", "state", "gender"}:
            value = entities.get(key)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            payload[key] = text_value

    for key in ["age", "income", "academic_percentage", "confidence"]:
        source_value = parsed.get(key)
        if source_value is None and key in {"age", "income"}:
            source_value = entities.get(key)
        number = _to_number(source_value)
        if number is not None:
            payload[key] = number

    payload["entities"] = {
        "age": payload.get("age"),
        "income": payload.get("income"),
        "gender": payload.get("gender"),
        "occupation": payload.get("occupation"),
        "state": payload.get("state"),
    }
    payload["bpl_status"] = _to_bool(parsed.get("bpl_status"))
    return payload


def decide_next_question_llm(
    profile: dict[str, Any],
    user_goal: str,
    previous_question: str | None = None,
    last_user_message: str | None = None,
) -> dict[str, str | None]:
    """
    Ask LLM to decide best missing next field.
    Returns strict payload: {"next_field": ..., "question_english": ..., "reason": ...}
    """
    payload: dict[str, str | None] = {
        "next_field": None,
        "question_english": None,
        "reason": "No missing field identified",
    }
    request_payload = {
        "profile": profile or {},
        "user_goal": user_goal or "unknown",
        "previous_question": previous_question,
        "last_user_message": last_user_message or "",
    }
    parsed = router.generate_json(json.dumps(request_payload, ensure_ascii=False), NEXT_QUESTION_PROMPT)
    if not isinstance(parsed, dict):
        return payload

    next_field = str(parsed.get("next_field") or "null").strip().lower()
    if next_field == "annual_income":
        next_field = "income"
    if next_field not in ALLOWED_NEXT_FIELDS:
        next_field = "null"

    reason = str(parsed.get("reason") or "").strip() or payload["reason"]
    question_english = str(parsed.get("question_english") or "").strip() or None

    profile_data = profile or {}
    if next_field == "income":
        if profile_data.get("income") is not None or profile_data.get("annual_income") is not None:
            next_field = "null"
    elif next_field != "null" and profile_data.get(next_field) not in (None, ""):
        next_field = "null"

    payload["next_field"] = None if next_field == "null" else next_field
    payload["question_english"] = question_english
    payload["reason"] = reason
    return payload


RESPONSE_FORMATTER_PROMPT = """SYSTEM ROLE:
You are YojnaSetuBot response formatter.

CRITICAL RULE:
You MUST ONLY format schemes passed from backend.

You MUST NEVER:
- change category
- mix categories
- include unrelated schemes

\u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513
STRICT VALIDATION BEFORE RESPONSE
\u2517\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u251b

For each scheme:

1. Check category matches user category
2. Check state matches user state OR "All India"
3. If not \u2192 DROP scheme

\u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513
OUTPUT FORMAT
\u2517\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u251b

\u1f539 Scheme Name: <name>  
\u1f4cd Benefit: <1 line>  
\u2705 Why this matches: Based on category=<category>, state=<state>  
\u1f4c4 Documents Required: <docs>  

Max 5 schemes only.

\u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513

IF ALL SCHEMES DROPPED:

Say:

"No schemes found for your category and state. Showing national schemes."

Then retry with ONLY "All India" schemes.

\u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513

NEVER OUTPUT RAW TEXT  
NEVER OUTPUT UNFILTERED DATA  
"""



def format_schemes_llm(cards: list[dict], profile: dict, language: str) -> str:
    import json
    
    user_category = profile.get("category", "unknown")
    user_state = profile.get("state", "unknown")
    
    # Hard-filter schemes BEFORE sending to LLM (deterministic, no LLM trust)
    filtered_cards = []
    for card in cards:
        scheme_cat = str(card.get("category") or card.get("scheme_category") or "").strip().lower()
        scheme_state = str(card.get("state") or "").strip().lower()
        user_state_norm = str(user_state).strip().lower()
        user_cat_norm = str(user_category).strip().lower()
        
        # Category gate
        if user_cat_norm and user_cat_norm != "unknown":
            if scheme_cat and scheme_cat != "unknown" and scheme_cat != user_cat_norm:
                continue
        
        # State gate
        if user_state_norm and user_state_norm != "unknown":
            if scheme_state and scheme_state not in {user_state_norm, "all india", "national", "central", "india"}:
                continue
        
        filtered_cards.append(card)
    
    if not filtered_cards:
        filtered_cards = [c for c in cards if str(c.get("state") or "").strip().lower() in {"all india", "national", "central", "india", ""}]
    
    if not filtered_cards:
        return ""
    
    payload = {
        "user_profile": {
            "category": user_category,
            "state": user_state,
            "language": language
        },
        "schemes": filtered_cards[:5]
    }
    
    lang_instruction = f"LANGUAGE LOCK: ALL output MUST be in language code \"{language}\". NO English mixing."
    
    prompt = f"{RESPONSE_FORMATTER_PROMPT}\n\n{lang_instruction}\n\nPayload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    
    try:
        result = router.generate_text(prompt, temperature=0.0)
        if result and len(result.strip()) > 10:
            return result.strip()
    except Exception:
        pass
    return ""

SCALE_WORDS = {
    "en": {
        "lakh": "lakh",
        "lac": "lakh",
        "cr": "crore",
        "crore": "crore",
        "k": "thousand",
        "thousand": "thousand",
    },
    "hi": {
        "\u0932\u093e\u0916": "lakh",
        "\u0915\u0930\u094b\u0921\u093c": "crore",
        "\u0939\u091c\u093e\u0930": "thousand",
    },
}
