"""
Safe multi-layer cache service for YojnaSetuBot.

Cache types:
1) extraction: message understanding JSON
2) rag: retrieved candidate scheme list
3) response: final response text for a profile+scheme signature
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any

from core.logger import get_logger
from services.confidence_service import CONFIDENCE_THRESHOLD

logger = get_logger("services.cache_service")

COLLECTION_NAME = "llm_cache"

TTL_SECONDS = {
    "extraction": 7 * 24 * 3600,
    "rag": 7 * 24 * 3600,
    "response": 24 * 3600,
}

PROFILE_SIG_FIELDS = [
    "category",
    "state",
    "income",
    "annual_income",
    "age",
    "gender",
    "caste_category",
    "academic_percentage",
    "bpl_status",
]


def _utcnow() -> datetime:
    return datetime.utcnow()


def _expires_for(cache_type: str) -> datetime:
    return _utcnow() + timedelta(seconds=TTL_SECONDS.get(cache_type, 24 * 3600))


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _get_collection():
    try:
        from models.db_client import db_client

        if db_client.db is None:
            return None
        col = db_client.db[COLLECTION_NAME]
        col.create_index("expires_at", expireAfterSeconds=0, background=True)
        return col
    except Exception as exc:
        logger.debug(f"Cache collection unavailable: {exc}")
        return None


def normalize_cache_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def profile_signature(profile: dict | None) -> str:
    source = profile or {}
    payload: dict[str, Any] = {}
    for field in PROFILE_SIG_FIELDS:
        value = source.get(field)
        if field == "income" and value is None:
            value = source.get("annual_income")
        if field == "annual_income" and value is None:
            value = source.get("income")
        payload[field] = value
    return _hash(_to_json(payload))


def extraction_key(
    text: str,
    language: str,
    expected_field: str | None,
    category: str | None = None,
    state: str | None = None,
    income: int | float | None = None,
    age: int | None = None,
) -> str:
    payload = {
        "type": "extraction",
        "text": normalize_cache_text(text),
        "language": (language or "en").strip().lower(),
        "expected_field": (expected_field or "").strip().lower() or None,
        "category": normalize_cache_text(category or ""),
        "state": normalize_cache_text(state or ""),
        "income": str(income) if income is not None else None,
        "age": str(age) if age is not None else None,
    }
    return _hash(_to_json(payload))


def rag_key(category: str, state: str | None, language: str | None, subcategory: str | None = None, profile: dict | None = None) -> str:
    payload = {
        "type": "rag",
        "category": normalize_cache_text(category),
        "subcategory": normalize_cache_text(subcategory or ""),
        "state": normalize_cache_text(state or ""),
        "language": (language or "en").strip().lower(),
        "profile_sig": profile_signature(profile) if profile else None,
    }
    return _hash(_to_json(payload))


def response_key(profile: dict, scheme_ids: list[str], language: str | None) -> str:
    payload = {
        "type": "response",
        "profile_signature": profile_signature(profile),
        "category": normalize_cache_text((profile or {}).get("category") or ""),
        "state": normalize_cache_text((profile or {}).get("state") or ""),
        "scheme_ids": list(scheme_ids or []),
        "language": (language or "en").strip().lower(),
    }
    return _hash(_to_json(payload))


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _get_doc(key: str, expected_type: str):
    col = _get_collection()
    if col is None:
        return None
    try:
        doc = col.find_one({"_id": key, "type": expected_type})
        if not doc:
            return None
        if doc.get("expires_at") and doc["expires_at"] <= _utcnow():
            return None
        return doc
    except Exception as exc:
        logger.debug(f"Cache read error: {exc}")
        return None


def _set_doc(
    *,
    key: str,
    cache_type: str,
    input_str: str,
    response: Any,
    language: str = "en",
    expected_field: str | None = None,
    profile_sig: str | None = None,
    category: str | None = None,
    state: str | None = None,
    confidence: float | None = None,
    model_name: str = "unknown",
    latency_ms: int = 0,
) -> None:
    if response in (None, "", [], {}):
        return
    normalized_confidence = _normalize_confidence(confidence)
    if normalized_confidence is not None and normalized_confidence < CONFIDENCE_THRESHOLD:
        logger.info({"event": "cache_skip_low_confidence", "type": cache_type, "confidence": normalized_confidence})
        return
    col = _get_collection()
    if col is None:
        return
    try:
        now = _utcnow()
        doc = {
            "_id": key,
            "type": cache_type,
            "input_str": str(input_str or "")[:2000],
            "response": response,
            "profile_signature": profile_sig,
            "profile_hash": profile_sig,
            "language": (language or "en").strip().lower(),
            "expected_field": (expected_field or "").strip().lower() or None,
            "category": normalize_cache_text(category or ""),
            "state": normalize_cache_text(state or ""),
            "confidence": normalized_confidence,
            "model_name": model_name,
            "latency_ms": int(latency_ms or 0),
            "created_at": now,
            "expires_at": _expires_for(cache_type),
        }
        col.replace_one({"_id": key}, doc, upsert=True)
    except Exception as exc:
        logger.debug(f"Cache write error: {exc}")


def get_extraction_cache(
    text: str,
    language: str = "en",
    expected_field: str | None = None,
    category: str | None = None,
    state: str | None = None,
    income: int | float | None = None,
    age: int | None = None,
) -> dict | None:
    key = extraction_key(text, language, expected_field, category=category, state=state, income=income, age=age)
    doc = _get_doc(key, "extraction")
    if doc and isinstance(doc.get("response"), dict):
        logger.info({"event": "extraction_cache_hit", "language": language, "expected_field": expected_field})
        return doc["response"]
    logger.info({"event": "extraction_cache_miss", "language": language, "expected_field": expected_field})
    return None


def set_extraction_cache(
    text: str,
    result: dict,
    language: str = "en",
    expected_field: str | None = None,
    model_name: str = "unknown",
    latency_ms: int = 0,
    category: str | None = None,
    state: str | None = None,
    income: int | float | None = None,
    age: int | None = None,
) -> None:
    if not isinstance(result, dict) or not result:
        return
    key = extraction_key(text, language, expected_field, category=category, state=state, income=income, age=age)
    _set_doc(
        key=key,
        cache_type="extraction",
        input_str=text,
        response=result,
        language=language,
        expected_field=expected_field,
        category=category,
        state=state,
        confidence=result.get("confidence"),
        model_name=model_name,
        latency_ms=latency_ms,
    )


def get_rag_cache(
    category: str,
    state: str | None = None,
    language: str | None = "en",
    subcategory: str | None = None,
    profile: dict | None = None,
) -> list | None:
    key = rag_key(category, state, language, subcategory=subcategory, profile=profile)
    doc = _get_doc(key, "rag")
    if doc and isinstance(doc.get("response"), list):
        logger.info({"event": "rag_cache_hit", "category": category, "subcategory": subcategory, "state": state, "language": language})
        return doc["response"]
    logger.info({"event": "rag_cache_miss", "category": category, "subcategory": subcategory, "state": state, "language": language})
    return None


def set_rag_cache(
    category: str,
    schemes: list,
    state: str | None = None,
    language: str | None = "en",
    subcategory: str | None = None,
    profile: dict | None = None,
    model_name: str = "deterministic",
    latency_ms: int = 0,
) -> None:
    if not isinstance(schemes, list) or not schemes:
        return
    key = rag_key(category, state, language, subcategory=subcategory, profile=profile)
    _set_doc(
        key=key,
        cache_type="rag",
        input_str=f"{category}|{subcategory}|{state}",
        response=schemes,
        language=language or "en",
        category=category,
        state=state,
        confidence=1.0,
        model_name=model_name,
        latency_ms=latency_ms,
    )


def get_response_cache(profile: dict, scheme_ids: list[str], language: str | None = "en") -> str | None:
    key = response_key(profile, scheme_ids, language)
    doc = _get_doc(key, "response")
    if doc and isinstance(doc.get("response"), str):
        logger.info({"event": "response_cache_hit", "language": language, "scheme_count": len(scheme_ids or [])})
        return doc["response"]
    logger.info({"event": "response_cache_miss", "language": language, "scheme_count": len(scheme_ids or [])})
    return None


def set_response_cache(
    profile: dict,
    scheme_ids: list[str],
    response_text: str,
    language: str | None = "en",
    confidence: float | None = None,
    model_name: str = "deterministic",
    latency_ms: int = 0,
) -> None:
    if not response_text or not isinstance(response_text, str):
        return
    key = response_key(profile, scheme_ids, language)
    _set_doc(
        key=key,
        cache_type="response",
        input_str="|".join(scheme_ids or []),
        response=response_text,
        language=language or "en",
        profile_sig=profile_signature(profile),
        category=str((profile or {}).get("category") or ""),
        state=str((profile or {}).get("state") or ""),
        confidence=confidence,
        model_name=model_name,
        latency_ms=latency_ms,
    )
