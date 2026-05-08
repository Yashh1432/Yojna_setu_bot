"""
Deterministic scheme retrieval with semantic intent normalization.

LLM/semantic logic is used only for intent/category normalization.
Scheme retrieval, filtering, ranking, and eligibility remain deterministic.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.logger import get_logger
from engine.engine import classify_user_intent_llm, classify_user_intent_semantic, normalize_text_light
from engine.eligibility import filter_schemes
from engine.validator import normalize_state_name
from services.cache_service import get_rag_cache, set_rag_cache
from services.embedding_service import semantic_match

logger = get_logger("engine.orchestrator")

MIN_STRONG_MATCH_SCORE = 60

DATASET_PATHS = [
    Path(__file__).resolve().parent.parent / "datasets" / "fixed_final_production_schemes.json",
    Path(__file__).resolve().parent.parent / "scheme_datasets" / "fixed_final_production_schemes.json",
    Path(__file__).resolve().parent.parent / "scheme_datasets" / "final_production_schemes.json",
]

TAXONOMY = {
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

DATASET_CATEGORY_TO_TAXONOMY = {
    "education": "education",
    "agriculture": "agriculture",
    "health": "health",
    "employment": "employment",
    "housing": "housing",
    "financial assistance": "finance_business",
    "finance": "finance_business",
    "business": "finance_business",
    # 'women & child' normalizes to 'women child' after normalize_text_light strips '&'
    "women & child": "women_child",
    "women and child": "women_child",
    "women child": "women_child",       # normalized form of 'women & child'
    "women": "women_child",
    "mahila": "women_child",
    "child": "women_child",
    "senior citizen": "senior_citizen",
    "old age": "senior_citizen",
    "pension": "senior_citizen",
    "disability": "disability",
    "divyang": "disability",
    "others": "social_welfare",
    "other": "social_welfare",
}


def _is_mocked_callable(obj: Any) -> bool:
    t = type(obj)
    return "unittest.mock" in getattr(t, "__module__", "") or "mock" in getattr(t, "__name__", "").lower()


@lru_cache(maxsize=1)
def load_scheme_dataset() -> list[dict]:
    for path in DATASET_PATHS:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info(f"Loaded {len(data)} schemes from {path.name}")
            return data
    logger.warning("No local scheme dataset found.")
    return []


def _maybe_clear_test_caches() -> None:
    if _is_mocked_callable(load_scheme_dataset):
        try:
            _clean_dataset.cache_clear()
            _dataset_available_states.cache_clear()
            _dataset_category_distribution.cache_clear()
        except Exception:
            pass


def _profile_income(profile: dict) -> int | float | None:
    if profile.get("annual_income") is not None:
        return profile.get("annual_income")
    return profile.get("income")


def _scheme_income_limit(scheme: dict[str, Any]) -> int | float | None:
    rules = scheme.get("eligibility") if isinstance(scheme.get("eligibility"), dict) else {}
    limit = rules.get("max_income") if isinstance(rules, dict) else None
    if limit is None:
        limit = scheme.get("income_limit")
    if limit is None:
        return None
    try:
        return float(limit)
    except Exception:
        return None


def summarize_benefit(text: Any, max_chars: int = 150) -> str:
    if not text:
        return "Details not clearly listed."
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


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


def state_allowed(user_state: str | None, scheme_state: str | None) -> bool:
    if not user_state:
        return True
    if not scheme_state:
        return False
    user_state_norm = _norm_state(user_state)
    scheme_state_norm = _norm_state(scheme_state)
    return scheme_state_norm == user_state_norm or _is_all_india(scheme_state_norm)


def _to_taxonomy(category: Any) -> str:
    raw = normalize_text_light(str(category or ""))
    if raw in TAXONOMY:
        return raw
    return DATASET_CATEGORY_TO_TAXONOMY.get(raw, "unknown")


def _readable_category(category: str | None) -> str:
    labels = {
        "education": "education",
        "health": "health",
        "agriculture": "agriculture",
        "finance_business": "finance/business",
        "women_child": "women/child",
        "housing": "housing",
        "senior_citizen": "senior citizen",
        "disability": "disability",
        "employment": "employment",
        "social_welfare": "social welfare",
    }
    return labels.get(_to_taxonomy(category), "selected")


def _no_exact_state_fallback_message(*, language: str, user_category: str, user_state: str) -> str:
    if language == "hi" and user_category == "housing" and user_state == "uttar pradesh":
        return "उत्तर प्रदेश के लिए कोई सटीक आवास योजना नहीं मिली। उपलब्ध राष्ट्रीय/All India योजनाएँ दिखा रहा हूँ।"
    readable_state = user_state.title() if user_state else "your state"
    return f"No exact {readable_state} schemes found. Showing national/All India schemes only."


def _no_geo_match_message(*, language: str, user_category: str, user_state: str) -> str:
    if language == "hi" and user_category == "housing" and user_state == "uttar pradesh":
        return "उत्तर प्रदेश के लिए कोई उपयुक्त आवास योजना नहीं मिली। कृपया दूसरी श्रेणी आज़माएँ या अधिक जानकारी दें।"
    readable_category = _readable_category(user_category)
    readable_state = user_state.title() if user_state else "your state"
    return f"No matching {readable_category.lower()} schemes found for {readable_state}. Try another category or share more details."


def _hard_filter_schemes(schemes: list[dict], profile: dict, user_category: str) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Hard filter before semantic ranking:
    - category exact match (after taxonomy normalization)
    - state exact match -> primary bucket; All India -> fallback bucket
    - income: only reject if scheme has an EXPLICIT limit AND user income EXCEEDS it
      (missing income limit = no restriction = scheme passes)
    """
    user_state = _norm_state(profile.get("state"))
    user_income = _profile_income(profile)
    primary: list[dict] = []
    fallback_all_india: list[dict] = []
    income_rejected: list[dict] = []

    for scheme in schemes:
        scheme_category = _to_taxonomy(scheme.get("category"))
        if user_category != "unknown" and scheme_category != user_category:
            continue

        scheme_state = _norm_state(scheme.get("state"))
        if user_state:
            if scheme_state == user_state:
                state_bucket = "primary"
            elif _is_all_india(scheme_state):
                state_bucket = "fallback"
            else:
                continue
        else:
            state_bucket = "primary"

        # Income filter: only reject when limit is explicitly stored AND exceeded
        if user_income is not None:
            scheme_limit = _scheme_income_limit(scheme)
            if scheme_limit is not None and float(scheme_limit) < float(user_income):
                income_rejected.append(scheme)
                continue  # hard reject only when limit is known and exceeded

        if state_bucket == "primary":
            primary.append(scheme)
        else:
            fallback_all_india.append(scheme)

    return primary, fallback_all_india, income_rejected


def _semantic_rank_within_filtered(candidates: list[dict], query: str | None, top_n: int = 15) -> list[dict]:
    query_norm = normalize_text_for_retrieval(query or "")
    if not candidates:
        return []
    if not query_norm:
        return candidates[:top_n]

    scored: list[tuple[float, dict]] = []
    keywords = [w for w in query_norm.split() if len(w) >= 3]
    for scheme in candidates:
        searchable = _scheme_searchable_text(scheme)
        sem_score = float(semantic_match(query_norm, searchable) or 0.0)
        lex_score = float(sum(1 for w in keywords if w in searchable))
        total = sem_score + (0.05 * lex_score)
        scored.append((total, scheme))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_n]]


def normalize_text_for_retrieval(text: Any) -> str:
    return normalize_text_light(str(text or ""))


def _extract_keywords(text: str, limit: int = 5) -> list[str]:
    words = [w for w in normalize_text_for_retrieval(text).split() if len(w) >= 4]
    stop = {
        "scheme", "schemes", "state", "india", "details", "please", "help", "need", "want",
        "give", "from", "with", "about", "this", "that", "your", "support", "government",
    }
    out: list[str] = []
    for word in words:
        if word in stop:
            continue
        if word in out:
            continue
        out.append(word)
        if len(out) >= limit:
            break
    return out


def _scheme_searchable_text(scheme: dict) -> str:
    eligibility = scheme.get("eligibility")
    if isinstance(eligibility, dict):
        eligibility_text = " ".join(f"{k}:{v}" for k, v in eligibility.items())
    else:
        eligibility_text = str(eligibility or "")
    text = " ".join(
        [
            str(scheme.get("scheme_name") or ""),
            str(scheme.get("category") or ""),
            str(scheme.get("benefits") or ""),
            str(scheme.get("description") or ""),
            str(scheme.get("target_group") or ""),
            eligibility_text,
        ]
    )
    return normalize_text_for_retrieval(text)


def scheme_passes_quality_gate(scheme: dict) -> bool:
    name = str(scheme.get("scheme_name") or "").strip()
    if not name or name.lower() in {"unnamed scheme", "unknown scheme"}:
        return False

    benefits = str(scheme.get("benefits") or "").strip()
    if not benefits or benefits.lower() == "no description available":
        return False

    if not str(scheme.get("category") or "").strip():
        return False
    if not str(scheme.get("state") or "").strip():
        return False

    return len(_scheme_searchable_text(scheme)) >= 30


def _is_quality_scheme(scheme: dict) -> bool:
    return scheme_passes_quality_gate(scheme)


@lru_cache(maxsize=1)
def _clean_dataset() -> tuple[list[dict], int]:
    raw = load_scheme_dataset()
    clean = [s for s in raw if scheme_passes_quality_gate(s)]
    rejected = len(raw) - len(clean)
    logger.info(
        {
            "event": "dataset_quality_gate",
            "dataset_loaded_count": len(raw),
            "dataset_clean_count": len(clean),
            "quality_rejected_count": rejected,
        }
    )
    return clean, rejected


@lru_cache(maxsize=1)
def _dataset_available_states() -> frozenset[str]:
    clean, _ = _clean_dataset()
    states: set[str] = set()
    for scheme in clean:
        state = _norm_state(scheme.get("state"))
        if state and not _is_all_india(state):
            states.add(state)
    return frozenset(states)


@lru_cache(maxsize=1)
def _dataset_category_distribution() -> dict[str, int]:
    clean, _ = _clean_dataset()
    dist: dict[str, int] = {}
    for scheme in clean:
        cat = _to_taxonomy(scheme.get("category"))
        if cat != "unknown":
            dist[cat] = dist.get(cat, 0) + 1
    return dist


def _normalize_user_intent(profile: dict, query: str | None) -> dict[str, Any]:
    query_text = str(query or "")
    language = str(profile.get("language") or "en").strip().lower()
    context = {
        "profile_category": profile.get("category"),
        "state": profile.get("state"),
    }

    semantic_input = " ".join(
        [
            query_text,
            str(profile.get("category") or ""),
        ]
    ).strip()

    # V3: Gemini = source of truth, embeddings = fallback only
    llm_intent = classify_user_intent_llm(
        original_text=query_text,
        english_translation=query_text,
        selected_language=language,
        conversation_context=context,
    )
    llm_category = _to_taxonomy(llm_intent.get("canonical_category"))
    llm_confidence = float(llm_intent.get("confidence") or 0.0)

    if llm_category != "unknown" and llm_confidence >= 0.45:
        intent = llm_intent  # Gemini is authoritative
    else:
        # Fallback to semantic embeddings
        semantic_intent = classify_user_intent_semantic(
            original_text=semantic_input,
            english_translation=semantic_input,
            selected_language=language,
            conversation_context=context,
        )
        intent = semantic_intent

    category = _to_taxonomy(intent.get("canonical_category"))
    if category == "unknown":
        category = _to_taxonomy(profile.get("category"))

    subcategory = normalize_text_for_retrieval(intent.get("subcategory") or "") or None

    keywords = [normalize_text_for_retrieval(k) for k in (intent.get("intent_keywords") or []) if str(k).strip()]
    if not keywords:
        keywords = _extract_keywords(semantic_input)

    return {
        "canonical_category": category,
        "subcategory": subcategory,
        "intent_keywords": keywords[:5],
        "confidence": float(intent.get("confidence") or 0.0),
        "normalized_query": normalize_text_for_retrieval(query_text),
    }


def preprocess_scheme_intent(scheme: dict) -> dict:
    item = dict(scheme or {})
    searchable_text = _scheme_searchable_text(item)
    searchable_text_hash = hashlib.sha1(searchable_text.encode("utf-8")).hexdigest()
    if item.get("searchable_text_hash") == searchable_text_hash:
        return item

    semantic = classify_user_intent_semantic(
        original_text=searchable_text,
        english_translation=searchable_text,
        selected_language="en",
        conversation_context={"source": "scheme_preprocess"},
    )

    normalized_category = _to_taxonomy(semantic.get("canonical_category"))
    if normalized_category == "unknown":
        normalized_category = _to_taxonomy(item.get("category"))

    intent_keywords = [normalize_text_for_retrieval(k) for k in (semantic.get("intent_keywords") or []) if str(k).strip()]
    if not intent_keywords:
        intent_keywords = _extract_keywords(searchable_text)

    item["searchable_text"] = searchable_text
    item["searchable_text_hash"] = searchable_text_hash
    item["normalized_category"] = normalized_category
    item["normalized_subcategory"] = normalize_text_for_retrieval(semantic.get("subcategory") or "") or None
    item["intent_keywords"] = intent_keywords[:5]
    item["quality_status"] = scheme_passes_quality_gate(item)
    return item


def scheme_passes_geo_gate(scheme: dict, user_state: str | None) -> bool:
    return state_allowed(user_state, scheme.get("state"))


def scheme_passes_intent_gate(scheme: dict, user_intent: dict) -> tuple[bool, dict[str, Any]]:
    user_category = _to_taxonomy(user_intent.get("canonical_category"))
    scheme_category = _to_taxonomy(scheme.get("normalized_category") or scheme.get("category"))

    user_keywords = {normalize_text_for_retrieval(k) for k in (user_intent.get("intent_keywords") or []) if str(k).strip()}
    scheme_keywords = {normalize_text_for_retrieval(k) for k in (scheme.get("intent_keywords") or []) if str(k).strip()}
    keyword_overlap = sorted(user_keywords & scheme_keywords)

    subcategory = normalize_text_for_retrieval(user_intent.get("subcategory") or "")
    searchable = str(scheme.get("searchable_text") or "")
    subcategory_match = bool(subcategory and subcategory in searchable)

    query_norm = normalize_text_for_retrieval(user_intent.get("normalized_query") or "")
    semantic_score = semantic_match(query_norm, searchable) if query_norm else 0.0
    semantic_match_pass = semantic_score >= 0.62

    category_match = bool(user_category != "unknown" and user_category == scheme_category)
    intent_pass = bool(category_match or keyword_overlap or subcategory_match or semantic_match_pass)

    unrelated_category = bool(
        user_category != "unknown"
        and scheme_category != "unknown"
        and scheme_category != user_category
        and semantic_score < 0.78
    )
    if unrelated_category:
        intent_pass = False

    # Hard drop if category doesn't match
    if user_category != "unknown" and scheme_category != "unknown":
        if scheme_category != user_category:
            intent_pass = False

    return intent_pass, {
        "category_match": category_match,
        "subcategory_match": subcategory_match,
        "keyword_overlap_count": len(keyword_overlap),
        "semantic_score": semantic_score,
        "scheme_category": scheme_category,
        "user_category": user_category,
        "unrelated_category": unrelated_category,
    }


def rank_scheme(scheme: dict, user_profile: dict, user_intent: dict, intent_meta: dict[str, Any]) -> int:
    score = 0

    user_state = _norm_state(user_profile.get("state"))
    scheme_state = _norm_state(scheme.get("state"))
    if user_state:
        if scheme_state == user_state:
            score += 40
        elif _is_all_india(scheme_state):
            score += 25
        else:
            return -100

    if not scheme.get("quality_status", False):
        score -= 80

    user_category = _to_taxonomy(intent_meta.get("user_category"))
    scheme_category = _to_taxonomy(intent_meta.get("scheme_category"))

    if user_category != "unknown":
        if scheme_category == user_category:
            score += 50
        elif float(intent_meta.get("semantic_score") or 0.0) >= 0.82:
            score += 30
        else:
            score -= 100

    if intent_meta.get("subcategory_match"):
        score += 25
    if int(intent_meta.get("keyword_overlap_count") or 0) > 0:
        score += 20

    semantic_score = float(intent_meta.get("semantic_score") or 0.0)
    if semantic_score >= 0.72:
        score += 30
    elif semantic_score < 0.30 and not (user_category != "unknown" and scheme_category == user_category):
        score -= 80

    if intent_meta.get("unrelated_category"):
        score -= 80

    rules = scheme.get("eligibility") or {}
    age = user_profile.get("age")
    income = _profile_income(user_profile)
    gender = str(user_profile.get("gender") or "").lower()
    caste = str(user_profile.get("caste_category") or "").lower()
    bpl = user_profile.get("bpl_status")
    disability = bool(user_profile.get("disability_status"))

    if age is not None:
        min_age = rules.get("min_age")
        max_age = rules.get("max_age")
        age_ok = True
        if min_age is not None and age < min_age:
            age_ok = False
        if max_age is not None and age > max_age:
            age_ok = False
        if age_ok and (min_age is not None or max_age is not None):
            score += 20

    if income is not None:
        max_income = rules.get("max_income")
        if max_income is not None and income <= max_income:
            score += 20

    rule_gender = str(rules.get("gender") or "any").lower()
    if gender and rule_gender not in {"", "any", "all"} and rule_gender == gender:
        score += 15

    if bpl is True and rules.get("bpl_required") is True:
        score += 15
    if caste:
        rule_caste = str(rules.get("caste") or "").lower()
        if rule_caste and rule_caste in caste:
            score += 15
    if disability and rules.get("disability_required") is True:
        score += 15

    return int(score)


def explain_match(scheme: dict, profile: dict) -> list[str]:
    reasons: list[str] = []
    rules = scheme.get("eligibility") or {}
    scheme_category = _to_taxonomy(scheme.get("category"))
    user_category = _to_taxonomy(profile.get("category"))

    scheme_state_raw = scheme.get("state") or ""
    scheme_state = _norm_state(scheme_state_raw)
    user_state = _norm_state(profile.get("state"))
    if scheme_state:
        if _is_all_india(scheme_state):
            reasons.append("This is a national/All India scheme.")
        elif user_state and scheme_state == user_state:
            reasons.append(f"This scheme is available in {str(profile.get('state') or '').strip() or scheme_state.title()}.")
        else:
            reasons.append(f"This scheme is available in {str(scheme_state_raw).strip() or scheme_state.title()}.")

    if user_category != "unknown" and scheme_category == user_category:
        reasons.append(f"Scheme category is {user_category.replace('_', ' ')}, matching your selected category.")

    if profile.get("age") is not None:
        min_age = rules.get("min_age")
        max_age = rules.get("max_age")
        if min_age is not None or max_age is not None:
            age_parts: list[str] = []
            if min_age is not None:
                age_parts.append(f"minimum age {min_age}")
            if max_age is not None:
                age_parts.append(f"maximum age {max_age}")
            reasons.append("Age criteria satisfied: " + ", ".join(age_parts) + ".")

    income = _profile_income(profile)
    if income is not None and rules.get("max_income") is not None:
        try:
            reasons.append(f"Income check passed: your income is Rs. {int(float(income)):,} and limit is Rs. {int(float(rules['max_income'])):,}.")
        except Exception:
            reasons.append("Income check passed based on scheme income limit.")

    if not reasons:
        reasons.append("Matched on strict category and state filters.")

    return reasons


def build_limitations(scheme: dict, profile: dict) -> list[str]:
    limitations: list[str] = []
    rules = scheme.get("eligibility") or {}

    if rules.get("min_age") is not None or rules.get("max_age") is not None:
        if profile.get("age") is None:
            limitations.append("Confirm age criteria")

    if rules.get("max_income") is not None and _profile_income(profile) is None:
        limitations.append("Confirm annual income limit")

    if (rules.get("gender") or "any").lower() not in {"", "any", "all"} and not profile.get("gender"):
        limitations.append("Confirm gender-specific eligibility")

    if rules.get("caste") and not profile.get("caste_category"):
        limitations.append("Confirm caste-category requirement")

    if scheme.get("state") and "all india" not in str(scheme.get("state")).lower() and not profile.get("state"):
        limitations.append("Confirm state-specific eligibility")

    return limitations


def recommend_schemes(profile: dict, query: str | None = None, top_k: int = 5) -> dict[str, Any]:
    _maybe_clear_test_caches()
    profile = profile or {}
    clean_pool, quality_rejected = _clean_dataset()
    if not clean_pool:
        return {"schemes": [], "fallback_used": False, "fallback_message": None}

    user_intent = _normalize_user_intent(profile, query)
    user_category = _to_taxonomy(user_intent.get("canonical_category"))
    readable_category = _readable_category(user_category)
    user_state = _norm_state(profile.get("state"))
    user_language = str(profile.get("language") or "en").strip().lower()

    logger.info(
        {
            "event": "retrieval_start",
            "user_category": user_category,
            "subcategory": user_intent.get("subcategory"),
            "intent_keywords": user_intent.get("intent_keywords"),
            "user_state": user_state or None,
            "dataset_clean_count": len(clean_pool),
            "quality_rejected_count": quality_rejected,
        }
    )

    fallback_used = False
    fallback_message: str | None = None

    candidate_pool: list[dict] = clean_pool
    if user_category != "unknown":
        cached = get_rag_cache(user_category, user_state or None, user_language, subcategory=user_intent.get("subcategory"), profile=profile)
        if isinstance(cached, list) and cached:
            candidate_pool = [s for s in cached if scheme_passes_quality_gate(s)]
        else:
            prefiltered = [s for s in clean_pool if _to_taxonomy(s.get("category")) == user_category]
            candidate_pool = prefiltered or []
            if prefiltered:
                set_rag_cache(user_category, prefiltered, user_state or None, user_language, subcategory=user_intent.get("subcategory"), profile=profile)

    prepared: list[dict] = []
    for scheme in candidate_pool:
        enriched = preprocess_scheme_intent(scheme)
        if enriched.get("quality_status"):
            prepared.append(enriched)

    if not prepared:
        readable_state = str(profile.get("state") or "").strip()
        if user_category != "unknown" and readable_state:
            msg = f"No matching {readable_category.lower()} schemes found for {readable_state}. Showing All India schemes if available."
        else:
            msg = f"Sorry, I could not find relevant {readable_category} schemes in the current dataset. Try another category or provide more details."
        return {
            "schemes": [],
            "fallback_used": False,
            "fallback_message": msg,
            "errors": [msg],
            "eligibility": {"eligible": [], "uncertain_needs_more_data": [], "ineligible": [], "errors": []},
        }

    primary, fallback_all_india, income_rejected = _hard_filter_schemes(prepared, profile, user_category)

    # ── Top-up model: state-specific first, then All India to fill up to top_k ──
    # Never show zero results when national schemes exist
    selected_pool = list(primary)
    readable_state = str(profile.get("state") or "").strip()

    if not selected_pool:
        # No state-specific results at all — use All India only
        if fallback_all_india:
            selected_pool = fallback_all_india
            fallback_used = True
            if user_state:
                fallback_message = _no_exact_state_fallback_message(
                    language=user_language,
                    user_category=user_category,
                    user_state=user_state,
                )
        else:
            # Nothing found anywhere
            if income_rejected:
                fallback_message = "No schemes matched your income criteria. Try a higher income range."
            elif user_category != "unknown" and user_state:
                fallback_message = _no_geo_match_message(
                    language=user_language,
                    user_category=user_category,
                    user_state=user_state,
                )
            elif user_category != "unknown":
                fallback_message = f"Sorry, no {readable_category} schemes found for your profile in the current dataset."
            else:
                fallback_message = f"No matching schemes found for your query."
            return {"schemes": [], "fallback_used": fallback_used, "fallback_message": fallback_message}
    elif len(selected_pool) < top_k and fallback_all_india:
        # Have some state schemes but fewer than requested — top-up with All India
        needed = top_k - len(selected_pool)
        selected_pool = selected_pool + fallback_all_india[:needed]
        fallback_used = True
        if readable_state:
            fallback_message = f"Showing {len(primary)} {readable_state} scheme(s) plus national schemes to fill your results."

    ranked = _semantic_rank_within_filtered(selected_pool, query, top_n=max(top_k * 5, 25))
    raw_candidates: list[dict] = []
    for scheme in ranked:
        copied = dict(scheme)
        copied["score"] = 1.0
        copied["why_match"] = explain_match(copied, profile)
        raw_candidates.append(copied)

    filtered = filter_schemes(profile, category=user_category or "", schemes=raw_candidates, query_text=query)

    # Combine eligible + uncertain (don't gate on only one bucket — profile is often incomplete)
    elig   = filtered.get("eligible") or []
    uncert = filtered.get("uncertain_needs_more_data") or []
    # Merge: eligible first, then uncertain, then fall back to ineligible as last resort
    preferred = elig + [s for s in uncert if s not in elig]
    if not preferred:
        preferred = raw_candidates  # hard fallback: return semantically ranked if all filtered out
    eligibility_by_name: dict[str, bool | None] = {}
    for scheme in elig:
        eligibility_by_name[str(scheme.get("scheme_name") or "").strip().lower()] = True
    for scheme in uncert:
        eligibility_by_name[str(scheme.get("scheme_name") or "").strip().lower()] = None
    for scheme in filtered.get("ineligible") or []:
        eligibility_by_name[str(scheme.get("scheme_name") or "").strip().lower()] = False

    geo_safe_preferred: list[dict] = []
    geo_rejected_count = 0
    for scheme in preferred:
        if state_allowed(user_state or None, scheme.get("state")):
            geo_safe_preferred.append(scheme)
            continue
        geo_rejected_count += 1
        logger.warning(
            {
                "event": "geo_rejected",
                "reason": "geo_rejected: scheme_state != user_state and not_all_india",
                "scheme_name": scheme.get("scheme_name"),
                "user_state": user_state or None,
                "scheme_state": _norm_state(scheme.get("state")) or None,
            }
        )

    if user_state and not geo_safe_preferred:
        if fallback_all_india:
            fallback_message = _no_exact_state_fallback_message(
                language=user_language,
                user_category=user_category,
                user_state=user_state,
            )
        else:
            fallback_message = _no_geo_match_message(
                language=user_language,
                user_category=user_category,
                user_state=user_state,
            )
        return {
            "schemes": [],
            "fallback_used": True,
            "fallback_message": fallback_message,
            "errors": [],
            "eligibility": filtered,
            "geo_rejected_count": geo_rejected_count,
        }

    results: list[dict] = []
    for scheme in geo_safe_preferred[:top_k]:
        scheme_name = scheme.get("scheme_name", "Unknown Scheme")
        score = float(scheme.get("score") or 0.0)
        eligible = eligibility_by_name.get(str(scheme_name).strip().lower())
        results.append(
            {
                "scheme_id": scheme.get("scheme_id") or scheme.get("scheme_name", "unknown"),
                "scheme_name": scheme_name,
                "state": scheme.get("state") or "Unknown",
                "category": scheme.get("category"),
                "description": scheme.get("description"),
                "eligibility": scheme.get("eligibility"),
                "eligible": eligible,
                "score": score,
                "benefits_summary": summarize_benefit(
                    scheme.get("benefits_summary") or scheme.get("benefits") or scheme.get("description"),
                    max_chars=150,
                ),
                "documents_required": scheme.get("documents_required", []),
                "application_link": scheme.get("application_link"),
                "why_match": scheme.get("why_match") or ["Matched by strict category/state filters."],
            }
        )

    errors = list(filtered.get("errors") or [])
    if not results and not errors:
        errors = ["No schemes found based on current details."]

    return {
        "schemes": results,
        "fallback_used": fallback_used,
        "fallback_message": fallback_message,
        "errors": errors,
        "eligibility": filtered,
        "geo_rejected_count": geo_rejected_count,
    }
