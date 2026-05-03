"""
Deterministic eligibility filtering owned by Python, not the LLM.
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger
from services.confidence_service import CONFIDENCE_THRESHOLD, final_confidence

logger = get_logger("engine.eligibility")


def detect_fake_schemes(query: str) -> bool:
    """
    Placeholder for RAG-based fake scheme detection.
    Flags obviously unrealistic queries.
    """
    if not query:
        return False

    unrealistic_keywords = ["1 crore", "free money", "instant luxury car", "paisa double", "lottery", "win cash"]
    q_lower = query.lower()
    for word in unrealistic_keywords:
        if word in q_lower:
            logger.warning(f"Fake Detection: Query flagged for Realistic Check: '{query}'")
            return True
    return False


def _profile_view(raw_profile: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_profile.get("profile"), dict):
        return raw_profile["profile"]
    return raw_profile


def _income(profile: dict[str, Any]) -> int | None:
    value = profile.get("annual_income")
    if value is None:
        value = profile.get("income")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _normalize_gender(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    return lowered or None


def _scheme_payload(
    scheme: dict[str, Any],
    reasons: list[str],
    eligible: bool,
    score: float,
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "scheme_name": scheme.get("scheme_name", "Unknown Scheme"),
        "state": scheme.get("state") or "Unknown",
        "benefits_summary": scheme.get("benefits_summary") or scheme.get("benefits") or scheme.get("description") or "Details available.",
        "documents_required": scheme.get("documents_required", scheme.get("documentsRequired", [])),
        "application_link": scheme.get("application_link", ""),
        "why_match": reasons or ["Relevant based on your profile"],
        "eligible": eligible,
        "score": score,
        "missing_fields": missing_fields or [],
    }


def filter_schemes(
    raw_profile: dict,
    category: str,
    schemes: list,
    sub_context: str = None,
    query_text: str | None = None,
) -> dict:
    """
    Deterministic filter engine.
    Hard filters decide eligibility. Missing required user fields move a scheme
    to uncertain_needs_more_data instead of silently rejecting it.
    """
    if not schemes:
        logger.warning("No schemes provided to filter_schemes.")
        return {"eligible": [], "uncertain_needs_more_data": [], "ineligible": [], "errors": ["No schemes provided."]}

    profile = _profile_view(raw_profile or {})
    user_age = profile.get("age")
    user_income = _income(profile)
    user_gender = _normalize_gender(profile.get("gender"))
    user_bpl = profile.get("bpl_status")

    eligible_schemes: list[dict[str, Any]] = []
    uncertain_schemes: list[dict[str, Any]] = []
    ineligible_schemes: list[dict[str, Any]] = []

    logger.info({"event": "eligibility_filter_start", "category": category, "scheme_count": len(schemes)})

    for scheme in schemes:
        rules = scheme.get("eligibility") or {}
        missing_fields: list[str] = []
        rejected_reasons: list[str] = []

        min_age = rules.get("min_age")
        max_age = rules.get("max_age")
        max_income = rules.get("max_income")
        scheme_gender = _normalize_gender(rules.get("gender") or scheme.get("gender"))
        bpl_required = rules.get("bpl_required") if rules.get("bpl_required") is not None else scheme.get("bpl_required")

        if max_income is not None:
            if user_income is None:
                missing_fields.append("annual_income")
            elif user_income > max_income:
                rejected_reasons.append(f"Income exceeds limit of Rs. {int(max_income):,}.")

        if min_age is not None or max_age is not None:
            if user_age is None:
                missing_fields.append("age")
            else:
                if min_age is not None and user_age < min_age:
                    rejected_reasons.append(f"Minimum age is {min_age}.")
                if max_age is not None and user_age > max_age:
                    rejected_reasons.append(f"Maximum age is {max_age}.")

        if scheme_gender and scheme_gender not in {"all", "any"}:
            if not user_gender:
                missing_fields.append("gender")
            elif user_gender != scheme_gender:
                rejected_reasons.append(f"Only for {scheme_gender} applicants.")

        if bpl_required is True:
            if user_bpl is None:
                missing_fields.append("bpl_status")
            elif user_bpl is False:
                rejected_reasons.append("Requires BPL status.")

        score = final_confidence({"profile": profile}, schemes, scheme)
        if rejected_reasons:
            ineligible_schemes.append(_scheme_payload(scheme, rejected_reasons, False, score))
            continue

        if missing_fields:
            reasons = [f"Need {field.replace('_', ' ')} to confirm eligibility." for field in dict.fromkeys(missing_fields)]
            uncertain_schemes.append(_scheme_payload(scheme, reasons, False, score, list(dict.fromkeys(missing_fields))))
            continue

        eligible_reasons = scheme.get("why_match") or ["Matches your profile criteria"]
        if isinstance(eligible_reasons, str):
            eligible_reasons = [eligible_reasons]
        if score < CONFIDENCE_THRESHOLD and not eligible_reasons:
            eligible_reasons = ["Low-confidence but rules satisfied."]
        eligible_schemes.append(_scheme_payload(scheme, list(eligible_reasons), True, score))

    eligible_schemes.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    uncertain_schemes.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    ineligible_schemes.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    logger.info(
        {
            "event": "eligibility_filter_summary",
            "eligible": len(eligible_schemes),
            "uncertain_needs_more_data": len(uncertain_schemes),
            "ineligible": len(ineligible_schemes),
        }
    )

    return {
        "eligible": eligible_schemes[:5],
        "uncertain_needs_more_data": uncertain_schemes[:5],
        "ineligible": ineligible_schemes[:5],
        "errors": [],
    }
