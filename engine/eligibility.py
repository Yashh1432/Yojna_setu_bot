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


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


def _occupation_matches(user_occupation: Any, required_occupation: Any) -> tuple[bool | None, str]:
    required = str(required_occupation or "").strip().lower()
    if not required:
        return None, ""
    user_value = str(user_occupation or "").strip().lower()
    if not user_value:
        return False, "occupation"
    required_tokens = [token.strip() for token in required.replace("/", ",").split(",") if token.strip()]
    if not required_tokens:
        required_tokens = [required]
    if any(token in user_value for token in required_tokens):
        return True, ""
    return False, required_tokens[0]


def evaluate_scheme_eligibility_for_profile(raw_profile: dict[str, Any], scheme: dict[str, Any]) -> dict[str, Any]:
    profile = _profile_view(raw_profile or {})
    rules = scheme.get("eligibility") or {}

    user_age = profile.get("age")
    user_income = _income(profile)
    user_gender = _normalize_gender(profile.get("gender"))
    user_bpl = _normalize_bool(profile.get("bpl_status"))
    user_disability = _normalize_bool(profile.get("disability_status"))
    user_caste = str(profile.get("caste_category") or "").strip().lower()
    user_occupation = profile.get("occupation")
    user_percentage = profile.get("academic_percentage")

    missing_fields: list[str] = []
    rejected_reasons: list[str] = []
    rejection_code: str | None = None

    min_age = rules.get("min_age")
    max_age = rules.get("max_age")
    max_income = rules.get("max_income")
    scheme_gender = _normalize_gender(rules.get("gender") or scheme.get("gender"))
    bpl_required = rules.get("bpl_required") if rules.get("bpl_required") is not None else scheme.get("bpl_required")
    disability_required = rules.get("disability_required") if rules.get("disability_required") is not None else scheme.get("disability_required")
    occupation_required = rules.get("occupation") or scheme.get("occupation")
    caste_required = str(rules.get("caste") or scheme.get("caste") or "").strip().lower()
    min_percentage = rules.get("min_percentage") if rules.get("min_percentage") is not None else rules.get("academic_percentage")

    if max_income is not None:
        if user_income is None:
            missing_fields.append("annual_income")
        elif user_income > max_income:
            rejected_reasons.append(f"Income exceeds limit of Rs. {int(max_income):,}.")
            rejection_code = rejection_code or "income_exceeded"

    if min_age is not None or max_age is not None:
        if user_age is None:
            missing_fields.append("age")
        else:
            if min_age is not None and user_age < min_age:
                rejected_reasons.append(f"Minimum age is {min_age}.")
                rejection_code = rejection_code or "age_mismatch"
            if max_age is not None and user_age > max_age:
                rejected_reasons.append(f"Maximum age is {max_age}.")
                rejection_code = rejection_code or "age_mismatch"

    if scheme_gender and scheme_gender not in {"all", "any"}:
        if not user_gender:
            missing_fields.append("gender")
        elif user_gender != scheme_gender:
            rejected_reasons.append(f"Only for {scheme_gender} applicants.")
            rejection_code = rejection_code or "gender_mismatch"

    if _normalize_bool(bpl_required) is True:
        if user_bpl is None:
            missing_fields.append("bpl_status")
        elif user_bpl is False:
            rejected_reasons.append("Requires BPL status.")
            rejection_code = rejection_code or "bpl_mismatch"

    if _normalize_bool(disability_required) is True:
        if user_disability is None:
            missing_fields.append("disability_status")
        elif user_disability is False:
            rejected_reasons.append("Requires disability status.")
            rejection_code = rejection_code or "disability_mismatch"

    occ_match, occ_info = _occupation_matches(user_occupation, occupation_required)
    if occ_match is False:
        if occ_info == "occupation":
            missing_fields.append("occupation")
        else:
            rejected_reasons.append(f"Requires occupation: {occ_info}.")
            rejection_code = rejection_code or "occupation_mismatch"

    if caste_required:
        if not user_caste:
            missing_fields.append("caste_category")
        elif caste_required not in user_caste:
            rejected_reasons.append(f"Requires caste category: {caste_required}.")
            rejection_code = rejection_code or "caste_mismatch"

    if min_percentage is not None:
        if user_percentage is None:
            missing_fields.append("academic_percentage")
        else:
            try:
                if float(user_percentage) < float(min_percentage):
                    rejected_reasons.append(f"Minimum academic percentage is {min_percentage}.")
                    rejection_code = rejection_code or "academic_percentage_mismatch"
            except Exception:
                missing_fields.append("academic_percentage")

    if rejected_reasons:
        return {
            "status": "ineligible",
            "reasons": rejected_reasons,
            "missing_fields": [],
            "rejection_code": rejection_code or "eligibility_mismatch",
        }

    if missing_fields:
        unique_missing = list(dict.fromkeys(missing_fields))
        reasons = [f"Need {field.replace('_', ' ')} to confirm eligibility." for field in unique_missing]
        return {
            "status": "uncertain",
            "reasons": reasons,
            "missing_fields": unique_missing,
            "rejection_code": "missing_required_field",
        }

    return {
        "status": "eligible",
        "reasons": [],
        "missing_fields": [],
        "rejection_code": None,
    }


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

    eligible_schemes: list[dict[str, Any]] = []
    uncertain_schemes: list[dict[str, Any]] = []
    ineligible_schemes: list[dict[str, Any]] = []

    logger.info({"event": "eligibility_filter_start", "category": category, "scheme_count": len(schemes)})

    for scheme in schemes:
        eval_result = evaluate_scheme_eligibility_for_profile(profile, scheme)
        missing_fields: list[str] = list(eval_result.get("missing_fields") or [])
        rejected_reasons: list[str] = list(eval_result.get("reasons") or []) if eval_result.get("status") == "ineligible" else []

        score = final_confidence({"profile": profile}, schemes, scheme)
        if eval_result.get("status") == "ineligible":
            ineligible_schemes.append(_scheme_payload(scheme, rejected_reasons, False, score))
            continue

        if eval_result.get("status") == "uncertain":
            reasons = list(eval_result.get("reasons") or [])
            uncertain_schemes.append(_scheme_payload(scheme, reasons, False, score, missing_fields))
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
        "eligible": eligible_schemes[:10],
        "uncertain_needs_more_data": uncertain_schemes[:10],
        "ineligible": ineligible_schemes[:10],
        "errors": [],
    }
