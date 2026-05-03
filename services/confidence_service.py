"""
Confidence Service – calculates confidence scores for extraction, semantic RAG, and rule matching.
Provides a configurable confidence threshold (default 0.6) via environment variable.
"""

from typing import List, Dict
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Confidence threshold below which the bot should ask the user for clarification.
# Can be overridden via environment variable CONFIDENCE_THRESHOLD.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# ---------------------------------------------------------------------------
# Extraction confidence – how complete the LLM extraction payload is.
# ---------------------------------------------------------------------------
def extraction_confidence(profile: dict) -> float:
    """Return a confidence score (0‑1) based on presence of key profile fields.

    The score is the fraction of the four core fields (age, income,
    state, gender) that are present in the extracted profile.
    """
    normalized = profile.get("profile") if isinstance(profile, dict) and isinstance(profile.get("profile"), dict) else profile
    normalized = normalized or {}
    total = 4
    present = sum(bool(normalized.get(k)) for k in ["age", "income", "state", "gender"])
    return present / total

# ---------------------------------------------------------------------------
# Semantic RAG confidence – average of top‑5 similarity scores.
# ---------------------------------------------------------------------------
def rag_confidence(rag_results: List[dict]) -> float:
    """Calculate confidence from semantic search results.

    Returns the mean of the top‑5 cosine similarity scores (0‑1). If no results are
    available the confidence is 0.0.
    """
    if not rag_results:
        return 0.0
    top_scores = [r.get("score", 0.0) for r in rag_results[:5]]
    return float(sum(top_scores) / len(top_scores))

# ---------------------------------------------------------------------------
# Rule‑based confidence – how well the scheme's hard eligibility rules match.
# ---------------------------------------------------------------------------
def rule_confidence(profile: dict, scheme: dict) -> float:
    """Return a confidence score (0‑1) based on rule matching.

    The function checks two core rule categories: income and age.
    Each satisfied rule contributes 1 point; missing rules are considered a
    neutral pass (score + 1). The final score is the fraction of the checks.
    """
    total = 2
    score = 0
    rules = scheme.get("eligibility", {})

    # Income – must be <= max_income if defined.
    if rules.get("max_income") and profile.get("income") is not None:
        if profile["income"] <= rules["max_income"]:
            score += 1
    else:
        score += 1

    # Age – must be >= min_age if defined.
    if rules.get("min_age") and profile.get("age") is not None:
        if profile["age"] >= rules["min_age"]:
            score += 1
    else:
        score += 1

    return score / total

# ---------------------------------------------------------------------------
# Final confidence – weighted combination of the three components.
# ---------------------------------------------------------------------------
def final_confidence(profile: dict, rag_results: List[dict], scheme: dict) -> float:
    """Combine extraction, RAG, and rule confidences into a single score.

    Weights:
        * Extraction clarity – 40 %
        * Semantic relevance – 30 %
        * Hard rule matching – 30 %
    The result is rounded to two decimal places.
    """
    e = extraction_confidence(profile)
    r = rag_confidence(rag_results)
    rule = rule_confidence(profile, scheme)
    return round(0.4 * e + 0.3 * r + 0.3 * rule, 2)


# ---------------------------------------------------------------------------
# Confidence tier classification — makes scores actionable.
# ---------------------------------------------------------------------------
def classify_confidence(score: float) -> str:
    """Classify a confidence score into a human-readable tier.

    Tiers:
        * high   — score >= 0.8. Strong match, show confidently.
        * medium — score >= 0.6. Acceptable, show with mild disclaimer.
        * low    — score < 0.6. Weak match, consider escalation.
    """
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    else:
        return "low"

