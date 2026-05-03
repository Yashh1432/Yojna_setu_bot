"""
Unified Input Sanitizer — ALL modules must pass input through this layer first.
Prevents SQL-injection-style attacks, strips noise, and normalizes currency/number formats.
"""

import re
import html

# Indian number word mappings
LAKH = 100_000
CRORE = 10_000_000

def sanitize_text(text: str) -> str:
    """
    Single entry point for all user inputs before ANY processing.
    Strip HTML, control chars, and excessive whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                      # &amp; → &
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars
    text = re.sub(r'\s+', ' ', text)                # Collapse whitespace
    return text.strip()[:1000]                      # Hard cap: 1000 chars


def normalize_numeric_text(text: str) -> str:
    """
    Normalise Indian currency/number expressions for reliable regex parsing.
    Must be applied to eligibility_criteria text AND user input before regex.
    
    Examples:
        "₹2,00,000"         → "200000"
        "2 lakh"            → "200000"
        "1.5 lakh"          → "150000"
        "50 thousand"       → "50000"
        "Rs. 1,50,000"      → "150000"
        "annual income 2L"  → "annual income 200000"
    """
    text = text.lower()

    # Remove commas inside numbers: 2,00,000 → 200000
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)   # Second pass for ×× commas

    # Rupee symbols / "rs" → drop (leave number intact)
    text = re.sub(r'₹|rs\.?\s*', '', text)

    # Handle decimal lakh:  1.5 lakh, 2lakh, 2 lac, 2l → 150000, 200000
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|\bl\b|l\b)',
        lambda m: str(int(float(m.group(1)) * LAKH)),
        text
    )

    # Handle crore
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:crores?|cr\b)',
        lambda m: str(int(float(m.group(1)) * CRORE)),
        text
    )

    # Handle thousand / k
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:thousands?|k\b)',
        lambda m: str(int(float(m.group(1)) * 1000)),
        text
    )

    return text


def parse_income_range(text: str) -> dict | None:
    """
    Detect and extract income ranges from user text.
    Returns:
        None                             → if no income mentioned
        {"val": int, "is_range": False}  → exact value
        {"val": int, "min": int, "max": int, "is_range": True}
                                         → range (val = upper bound / safe path)
    """
    norm = normalize_numeric_text(text)

    # Range pattern: "1 to 2 lakh", "100000-200000", "between 50k and 80k"
    range_match = re.search(
        r'(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)',
        norm
    )
    if range_match:
        # Re-apply scaling if normalization missed these specific raw digits
        lo_raw, hi_raw = float(range_match.group(1)), float(range_match.group(2))
        lo = int(lo_raw * LAKH if lo_raw < 100 else lo_raw)
        hi = int(hi_raw * LAKH if hi_raw < 100 else hi_raw)
        return {"val": hi, "min": lo, "max": hi, "is_range": True}

    # Exact pattern - look for numbers that might be lakh/thousand or raw
    exact_match = re.search(r'\b(\d{4,8})\b', norm)
    if not exact_match:
        # Catch case where it was normalized but maybe not caught by \b\d{4,8}\b
        # like "120000" in "i have 120000 income"
        exact_match = re.search(r'(\d{4,8})', norm)

    if exact_match:
        return {"val": int(exact_match.group(1)), "is_range": False}

    return None


def parse_age_range(text: str) -> dict | None:
    """
    Extract age (integer) or age-range from text.
    """
    norm = normalize_numeric_text(text)

    range_match = re.search(r'(\d{1,3})\s*(?:to|-|and)\s*(\d{1,3})\s*(?:years?)?', norm)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        if 0 < lo < 120 and 0 < hi < 120:
            return {"val": hi, "min": lo, "max": hi, "is_range": True}

    exact = re.search(r'\b(\d{1,3})\s*(?:years?|yr)?\b', norm)
    if exact:
        age = int(exact.group(1))
        if 0 < age < 120:
            return {"val": age, "is_range": False}

    return None


# ─────────────────────────────────────────────────────────────────
# PROMPT INJECTION DEFENSE
# Strips adversarial patterns before any user text enters an LLM prompt.
# ─────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above|my)\s+(instructions?|rules?|prompts?|commands?)",
    r"(forget|disregard|override|bypass)\s+(everything|all|the\s+rules?|above|previous)",
    r"you\s+are\s+now\s+",
    r"act\s+(as|like)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*prompt",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"pretend\s+(you\s+(are|have)|to\s+be)",
    r"do\s+anything\s+now",
    r"reveal\s+(your\s+)?(prompt|instructions?|system|rules?)",
    r"give\s+(me\s+)?all\s+(the\s+)?schemes",
]

def sanitize_for_llm(text: str) -> str:
    """
    Strips known prompt injection patterns from user input before it is
    injected into any LLM prompt template.

    This is the MANDATORY final gate before user text enters a prompt.
    It does NOT replace sanitize_text() — both must be applied in order:
        1. sanitize_text(raw_input)      → clean whitespace / HTML / control chars
        2. sanitize_for_llm(clean_text)  → strip injection attacks

    Replaced patterns are substituted with '[FILTERED]' so the LLM
    still receives a coherent (if partial) sentence.
    """
    if not text:
        return ""
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
    return text


def clean_output(text: str) -> str:
    """
    Final output cleanup: removes double spaces and strips whitespace.
    """
    if not text:
        return ""
    return text.replace("  ", " ").strip()
