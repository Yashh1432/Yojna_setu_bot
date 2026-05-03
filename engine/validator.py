"""
Validation + normalization helpers used by the bot state machine and tests.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from core.logger import get_logger

logger = get_logger("engine.validator")

INDIAN_STATES = [
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "delhi",
    "jammu",
    "kashmir",
    "pondicherry",
    "chandigarh",
    "ladakh",
    "andaman",
    "nicobar",
    "lakshadweep",
    "dadra",
    "daman",
    "diu",
]

GENDER_NORMALIZE = {
    "male": "male",
    "m": "male",
    "man": "male",
    "boy": "male",
    "purush": "male",
    "aadmi": "male",
    "ladka": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "girl": "female",
    "lady": "female",
    "mahila": "female",
    "aurat": "female",
    "stri": "female",
    "stree": "female",
    "ladki": "female",
    "other": "other",
    "others": "other",
    "transgender": "other",
}

# ── Category aliases mapping to ACTUAL dataset category values ──
# Dataset categories: Education, Agriculture, Employment, Senior Citizen,
# Health, Women & Child, Financial Assistance, Housing, Others, Disability
CATEGORY_ALIAS: dict[str, str] = {
    # Education
    "student": "education",
    "scholarship": "education",
    "education": "education",
    "school": "education",
    "college": "education",
    "university": "education",
    "vidya": "education",
    "shiksha": "education",
    "शिक्षा": "education",
    "શિક્ષા": "education",
    "கல்வி": "education",
    "విద్య": "education",
    "ಶಿಕ್ಷಣ": "education",
    "শিক্ষা": "education",
    "تعلیم": "education",
    # Agriculture
    "farmer": "agriculture",
    "kisan": "agriculture",
    "khedut": "agriculture",
    "agriculture": "agriculture",
    "agri": "agriculture",
    "crop": "agriculture",
    "खेती": "agriculture",
    "કૃષિ": "agriculture",
    "விவசாயம்": "agriculture",
    "వ్యవసాయం": "agriculture",
    "ಕೃಷಿ": "agriculture",
    "কৃষি": "agriculture",
    "زراعت": "agriculture",
    # Senior Citizen
    "senior citizen": "senior_citizen",
    "senior": "senior_citizen",
    "old age": "senior_citizen",
    "old-age": "senior_citizen",
    "oldage": "senior_citizen",
    "pension": "senior_citizen",
    "elderly": "senior_citizen",
    "vridh": "senior_citizen",
    "vruddh": "senior_citizen",
    "vayo": "senior_citizen",
    "वृद्ध": "senior_citizen",
    "वृद्धा": "senior_citizen",
    "पेंशन": "senior_citizen",
    "बुजुर्ग": "senior_citizen",
    "વૃદ્ધ": "senior_citizen",
    "પેન્શન": "senior_citizen",
    "வயதானவர்": "senior_citizen",
    "ஓய்வூதியம்": "senior_citizen",
    "వృద్ధ": "senior_citizen",
    "పెన్షన్": "senior_citizen",
    "ಹಿರಿಯ": "senior_citizen",
    "ಪಿಂಚಣಿ": "senior_citizen",
    "বয়স্ক": "senior_citizen",
    "পেনশন": "senior_citizen",
    "بزرگ": "senior_citizen",
    "پینشن": "senior_citizen",
    # Health
    "health": "health",
    "aarogya": "health",
    "arogya": "health",
    "medical": "health",
    "hospital": "health",
    "treatment": "health",
    "स्वास्थ्य": "health",
    "आरोग्य": "health",
    "આરોગ્ય": "health",
    "சுகாதாரம்": "health",
    "ఆరోగ్యం": "health",
    "ಆರೋಗ್ಯ": "health",
    "স্বাস্থ্য": "health",
    "صحت": "health",
    # Women & Child
    "women": "women_child",
    "mahila": "women_child",
    "woman": "women_child",
    "widow": "women_child",
    "vidhwa": "women_child",
    "vidhva": "women_child",
    "child": "women_child",
    "girl": "women_child",
    "beti": "women_child",
    "महिला": "women_child",
    "विधवा": "women_child",
    "મહિલા": "women_child",
    "વિધવા": "women_child",
    "பெண்கள்": "women_child",
    "మహిళ": "women_child",
    "ಮಹಿಳೆ": "women_child",
    "মহিলা": "women_child",
    "خواتین": "women_child",
    # Employment
    "job": "employment",
    "employment": "employment",
    "skill": "employment",
    "naukri": "employment",
    "rojgar": "employment",
    "work": "employment",
    "apprentice": "employment",
    "रोजगार": "employment",
    "નોકરી": "employment",
    "வேலை": "employment",
    "ఉపాధి": "employment",
    "ಉದ್ಯೋಗ": "employment",
    "কর্মসংস্থান": "employment",
    "روزگار": "employment",
    # Financial Assistance
    "loan": "finance_business",
    "finance": "finance_business",
    "business": "finance_business",
    "subsidy": "finance_business",
    "credit": "finance_business",
    "entrepreneur": "finance_business",
    "bank": "finance_business",
    "mudra": "finance_business",
    "ऋण": "finance_business",
    "લોન": "finance_business",
    "கடன்": "finance_business",
    "రుణం": "finance_business",
    "ಸಾಲ": "finance_business",
    "ঋণ": "finance_business",
    "قرض": "finance_business",
    # Housing
    "housing": "housing",
    "house": "housing",
    "awas": "housing",
    "home": "housing",
    "shelter": "housing",
    "आवास": "housing",
    "ઘર": "housing",
    "வீடு": "housing",
    "ఇల్లు": "housing",
    "ಮನೆ": "housing",
    "বাসস্থান": "housing",
    "مکان": "housing",
    # Disability
    "disability": "disability",
    "disabled": "disability",
    "divyang": "disability",
    "handicap": "disability",
    "handicapped": "disability",
    "दिव्यांग": "disability",
    "विकलांग": "disability",
    "દિવ્યાંગ": "disability",
    "மாற்றுத்திறனாளி": "disability",
    "వికలాంగ": "disability",
    "ವಿಕಲಚೇತನ": "disability",
    "প্রতিবন্ধী": "disability",
    "معذور": "disability",
    # Others
    "other": "social_welfare",
    "general": "social_welfare",
}

# ── Subcategory keywords for secondary matching within a category ──
SUBCATEGORY_KEYWORDS: dict[str, list[str]] = {
    "pension": ["pension", "पेंशन", "પેન્શન", "ஓய்வூதியம்", "పెన్షన్", "ಪಿಂಚಣಿ", "পেনশন", "پینشن"],
    "scholarship": ["scholarship", "छात्रवृत्ति", "શિષ્યવૃત્તિ", "உதவித்தொகை", "స్కాలర్‌షిప్", "ವಿದ್ಯಾರ್ಥಿವೇತನ", "বৃত্তি"],
    "subsidy": ["subsidy", "सब्सिडी", "સબસિડી", "மானியம்", "సబ్సిడీ", "ಸಬ್ಸಿಡಿ", "ভর্তুকি"],
    "insurance": ["insurance", "bima", "बीमा", "વીમો", "காப்பீடு", "బీమా", "ವಿಮೆ", "বীমা"],
    "training": ["training", "प्रशिक्षण", "તાલીમ", "பயிற்சி", "శిక్షణ", "ತರಬೇತಿ", "প্রশিক্ষণ"],
    "maternity": ["maternity", "pregnancy", "pregnant", "मातृत्व", "માતૃત્વ", "கர்ப்பிணி", "గర్భిణి"],
    "ration": ["ration", "bpl", "राशन", "રાશન", "ரேஷன்", "రేషన్", "ರೇಷನ್", "রেশন"],
    "pilgrimage": ["pilgrimage", "tirthdarshan", "tirth", "तीर्थ", "તીર્થ", "யாத்திரை", "తీర్థయాత్ర"],
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "education": ["student", "scholarship", "education", "school", "college", "university", "vidya", "shiksha"],
    "agriculture": ["farmer", "kisan", "khedut", "agriculture", "agri", "crop", "kheti"],
    "health": ["health", "aarogya", "arogya", "આરોગ્ય", "medical", "hospital", "treatment"],
    "employment": ["job", "rojgar", "employment", "skill", "work", "naukri", "apprentice"],
    "women & child": ["women", "mahila", "widow", "vidhwa", "vidhva", "single mother", "beti", "child", "girl"],
    "financial assistance": ["loan", "finance", "business", "subsidy", "credit", "entrepreneur", "mudra", "bank"],
    "housing": ["house", "awas", "housing", "home", "shelter"],
    "senior citizen": ["senior", "old age", "pension", "elderly", "vridh", "vruddh", "vayo", "बुजुर्ग", "વૃદ્ધ"],
    "disability": ["disability", "disabled", "divyang", "handicap", "दिव्यांग", "विकलांग", "દિવ્યાંગ"],
    "others": ["other", "general"],
}


def normalize_category(text: str) -> tuple[str | None, str | None]:
    """Normalize raw text to (canonical_category, subcategory).

    canonical_category matches actual dataset values (title-case).
    subcategory is an optional secondary signal for ranking.
    """
    if not text:
        return None, None
    source = str(text).strip().lower()
    if not source:
        return None, None

    # 1. Direct alias lookup (exact or substring)
    canonical: str | None = None
    for alias, cat in CATEGORY_ALIAS.items():
        if alias in source:
            canonical = cat
            break

    # 2. Keyword scan fallback
    if not canonical:
        for cat_key, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in source for kw in keywords):
                # Map to title-case dataset value
                canonical = CATEGORY_ALIAS.get(cat_key) or cat_key.title()
                break

    # 3. Subcategory detection
    subcategory: str | None = None
    for sub_key, sub_keywords in SUBCATEGORY_KEYWORDS.items():
        if any(kw in source for kw in sub_keywords):
            subcategory = sub_key
            break

    return canonical, subcategory


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if parsed.is_integer():
        return int(parsed)
    return int(round(parsed))


def _normalize_income(value) -> int | None:
    parsed = _coerce_int(value)
    if parsed is None:
        return None
    if 0 < parsed < 100:
        parsed *= 100000
    return parsed if 0 <= parsed <= 100_000_000 else None


def normalize_state_name(raw_state: Any) -> str | None:
    if raw_state is None:
        return None
    state_str = str(raw_state).strip().lower()
    if not state_str:
        return None

    for state in INDIAN_STATES:
        if state_str == state or state_str in state or state in state_str:
            return state

    matches = get_close_matches(state_str, INDIAN_STATES, n=1, cutoff=0.75)
    return matches[0] if matches else None


def normalize_income_value(value: Any) -> int | None:
    return _normalize_income(value)


def normalize_gender_value(value: Any) -> str | None:
    if value is None:
        return None
    return GENDER_NORMALIZE.get(str(value).strip().lower())


def infer_scheme_category(*texts: Any) -> str | None:
    merged = " ".join(str(part or "").strip().lower() for part in texts if str(part or "").strip())
    if not merged:
        return None

    # Fast path: exact/substring keyword match (covers native scripts too).
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(str(keyword).lower() in merged for keyword in keywords):
            return category

    # Fuzzy/phonetic-ish path for romanized typos:
    # - Tokenize ASCII words only
    # - Match tokens to ASCII keywords by limited edit distance
    tokens = re.findall(r"[a-z]{3,}", merged)
    if not tokens:
        return None

    def _levenshtein_limited(a: str, b: str, max_dist: int) -> int:
        if a == b:
            return 0
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        if len(a) > len(b):
            a, b = b, a
        previous = list(range(len(a) + 1))
        for i, ch_b in enumerate(b, start=1):
            current = [i]
            # Early exit: track best value in the row.
            best = current[0]
            for j, ch_a in enumerate(a, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (0 if ch_a == ch_b else 1)
                cost = min(insert_cost, delete_cost, replace_cost)
                current.append(cost)
                if cost < best:
                    best = cost
            previous = current
            if best > max_dist:
                return max_dist + 1
        return previous[-1]

    def _token_matches_keyword(token: str, keyword: str) -> bool:
        token = token.strip().lower()
        keyword = keyword.strip().lower()
        if not token or not keyword:
            return False
        if token == keyword:
            return True
        if len(token) < 4 or len(keyword) < 4:
            return False
        # Avoid noisy matches like "bank" -> "bangla"
        if token[0] != keyword[0]:
            return False
        max_dist = 1 if max(len(token), len(keyword)) <= 7 else 2
        return _levenshtein_limited(token, keyword, max_dist=max_dist) <= max_dist

    for category, keywords in CATEGORY_KEYWORDS.items():
        ascii_keywords = [kw for kw in keywords if re.fullmatch(r"[a-z][a-z ]{2,}", str(kw).lower())]
        if not ascii_keywords:
            continue
        for token in tokens:
            if any(_token_matches_keyword(token, kw) for kw in ascii_keywords):
                return category
    return None


def validate_entities(entities: dict) -> dict:
    """
    Normalize entity fields into a flat dictionary.
    Expected shape: {"age": int?, "income": int?, "state": str?, "gender": str?, ...}
    """
    entities = entities or {}
    out: dict = {}

    age = _coerce_int(entities.get("age"))
    if age is not None and 1 <= age <= 120:
        out["age"] = age

    income = _normalize_income(entities.get("income"))
    if income is not None:
        out["income"] = income

    state = normalize_state_name(entities.get("state"))
    if state:
        out["state"] = state

    gender_raw = entities.get("gender")
    if gender_raw is not None:
        gender = normalize_gender_value(gender_raw)
        if gender:
            out["gender"] = gender
            if gender == "female" and not entities.get("category"):
                out["category"] = "Women"

    category_parts = [
        str(entities.get("category") or "").strip().lower(),
        str(entities.get("sub_context") or "").strip().lower(),
        str(entities.get("raw_text") or "").strip().lower(),
    ]
    category_search = " ".join(part for part in category_parts if part)

    if "category" not in out and category_search:
        inferred = infer_scheme_category(category_search)
        if inferred:
            out["category"] = inferred.title()
        else:
            for key, value in CATEGORY_ALIAS.items():
                if key in category_search:
                    out["category"] = value
                    break

    sub_context = str(entities.get("sub_context") or "").strip().lower()
    if sub_context:
        out["sub_context"] = sub_context

    logger.debug({"event": "validate_entities", "normalized": out})
    return out
