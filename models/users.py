from copy import deepcopy
from datetime import datetime

from core.logger import get_logger
from models.db_client import db_client

logger = get_logger("models.users")

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
    "disability_status",
    "student_status",
    "farmer_status",
    "business_status",
    "bpl_status",
]

BOOLEAN_FIELDS = {
    "disability_status",
    "student_status",
    "farmer_status",
    "business_status",
    "bpl_status",
}


def _default_profile():
    return {
        "age": None,
        "annual_income": None,
        "income": None,
        "occupation": None,
        "education_level": None,
        "state": None,
        "gender": None,
        "category": None,
        "academic_percentage": None,
        "caste_category": None,
        "disability_status": None,
        "student_status": None,
        "farmer_status": None,
        "business_status": None,
        "bpl_status": None,
    }


def _default_user(phone_number: str):
    now = datetime.utcnow()
    return {
        "phone_number": phone_number,
        "language": None,
        "conv_state": "active",
        "profile": _default_profile(),
        "asked_fields": [],
        "profile_sources": {},
        "last_schemes": [],
        "selected_scheme": None,
        "created_at": now,
        "updated_at": now,
        "last_active": now,
    }


def _normalize_bool(value):
    if isinstance(value, bool) or value is None:
        return value

    lowered = str(value).strip().lower()
    if lowered in {"yes", "true", "1", "y"}:
        return True
    if lowered in {"no", "false", "0", "n"}:
        return False
    return None


def _normalize_profile_values(profile: dict, include_defaults: bool = False) -> dict:
    normalized = _default_profile() if include_defaults else {}

    for field in PROFILE_FIELDS:
        if field not in profile:
            continue

        value = profile[field]
        if field in BOOLEAN_FIELDS:
            normalized[field] = _normalize_bool(value)
        elif value in ("", []):
            normalized[field] = None
        else:
            normalized[field] = value

    annual_income = normalized.get("annual_income")
    income = normalized.get("income")
    if "annual_income" in normalized and "income" not in normalized:
        normalized["income"] = annual_income
    if "income" in normalized and "annual_income" not in normalized:
        normalized["annual_income"] = income

    return normalized


class UserModel:
    def __init__(self):
        self.collection = db_client.db["users"] if db_client.db is not None else None
        self.memory_store = {}

    def normalize_user(self, user_doc):
        if not user_doc:
            return None

        normalized = deepcopy(user_doc)
        profile = normalized.get("profile") or {}

        merged_profile = _default_profile()
        merged_profile.update(_normalize_profile_values(profile, include_defaults=False))

        normalized["profile"] = merged_profile
        normalized.setdefault("asked_fields", [])
        normalized.setdefault("profile_sources", {})
        normalized.setdefault("language", None)
        normalized.setdefault("conv_state", "active")
        normalized.setdefault("last_schemes", [])
        normalized.setdefault("selected_scheme", None)
        return normalized

    def get_user(self, phone_number):
        if self.collection is not None:
            doc = self.collection.find_one({"phone_number": phone_number})
            return self.normalize_user(doc)

        return self.normalize_user(self.memory_store.get(phone_number))

    def create_or_get_user(self, phone_number):
        existing = self.get_user(phone_number)
        if existing:
            return existing, False

        user = _default_user(phone_number)
        if self.collection is not None:
            self.collection.insert_one(deepcopy(user))
        else:
            self.memory_store[phone_number] = deepcopy(user)
        return self.normalize_user(user), True

    def update_user(self, phone_number, data):
        user, _ = self.create_or_get_user(phone_number)
        updated = deepcopy(user)

        for key, value in data.items():
            if key == "profile" and isinstance(value, dict):
                merged_profile = deepcopy(updated.get("profile") or _default_profile())
                merged_profile.update(_normalize_profile_values(value, include_defaults=False))
                updated["profile"] = _normalize_profile_values(merged_profile, include_defaults=True)
            elif key == "asked_fields" and isinstance(value, list):
                updated["asked_fields"] = value
            elif key == "profile_sources" and isinstance(value, dict):
                updated["profile_sources"].update(value)
            else:
                updated[key] = value

        now = datetime.utcnow()
        updated["updated_at"] = now
        updated["last_active"] = now

        if self.collection is not None:
            self.collection.update_one(
                {"phone_number": phone_number},
                {"$set": updated},
                upsert=True,
            )
        else:
            self.memory_store[phone_number] = updated
        return self.normalize_user(updated)

    def update_profile_with_priority(self, phone_number, new_entities, source="manual"):
        priority = {"manual": 10, "ocr": 5, "llm": 1}
        user, _ = self.create_or_get_user(phone_number)
        profile = deepcopy(user.get("profile") or {})
        sources = deepcopy(user.get("profile_sources") or {})
        incoming_priority = priority.get(source, 0)

        normalized_entities = _normalize_profile_values(new_entities or {}, include_defaults=False)

        for key, value in normalized_entities.items():
            if value in (None, "", []):
                continue
            current_priority = priority.get(sources.get(key), 0)
            if incoming_priority >= current_priority:
                profile[key] = value
                sources[key] = source

        return self.update_user(
            phone_number,
            {"profile": profile, "profile_sources": sources},
        )

    def reset_user(self, phone_number):
        current, _ = self.create_or_get_user(phone_number)
        keep_language = current.get("language")
        reset_doc = _default_user(phone_number)
        reset_doc["language"] = keep_language
        if self.collection is not None:
            self.collection.update_one(
                {"phone_number": phone_number},
                {"$set": reset_doc},
                upsert=True,
            )
        else:
            self.memory_store[phone_number] = reset_doc
        return self.normalize_user(reset_doc)


user_model = UserModel()
