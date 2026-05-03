"""
RAG (Retrieval-Augmented Generation) Layer — MongoDB Scheme Retrieval.
Grounds eligibility decisions in MongoDB data, not LLM hallucination.
"""

import json
import os
import logging
from datetime import datetime
import uuid

logger = logging.getLogger("engine.rag")

SCHEMES_MONGO_COLLECTION = "schemes_structured"

# ── Quality gate (mirrors orchestrator logic) ──
def _is_quality_scheme(scheme: dict) -> bool:
    """Reject garbage entries before returning."""
    name = str(scheme.get("scheme_name") or "").strip()
    if not name or name.lower() in {"unnamed scheme", "unknown scheme", ""}:
        return False
    benefits = str(scheme.get("benefits") or scheme.get("description") or "").strip()
    if not benefits or len(benefits) < 20:
        return False
    if "no description available" in benefits.lower():
        return False
    return True

def _get_db():
    """Returns MongoDB db handle or None."""
    try:
        from models.db_client import db_client
        return db_client.db
    except Exception:
        return None

def sync_schemes_to_mongo(json_path: str | None = None) -> int:
    """
    One-time startup sync: load full_preprocessed_schemes.json → MongoDB.
    Skips if already populated. Returns count of documents in collection.
    Note: Schema V2 now requires background migration via scripts/migrate_schema.py.
    This function remains as an emergency fallback structural insert.
    """
    db = _get_db()
    if db is None:
        logger.warning("RAG: MongoDB unavailable, skipping sync.")
        return 0

    existing = db[SCHEMES_MONGO_COLLECTION].count_documents({})
    if existing > 0:
        logger.info(f"RAG: MongoDB already has {existing} schemes. Skipping sync.")
        return existing

    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'final_production_schemes.json'
        )

    try:
        logger.info(f"RAG: Syncing schemes from {json_path} to MongoDB...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            logger.error("RAG: JSON is empty or not a list.")
            return 0

        # Map to V2 structured format if not already compliant
        v2_data = []
        for d in data:
            if "scheme_id" not in d:
                 d["scheme_id"] = uuid.uuid4().hex
            if hasattr(d.get("eligibility"), "copy"):
                 pass # already dict
            elif "eligibility_criteria" in d:
                 # Minimal shim if not parsed properly by processor script
                 d["eligibility"] = {"min_age": None, "max_age": None, "max_income": None}
            v2_data.append(d)

        batch_size = 500
        total = 0
        for i in range(0, len(v2_data), batch_size):
            batch = v2_data[i:i + batch_size]
            try:
                db[SCHEMES_MONGO_COLLECTION].insert_many(batch, ordered=False)
                total += len(batch)
            except Exception as be:
                logger.warning(f"RAG: Batch {i//batch_size} insert error: {be}")

        logger.info(f"RAG: Synced {total} schemes to MongoDB.")
        return total

    except Exception as e:
        logger.error(f"RAG: Sync failed: {e}")
        return 0


def retrieve_schemes(category: str, state: str | None = None, language: str | None = "en", limit: int = 10) -> list:
    """
    Cache Layer 2 — RAG CACHE.
    Key: MD5("rag:" + category).
    Hit  → return cached scheme list instantly (0 DB queries).
    Miss → query MongoDB, cache result, return.
    """
    from services.cache_service import get_rag_cache, set_rag_cache

    # ── Cache read ──────────────────────────────────────────────
    cached = get_rag_cache(category, state=state, language=language)
    if cached is not None and isinstance(cached, list):
        logger.debug(f"RAG cache HIT for category '{category}' ({len(cached)} schemes)")
        return cached

    # ── MongoDB query ────────────────────────────────────────────
    db = _get_db()
    if db is None:
        logger.warning("RAG: MongoDB unavailable for scheme retrieval.")
        return []

    try:
        cursor = db[SCHEMES_MONGO_COLLECTION].find(
            {"category": {"$regex": category, "$options": "i"}},
            {
                "scheme_id": 1,
                "scheme_name": 1,
                "category": 1,
                "eligibility": 1,
                "benefits": 1,
                "state": 1,
                "documents_required": 1,
                "application_link": 1,
                "eligibility_criteria": 1,
                "_id": 0,
            }
        ).limit(limit)

        schemes = list(cursor)

        if not schemes:
            # Fallback: text search on name / benefits
            cursor = db[SCHEMES_MONGO_COLLECTION].find(
                {"$or": [
                    {"scheme_name": {"$regex": category, "$options": "i"}},
                    {"benefits": {"$regex": category, "$options": "i"}}
                ]}
            ).limit(limit)
            schemes = list(cursor)

        logger.info(f"RAG: Retrieved {len(schemes)} schemes for category '{category}'")

        # ── Quality gate ──
        schemes = [s for s in schemes if _is_quality_scheme(s)]
        logger.info(f"RAG: {len(schemes)} schemes after quality gate for category '{category}'")

        # ── Cache write ──────────────────────────────────────────
        if schemes:
            set_rag_cache(category, schemes, state=state, language=language)

        return schemes

    except Exception as e:
        logger.error(f"RAG: Query failed for category '{category}': {e}")
        return []
