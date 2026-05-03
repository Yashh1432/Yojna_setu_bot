from datetime import datetime, timedelta

from core.logger import get_logger
from models.db_client import db_client

logger = get_logger("models.llm_cache")


class LLMCacheModel:
    def __init__(self):
        self.collection = db_client.db["ai_response_cache"] if db_client.db is not None else None
        self.memory_cache = {}

    def get_cache(self, query_hash: str):
        now = datetime.utcnow()
        if self.collection is not None:
            doc = self.collection.find_one({"query_hash": query_hash, "expires_at": {"$gt": now}})
            return doc.get("response") if doc else None

        item = self.memory_cache.get(query_hash)
        if not item:
            return None
        if item["expires_at"] <= now:
            self.memory_cache.pop(query_hash, None)
            return None
        return item["response"]

    def set_cache(
        self,
        query_hash: str,
        input_str: str,
        response: str,
        model_name: str = "unknown",
        latency_ms: int = 0,
    ):
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=24)

        if self.collection is not None:
            self.collection.update_one(
                {"query_hash": query_hash},
                {
                    "$set": {
                        "query_hash": query_hash,
                        "input_str": input_str,
                        "response": response,
                        "model_name": model_name,
                        "latency_ms": latency_ms,
                        "created_at": now,
                        "expires_at": expires_at,
                    }
                },
                upsert=True,
            )
            return

        self.memory_cache[query_hash] = {
            "response": response,
            "expires_at": expires_at,
        }


llm_cache_model = LLMCacheModel()
