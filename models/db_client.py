import os

from core.logger import get_logger

logger = get_logger("models.db_client")

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - optional dependency at runtime
    MongoClient = None


class DBClient:
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        if not self.uri or MongoClient is None:
            logger.warning("MongoDB unavailable. Running in local demo mode.")
            return

        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
                socketTimeoutMS=5000,
                retryWrites=True,
                retryReads=True,
                appName="YojnaSetuBot",
            )
            self.client.admin.command("ping")
            self.db = self.client["welfare_chatbot"]
            logger.info("Connected to MongoDB.")
        except Exception as exc:
            logger.warning(f"MongoDB connection failed. Falling back to local mode: {exc}")
            self.client = None
            self.db = None


db_client = DBClient()
