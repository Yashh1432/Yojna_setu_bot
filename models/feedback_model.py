"""
Feedback Model — stores user ratings per response for quality tracking.
Enables the feedback loop: user rates bot responses, data informs improvements.
"""

from datetime import datetime
from core.logger import get_logger
from models.db_client import db_client

logger = get_logger("models.feedback_model")


class FeedbackModel:
    def __init__(self):
        self.collection = db_client.db["feedback"] if db_client.db is not None else None

    def store_feedback(self, phone_number: str, trace_id: str, rating: int, comment: str = "") -> bool:
        """
        Store user feedback for a specific response.
        
        Args:
            phone_number: User identifier
            trace_id: The trace_id from the response being rated
            rating: 1-5 star rating
            comment: Optional text feedback
            
        Returns:
            True if stored successfully, False otherwise.
        """
        if self.collection is None:
            logger.warning({"event": "feedback_skip", "reason": "db_unavailable"})
            return False

        if not (1 <= rating <= 5):
            logger.warning({"event": "feedback_invalid", "rating": rating, "reason": "out_of_range"})
            return False

        try:
            self.collection.insert_one({
                "phone_number": phone_number,
                "trace_id": trace_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.utcnow()
            })
            logger.info({
                "event": "feedback_stored",
                "phone": phone_number,
                "trace_id": trace_id,
                "rating": rating
            })
            return True
        except Exception as e:
            logger.error({"event": "feedback_error", "error": str(e)})
            return False

    def get_average_rating(self, limit: int = 100) -> float:
        """Returns the average rating from the most recent `limit` feedback entries."""
        if self.collection is None:
            return 0.0
        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            ratings = [doc["rating"] for doc in cursor if "rating" in doc]
            return round(sum(ratings) / len(ratings), 2) if ratings else 0.0
        except Exception:
            return 0.0


feedback_model = FeedbackModel()
