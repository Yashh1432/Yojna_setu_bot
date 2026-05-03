from models.db_client import db_client
import logging
from datetime import datetime
import uuid

logger = logging.getLogger("models.messages")

class MessagesModel:
    def __init__(self):
        self.sessions = db_client.db['sessions'] if db_client.db is not None else None
        self.messages = db_client.db['messages'] if db_client.db is not None else None

    def create_or_update_session(self, phone_number):
        if self.sessions is None:
            return None
            
        # Find active session or create new
        # Simplified: we use phone_number as session_id to maintain a single continuous thread
        # for ease of WhatsApp integration, or we create a new one.
        # Here we just look up by phone_number.
        session = self.sessions.find_one({"phone_number": phone_number})
        now = datetime.utcnow()
        if session:
            self.sessions.update_one(
                {"_id": session["_id"]},
                {"$set": {"last_active": now}}
            )
            return session["session_id"]
        else:
            session_id = uuid.uuid4().hex
            self.sessions.insert_one({
                "session_id": session_id,
                "phone_number": phone_number,
                "created_at": now,
                "last_active": now
            })
            return session_id

    def log_message(self, phone_number, sender, msg_type, text, audio_url=None):
        if self.messages is None:
            return
            
        session_id = self.create_or_update_session(phone_number)
        if not session_id:
            return
            
        msg = {
            "message_id": uuid.uuid4().hex,
            "session_id": session_id,
            "sender": sender, # 'user' or 'bot'
            "message_type": msg_type, # 'text' or 'voice'
            "text": text,
            "audio_url": audio_url,
            "timestamp": datetime.utcnow()
        }
        
        try:
            self.messages.insert_one(msg)
        except Exception as e:
            logger.error(f"Failed to log message: {e}")

messages_model = MessagesModel()
