import unittest
from unittest.mock import patch

from engine.state_manager import handle_message
from models.users import user_model


class TestConversationalFlow(unittest.TestCase):
    @staticmethod
    def _payload(**kwargs):
        base = {
            "language": "en",
            "intent": "unknown",
            "occupation": None,
            "category": None,
            "age": None,
            "income": None,
            "state": None,
            "gender": None,
            "caste_category": None,
            "academic_percentage": None,
            "bpl_status": None,
            "confidence": 0.2,
        }
        base.update(kwargs)
        return base

    def _fake_extract(self, message: str, **kwargs):
        lowered = str(message).strip().lower()
        if "student" in lowered and "karnataka" in lowered:
            return self._payload(
                intent="scheme_search",
                occupation="student",
                category="education",
                age=25,
                state="Karnataka",
                confidence=0.95,
            )
        if "documents" in lowered:
            return self._payload(intent="general_query", confidence=0.8)
        return self._payload()

    def setUp(self):
        user_model.memory_store.clear()
        self._next_q_patcher = patch(
            "engine.state_manager.decide_next_question_llm",
            return_value={"next_field": None, "question_english": None, "reason": "offline-test"},
        )
        self._next_q_patcher.start()
        self._extract_patcher = patch("engine.state_manager.extract_profile_llm", side_effect=self._fake_extract)
        self._extract_patcher.start()
        self._lang_pick_patcher = patch("engine.state_manager.infer_language_selection_llm", side_effect=self._fake_language_pick)
        self._lang_pick_patcher.start()
        self._tx_to_en_patcher = patch("engine.state_manager.translate_to_english", side_effect=lambda text, source_lang: text)
        self._tx_to_en_patcher.start()
        self._tx_from_en_patcher = patch("engine.state_manager.translate_from_english", side_effect=lambda text, target_lang: text)
        self._tx_from_en_patcher.start()

    def _fake_language_pick(self, message: str):
        lowered = str(message).strip().lower()
        if lowered in {"english"}:
            return {"selected_language": "en", "confidence": 0.95}
        if lowered in {"bengali"}:
            return {"selected_language": "bn", "confidence": 0.95}
        return {"selected_language": None, "confidence": 0.0}

    def tearDown(self):
        self._extract_patcher.stop()
        self._next_q_patcher.stop()
        self._lang_pick_patcher.stop()
        self._tx_to_en_patcher.stop()
        self._tx_from_en_patcher.stop()

    def test_first_turn_always_asks_language_bilingual(self):
        response, state = handle_message("u1", "hello")
        self.assertEqual(state, "awaiting_language")
        text = response["response"].lower()
        self.assertIn("please choose your preferred language", text)
        self.assertIn("1", response["response"])

    def test_language_selection_then_category_followup(self):
        response, state = handle_message("u2", "bengali")
        self.assertEqual(state, "collecting_profile")
        self.assertIn("1", response["response"])
        user = user_model.get_user("u2")
        self.assertEqual(user.get("last_question_field"), "category")

    def test_free_speech_scholarship_asks_related_followup(self):
        handle_message("u3", "english")
        response, state = handle_message(
            "u3",
            "I am a student having 25 age and from Karnataka give me a list of scholarships",
        )
        self.assertEqual(state, "collecting_profile")
        lowered = response["response"].lower()
        self.assertTrue("percentage" in lowered or "income" in lowered)

    def test_language_change_command(self):
        handle_message("u4", "english")
        response, state = handle_message("u4", "change language")
        self.assertEqual(state, "awaiting_language")
        self.assertIn("choose your preferred language", response["response"].lower())

    def test_help_command(self):
        handle_message("u5", "english")
        response, state = handle_message("u5", "help")
        self.assertEqual(state, "active")
        self.assertIn("help", response["response"].lower())

    def test_query_between_followups_gets_answer(self):
        handle_message("u6", "english")
        response, state = handle_message("u6", "What documents are needed for scholarship schemes?")
        self.assertIn(state, {"active", "showing_schemes", "collecting_profile"})
        self.assertTrue(len(response.get("response", "")) > 10)


if __name__ == "__main__":
    unittest.main()

