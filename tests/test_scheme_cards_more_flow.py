import unittest
from unittest.mock import patch

from engine.state_manager import handle_message
from models.users import user_model


def _scheme(name: str, state: str = "Gujarat", category: str = "Housing") -> dict:
    return {
        "scheme_id": name,
        "scheme_name": name,
        "state": state,
        "category": category,
        "description": "Housing support for eligible families.",
        "eligibility": {},
        "benefits_summary": "Housing support",
        "documents_required": ["ID"],
        "application_link": "https://example.org",
        "why_match": ["Matched by profile"],
        "eligible": True,
        "score": 0.9,
    }


class TestSchemeCardsMoreFlow(unittest.TestCase):
    def setUp(self):
        user_model.memory_store.clear()
        self._tx_to_en = patch("engine.state_manager.translate_to_english", side_effect=lambda text, source_lang: text)
        self._tx_from_en = patch("engine.state_manager.translate_from_english", side_effect=lambda text, target_lang: text)
        self._tx_from_en_meta = patch(
            "engine.state_manager.translate_from_english_with_meta",
            side_effect=lambda text, target_lang: (text, {"translation_failed": False}),
        )
        self._next_q = patch(
            "engine.state_manager.decide_next_question_llm",
            return_value={"next_field": None, "question_english": None, "reason": "test"},
        )
        self._extract = patch(
            "engine.state_manager.extract_profile_llm",
            return_value={
                "intent": "profile_update",
                "confidence": 0.9,
                "category": None,
                "state": None,
                "age": None,
                "income": None,
                "occupation": None,
                "language": "gu",
            },
        )
        self._tx_to_en.start()
        self._tx_from_en.start()
        self._tx_from_en_meta.start()
        self._next_q.start()
        self._extract.start()

    def tearDown(self):
        self._tx_to_en.stop()
        self._tx_from_en.stop()
        self._tx_from_en_meta.stop()
        self._next_q.stop()
        self._extract.stop()

    @patch("engine.state_manager._query_schemes_direct", return_value=[])
    @patch("engine.state_manager.recommend_schemes")
    def test_no_intro_when_schemes_empty(self, mock_recommend, _mock_direct):
        mock_recommend.return_value = {
            "schemes": [],
            "fallback_used": True,
            "fallback_message": "No matching housing schemes found for Gujarat. Showing All India schemes if available.",
            "errors": [],
        }
        phone = "t_no_intro_empty"
        user_model.update_user(
            phone,
            {
                "language": "en",
                "conv_state": "collecting_profile",
                "last_question_field": "annual_income",
                "profile": {"category": "housing", "state": "Gujarat", "age": 45, "language": "en"},
            },
        )
        response, state = handle_message(phone, "300000")
        self.assertEqual(state, "showing_schemes")
        self.assertEqual(response.get("schemes"), [])
        self.assertIn("No matching housing schemes found for Gujarat. Showing All India schemes if available.", response.get("response", ""))
        self.assertNotIn("તમારા માટે મળતી યોજનાઓ", response.get("response", ""))

    @patch("engine.state_manager._query_schemes_direct", return_value=[])
    @patch("engine.state_manager.recommend_schemes")
    def test_more_returns_next_batch(self, mock_recommend, _mock_direct):
        mock_recommend.return_value = {
            "schemes": [_scheme("A"), _scheme("B"), _scheme("C"), _scheme("D")],
            "fallback_used": False,
            "fallback_message": None,
            "errors": [],
        }
        phone = "t_more_next_batch"
        user_model.update_user(
            phone,
            {
                "language": "en",
                "conv_state": "showing_schemes",
                "last_schemes": [_scheme("A"), _scheme("B")],
                "last_schemes_cursor": 2,
                "profile": {"category": "housing", "state": "Gujarat", "age": 45, "annual_income": 300000, "language": "en"},
            },
        )
        response, state = handle_message(phone, "વધુ")
        self.assertEqual(state, "showing_schemes")
        self.assertEqual(len(response.get("schemes") or []), 2)
        names = [s.get("scheme_name") for s in response.get("schemes") or []]
        self.assertEqual(names, ["C", "D"])

    @patch("engine.state_manager._query_schemes_direct", return_value=[])
    @patch("engine.state_manager.recommend_schemes")
    def test_more_returns_no_more_when_exhausted(self, mock_recommend, _mock_direct):
        mock_recommend.return_value = {
            "schemes": [_scheme("A"), _scheme("B")],
            "fallback_used": False,
            "fallback_message": None,
            "errors": [],
        }
        phone = "t_more_exhausted"
        user_model.update_user(
            phone,
            {
                "language": "en",
                "conv_state": "showing_schemes",
                "last_schemes": [_scheme("A"), _scheme("B")],
                "last_schemes_cursor": 2,
                "profile": {"category": "housing", "state": "Gujarat", "age": 45, "annual_income": 300000, "language": "en"},
            },
        )
        response, state = handle_message(phone, "give me more")
        self.assertEqual(state, "showing_schemes")
        self.assertEqual(response.get("schemes"), [])
        self.assertIn("No more schemes found.", response.get("response", ""))


if __name__ == "__main__":
    unittest.main()
