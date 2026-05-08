import unittest
from unittest.mock import patch

from engine.state_manager import MENU_TEXT, MENU_TEXTS, handle_message
from engine.responses import OPTIONS_MENU
from models.users import user_model


class TestMenuNumberRouting(unittest.TestCase):
    def setUp(self):
        user_model.memory_store.clear()

    def _seed_user(self, phone: str, *, language="en", conv_state="active", expected_field=None, profile=None):
        user_model.update_user(
            phone,
            {
                "language": language,
                "conv_state": conv_state,
                "last_question_field": expected_field,
                "profile": profile or {},
            },
        )

    def test_case_1_hindi_category_digit_2_starts_language_change(self):
        phone = "menu_case_1"
        self._seed_user(phone, language="hi", conv_state="collecting_profile", expected_field="category")

        response, state = handle_message(phone, "2")
        user = user_model.get_user(phone)

        self.assertEqual(state, "awaiting_language")
        self.assertIsNone(user.get("language"))
        self.assertIn("Please choose your preferred language", str(response.get("response") or ""))

    @patch("engine.state_manager._translate_to_english_safe", side_effect=lambda text, language: text)
    @patch("engine.state_manager.extract_profile_llm")
    def test_case_2_hindi_health_text_still_selects_health(self, mock_extract, _mock_translate):
        mock_extract.return_value = {
            "language": "hi",
            "intent": "profile_update",
            "scheme_name": None,
            "entities": {"age": None, "income": None, "gender": None, "occupation": None, "state": None},
            "category": "health",
            "age": None,
            "income": None,
            "occupation": None,
            "education_level": None,
            "state": None,
            "gender": None,
            "caste_category": None,
            "academic_percentage": None,
            "bpl_status": None,
            "answer_english": None,
            "confidence": 0.95,
        }

        phone = "menu_case_2"
        self._seed_user(phone, language="hi", conv_state="collecting_profile", expected_field="category")

        response, state = handle_message(phone, "स्वास्थ्य")
        user = user_model.get_user(phone)

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user["profile"]["category"], "health")
        self.assertEqual(user.get("last_question_field"), "state")
        self.assertNotIn("Please choose your preferred language", str(response.get("response") or ""))

    def test_case_3_age_digit_stays_numeric(self):
        phone = "menu_case_3"
        self._seed_user(phone, language="en", conv_state="collecting_profile", expected_field="age")

        _response, state = handle_message(phone, "2")
        user = user_model.get_user(phone)

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user["profile"]["age"], 2)
        self.assertNotEqual(user.get("conv_state"), "awaiting_language")

    def test_case_4_income_number_stays_numeric(self):
        phone = "menu_case_4"
        self._seed_user(phone, language="en", conv_state="collecting_profile", expected_field="annual_income")

        _response, state = handle_message(phone, "300000")
        user = user_model.get_user(phone)

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user["profile"]["annual_income"], 300000)
        self.assertEqual(user["profile"]["income"], 300000)

    def test_case_5_all_menus_show_only_1_2_3(self):
        self.assertEqual(MENU_TEXT, "1 Reset\n2 Change Language\n3 Help")
        for text in MENU_TEXTS.values():
            self.assertIn("1", text)
            self.assertIn("2", text)
            self.assertIn("3", text)
            self.assertNotIn("4", text)
            self.assertNotIn("Eligibility", text)
        for text in OPTIONS_MENU.values():
            self.assertEqual(text, "1 Reset | 2 Change Language | 3 Help")

    def test_case_6_digit_4_returns_invalid_menu_message(self):
        phone = "menu_case_6"
        self._seed_user(phone, language="hi", conv_state="collecting_profile", expected_field="category")

        response, state = handle_message(phone, "4")

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(response.get("intent"), "HELP")
        self.assertEqual(response.get("schemes"), [])
        self.assertNotIn("eligibility", str(response.get("response") or "").lower())
        self.assertNotIn("Please choose your preferred language", str(response.get("response") or ""))


if __name__ == "__main__":
    unittest.main()
