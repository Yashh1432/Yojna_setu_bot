import unittest
from pathlib import Path
from unittest.mock import patch

from engine import orchestrator
from engine.engine import extract_profile_llm
from engine.state_manager import handle_message
from models.users import user_model


def _llm_payload(**kwargs):
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
        "confidence": 0.9,
    }
    base.update(kwargs)
    return base


class TestStateOutputFixes(unittest.TestCase):
    def setUp(self):
        user_model.memory_store.clear()
        self._next_q_patcher = patch(
            "engine.state_manager.decide_next_question_llm",
            return_value={"next_field": None, "question_english": None, "reason": "offline-test"},
        )
        self._next_q_patcher.start()
        self._lang_pick_patcher = patch("engine.state_manager.infer_language_selection_llm", side_effect=self._fake_language_pick)
        self._lang_pick_patcher.start()
        self._tx_to_en_patcher = patch("engine.state_manager.translate_to_english", side_effect=lambda text, source_lang: text)
        self._tx_to_en_patcher.start()
        self._tx_from_en_patcher = patch("engine.state_manager.translate_from_english", side_effect=lambda text, target_lang: text)
        self._tx_from_en_patcher.start()

    @staticmethod
    def _fake_language_pick(message: str):
        lowered = str(message).strip().lower()
        mapping = {
            "english": "en",
            "hindi": "hi",
            "bengali": "bn",
            "kannad": "kn",
            "ಕನ್ನಡ": "kn",
            "tamil": "ta",
            "urdu": "ur",
        }
        code = mapping.get(lowered)
        return {"selected_language": code, "confidence": 0.95 if code else 0.0}

    def tearDown(self):
        self._next_q_patcher.stop()
        self._lang_pick_patcher.stop()
        self._tx_to_en_patcher.stop()
        self._tx_from_en_patcher.stop()

    def test_static_semantic_mappings_removed_from_state_manager(self):
        state_manager_path = Path(__file__).resolve().parent.parent / "engine" / "state_manager.py"
        source = state_manager_path.read_text(encoding="utf-8")
        self.assertNotIn("OCCUPATION_HINTS", source)
        self.assertNotIn("STATE_TEXT_ALIASES", source)
        self.assertNotIn("CATEGORY_REQUIRED_FIELDS", source)
        self.assertNotIn("_heuristic_extract_profile", source)

    @patch("engine.engine.router.generate_json")
    def test_khedutt_profile_extraction_via_llm(self, mock_generate_json):
        mock_generate_json.return_value = {
            "language": "gu",
            "intent": "profile_update",
            "occupation": "farmer",
            "category": "agriculture",
            "age": None,
            "income": None,
            "state": None,
            "gender": None,
            "caste_category": None,
            "academic_percentage": None,
            "bpl_status": None,
            "confidence": 0.93,
        }
        out = extract_profile_llm("khedutt")
        self.assertEqual(out.get("occupation"), "farmer")
        self.assertEqual(out.get("category"), "agriculture")

    @patch("engine.state_manager.extract_profile_llm")
    def test_language_typo_kannad_is_accepted(self, mock_extract):
        mock_extract.return_value = _llm_payload(language=None, intent="unknown")

        phone = "t_lang_typo_kannad"
        response, state = handle_message(phone, "kannad")

        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user.get("language"), "kn")
        self.assertEqual(user.get("last_question_field"), "category")

    @patch("engine.state_manager.extract_profile_llm")
    def test_language_typo_gujrathi_is_accepted(self, mock_extract):
        mock_extract.return_value = _llm_payload(language=None, intent="unknown")

        phone = "t_lang_typo_gujrathi"
        response, state = handle_message(phone, "gujrathi")

        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user.get("language"), "gu")
        self.assertEqual(user.get("last_question_field"), "category")

    @patch("engine.state_manager.extract_profile_llm")
    def test_language_script_kannada_is_accepted(self, mock_extract):
        mock_extract.return_value = _llm_payload(language=None, intent="unknown")

        phone = "t_lang_script_kannada"
        response, state = handle_message(phone, "ಕನ್ನಡ")

        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user.get("language"), "kn")
        self.assertEqual(user.get("last_question_field"), "category")

    @patch("engine.state_manager.extract_profile_llm")
    def test_repeated_invalid_language_shows_hint(self, mock_extract):
        mock_extract.return_value = _llm_payload(language=None, intent="unknown")

        phone = "t_invalid_lang_hint"
        response1, state1 = handle_message(phone, "zzz")
        response2, state2 = handle_message(phone, "zzz")

        self.assertEqual(state1, "awaiting_language")
        self.assertEqual(state2, "awaiting_language")
        self.assertIn("please choose your preferred language", response1.get("response", "").lower())
        self.assertIn("please choose your preferred language", response2.get("response", "").lower())

    @patch("engine.state_manager.decide_next_question_llm")
    @patch("engine.state_manager.extract_profile_llm")
    def test_numeric_field_priority_blocks_menu_conflict(self, mock_extract, mock_next_q):
        def fake_extract(message: str, **kwargs):
            lowered = message.strip().lower()
            if lowered == "student":
                return _llm_payload(intent="profile_update", occupation="student", category="education", state="Karnataka")
            if lowered == "2":
                return _llm_payload(intent="change_language")
            return _llm_payload()

        mock_extract.side_effect = fake_extract
        mock_next_q.return_value = {
            "next_field": "academic_percentage",
            "question_english": "What is your latest academic percentage?",
            "reason": "Need academic percentage for education flows.",
        }

        phone = "t_numeric_priority"
        handle_message(phone, "hindi")
        handle_message(phone, "student")
        response, state = handle_message(phone, "2")

        user = user_model.get_user(phone)
        profile = user.get("profile", {})
        self.assertEqual(user.get("language"), "hi")
        self.assertEqual(profile.get("annual_income"), 200000)
        self.assertNotEqual(state, "awaiting_language")
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.extract_profile_llm")
    def test_numeric_income_answer_uses_lakh_rule_not_menu(self, mock_extract):
        def fake_extract(message: str, **kwargs):
            lowered = message.strip().lower()
            if lowered == "education":
                return _llm_payload(intent="profile_update", category="education")
            if lowered == "gujarat":
                return _llm_payload(intent="profile_update", state="Gujarat")
            return _llm_payload()

        mock_extract.side_effect = fake_extract

        phone = "t_income_numeric_rule"
        handle_message(phone, "english")
        handle_message(phone, "education")
        handle_message(phone, "gujarat")
        response, state = handle_message(phone, "2")

        user = user_model.get_user(phone)
        profile = user.get("profile", {})
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(profile.get("annual_income"), 200000)
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.extract_profile_llm")
    def test_category_fuzzy_phonetic_aarogyaa_infers_health(self, mock_extract):
        # Force LLM to not provide category; Python should infer from fuzzy keyword.
        mock_extract.return_value = _llm_payload(intent="profile_update", category=None)

        phone = "t_cat_fuzzy_aarogyaa"
        handle_message(phone, "english")

        response, state = handle_message(phone, "aarogyaa")
        user = user_model.get_user(phone)
        profile = user.get("profile", {})

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(profile.get("category"), "health")
        self.assertEqual(user.get("last_question_field"), "state")
        self.assertTrue(response.get("response", ""))

    @patch("engine.state_manager.extract_profile_llm")
    def test_explicit_commands_work_during_profile_collection(self, mock_extract):
        def fake_extract(message: str, **kwargs):
            lowered = message.strip().lower()
            if lowered == "hindi":
                return _llm_payload(language="hi", intent="unknown")
            return _llm_payload()

        mock_extract.side_effect = fake_extract

        phone = "t_explicit_cmd_stage"
        handle_message(phone, "hindi")

        response1, state1 = handle_message(phone, "reset")
        self.assertEqual(state1, "awaiting_language")
        self.assertIn("choose your preferred language", response1.get("response", "").lower())

        handle_message(phone, "hindi")
        response2, state2 = handle_message(phone, "change language")
        self.assertEqual(state2, "awaiting_language")
        self.assertIn("choose your preferred language", response2.get("response", "").lower())

        handle_message(phone, "hindi")
        response3, state3 = handle_message(phone, "help")
        self.assertEqual(state3, "active")
        self.assertIn("help", response3.get("response", "").lower())

        handle_message(phone, "hindi")
        response4, state4 = handle_message(phone, "check eligibility")
        self.assertIn(state4, {"collecting_profile", "showing_schemes"})
        self.assertTrue(response4.get("response", ""))

    @patch("engine.state_manager.extract_profile_llm")
    def test_digit_menu_does_not_override_profile_collection(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="unknown")
        phone = "t_digit_not_menu_collecting"
        handle_message(phone, "kannad")
        response, state = handle_message(phone, "2")
        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user.get("last_question_field"), "category")
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.extract_profile_llm")
    def test_non_menu_numeric_when_category_expected_reasks(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="unknown")
        phone = "t_numeric_occ_reask"
        handle_message(phone, "hindi")
        response, state = handle_message(phone, "22")
        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual(user.get("last_question_field"), "category")
        self.assertTrue(response.get("response", ""))

    def test_explicit_reset_text_always_works_mid_flow(self):
        phone = "t_explicit_reset_mid_flow"
        handle_message(phone, "english")
        response, state = handle_message(phone, "reset")
        self.assertEqual(state, "awaiting_language")
        self.assertIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.decide_next_question_llm")
    @patch("engine.state_manager.extract_profile_llm")
    def test_interruption_answers_then_resumes_expected_field(self, mock_extract, mock_next_q):
        def fake_extract(message: str, **kwargs):
            lowered = message.strip().lower()
            if lowered == "student":
                return _llm_payload(intent="profile_update", occupation="student", category="education", state="Karnataka")
            if lowered == "what is scholarship?":
                return _llm_payload(
                    intent="general_query",
                    answer_english="A scholarship is financial help given to students for education.",
                )
            return _llm_payload()

        mock_extract.side_effect = fake_extract
        mock_next_q.return_value = {
            "next_field": "income",
            "question_english": "Now, what is your annual income?",
            "reason": "Need income to continue eligibility checks.",
        }

        phone = "t_interrupt_resume"
        handle_message(phone, "english")
        response1, state1 = handle_message(phone, "student")
        self.assertEqual(state1, "collecting_profile")
        self.assertIn("income", response1.get("response", "").lower())

        response2, state2 = handle_message(phone, "What is scholarship?")
        text = response2.get("response", "").lower()
        self.assertEqual(state2, "collecting_profile")
        self.assertIn("scholarship is financial help", text)
        self.assertIn("income", text)

    @patch("engine.state_manager.extract_profile_llm")
    def test_expected_occupation_input_forces_progress_when_llm_returns_null(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="unknown", occupation=None)

        phone = "t_occ_expected_force_progress"
        user_model.update_user(
            phone,
            {
                "language": "en",
                "conv_state": "collecting_profile",
                "last_question_field": "occupation",
                "profile": {
                    "category": "education",
                    "state": "Gujarat",
                    "age": 25,
                    "annual_income": 200000,
                    "income": 200000,
                },
            },
        )

        response, state = handle_message(phone, "student")
        user = user_model.get_user(phone)
        profile = user.get("profile", {})

        self.assertEqual(state, "collecting_profile")
        self.assertEqual(profile.get("occupation"), "student")
        self.assertNotEqual(user.get("last_question_field"), "occupation")
        self.assertNotIn("what is your occupation", response.get("response", "").lower())

    @patch("engine.state_manager.decide_next_question_llm")
    @patch("engine.state_manager.extract_profile_llm")
    def test_general_query_health_branch_translates_answer_and_followup_in_gujarati(self, mock_extract, mock_next_q):
        answer_en = "For health-related government schemes in India, please provide your state and income details for accurate information."
        question_en = "What is your state?"
        answer_gu = "ભારતમાં આરોગ્ય સંબંધિત સરકારી યોજનાઓ માટે, કૃપા કરીને ચોક્કસ માહિતી માટે તમારું રાજ્ય અને આવક જણાવો."
        question_gu = "તમે કયા રાજ્યમાં રહો છો?"

        def fake_translate(text: str, lang: str):
            if lang != "gu":
                return text
            mapping = {
                answer_en: answer_gu,
                question_en: question_gu,
                "1 Reset\n2 Change Language\n3 Help": "1 રીસેટ\n2 ભાષા બદલો\n3 મદદ",
            }
            return mapping.get(text, text)

        mock_extract.return_value = _llm_payload(
            intent="general_query",
            category="health",
            answer_english=answer_en,
        )
        mock_next_q.return_value = {
            "next_field": "state",
            "question_english": question_en,
            "reason": "Need state for health schemes.",
        }

        phone = "t_gu_health_translate"
        with patch("engine.state_manager.translate_from_english", side_effect=fake_translate) as tx_mock:
            handle_message(phone, "gujarati")
            response, state = handle_message(phone, "આરોગ્ય")

        text = response.get("response", "")
        self.assertEqual(state, "collecting_profile")
        self.assertNotIn("For health-related", text)
        self.assertNotIn("What is your state?", text)
        self.assertIn(answer_gu, text)
        self.assertIn(question_gu, text)
        self.assertTrue(any(call.args == (answer_en, "gu") for call in tx_mock.mock_calls))

    @patch("engine.state_manager.extract_profile_llm")
    def test_hindi_health_followup_has_no_english_leak(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="profile_update", category="health")

        def fake_translate(text: str, lang: str):
            mapping = {
                ("Language updated to Hindi.\n\nWhich type of scheme do you need (education, agriculture, health, employment, housing, finance, women)?", "hi"): "भाषा Hindi में अपडेट की गई.\n\nआपको किस प्रकार की योजना चाहिए?",
                ("Which state do you live in?", "hi"): "आप किस राज्य में रहते हैं?",
                ("1 Reset\n2 Change Language\n3 Help", "hi"): "1 रीसेट\n2 भाषा बदलें\n3 सहायता",
            }
            return mapping.get((text, lang), text)

        phone = "t_hi_health_followup"
        with patch("engine.state_manager.translate_from_english", side_effect=fake_translate):
            handle_message(phone, "hindi")
            response, state = handle_message(phone, "health")

        self.assertEqual(state, "collecting_profile")
        text = response.get("response", "")
        self.assertNotIn("What is your state?", text)
        self.assertIn("राज्य", text)

    @patch("engine.state_manager.extract_profile_llm")
    def test_kannada_health_followup_has_no_english_leak(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="profile_update", category="health")

        def fake_translate(text: str, lang: str):
            mapping = {
                ("Language updated to Kannada.\n\nWhich type of scheme do you need (education, agriculture, health, employment, housing, finance, women)?", "kn"): "ಭಾಷೆ Kannada ಗೆ ಸೆಟ್ ಮಾಡಲಾಗಿದೆ.\n\nನಿಮಗೆ ಯಾವ ಯೋಜನೆ ಬೇಕು?",
                ("Which state do you live in?", "kn"): "ನೀವು ಯಾವ ರಾಜ್ಯದಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
                ("1 Reset\n2 Change Language\n3 Help", "kn"): "1 ರೀಸೆಟ್\n2 ಭಾಷೆ ಬದಲಿಸಿ\n3 ಸಹಾಯ",
            }
            return mapping.get((text, lang), text)

        phone = "t_kn_health_followup"
        with patch("engine.state_manager.translate_from_english", side_effect=fake_translate):
            handle_message(phone, "kannad")
            response, state = handle_message(phone, "health")

        self.assertEqual(state, "collecting_profile")
        text = response.get("response", "")
        self.assertNotIn("What is your state?", text)
        self.assertIn("ರಾಜ್ಯ", text)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_gujarat_user_gets_only_gujarat_or_national(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Gujarat Farmer Support",
                "state": "Gujarat",
                "category": "agriculture",
                "description": "Gujarat agriculture support scheme.",
                "benefits": "Support for farmers in Gujarat.",
                "documents_required": ["ID", "Land Record"],
                "eligibility": {},
            },
            {
                "scheme_name": "Jharkhand Farmer Support",
                "state": "Jharkhand",
                "category": "agriculture",
                "description": "Jharkhand agriculture support scheme.",
                "benefits": "Support for farmers in Jharkhand.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Farmer Support",
                "state": "All India",
                "category": "agriculture",
                "description": "All India support for farmers.",
                "benefits": "Pan-India support for farmers.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
        ]

        profile = {"occupation": "farmer", "category": "agriculture", "state": "Gujarat"}
        result = orchestrator.recommend_schemes(profile, query="farmer schemes", top_k=10)
        schemes = result["schemes"]
        names = {s["scheme_name"] for s in schemes}
        states = {str(s["state"]).strip().lower() for s in schemes}

        self.assertIn("Gujarat Farmer Support", names)
        self.assertIn("National Farmer Support", names)
        self.assertNotIn("Jharkhand Farmer Support", names)
        self.assertTrue(states.issubset({"gujarat", "all india"}))

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_senior_category_hard_filter_blocks_unrelated_schemes(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Karnataka Senior Pension",
                "state": "Karnataka",
                "category": "Senior Citizen",
                "description": "Monthly pension support for senior citizens.",
                "benefits": "Old age pension for eligible seniors.",
                "documents_required": ["ID", "Age Proof"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Senior Pension",
                "state": "All India",
                "category": "Senior Citizen",
                "description": "National pension support for elderly citizens.",
                "benefits": "Pension for senior citizens across India.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "Cancer Care Mission",
                "state": "All India",
                "category": "Health",
                "description": "Cancer treatment assistance.",
                "benefits": "Financial aid for cancer patients.",
                "documents_required": ["Medical Documents"],
                "eligibility": {},
            },
            {
                "scheme_name": "Disability Grant",
                "state": "All India",
                "category": "Disability",
                "description": "Disability support grant.",
                "benefits": "Monthly assistance for disabled citizens.",
                "documents_required": ["Disability Certificate"],
                "eligibility": {},
            },
        ]

        profile = {"category": "senior citizen", "state": "Karnataka", "annual_income": 300000}
        result = orchestrator.recommend_schemes(profile, query="senior citizen pension", top_k=10)
        names = {s["scheme_name"] for s in result["schemes"]}

        self.assertIn("Karnataka Senior Pension", names)
        self.assertIn("National Senior Pension", names)
        self.assertNotIn("Cancer Care Mission", names)
        self.assertNotIn("Disability Grant", names)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_no_category_match_returns_no_relevant_message(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Cancer Care Mission",
                "state": "All India",
                "category": "Health",
                "description": "Cancer treatment assistance.",
                "benefits": "Financial aid for cancer patients.",
                "documents_required": ["Medical Documents"],
                "eligibility": {},
            }
        ]

        profile = {"category": "senior citizen", "state": "Karnataka", "annual_income": 300000}
        result = orchestrator.recommend_schemes(profile, query="senior citizen pension", top_k=10)
        self.assertEqual(result.get("schemes"), [])
        self.assertIn("could not find relevant", str(result.get("fallback_message", "")).lower())

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_hindi_senior_profile_rejects_unrelated_national_schemes(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "National Senior Pension",
                "state": "All India",
                "category": "Senior Citizen",
                "description": "Old age pension support for senior citizens.",
                "benefits": "Monthly pension for elderly citizens.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Cancer Aid",
                "state": "All India",
                "category": "Health",
                "description": "Cancer treatment support scheme.",
                "benefits": "Aid for cancer patients.",
                "documents_required": ["Medical Records"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Disability Grant",
                "state": "All India",
                "category": "Disability",
                "description": "Support for disabled citizens.",
                "benefits": "Monthly disability assistance.",
                "documents_required": ["Disability Certificate"],
                "eligibility": {},
            },
            {
                "scheme_name": "Ex-Servicemen Welfare",
                "state": "All India",
                "category": "Others",
                "description": "Support for ex-servicemen.",
                "benefits": "Benefits for armed forces veterans.",
                "documents_required": ["Service ID"],
                "eligibility": {},
            },
        ]

        profile = {"category": "वरिष्ठ नागरिक", "state": "Karnataka", "annual_income": 300000}
        result = orchestrator.recommend_schemes(profile, query="वरिष्ठ नागरिक पेंशन", top_k=10)
        names = {s["scheme_name"] for s in result["schemes"]}

        self.assertIn("National Senior Pension", names)
        self.assertNotIn("National Cancer Aid", names)
        self.assertNotIn("National Disability Grant", names)
        self.assertNotIn("Ex-Servicemen Welfare", names)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_gujarati_health_profile_rejects_business_and_funeral(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "National Health Protection",
                "state": "All India",
                "category": "Health",
                "description": "Healthcare support for families.",
                "benefits": "Hospital and treatment assistance.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Startup Loan",
                "state": "All India",
                "category": "Financial Assistance",
                "description": "Startup and MSME business financing.",
                "benefits": "Low interest business loan support.",
                "documents_required": ["Business Proof"],
                "eligibility": {},
            },
            {
                "scheme_name": "Funeral Support Grant",
                "state": "All India",
                "category": "Others",
                "description": "Financial assistance for funeral rites.",
                "benefits": "Support for death and funeral expenses.",
                "documents_required": ["Death Certificate"],
                "eligibility": {},
            },
        ]

        profile = {"category": "આરોગ્ય", "state": "Gujarat", "annual_income": 200000}
        result = orchestrator.recommend_schemes(profile, query="આરોગ્ય યોજના", top_k=10)
        names = {s["scheme_name"] for s in result["schemes"]}

        self.assertIn("National Health Protection", names)
        self.assertNotIn("National Startup Loan", names)
        self.assertNotIn("Funeral Support Grant", names)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_marathi_business_profile_rejects_funeral_category(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "National MSME Subsidy",
                "state": "All India",
                "category": "Financial Assistance",
                "description": "Business subsidy support for MSMEs.",
                "benefits": "Loan and subsidy support for entrepreneurs.",
                "documents_required": ["ID", "Business Details"],
                "eligibility": {},
            },
            {
                "scheme_name": "Antim Sanskar Sahayata",
                "state": "All India",
                "category": "Others",
                "description": "Support for funeral and death rites.",
                "benefits": "Financial support for antim sanskar.",
                "documents_required": ["Death Certificate"],
                "eligibility": {},
            },
        ]

        profile = {"category": "व्यवसाय", "state": "Maharashtra", "annual_income": 200000}
        result = orchestrator.recommend_schemes(profile, query="व्यवसाय कर्ज योजना", top_k=10)
        names = {s["scheme_name"] for s in result["schemes"]}

        self.assertIn("National MSME Subsidy", names)
        self.assertNotIn("Antim Sanskar Sahayata", names)

    @patch("engine.state_manager.recommend_schemes")
    @patch("engine.state_manager.extract_profile_llm")
    def test_maharastra_input_normalizes_to_maharashtra(self, mock_extract, mock_recommend):
        mock_extract.return_value = _llm_payload(
            intent="scheme_search",
            occupation="farmer",
            category="agriculture",
            age=50,
            income=300000,
            state="Maharashtra",
            confidence=0.95,
        )

        def fake_recommend(profile, query=None, top_k=5):
            self.assertEqual(profile.get("state"), "Maharashtra")
            return {
                "schemes": [
                    {
                        "scheme_name": "Maharashtra Kisan Support",
                        "state": "Maharashtra",
                        "benefits_summary": "State support",
                        "why_match": ["Available in Maharashtra"],
                        "documents_required": ["ID"],
                        "application_link": None,
                    },
                    {
                        "scheme_name": "National Kisan Support",
                        "state": "All India",
                        "benefits_summary": "National support",
                        "why_match": ["National / All India scheme"],
                        "documents_required": ["ID"],
                        "application_link": None,
                    },
                ],
                "fallback_used": False,
                "fallback_message": None,
            }

        mock_recommend.side_effect = fake_recommend
        phone = "t_maharastra_norm"
        handle_message(phone, "english")
        response, _ = handle_message(phone, "kisan hu 50 saal ka hun income 3 lakh from maharastra")

        user = user_model.get_user(phone)
        self.assertEqual(user.get("profile", {}).get("state"), "Maharashtra")
        states = {str(s.get("state", "")).lower() for s in response.get("schemes", [])}
        self.assertTrue(states.issubset({"maharashtra", "all india"}))

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_reason_uses_actual_scheme_state(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "National Health Plan",
                "state": "All India",
                "category": "health",
                "description": "Health support",
                "benefits": "Health coverage",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "Gujarat Health Plan",
                "state": "Gujarat",
                "category": "health",
                "description": "Gujarat health support",
                "benefits": "State health coverage",
                "documents_required": ["ID"],
                "eligibility": {},
            },
        ]

        profile = {"occupation": "worker", "category": "health", "state": "Gujarat"}
        result = orchestrator.recommend_schemes(profile, query="health scheme", top_k=10)
        reasons_by_name = {s["scheme_name"]: " | ".join(s.get("why_match") or []) for s in result["schemes"]}
        all_reasons = " ".join(reasons_by_name.values())

        self.assertIn("national/All India scheme", reasons_by_name.get("National Health Plan", ""))
        self.assertIn("available in Gujarat", reasons_by_name.get("Gujarat Health Plan", ""))
        self.assertNotIn("Available for Gujarat", all_reasons)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_benefits_summary_capped_to_150_chars(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Long Benefit Scheme",
                "state": "All India",
                "category": "finance",
                "description": "d" * 500,
                "benefits": "b" * 500,
                "documents_required": ["ID"],
                "eligibility": {},
            }
        ]

        profile = {"occupation": "farmer", "category": "finance", "state": "Gujarat"}
        result = orchestrator.recommend_schemes(profile, query="loan support", top_k=5)
        self.assertTrue(result["schemes"])
        summary = result["schemes"][0].get("benefits_summary", "")
        self.assertLessEqual(len(summary), 150)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_national_fallback_when_no_state_match(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Jharkhand Crop Support",
                "state": "Jharkhand",
                "category": "agriculture",
                "description": "Jharkhand-only scheme.",
                "benefits": "State-only support.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Crop Support",
                "state": "All India",
                "category": "agriculture",
                "description": "National scheme.",
                "benefits": "National support.",
                "documents_required": ["ID"],
                "eligibility": {},
            },
        ]

        profile = {"occupation": "farmer", "category": "agriculture", "state": "Gujarat"}
        result = orchestrator.recommend_schemes(profile, query="crop", top_k=10)
        names = {s["scheme_name"] for s in result["schemes"]}

        self.assertTrue(result["fallback_used"])
        self.assertEqual(
            result["fallback_message"],
            "No exact Gujarat schemes found. Showing national/All India schemes only.",
        )
        self.assertIn("National Crop Support", names)
        self.assertNotIn("Jharkhand Crop Support", names)

    def test_frontend_uses_response_and_card_separation(self):
        script_path = Path(__file__).resolve().parent.parent / "frontend" / "script.js"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn('const responseText = (data.response || "").replace(/\\n/g, "<br>");', script)
        self.assertIn('const stateLabel = (s.state && s.state.trim()) ? s.state.trim() : "All India";', script)
        self.assertIn('State-level Schemes', script)
        self.assertIn('National-level Schemes', script)
        self.assertNotIn("State: Not specified", script)

    def test_api_contract_keeps_response_and_schemes_separate(self):
        routes_path = Path(__file__).resolve().parent.parent / "api" / "routes.py"
        routes = routes_path.read_text(encoding="utf-8")

        self.assertIn('response_text = out.get("response") or out.get("message") or ""', routes)
        self.assertIn('out.pop("message", None)', routes)
        self.assertIn('if not isinstance(out.get("schemes"), list):', routes)

    @patch("engine.state_manager.recommend_schemes")
    @patch("engine.state_manager.extract_profile_llm")
    def test_complete_profile_never_returns_empty_response(self, mock_extract, mock_recommend):
        mock_extract.return_value = _llm_payload(
            intent="scheme_search",
            occupation="farmer",
            category="agriculture",
            age=45,
            income=200000,
            state="Karnataka",
            confidence=0.95,
        )
        mock_recommend.return_value = {
            "schemes": [
                {
                    "scheme_name": "Karnataka Farmer Relief",
                    "state": "Karnataka",
                    "benefits_summary": "Income support for eligible farmers.",
                    "why_match": ["Available in Karnataka"],
                    "documents_required": ["ID"],
                    "application_link": "https://example.org/apply",
                }
            ],
            "fallback_used": False,
            "fallback_message": None,
        }

        phone = "t_full_profile_non_empty"
        handle_message(phone, "english")
        response, _ = handle_message(phone, "farmer from Karnataka income 2 lakh age 45")
        self.assertTrue(response.get("response", "").strip())
        self.assertTrue(response.get("schemes"))

    @patch("engine.state_manager.extract_profile_llm")
    def test_tamil_free_speech_skips_onboarding_prompt(self, mock_extract):
        mock_extract.return_value = _llm_payload(
            language="ta",
            intent="scheme_search",
            occupation="farmer",
            category="agriculture",
            age=45,
            income=200000,
            state="Karnataka",
            confidence=0.95,
        )
        phone = "t_tamil_bootstrap"
        response, state = handle_message(phone, "நான் விவசாயி, வயது 45, வருமானம் 2 லட்சம், கர்நாடகா")
        user = user_model.get_user(phone)
        self.assertEqual(user.get("language"), "ta")
        self.assertNotEqual(state, "awaiting_language")
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.extract_profile_llm")
    def test_urdu_free_speech_skips_onboarding_prompt(self, mock_extract):
        mock_extract.return_value = _llm_payload(
            language="ur",
            intent="scheme_search",
            occupation="farmer",
            category="agriculture",
            age=40,
            income=200000,
            state="Karnataka",
            confidence=0.95,
        )
        phone = "t_urdu_bootstrap"
        response, state = handle_message(phone, "میں کسان ہوں، عمر 40، انکم 2 لاکھ، کرناٹک سے")
        user = user_model.get_user(phone)
        self.assertEqual(user.get("language"), "ur")
        self.assertNotEqual(state, "awaiting_language")
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.state_manager.recommend_schemes")
    @patch("engine.state_manager.extract_profile_llm")
    def test_response_text_never_contains_raw_scheme_dump(self, mock_extract, mock_recommend):
        mock_extract.return_value = _llm_payload(
            intent="scheme_search",
            occupation="farmer",
            category="agriculture",
            age=45,
            income=200000,
            state="Karnataka",
            confidence=0.95,
        )
        long_benefit = "x" * 500
        mock_recommend.return_value = {
            "schemes": [
                {
                    "scheme_name": "Any Scheme",
                    "state": "All India",
                    "benefits_summary": long_benefit,
                    "why_match": ["National / All India scheme"],
                    "documents_required": ["ID"],
                    "application_link": "https://example.org",
                }
            ],
            "fallback_used": False,
            "fallback_message": None,
        }
        phone = "t_no_raw_dump"
        handle_message(phone, "english")
        response, _ = handle_message(phone, "show schemes")
        self.assertLess(len(response.get("response", "")), 300)
        self.assertTrue(response.get("schemes"))

    @patch("engine.state_manager.extract_profile_llm")
    def test_language_persists_after_profile_turn(self, mock_extract):
        def fake_extract(message: str, **kwargs):
            lowered = message.strip().lower()
            if lowered == "farmer":
                return _llm_payload(intent="profile_update", occupation="farmer", category="agriculture")
            return _llm_payload()

        mock_extract.side_effect = fake_extract
        phone = "t_language_persist"
        handle_message(phone, "hindi")
        handle_message(phone, "farmer")
        user = user_model.get_user(phone)
        self.assertEqual(user.get("language"), "hi")

    @patch("engine.state_manager.translate_from_english")
    @patch("engine.state_manager.extract_profile_llm")
    @patch("engine.state_manager.infer_language_selection_llm")
    def test_followup_text_passes_translation_layer(self, mock_lang_pick, mock_extract, mock_translate):
        mock_lang_pick.return_value = {"selected_language": "ur", "confidence": 0.95}
        mock_extract.return_value = _llm_payload(intent="unknown")

        def fake_translate(text: str, target_lang: str):
            if target_lang == "ur":
                return f"[ur]{text}"
            return text

        mock_translate.side_effect = fake_translate

        response, _ = handle_message("t_translation_path", "urdu")
        self.assertIn("[ur]", response.get("response", ""))

    @patch("engine.state_manager.extract_profile_llm")
    def test_vidhwa_infers_women_category_without_redundant_category_question(self, mock_extract):
        mock_extract.return_value = _llm_payload(intent="unknown")
        phone = "t_vidhwa_category"
        handle_message(phone, "english")
        response, state = handle_message(phone, "vidhwa")
        user = user_model.get_user(phone)
        self.assertEqual(state, "collecting_profile")
        self.assertEqual((user.get("profile") or {}).get("category"), "women")
        self.assertNotIn("which type of scheme do you need", response.get("response", "").lower())


if __name__ == "__main__":
    unittest.main()
