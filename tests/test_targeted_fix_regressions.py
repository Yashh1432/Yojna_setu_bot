import unittest
from unittest.mock import patch

from engine.engine import classify_user_intent_llm, infer_language_selection_llm
from engine.orchestrator import recommend_schemes
from engine.rag import retrieve_schemes
import engine.state_manager as state_manager
from engine.state_manager import handle_message
from engine.validator import infer_scheme_category, normalize_state_name
from models.users import user_model
from services import cache_service, translation_service


class _InMemoryCollection:
    def __init__(self):
        self.docs = {}

    def create_index(self, *args, **kwargs):
        return None

    def find_one(self, query):
        doc = self.docs.get(query.get("_id"))
        if not doc:
            return None
        if query.get("type") and doc.get("type") != query.get("type"):
            return None
        return dict(doc)

    def replace_one(self, query, doc, upsert=False):
        self.docs[query.get("_id")] = dict(doc)
        return None


class _FakeCursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def limit(self, _limit):
        return self

    def __iter__(self):
        return iter(self.docs)


class _RecordingCollection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def find(self, query, projection=None):
        self.calls.append({"query": query, "projection": projection})
        index = len(self.calls) - 1
        docs = self.responses[index] if index < len(self.responses) else []
        return _FakeCursor(docs)


class _FakeDB:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, _name):
        return self.collection


def _llm_payload(**kwargs):
    payload = {
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
    payload.update(kwargs)
    return payload


class TestTargetedFixRegressions(unittest.TestCase):
    def setUp(self):
        user_model.memory_store.clear()
        self.cache_collection = _InMemoryCollection()
        self.cache_patcher = patch("services.cache_service._get_collection", return_value=self.cache_collection)
        self.cache_patcher.start()

    def tearDown(self):
        self.cache_patcher.stop()

    @patch("engine.engine.router.generate_json", return_value=None)
    def test_language_typos_kannad_and_malyalam_still_work(self, _mock_generate_json):
        self.assertEqual(infer_language_selection_llm("kannad").get("selected_language"), "kn")
        self.assertEqual(infer_language_selection_llm("malyalam").get("selected_language"), "ml")

    @patch("engine.state_manager.translate_to_english", side_effect=lambda text, source_lang: text)
    @patch("engine.state_manager.translate_from_english", side_effect=lambda text, target_lang: text)
    @patch("engine.state_manager.translate_from_english_with_meta", side_effect=lambda text, target_lang: (text, {"translation_failed": False}))
    @patch("engine.state_manager.decide_next_question_llm", return_value={"next_field": "age", "question_english": "What is your age?", "reason": "need age"})
    @patch("engine.state_manager.extract_profile_llm", return_value=_llm_payload())
    def test_numeric_answer_two_is_treated_as_income_not_menu(self, *_mocks):
        phone = "t_numeric_income_only"
        user_model.update_user(
            phone,
            {
                "language": "en",
                "conv_state": "collecting_profile",
                "last_question_field": "annual_income",
                "profile": {"category": "education", "state": "Gujarat"},
            },
        )
        response, state = handle_message(phone, "2")
        profile = (user_model.get_user(phone) or {}).get("profile", {})
        self.assertEqual(profile.get("annual_income"), 200000)
        self.assertEqual(profile.get("income"), 200000)
        self.assertEqual(state, "collecting_profile")
        self.assertNotIn("choose your preferred language", response.get("response", "").lower())

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_gujarat_profile_returns_only_gujarat_or_all_india(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "Gujarat Farmer Support",
                "state": "Gujarat",
                "category": "agriculture",
                "description": "Gujarat scheme",
                "benefits": "Support for Gujarat farmers",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Farmer Support",
                "state": "All India",
                "category": "agriculture",
                "description": "National scheme",
                "benefits": "Support for Indian farmers",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "Kerala Farmer Support",
                "state": "Kerala",
                "category": "agriculture",
                "description": "Kerala scheme",
                "benefits": "Support for Kerala farmers",
                "documents_required": ["ID"],
                "eligibility": {},
            },
        ]
        result = recommend_schemes({"category": "agriculture", "state": "Gujarat"}, query="farmer support", top_k=5)
        states = {str(item.get("state")).strip().lower() for item in result.get("schemes", [])}
        self.assertTrue(states.issubset({"gujarat", "all india"}))

    @patch("engine.engine.router.generate_json")
    def test_compound_intent_keeps_primary_and_secondary_context(self, mock_generate_json):
        intent = classify_user_intent_llm("student disability scholarship")
        mock_generate_json.assert_not_called()
        self.assertEqual(intent.get("canonical_category"), "education")
        self.assertIn("disability", intent.get("intent_keywords") or [])
        self.assertEqual(intent.get("subcategory"), "scholarship")

    def test_low_confidence_or_empty_outputs_are_not_cached(self):
        cache_service.set_extraction_cache(
            "medical help",
            {"intent": "profile_update", "category": "health", "confidence": 0.1},
            language="en",
            expected_field="category",
        )
        self.assertIsNone(cache_service.get_extraction_cache("medical help", language="en", expected_field="category"))

        profile = {"category": "health", "state": "Gujarat", "income": 200000, "age": 40}
        cache_service.set_response_cache(profile, ["S1"], "", language="en", confidence=0.95)
        self.assertEqual(len(self.cache_collection.docs), 0)

        cache_service.set_response_cache(profile, ["S1"], "response", language="en", confidence=0.1)
        self.assertEqual(len(self.cache_collection.docs), 0)

    @patch("services.translation_service.logger.error")
    def test_translation_failure_is_logged(self, mock_error):
        translation_service._translator._cache.clear()
        with patch.object(translation_service._translator, "_load", return_value=False):
            text, meta = translation_service.translate_from_english_with_meta("Unique health support text", "hi")
        self.assertEqual(text, "Unique health support text")
        self.assertTrue(meta.get("translation_failed"))
        mock_error.assert_called()

    @patch("engine.state_manager.translate_to_english", side_effect=lambda text, source_lang: text)
    @patch("engine.state_manager.translate_from_english", side_effect=lambda text, target_lang: text)
    @patch("engine.state_manager.translate_from_english_with_meta", side_effect=lambda text, target_lang: (text, {"translation_failed": True}))
    def test_translation_failure_sets_internal_metadata_without_crashing_chat(self, *_mocks):
        phone = "t_translation_metadata"
        user_model.update_user(phone, {"language": "hi", "conv_state": "active", "last_question_field": None, "profile": {}})
        response, state = handle_message(phone, "help")
        self.assertEqual(state, "active")
        self.assertTrue(response.get("response"))
        self.assertTrue((response.get("internal_metadata") or {}).get("translation_failed"))

    def test_mongo_retrieval_uses_normalized_category_first(self):
        collection = _RecordingCollection(
            [[{"scheme_name": "Health Aid", "benefits": "Long enough benefits for quality", "state": "All India"}]]
        )
        with patch("engine.rag._get_db", return_value=_FakeDB(collection)):
            schemes = retrieve_schemes("medical help", language="en")
        self.assertEqual(len(collection.calls), 1)
        self.assertTrue(schemes)
        first_query = collection.calls[0]["query"]
        self.assertIn("normalized_category", str(first_query))

    def test_mongo_retrieval_falls_back_to_regex_second(self):
        collection = _RecordingCollection(
            [
                [],
                [{"scheme_name": "Health Aid", "benefits": "Long enough benefits for quality", "state": "All India"}],
            ]
        )
        with patch("engine.rag._get_db", return_value=_FakeDB(collection)):
            schemes = retrieve_schemes("health", language="en")
        self.assertEqual(len(collection.calls), 2)
        self.assertTrue(schemes)
        second_query = collection.calls[1]["query"]
        self.assertIn("$regex", str(second_query))

    def test_state_normalization_supports_up_variants_and_awash_housing(self):
        self.assertEqual(normalize_state_name("UP"), "uttar pradesh")
        self.assertEqual(normalize_state_name("Uttar Pradesh"), "uttar pradesh")
        self.assertEqual(normalize_state_name("उत्तर प्रदेश"), "uttar pradesh")
        self.assertEqual(infer_scheme_category("awash support"), "housing")

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_up_housing_rejects_other_states_and_keeps_only_up_or_all_india(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "AP Housing",
                "state": "Andhra Pradesh",
                "category": "housing",
                "description": "AP housing support",
                "benefits": "AP only housing benefit for urban poor families",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "MP Housing",
                "state": "Madhya Pradesh",
                "category": "housing",
                "description": "MP housing support",
                "benefits": "MP only housing benefit for low income families",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "UP Housing",
                "state": "Uttar Pradesh",
                "category": "housing",
                "description": "UP housing support",
                "benefits": "UP housing benefit for eligible applicants",
                "documents_required": ["ID"],
                "eligibility": {},
            },
            {
                "scheme_name": "National Housing",
                "state": "All India",
                "category": "housing",
                "description": "National housing support",
                "benefits": "Housing support across India",
                "documents_required": ["ID"],
                "eligibility": {},
            },
        ]
        result = recommend_schemes(
            {"category": "housing", "state": "UP", "language": "hi", "age": 45, "income": 300000},
            query="आवास योजना",
            top_k=5,
        )
        states = {str(item.get("state")).strip().lower() for item in result.get("schemes", [])}
        self.assertTrue(states.issubset({"uttar pradesh", "all india"}))
        names = {str(item.get("scheme_name")).strip() for item in result.get("schemes", [])}
        self.assertNotIn("AP Housing", names)
        self.assertNotIn("MP Housing", names)
        reasons = " ".join(" ".join(item.get("why_match") or []) for item in result.get("schemes", []))
        self.assertNotIn("All India level", reasons)

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_up_housing_no_exact_state_uses_hindi_national_fallback_message(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "National Housing",
                "state": "All India",
                "category": "housing",
                "description": "National housing support",
                "benefits": "Housing support across India",
                "documents_required": ["ID"],
                "eligibility": {},
            }
        ]
        result = recommend_schemes(
            {"category": "housing", "state": "Uttar Pradesh", "language": "hi", "age": 45, "income": 300000},
            query="आवास",
            top_k=5,
        )
        self.assertTrue(result.get("schemes"))
        self.assertEqual(
            result.get("fallback_message"),
            "उत्तर प्रदेश के लिए कोई सटीक आवास योजना नहीं मिली। उपलब्ध राष्ट्रीय/All India योजनाएँ दिखा रहा हूँ।",
        )

    @patch("engine.orchestrator.load_scheme_dataset")
    def test_up_housing_no_state_or_national_returns_empty_with_hindi_message(self, mock_dataset):
        mock_dataset.return_value = [
            {
                "scheme_name": "MP Housing",
                "state": "Madhya Pradesh",
                "category": "housing",
                "description": "MP housing support",
                "benefits": "MP only housing benefit for low income families",
                "documents_required": ["ID"],
                "eligibility": {},
            }
        ]
        result = recommend_schemes(
            {"category": "housing", "state": "Uttar Pradesh", "language": "hi", "age": 45, "income": 300000},
            query="आवास",
            top_k=5,
        )
        self.assertEqual(result.get("schemes"), [])
        self.assertEqual(
            result.get("fallback_message"),
            "उत्तर प्रदेश के लिए कोई उपयुक्त आवास योजना नहीं मिली। कृपया दूसरी श्रेणी आज़माएँ या अधिक जानकारी दें।",
        )

    @patch("engine.orchestrator.filter_schemes")
    @patch("engine.orchestrator.load_scheme_dataset")
    def test_final_geo_safety_filter_rejects_cross_state_leak(self, mock_dataset, mock_filter):
        mock_dataset.return_value = [
            {
                "scheme_name": "UP Housing",
                "state": "Uttar Pradesh",
                "category": "housing",
                "description": "UP housing support",
                "benefits": "UP housing benefit for eligible applicants",
                "documents_required": ["ID"],
                "eligibility": {},
            }
        ]
        mock_filter.return_value = {
            "eligible": [
                {
                    "scheme_name": "MP Housing Leak",
                    "state": "Madhya Pradesh",
                    "category": "housing",
                    "description": "Leak",
                    "benefits": "Leak benefit that must be geo-rejected",
                    "documents_required": ["ID"],
                    "eligibility": {},
                    "score": 0.9,
                    "why_match": ["Leak candidate"],
                }
            ],
            "uncertain_needs_more_data": [],
            "ineligible": [],
            "errors": [],
        }
        result = recommend_schemes(
            {"category": "housing", "state": "Uttar Pradesh", "language": "hi"},
            query="housing",
            top_k=5,
        )
        self.assertEqual(result.get("schemes"), [])
        self.assertEqual(result.get("geo_rejected_count"), 1)

    @patch("engine.state_manager.logger.info")
    def test_final_output_geo_filter_blocks_cross_state_cards(self, mock_info):
        schemes = [
            {
                "scheme_name": "Karnataka Employment",
                "state": "Karnataka",
                "category": "employment",
                "description": "State employment support in Karnataka",
                "benefits_summary": "Employment help for Karnataka residents",
                "documents_required": ["ID"],
                "application_link": "https://example.org/karnataka",
                "eligibility": {},
            },
            {
                "scheme_name": "Goa Employment",
                "state": "Goa",
                "category": "employment",
                "description": "State employment support in Goa",
                "benefits_summary": "Employment help for Goa residents",
                "documents_required": ["ID"],
                "application_link": "https://example.org/goa",
                "eligibility": {},
            },
            {
                "scheme_name": "Arunachal Employment",
                "state": "Arunachal Pradesh",
                "category": "employment",
                "description": "State employment support in Arunachal Pradesh",
                "benefits_summary": "Employment help for Arunachal residents",
                "documents_required": ["ID"],
                "application_link": "https://example.org/arunachal",
                "eligibility": {},
            },
            {
                "scheme_name": "National Employment",
                "state": "All India",
                "category": "employment",
                "description": "National employment support",
                "benefits_summary": "Employment help across India",
                "documents_required": ["ID"],
                "application_link": "https://example.org/national",
                "eligibility": {},
            },
        ]
        cards = state_manager._normalize_scheme_cards(
            schemes,
            profile={"state": "Karnataka", "category": "employment", "language": "en"},
            max_cards=None,
        )
        states = {str(card.get("state") or "").strip().lower() for card in cards}
        self.assertTrue(states.issubset({"karnataka", "all india"}))
        names = {str(card.get("scheme_name") or "").strip() for card in cards}
        self.assertNotIn("Goa Employment", names)
        self.assertNotIn("Arunachal Employment", names)
        self.assertIn("Karnataka Employment", names)
        self.assertIn("National Employment", names)

        logged_payloads = [args[0] for args, _kwargs in mock_info.call_args_list if args and isinstance(args[0], dict)]
        geo_logs = [payload for payload in logged_payloads if payload.get("event") == "final_geo_filter_applied"]
        self.assertTrue(geo_logs)
        self.assertEqual(geo_logs[-1].get("geo_rejected_count"), 2)


if __name__ == "__main__":
    unittest.main()
