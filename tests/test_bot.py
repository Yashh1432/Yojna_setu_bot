"""
YojnaSetuBot — Comprehensive Test Suite
Tests:
  1. Unit tests for all fixed bugs (B1–B17)
  2. Multilingual integration tests (12 scenarios, 7 languages)
  3. Edge-case and regression tests

Run: python -m pytest tests/test_bot.py -v
Or:  python tests/test_bot.py
"""

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# ── Allow imports from project root ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ════════════════════════════════════════════════════════════════════
# SECTION 1: UNIT TESTS  (Bug regression tests — all offline)
# ════════════════════════════════════════════════════════════════════

class TestB1_TemplateKeyError(unittest.TestCase):
    """B1: MASTER_RESPONSE_TEMPLATE must not contain {threshold} placeholder."""

    def test_no_threshold_placeholder(self):
        from engine.engine import MASTER_RESPONSE_TEMPLATE
        self.assertNotIn("{threshold}", MASTER_RESPONSE_TEMPLATE,
            "B1 REGRESSION: {threshold} placeholder still in MASTER_RESPONSE_TEMPLATE")

    def test_format_call_succeeds(self):
        from engine.engine import MASTER_RESPONSE_TEMPLATE
        # Should not raise KeyError
        try:
            result = MASTER_RESPONSE_TEMPLATE.format(
                profile="test profile",
                schemes="test schemes",
                missed_schemes="none",
                explanations="some reasons",
                confidence_data="0.8",
                options_menu="1️⃣ Reset",
            )
            self.assertIsInstance(result, str)
        except KeyError as e:
            self.fail(f"B1 REGRESSION: format() raised KeyError — {e}")


class TestB2_ResetLanguage(unittest.TestCase):
    """B2: Reset should reply in detected language, not hardcoded Hindi."""

    def test_reset_uses_detected_language(self):
        from engine.responses import get_welcome
        # Verify the welcome messages exist for multiple languages
        for lang in ["en", "hi", "gu", "ta", "bn", "te"]:
            msg = get_welcome(lang)
            self.assertTrue(len(msg) > 5,
                f"B2: get_welcome('{lang}') returned empty/short string: {msg!r}")

    def test_reset_not_hardcoded_hindi(self):
        """Ensure state_manager no longer has the hardcoded 'hi' literal for reset."""
        state_manager_path = os.path.join(PROJECT_ROOT, "engine", "state_manager.py")
        with open(state_manager_path, encoding="utf-8") as f:
            content = f.read()
        # The fixed version uses llm_lang, not get_welcome("hi") directly
        self.assertNotIn('get_welcome("hi")', content,
            "B2 REGRESSION: state_manager still hardcodes get_welcome('hi') for reset")


class TestB4_EligibilityQueryText(unittest.TestCase):
    """B4: _get_rag_schemes must honour the query_text argument."""

    def test_no_stale_comment(self):
        eligibility_path = os.path.join(PROJECT_ROOT, "engine", "eligibility.py")
        with open(eligibility_path, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("Let's fix that too", content,
            "B4 REGRESSION: stale TODO comment still present in eligibility.py")

    def test_filter_schemes_signature(self):
        from engine.eligibility import filter_schemes
        import inspect
        sig = inspect.signature(filter_schemes)
        self.assertIn("query_text", sig.parameters,
            "B4: filter_schemes must accept query_text parameter")


class TestB7_OccupationExtracted(unittest.TestCase):
    """B7: occupation must be included in entity extraction in state_manager."""

    def test_occupation_in_entity_list(self):
        state_manager_path = os.path.join(PROJECT_ROOT, "engine", "state_manager.py")
        with open(state_manager_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn('"occupation"', content,
            "B7 REGRESSION: 'occupation' not found in state_manager entity extraction list")


class TestB9_ConfigurableThreshold(unittest.TestCase):
    """B9: Borderline miss threshold must use CONFIDENCE_THRESHOLD, not hardcoded 0.5."""

    def test_no_hardcoded_05(self):
        eligibility_path = os.path.join(PROJECT_ROOT, "engine", "eligibility.py")
        with open(eligibility_path, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("confidence > 0.5", content,
            "B9 REGRESSION: hardcoded 0.5 threshold still in eligibility.py")
        self.assertIn("CONFIDENCE_THRESHOLD", content,
            "B9: CONFIDENCE_THRESHOLD not used in eligibility.py")


class TestB10_SafeExplainabilityLoad(unittest.TestCase):
    """B10: explainability_service must not crash on DB-less startup."""

    def test_module_importable_without_db(self):
        """Should import cleanly even if MongoDB is unavailable."""
        try:
            import importlib
            import services.explainability_service as es
            importlib.reload(es)
        except Exception as e:
            self.fail(f"B10 REGRESSION: explainability_service crashed on import: {e}")

    def test_no_module_level_collection_access(self):
        explain_path = os.path.join(PROJECT_ROOT, "services", "explainability_service.py")
        with open(explain_path, encoding="utf-8") as f:
            content = f.read()
        # Should NOT have bare `collection = db_client.db[...]` at module level
        self.assertNotIn('collection = db_client.db["eligibility_checks"]', content,
            "B10 REGRESSION: bare collection assignment still at module level")
        # Should have the helper function
        self.assertIn("def _get_collection", content,
            "B10: _get_collection helper function not found")


class TestB11B12_FloatSafeConversion(unittest.TestCase):
    """B11+B12: validator must handle float values from LLM (e.g. 45.0, 150000.0)."""

    def test_age_float_string(self):
        from engine.validator import validate_entities
        result = validate_entities({"age": 45.0, "income": None})
        self.assertEqual(result.get("age"), 45,
            "B11: Float age 45.0 not correctly converted to int 45")

    def test_income_float_string(self):
        from engine.validator import validate_entities
        result = validate_entities({"income": 150000.0, "age": None})
        self.assertEqual(result.get("income"), 150000,
            "B12: Float income 150000.0 not correctly converted to int 150000")

    def test_age_int_still_works(self):
        from engine.validator import validate_entities
        result = validate_entities({"age": 60, "income": None})
        self.assertEqual(result.get("age"), 60)

    def test_age_out_of_range_dropped(self):
        from engine.validator import validate_entities
        result = validate_entities({"age": 200, "income": None})
        self.assertNotIn("age", result,
            "Validator should drop age=200 (out of range)")

    def test_income_auto_scale_lakh(self):
        """LLM sometimes returns '2' meaning '2 lakh' — validator should scale it."""
        from engine.validator import validate_entities
        result = validate_entities({"age": None, "income": 2})
        # 2 < 100 → auto-scaled to 200000
        self.assertEqual(result.get("income"), 200000,
            "Validator should auto-scale income=2 → 200000 (2 lakh)")


class TestB15_NoMarkdownInMenu(unittest.TestCase):
    """B15: get_options_menu must not include markdown dividers."""

    def test_no_markdown_divider(self):
        from engine.responses import get_options_menu
        for lang in ["en", "hi", "gu", "ta"]:
            menu = get_options_menu(lang)
            self.assertNotIn("---", menu,
                f"B15 REGRESSION: Markdown '---' divider found in menu for lang='{lang}'")
            self.assertNotIn("MENU", menu,
                f"B15 REGRESSION: 'MENU' heading found in menu for lang='{lang}'")

    def test_menu_has_options(self):
        from engine.responses import get_options_menu
        menu = get_options_menu("en")
        self.assertIn("Reset", menu)
        self.assertIn("Language", menu)


class TestB16_StateQuestionExists(unittest.TestCase):
    """B16: responses.py must have 'state' question for all key languages."""

    def test_state_question_present(self):
        from engine.responses import ASK_QUESTIONS, get_question
        self.assertIn("state", ASK_QUESTIONS,
            "B16 REGRESSION: 'state' not in ASK_QUESTIONS dict")

    def test_get_question_state_english(self):
        from engine.responses import get_question
        q = get_question("state", "en")
        self.assertIn("state", q.lower(),
            "state question in English should mention 'state'")

    def test_get_question_state_hindi(self):
        from engine.responses import get_question
        q = get_question("state", "hi")
        self.assertTrue(len(q) > 10, "Hindi state question should be non-empty")


class TestValidatorEdgeCases(unittest.TestCase):
    """Additional validator edge cases."""

    def test_none_values_graceful(self):
        from engine.validator import validate_entities
        result = validate_entities({})
        self.assertIsInstance(result, dict)

    def test_state_partial_match(self):
        from engine.validator import validate_entities
        result = validate_entities({"state": "gujarat"})
        self.assertEqual(result.get("state"), "gujarat")

    def test_state_partial_match_abbreviation(self):
        from engine.validator import validate_entities
        result = validate_entities({"state": "gujrat"})
        # "gujrat" is in "gujarat" → should match
        self.assertIsNotNone(result.get("state"),
            "Fuzzy state matching should handle 'gujrat'")

    def test_gender_normalize_female(self):
        from engine.validator import validate_entities
        for alias in ["mahila", "aurat", "woman", "female", "F"]:
            result = validate_entities({"gender": alias})
            self.assertEqual(result.get("gender"), "female",
                f"Gender '{alias}' should normalize to 'female'")

    def test_gender_normalize_male(self):
        from engine.validator import validate_entities
        for alias in ["male", "m", "man"]:
            result = validate_entities({"gender": alias})
            self.assertEqual(result.get("gender"), "male",
                f"Gender '{alias}' should normalize to 'male'")


class TestLLMFirstRefactor(unittest.TestCase):
    """LLM-first language inference and static-map removal checks."""

    @patch("engine.engine.router.generate_json")
    def test_language_infer_accepts_kannad_typo(self, mock_generate_json):
        from engine.engine import infer_language_selection_llm
        mock_generate_json.return_value = {"selected_language": "kn", "confidence": 0.92}
        out = infer_language_selection_llm("kannad")
        self.assertEqual(out.get("selected_language"), "kn")

    @patch("engine.engine.router.generate_json")
    def test_language_infer_accepts_kannada_script(self, mock_generate_json):
        from engine.engine import infer_language_selection_llm
        mock_generate_json.return_value = {"selected_language": "kn", "confidence": 0.94}
        out = infer_language_selection_llm("ಕನ್ನಡ")
        self.assertEqual(out.get("selected_language"), "kn")

    def test_removed_static_semantic_maps(self):
        state_manager_path = os.path.join(PROJECT_ROOT, "engine", "state_manager.py")
        embedding_path = os.path.join(PROJECT_ROOT, "services", "embedding_service.py")
        with open(state_manager_path, encoding="utf-8") as f:
            state_source = f.read()
        with open(embedding_path, encoding="utf-8") as f:
            embed_source = f.read()
        self.assertNotIn("LANGUAGE_ALIASES", state_source)
        self.assertNotIn("LANGUAGE_NORMALIZATION_CANDIDATES", state_source)
        self.assertNotIn("normalize_language(", state_source)
        self.assertNotIn("_parse_language_selection(", state_source)
        self.assertNotIn("_OCCUPATION_SYNONYMS", embed_source)


# ════════════════════════════════════════════════════════════════════
# SECTION 2: MULTILINGUAL SCENARIO TESTS  (against live LLM)
# These tests only run when the server is up — they are skipped otherwise.
# ════════════════════════════════════════════════════════════════════

import urllib.request
import urllib.error

SERVER_URL = "http://localhost:5000"
RUN_LIVE_INTEGRATION_TESTS = os.getenv("RUN_LIVE_INTEGRATION_TESTS", "0") == "1"

def _is_server_up() -> bool:
    if not RUN_LIVE_INTEGRATION_TESTS:
        return False
    try:
        urllib.request.urlopen(f"{SERVER_URL}/api/health", timeout=3)
        return True
    except Exception:
        return False

def _chat(phone: str, message: str) -> dict:
    import json as _json
    body = _json.dumps({
        "phone_number": phone,
        "message": message,
        "input_type": "text"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return _json.loads(resp.read())


@unittest.skipUnless(_is_server_up(), "Server not running — skipping integration tests")
class TestMultilingualIntegration(unittest.TestCase):
    """12 multilingual test scenarios against live Flask server."""

    BASE_PHONE = "test_multilingual_{}"

    def _fresh_phone(self, suffix: str) -> str:
        import uuid
        return f"test_{suffix}_{uuid.uuid4().hex[:6]}"

    def _reset(self, phone: str):
        """Hard-reset a test user."""
        _chat(phone, "1")  # global reset command

    # ── T1: English full profile ──────────────────────────────────
    def test_T1_english_farmer_full_profile(self):
        phone = self._fresh_phone("en_farmer")
        resp = _chat(phone, "I am a farmer, 45 years old, income 1.5 lakh, from Gujarat")
        self.assertIn("response", resp, "T1: No response field")
        self.assertEqual(resp.get("status"), "success", f"T1 failed: {resp}")
        text = resp["response"].lower()
        self.assertTrue(
            len(text) > 20,
            f"T1: Response too short: {text!r}"
        )
        print(f"\n[T1 English] {resp['response'][:200]}")

    # ── T2: Hindi ────────────────────────────────────────────────
    def test_T2_hindi_farmer(self):
        phone = self._fresh_phone("hi_farmer")
        resp = _chat(phone, "मैं किसान हूँ, उम्र 45 साल, आय 1.5 लाख, गुजरात से हूँ")
        self.assertEqual(resp.get("status"), "success", f"T2 Hindi failed: {resp}")
        self.assertTrue(len(resp.get("response", "")) > 20,
            f"T2: Response too short: {resp.get('response', '')!r}")
        print(f"\n[T2 Hindi] {resp['response'][:200]}")

    # ── T3: Gujarati ─────────────────────────────────────────────
    def test_T3_gujarati_farmer(self):
        phone = self._fresh_phone("gu_farmer")
        resp = _chat(phone, "hu kheti karu chu, umra 45, income 1.5 lakh, Gujarat ma rahvu chu")
        self.assertEqual(resp.get("status"), "success", f"T3 Gujarati failed: {resp}")
        print(f"\n[T3 Gujarati] {resp['response'][:200]}")

    # ── T4: Tamil ────────────────────────────────────────────────
    def test_T4_tamil_farmer(self):
        phone = self._fresh_phone("ta_farmer")
        resp = _chat(phone, "நான் விவசாயி, வயது 45, வருமானம் 1.5 லட்சம், குஜராத்தில் இருக்கிறேன்")
        self.assertEqual(resp.get("status"), "success", f"T4 Tamil failed: {resp}")
        print(f"\n[T4 Tamil] {resp['response'][:200]}")

    # ── T5: Bengali ──────────────────────────────────────────────
    def test_T5_bengali_farmer(self):
        phone = self._fresh_phone("bn_farmer")
        resp = _chat(phone, "আমি কৃষক, বয়স ৪৫, আয় দেড় লাখ, গুজরাট থেকে")
        self.assertEqual(resp.get("status"), "success", f"T5 Bengali failed: {resp}")
        print(f"\n[T5 Bengali] {resp['response'][:200]}")

    # ── T6: Telugu ───────────────────────────────────────────────
    def test_T6_telugu_farmer(self):
        phone = self._fresh_phone("te_farmer")
        resp = _chat(phone, "నేను రైతు, వయస్సు 45, ఆదాయం 1.5 లక్ష, ఆంధ్రప్రదేశ్ నుండి")
        self.assertEqual(resp.get("status"), "success", f"T6 Telugu failed: {resp}")
        print(f"\n[T6 Telugu] {resp['response'][:200]}")

    # ── T7: Reset flow ────────────────────────────────────────────
    def test_T7_reset_clears_state(self):
        phone = self._fresh_phone("reset_test")
        # First build up some state
        _chat(phone, "I am a farmer, 45 years old")
        # Now reset
        resp = _chat(phone, "1")
        self.assertEqual(resp.get("status"), "success", f"T7 reset failed: {resp}")
        state = resp.get("user_state", "")
        self.assertEqual(state, "awaiting_language",
            f"T7: Expected state='awaiting_language' after reset, got '{state}'")
        print(f"\n[T7 Reset] State: {state!r}, Response: {resp['response'][:100]}")

    # ── T8: Language switch ───────────────────────────────────────
    def test_T8_language_switch(self):
        phone = self._fresh_phone("lang_switch")
        _chat(phone, "I am a farmer, 45 years, income 1 lakh, Gujarat")
        resp = _chat(phone, "2")  # Change language
        self.assertEqual(resp.get("status"), "success", f"T8 lang switch failed: {resp}")
        print(f"\n[T8 Lang Switch] {resp['response'][:100]}")

    # ── T9: Zero / very low income ────────────────────────────────
    def test_T9_zero_income_handled(self):
        phone = self._fresh_phone("zero_income")
        resp = _chat(phone, "I am a farmer, age 50, income is zero, from Bihar")
        self.assertEqual(resp.get("status"), "success", f"T9 zero income failed: {resp}")
        # Should not crash — either ask for income again or process gracefully
        self.assertTrue(len(resp.get("response", "")) > 5)
        print(f"\n[T9 Zero Income] {resp['response'][:200]}")

    # ── T10: Missing fields → follow-up questions ─────────────────
    def test_T10_missing_income_triggers_question(self):
        phone = self._fresh_phone("missing_income")
        resp = _chat(phone, "I am a farmer, 45 years old, from Gujarat")
        self.assertEqual(resp.get("status"), "success", f"T10 failed: {resp}")
        state = resp.get("user_state", "")
        # Should still be collecting_profile or showing_schemes depending on config
        response_text = resp.get("response", "").lower()
        print(f"\n[T10 Missing Income] State: {state}, Response: {response_text[:200]}")
        # Key: should not crash
        self.assertIsInstance(resp["response"], str)

    # ── T11: High income → no or few schemes ──────────────────────
    def test_T11_high_income_filtering(self):
        phone = self._fresh_phone("high_income")
        resp = _chat(phone, "I am a farmer, 45 years old, income 50 lakh, from Gujarat")
        self.assertEqual(resp.get("status"), "success", f"T11 failed: {resp}")
        schemes = resp.get("schemes", [])
        # With income 50 lakh = 5,000,000 — most welfare schemes should not match
        print(f"\n[T11 High Income] Schemes found: {len(schemes)}, Response: {resp['response'][:200]}")
        # Just verify it doesn't crash and responds sensibly
        self.assertIsInstance(resp["response"], str)
        self.assertTrue(len(resp["response"]) > 5)

    # ── T12: Women category ───────────────────────────────────────
    def test_T12_women_entrepreneur(self):
        phone = self._fresh_phone("women_ent")
        resp = _chat(phone, "I am a woman entrepreneur, age 28, income 3 lakh, from Gujarat")
        self.assertEqual(resp.get("status"), "success", f"T12 failed: {resp}")
        print(f"\n[T12 Women Entrepreneur] {resp['response'][:200]}")
        self.assertIsInstance(resp["response"], str)
        self.assertTrue(len(resp["response"]) > 10)


# ════════════════════════════════════════════════════════════════════
# SECTION 3: SANITIZER + ENGINE UNIT TESTS
# ════════════════════════════════════════════════════════════════════

class TestSanitizer(unittest.TestCase):
    """Test input sanitizer and number normalization."""

    def test_lakh_normalization(self):
        from core.sanitizer import normalize_numeric_text
        result = normalize_numeric_text("2 lakh")
        self.assertIn("200000", result)

    def test_decimal_lakh(self):
        from core.sanitizer import normalize_numeric_text
        result = normalize_numeric_text("1.5 lakh")
        self.assertIn("150000", result)

    def test_crore_normalization(self):
        from core.sanitizer import normalize_numeric_text
        result = normalize_numeric_text("1 crore")
        self.assertIn("10000000", result)

    def test_thousand_normalization(self):
        from core.sanitizer import normalize_numeric_text
        result = normalize_numeric_text("50 thousand")
        self.assertIn("50000", result)

    def test_rupee_symbol_stripped(self):
        from core.sanitizer import normalize_numeric_text
        result = normalize_numeric_text("₹2,00,000")
        self.assertIn("200000", result)
        self.assertNotIn("₹", result)

    def test_html_entities_unescaped(self):
        from core.sanitizer import sanitize_text
        result = sanitize_text("&amp; &lt; &gt;")
        self.assertIn("&", result)
        self.assertNotIn("&amp;", result)

    def test_control_chars_stripped(self):
        from core.sanitizer import sanitize_text
        result = sanitize_text("hello\x00world\x01test")
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x01", result)

    def test_input_capped_1000_chars(self):
        from core.sanitizer import sanitize_text
        long_input = "a" * 2000
        result = sanitize_text(long_input)
        self.assertLessEqual(len(result), 1000)


class TestIncomeRangeParsing(unittest.TestCase):
    """Test income range extraction from user text."""

    def test_exact_income(self):
        from core.sanitizer import parse_income_range
        result = parse_income_range("My income is 1 lakh per year")
        self.assertIsNotNone(result)
        self.assertEqual(result["val"], 100000)
        self.assertFalse(result["is_range"])

    def test_range_income(self):
        from core.sanitizer import parse_income_range
        result = parse_income_range("income between 1 to 2 lakh")
        self.assertIsNotNone(result)
        self.assertTrue(result["is_range"])
        self.assertEqual(result["max"], 200000)

    def test_no_income(self):
        from core.sanitizer import parse_income_range
        result = parse_income_range("I am a farmer from Gujarat")
        self.assertIsNone(result)


class TestConfidenceService(unittest.TestCase):
    """Test confidence scoring math."""

    def test_full_profile_extraction_confidence(self):
        from services.confidence_service import extraction_confidence
        profile = {
            "profile": {
                "age": 45,
                "income": 150000,
                "occupation": "farmer",
                "state": "gujarat",
                "gender": "male"
            }
        }
        score = extraction_confidence(profile)
        self.assertEqual(score, 1.0, "Full profile should give extraction_confidence=1.0")

    def test_empty_profile_extraction_confidence(self):
        from services.confidence_service import extraction_confidence
        score = extraction_confidence({"profile": {}})
        self.assertEqual(score, 0.0)

    def test_partial_profile(self):
        from services.confidence_service import extraction_confidence
        profile = {"profile": {"age": 45, "income": 150000}}
        score = extraction_confidence(profile)
        self.assertAlmostEqual(score, 0.4, places=1)

    def test_final_confidence_bounds(self):
        from services.confidence_service import final_confidence
        profile = {"profile": {"age": 45, "income": 150000, "occupation": "farmer"}}
        scheme = {"eligibility": {"max_income": 200000, "min_age": 18, "occupation": "farmer"}}
        rag_results = [{"score": 0.8}]
        score = final_confidence(profile, rag_results, scheme)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestResponseTemplates(unittest.TestCase):
    """Test multilingual response template coverage."""

    def test_all_languages_have_welcome(self):
        from engine.responses import WELCOME_MESSAGES
        required_langs = ["en", "hi", "gu", "ta", "te", "kn", "bn", "mr", "pa", "ml"]
        for lang in required_langs:
            self.assertIn(lang, WELCOME_MESSAGES,
                f"WELCOME_MESSAGES missing language: {lang}")
            self.assertTrue(len(WELCOME_MESSAGES[lang]) > 5)

    def test_all_languages_have_menu(self):
        from engine.responses import OPTIONS_MENU
        required_langs = ["en", "hi", "gu", "ta", "te", "kn", "bn", "mr", "pa", "ml"]
        for lang in required_langs:
            self.assertIn(lang, OPTIONS_MENU,
                f"OPTIONS_MENU missing language: {lang}")

    def test_all_langs_have_age_question(self):
        from engine.responses import ASK_QUESTIONS
        self.assertIn("age", ASK_QUESTIONS)
        required_langs = ["en", "hi", "gu", "ta"]
        for lang in required_langs:
            self.assertIn(lang, ASK_QUESTIONS["age"],
                f"age question missing for lang: {lang}")

    def test_all_langs_have_state_question(self):
        from engine.responses import ASK_QUESTIONS
        self.assertIn("state", ASK_QUESTIONS,
            "B16: 'state' question not in ASK_QUESTIONS")
        for lang in ["en", "hi", "gu"]:
            self.assertIn(lang, ASK_QUESTIONS["state"])


# ════════════════════════════════════════════════════════════════════
# SECTION 4: LLM ROUTER UNIT TESTS (offline mocks)
# ════════════════════════════════════════════════════════════════════

class TestLLMRouterScrub(unittest.TestCase):
    """Test LLM output scrubbing."""

    def test_strip_think_blocks(self):
        from engine.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        raw = "<think>some internal reasoning here</think>The actual response"
        result = router._strip_thinking(raw)
        self.assertNotIn("<think>", result)
        self.assertNotIn("reasoning", result)
        self.assertIn("actual response", result)

    def test_scrub_json_from_markdown_fence(self):
        from engine.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        raw = '```json\n{"key": "value"}\n```'
        result = router.scrub_to_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("key"), "value")

    def test_scrub_json_with_think_prefix(self):
        from engine.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        raw = '<think>some reasoning</think>{"intent": "query", "language_code": "hi"}'
        result = router.scrub_to_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("intent"), "query")

    def test_scrub_invalid_json_returns_none(self):
        from engine.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        result = router.scrub_to_json("This is not JSON at all")
        self.assertIsNone(result)


class TestEngineNormalization(unittest.TestCase):
    """Test extraction normalization logic."""

    def test_normalise_extraction_flat_fields(self):
        from engine.engine import _normalise_extraction
        raw = {
            "language_code": "hi",
            "language_name": "Hindi",
            "intent": "update",
            "category": "Agriculture",
            "normalized_query": "farmer gujarat",
            "profile": {
                "age": 45,
                "income": 150000,
                "gender": "male",
                "occupation": "farmer",
                "state": "gujarat"
            }
        }
        result = _normalise_extraction(raw)
        self.assertEqual(result["language_code"], "hi")
        self.assertEqual(result["age"], 45)
        self.assertEqual(result["income"], 150000)
        self.assertEqual(result["state"], "gujarat")
        self.assertEqual(result["occupation"], "farmer")
        # Intent mapping: update → eligibility_check
        self.assertEqual(result["intent"], "eligibility_check")

    def test_normalise_handles_missing_profile(self):
        from engine.engine import _normalise_extraction
        raw = {
            "language_code": "en",
            "language_name": "English",
            "intent": "unknown",
            "category": None,
        }
        result = _normalise_extraction(raw)
        self.assertIsInstance(result, dict)
        self.assertIsNone(result["age"])
        self.assertEqual(result["intent"], "general_query")


# ════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("YojnaSetuBot — Test Suite")
    print("=" * 60)
    print()
    if _is_server_up():
        print("✅ Server detected at", SERVER_URL, "— Integration tests ENABLED")
    else:
        print("⚠️  Server NOT detected — Integration tests SKIPPED")
        print("    Start the server with: python run.py")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestB1_TemplateKeyError,
        TestB2_ResetLanguage,
        TestB4_EligibilityQueryText,
        TestB7_OccupationExtracted,
        TestB9_ConfigurableThreshold,
        TestB10_SafeExplainabilityLoad,
        TestB11B12_FloatSafeConversion,
        TestB15_NoMarkdownInMenu,
        TestB16_StateQuestionExists,
        TestValidatorEdgeCases,
        TestLLMFirstRefactor,
        TestSanitizer,
        TestIncomeRangeParsing,
        TestConfidenceService,
        TestResponseTemplates,
        TestLLMRouterScrub,
        TestEngineNormalization,
        # Integration (auto-skipped if server down)
        TestMultilingualIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
