import unittest
from unittest.mock import patch

from engine.llm_router import LLMRouter


class TestLLMRouterFallback(unittest.TestCase):
    def setUp(self):
        self.router = LLMRouter()

    def test_generate_json_returns_deterministic_fallback_when_all_tiers_fail(self):
        with (
            patch.object(self.router, "_get_cache", return_value=None),
            patch.object(self.router, "call_gemini", return_value=None),
            patch.object(self.router, "call_sarvam", return_value=None),
        ):
            self.router._last_gemini_error = "gemini_missing"
            self.router._last_sarvam_error = "sarvam_missing"
            out = self.router.generate_json("prompt", "sys")

        self.assertIsInstance(out, dict)
        self.assertTrue(out.get("_fallback"))
        self.assertGreater(len(str(out.get("reason") or "")), 0)

    def test_generate_text_returns_non_empty_fallback_when_all_tiers_fail(self):
        with (
            patch.object(self.router, "_get_cache", return_value=None),
            patch.object(self.router, "call_gemini", return_value=None),
            patch.object(self.router, "call_sarvam", return_value=None),
        ):
            self.router._last_gemini_error = "gemini_missing"
            self.router._last_sarvam_error = "sarvam_missing"
            out = self.router.generate_text("prompt", "sys")

        self.assertIsInstance(out, str)
        self.assertGreater(len(out.strip()), 0)

    def test_generate_text_uses_sarvam_when_gemini_fails(self):
        with (
            patch.object(self.router, "_get_cache", return_value=None),
            patch.object(self.router, "call_gemini", return_value=None),
            patch.object(self.router, "call_sarvam", return_value="sarvam response"),
            patch.object(self.router, "_set_cache"),
        ):
            out = self.router.generate_text("prompt", "sys")

        self.assertEqual(out, "sarvam response")


if __name__ == "__main__":
    unittest.main()
