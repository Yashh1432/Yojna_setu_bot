import unittest
from unittest.mock import patch

from services import cache_service


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


class TestCacheSystem(unittest.TestCase):
    def setUp(self):
        self.col = _InMemoryCollection()
        self.col_patcher = patch("services.cache_service._get_collection", return_value=self.col)
        self.col_patcher.start()

    def tearDown(self):
        self.col_patcher.stop()

    def test_extraction_cache_hit_same_language_and_expected_field(self):
        payload = {"intent": "profile_update", "category": "health"}
        cache_service.set_extraction_cache("aarogya", payload, language="gu", expected_field="category")
        hit = cache_service.get_extraction_cache("aarogya", language="gu", expected_field="category")
        self.assertEqual(hit, payload)

    def test_extraction_cache_miss_different_expected_field(self):
        payload = {"intent": "profile_update", "category": "health"}
        cache_service.set_extraction_cache("aarogya", payload, language="gu", expected_field="category")
        miss = cache_service.get_extraction_cache("aarogya", language="gu", expected_field="state")
        self.assertIsNone(miss)

    def test_rag_cache_not_reused_for_different_state(self):
        schemes = [{"scheme_name": "Health A", "state": "Gujarat"}]
        cache_service.set_rag_cache("health", schemes, state="gujarat", language="gu")
        miss = cache_service.get_rag_cache("health", state="karnataka", language="gu")
        self.assertIsNone(miss)

    def test_response_cache_hit_same_profile_signature_and_language(self):
        profile = {"category": "health", "state": "Gujarat", "income": 200000, "age": 40}
        schemes = ["S1", "S2"]
        cache_service.set_response_cache(profile, schemes, "ગુજરાતી જવાબ", language="gu")
        hit = cache_service.get_response_cache(profile, schemes, language="gu")
        self.assertEqual(hit, "ગુજરાતી જવાબ")

    def test_response_cache_miss_when_profile_changes(self):
        profile_a = {"category": "health", "state": "Gujarat", "income": 200000, "age": 40}
        profile_b = {"category": "health", "state": "Gujarat", "income": 300000, "age": 40}
        schemes = ["S1", "S2"]
        cache_service.set_response_cache(profile_a, schemes, "ગુજરાતી જવાબ", language="gu")
        miss = cache_service.get_response_cache(profile_b, schemes, language="gu")
        self.assertIsNone(miss)

    def test_response_cache_not_shared_across_states(self):
        profile_gu = {"category": "agriculture", "state": "Gujarat", "income": 100000}
        profile_jh = {"category": "agriculture", "state": "Jharkhand", "income": 100000}
        schemes = ["Farmer Aid"]
        cache_service.set_response_cache(profile_gu, schemes, "only gujarat", language="en")
        miss = cache_service.get_response_cache(profile_jh, schemes, language="en")
        self.assertIsNone(miss)

    def test_response_cache_not_shared_across_languages(self):
        profile = {"category": "health", "state": "Gujarat", "income": 200000}
        schemes = ["Health Aid"]
        cache_service.set_response_cache(profile, schemes, "Gujarati text", language="gu")
        miss = cache_service.get_response_cache(profile, schemes, language="hi")
        self.assertIsNone(miss)

    def test_low_confidence_extraction_is_not_cached(self):
        payload = {"intent": "profile_update", "category": "health", "confidence": 0.2}
        cache_service.set_extraction_cache("medical help", payload, language="en", expected_field="category")
        miss = cache_service.get_extraction_cache("medical help", language="en", expected_field="category")
        self.assertIsNone(miss)

    def test_response_cache_stores_context_metadata(self):
        profile = {"category": "health", "state": "Gujarat", "income": 200000, "age": 40}
        schemes = ["Health Aid"]
        cache_service.set_response_cache(profile, schemes, "Health response", language="en", confidence=0.95)
        self.assertEqual(len(self.col.docs), 1)
        doc = next(iter(self.col.docs.values()))
        self.assertEqual(doc.get("category"), "health")
        self.assertEqual(doc.get("state"), "gujarat")
        self.assertEqual(doc.get("language"), "en")
        self.assertEqual(doc.get("confidence"), 0.95)
        self.assertTrue(doc.get("profile_hash"))
        self.assertIn("created_at", doc)
        self.assertIn("expires_at", doc)


if __name__ == "__main__":
    unittest.main()
