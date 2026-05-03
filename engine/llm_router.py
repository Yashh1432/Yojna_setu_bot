import os
import json
import re
import logging
import hashlib
import time
from datetime import datetime
from dotenv import load_dotenv

# Upgraded Gemini SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Sarvam AI Support
try:
    from sarvamai import SarvamAI
    HAS_SARVAM = True
except ImportError:
    HAS_SARVAM = False

from core.logger import get_logger

load_dotenv(override=True)

logger = get_logger("engine.llm_router")


class LLMRouter:
    def __init__(self):
        # ── Tier 1: Gemini API (PRIMARY with key rotation) ──────
        self.gemini_keys: list[str] = []
        self.gemini_active = False
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._key_index = 0
        self._key_cooldown: dict[int, float] = {}  # {key_index: cooldown_until_monotonic}
        self._key_usage: dict[int, dict] = {}       # {key_index: {"count": int, "window_start": float}}
        self._rpm_limit = int(os.getenv("GEMINI_RPM_LIMIT", "14"))  # per-key requests per minute

        if HAS_GENAI:
            keys_raw = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
            self.gemini_keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
            if self.gemini_keys:
                self.gemini_active = True
                logger.info(f"LLMRouter: Gemini (V2 SDK) configured with {len(self.gemini_keys)} API key(s).")
            else:
                logger.warning("LLMRouter: No Gemini API keys found.")
        else:
            logger.warning("LLMRouter: google-genai package not found.")

        # ── Tier 2: Sarvam AI (Secondary Fallback) ──────────────
        self.sarvam_active = False
        sarvam_key = os.getenv("SARVAM_API_KEY")
        if HAS_SARVAM and sarvam_key:
            try:
                self.sarvam_client = SarvamAI(api_subscription_key=sarvam_key)
                self.sarvam_model = os.getenv("SARVAM_MODEL", "sarvam-m")
                self.sarvam_active = True
                logger.info(f"LLMRouter: Sarvam AI configured as fallback (model: {self.sarvam_model}).")
            except Exception as e:
                logger.warning(f"LLMRouter: Sarvam config error: {e}")

        self._ensure_cache_ttl_index()

    def _ensure_cache_ttl_index(self):
        """Create TTL indexes for legacy and new cache collections."""
        try:
            from models.db_client import db_client
            if db_client.db is not None:
                db_client.db["ai_response_cache"].create_index(
                    "created_at",
                    expireAfterSeconds=86400,  # 24 hours
                    background=True
                )
                # New cache collection used by services.cache_service.
                db_client.db["llm_cache"].create_index(
                    "expires_at",
                    expireAfterSeconds=0,
                    background=True,
                )
        except Exception as e:
            logger.warning(f"LLMRouter: Could not create TTL index: {e}")

    # ─────────────────────────────────────────────────────────────
    # KEY ROTATION ENGINE
    # ─────────────────────────────────────────────────────────────

    def _next_gemini_key(self) -> tuple[int | None, str | None]:
        """Round-robin key selection with cooldown + pre-rotation."""
        if not self.gemini_keys:
            return None, None

        now = time.monotonic()
        tried = 0
        while tried < len(self.gemini_keys):
            idx = self._key_index % len(self.gemini_keys)
            self._key_index += 1
            tried += 1

            # Skip if in cooldown
            if self._key_cooldown.get(idx, 0) > now:
                continue

            # Pre-rotation: skip if approaching RPM limit
            usage = self._key_usage.get(idx, {"count": 0, "window_start": now})
            if now - usage["window_start"] < 60:
                if usage["count"] >= self._rpm_limit:
                    logger.debug(f"LLMRouter: Key #{idx} pre-rotated (RPM limit {self._rpm_limit})")
                    continue
            else:
                # Reset window
                self._key_usage[idx] = {"count": 0, "window_start": now}

            return idx, self.gemini_keys[idx]

        return None, None

    def _record_usage(self, idx: int) -> None:
        """Track successful call for pre-rotation."""
        now = time.monotonic()
        usage = self._key_usage.get(idx, {"count": 0, "window_start": now})
        if now - usage["window_start"] >= 60:
            usage = {"count": 0, "window_start": now}
        usage["count"] += 1
        self._key_usage[idx] = usage

    def _mark_cooldown(self, idx: int, duration: float = 60.0) -> None:
        """Mark a key as rate-limited for `duration` seconds."""
        self._key_cooldown[idx] = time.monotonic() + duration
        logger.info(f"LLMRouter: Gemini key #{idx} rate-limited, cooldown {duration}s")

    # ─────────────────────────────────────────────────────────────
    # TIER 1: Gemini API — PRIMARY (with key rotation)
    # ─────────────────────────────────────────────────────────────

    def call_gemini(self, prompt: str, sys_prompt: str) -> str | None:
        """Tier 1: Gemini API with round-robin key rotation (V2 SDK)."""
        if not self.gemini_active:
            return None

        # Safety settings for the new SDK
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]

        for attempt in range(len(self.gemini_keys)):
            idx, key = self._next_gemini_key()
            if key is None:
                logger.warning("LLMRouter: All Gemini keys exhausted/rate-limited.")
                break

            try:
                client = genai.Client(api_key=key)
                res = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        temperature=0.3,
                        max_output_tokens=1024,
                        safety_settings=safety_settings
                    )
                )
                if res and res.text:
                    self._record_usage(idx)
                    logger.debug(f"LLMRouter: Gemini responded (key #{idx}).")
                    return res.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    self._mark_cooldown(idx, duration=60.0)
                else:
                    logger.warning(f"LLMRouter: Gemini error (key #{idx}): {e}")
                    self._mark_cooldown(idx, duration=10.0)
                continue

        return None

    # ─────────────────────────────────────────────────────────────
    # TIER 2: Sarvam AI — FALLBACK
    # ─────────────────────────────────────────────────────────────

    def call_sarvam(self, prompt: str, sys_prompt: str) -> str | None:
        """Tier 2: Sarvam AI fallback."""
        if not self.sarvam_active:
            return None
        try:
            # Fixed Sarvam call: completions() instead of completions.create()
            res = self.sarvam_client.chat.completions(
                model=self.sarvam_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            # Response is a CreateChatCompletionResponse object
            text = res.choices[0].message.content.strip()
            if text:
                logger.debug(f"LLMRouter: Sarvam AI ({self.sarvam_model}) responded.")
                return text
        except Exception as e:
            logger.warning(f"LLMRouter: Sarvam error: {e}")
        return None

    # ─────────────────────────────────────────────────────────────
    # CACHE: MongoDB with TTL
    # ─────────────────────────────────────────────────────────────

    def _get_cache(self, prompt: str, sys_prompt: str) -> str | None:
        try:
            from models.llm_cache import llm_cache_model
            hash_key = hashlib.sha256(f"{sys_prompt}\n\n{prompt}".encode()).hexdigest()
            return llm_cache_model.get_cache(hash_key)
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        return None

    def _set_cache(self, prompt: str, sys_prompt: str, response: str, model_used: str = "unknown", latency: int = 0):
        if not response:
            return
        try:
            from models.llm_cache import llm_cache_model
            hash_key = hashlib.sha256(f"{sys_prompt}\n\n{prompt}".encode()).hexdigest()
            llm_cache_model.set_cache(
                query_hash=hash_key,
                input_str=prompt,
                response=response,
                model_name=model_used,
                latency_ms=latency
            )
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def generate_text(self, prompt: str, sys_prompt: str = "") -> str | None:
        cached = self._get_cache(prompt, sys_prompt)
        if cached:
            return cached

        start_time = time.monotonic()
        model_used = "unknown"

        res = self.call_gemini(prompt, sys_prompt)
        if res:
            model_used = f"gemini:{self.gemini_model_name}"
        else:
            res = self.call_sarvam(prompt, sys_prompt)
            if res:
                model_used = f"sarvam:{self.sarvam_model}"

        if not res:
            logger.error("LLMRouter: All tiers failed (generate_text).")
            return None

        latency_ms = int((time.monotonic() - start_time) * 1000)
        res = self._strip_thinking(res)
        self._set_cache(prompt, sys_prompt, res, model_used=model_used, latency=latency_ms)
        return res

    def generate_json(self, prompt: str, sys_prompt: str = "") -> dict | None:
        cached = self._get_cache(prompt, sys_prompt)
        if cached:
            return self.scrub_to_json(cached)

        start_time = time.monotonic()
        model_used = "unknown"

        res = self.call_gemini(prompt, sys_prompt)
        if res:
            model_used = f"gemini:{self.gemini_model_name}"
        else:
            res = self.call_sarvam(prompt, sys_prompt)
            if res:
                model_used = f"sarvam:{self.sarvam_model}"

        if not res:
            logger.error("LLMRouter: All tiers failed (generate_json).")
            return None

        parsed = self.scrub_to_json(res)

        if parsed:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._set_cache(prompt, sys_prompt, res, model_used=model_used, latency=latency_ms)
            return parsed

        return None

    def scrub_to_json(self, text: str) -> dict | None:
        """Extract valid JSON from LLM output, stripping markdown fences and think blocks."""
        if not text:
            return None
        raw = self._strip_thinking(text)
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'(\{.*\})', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip <think>...</think> reasoning blocks from LLM output."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


router = LLMRouter()
