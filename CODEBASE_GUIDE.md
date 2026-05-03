# YojnaSetuBot — Detailed Codebase Manual

This manual provides a developer-level breakdown of the project architecture, including function-level details and internal logic flows.

---

## 📂 Project Root
- **`run.py`**: 
  - **Purpose**: Server entry point.
  - **Logic**: Loads `.env` via `load_dotenv()`. Configures a centralized `logging.FileHandler` for `app.log`. Initializes the Flask app, CORS, and Rate Limiter. Fixes the Windows duplicate-process bug by setting `use_reloader=False`.
- **`CODEBASE_GUIDE.md`**: This document.
- **`DIAGNOSTIC_REPORT.txt`**: A runtime snapshot of system health, versioning, and database connectivity.

---

## 📁 `api/` — Web Interface Layer
- **`routes.py`**:
  - **`chat()`**: Main conversational loop. Detects input type (JSON vs Multipart Voice). Converts speech to text if needed using `voice.py`. Routes the message to `state_manager.handle_message()`.
  - **`upload()`**: Document OCR gateway. Receives image files, saves them to a secure temp path, and triggers `ocr.py`. Automates profile updates based on extracted OCR text.
  - **Maintenance**: Periodically executes `cleanup_old_files()` to purge transient `.wav` and `.png` artifacts.

---

## 📁 `core/` — Infrastructure & Security
- **`sanitizer.py`**:
  - **`sanitize_text()`**: Strips HTML entities, control characters, and enforces a 1000-character hard cap on all user inputs.
  - **`normalize_numeric_text()`**: The core currency/number parser. Converts "2 lakh", "1.5L", "50 hazar", and "Rs 1,50,000" into standard integers (e.g., `150000`).
- **`limiter.py`**: Global rate-limiting configuration (e.g., 60 req/min for chat, 10 req/min for OCR) to protect cloud API budgets.
- **`cleanup.py`**: File system janitor. Checks timestamps of files in static directories and removes any older than 600 seconds.

---

## 📁 `engine/` — Pipeline Intelligence (The "Brain")
- **`state_manager.py`**:
  - **States**: `awaiting_language`, `collecting_profile`, `showing_schemes`.
  - **Logic**: Implements global command interrupts. If input is "1" or "reset", it clears the session (`language=None`, `profile={}`). If input is "2", it resets the language state.
  - **Flow**: Assembles the profile field-by-field and determines if it is "Complete" (Category + Age + Income) before surfacing schemes.
- **`engine.py`**:
  - **`parse_message_llm()`**: LLM Call #1. Extracts profile entities into JSON. Maps raw LLM intents (e.g., `update`) to system-level intents (e.g., `eligibility_check`).
  - **`generate_final_response()`**: LLM Call #2. Renders eligibility decisions into warm prose. It receives a "Context Packet" containing eligible schemes, missed borderline schemes, and strict rule-engine reasoning.
- **`pipeline.py`**:
  - **The Blueprint**: Strictly enforces the sequence: **Normalize** (LLM #1) → **Retrieve** (RAG) → **Filter** (Rule Engine) → **Rank** (Confidence Math) → **Explain** (Audit Log) → **Translate** (IndicTrans2).
- **`llm_router.py`**:
  - **`LLMRouter`**: Implements the 4-tier cascade: **Ollama (Local Mistral) → Google Gemini → Sarvam (Indic-specialized) → OpenAI GPT-4o-mini**.
  - **Scrubbing**: `scrub_to_json()` uses regex to extract JSON blocks and `re.sub` to strip internal `<think>` blocks from model outputs.
  - **Performance**: Now tracks `model_used` and `latency_ms` for every LLM call to optimize the cascade and cache quality.
- **`eligibility.py`**:
  - **`filter_schemes()`**: The Hybrid Rule Engine. Runs semantic results from MongoDB through hard-coded logic filters for Age, Income, Gender, and State.
  - **Borderline Detection**: Schemes that slightly miss the threshold are tagged as "missed detection" to be surfaced as additional suggestions.
- **`validator.py`**: 
  - **`validate_entities()`**: Sanitizes LLM outputs. Normalizes genders (`aurat` → `female`) and scales ambiguous numbers (handles "2" specifically as "200000" when income is expected).
  - **`resolve_conflict()`**: Implements **Truth Source Priority**: *Manual User Input > OCR > LLM Extraction*.
- **`responses.py`**: The template bank for all 13 supported languages. Contains the numeric menu templates and the 4-question profile rotation.

---

## 📁 `services/` — Functional Modules
- **`translation_service.py`**:
  - **`TranslationPipeline`**: Lazy-loads the `ai4bharat/indictrans2-en-indic-1B` model.
  - **Logic**: Maps ISO codes to NLLB tags. Uses `tgt_lang_id` on the model generate call for high-fidelity translation from English canonical state.
- **`voice.py`**:
  - **STT**: Uses OpenAI Whisper. Injects language hints from the session to improve transcription accuracy for Indian dialects.
  - **TTS**: Maps ISO codes to Google's BCP-47 tags (e.g., `hi` → `hi-IN`) and generates `.mp3` via `gTTS`.
- **`rag_service.py`**: Performs O(N) semantic scan over MongoDB using cosine similarity.
- **`explainability_service.py`**: Writes to the `eligibility_checks` collection. Captures exactly why a user was rejected (e.g., *"Age 45 > Max Age 40"*).
- **`confidence_service.py`**: Calculates the **Truth Score**. Extraction Confidence (profile completeness) x Semantic Score (RAG relevance) x Rule Confidence (hard match quality).

---

## 📁 `models/` — Data Persistence
- **`db_client.py`**: Singleton connection to MongoDB Atlas. Enforces TTL indexes on cache and strict field validation on the `schemes_structured` collection.
- **`users.py`**:
  - **`normalize_user()`**: Critical on-read transformer. Real-time migration of legacy V1 user documents to the V2 nested profile structure.
- **`llm_cache.py`**: Implements a 3rd-layer cache indexed by SHA-256 hashes of input text. Now records the specific model used and the generation latency to provide analytics and ensure deterministic responses.

---

## 📁 `frontend/` — User Interface
- **`index.html`**: SPA layout with a custom audio visualizer.
- **`main.js`**: Manages the Web Audio API for recording, handles AJAX polling for long-running LLM tasks, and implements the "typing" indicator effect.
- **`style.css`**: Defines the "Glassmorphism" theme and the mobile-first WhatsApp chat layout.
- **`static/audio/`**: Transient store for voice messages; automatically pruned by `core/cleanup.py`.

---

## 📁 `tests/` — Quality Assurance
- **`test_bot.py`**: 
  - **Unit**: Validates number normalization, entity resolution, and template formatting.
  - **Integration**: Simulates full 7-language conversations to verify the 4-tier LLM cascade and cache durability.
