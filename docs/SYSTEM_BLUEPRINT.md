# 🇮🇳 YojnaSetuBot — FINAL MASTER SYSTEM PROMPT
## Complete Production Blueprint for Multilingual Welfare Intelligence Assistant
### Web-Based WhatsApp-Style Interface (No WhatsApp Deployment Required)

---

## ⚙️ SECTION 0: WHAT YOU ARE BUILDING

You are building **YojnaSetuBot**, a multilingual, voice-first AI chatbot that runs on a **custom WhatsApp-style web interface** (no WhatsApp API required). Citizens interact via this web UI using text or voice, and the bot helps them discover and verify Indian government welfare schemes.

**Stack at a Glance:**
- Frontend: WhatsApp-style HTML/CSS/JS web page
- Backend: Python 3.10+ Flask REST API
- Database: MongoDB Atlas (`welfare_chatbot` database)
- Primary LLM: mistral via Ollama (local)
- Secondary LLM: Google Gemini 1.5 Pro/Flash (API fallback)
- Tertiary LLM: Sarvam AI `sarvam-m` (Indic language generation)
- Emergency Fallback LLM: OpenAI GPT-4o-mini (via LangChain)
- STT: OpenAI Whisper (base/tiny) → AI4Bharat IndicWhisper (refinement)
- TTS: Indic Parler-TTS (primary) → gTTS (fallback)
- OCR: Tesseract (`eng+hin+guj+tam+kan`)

---

## 🖥️ SECTION 1: FRONTEND — WHATSAPP-STYLE WEB UI

### 1.1 UI Layout
Build a single-page application that visually mimics WhatsApp Web:

```
┌─────────────────────────────────────────────────────────┐
│  🟢 YojnaSetuBot          [🌐 Language] [🔄 Reset]      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   [Bot Bubble] Namaste! Aap kaunsi bhasha mein baat     │
│                karna chahte hain?                       │
│                                    [User Bubble] Hindi  │
│   [Bot Bubble] बहुत अच्छा! आप किस श्रेणी की योजनाएं  │
│                जानना चाहते हैं?                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  📎 [Attach Doc]  [🎤 Hold to Record]  [Type...]  [➤]  │
└─────────────────────────────────────────────────────────┘
```

### 1.2 UI Features Required
- **Chat bubbles**: Left (bot) / Right (user), with timestamps
- **Voice recording**: Hold-to-record mic button → sends audio blob to `/api/chat`
- **Voice playback**: Bot responses that came from TTS show a ▶️ play button inline in the bubble
- **Document upload**: Paper clip icon → opens file picker → sends to `/api/upload`
- **Language badge**: Shows current detected language in the header (e.g., "🌐 हिंदी")
- **Typing indicator**: Animated dots while bot is processing
- **Bottom menu bar** (always visible after first response):
  ```
  [1️⃣ Reset] [2️⃣ Change Language] [3️⃣ Check Eligibility] [4️⃣ Help]
  ```
- **Scheme cards**: When schemes are returned, render them as expandable cards with:
  - Scheme name, Category badge, State, Benefits summary
  - "Why Eligible" accordion section
  - Application link button

### 1.3 API Calls from Frontend
```javascript
// Text message
POST /api/chat
{ "phone_number": "web_user_abc123", "message": "मुझे खेती की योजनाएं चाहिए", "input_type": "text" }

// Voice message
POST /api/chat  (multipart/form-data)
{ "phone_number": "web_user_abc123", "audio": <blob>, "input_type": "voice" }

// Document upload
POST /api/upload  (multipart/form-data)
{ "phone_number": "web_user_abc123", "document": <file>, "document_type": "aadhaar" }
```

### 1.4 Session Identity
Since there is no WhatsApp phone number, generate a `web_user_<uuid4>` on first page load and store in `localStorage`. Use this as the `phone_number` field throughout.

---

## 🗂️ SECTION 2: FOLDER STRUCTURE

```
YojnaSetuBot/
│
├── .env                         # All secrets: API keys, DB URI, tokens
├── .gitignore
├── requirements.txt
├── README.md
├── run.py                       # Flask app entry point
│
├── frontend/                    # WhatsApp-style web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── core/
│   ├── config.py                # Loads .env variables
│   └── logger.py                # Centralized logging
│
├── api/
│   ├── routes.py                # /api/chat and /api/upload endpoints
│   └── middleware.py            # Input validation, CORS, rate limiting
│
├── engine/
│   ├── engine.py                # All 12 LLM prompt functions
│   ├── llm_router.py            # mistral → Gemini → Sarvam → GPT-4o-mini routing
│   ├── state_manager.py         # Bot state machine (awaiting_language → chatting)
│   ├── eligibility.py           # Rule engine: scheme filtering logic
│   └── cache.py                 # MongoDB-backed SHA256 prompt cache
│
├── services/
│   ├── stt.py                   # Whisper + IndicWhisper STT pipeline
│   ├── tts.py                   # Indic Parler-TTS + gTTS pipeline
│   ├── ocr.py                   # Tesseract OCR + entity extraction
│   └── scrubber.py              # Post-processing: strip <think>, markdown, etc.
│
├── models/
│   ├── db_client.py             # MongoDB Atlas connection (pymongo + certifi)
│   ├── users.py                 # User CRUD
│   ├── schemes.py               # Scheme queries and RAG retrieval
│   └── analytics.py            # Chat session + eligibility logging
│
└── datasets/
    └── schemes_data.json        # Seed data for welfare schemes
```

---

## 🤖 SECTION 3: LLM ROUTING ARCHITECTURE

### 3.1 The 4-Tier LLM Cascade

```
User Input
    │
    ▼
[Tier 1] mistral via Ollama (LOCAL — port 11434)
    │  ✅ Primary for all tasks
    │  ❌ If Ollama not running or timeout > 8s
    ▼
[Tier 2] Google Gemini 1.5 Flash / Pro (API)
    │  ✅ Fallback for complex reasoning, long context, Indic text
    │  ❌ If Gemini quota exceeded or API error
    ▼
[Tier 3] Sarvam AI sarvam-m (API)
    │  ✅ Specialized for Hindi, Gujarati, Tamil, Kannada, Bengali
    │  ❌ If Sarvam API error
    ▼
[Tier 4] OpenAI GPT-4o-mini via LangChain (API)
    ✅ Emergency fallback — always available
```
Logic → mistral
Language handling → Sarvam / Translation layer

Input → Detect language
      → Translate to English (optional)
      → mistral processing
      → Translate back to user language
    
### 3.2 `llm_router.py` Logic (Pseudocode)

```python
def call_llm(prompt: str, language: str, task: str) -> str:
    # 1. Check cache first (SHA256 hash of prompt)
    cached = cache.get(prompt)
    if cached: return cached

    # 2. Try mistral (Ollama local)
    try:
        if is_ollama_running():
            response = call_ollama_llama3(prompt, timeout=8)
            cache.set(prompt, response)
            return response
    except: pass

    # 3. Try Gemini Flash
    try:
        response = call_gemini(prompt, model="gemini-1.5-flash")
        cache.set(prompt, response)
        return response
    except: pass

    # 4. Try Sarvam AI (for Indic languages)
    if language in ["hi", "gu", "ta", "kn", "bn"]:
        try:
            response = call_sarvam(prompt)
            cache.set(prompt, response)
            return response
        except: pass

    # 5. Emergency: GPT-4o-mini
    response = call_openai_gpt4o_mini(prompt)
    cache.set(prompt, response)
    return response
```

---

## 📋 SECTION 4: THE 12-PROMPT ENGINE (`engine.py`)

Each function is a standalone, single-responsibility LLM call. All output passes through `scrubber.py` before being returned.

---

### PROMPT 1: `detect_language_llm(text: str) -> str`

```python
SYSTEM = """
You are a language detection expert. Identify the language of the input text.
Return ONLY a JSON object with no extra text:
{"language_code": "hi", "language_name": "Hindi", "script": "Devanagari"}

Supported languages and codes:
- English: en
- Hindi: hi  
- Gujarati: gu
- Tamil: ta
- Kannada: kn
- Bengali: bn
- Marathi: mr
- Telugu: te
- Punjabi: pa
- Malayalam: ml

If multiple languages are detected, return the dominant one.
Never explain. Return only the JSON.
"""
USER = f"Detect the language of this text: {text}"
```

---

### PROMPT 2: `detect_intent_llm(text: str, language: str) -> str`

```python
SYSTEM = """
You are an intent classifier for a government welfare scheme chatbot in India.
The user is speaking in {language}.

Classify the user's message into ONE of these intents:
- GREET: Hello, hi, namaste, kem cho, vanakkam
- SELECT_LANGUAGE: User is choosing or changing their language
- SELECT_CATEGORY: User wants to pick a scheme domain/category
- FREE_SPEECH_PROFILE: User is describing their personal situation (age, income, occupation)
- ASK_SCHEME_INFO: User wants details about a specific scheme
- UPLOAD_DOCUMENT: User wants to upload a document for verification
- CHECK_ELIGIBILITY: User wants to verify their eligibility
- MENU_RESET: User wants to start over
- MENU_CHANGE_LANGUAGE: User wants to change language
- MENU_HELP: User needs help with navigation
- FAKE_SCHEME_CHECK: User is asking if a scheme is real
- FAMILY_RECOMMENDATION: User wants schemes for family members
- GENERAL_QUERY: Any other welfare-related query

Return ONLY a JSON object:
{"intent": "FREE_SPEECH_PROFILE", "confidence": 0.92}
Never explain. Return only JSON.
"""
```

---

### PROMPT 3: `detect_category_llm(text: str, language: str) -> str`

```python
SYSTEM = """
You are a domain/category classifier for Indian government welfare schemes.
The user may describe their category using native language words, synonyms, dialects, or slang.

You must intelligently map ANY user input to ONE of these official categories:
- Agriculture (खेती, ખેતી, khedut, krishi, farm, farming, kisan, kisaan)
- Education (शिक्षा, ભણતર, scholarship, school, college, vidya)
- Health (स्वास्थ्य, આरोग्ય, hospital, medical, bimari, arogya)
- Housing (आवास, ઘर, ghar, makaan, awas, housing)
- Employment (रोजगार, રોજગાર, naukri, job, rozgaar, employment)
- Women & Child (महिला, સ્ત્રી, mahila, women, beti, child)
- Senior Citizens (वृद्ध, વૃદ્ધ, bujurg, pension, old age, vriddh)
- Disability (दिव्यांग, divyang, viklang, disabled)
- Financial Assistance (आर्थिक, loan, mudra, subsidy, financial)
- Scholarships (छात्रवृत्ति, scholarship, merit)

Do NOT use mapping dictionaries. Use deep language understanding to detect ANY synonym, 
native word, misspelling, or dialect variant and map it to the closest official category.

Return ONLY JSON:
{"category": "Agriculture", "confidence": 0.95, "detected_keywords": ["kheti", "khedut"]}
"""
```

---

### PROMPT 4: `extract_entities_llm(text: str, language: str) -> dict`

```python
SYSTEM = """
You are a demographic entity extractor for an Indian government welfare chatbot.
The user is speaking in {language}. Extract ALL available demographic information.

Extract these entities if present:
- age: (integer, e.g., 45)
- income: (annual in INR, e.g., 150000)
- state: (Indian state name in English, e.g., "Gujarat")
- gender: ("male", "female", "other")
- occupation: (e.g., "farmer", "student", "unemployed")
- caste_category: ("general", "obc", "sc", "st")
- family_size: (integer)
- has_disability: (boolean)
- is_student: (boolean)
- marital_status: ("married", "unmarried", "widow", "widower")

For income: understand phrases like "2 lakh", "₹2,00,000", "do lakh", "2 लाख", "2 લાખ" — all mean 200000.
For age: understand "60 saal", "साठ साल", "sixty years" — all mean 60.
For state: understand regional names like "Gujarat", "Gujrat", "ગુજરાત" — all mean Gujarat.

Return ONLY JSON with found entities. Omit keys that are not mentioned:
{"age": 62, "income": 150000, "state": "Karnataka", "gender": "male"}
"""
```

---

### PROMPT 5: `extract_entities_from_ocr_llm(ocr_text: str, language: str) -> dict`

```python
SYSTEM = """
You are an expert at parsing OCR-extracted text from Indian government documents.
The OCR text may contain noise, typos, and formatting artifacts.

From the raw OCR text, extract:
- name: Full name of the document holder
- dob: Date of birth (normalize to YYYY-MM-DD)
- age: Calculate from DOB if possible
- aadhaar_number: 12-digit Aadhaar number (if present)
- income: Annual income amount in INR
- caste_category: Caste certificate category (SC/ST/OBC/General)
- state: State name
- address: Full address if present
- gender: Gender of holder

Intelligently handle:
- Dates in DD/MM/YYYY or DD-MM-YYYY format
- Numbers with commas (2,00,000 = 200000)
- Hindi/Gujarati text mixed with English
- Partial or garbled text from poor scans

Return ONLY JSON:
{"name": "Ramesh Kumar", "dob": "1962-05-14", "age": 62, "income": 120000, "state": "Rajasthan"}
"""
```

---

### PROMPT 6: `generate_greeting(language: str) -> str`

```python
SYSTEM = """
You are YojnaSetuBot, a warm and friendly Senior Government Advisor helping Indian citizens find welfare schemes.

Generate a warm, natural greeting in {language} using its NATIVE SCRIPT ONLY.
Rules:
- Use natural filler words appropriate to the language (Ji, Namaste, Kem cho, Vanakkam, etc.)
- Briefly explain what you can help with (government welfare schemes)
- Ask the user what they are looking for OR to describe their situation
- Sound like a helpful elder relative, NOT a robot
- NO markdown formatting
- NO English words if language is not English
- Keep it under 3 sentences
- End by asking them to describe their need or choose a category

NEVER use: *, #, <think>, bullet points, numbered lists in the response.
"""
```

---

### PROMPT 7: `generate_follow_up_question(missing_fields: list, language: str) -> str`

```python
SYSTEM = """
You are YojnaSetuBot, a warm Senior Government Advisor.
You are collecting missing information from the user to find matching welfare schemes.

Missing fields that still need to be collected: {missing_fields}

Generate a SINGLE, natural conversational question in {language} (native script only) 
that asks for the MOST IMPORTANT missing piece of information.

Priority order: income > age > state > occupation > gender > caste_category

Rules:
- Ask for ONLY ONE piece of information per message
- Sound natural and warm, like asking a family member
- Give examples of expected answers inline (e.g., "जैसे ₹1 लाख, ₹2 लाख")
- NO markdown, NO bullet points, NO English in Indic responses
- Use appropriate honorifics for the language (Aap, Tame, Neenga, etc.)
"""
```

---

### PROMPT 8: `generate_options_menu(language: str) -> str`

```python
SYSTEM = """
Generate the 4-option navigation menu for YojnaSetuBot in {language} using NATIVE SCRIPT ONLY.

The menu should present these 4 options naturally, embedded in a warm closing sentence:

1️⃣ Reset / Start Over — Begin fresh from the beginning
2️⃣ Change Language — Switch to a different language  
3️⃣ Check Eligibility Again — Re-run with different details
4️⃣ Help — Learn what the bot can do

Format as a warm, conversational closing line followed by the numbered options.
NO markdown. NO English in Indic language responses.
Example format (do NOT use this exact text, generate fresh):
"आप इनमें से कोई भी विकल्प चुन सकते हैं:"
1️⃣ नई शुरुआत | 2️⃣ भाषा बदलें | 3️⃣ पात्रता जांचें | 4️⃣ सहायता
"""
```

---

### PROMPT 9: `generate_scheme_explanation(profile: dict, top_schemes: list, language: str) -> str`

```python
SYSTEM = """
You are YojnaSetuBot, a warm, knowledgeable Senior Government Advisor.
You have found matching welfare schemes for this citizen.

USER PROFILE: {profile}
MATCHING SCHEMES (from database): {top_schemes}

Generate a warm, conversational response in {language} (NATIVE SCRIPT ONLY) that:

1. Opens with a warm, personalized line acknowledging their situation
2. For EACH scheme (maximum 5), explain in this natural flow:
   - Scheme name (in native language if possible)
   - What benefit they will receive (in simple, plain language)
   - WHY they are eligible (list matched conditions: age ✔, income ✔, state ✔)
   - Key documents needed (1-2 most important)
   - How to apply (in one simple line)

3. If any scheme does NOT match perfectly, explain what condition is missing.

Rules:
- Sound like a helpful, knowledgeable relative explaining schemes at home
- NO markdown, NO asterisks, NO bullet symbols
- NO English words if language is not English
- Use simple vocabulary — imagine explaining to a 50-year-old farmer
- Use natural connectors: "इसके साथ ही...", "और एक अच्छी बात...", etc.
- End with the options menu (call generate_options_menu separately and append)

CRITICAL: Only mention schemes from the provided database list. 
NEVER invent or hallucinate scheme names or benefits.
"""
```

---

### PROMPT 10: `translate_system_message(text: str, target_language: str) -> str`

```python
SYSTEM = """
Translate the following system/error message to {target_language} using NATIVE SCRIPT ONLY.

Rules:
- Maintain the meaning exactly
- Use simple, clear vocabulary
- Sound warm, not robotic
- NO markdown formatting
- For technical terms with no native equivalent, use the simplest possible native paraphrase

Text to translate: {text}
"""
```

---

### PROMPT 11: `clean_voice_input(raw_stt_text: str) -> str`

```python
SYSTEM = """
You are cleaning raw Speech-to-Text output from an Indian language voice assistant.

The input may contain:
- Filler words: "um", "uh", "hmm", "aaa", "haan haan"
- Repeated words: "mera mera naam" → "mera naam"
- STT artifacts: "[inaudible]", "[noise]", "..."
- Fragmented sentences from pauses mid-speech
- Mixed scripts from code-switching

Clean the text by:
1. Removing filler/repeated words
2. Removing STT artifacts and noise markers
3. Joining fragmented sentences into coherent text
4. Preserving the ORIGINAL LANGUAGE AND SCRIPT (do not translate)
5. Preserving all meaningful content

Return ONLY the cleaned text. No explanation.
"""
```

---

### PROMPT 12: `optimize_for_tts(text: str, language: str) -> str`

```python
SYSTEM = """
Optimize the following text for Text-to-Speech (TTS) audio generation in {language}.

Transformations required:
1. Expand abbreviations: "PM" → "प्रधानमंत्री", "Rs." → "रुपये", "₹2L" → "दो लाख रुपये"
2. Spell out URLs: "pmkisan.gov.in" → "पीएम किसान सरकारी वेबसाइट"
3. Remove ALL special characters: *, #, 1️⃣, ✔, →, |, /, \
4. Remove markdown formatting completely
5. Convert digits to words where natural: "5" → "पाँच" (in Hindi)
6. Add natural pauses using commas where a speaker would pause
7. Remove parenthetical asides that don't work in speech
8. Ensure sentences are complete and flow naturally when spoken aloud

Return ONLY the TTS-optimized text. No explanation.
"""
```

---

## 🔄 SECTION 5: STATE MACHINE (`state_manager.py`)

```
States:
├── awaiting_language     → First message ever; detect or ask for language
├── awaiting_category     → Language set; ask for category or detect from free speech  
├── collecting_profile    → Category set; extract entities, ask for missing fields
├── showing_schemes       → Profile complete; display filtered schemes
├── awaiting_document     → User triggered OCR flow; waiting for document upload
└── chatting              → General follow-up questions within session

Global Menu Interceptor (runs BEFORE state machine on EVERY message):
If input is "1" / "१" / "૧" → RESET → delete session → go to awaiting_language
If input is "2" / "२" / "૨" → go to awaiting_language (keep profile)
If input is "3" / "३" / "૩" → go to collecting_profile (keep language)
If input is "4" / "४" / "૪" → generate_help_message(language)
```

### State Transition Logic

```python
def handle_message(phone_number, message, input_type):
    user = db.users.find_one({"phone_number": phone_number})
    
    # Step 1: Global menu interceptor
    if is_menu_command(message):
        return handle_menu_command(message, user)
    
    state = user.get("conv_state", "awaiting_language")
    
    if state == "awaiting_language":
        lang = detect_language_llm(message)
        db.users.update(phone_number, {"language": lang, "conv_state": "awaiting_category"})
        return generate_greeting(lang)
    
    elif state == "awaiting_category":
        intent = detect_intent_llm(message, user.language)
        if intent == "SELECT_CATEGORY" or intent == "FREE_SPEECH_PROFILE":
            category = detect_category_llm(message, user.language)
            entities = extract_entities_llm(message, user.language)
            db.users.update(phone_number, {
                "category": category,
                "conv_data": entities,
                "conv_state": "collecting_profile"
            })
            # Check if profile is already complete from free speech
            missing = get_missing_fields(entities, category)
            if not missing:
                return trigger_scheme_search(user)
            return generate_follow_up_question(missing, user.language)
    
    elif state == "collecting_profile":
        new_entities = extract_entities_llm(message, user.language)
        merged = {**user.conv_data, **new_entities}
        db.users.update(phone_number, {"conv_data": merged})
        missing = get_missing_fields(merged, user.category)
        if not missing:
            db.users.update(phone_number, {"conv_state": "showing_schemes"})
            return trigger_scheme_search(user)
        return generate_follow_up_question(missing, user.language)
    
    elif state in ["showing_schemes", "chatting"]:
        # Handle follow-up questions, free speech updates, re-searches
        intent = detect_intent_llm(message, user.language)
        if intent == "CHECK_ELIGIBILITY":
            return trigger_scheme_search(user)
        if intent == "UPLOAD_DOCUMENT":
            db.users.update(phone_number, {"conv_state": "awaiting_document"})
            return translate_system_message("Please upload your document now.", user.language)
        # General conversational response
        return generate_conversational_reply(message, user)
    
    elif state == "awaiting_document":
        return translate_system_message("Please use the 📎 attachment button to upload your document.", user.language)
```

---

## 🔍 SECTION 6: ELIGIBILITY ENGINE (`eligibility.py`)

### 6.1 Scheme Filtering (RAG + Rule Engine Hybrid)

```python
def filter_schemes(profile: dict, category: str) -> list:
    # Step 1: MongoDB keyword + category pre-filter
    base_query = {
        "category": {"$regex": category, "$options": "i"},
        "$or": [
            {"state": profile.get("state", "")},
            {"state": "All India"},
            {"state": "National"}
        ]
    }
    candidate_schemes = db.schemes.find(base_query).limit(50)
    
    # Step 2: Rule engine eligibility check
    eligible = []
    for scheme in candidate_schemes:
        result = check_eligibility_rules(scheme, profile)
        if result["eligible"]:
            eligible.append({**scheme, "eligibility_reasons": result["reasons"]})
    
    # Step 3: Rank by match completeness
    ranked = rank_schemes(eligible, profile)
    return ranked[:5]  # Return top 5

def check_eligibility_rules(scheme: dict, profile: dict) -> dict:
    rules = scheme.get("eligibility_rules", {})
    reasons = []
    not_eligible_reasons = []
    
    if "min_age" in rules and profile.get("age"):
        if profile["age"] >= rules["min_age"]:
            reasons.append(f"Age {profile['age']} meets minimum age {rules['min_age']} ✔")
        else:
            not_eligible_reasons.append(f"Age requirement not met (need {rules['min_age']}+)")
    
    if "max_income" in rules and profile.get("income"):
        if profile["income"] <= rules["max_income"]:
            reasons.append(f"Income ₹{profile['income']} within limit ₹{rules['max_income']} ✔")
        else:
            not_eligible_reasons.append(f"Income exceeds limit of ₹{rules['max_income']}")
    
    # ... similar checks for gender, caste, occupation, etc.
    
    eligible = len(not_eligible_reasons) == 0
    return {"eligible": eligible, "reasons": reasons, "unmet": not_eligible_reasons}
```

---

## 🎤 SECTION 7: STT PIPELINE (`services/stt.py`)

```python
def transcribe_audio(audio_file_path: str, language_code: str = None) -> str:
    """
    Primary: OpenAI Whisper (base/tiny) — fast, general purpose
    Optional Refinement: AI4Bharat IndicWhisper — for Indic language accuracy
    """
    
    # Step 1: Whisper base transcription
    import whisper
    model = whisper.load_model("base")  # or "tiny" for speed
    
    options = {}
    if language_code and language_code != "en":
        options["language"] = WHISPER_LANGUAGE_MAP.get(language_code, None)
    
    result = model.transcribe(audio_file_path, **options)
    raw_text = result["text"]
    
    # Step 2: Optional IndicWhisper refinement for Indic languages
    if language_code in ["hi", "gu", "ta", "kn", "bn", "mr", "te"]:
        try:
            refined = refine_with_indic_whisper(audio_file_path, language_code)
            if refined and len(refined) > len(raw_text) * 0.8:  # Use if plausible
                raw_text = refined
        except Exception as e:
            logger.warning(f"IndicWhisper refinement failed: {e}, using Whisper output")
    
    # Step 3: Clean STT artifacts using LLM
    cleaned = clean_voice_input(raw_text)
    return cleaned

WHISPER_LANGUAGE_MAP = {
    "hi": "hindi", "gu": "gujarati", "ta": "tamil",
    "kn": "kannada", "bn": "bengali", "mr": "marathi",
    "te": "telugu", "pa": "punjabi", "ml": "malayalam"
}
```

---

## 🔊 SECTION 8: TTS PIPELINE (`services/tts.py`)

```python
def synthesize_speech(text: str, language_code: str) -> str:
    """
    Primary: Indic Parler-TTS — natural Indic voices
    Fallback: gTTS — reliable, language-wide coverage
    Returns: path to audio file
    """
    
    # Step 1: Optimize text for TTS
    tts_ready_text = optimize_for_tts(text, language_code)
    
    output_path = f"/tmp/tts_{uuid4()}.mp3"
    
    # Step 2: Try Indic Parler-TTS (primary)
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        # ... Parler-TTS generation code
        # Use appropriate voice description for language
        voice_desc = PARLER_VOICE_MAP.get(language_code, "A female speaker with a warm voice")
        audio = parler_model.generate(tts_ready_text, voice_desc)
        save_audio(audio, output_path)
        return output_path
    except Exception as e:
        logger.warning(f"Parler-TTS failed: {e}, falling back to gTTS")
    
    # Step 3: Fallback to gTTS
    from gtts import gTTS
    tts = gTTS(text=tts_ready_text, lang=language_code, slow=False)
    tts.save(output_path)
    return output_path

def should_generate_tts(input_type: str) -> bool:
    """Only generate TTS if the user's input was voice"""
    return input_type == "voice"
```

---

## 📄 SECTION 9: OCR PIPELINE (`services/ocr.py`)

```python
def process_document(image_path: str, document_type: str, language: str) -> dict:
    """
    1. Run Tesseract OCR with multi-language support
    2. Extract entities using LLM prompt 5
    3. Compare with user profile and scheme rules
    4. Store result in documents_ocr collection
    """
    import pytesseract
    from PIL import Image
    
    # Multi-language OCR config
    lang_config = {
        "hi": "eng+hin",
        "gu": "eng+hin+guj",
        "ta": "eng+hin+tam",
        "kn": "eng+hin+kan",
        "en": "eng"
    }
    ocr_lang = lang_config.get(language, "eng+hin")
    
    # Step 1: Run OCR
    image = Image.open(image_path)
    raw_ocr_text = pytesseract.image_to_string(image, lang=ocr_lang)
    
    # Step 2: Extract structured data using LLM
    extracted = extract_entities_from_ocr_llm(raw_ocr_text, language)
    
    # Step 3: Verify against user profile
    verification = verify_against_profile(extracted, phone_number)
    
    # Step 4: Store in MongoDB
    db.documents_ocr.insert({
        "phone_number": phone_number,
        "document_type": document_type,
        "extracted_data": extracted,
        "verification_status": verification["status"],
        "timestamp": datetime.utcnow()
    })
    
    return {"extracted": extracted, "verification": verification}
```

---

## 🧹 SECTION 10: OUTPUT SCRUBBER (`services/scrubber.py`)

```python
import re

def scrub_llm_output(text: str) -> str:
    """
    Enforce clean output REGARDLESS of what LLM generates.
    Called on EVERY LLM response before returning to user.
    """
    
    # 1. Remove <think>...</think> reasoning blocks (including nested)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove all markdown formatting
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # Bold/italic
    text = re.sub(r'#{1,6}\s+', '', text)                 # Headers
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
    
    # 3. Remove repeated trailing sentences
    sentences = text.split('।') if '।' in text else text.split('.')
    seen = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s_clean)
    
    # 4. Remove leaked internal tags
    text = re.sub(r'<[^>]+>', '', '।'.join(unique_sentences) if '।' in text else '. '.join(unique_sentences))
    
    # 5. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()
```

---

## 💾 SECTION 11: MONGODB CACHE (`engine/cache.py`)

```python
import hashlib, json
from datetime import datetime, timedelta

def get_cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

def get_cached_response(prompt: str) -> str | None:
    key = get_cache_key(prompt)
    cached = db.ai_response_cache.find_one({
        "hash": key,
        "created_at": {"$gt": datetime.utcnow() - timedelta(hours=24)}
    })
    return cached["response"] if cached else None

def cache_response(prompt: str, response: str):
    key = get_cache_key(prompt)
    db.ai_response_cache.update_one(
        {"hash": key},
        {"$set": {"response": response, "created_at": datetime.utcnow()}},
        upsert=True
    )
    # Target: sub-50ms cache hit latency via MongoDB index on "hash" field
```

---

## 🌐 SECTION 12: FLASK API ROUTES (`api/routes.py`)

### POST `/api/chat`

```
Input:  { phone_number, message, input_type: "text"|"voice", audio_file? }
Output: { response_text, response_audio_url?, schemes?, language, state }

Flow:
1. If input_type == "voice": STT → cleaned_text
2. Global menu interceptor check
3. State machine routing
4. LLM call (with cache check)
5. Scrub output
6. If original input was voice: TTS → audio file URL
7. Log to chat_sessions
8. Return JSON response
```

### POST `/api/upload`

```
Input:  { phone_number, document (multipart), document_type }
Output: { extracted_data, verification_status, merged_profile, message }

Flow:
1. Save uploaded file to /tmp/
2. Run OCR pipeline
3. Merge extracted data with user profile in MongoDB
4. Check if profile is now complete → trigger scheme search if yes
5. Return verification result + updated profile summary
```

### GET `/api/health`

```
Returns: { status, ollama_status, gemini_status, mongodb_status, timestamp }
```

---

## 📊 SECTION 13: MONGODB SCHEMA (Complete)

### Collection: `users`
```json
{
  "phone_number": "web_user_abc123",
  "conv_state": "collecting_profile",
  "conv_data": {
    "age": 62, "income": 120000, "state": "Gujarat",
    "occupation": "farmer", "gender": "male", "caste_category": "obc"
  },
  "language": "gu",
  "category": "Agriculture",
  "created_at": "ISODate",
  "last_active": "ISODate"
}
```

### Collection: `schemes`
```json
{
  "scheme_id": "PM_KISAN_001",
  "scheme_name": "PM-KISAN Samman Nidhi",
  "category": "Agriculture",
  "state": "All India",
  "eligibility_rules": {
    "min_age": 18, "max_age": null,
    "max_income": 200000,
    "gender": "any",
    "occupation": ["farmer", "agriculture"]
  },
  "income_limit": 200000,
  "benefits": "₹6000 per year in 3 installments of ₹2000",
  "documents_required": ["Aadhaar", "Bank Passbook", "Land Records"],
  "application_link": "https://pmkisan.gov.in",
  "is_verified": true
}
```

### Collection: `ai_response_cache`
```json
{
  "hash": "sha256_hash_of_prompt",
  "response": "cached LLM response",
  "created_at": "ISODate"
}
```
Index: `{ "hash": 1 }` (unique), `{ "created_at": 1 }` (TTL: 24 hours)

---

## 🔒 SECTION 14: ENVIRONMENT VARIABLES (`.env`)

```env
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/welfare_chatbot?retryWrites=true&w=majority

# LLM APIs
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
SARVAM_API_KEY=your_sarvam_key

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Flask
FLASK_SECRET_KEY=your_secret_key
FLASK_ENV=development
PORT=5000

# TTS/STT
WHISPER_MODEL=base
PARLER_TTS_MODEL=ai4bharat/indic-parler-tts
TESSERACT_CMD=/usr/bin/tesseract
```

---

## 🚀 SECTION 15: REQUIREMENTS.TXT

```
flask==3.0.0
flask-cors==4.0.0
pymongo==4.6.1
certifi==2024.2.2
python-dotenv==1.0.1
openai-whisper==20231117
pytesseract==0.3.10
Pillow==10.2.0
gTTS==2.5.1
google-generativeai==0.5.0
openai==1.14.0
langchain==0.1.12
langchain-openai==0.0.8
requests==2.31.0
python-multipart==0.0.9
torch==2.2.0
transformers==4.38.2
```

---

## ✅ SECTION 16: KEY RULES (NEVER BREAK THESE)

1. **SCRUBBER IS MANDATORY** — Every single LLM output passes through `scrub_llm_output()` before reaching the user. No exceptions.
2. **CACHE FIRST** — Before calling ANY LLM, always check the SHA256 cache. Every response gets cached after generation.
3. **LLaMA3 PRIMARY** — Always try Ollama/mistral first. Only cascade to Gemini, Sarvam, GPT-4o-mini on failure.
4. **NATIVE SCRIPT ONLY** — When language is Hindi/Gujarati/Tamil/Kannada, responses must be in that native script. Zero English leakage, zero Romanization.
5. **TTS ONLY ON VOICE INPUT** — Only generate audio response if `input_type == "voice"`. Text inputs get text-only response.
6. **NEVER HALLUCINATE SCHEMES** — `generate_scheme_explanation` must ONLY use schemes passed to it from the database query. No invented scheme names or benefits.
7. **GLOBAL MENU ALWAYS WORKS** — Menu commands (1/2/3/4 and regional variants) must be intercepted BEFORE the state machine on every single message.
8. **PROFILE MERGING** — Entity extraction results always MERGE into existing profile (never overwrite), so data accumulates across messages.
9. **ONE QUESTION AT A TIME** — `generate_follow_up_question` always asks for exactly ONE missing field, not multiple at once.
10. **OCR MERGES INTO PROFILE** — Extracted OCR data merges into `conv_data`, and if profile becomes complete, automatically triggers scheme search.

---

*YojnaSetuBot Master Prompt v1.0 — Complete Production Blueprint*
*Web-Based WhatsApp-Style Interface | No WhatsApp Deployment Required*
