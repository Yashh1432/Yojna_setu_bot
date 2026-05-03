from __future__ import annotations

LANGS = ["en", "hi", "gu", "ta", "te", "kn", "bn", "mr", "pa", "ml", "or", "as", "ur"]

WELCOME_MESSAGES = {
    "en": "Please choose your language: English / Hindi / Gujarati",
    "hi": "Please choose your language: English / Hindi / Gujarati",
    "gu": "Please choose your language: English / Hindi / Gujarati",
    "ta": "Please choose your language: English / Hindi / Gujarati",
    "te": "Please choose your language: English / Hindi / Gujarati",
    "kn": "Please choose your language: English / Hindi / Gujarati",
    "bn": "Please choose your language: English / Hindi / Gujarati",
    "mr": "Please choose your language: English / Hindi / Gujarati",
    "pa": "Please choose your language: English / Hindi / Gujarati",
    "ml": "Please choose your language: English / Hindi / Gujarati",
    "or": "Please choose your language: English / Hindi / Gujarati",
    "as": "Please choose your language: English / Hindi / Gujarati",
    "ur": "Please choose your language: English / Hindi / Gujarati",
}

HELP_MESSAGES = {
    code: "You can describe your needs naturally. Example: I am a student from Gujarat with family income 2 lakh."
    for code in LANGS
}

CLARIFICATION_PROMPTS = {
    code: "Could you clarify what kind of scheme or support you need?" for code in LANGS
}

PROFILE_ACK_MESSAGES = {
    code: "Language updated. Tell me what kind of scheme you need." for code in LANGS
}

SCHEME_INTRO_MESSAGES = {
    code: "Based on your profile, these schemes look relevant:" for code in LANGS
}

CHECK_AGAIN_MESSAGES = {
    code: "I could not find a strong match yet. Share income and state for better suggestions."
    for code in LANGS
}

OPTIONS_MENU = {code: "1 Reset | 2 Change Language | 3 Help | 4 Check Eligibility" for code in LANGS}

ASK_QUESTIONS = {
    "age": {code: "What is your age?" for code in LANGS},
    "income": {code: "What is your approximate annual family income?" for code in LANGS},
    "state": {code: "Which state do you live in?" for code in LANGS},
    "category": {code: "If relevant, what is your category (General/OBC/SC/ST)?" for code in LANGS},
}


def _pick(language_code: str | None) -> str:
    code = str(language_code or "en").lower().strip()
    return code if code in LANGS else "en"


def get_welcome(language_code: str = "en") -> str:
    return WELCOME_MESSAGES[_pick(language_code)]


def get_help_menu(language_code: str = "en") -> str:
    return HELP_MESSAGES[_pick(language_code)]


def get_clarification_prompt(language_code: str = "en") -> str:
    return CLARIFICATION_PROMPTS[_pick(language_code)]


def get_profile_ack(language_code: str = "en") -> str:
    return PROFILE_ACK_MESSAGES[_pick(language_code)]


def get_scheme_intro(language_code: str = "en") -> str:
    return SCHEME_INTRO_MESSAGES[_pick(language_code)]


def get_check_again_message(language_code: str = "en") -> str:
    return CHECK_AGAIN_MESSAGES[_pick(language_code)]


def get_options_menu(language_code: str = "en") -> str:
    return OPTIONS_MENU[_pick(language_code)]


def get_followup_question(language_code: str, field: str, profile: dict | None = None) -> str:
    code = _pick(language_code)
    return ASK_QUESTIONS.get(field, ASK_QUESTIONS["category"]).get(code, ASK_QUESTIONS["category"]["en"])


def get_question(field: str, language_code: str = "en") -> str:
    return get_followup_question(language_code, field)


def get_apply_instructions(scheme: dict | None, lang: str = "en") -> str:
    if not scheme:
        return "Please tell me which scheme you want to apply for."
    link = scheme.get("application_link") or "https://www.india.gov.in/"
    return f"To apply for {scheme.get('scheme_name', 'this scheme')}, use the official link: {link}"
