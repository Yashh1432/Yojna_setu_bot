from datetime import datetime
import os
import uuid

from flask import Blueprint, abort, jsonify, request

from core.cleanup import cleanup_old_files
from core.limiter import limiter
from core.logger import get_logger, set_trace_id
from core.sanitizer import sanitize_text
from engine.validator import (
    is_scheme_allowed_for_user,
    is_true_national_scheme,
    is_user_state_scheme,
    normalize_state_for_geo,
)
from engine.state_manager import MENU_TEXT, MENU_TEXTS, handle_message
from models.db_client import db_client
from models.feedback_model import feedback_model
from models.users import user_model
from services.translation_service import translate_from_english
from services.voice import _VOICE_ENABLED, clean_stt_text, speech_to_text, text_to_speech

logger = get_logger("api.routes")

api_bp = Blueprint('api', __name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUDIO_DIR = os.path.join(_BASE_DIR, "frontend", "static", "audio")

SAFE_EMPTY_RESPONSE_FALLBACK = {
    "en": "I could not complete the search safely. Please try again or provide more details.",
    "hi": "I could not complete the search safely. Please try again or provide more details.",
    "ta": "I could not complete the search safely. Please try again or provide more details.",
    "ur": "I could not complete the search safely. Please try again or provide more details.",
}

def _final_geo_filter_schemes(schemes: list, user_state: str | None):
    user_state_norm = normalize_state_for_geo(user_state)
    if not user_state_norm:
        return list(schemes or []), []

    filtered = []
    rejected = []
    for item in schemes or []:
        scheme = dict(item or {})
        scheme_name = str(scheme.get("scheme_name") or "Unknown Scheme")
        scheme_state_raw = str(scheme.get("state") or "").strip()
        scheme_state_norm = normalize_state_for_geo(scheme_state_raw)
        geo_allowed = bool(scheme_state_norm and is_scheme_allowed_for_user(scheme, user_state_norm))

        if not geo_allowed:
            rejected.append(
                {
                    "scheme_name": scheme_name,
                    "scheme_state": scheme_state_raw or None,
                    "user_state": user_state,
                    "reason": "geo_rejected: scheme_state != user_state and not_all_india",
                }
            )
            continue

        if is_user_state_scheme(scheme, user_state_norm):
            scheme["why_match"] = [f"This scheme is available in {str(user_state or '').strip() or scheme_state_raw}."]
        elif is_true_national_scheme(scheme):
            scheme["why_match"] = ["This is a national/All India scheme."]

        filtered.append(scheme)

    return filtered, rejected


@api_bp.route('/chat', methods=['POST'])
@limiter.limit("60/minute")
def chat():
    """
    Phase 2 endpoint routing logic via state machine.
    Expects structured payload (JSON logic) OR Multipart (Voice logics)
    """
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)

    cleanup_old_files(_AUDIO_DIR, max_age_seconds=600)

    if request.is_json:
        data = request.get_json(silent=True) or {}
        phone_number = data.get('phone_number')
        message = data.get('message', '')
        input_type = data.get('input_type', 'text')
    else:
        phone_number = request.form.get('phone_number')
        input_type = request.form.get('input_type', 'text')
        message = request.form.get('message', '')

    input_type = str(input_type or "text").strip().lower()

    if not phone_number:
        phone_number = "anonymous_user"
        logger.info({"event": "anonymous_user", "trace_id": trace_id})

    logger.info({
        "event": "request_start",
        "trace_id": trace_id,
        "phone": phone_number,
        "input_type": input_type,
        "message_preview": message[:50],
    })

    user_lang_hint = "en"
    voice_audio_received = False
    if input_type == "voice":
        # ------------------------------------------------------------------
        # Guard: VOICE_ENABLED=false — return clear message, text chat unaffected
        # ------------------------------------------------------------------
        if not _VOICE_ENABLED:
            logger.info({"event": "voice_disabled", "trace_id": trace_id, "phone": phone_number})
            return jsonify({
                "response": "Voice input is currently disabled. Please use text chat.",
                "audio_url": "",
                "status": "success",
                "schemes": [],
                "errors": [],
                "user_state": "voice_disabled",
                "trace_id": trace_id,
            }), 200

        try:
            user_doc = user_model.get_user(phone_number)
            if not user_doc:
                user_doc, _ = user_model.create_or_get_user(phone_number)
            if user_doc and user_doc.get("language"):
                user_lang_hint = user_doc["language"]
        except Exception:
            pass

        audio_file = request.files.get('audio')
        if audio_file:
            voice_audio_received = True
            original_name = audio_file.filename or ""
            extension = os.path.splitext(original_name)[1].lower()
            allowed_exts = {".wav", ".webm", ".mp3", ".m4a", ".ogg", ".mp4", ".mpeg"}
            if extension not in allowed_exts:
                logger.warning(
                    f"Voice route: unexpected extension '{extension}' for file '{original_name}' "
                    f"— defaulting to .webm | trace_id={trace_id}"
                )
                extension = ".webm"

            tmp_path = os.path.join(_AUDIO_DIR, f"tmp_{uuid.uuid4().hex}{extension}")
            try:
                audio_file.save(tmp_path)
                saved_size = os.path.getsize(tmp_path) if os.path.isfile(tmp_path) else 0
                logger.info(
                    f"Voice route: audio saved | path={tmp_path} | ext={extension} "
                    f"| size_bytes={saved_size} | lang_hint={user_lang_hint} | trace_id={trace_id}"
                )
                raw_text = speech_to_text(tmp_path, language_hint=user_lang_hint)
                message = clean_stt_text(raw_text, user_lang_hint)
                logger.info(
                    f"Voice route: STT result | text_len={len(message)} "
                    f"| preview={message[:60]!r} | trace_id={trace_id}"
                )
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            logger.warning(
                f"Voice route: no audio file in request | trace_id={trace_id} | phone={phone_number}"
            )

        if len(message.strip()) < 3:
            logger.warning(
                f"Voice route: STT returned insufficient text ('{message.strip()}') "
                f"— returning retry prompt | trace_id={trace_id}"
            )
            error_menu = MENU_TEXTS.get(user_lang_hint, MENU_TEXT)
            error_prompt = "Could not understand audio, please repeat"
            if user_lang_hint != "en":
                translated_prompt = translate_from_english(error_prompt, user_lang_hint)
                if translated_prompt and translated_prompt.strip():
                    error_prompt = translated_prompt
            error_response = f"{error_prompt}\n\n{error_menu}"
            # TTS is best-effort — if it fails, text response is still returned
            error_audio = text_to_speech(error_response, language_code=user_lang_hint) if voice_audio_received else ""

            return jsonify({
                "response": error_response,
                "audio_url": error_audio or "",
                "status": "success",
                "schemes": [],
                "errors": [],
                "user_state": "voice_prompt_error",
                "trace_id": trace_id,
            }), 200

    from models.messages_model import messages_model

    messages_model.log_message(
        phone_number=phone_number,
        sender='user',
        msg_type=input_type,
        text=message,
    )

    message = sanitize_text(message)
    response_msg, new_state = handle_message(phone_number, message)

    out = {
        "user_state": new_state,
        "status": "success",
        "schemes": [],
        "errors": [],
        "fallback_used": False,
        "audio_url": "",
        "trace_id": trace_id,
    }

    if isinstance(response_msg, dict):
        out.update(response_msg)
    else:
        out["response"] = response_msg

    user_state_for_geo = None
    try:
        latest_user = user_model.get_user(phone_number) or {}
        profile = latest_user.get("profile") if isinstance(latest_user.get("profile"), dict) else {}
        user_state_for_geo = profile.get("state")
    except Exception:
        user_state_for_geo = None

    rejected_schemes = []
    if isinstance(out.get("schemes"), list):
        filtered_schemes, rejected_schemes = _final_geo_filter_schemes(out.get("schemes") or [], user_state_for_geo)
        out["schemes"] = filtered_schemes
        if rejected_schemes:
            logger.warning(
                {
                    "event": "final_geo_filter",
                    "trace_id": trace_id,
                    "user_state": user_state_for_geo,
                    "rejected_count": len(rejected_schemes),
                    "rejected_schemes": rejected_schemes,
                }
            )
    if user_state_for_geo and isinstance(out.get("schemes"), list) and not out.get("schemes") and rejected_schemes:
        out["response"] = (
            f"No matching schemes found for {user_state_for_geo}. "
            "Showing All India schemes if available."
        )

    response_text = out.get("response") or out.get("message") or ""
    if not response_text or not str(response_text).strip():
        try:
            user_doc, _ = user_model.create_or_get_user(phone_number)
            lang = (user_doc or {}).get("language") or "en"
        except Exception:
            lang = "en"
        safe_text = SAFE_EMPTY_RESPONSE_FALLBACK.get(lang, SAFE_EMPTY_RESPONSE_FALLBACK["en"])
        menu_text = MENU_TEXTS.get(lang, MENU_TEXT)
        if lang != "en":
            translated_safe = translate_from_english(SAFE_EMPTY_RESPONSE_FALLBACK["en"], lang)
            if translated_safe and translated_safe.strip():
                safe_text = translated_safe
            translated_menu = translate_from_english(MENU_TEXT, lang)
            if translated_menu and translated_menu.strip():
                menu_text = translated_menu
        response_text = f"{safe_text}\n\n{menu_text}"
        logger.error({"event": "empty_response_guard", "trace_id": trace_id, "phone": phone_number, "lang": lang})
    out["response"] = response_text
    out.pop("message", None)
    if not isinstance(out.get("schemes"), list):
        out["schemes"] = []

    if input_type == "voice" and _VOICE_ENABLED and voice_audio_received:
        user_lang = user_lang_hint
        try:
            user_doc, _ = user_model.create_or_get_user(phone_number)
            if user_doc:
                user_lang = user_doc.get("language") or "en"
        except Exception:
            pass
        # TTS is best-effort — text response is already set in out["response"]
        audio_url = text_to_speech(response_text, language_code=user_lang)
        out["audio_url"] = audio_url or ""  # empty string, never None
    elif input_type != "voice":
        out.pop("audio_url", None)

    try:
        messages_model.log_message(
            phone_number=phone_number,
            sender='bot',
            msg_type='voice' if out.get("audio_url") else 'text',
            text=response_text,
            audio_url=out.get("audio_url"),
        )
    except Exception:
        pass

    logger.info({
        "event": "request_complete",
        "trace_id": trace_id,
        "phone": phone_number,
        "state": new_state,
        "status": out.get("status"),
    })

    return jsonify(out), 200


@api_bp.route('/upload', methods=['POST'])
@limiter.limit("10/minute")
def upload():
    cleanup_old_files(os.path.join(_BASE_DIR, "frontend", "static"), max_age_seconds=300)

    """
    Phase 3: OCR Integration Wrapper hooking uploads into DB abstractions seamlessly validating profiles dynamically.
    """
    from services.ocr import extract_entities_from_ocr, extract_text_from_image

    phone_number = request.form.get('phone_number')
    if not phone_number:
        return jsonify({"status": "error", "message": "Missing phone_number"}), 400

    if 'image' not in request.files:
        abort(400, description="Missing image file in request")

    image_file = request.files['image']

    tmp_path = os.path.join(_BASE_DIR, "frontend", "static", f"tmp_ocr_{uuid.uuid4().hex}.png")
    image_file.save(tmp_path)

    extracted_raw = extract_text_from_image(tmp_path)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not extracted_raw:
        logger.error(f"OCR Failed for user {phone_number}")
        return jsonify({
            "status": "error",
            "message": "OCR failed, please upload clearer image"
        }), 200

    logger.info(f"OCR raw text for {phone_number}: {extracted_raw[:100]}...")

    ocr_result = extract_entities_from_ocr(extracted_raw)
    entities = ocr_result.get("entities", {}) if isinstance(ocr_result, dict) else {}
    logger.info(f"OCR extracted entities for {phone_number}: {ocr_result}")

    ocr_doc = {
        "phone_number": phone_number,
        "raw_text": extracted_raw,
        "entities": entities,
        "ocr_result": ocr_result,
        "timestamp": datetime.utcnow(),
        "status": "requires_human_review"
    }

    if db_client.db is not None:
        db_client.db['documents_ocr'].insert_one(ocr_doc)

        user_model.update_profile_with_priority(phone_number, entities, source="ocr")

        user, _ = user_model.create_or_get_user(phone_number)
        if user and not user.get("name") and entities.get("name"):
            user_model.update_user(phone_number, {"name": entities["name"]})

    return jsonify({
        "status": "success",
        "message": "Document processed successfully",
        "extracted_data": entities
    }), 200


@api_bp.route('/chat/feedback', methods=['POST'])
@limiter.limit("30/minute")
def chat_feedback():
    """
    Records user feedback for a specific bot response.
    Expects: phone_number, trace_id, rating (1-5), optional comment.
    """
    data = request.get_json(silent=True) or {}
    phone_number = data.get('phone_number')
    trace_id = data.get('trace_id')
    rating = data.get('rating')
    comment = data.get('comment', "")

    if not all([phone_number, trace_id, rating]):
        return jsonify({"status": "error", "message": "Missing required fields (phone_number, trace_id, rating)"}), 400

    success = feedback_model.store_feedback(phone_number, trace_id, int(rating), comment)

    if success:
        return jsonify({"status": "success", "message": "Feedback recorded"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to record feedback"}), 500
