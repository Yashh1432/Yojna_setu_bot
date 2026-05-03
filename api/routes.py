from flask import Blueprint, request, jsonify, abort
import os
import uuid
from core.logger import get_logger, set_trace_id
from core.limiter import limiter
from core.sanitizer import sanitize_text
from engine.state_manager import MENU_TEXT, MENU_TEXTS, handle_message
from models.db_client import db_client
from datetime import datetime
from services.voice import speech_to_text, text_to_speech, clean_stt_text
from services.translation_service import translate_from_english
from core.cleanup import cleanup_old_files

from models.users import user_model
from models.feedback_model import feedback_model

logger = get_logger("api.routes")

api_bp = Blueprint('api', __name__)

# Fix B14: absolute base dir so paths work regardless of CWD
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUDIO_DIR = os.path.join(_BASE_DIR, "frontend", "static", "audio")

SAFE_EMPTY_RESPONSE_FALLBACK = {
    "en": "I could not complete the search safely. Please try again or provide more details.",
    "hi": "I could not complete the search safely. Please try again or provide more details.",
    "ta": "I could not complete the search safely. Please try again or provide more details.",
    "ur": "I could not complete the search safely. Please try again or provide more details.",
}

@api_bp.route('/chat', methods=['POST'])
@limiter.limit("60/minute")
def chat():
    """
    Phase 2 endpoint routing logic via state machine.
    Expects structured payload (JSON logic) OR Multipart (Voice logics)
    """
    # â”€â”€ TRACE ID: unique per request for end-to-end correlation â”€â”€
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)

    # Perform maintenance cleanup (Fix B14: absolute path)
    cleanup_old_files(_AUDIO_DIR, max_age_seconds=600)
    # 1. Flexible Payload Binding Resolving
    if request.is_json:
        data = request.get_json(silent=True) or {}
        phone_number = data.get('phone_number')
        message = data.get('message', '')
        input_type = data.get('input_type', 'text')
    else:
        # Voice pipeline upload handling
        phone_number = request.form.get('phone_number')
        input_type = request.form.get('input_type', 'text')
        message = request.form.get('message', '')

    if not phone_number:
        # User requested flexible/temporary phone numbers
        phone_number = "anonymous_user"
        logger.info({"event": "anonymous_user", "trace_id": trace_id})

    logger.info({"event": "request_start", "trace_id": trace_id, "phone": phone_number, "input_type": input_type, "message_preview": message[:50]})

    # 2. Voice Sub-System Overrides
    if input_type == "voice":
        if 'audio' in request.files:
            audio_file = request.files['audio']
            
            # Persist temporarily for processor logic securely isolated
            tmp_path = os.path.join(_AUDIO_DIR, f"tmp_{uuid.uuid4().hex}.wav")
            audio_file.save(tmp_path)
            
            # STT Extract â€” pass user's language hint for better accuracy (FIX 7)
            user_lang_hint = "en"
            try:
                from models.users import user_model as um
                u = um.get_user(phone_number)
                if u and u.get("language"):
                    user_lang_hint = u["language"]
            except Exception:
                pass
            raw_text = speech_to_text(tmp_path, language_hint=user_lang_hint)
            message = clean_stt_text(raw_text, user_lang_hint)
            
            # Cleanup securely
            try:
                os.remove(tmp_path)
            except:
                pass
                
        # 3. Explicit Protection Validating Bound Layer Limits Check!
        if len(message.strip()) < 3:
            # Trap native generation explicitly overriding standard loop executions
            error_menu = MENU_TEXTS.get(user_lang_hint, MENU_TEXT)
            error_response = f"Could not understand audio, please repeat\n\n{error_menu}"
            error_audio = text_to_speech(error_response)
            
            return jsonify({
                "response": error_response,
                "audio_url": error_audio,
                "status": "success",
                "schemes": [],
                "errors": [],
                "user_state": "voice_prompt_error"
            }), 200

    # Log User Message
    from models.messages_model import messages_model
    messages_model.log_message(
        phone_number=phone_number,
        sender='user',
        msg_type=input_type,
        text=message
    )

    # Flow logic via engine (message sanitized at routes boundary)
    message = sanitize_text(message)
    response_msg, new_state = handle_message(phone_number, message)


    out = {
        "user_state": new_state,
        "status": "success",
        "schemes": [],
        "errors": [],
        "fallback_used": False,
        "trace_id": trace_id
    }

    if isinstance(response_msg, dict):
        out.update(response_msg)
    else:
        out["response"] = response_msg

    # Canonical response contract: short chat text in `response`, cards in `schemes`.
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
    
    if input_type == "voice":
        # Retrieve user's session language for TTS
        user_lang = "en"
        try:
            from models.users import user_model
            u, _ = user_model.create_or_get_user(phone_number)
            if u:
                user_lang = u.get("language") or "en"
        except Exception:
            pass
        output_audio_path = text_to_speech(response_text, language_code=user_lang)
        out["audio_url"] = output_audio_path

    # Formative Logging
    try:
        messages_model.log_message(
            phone_number=phone_number,
            sender='bot',
            msg_type='voice' if out.get("audio_url") else 'text',
            text=response_text,
            audio_url=out.get("audio_url")
        )
    except Exception:
        pass

    logger.info({"event": "request_complete", "trace_id": trace_id, "phone": phone_number, "state": new_state, "status": out.get("status")})

    return jsonify(out), 200

@api_bp.route('/upload', methods=['POST'])
@limiter.limit("10/minute")
def upload():
    # Perform maintenance cleanup for any leftover OCR artifacts (Fix B14: absolute path)
    cleanup_old_files(os.path.join(_BASE_DIR, "frontend", "static"), max_age_seconds=300)

    """
    Phase 3: OCR Integration Wrapper hooking uploads into DB abstractions seamlessly validating profiles dynamically.
    """
    from services.ocr import extract_text_from_image, extract_entities_from_ocr
    
    phone_number = request.form.get('phone_number')
    if not phone_number:
        return jsonify({"status": "error", "message": "Missing phone_number"}), 400
        
    if 'image' not in request.files:
        abort(400, description="Missing image file in request")
        
    image_file = request.files['image']
    
    # 1. Store securely in tmp explicitly limiting size logic
    tmp_path = os.path.join(_BASE_DIR, "frontend", "static", f"tmp_ocr_{uuid.uuid4().hex}.png")
    image_file.save(tmp_path)
    
    # 2. Native OCR wrapper execution
    extracted_raw = extract_text_from_image(tmp_path)
    
    # Clean up file isolation (if we need to keep documents for admins, we can skip removal, let's keep it simple cleanup)
    try:
        os.remove(tmp_path)
    except:
        pass
        
    # 3. Handle graceful fallback bounding LLMs out of loop explicitly 
    if not extracted_raw:
        logger.error(f"OCR Failed for user {phone_number}")
        return jsonify({
            "status": "error", 
            "message": "OCR failed, please upload clearer image"
        }), 200

    logger.info(f"OCR raw text for {phone_number}: {extracted_raw[:100]}...")
        
    # 4. Extract entity structures natively
    ocr_result = extract_entities_from_ocr(extracted_raw)
    entities = ocr_result.get("entities", {}) if isinstance(ocr_result, dict) else {}
    logger.info(f"OCR extracted entities for {phone_number}: {ocr_result}")

    
    # 5. Connect mappings to Database limits successfully avoiding overwrites mapping directly back locally
    ocr_doc = {
        "phone_number": phone_number,
        "raw_text": extracted_raw,
        "entities": entities,
        "ocr_result": ocr_result,
        "timestamp": datetime.utcnow(),
        "status": "requires_human_review" # Answering the trust bounds naturally
    }
    
    if db_client.db is not None:
        db_client.db['documents_ocr'].insert_one(ocr_doc)
        
        # 5. Connect mappings with Source Priority Enforced: Manual > OCR > LLM
        from models.users import user_model
        user_model.update_profile_with_priority(phone_number, entities, source="ocr")
        
        # Check for name update separately (not in profile nested)
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

