import os
import re
import logging
from PIL import Image

logger = logging.getLogger("services.ocr")

# 1. Absolute Path Management
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False
    return _nlp if _nlp is not False else None

def _get_pytesseract():
    try:
        import pytesseract
        tess_path = os.getenv("TESSERACT_CMD")
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
        return pytesseract
    except ImportError:
        return None

def clean_ocr_text(text: str) -> str:
    """Standardizes OCR text without destructive normalization."""
    if not text: return ""
    return text.strip()

def _extract_digits(text: str) -> str:
    """Safely normalizes digits for numeric fields only (Fix Rule 10)."""
    if not text: return ""
    # Map O to 0 only when in a potential numeric context
    return text.replace("O", "0").replace("o", "0").replace(",", "").strip()

def extract_text_from_image(file_path: str) -> str:
    try:
        tess = _get_pytesseract()
        if not tess: return ""
        raw_text = tess.image_to_string(Image.open(file_path))
        return clean_ocr_text(raw_text)
    except Exception as e:
        logger.error(f"OCR Exception: {e}")
        return ""

def extract_entities_from_ocr(text: str, language_code: str = "en") -> dict:
    entities = {"name": None, "income": None, "document_type": "Unknown"}
    
    if not text:
        msg_map = {
            "hi": "माफ़ कीजिये, मैं दस्तावेज़ को पढ़ नहीं पाया। कृपया साफ़ फोटो भेजें।",
            "gu": "દિલગીર છું, હું દસ્તાવેજ વાંચી શક્યો નથી. મહેરબાની કરીને સ્પષ્ટ ફોટો મોકલો.",
            "en": "I'm sorry, I couldn't read the document. Please send a clearer photo."
        }
        return {
            "entities": entities, "source": "ocr", "status": "error",
            "error_msg": msg_map.get(language_code, msg_map["en"])
        }

    # Income Detection (Harden Rule 10)
    # Pattern: Look for currency symbols or keywords followed by numbers and optional multipliers (k, lakh)
    income_match = re.search(r'(income|salary|rs\.?|₹)\s*[:\-]?\s*([\dOo,]+)', text, re.IGNORECASE)
    if income_match:
        try: 
            raw_val = _extract_digits(income_match.group(2))
            entities["income"] = int(re.sub(r'[^\d]', '', raw_val))
        except: pass

    # Document Type
    if "aadhaar" in text or "uid" in text: entities["document_type"] = "Aadhaar"
    elif "pan" in text or "permanent account number" in text: entities["document_type"] = "PAN Card"
    elif "income certificate" in text or "tahsildar" in text: entities["document_type"] = "Income Certificate"
        
    # Name Detection
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text.title())
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text) > 2:
                entities["name"] = ent.text.title()
                break
                
    if entities["name"] is None:
        name_match = re.search(r'name\s*[:\-]?\s*([a-z\s]+)(?:\n|\d|$)', text, re.IGNORECASE)
        if name_match:
            extracted = name_match.group(1).strip()
            if len(extracted) > 2: entities["name"] = extracted.title()
            
    return {"entities": entities, "source": "ocr", "status": "success"}
