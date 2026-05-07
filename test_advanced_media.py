import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.ocr import extract_entities_from_ocr
from services.voice import text_to_speech, speech_to_text

def run_advanced_tests():
    print("=== 1. Testing OCR Entity Extraction with Different Parameters ===")
    
    # Test 1: Empty text with different languages (should return localized error)
    print("\n--- OCR Test 1: Empty Text ---")
    print("English:", extract_entities_from_ocr("", language_code="en")["error_msg"])
    print("Hindi:", extract_entities_from_ocr("", language_code="hi")["error_msg"])
    print("Gujarati:", extract_entities_from_ocr("", language_code="gu")["error_msg"])
    
    # Test 2: Different Simulated OCR Texts
    print("\n--- OCR Test 2: Synthetic Documents ---")
    doc_aadhaar = "Name: Rahul Sharma\nDOB: 12/05/1990\nAadhaar: 1234 5678 9012"
    doc_income = "Income Certificate\nThis is to certify that Name : Sunita Devi has an annual income of Rs. 45,000 only."
    doc_pan = "INCOME TAX DEPARTMENT\nPermanent Account Number\nName: Amit Kumar"
    
    print("Aadhaar Extract:", extract_entities_from_ocr(doc_aadhaar))
    print("Income Extract:", extract_entities_from_ocr(doc_income))
    print("PAN Extract:", extract_entities_from_ocr(doc_pan))

    print("\n=== 2. Testing TTS with Different Languages ===")
    
    texts = {
        "en": "I am looking for a farming scheme.",
        "hi": "मैं खेती की योजना ढूंढ रहा हूं।",
        "gu": "હું ખેતીની યોજના શોધી રહ્યો છું.",
        "mr": "मी शेतीची योजना शोधत आहे."
    }
    
    tts_paths = {}
    for lang, text in texts.items():
        print(f"Generating TTS for [{lang}]: {text.encode('utf-8')}")
        path = text_to_speech(text, lang)
        tts_paths[lang] = path
        print(f"Output path: {path}")

    print("\n=== 3. Testing STT with Different Language Hints ===")
    
    for lang, path in tts_paths.items():
        if not path:
            continue
        
        abs_path = os.path.join(os.path.dirname(__file__), "frontend", path.lstrip('/'))
        if not os.path.exists(abs_path):
            print(f"Audio file not found: {abs_path}")
            continue
            
        print(f"\n--- Testing Audio: {lang} ---")
        
        # Test STT with CORRECT hint
        stt_correct = speech_to_text(abs_path, language_hint=lang)
        print(f"STT Output (Hint: {lang}): {stt_correct.encode('utf-8')}")
        
        # Test STT with WRONG hint
        wrong_hint = "en" if lang != "en" else "hi"
        stt_wrong = speech_to_text(abs_path, language_hint=wrong_hint)
        print(f"STT Output (Hint: {wrong_hint}): {stt_wrong.encode('utf-8')}")

if __name__ == "__main__":
    run_advanced_tests()
