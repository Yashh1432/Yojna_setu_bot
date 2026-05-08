import sys
import os

# Ensure we can import from services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from services.translation_service import translate_from_english_with_meta, translate_to_english_with_meta

def run_tests():
    print("Testing Translation Fallback & Cache Detection...\n")
    
    # 1. English to Gujarati
    print("Test 1: English to Gujarati")
    text_en = "Hello, how can I help you with government schemes?"
    out, meta = translate_from_english_with_meta(text_en, "gu")
    print(f"Input: {text_en}")
    print(f"Output: {out}")
    print(f"Meta: {meta}")
    if meta.get("translation_failed"):
        print("Result: FALLBACK USED (Model failed to load/translate)\n")
    else:
        print("Result: INDICTRANS2 LOADED SUCCESSFULLY\n")

    # 2. Gujarati to English
    print("Test 2: Gujarati to English")
    text_gu = "મને સરકારી યોજનાઓ વિશે માહિતી આપો"
    out, meta = translate_to_english_with_meta(text_gu, "gu")
    print(f"Input: {text_gu}")
    print(f"Output: {out}")
    print(f"Meta: {meta}")
    if meta.get("translation_failed"):
        print("Result: FALLBACK USED (Model failed to load/translate)\n")
    else:
        print("Result: INDICTRANS2 LOADED SUCCESSFULLY\n")

    # 3. English to Hindi
    print("Test 3: English to Hindi")
    text_en_hi = "This is a test for Hindi translation."
    out, meta = translate_from_english_with_meta(text_en_hi, "hi")
    print(f"Input: {text_en_hi}")
    print(f"Output: {out}")
    print(f"Meta: {meta}")
    if meta.get("translation_failed"):
        print("Result: FALLBACK USED (Model failed to load/translate)\n")
    else:
        print("Result: INDICTRANS2 LOADED SUCCESSFULLY\n")

if __name__ == "__main__":
    run_tests()
