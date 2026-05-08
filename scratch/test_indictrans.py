import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from services.translation_service import translate_from_english, translate_to_english, _translator

print("--- INDICTRANS2 DIAGNOSTIC ---")

# 1. Check if transformers is available
try:
    import torch
    import transformers
    print(f"SUCCESS: PyTorch version: {torch.__version__}")
    print(f"SUCCESS: Transformers version: {transformers.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"FAILURE: Required libraries missing: {e}")
    sys.exit(1)

# 2. Check local_files_only setting
print(f"Setting - local_files_only: {_translator.local_files_only}")

# 3. Check model paths
print(f"EN-INDIC Model Path: {_translator.en_indic_path}")
print(f"INDIC-EN Model Path: {_translator.indic_en_path}")

# 4. Test Translation (English to Hindi)
print("\n--- TEST 1: English -> Hindi ---")
start_time = time.time()
test_text = "How can I help you today?"
result_hi = translate_from_english(test_text, "hi")
duration = time.time() - start_time

print(f"Input: {test_text}")
print(f"Output: {result_hi}")
print(f"Duration: {duration:.2f}s")

if result_hi == test_text:
    print("RESULT: Translation FAILED (returned original text).")
else:
    print("RESULT: Translation SUCCESS!")

# 5. Test Translation (Hindi -> English)
print("\n--- TEST 2: Hindi -> English ---")
start_time = time.time()
test_text_hi = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"
result_en = translate_to_english(test_text_hi, "hi")
duration = time.time() - start_time

print(f"Input: {test_text_hi}")
print(f"Output: {result_en}")
print(f"Duration: {duration:.2f}s")

if result_en == test_text_hi:
    print("RESULT: Translation FAILED (returned original text).")
else:
    print("RESULT: Translation SUCCESS!")

print("\n--- SUMMARY ---")
if result_hi != test_text and result_en != test_text_hi:
    print("IndicTrans2 is working correctly.")
else:
    print("IndicTrans2 is NOT working. It is falling back to original text.")
    print("Check logs/error.log for detailed loading errors.")
