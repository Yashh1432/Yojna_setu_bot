import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.ocr import extract_text_from_image, extract_entities_from_ocr
from services.voice import text_to_speech, speech_to_text

def create_test_image(filename="test_ocr_image.png"):
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Use default font
    try:
        font = ImageFont.load_default()
        d.text((10, 10), "Name: Rahul Sharma", fill=(0, 0, 0), font=font)
        d.text((10, 50), "Income: Rs. 50,000", fill=(0, 0, 0), font=font)
        d.text((10, 90), "Aadhaar Card", fill=(0, 0, 0), font=font)
    except Exception as e:
        print(f"Error drawing text: {e}")
    img.save(filename)
    return filename

def run_tests():
    print("=== Testing OCR ===")
    img_path = create_test_image()
    print(f"Generated test image: {img_path}")
    text = extract_text_from_image(img_path)
    print(f"OCR Extracted Text:\n{text}")
    
    entities = extract_entities_from_ocr(text)
    print(f"OCR Entities:\n{entities}")

    print("\n=== Testing TTS ===")
    test_text_en = "Hello, welcome to Yojna Setu bot. How can I help you today?"
    print(f"Input text (en): {test_text_en}")
    tts_path_en = text_to_speech(test_text_en, "en")
    print(f"TTS Output path (en): {tts_path_en}")
    
    test_text_hi = "नमस्ते, योजना सेतु बोट में आपका स्वागत है।"
    print(f"Input text (hi): {test_text_hi}")
    tts_path_hi = text_to_speech(test_text_hi, "hi")
    print(f"TTS Output path (hi): {tts_path_hi}")

    print("\n=== Testing STT ===")
    if tts_path_en:
        # Resolve to absolute path
        abs_tts_path_en = os.path.join(os.path.dirname(__file__), "frontend", tts_path_en.lstrip('/'))
        if os.path.exists(abs_tts_path_en):
            print(f"Testing STT on English audio: {abs_tts_path_en}")
            stt_en = speech_to_text(abs_tts_path_en, "en")
            print(f"STT Output (en): {stt_en}")
        else:
            print(f"Audio file not found: {abs_tts_path_en}")
    else:
        print("TTS failed, skipping STT for English")

    if tts_path_hi:
        abs_tts_path_hi = os.path.join(os.path.dirname(__file__), "frontend", tts_path_hi.lstrip('/'))
        if os.path.exists(abs_tts_path_hi):
            print(f"Testing STT on Hindi audio: {abs_tts_path_hi}")
            stt_hi = speech_to_text(abs_tts_path_hi, "hi")
            print(f"STT Output (hi): {stt_hi}")
        else:
            print(f"Audio file not found: {abs_tts_path_hi}")
    else:
        print("TTS failed, skipping STT for Hindi")

if __name__ == "__main__":
    run_tests()
