import os
import sys
import json
import time

# Add root directory to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.orchestrator import orchestrator
from core.logger import get_logger

logger = get_logger("tests.eval_pipeline")

def run_evaluation(dataset_path: str):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    correct_category = 0
    correct_profile = 0
    
    print(f"\n--- Starting Evaluation Pipeline ({total} cases) ---\n")
    
    mock_user = {
        "phone_number": "test_user",
        "profile": {},
        "language": "en"
    }

    for i, case in enumerate(dataset, 1):
        input_text = case["input"]
        print(f"[{i}/{total}] Testing: {input_text[:50]}...")
        
        # Run through orchestrator
        # Note: We pass a fresh mock user to avoid state pollution between cases
        res = orchestrator("test_user", input_text, mock_user)
        
        entities = res.get("entities", {})
        
        # 1. Verify Category
        if "expected_category" in case:
            actual_cat = res.get("parsed", {}).get("category")
            if actual_cat == case["expected_category"]:
                correct_category += 1
            else:
                print(f"   [FAIL] Category Mismatch: Expected '{case['expected_category']}', got '{actual_cat}'")

        # 2. Verify Profile Data
        field_match = True
        for field in ["expected_age", "expected_income", "expected_state"]:
            if field in case:
                expected_val = case[field]
                actual_field = field.replace("expected_", "")
                actual_val = entities.get(actual_field)
                if actual_val != expected_val:
                    field_match = False
                    print(f"   [FAIL] Profile Mismatch ({actual_field}): Expected {expected_val}, got {actual_val}")
        
        if field_match:
            correct_profile += 1

    print("\n--- Evaluation Results ---")
    print(f"Category Accuracy: {correct_category/total:.2%}")
    print(f"Profile Extraction Accuracy: {correct_profile/total:.2%}")
    print("---------------------------\n")

if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    run_evaluation(dataset_file)
