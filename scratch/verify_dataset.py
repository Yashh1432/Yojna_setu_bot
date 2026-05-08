import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    from engine.orchestrator import load_scheme_dataset, DATASET_PATHS
    from services.embedding_service import semantic_match
    
    print("--- DATASET CHECK ---")
    print("Checking dataset paths...")
    for p in DATASET_PATHS:
        exists = p.exists()
        print(f"Path: {p} - Exists: {exists}")
        
    print("\nAttempting to load dataset using load_scheme_dataset()...")
    data = load_scheme_dataset()
    if data:
        print(f"SUCCESS: Loaded {len(data)} schemes.")
        if len(data) > 0:
            print(f"Sample scheme: {data[0].get('scheme_name', 'No Name')}")
    else:
        print("FAILURE: Dataset is empty or could not be loaded.")

    print("\n--- EMBEDDING CHECK ---")
    print("Attempting to load embedding model and perform semantic match...")
    score = semantic_match("farmer", "agriculture worker")
    print(f"Semantic match score ('farmer' vs 'agriculture worker'): {score:.4f}")
    if score > 0.5:
        print("SUCCESS: Embedding model is working.")
    else:
        print("FAILURE: Embedding model returned low score or failed to load.")

except Exception as e:
    print(f"ERROR: An exception occurred: {e}")
    import traceback
    traceback.print_exc()
