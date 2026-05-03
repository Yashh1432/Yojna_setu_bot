import sys
import os

# Add parent dir to path so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from models.db_client import db_client
from services.embedding_service import embed_text

def run_migration():
    db = db_client.db
    if db is None:
        print("Database connection failed.")
        return

    collection = db["schemes_structured"]
    schemes = list(collection.find())
    
    if not schemes:
        print("No schemes found in database.")
        return

    print(f"Generating embeddings for {len(schemes)} schemes...")
    
    updated_count = 0
    for scheme in schemes:
        # Build embedding text if not natively present
        text = str(scheme.get("embedding_text") or "")
        
        if not text.strip():
            name = scheme.get('scheme_name') or scheme.get('name', '')
            desc = scheme.get('benefits') or scheme.get('description', '')
            cat = scheme.get('category', '')
            state = scheme.get('state', '')
            text = f"{name} {desc} {cat} {state}"

        if not text.strip():
            continue

        vector = embed_text(text)

        collection.update_one(
            {"_id": scheme["_id"]},
            {"$set": {
                "embedding_text": text,
                "embedding": vector
            }}
        )
        updated_count += 1
        print(f"  Processed: {scheme.get('scheme_name') or scheme.get('_id')}")

    print(f"Embeddings generated and saved for {updated_count} schemes.")

if __name__ == "__main__":
    run_migration()
