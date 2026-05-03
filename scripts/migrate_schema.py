import sys
import os
import logging
import argparse
import json
from datetime import datetime
from pymongo import UpdateOne

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.db_client import db_client
from models.users import user_model
from models.data_validation import validate_scheme

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_schema")

def migrate_users(dry_run=True):
    logger.info(f"--- Starting User Migration (Schema V2) ---")
    logger.info(f"Dry Run Mode: {dry_run}")
    
    if db_client.db is None:
        logger.error("Database connection failed.")
        return

    users_collection = db_client.db['users']
    
    # 1. Expand Phase: Identify users not yet migrated (or missing schema_version)
    query = {"schema_version": {"$ne": 2}}
    total_to_migrate = users_collection.count_documents(query)
    logger.info(f"Found {total_to_migrate} users requiring migration.")
    
    if total_to_migrate == 0:
        logger.info("User migration complete!")
        return

    cursor = users_collection.find(query)
    migrated, skipped, failed = 0, 0, 0
    updates = []
    
    for doc in cursor:
        try:
            phone_number = doc.get("phone_number")
            if not phone_number:
                skipped += 1
                continue
            
            # Normalize using the existing robust method
            normalized_doc = user_model.normalize_user(doc)
            
            if not dry_run:
                # dual-write: keep conv_data, add profile, set schema_version
                updates.append(UpdateOne(
                    {"phone_number": phone_number},
                    {"$set": {
                        "profile": normalized_doc.get("profile", {}),
                        "schema_version": 2
                    }}
                ))
            
            migrated += 1
            if len(updates) >= 100 and not dry_run:
                users_collection.bulk_write(updates, ordered=False)
                updates = []
                logger.info(f"Batched {migrated}/{total_to_migrate} users...")
                
        except Exception as e:
            logger.error(f"Failed to migrate user {doc.get('phone_number', 'unknown')}: {e}")
            failed += 1

    if updates and not dry_run:
        users_collection.bulk_write(updates, ordered=False)
        
    logger.info(f"User Migration Summary: Migrated={migrated}, Skipped={skipped}, Failed={failed}")

def migrate_schemes(dry_run=True):
    logger.info(f"--- Starting Schemes Migration (Schema V2) ---")
    if db_client.db is None:
        logger.error("Database connection failed.")
        return

    schemes_coll = db_client.db['schemes_structured']
    
    json_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'final_production_schemes.json')
    if not os.path.exists(json_path):
        logger.error(f"Cannot find schemes file at {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    logger.info(f"Loaded {len(data)} schemes from JSON.")
    
    migrated, skipped, failed = 0, 0, 0
    updates = []
    
    for scheme_doc in data:
        try:
            # db constraint validation simulation
            if not validate_scheme(scheme_doc):
                logger.warning(f"Validation failed for scheme {scheme_doc.get('scheme_id', 'Unknown')}. Skipping.")
                skipped += 1
                continue
                
            if not dry_run:
                updates.append(UpdateOne(
                    {"scheme_id": scheme_doc["scheme_id"]},
                    {"$set": scheme_doc},
                    upsert=True
                ))
                
            migrated += 1
            if len(updates) >= 100 and not dry_run:
                schemes_coll.bulk_write(updates, ordered=False)
                updates = []
                
        except Exception as e:
            logger.error(f"Failed to migrate scheme {scheme_doc.get('scheme_id', 'Unknown')}: {e}")
            failed += 1

    if updates and not dry_run:
        schemes_coll.bulk_write(updates, ordered=False)
        
    logger.info(f"Schemes Migration Summary: Upserted={migrated}, Skipped={skipped}, Failed={failed}")

def generate_lockfile():
    lock = os.path.join(os.path.dirname(__file__), 'migration.lock')
    if os.path.exists(lock):
        logger.error("Migration is already running or didn't finish properly. Clear migration.lock to retry.")
        return False
    with open(lock, 'w') as f:
        f.write(datetime.utcnow().isoformat())
    return True

def clear_lockfile():
    lock = os.path.join(os.path.dirname(__file__), 'migration.lock')
    if os.path.exists(lock):
        os.remove(lock)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to DB")
    parser.add_argument("--users-only", action="store_true", help="Skip schemes migration")
    parser.add_argument("--schemes-only", action="store_true", help="Skip users migration")
    args = parser.parse_args()

    if not args.dry_run and not generate_lockfile():
        sys.exit(1)

    try:
        # Enforce indexes First
        if not args.dry_run:
            logger.info("Enforcing strict DB indexes and validation constraints...")
            db_client.ensure_indexes_and_validation()

        if not args.schemes_only:
            migrate_users(dry_run=args.dry_run)
            
        if not args.users_only:
            migrate_schemes(dry_run=args.dry_run)
            
        logger.info("Migration orchestrator finished successfully!")
    finally:
        if not args.dry_run:
            clear_lockfile()
