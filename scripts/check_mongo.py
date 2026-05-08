"""
scripts/check_mongo.py
----------------------
Diagnostic script — tests MongoDB connectivity for YojnaSetuBot.
Safe to run at any time. Never prints the full URI or password.

Usage:
    python scripts/check_mongo.py
"""

import sys
import os
import time
from urllib.parse import urlparse

# ── Path setup so we can import from project root ─────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Load .env before anything else ────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    print("[.env] Loaded successfully.")
except ImportError:
    print("[.env] python-dotenv not installed — using system environment only.")

# ── Helpers ───────────────────────────────────────────────────────────────

def sanitize_uri(uri: str) -> str:
    try:
        p = urlparse(uri)
        host = p.hostname or "<unknown>"
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://<credentials-hidden>@{host}{port}{p.path}"
    except Exception:
        return "<uri-parse-error>"


def classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "dns operation timed out" in msg or "resolution lifetime expired" in msg:
        return (
            "DNS_TIMEOUT — Cannot resolve the cluster hostname.\n"
            "  → Check internet / proxy / VPN.\n"
            "  → Try: nslookup <your-cluster-hostname>\n"
            "  → If on Atlas, make sure the cluster is not paused."
        )
    if "timed out" in msg or "connection timed out" in msg:
        return (
            "CONNECT_TIMEOUT — Host resolved but connection refused/timed out.\n"
            "  → Check Atlas IP Whitelist (Network Access).\n"
            "  → Add your current IP or 0.0.0.0/0 for local testing."
        )
    if "authentication failed" in msg or "authenticationfailed" in msg:
        return (
            "AUTH_FAILED — Wrong username or password.\n"
            "  → Double-check MONGODB_URI credentials in .env."
        )
    if "nodelist" in msg or "invalid uri" in msg:
        return (
            "INVALID_URI — MONGODB_URI format is wrong.\n"
            "  → Expected: mongodb+srv://user:password@cluster.mongodb.net/dbname"
        )
    return f"UNKNOWN_ERROR — {str(exc)[:300]}"


# ── Main check ────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  YojnaSetuBot — MongoDB Connectivity Check")
    print("=" * 60)

    uri = os.getenv("MONGODB_URI", "").strip()

    # ── 1. Check URI presence ──────────────────────────────────────────
    if not uri:
        print("\n[FAIL] MONGODB_URI is NOT set in the environment/.env file.")
        print("  Fix: Add MONGODB_URI=mongodb+srv://user:pass@cluster/db to .env")
        print("\n[STATUS] Fallback mode: LOCAL (JSON datasets) — bot works fine.")
        return

    print(f"\n[OK]   MONGODB_URI is set.")
    print(f"       Sanitized URI: {sanitize_uri(uri)}")

    # ── 2. Check pymongo import ────────────────────────────────────────
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
        print("[OK]   pymongo is installed.")
    except ImportError:
        print("[FAIL] pymongo is not installed.")
        print("  Fix: pip install pymongo[srv]")
        print("\n[STATUS] Fallback mode: LOCAL — bot works fine.")
        return

    # ── 3. Attempt connection ──────────────────────────────────────────
    print("\n[...] Attempting MongoDB ping (timeout = 3 s) …")
    t0 = time.monotonic()
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
            appName="YojnaSetuBotDiag",
        )
        client.admin.command("ping")
        elapsed = time.monotonic() - t0
        print(f"[OK]   Ping succeeded in {elapsed:.2f}s.")

        db = client["welfare_chatbot"]
        col_names = db.list_collection_names()
        print(f"[OK]   Database 'welfare_chatbot' accessible.")
        print(f"       Collections found: {col_names or ['<none yet>']}")
        print("\n[STATUS] CONNECTED — MongoDB is live and working.")

    except ServerSelectionTimeoutError as exc:
        elapsed = time.monotonic() - t0
        reason = classify_error(exc)
        print(f"[FAIL] Connection failed after {elapsed:.2f}s.")
        print(f"       Reason: {reason}")
        print("\n[STATUS] Fallback mode: LOCAL — bot still works fine.")

    except Exception as exc:
        elapsed = time.monotonic() - t0
        reason = classify_error(exc)
        print(f"[FAIL] Unexpected error after {elapsed:.2f}s.")
        print(f"       Reason: {reason}")
        print("\n[STATUS] Fallback mode: LOCAL — bot still works fine.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
