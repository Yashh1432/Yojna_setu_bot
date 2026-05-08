"""
MongoDB client for YojnaSetuBot.
---------------------------------
Fallback behaviour
  If MongoDB is unavailable for ANY reason, the bot continues in local mode.
  Local mode is fully functional — it reads from JSON scheme datasets.

Common failure reasons and how to fix them
  1. MONGODB_URI not set in .env
       → Add MONGODB_URI=mongodb+srv://... to your .env file.

  2. DNS timeout ("The DNS operation timed out")
       → Your network cannot resolve the Atlas cluster hostname.
       → Check your internet connection or corporate proxy/firewall settings.
       → Verify the hostname by: nslookup <cluster-hostname>

  3. IP whitelist error (Atlas)
       → Go to Atlas → Network Access → Add your current IP address.
       → If deploying on a server, add the server's public IP or allow 0.0.0.0/0
         for testing (never in production).

  4. Atlas cluster paused
       → Log in to MongoDB Atlas, open the project, and resume the cluster.
       → Free-tier clusters pause automatically after 60 days of inactivity.

  5. Wrong username / password in URI
       → Atlas returns an AuthenticationFailed error.
       → Double-check MONGODB_URI credentials.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlparse

from core.logger import get_logger

logger = get_logger("models.db_client")

try:
    from pymongo import MongoClient
    from pymongo.errors import (
        ConfigurationError,
        ConnectionFailure,
        OperationFailure,
        ServerSelectionTimeoutError,
    )
    _HAS_PYMONGO = True
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore[assignment]
    _HAS_PYMONGO = False
    # Stub exception types so except clauses don't break
    class _StubExc(Exception):
        pass
    ConfigurationError = ConnectionFailure = OperationFailure = ServerSelectionTimeoutError = _StubExc  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_uri(uri: str) -> str:
    """Return the URI with credentials stripped — safe to log."""
    try:
        parsed = urlparse(uri)
        # Remove user:password from netloc
        host = parsed.hostname or "<unknown-host>"
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or ""
        return f"{parsed.scheme}://<credentials-hidden>@{host}{port}{path}"
    except Exception:
        return "<uri-parse-error>"


def _classify_error(exc: Exception) -> str:
    """Return a short human-readable reason for a connection failure."""
    msg = str(exc).lower()
    if "dns operation timed out" in msg or "resolution lifetime expired" in msg:
        return (
            "DNS_TIMEOUT — Cannot resolve the cluster hostname. "
            "Check your internet connection or DNS settings."
        )
    if "timed out" in msg or "connection timed out" in msg:
        return (
            "CONNECT_TIMEOUT — Host resolved but connection refused/timed out. "
            "Check Atlas IP Whitelist or firewall rules."
        )
    if "authentication failed" in msg or "authenticationfailed" in msg:
        return "AUTH_FAILED — Wrong username or password in MONGODB_URI."
    if "nodelist" in msg or "invalid uri" in msg:
        return "INVALID_URI — MONGODB_URI format is incorrect."
    if "ssl" in msg or "tls" in msg:
        return "TLS_ERROR — SSL/TLS handshake failed."
    if "network error" in msg:
        return "NETWORK_ERROR — General network failure."
    return f"UNKNOWN — {str(exc)[:200]}"


# ---------------------------------------------------------------------------
# DBClient
# ---------------------------------------------------------------------------

class DBClient:
    """
    Thin wrapper around PyMongo with safe fallback to local mode.

    Usage:
        from models.db_client import db_client
        if db_client.is_connected():
            db_client.db["collection"].find_one(...)
    """

    def __init__(self) -> None:
        self.uri: str | None = os.getenv("MONGODB_URI")
        self.client: MongoClient | None = None
        self.db = None
        self._local_mode: bool = True
        self._last_failure_reason: str = ""
        self._last_log_ts: float = 0.0          # noise-reduction timestamp
        self._log_cooldown_sec: float = 60.0    # suppress duplicate warnings

        self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Return True if MongoDB is live and the DB handle is available."""
        return self.db is not None and not self._local_mode

    def reconnect(self) -> bool:
        """
        Attempt to re-establish the MongoDB connection.
        Safe to call at any time — never raises.
        Returns True if reconnection succeeded.
        """
        logger.info("Attempting MongoDB reconnect …")
        self.client = None
        self.db = None
        self._local_mode = True
        self._connect()
        return self.is_connected()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Try to connect once, with a hard cap of ~3 s, then fall back."""
        if not self.uri:
            self._warn_once(
                "MONGODB_URI is not set in environment/.env. "
                "Running in local mode (JSON dataset). "
                "To use MongoDB, add MONGODB_URI=mongodb+srv://... to your .env file."
            )
            return

        if not _HAS_PYMONGO:
            self._warn_once(
                "pymongo is not installed. Running in local mode. "
                "Install with: pip install pymongo[srv]"
            )
            return

        sanitized = _sanitize_uri(self.uri)
        logger.info(f"Connecting to MongoDB: {sanitized} (timeout=3s) …")

        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
                socketTimeoutMS=5000,
                retryWrites=True,
                retryReads=True,
                appName="YojnaSetuBot",
            )
            # Ping forces an actual network round-trip
            self.client.admin.command("ping")
            self.db = self.client["welfare_chatbot"]
            self._local_mode = False
            self._last_failure_reason = ""
            self.ensure_scheme_indexes()
            logger.info(f"MongoDB connected successfully → {sanitized}")

        except ServerSelectionTimeoutError as exc:
            reason = _classify_error(exc)
            self._handle_failure(sanitized, reason)

        except ConfigurationError as exc:
            reason = f"CONFIG_ERROR — {exc}"
            self._handle_failure(sanitized, reason)

        except OperationFailure as exc:
            reason = _classify_error(exc)
            self._handle_failure(sanitized, reason)

        except Exception as exc:
            reason = _classify_error(exc)
            self._handle_failure(sanitized, reason)

    def _handle_failure(self, sanitized_uri: str, reason: str) -> None:
        self.client = None
        self.db = None
        self._local_mode = True
        self._last_failure_reason = reason

        self._warn_once(
            f"MongoDB connection failed — falling back to local mode.\n"
            f"  Host (sanitized): {sanitized_uri}\n"
            f"  Reason: {reason}\n"
            "  The bot will continue using local JSON datasets. "
            "State will not be persisted to the cloud."
        )

    def _warn_once(self, message: str) -> None:
        """Log a warning, but suppress duplicates within the cooldown window."""
        now = time.monotonic()
        if now - self._last_log_ts >= self._log_cooldown_sec:
            logger.warning(message)
            self._last_log_ts = now

    # ------------------------------------------------------------------
    # Schema helpers (unchanged API)
    # ------------------------------------------------------------------

    def ensure_scheme_indexes(self) -> None:
        if self.db is None:
            return
        try:
            col = self.db["schemes_structured"]
            col.create_index("normalized_category", background=True)
            col.create_index("category", background=True)
            col.create_index("state", background=True)
            col.create_index("scheme_name", background=True)
        except Exception as exc:
            logger.warning(f"Failed to ensure scheme indexes: {exc}")

    def ensure_indexes_and_validation(self) -> None:
        self.ensure_scheme_indexes()


# Module-level singleton — imported everywhere else in the codebase
db_client = DBClient()
