import logging
import os
import json
import threading

# ── Absolute Path Management ──────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_BASE_DIR, "logs")
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# ── REQUEST CONTEXT (Thread-local) ──────────────────────────────
# Carries trace_id through the entire request lifecycle without
# needing to pass it as a function argument.
_request_context = threading.local()

def set_trace_id(trace_id: str):
    """Set the trace_id for the current request thread."""
    _request_context.trace_id = trace_id

def get_trace_id() -> str:
    """Get the trace_id for the current request thread."""
    return getattr(_request_context, "trace_id", "no-trace")


# ── STRUCTURED JSON FORMATTER ───────────────────────────────────
class StructuredFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line.
    If the log message is a dict, its keys are merged into the entry.
    The trace_id is automatically attached from thread-local context.
    """
    def format(self, record):
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "trace_id": get_trace_id(),
        }

        # If message is already a dict (structured event), merge it
        if isinstance(record.msg, dict):
            log_entry["event"] = record.msg.get("event", "log")
            log_entry.update(record.msg)
        else:
            log_entry["message"] = record.getMessage()

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def get_logger(name: str):
    logger = logging.getLogger(name)

    # If the logger doesn't have handlers yet, configure it
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Structured JSON formatter for machine-readable logs
        formatter = StructuredFormatter()

        # 1. Main Application Handler (All Logs)
        app_handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)

        # 2. Error Handler (Only ERROR and CRITICAL)
        error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # Optional: Prevent log propagation to avoid duplicate console logs
        logger.propagate = False

    return logger

# ── SYSTEM PULSE (Verification) ────────────────────────────────
_pulse_logger = get_logger("core.logger")
try:
    _pulse_logger.info({"event": "system_pulse", "msg": "Logging system initialized successfully."})
except Exception as e:
    print(f"CRITICAL: Logging failure: {e}")
