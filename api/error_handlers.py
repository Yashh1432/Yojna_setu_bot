from flask import jsonify, current_app
from werkzeug.exceptions import HTTPException
import traceback
import logging

logger = logging.getLogger("api.errors")

def register_error_handlers(app):
    """
    Registers centralized error handlers for the Flask application.
    Separates HTTP-level errors from unexpected system exceptions.
    """

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle Flask-raised HTTP exceptions (400, 404, 405, etc)."""
        
        # 1. Silently handle favicon and static noise to prevent log pollution
        from flask import request
        is_static_noise = any(noise in request.path for noise in ["favicon.ico", "robots.txt"])
        
        # 2. Log based on severity
        if not is_static_noise:
            if e.code < 500:
                logger.warning(f"Client-side Exception [{e.code}]: {e.name} at {request.path} - {e.description}")
            else:
                # 5xx level but raised via abort() or Werkzeug
                logger.error(f"HTTP Server Exception [{e.code}]: {e.name} at {request.path} - {e.description}")

        # 3. Standardized JSON response
        response = {
            "status": "error",
            "message": e.description,
            "code": e.code
        }
        return jsonify(response), e.code

    @app.errorhandler(Exception)
    def handle_general_exception(e):
        """Handle all unhandled non-HTTP exceptions (Internal Server Errors)."""
        
        # 1. Detailed logging with stack trace (Crucial for debugging 500s)
        logger.error(f"CRITICAL: Unhandled system failure: {str(e)}\n{traceback.format_exc()}")
        
        # 2. Standardized JSON response (Never expose raw stack traces to the frontend)
        code = 500
        response = {
            "status": "error",
            "message": "An unexpected internal server error occurred. Please try again later.",
            "code": code
        }

        # 3. Optional debug details (Safe for development)
        if app.debug:
            response["debug_info"] = str(e)
            
        return jsonify(response), code
