import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# 1. Absolute Path Management
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load environment variables FIRST using absolute path
load_dotenv(os.path.join(_BASE_DIR, ".env"))

# 2. Setup Central Logging
from core.logger import get_logger
logger = get_logger("run")

from core.limiter import limiter
from api.routes import api_bp
from api.error_handlers import register_error_handlers
from models.db_client import db_client

def create_app():
    # 3. Path-Aware Flask Instance
    static_dir = os.path.join(_BASE_DIR, 'frontend')
    app = Flask(__name__, static_folder=static_dir, static_url_path='/')
    CORS(app)
    
    # ── UTF-8 Enforcement: Raw Unicode in JSON + Explicit Charset ──
    app.config["JSON_AS_ASCII"] = False
    try:
        app.json.ensure_ascii = False  # Flask 2.3+
    except AttributeError:
        pass

    @app.after_request
    def add_header(response):
        if response.mimetype == 'application/json':
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    limiter.init_app(app)
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    # 4. Health Check Endpoint
    @app.route('/api/health')
    def health_check():
        db_status = "connected" if db_client.db is not None else "disconnected"
        return jsonify({
            "status": "healthy",
            "mongodb": db_status,
            "version": "3.0.0",
            "os": os.name
        }), 200
        
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    logger.info(f"YojnaSetuBot starting on port {port} (DIR: {_BASE_DIR})")
    
    # PERMANENT WINDOWS FIX: Disable reloader to prevent [WinError 10038]
    # Threaded is True by default in Flask 1.0+, which is fine.
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
