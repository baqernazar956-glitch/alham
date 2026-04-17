# app/__init__.py أو flask_book_recommendation/__init__.py
import sys
import io
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import logging
# Force reload trigger v2
from logging.handlers import RotatingFileHandler
# Force reload trigger v3
from flask import Flask, redirect, url_for, jsonify, request
from flask_cors import CORS
from flask_compress import Compress
from .config import Config
from .extensions import db, login_manager, migrate, csrf, cache, jwt
from .models import User
from .routes.main import main_bp
from .routes.auth import auth_bp
from .routes.preferences import prefs_bp
from .routes.my_google_books import google_bp
from .routes.explore import explore_bp
from .routes.public import public_bp
from .routes.api import api_bp
from .routes.onboarding import onboarding_bp
from .routes.admin import admin_bp
from .routes.realtime_api import realtime_bp



def setup_logging(app):
    """إعداد نظام Logging للتطبيق"""
    if not app.debug:

        # إنشاء ملف log مع rotation
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.addHandler(file_handler)
        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.debug('Application startup')


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # تهيئة Extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    cache.init_app(app)
    jwt.init_app(app)  # JWT للـ API
    Compress(app)  # ⚡ Gzip/Brotli compression for all responses
    
    # Custom Jinja2 Tests
    app.jinja_env.tests['contains'] = lambda container, item: item in container if container else False
    
    # تفعيل CORS للـ API فقط (للسماح لـ Flutter بالاتصال)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # إعداد Logging
    setup_logging(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))
    
    @login_manager.unauthorized_handler
    def unauthorized():
        return redirect(url_for("auth.login"))

    # تسجيل المسارات (Blueprints)
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(prefs_bp)
    app.register_blueprint(google_bp)
    app.register_blueprint(explore_bp)   # صفحة الاستكشاف
    app.register_blueprint(public_bp)
    
    # تسجيل REST API Blueprint
    # إعفاء API من CSRF (يستخدم JWT بدلاً منه)
    csrf.exempt(api_bp)
    app.register_blueprint(api_bp)
    
    # Onboarding
    csrf.exempt(onboarding_bp) # API also uses JWT/Token often
    app.register_blueprint(onboarding_bp)
    
    app.register_blueprint(admin_bp)

    # Real-time interaction tracking & dynamic feed
    csrf.exempt(realtime_bp)
    app.register_blueprint(realtime_bp)

    # Home page -> Explore
    # @app.route("/")
    # def index():
    #     return redirect(url_for("explore.index"))

    # فحص سريع لحالة السيرفر
    @app.route("/ping")
    def ping():
        return jsonify(status="ok")

    # ⚡ Performance: Static files caching + smart response headers
    @app.after_request
    def add_performance_headers(response):
        # Cache static assets for 1 week
        if request.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'public, max-age=604800'
        # Cache images from external sources
        if response.content_type and 'image' in response.content_type:
            response.headers['Cache-Control'] = 'public, max-age=86400'
        return response

    # معالجة الأخطاء
    @app.errorhandler(404)
    def not_found(e):
        app.logger.warning(f"404 error: {request.url}")
        return "الصفحة غير موجودة، جرّب / أو /explore", 404
    
    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error(f"500 error: {e}", exc_info=True)
        db.session.rollback()
        return "حدث خطأ داخلي. يرجى المحاولة لاحقاً.", 500

    # إنشاء الجداول (فقط في حالة عدم وجود migrations)
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            app.logger.warning(f"Could not create tables (may already exist): {e}")

    # Start the Daily Embedding Updater Scheduler
    try:
        from training.scheduler import start_scheduler
        start_scheduler(app)
    except Exception as e:
        app.logger.warning(f"Could not start scheduler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 🧠 Initialize Unified Neural Recommendation Pipeline
    # Models loaded ONCE at startup — never reloaded
    # ═══════════════════════════════════════════════════════════════
    try:
        from ai_book_recommender.unified_pipeline import get_unified_engine
        unified_engine = get_unified_engine()
        unified_engine.flask_app = app
        app.logger.info("[SUCCESS] UnifiedRecommendationPipeline loaded at startup")
    except Exception as e:
        app.logger.warning(f"[WARNING] Could not initialize UnifiedRecommendationPipeline: {e}")

    return app


# For Gunicorn in production (Render) or local testing
# Usage: gunicorn flask_book_recommendation.app:app
if __name__ == "__main__" or os.environ.get("FLASK_RUN_FROM_CLI") == "true":
    app = create_app()
else:
    # When imported by unified_server.py, we don't want a module-level 'app' 
    # that independently calls create_app() at import time.
    app = None
