# routes/api/__init__.py
"""
REST API Blueprint - يوفر endpoints لتطبيق Flutter
"""
from flask import Blueprint

# إنشاء الـ Blueprint الرئيسي للـ API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# استيراد وتسجيل الـ sub-blueprints
from .auth import api_auth_bp
from .books import api_books_bp
from .user import api_user_bp
from .ai import api_ai_bp

api_bp.register_blueprint(api_auth_bp)
api_bp.register_blueprint(api_books_bp)
api_bp.register_blueprint(api_user_bp)
api_bp.register_blueprint(api_ai_bp)
