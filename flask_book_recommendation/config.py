import os
# Force reload trigger
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-fallback-key"

    # قاعدة البيانات - التعامل مع Render
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = db_url
    elif os.environ.get("DB_URL"):
        SQLALCHEMY_DATABASE_URI = os.environ.get("DB_URL")
    else:
        # SQLite محلي لتسهيل التطوير
        db_path = os.path.join(BASE_DIR, "app.db").replace("\\", "/")
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + db_path

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # إعدادات Caching
    CACHE_TYPE = 'RedisCache' if os.environ.get('REDIS_URL') else 'SimpleCache'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_DEFAULT_TIMEOUT = 600
    
    # إعدادات CSRF و Session
    WTF_CSRF_ENABLED = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # إعدادات Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(BASE_DIR, "app.log")
    
    # إعدادات JWT للـ API
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or SECRET_KEY
    from datetime import timedelta
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)