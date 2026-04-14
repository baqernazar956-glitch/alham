from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_caching import Cache

# تعريف قاعدة البيانات
db = SQLAlchemy()

# تعريف نظام تسجيل الدخول
login_manager = LoginManager()

# تعريف نظام Migrations
migrate = Migrate()

# تعريف CSRF Protection
csrf = CSRFProtect()

# تعريف نظام Caching
cache = Cache()

# تعريف JWT للـ API
from flask_jwt_extended import JWTManager
jwt = JWTManager()