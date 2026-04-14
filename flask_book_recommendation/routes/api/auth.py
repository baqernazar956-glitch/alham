# routes/api/auth.py
"""
API Authentication endpoints - JWT-based login/register
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, 
    jwt_required, 
    get_jwt_identity,
    get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
from ...extensions import db, csrf
from ...models import User, UserPreference

api_auth_bp = Blueprint('api_auth', __name__, url_prefix='/auth')

# تخزين الـ tokens الملغاة (في الإنتاج استخدم Redis)
blacklisted_tokens = set()


@api_auth_bp.route('/register', methods=['POST'])
@csrf.exempt
def register():
    """
    تسجيل مستخدم جديد
    POST /api/auth/register
    Body: {"name": "...", "email": "...", "password": "..."}
    Returns: {"success": true, "token": "...", "user": {...}}
    """
    data = request.get_json() or {}
    
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    
    # التحقق من الحقول المطلوبة
    if not name or not email or not password:
        return jsonify({
            'success': False,
            'error': 'جميع الحقول مطلوبة (name, email, password)'
        }), 400
    
    # التحقق من عدم وجود المستخدم
    if User.query.filter_by(email=email).first():
        return jsonify({
            'success': False,
            'error': 'البريد الإلكتروني مسجل مسبقاً'
        }), 409
    
    # إنشاء المستخدم
    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        onboarding_completed=False
    )
    db.session.add(user)
    db.session.commit()
    
    # إنشاء JWT token
    token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'onboarding_completed': user.onboarding_completed
        }
    }), 201


@api_auth_bp.route('/login', methods=['POST'])
@csrf.exempt
def login():
    """
    تسجيل الدخول
    POST /api/auth/login
    Body: {"email": "...", "password": "..."}
    Returns: {"success": true, "token": "...", "user": {...}}
    """
    data = request.get_json() or {}
    
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    
    if not email or not password:
        return jsonify({
            'success': False,
            'error': 'البريد وكلمة المرور مطلوبان'
        }), 400
    
    # البحث عن المستخدم
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({
            'success': False,
            'error': 'البريد الإلكتروني غير مسجل'
        }), 404
    
    # التحقق من كلمة المرور
    try:
        if not check_password_hash(user.password_hash, password):
            return jsonify({
                'success': False,
                'error': 'كلمة المرور غير صحيحة'
            }), 401
    except Exception:
        return jsonify({
            'success': False,
            'error': 'خطأ في تنسيق كلمة المرور المخزنة'
        }), 500
    
    # إنشاء JWT token
    token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'onboarding_completed': user.onboarding_completed
        }
    })


@api_auth_bp.route('/logout', methods=['POST'])
@csrf.exempt
@jwt_required()
def logout():
    """
    تسجيل الخروج (إلغاء الـ token)
    POST /api/auth/logout
    Headers: Authorization: Bearer <token>
    """
    jti = get_jwt()['jti']
    blacklisted_tokens.add(jti)
    
    return jsonify({
        'success': True,
        'message': 'تم تسجيل الخروج بنجاح'
    })


@api_auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """
    معلومات المستخدم الحالي
    GET /api/auth/me
    Headers: Authorization: Bearer <token>
    """
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    
    if not user:
        return jsonify({
            'success': False,
            'error': 'المستخدم غير موجود'
        }), 404
    
    # جلب اهتمامات المستخدم
    preferences = UserPreference.query.filter_by(user_id=user.id).all()
    interests = [p.topic for p in preferences]
    
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'onboarding_completed': user.onboarding_completed,
            'interests': interests,
            'created_at': user.created_at.isoformat() if user.created_at else None
        }
    })


@api_auth_bp.route('/onboarding', methods=['POST'])
@jwt_required()
def complete_onboarding():
    """
    إكمال الـ onboarding واختيار الاهتمامات
    POST /api/auth/onboarding
    Body: {"interests": ["Programming", "AI", "Fiction"]}
    """
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    
    if not user:
        return jsonify({'success': False, 'error': 'المستخدم غير موجود'}), 404
    
    data = request.get_json() or {}
    interests = data.get('interests', [])
    
    if len(interests) < 3:
        return jsonify({
            'success': False,
            'error': 'يرجى اختيار 3 اهتمامات على الأقل'
        }), 400
    
    # حذف الاهتمامات القديمة
    UserPreference.query.filter_by(user_id=user.id).delete()
    
    # إضافة الاهتمامات الجديدة
    for interest in interests:
        pref = UserPreference(
            user_id=user.id,
            topic=interest,
            weight=100.0
        )
        db.session.add(pref)
    
    user.onboarding_completed = True
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'تم حفظ الاهتمامات بنجاح'
    })
