import os
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ..extensions import db, csrf, cache
from ..models import User, UserPreference, BookReview, BookStatus, UserBookView, Book

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# صفحة تسجيل الدخول
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Email not found", "error")
            return redirect(url_for("auth.login"))

        try:
            ok = check_password_hash(user.password_hash, password)
        except Exception:
            flash("Stored password invalid format. Recreate user.", "error")
            return redirect(url_for("auth.login"))

        if not ok:
            flash("Wrong password", "error")
            return redirect(url_for("auth.login"))

        login_user(user)

        # مسح الكاش المتعلق بهذا المستخدم فقط (أسرع بكثير من cache.clear())
        try:
            # بدلاً من مسح كل الكاش، نمسح فقط مفاتيح المستخدم الحالي
            cache.delete(f'user_{user.id}')
            cache.delete(f'recommendations_{user.id}')
            cache.delete(f'explore_{user.id}')
        except Exception:
            pass

        # إذا لم يكمل الـ onboarding، وجهه لصفحة الاهتمامات
        if not user.onboarding_completed:
            return redirect(url_for("auth.onboarding"))
        
        return redirect(url_for("explore.index"))

    return render_template("login.html")

# صفحة إنشاء الحساب
@auth_bp.route("/register", methods=["GET", "POST"])
@csrf.exempt  # استثناء مؤقت لحل مشكلة التسجيل
def register():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not name or not email or not password:
            flash("All fields are required", "error")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists", "error")
            return redirect(url_for("auth.register"))

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            onboarding_completed=False
        )
        db.session.add(user)
        db.session.commit()
        
        # تسجيل دخول تلقائي وتوجيه لصفحة الاهتمامات
        login_user(user)
        return redirect(url_for("auth.onboarding"))

    return render_template("register.html")

# صفحة اختيار الاهتمامات (Onboarding)
@auth_bp.route("/onboarding", methods=["GET", "POST"])
@csrf.exempt  # استثناء من CSRF لأن المستخدم مسجل دخول بالفعل
@login_required
def onboarding():
    # إذا أكمل الـ onboarding، وجهه للصفحة الرئيسية
    if current_user.onboarding_completed:
        return redirect(url_for("explore.index"))
    
    if request.method == "POST":
        interests = request.form.getlist("interests")
        
        if len(interests) < 3:
            flash("يرجى اختيار 3 اهتمامات على الأقل", "error")
            return redirect(url_for("auth.onboarding"))
        
        # ── 1. Clear previous selections ──
        from ..models import UserGenre, Genre
        UserGenre.query.filter_by(user_id=current_user.id).delete()
        UserPreference.query.filter_by(user_id=current_user.id).delete()
        
        # ── 2. Save to both UserGenre AND UserPreference ──
        all_genres = {g.name.lower(): g for g in Genre.query.all()}
        selected_genre_names = []
        
        for interest in interests:
            term = interest.strip()
            if not term:
                continue
            
            term_lower = term.lower()
            selected_genre_names.append(term)
            
            # Save as UserGenre if it matches a known genre
            if term_lower in all_genres:
                genre = all_genres[term_lower]
                db.session.add(UserGenre(user_id=current_user.id, genre_id=genre.id))
            
            # Always save as UserPreference with high weight
            existing_pref = UserPreference.query.filter_by(
                user_id=current_user.id, topic=term
            ).first()
            if not existing_pref:
                db.session.add(UserPreference(
                    user_id=current_user.id,
                    topic=term,
                    weight=150.0  # Explicit selection = highest priority
                ))
        
        # ── 3. Mark onboarding as completed ──
        current_user.onboarding_completed = True
        db.session.commit()
        
        # ── 4. Build User Embedding (cold-start initialization) ──
        try:
            from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
            user_embedding_manager.initialize_from_interests(current_user.id, selected_genre_names)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error initializing user embedding: {e}")
        
        # ── 5. Seed books from Google Books in background ──
        try:
            from ..interest_seeder import seed_books_for_interests
            app = current_app._get_current_object()
            seed_books_for_interests(current_user.id, selected_genre_names, app=app)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error seeding books: {e}")
        
        # ── 6. Clear user-specific caches only (much faster than cache.clear()) ──
        try:
            cache.delete(f'user_{current_user.id}')
            cache.delete(f'recommendations_{current_user.id}')
            cache.delete(f'explore_{current_user.id}')
        except Exception:
            pass
        
        # Clear pipeline neural cache for user
        try:
            from ai_book_recommender.unified_pipeline import get_unified_engine
            engine = get_unified_engine()
            engine.clear_user_cache(current_user.id)
        except Exception:
            pass
        
        flash("مرحباً! تم حفظ اهتماماتك بنجاح 🎉", "success")
        return redirect(url_for("explore.index"))
    
    return render_template("onboarding.html")

# تسجيل خروج
@auth_bp.route("/logout")
@login_required
def logout():
    # مسح الكاش المتعلق بهذا المستخدم فقط قبل تسجيل الخروج (أسرع)
    user_id = current_user.id
    try:
        cache.delete(f'user_{user_id}')
        cache.delete(f'recommendations_{user_id}')
        cache.delete(f'explore_{user_id}')
    except Exception:
        pass
    
    # مسح كاش pipeline العصبي للمستخدم الحالي
    try:
        from ai_book_recommender.unified_pipeline import get_unified_engine
        engine = get_unified_engine()
        engine.clear_user_cache(user_id)
    except Exception:
        pass
    
    logout_user()
    return redirect(url_for("auth.login"))

# صفحة الملف الشخصي
@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        bio = (request.form.get("bio") or "").strip()
        reading_goal = request.form.get("reading_goal")
        current_password = request.form.get("current_password") or ""
        new_password = request.form.get("new_password") or ""
        
        # تحديث الاسم
        if name and name != current_user.name:
            current_user.name = name
        
        # تحديث النبذة الشخصية
        current_user.bio = bio
        
        # تحديث هدف القراءة
        if reading_goal:
            try:
                current_user.reading_goal = int(reading_goal)
            except ValueError:
                pass
        
        # معالجة صورة الملف الشخصي
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{current_user.id}_{uuid.uuid4().hex[:8]}.{ext}"
                upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'profiles')
                os.makedirs(upload_folder, exist_ok=True)
                if current_user.profile_picture:
                    old_path = os.path.join(current_app.root_path, 'static', current_user.profile_picture.lstrip('/static/'))
                    if os.path.exists(old_path):
                        try: os.remove(old_path)
                        except: pass
                filepath = os.path.join(upload_folder, filename)
                file.save(filepath)
                current_user.profile_picture = f"/static/uploads/profiles/{filename}"
        
        # تغيير كلمة المرور
        if new_password:
            if not current_password:
                flash("أدخل كلمة المرور الحالية", "error")
                return redirect(url_for("auth.profile"))
            if not check_password_hash(current_user.password_hash, current_password):
                flash("كلمة المرور الحالية غير صحيحة", "error")
                return redirect(url_for("auth.profile"))
            current_user.password_hash = generate_password_hash(new_password)
        
        # تحديث الرتبة بناءً على الإنجازات
        finished_count = BookStatus.query.filter_by(user_id=current_user.id, status='finished').count()
        if finished_count >= 50: current_user.rank = "Legendary Reader"
        elif finished_count >= 20: current_user.rank = "Elite Bibliophile"
        elif finished_count >= 10: current_user.rank = "Avid Reader"
        elif finished_count >= 5: current_user.rank = "Dedicated Explorer"
        else: current_user.rank = "Novice Reader"
        
        db.session.commit()
        flash("تم تحديث معلومات الحساب بنجاح ✅", "success")
        return redirect(url_for("auth.profile"))
    
    # جلب المراجعات الخاصة بالمستخدم
    user_reviews = current_user.reviews.order_by(BookReview.created_at.desc()).limit(6).all()
    
    # === تحديث سلسلة النشاط (Reading Streak) ===
    today = datetime.utcnow().date()
    if not current_user.last_active_date:
        current_user.current_streak = 1
        current_user.last_active_date = today
    else:
        last_active = current_user.last_active_date
        delta = (today - last_active).days
        
        if delta == 1:
            # النشاط كان بالأمس، نزيد السلسلة
            current_user.current_streak += 1
            current_user.last_active_date = today
        elif delta > 1:
            # انقطاع النشاط لأكثر من يوم، نعيد السلسلة إلى 1
            current_user.current_streak = 1
            current_user.last_active_date = today
        # إذا كان delta == 0 (النشاط اليوم)، لا نفعل شيئاً
    
    db.session.commit()
    
    # === إحصائيات القراءة ===
    total_books = BookStatus.query.filter_by(user_id=current_user.id).count()
    books_finished = BookStatus.query.filter_by(user_id=current_user.id, status='finished').count()
    books_reading = BookStatus.query.filter_by(user_id=current_user.id, status='reading').count()
    books_later = BookStatus.query.filter_by(user_id=current_user.id, status='later').count()
    books_favorite = BookStatus.query.filter_by(user_id=current_user.id, status='favorite').count()
    total_reviews = current_user.reviews.count()
    
    # عدد الكتب المشاهدة
    total_views = UserBookView.query.filter_by(user_id=current_user.id).count()
    
    # حساب مدة العضوية
    member_since = current_user.created_at or datetime.utcnow()
    days_member = (datetime.utcnow() - member_since).days
    
    # آخر الكتب المشاهدة
    recent_views = (
        db.session.query(UserBookView, Book)
        .join(Book, UserBookView.book_id == Book.id, isouter=True)
        .filter(UserBookView.user_id == current_user.id)
        .order_by(UserBookView.last_viewed_at.desc())
        .limit(8)
        .all()
    )
    
    # حساب متوسط التقييم
    avg_rating = None
    if total_reviews > 0:
        from sqlalchemy import func
        avg_result = db.session.query(func.avg(BookReview.rating)).filter_by(user_id=current_user.id).scalar()
        avg_rating = round(float(avg_result), 1) if avg_result else None
    
    stats = {
        'total_books': total_books,
        'books_finished': books_finished,
        'books_reading': books_reading,
        'books_later': books_later,
        'books_favorite': books_favorite,
        'total_reviews': total_reviews,
        'total_views': total_views,
        'days_member': days_member,
        'member_since': member_since,
        'avg_rating': avg_rating,
        'streak': current_user.current_streak
    }
    
    return render_template("profile.html", user_reviews=user_reviews, stats=stats, recent_views=recent_views)


# إزالة صورة البروفايل
@auth_bp.route("/profile/remove-picture", methods=["POST"])
@login_required
def remove_profile_picture():
    if current_user.profile_picture:
        # حذف الملف القديم
        old_path = os.path.join(current_app.root_path, 'static', current_user.profile_picture.lstrip('/static/'))
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass
        current_user.profile_picture = None
        db.session.commit()
        flash("Profile picture removed successfully", "success")
    return redirect(url_for("auth.profile"))

# مستخدم تجريبي جاهز للاختبار
@auth_bp.route("/seed/demo")
def seed_demo_user():
    if not User.query.filter_by(email="admin@example.com").first():
        u = User(
            name="Admin",
            email="admin@example.com",
            password_hash=generate_password_hash("1234"),
            onboarding_completed=True
        )
        db.session.add(u)
        db.session.commit()
    return "User: admin@example.com / 1234"

