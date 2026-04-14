# routes/api/user.py
"""
API User endpoints - مكتبة المستخدم والتفضيلات
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ...extensions import db, cache
from ...models import (
    User, Book, UserPreference, BookStatus, 
    UserRatingCF, BookReview
)

api_user_bp = Blueprint('api_user', __name__, url_prefix='/user')


@api_user_bp.route('/library', methods=['GET'])
@jwt_required()
def get_library():
    """
    مكتبة المستخدم (الكتب المحفوظة)
    GET /api/user/library?status=favorite
    """
    user_id = int(get_jwt_identity())
    status_filter = request.args.get('status')  # favorite, later, finished
    
    query = BookStatus.query.filter_by(user_id=user_id)
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    statuses = query.order_by(BookStatus.created_at.desc()).all()
    
    books = []
    for bs in statuses:
        book = bs.book
        if book:
            books.append({
                'id': book.id,
                'gid': book.google_id,
                'title': book.title,
                'author': book.author,
                'cover_url': book.cover_url,
                'status': bs.status,
                'added_at': bs.created_at.isoformat() if bs.created_at else None
            })
    
    return jsonify({
        'success': True,
        'total': len(books),
        'books': books
    })


@api_user_bp.route('/library/<gid>', methods=['POST'])
@jwt_required()
def add_to_library(gid: str):
    """
    إضافة كتاب للمكتبة
    POST /api/user/library/<gid>
    Body: {"status": "favorite"} // favorite, later, finished
    """
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    status = data.get('status', 'later')
    
    if status not in ['favorite', 'later', 'finished']:
        return jsonify({
            'success': False,
            'error': 'الحالة يجب أن تكون: favorite, later, أو finished'
        }), 400
    
    # التحقق من وجود الكتاب أو إنشاؤه
    book = Book.query.filter_by(google_id=gid, owner_id=user_id).first()
    if not book:
        # جلب معلومات الكتاب من Google Books
        import requests
        try:
            resp = requests.get(f"https://www.googleapis.com/books/v1/volumes/{gid}", timeout=10)
            if resp.status_code == 200:
                info = resp.json().get('volumeInfo', {})
                book = Book(
                    google_id=gid,
                    title=info.get('title', 'بدون عنوان'),
                    author=', '.join(info.get('authors', [])),
                    description=info.get('description', ''),
                    cover_url=info.get('imageLinks', {}).get('thumbnail', ''),
                    owner_id=user_id
                )
                db.session.add(book)
                db.session.commit()
        except Exception:
            return jsonify({
                'success': False,
                'error': 'لم يتم العثور على الكتاب'
            }), 404
    
    if not book:
        return jsonify({
            'success': False,
            'error': 'لم يتم العثور على الكتاب'
        }), 404
    
    # إضافة أو تحديث الحالة
    book_status = BookStatus.query.filter_by(user_id=user_id, book_id=book.id).first()
    if book_status:
        book_status.status = status
    else:
        book_status = BookStatus(user_id=user_id, book_id=book.id, status=status)
        db.session.add(book_status)
    
    db.session.commit()
    
    # --- 🆕 User Embedding Update (Phase 2) ---
    try:
        from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
        user_embedding_manager.update_user_embedding(user_id, book_id=book.id)
    except Exception as e_emb:
        print(f"Embedding update error: {e_emb}")
    # ------------------------------------------
    
    # 🔥 إبطال كاش الصفحة الرئيسية لحظياً
    cache.delete(f"home_full_{user_id}")
    cache.delete(f"home_feed_{user_id}")
    cache.delete(f"home_recs_{user_id}")
    
    return jsonify({
        'success': True,
        'message': f'تم إضافة الكتاب كـ {status}'
    })


@api_user_bp.route('/library/<gid>', methods=['DELETE'])
@jwt_required()
def remove_from_library(gid: str):
    """
    حذف كتاب من المكتبة
    DELETE /api/user/library/<gid>
    """
    user_id = int(get_jwt_identity())
    
    book = Book.query.filter_by(google_id=gid, owner_id=user_id).first()
    if book:
        BookStatus.query.filter_by(user_id=user_id, book_id=book.id).delete()
        db.session.commit()
    
    # 🔥 إبطال كاش الصفحة الرئيسية لحظياً
    cache.delete(f"home_full_{user_id}")
    cache.delete(f"home_feed_{user_id}")
    cache.delete(f"home_recs_{user_id}")
    
    return jsonify({
        'success': True,
        'message': 'تم حذف الكتاب من المكتبة'
    })


@api_user_bp.route('/preferences', methods=['GET'])
@jwt_required()
def get_preferences():
    """
    اهتمامات المستخدم
    GET /api/user/preferences
    """
    user_id = int(get_jwt_identity())
    
    prefs = UserPreference.query.filter_by(user_id=user_id).all()
    interests = [{'topic': p.topic, 'weight': p.weight} for p in prefs]
    
    return jsonify({
        'success': True,
        'interests': interests
    })


@api_user_bp.route('/preferences', methods=['PUT'])
@jwt_required()
def update_preferences():
    """
    تحديث اهتمامات المستخدم
    PUT /api/user/preferences
    Body: {"interests": ["Programming", "AI", "Fiction"]}
    """
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    interests = data.get('interests', [])
    
    # حذف القديم
    UserPreference.query.filter_by(user_id=user_id).delete()
    
    # إضافة الجديد
    for topic in interests:
        pref = UserPreference(user_id=user_id, topic=topic, weight=100.0)
        db.session.add(pref)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'تم تحديث الاهتمامات'
    })


@api_user_bp.route('/rate/<gid>', methods=['POST'])
@jwt_required()
def rate_book(gid: str):
    """
    تقييم كتاب
    POST /api/user/rate/<gid>
    Body: {"rating": 5, "review": "كتاب رائع!"}
    """
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    rating = data.get('rating')
    review_text = data.get('review', '')
    
    if not rating or rating < 1 or rating > 5:
        return jsonify({
            'success': False,
            'error': 'التقييم يجب أن يكون من 1 إلى 5'
        }), 400
    
    # حفظ في UserRatingCF للتوصيات
    cf_rating = UserRatingCF.query.filter_by(user_id=user_id, google_id=gid).first()
    if cf_rating:
        cf_rating.rating = float(rating)
    else:
        cf_rating = UserRatingCF(user_id=user_id, google_id=gid, rating=float(rating))
        db.session.add(cf_rating)
    
    # حفظ المراجعة إن وجدت
    if review_text:
        review = BookReview.query.filter_by(user_id=user_id, google_id=gid).first()
        if review:
            review.rating = rating
            review.review_text = review_text
        else:
            review = BookReview(
                user_id=user_id,
                google_id=gid,
                rating=rating,
                review_text=review_text
            )
            db.session.add(review)
    
    db.session.commit()
    
    # --- 🆕 User Embedding Update (Phase 2) ---
    try:
        from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
        user_embedding_manager.update_user_embedding(user_id, google_id=gid)
    except Exception as e_emb:
        print(f"Embedding update error: {e_emb}")
    # ------------------------------------------

    # 🔥 إبطال كاش الصفحة الرئيسية لحظياً لتحديث التوصيات بناءً على التقييم
    cache.delete(f"home_full_{user_id}")
    cache.delete(f"home_feed_{user_id}")
    cache.delete(f"home_recs_{user_id}")
    
    return jsonify({
        'success': True,
        'message': 'تم حفظ التقييم'
    })


@api_user_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """
    إحصائيات المستخدم
    GET /api/user/stats
    """
    user_id = int(get_jwt_identity())
    
    # عدد الكتب حسب الحالة
    favorites = BookStatus.query.filter_by(user_id=user_id, status='favorite').count()
    later = BookStatus.query.filter_by(user_id=user_id, status='later').count()
    finished = BookStatus.query.filter_by(user_id=user_id, status='finished').count()
    
    # عدد التقييمات
    ratings_count = UserRatingCF.query.filter_by(user_id=user_id).count()
    
    # عدد المراجعات
    reviews_count = BookReview.query.filter_by(user_id=user_id).count()
    
    return jsonify({
        'success': True,
        'stats': {
            'favorites': favorites,
            'read_later': later,
            'finished': finished,
            'total_books': favorites + later + finished,
            'ratings': ratings_count,
            'reviews': reviews_count
        }
    })


@api_user_bp.route('/book-view', methods=['POST'])
@jwt_required()
def log_book_view():
    """
    تسجيل مشاهدة كتاب لتحسين التوصيات
    POST /api/user/book-view
    Body: {"google_id": "abc123", "source": "google", "book_info": {...}}
    """
    from ...models import UserBookView, Book
    from ...utils import update_user_preferences_from_behavior
    
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    
    google_id = data.get('google_id')
    book_id = data.get('book_id')
    source = data.get('source', 'unknown')
    book_info = data.get('book_info', {})
    
    if not google_id and not book_id:
        return jsonify({
            'success': False,
            'error': 'يجب تحديد google_id أو book_id'
        }), 400
    
    try:
        # البحث عن مشاهدة سابقة
        if google_id:
            view = UserBookView.query.filter_by(user_id=user_id, google_id=google_id).first()
        else:
            view = UserBookView.query.filter_by(user_id=user_id, book_id=book_id).first()
        
        if view:
            # تحديث عدد المشاهدات
            view.view_count = (view.view_count or 0) + 1
        else:
            # إنشاء مشاهدة جديدة
            view = UserBookView(
                user_id=user_id,
                google_id=google_id,
                book_id=book_id,
                view_count=1
            )
            db.session.add(view)
        
        db.session.commit()
        
        # --- 🆕 User Embedding Update (Phase 2) ---
        try:
            from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
            user_embedding_manager.update_user_embedding(user_id, book_id=book_id, google_id=google_id)
        except Exception as e_emb:
            print(f"Embedding update error: {e_emb}")
        # ------------------------------------------
        
        # تحديث التفضيلات تلقائياً
        if book_info:
            try:
                update_user_preferences_from_behavior(user_id, "view", book_info)
            except Exception as e:
                print(f"[BookView] Preferences update error: {e}")
        
        # 🔥 إبطال كاش الصفحة الرئيسية لحظياً لتحديث التوصيات بناءً على المشاهدة
        cache.delete(f"home_full_{user_id}")
        cache.delete(f"home_feed_{user_id}")
        cache.delete(f"home_recs_{user_id}")
        
        return jsonify({
            'success': True,
            'view_count': view.view_count
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[BookView] Error: {e}")
        return jsonify({
            'success': False,
            'error': 'حدث خطأ في تسجيل المشاهدة'
        }), 500


@api_user_bp.route('/behavior-profile', methods=['GET'])
@jwt_required()
def get_behavior_profile():
    """
    ملف سلوك المستخدم - تحليل الاهتمامات والأنماط
    GET /api/user/behavior-profile
    """
    from ...utils import get_user_behavior_profile
    
    user_id = int(get_jwt_identity())
    
    try:
        profile = get_user_behavior_profile(user_id)
        return jsonify({
            'success': True,
            'profile': profile
        })
    except Exception as e:
        print(f"[BehaviorProfile API] Error: {e}")
        return jsonify({
            'success': False,
            'error': 'حدث خطأ في تحليل السلوك'
        }), 500


@api_user_bp.route('/ai-recommendations', methods=['GET'])
@jwt_required()
def get_ai_recommendations():
    """
    توصيات مخصصة بالذكاء الاصطناعي
    GET /api/user/ai-recommendations?limit=12
    """
    from ...utils import get_ai_personalized_recommendations
    
    user_id = int(get_jwt_identity())
    limit = request.args.get('limit', 12, type=int)
    
    try:
        result = get_ai_personalized_recommendations(user_id, limit=limit)
        return jsonify({
            'success': result.get('success', False),
            'books': result.get('books', []),
            'ai_analysis': result.get('ai_analysis', ''),
            'suggested_topics': result.get('suggested_topics', [])
        })
    except Exception as e:
        print(f"[AI Recommendations API] Error: {e}")
        return jsonify({
            'success': False,
            'error': 'حدث خطأ في جلب التوصيات'
        }), 500
