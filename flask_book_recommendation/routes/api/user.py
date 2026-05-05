# routes/api/user.py
"""
API User endpoints - مكتبة المستخدم والتفضيلات
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ...extensions import db, cache
from datetime import datetime
from ...models import (
    User, Book, UserPreference, BookStatus, 
    UserRatingCF, BookReview, UserBookNote, SearchHistory,
    UserBookView
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
    
    if status not in ['favorite', 'later', 'finished', 'reading']:
        return jsonify({
            'success': False,
            'error': 'الحالة يجب أن تكون: favorite, later, finished, أو reading'
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
    
    try:
        from ai_book_recommender.unified_pipeline import get_unified_engine
        engine = get_unified_engine()
        engine.clear_user_cache(user_id)
    except Exception as e:
        print(f"Clear unified cache error: {e}")
        pass
        
    try:
        from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based
        cache.delete_memoized(get_homepage_sections)
        cache.delete_memoized(get_topic_based)
    except Exception:
        pass
    
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
    
    try:
        from ai_book_recommender.unified_pipeline import get_unified_engine
        engine = get_unified_engine()
        engine.clear_user_cache(user_id)
    except Exception as e:
        print(f"Clear unified cache error: {e}")
        pass
        
    try:
        from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based
        cache.delete_memoized(get_homepage_sections)
        cache.delete_memoized(get_topic_based)
    except Exception:
        pass
    
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
    
    # التحقق من وجود الكتاب أو إنشاؤه لضمان ظهوره في قسم Trending
    book = Book.query.filter_by(google_id=gid).first()
    if not book:
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
            pass

    # حفظ في UserRatingCF للتوصيات
    cf_rating = UserRatingCF.query.filter_by(user_id=user_id, google_id=gid).first()
    if cf_rating:
        cf_rating.rating = float(rating)
    else:
        cf_rating = UserRatingCF(user_id=user_id, google_id=gid, rating=float(rating))
        db.session.add(cf_rating)
    
    # دائماً حفظ التقييم في BookReview لكي يظهر في التوصيات الرائجة (Trending)
    review = BookReview.query.filter_by(user_id=user_id, google_id=gid).first()
    if review:
        review.rating = rating
        if review_text: review.review_text = review_text
        # ضمان ربط book_id لكي تظهر المراجعة في الويب أيضاً
        if not review.book_id and book:
            review.book_id = book.id
    else:
        review = BookReview(
            user_id=user_id,
            google_id=gid,
            rating=rating,
            review_text=review_text if review_text else None,
            book_id=book.id if book else None
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
    
    try:
        from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based, get_top_rated
        cache.delete_memoized(get_homepage_sections)
        cache.delete_memoized(get_topic_based)
        cache.delete_memoized(get_top_rated)
    except Exception:
        pass
    
    try:
        from ai_book_recommender.unified_pipeline import get_unified_engine
        engine = get_unified_engine()
        engine.clear_user_cache(user_id)
    except Exception as e:
        print(f"Clear unified cache error: {e}")
        pass
        
    try:
        from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based
        cache.delete_memoized(get_homepage_sections)
        cache.delete_memoized(get_topic_based)
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'message': 'تم حفظ التقييم'
    })


@api_user_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """
    User Statistics - Real metrics from DB
    GET /api/user/stats
    """
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404

    # Update activity streak
    user.update_activity()
    db.session.commit()

    # 1. Library Counts (by status)
    favorites = BookStatus.query.filter_by(user_id=user_id, status='favorite').count()
    later = BookStatus.query.filter_by(user_id=user_id, status='later').count()
    finished = BookStatus.query.filter_by(user_id=user_id, status='finished').count()
    
    # "Reading" defined as progress > 0 and not yet finished
    reading = BookStatus.query.filter(
        BookStatus.user_id == user_id, 
        BookStatus.status != 'finished',
        BookStatus.reading_progress > 0
    ).count()
    
    # 2. Activity Counts (reviews & views & pages)
    reviews_count = BookReview.query.filter_by(user_id=user_id).count()
    views_res = db.session.query(db.func.sum(UserBookView.view_count)).filter(UserBookView.user_id == user_id).scalar() or 0
    
    # Calculate total pages read
    total_pages_read = 0
    # Finished books
    finished_books_pages = db.session.query(Book.page_count).join(BookStatus).filter(
        BookStatus.user_id == user_id,
        BookStatus.status == 'finished'
    ).all()
    total_pages_read += sum(b.page_count or 0 for b in finished_books_pages)
    
    # Reading books (partial progress)
    reading_statuses = BookStatus.query.filter(
        BookStatus.user_id == user_id,
        BookStatus.status != 'finished',
        BookStatus.reading_progress > 0
    ).all()
    for bs in reading_statuses:
        if bs.book and bs.book.page_count:
            total_pages_read += int(bs.book.page_count * (bs.reading_progress / 100.0))
    
    # 3. Time & Streak
    days_member = (datetime.utcnow() - user.created_at).days if user.created_at else 0
    
    # 4. Ratings (Average)
    avg_rating_res = db.session.query(db.func.avg(BookReview.rating)).filter(BookReview.user_id == user_id).scalar()
    avg_rating = round(float(avg_rating_res), 1) if avg_rating_res else "—"

    # 5. Dynamic Rank Update
    if finished >= 50:
        user.rank = "Grand Librarian"
    elif finished >= 20:
        user.rank = "Scholar"
    elif finished >= 10:
        user.rank = "Avid Reader"
    elif finished >= 5:
        user.rank = "Bookworm"
    else:
        user.rank = "Novice Reader"
    
    db.session.commit()

    return jsonify({
        'success': True,
        'stats': {
            'days_member': days_member,
            'total_books': favorites + later + finished,
            'streak': user.current_streak or 0,
            'books_finished': finished,
            'total_reviews': reviews_count,
            'total_views': views_res,
            'total_pages_read': total_pages_read,
            'books_reading': reading,
            'books_later': later,
            'books_favorite': favorites,
            'avg_rating': avg_rating,
            'rank': user.rank
        }
    })


@api_user_bp.route('/log-search', methods=['POST'])
@jwt_required()
def log_search():
    """
    تسجيل عملية بحث المستخدم لتحديث التوصيات
    POST /api/user/log-search
    Body: {"query": "python programming"}
    """
    from datetime import datetime

    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    q = (data.get('query') or '').strip()

    if not q or len(q) < 2:
        return jsonify({'success': False, 'error': 'يرجى إدخال نص بحث صالح'}), 400

    try:
        # 1. حفظ في SearchHistory (نفس منطق الويب)
        history = SearchHistory(
            user_id=user_id,
            query=q,
            created_at=datetime.utcnow()
        )
        db.session.add(history)

        # 2. تحديث التفضيلات بناءً على كلمات البحث
        keywords = q.lower().split()
        valid_kw = [k for k in keywords if len(k) > 2]
        for kw in valid_kw[:3]:
            pref = UserPreference.query.filter_by(user_id=user_id, topic=kw).first()
            if pref:
                pref.weight += 40.0
                pref.updated_at = datetime.utcnow()
            else:
                pref = UserPreference(user_id=user_id, topic=kw, weight=100.0)
                db.session.add(pref)

        db.session.commit()

        # 🔥 إبطال كاش التوصيات لتحديثها فوراً
        cache.delete(f"home_full_{user_id}")
        cache.delete(f"home_feed_{user_id}")
        cache.delete(f"home_recs_{user_id}")
        
        try:
            from ai_book_recommender.unified_pipeline import get_unified_engine
            engine = get_unified_engine()
            engine.clear_user_cache(user_id)
        except Exception as e:
            print(f"Clear unified cache error: {e}")
            pass
            
        try:
            from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based
            cache.delete_memoized(get_homepage_sections)
            cache.delete_memoized(get_topic_based)
        except Exception:
            pass

        return jsonify({'success': True, 'message': 'تم تسجيل البحث بنجاح'})

    except Exception as e:
        db.session.rollback()
        print(f"[log-search] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_user_bp.route('/search-history', methods=['GET'])
@jwt_required()
def get_search_history():
    """
    جلب سجل بحث المستخدم من السيرفر
    GET /api/user/search-history?limit=10
    """
    user_id = int(get_jwt_identity())
    limit = request.args.get('limit', 10, type=int)

    try:
        searches = (SearchHistory.query
                    .filter_by(user_id=user_id)
                    .order_by(SearchHistory.created_at.desc())
                    .limit(limit)
                    .all())

        queries = []
        seen = set()
        for s in searches:
            q = s.query.strip()
            if q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)

        return jsonify({'success': True, 'queries': queries})
    except Exception as e:
        return jsonify({'success': True, 'queries': []})


@api_user_bp.route('/book-view', methods=['POST'])
@jwt_required()
def log_book_view():
    """
    تسجيل مشاهدة كتاب لتحسين التوصيات
    POST /api/user/book-view
    Body: {"google_id": "abc123", "source": "google", "book_info": {...}}
    """
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
        
        try:
            from ai_book_recommender.unified_pipeline import get_unified_engine
            engine = get_unified_engine()
            engine.clear_user_cache(user_id)
        except Exception as e:
            print(f"Clear unified cache error: {e}")
            pass
            
        try:
            from flask_book_recommendation.recommender import get_homepage_sections, get_topic_based
            cache.delete_memoized(get_homepage_sections)
            cache.delete_memoized(get_topic_based)
        except Exception:
            pass
        
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


@api_user_bp.route('/library/<gid>/status', methods=['GET'])
@jwt_required()
def get_book_status(gid: str):
    """
    التحقق من حالة كتاب في المكتبة
    GET /api/user/library/<gid>/status
    """
    user_id = int(get_jwt_identity())
    
    book = Book.query.filter_by(google_id=gid, owner_id=user_id).first()
    if not book:
        return jsonify({
            'success': True,
            'in_library': False,
            'status': None
        })
    
    book_status = BookStatus.query.filter_by(user_id=user_id, book_id=book.id).first()
    if not book_status:
        return jsonify({
            'success': True,
            'in_library': False,
            'status': None
        })
    
    return jsonify({
        'success': True,
        'in_library': True,
        'status': book_status.status,
        'reading_progress': book_status.reading_progress or 0
    })


@api_user_bp.route('/notes/<gid>', methods=['GET'])
@jwt_required()
def get_note(gid: str):
    """
    جلب ملاحظة كتاب
    GET /api/user/notes/<gid>
    """
    user_id = int(get_jwt_identity())
    
    note = UserBookNote.query.filter_by(user_id=user_id, google_id=gid).first()
    
    note_text = note.note_text if note else ''
    
    # إذا لم توجد ملاحظة في الجدول المخصص، ابحث في جدول الكتب (لمزامنة الويب)
    if not note_text:
        book = Book.query.filter_by(owner_id=user_id, google_id=gid).first()
        if book and book.notes:
            note_text = book.notes
    
    return jsonify({
        'success': True,
        'note_text': note_text,
        'updated_at': note.updated_at.isoformat() if note and note.updated_at else None
    })


@api_user_bp.route('/notes/<gid>', methods=['PUT'])
@jwt_required()
def save_note(gid: str):
    """
    حفظ/تحديث ملاحظة كتاب
    PUT /api/user/notes/<gid>
    Body: {"note_text": "..."}
    """
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    note_text = data.get('note_text', '')
    
    note = UserBookNote.query.filter_by(user_id=user_id, google_id=gid).first()
    if note:
        note.note_text = note_text
    else:
        book = Book.query.filter_by(google_id=gid).first()
        note = UserBookNote(
            user_id=user_id,
            google_id=gid,
            book_id=book.id if book else None,
            note_text=note_text
        )
        db.session.add(note)
    
    # مزامنة الملاحظة مع حقل Book.notes لكي تظهر في الويب أيضاً
    book = Book.query.filter_by(google_id=gid, owner_id=user_id).first()
    if book:
        book.notes = note_text
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'تم حفظ الملاحظة'
    })


@api_user_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """
    تحديث الملف الشخصي
    PUT /api/user/profile
    Body: {"name": "...", "bio": "...", "reading_goal": 12}
    """
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'success': False, 'error': 'المستخدم غير موجود'}), 404
    
    data = request.get_json() or {}
    
    if 'name' in data and data['name']:
        user.name = data['name'].strip()
    if 'bio' in data:
        user.bio = data['bio']
    if 'reading_goal' in data:
        user.reading_goal = int(data['reading_goal'])
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'bio': user.bio,
            'reading_goal': user.reading_goal,
            'rank': user.rank,
        }
    })


@api_user_bp.route('/library/<gid>/progress', methods=['PUT'])
@jwt_required()
def update_reading_progress(gid: str):
    """
    تحديث تقدم القراءة
    PUT /api/user/library/<gid>/progress
    Body: {"progress": 50}
    """
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}
    progress = data.get('progress', 0)
    
    book = Book.query.filter_by(google_id=gid, owner_id=user_id).first()
    if not book:
        return jsonify({'success': False, 'error': 'الكتاب غير موجود في المكتبة'}), 404
    
    book_status = BookStatus.query.filter_by(user_id=user_id, book_id=book.id).first()
    if not book_status:
        return jsonify({'success': False, 'error': 'الكتاب غير موجود في المكتبة'}), 404
    
    from datetime import datetime
    book_status.reading_progress = progress
    book_status.last_read_at = datetime.utcnow()
    
    if progress >= 100:
        book_status.status = 'finished'
        book_status.finished_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'تم تحديث التقدم',
        'progress': progress,
        'status': book_status.status
    })
