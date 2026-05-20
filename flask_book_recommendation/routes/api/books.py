# routes/api/books.py
"""
API Books endpoints - البحث والتوصيات وتفاصيل الكتب
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, jwt_required
from ...recommender import (
    get_mood_based_recommendations,
    get_recommendations_by_title,
    MOOD_MAPPING,
    _get_user_interests,
    _get_book_uid,
    _strict_interest_filter,
    _fetch_interest_books_from_api
)
from ...utils import (
    fetch_google_books,
    fetch_gutenberg_books,
    fetch_openlib_books,
    fetch_archive_books,
    fetch_itbook_books
)
from concurrent.futures import ThreadPoolExecutor
import requests

api_books_bp = Blueprint('api_books', __name__, url_prefix='/books')


@api_books_bp.route('/search', methods=['GET'])
def search_books():
    """
    البحث عن كتب
    GET /api/books/search?q=python&page=1&per_page=20
    """
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    should_log = request.args.get('log', 'true').lower() == 'true'
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'يرجى إدخال كلمة البحث (q)'
        }), 400
    
    # ── تسجيل البحث تلقائياً (نفس منطق الويب) ──
    try:
        if should_log:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity as _get_id
            import threading
            from datetime import datetime
            from flask import current_app
            verify_jwt_in_request(optional=True)
            uid = _get_id()
            if uid:
                uid = int(uid)
            app = current_app._get_current_object()
            
            if uid is None:
                return  # Skip search history for anonymous users
            def _bg_log(app, user_id, q):
                with app.app_context():
                    try:
                        from ...models import SearchHistory, UserPreference
                        from ...extensions import db, cache
                        history = SearchHistory(user_id=user_id, query=q, created_at=datetime.utcnow())
                        db.session.add(history)
                        keywords = q.lower().split()
                        for kw in [k for k in keywords if len(k) > 2][:3]:
                            pref = UserPreference.query.filter_by(user_id=user_id, topic=kw).first()
                            if pref:
                                pref.weight += 40.0
                                pref.updated_at = datetime.utcnow()
                            else:
                                pref = UserPreference(user_id=user_id, topic=kw, weight=100.0)
                                db.session.add(pref)
                        db.session.commit()
                        cache.delete(f"home_full_{user_id}")
                        cache.delete(f"home_feed_{user_id}")
                        cache.delete(f"home_recs_{user_id}")
                        try:
                            from ai_book_recommender.unified_pipeline import get_unified_engine
                            engine = get_unified_engine()
                            engine.clear_user_cache(user_id)
                        except: pass
                        try:
                            from ...recommender import get_homepage_sections, get_topic_based, get_last_search_recommendations
                            cache.delete_memoized(get_homepage_sections)
                            cache.delete_memoized(get_topic_based)
                            cache.delete_memoized(get_last_search_recommendations)
                        except: pass
                    except Exception as e:
                        try: db.session.rollback()
                        except: pass
                        print(f"[api/search] bg log error: {e}")
            threading.Thread(target=_bg_log, args=(app, uid, query), daemon=True).start()
    except: pass
    
    # البحث في المصادر المتعددة بالتوازي
    all_books = []
    
    def fetch_google():
        try:
            res = fetch_google_books(query, max_results=per_page, start_index=(page-1)*per_page)
            items = res[0] if isinstance(res, tuple) else res
            result = []
            for it in (items or []):
                vi = it.get("volumeInfo", {}) or {}
                links = vi.get("imageLinks", {}) or {}
                cover = links.get("thumbnail") or links.get("smallThumbnail")
                if cover and cover.startswith("http://"): 
                    cover = cover.replace("http://", "https://")
                    
                result.append({
                    "id": it.get("id"),
                    "title": vi.get("title") or "بدون عنوان",
                    "author": ", ".join(vi.get("authors", [])) if vi.get("authors") else "غير معروف",
                    "desc": vi.get("description") or "",
                    "cover": cover or "",
                    "cover_url": cover or "",
                    "source": "google",
                    "rating": vi.get("averageRating"),
                    "ratings_count": vi.get("ratingsCount"),
                })
            return result
        except Exception: return []
    
    def fetch_gut():
        try:
            return fetch_gutenberg_books(query, page=page)[:per_page] or []
        except: return []
    
    def fetch_ol():
        try:
            return fetch_openlib_books(query, limit=per_page, offset=(page-1)*per_page) or []
        except: return []

    def fetch_ia():
        try:
            return fetch_archive_books(query, limit=per_page) or []
        except: return []

    def fetch_it():
        try:
            return fetch_itbook_books(query, page=page, limit=per_page) or []
        except: return []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_google),
            executor.submit(fetch_gut),
            executor.submit(fetch_ol),
            executor.submit(fetch_ia),
            executor.submit(fetch_it)
        ]
        
        for future in futures:
            try:
                result = future.result(timeout=12)
                if result:
                    all_books.extend(result)
            except:
                pass
    
    # إزالة التكرارات بناءً على العنوان
    seen_titles = set()
    unique_books = []
    for book in all_books:
        title = (book.get('title') or '').lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_books.append(book)
    
    return jsonify({
        'success': True,
        'query': query,
        'page': page,
        'per_page': per_page,
        'total': len(unique_books),
        'books': unique_books[:per_page]
    })


@api_books_bp.route('/trending', methods=['GET'])
def get_trending_books():
    """
    الكتب الرائجة
    GET /api/books/trending?limit=12
    """
    limit = request.args.get('limit', 12, type=int)
    
    try:
        from ...recommender import get_trending
        books = get_trending(limit)
        return jsonify({
            'success': True,
            'books': books or []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'books': []
        })


@api_books_bp.route('/top-rated', methods=['GET'])
def get_top_rated_books():
    """
    الكتب الأعلى تقييماً من المجتمع
    GET /api/books/top-rated?limit=15
    """
    limit = request.args.get('limit', 15, type=int)
    
    try:
        from ...recommender import get_top_rated
        books = get_top_rated(limit)
        return jsonify({
            'success': True,
            'books': books or []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'books': []
        })



@api_books_bp.route('/mood-recommendations', methods=['GET'])
def get_mood_recs():
    """
    توصيات بناءً على المزاج
    GET /api/books/mood-recommendations?mood=happy&limit=12
    """
    mood = request.args.get('mood', '').strip()
    limit = request.args.get('limit', 12, type=int)
    
    if not mood:
        return jsonify({
            'success': False,
            'error': 'يرجى تحديد المزاج (mood)'
        }), 400
    
    try:
        books = get_mood_based_recommendations(mood, limit=limit)
        return jsonify({
            'success': True,
            'books': books
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'books': []
        })


@api_books_bp.route('/recommend-by-book', methods=['GET'])
def get_book_recs():
    """
    توصيات بناءً على كتاب معين
    GET /api/books/recommend-by-book?title=Harry Potter&limit=24
    """
    title = request.args.get('title', '').strip()
    limit = request.args.get('limit', 24, type=int)
    
    if not title:
        return jsonify({
            'success': False,
            'error': 'يرجى إدخال اسم الكتاب'
        }), 400
    
    try:
        books = get_recommendations_by_title(title, limit=limit)
        return jsonify({
            'success': True,
            'books': books
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'books': []
        })


@api_books_bp.route('/mood-meta', methods=['GET'])
def get_mood_meta():
    """
    بيانات الحالة المزاجية المتوفرة (العناوين والرموز التعبيرية)
    GET /api/books/mood-meta
    """
    return jsonify({
        'success': True,
        'moods': MOOD_MAPPING
    })


@api_books_bp.route('/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    """
    التوصيات المخصصة للمستخدم (Unified Neural Engine)
    يتطابق الآن مع منطق الويب الصارم (Gatekeeper + API Fallback)
    """
    user_id = int(get_jwt_identity())
    
    try:
        import time as _time
        from flask import current_app
        from ai_book_recommender.unified_pipeline import get_unified_engine
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        engine = get_unified_engine()
        if engine.flask_app is None:
            engine.flask_app = current_app._get_current_object()

        now_ts = _time.time()
        ctx = {"page": "home_mobile", "time": now_ts, "device": "mobile"}
        app_obj = current_app._get_current_object()

        # 1. Get user interests
        user_interests = _get_user_interests(user_id)
        
        # Helper to map and format books for JSON response
        def _map_books(recs):
            bl = []
            for b in (recs or []):
                if not b: continue
                # Handling both dict and model objects
                bid = b.get("id") or b.get("google_id") if isinstance(b, dict) else (getattr(b, 'google_id', None) or getattr(b, 'id', None))
                title = b.get("title") if isinstance(b, dict) else getattr(b, 'title', 'بدون عنوان')
                author = b.get("author") or (b.get("authors") or ["Unknown"])[0] if isinstance(b, dict) else getattr(b, 'author', 'غير معروف')
                cover = b.get("cover_url") or b.get("cover") if isinstance(b, dict) else getattr(b, 'cover_url', '')
                cats = b.get("categories", []) if isinstance(b, dict) else (getattr(b, 'categories', '').split(",") if getattr(b, 'categories', None) else [])
                
                bl.append({
                    "id": str(bid),
                    "title": title,
                    "author": author,
                    "cover_url": cover,
                    "categories": cats,
                    "rating": b.get("rating", 4.5) if isinstance(b, dict) else getattr(b, 'average_rating', 0.0),
                    "reason": b.get("algo_tag", "Neural Pipeline") if isinstance(b, dict) else "Community",
                    "algorithm_tag": b.get("algo_tag", "Neural Pipeline") if isinstance(b, dict) else "Community",
                    "preview_link": b.get("preview_link") or b.get("preview") if isinstance(b, dict) else getattr(b, 'preview_link', None),
                    "info_link": b.get("info_link") or b.get("info") if isinstance(b, dict) else getattr(b, 'info_link', None)
                })
            return bl

        # 2. Fetch sections in parallel (matching web home_feed logic)
        sections_data = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            def _run(func, *args, **kwargs):
                with app_obj.app_context():
                    return func(*args, **kwargs)

            futures = {
                executor.submit(_run, engine.recommend_full_stack, user_id=user_id, top_k=100, context=ctx): "recommended_for_you",
            }
            
            for f in as_completed(futures, timeout=15):
                name = futures[f]
                try:
                    sections_data[name] = f.result() or []
                except:
                    sections_data[name] = []

        # 3. Apply Gatekeeper & Fallback to "Recommended for you"
        main_recs = sections_data.get("recommended_for_you", [])
        if user_interests:
            # Apply strict filter
            filtered_recs = _strict_interest_filter(main_recs, user_interests, limit=60)
            
            # If too few, supplement from API (The "Missing Books" fix)
            if len(filtered_recs) < 5:
                api_supplement = _fetch_interest_books_from_api(user_interests, limit_per_interest=10)
                
                # Deduplicate against filtered
                seen_uids = { _get_book_uid(b) for b in filtered_recs }
                for ab in api_supplement:
                    uid = _get_book_uid(ab)
                    if uid not in seen_uids:
                        filtered_recs.append(ab)
                        seen_uids.add(uid)
            
            main_recs = filtered_recs

        # 4. Construct Final Sections List
        sections = []
        
        # Section: Recommended for You (Neural + Interests)
        if main_recs:
            sections.append({
                "title": "Recommend for you",
                "subtitle": "9-Stage Neural Pipeline • Deep Learning",
                "icon": "🧠",
                "books": _map_books(main_recs)
            })

        # Section: Because you searched (Dynamic Search)
        try:
            from flask_book_recommendation.models import SearchHistory, Book
            from flask_book_recommendation.extensions import db
            last_search_obj = SearchHistory.query.filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).first()
            if last_search_obj and last_search_obj.query:
                q = last_search_obj.query.title()
                dynamic_books = Book.query.filter(db.or_(Book.categories.ilike(f'%{q}%'), Book.title.ilike(f'%{q}%'))).limit(15).all()
                if dynamic_books:
                    sections.append({
                        "title": f"لأنك بحثت عن \"{q}\"",
                        "subtitle": "نتائج فورية لبحثك الأخير",
                        "icon": "🔍",
                        "books": _map_books(dynamic_books)
                    })
        except: pass

        return jsonify({
            'success': True,
            'sections': sections
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"Personalized Recs API Error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'sections': []
        })


@api_books_bp.route('/<gid>', methods=['GET'])
def get_book_detail(gid: str):
    """
    تفاصيل كتاب معين
    GET /api/books/<google_id>
    """
    book = None
    
    # محاولة جلب من Google Books
    try:
        url = f"https://www.googleapis.com/books/v1/volumes/{gid}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            info = data.get('volumeInfo', {})
            access = data.get('accessInfo', {})
            
            book = {
                'gid': gid,
                'title': info.get('title', 'بدون عنوان'),
                'authors': info.get('authors', []),
                'author': ', '.join(info.get('authors', ['غير معروف'])),
                'description': info.get('description', ''),
                'cover_url': info.get('imageLinks', {}).get('thumbnail', ''),
                'categories': info.get('categories', []),
                'publisher': info.get('publisher', ''),
                'published_date': info.get('publishedDate', ''),
                'page_count': info.get('pageCount', 0),
                'language': info.get('language', ''),
                'average_rating': info.get('averageRating', 0),
                'ratings_count': info.get('ratingsCount', 0),
                'preview_link': info.get('previewLink', ''),
                'info_link': info.get('infoLink', ''),
                'can_read': access.get('viewability') in ['ALL_PAGES', 'PARTIAL'],
                'epub_available': access.get('epub', {}).get('isAvailable', False),
                'pdf_available': access.get('pdf', {}).get('isAvailable', False),
                'source': 'google'
            }
    except Exception:
        pass
    
    # حساب التقييم المحلي من مراجعات المستخدمين
    try:
        from ...models import BookReview
        from ...extensions import db
        from sqlalchemy import func
        
        local_stats = db.session.query(
            func.avg(BookReview.rating),
            func.count(BookReview.id)
        ).filter(BookReview.google_id == gid).first()
        
        local_avg = round(float(local_stats[0]), 1) if local_stats and local_stats[0] else 0.0
        local_count = int(local_stats[1]) if local_stats and local_stats[1] else 0
        
        if book and local_count > 0:
            book['average_rating'] = local_avg
            book['ratings_count'] = local_count
            book['is_local_rating'] = True
        elif book:
            book['is_local_rating'] = False
    except Exception as e:
        current_app.logger.error(f"Local rating calc error: {e}")
    
    if not book:
        return jsonify({
            'success': False,
            'error': 'الكتاب غير موجود'
        }), 404
    
    return jsonify({
        'success': True,
        'book': book
    })


@api_books_bp.route('/categories', methods=['GET'])
def get_categories():
    """
    قائمة التصنيفات المتاحة
    GET /api/books/categories
    """
    categories = [
        {"id": "fiction", "name": "روايات", "name_en": "Fiction"},
        {"id": "science", "name": "علوم", "name_en": "Science"},
        {"id": "history", "name": "تاريخ", "name_en": "History"},
        {"id": "philosophy", "name": "فلسفة", "name_en": "Philosophy"},
        {"id": "psychology", "name": "علم نفس", "name_en": "Psychology"},
        {"id": "business", "name": "أعمال", "name_en": "Business"},
        {"id": "self-help", "name": "تطوير ذات", "name_en": "Self-Help"},
        {"id": "biography", "name": "سير ذاتية", "name_en": "Biography"},
        {"id": "programming", "name": "برمجة", "name_en": "Programming"},
        {"id": "ai", "name": "ذكاء اصطناعي", "name_en": "Artificial Intelligence"},
    ]
    
    return jsonify({
        'success': True,
        'categories': categories
    })


@api_books_bp.route('/category/<category_id>', methods=['GET'])
def get_books_by_category(category_id: str):
    """
    Get books by category ID
    GET /api/books/category/programming?page=1
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    try:
        all_books = []
        
        # Handle source-specific requests
        if category_id.startswith("source:"):
            source = category_id.split(":")[1]
            if source == "google":
                res = fetch_google_books(random.choice(RANDOM_TOPICS), max_results=per_page, start_index=(page-1)*per_page)
                items = res[0] if isinstance(res, (tuple, list)) else res
                for it in (items or []):
                    vi = it.get("volumeInfo", {}) or {}
                    links = vi.get("imageLinks", {}) or {}
                    cover = links.get("thumbnail") or links.get("smallThumbnail")
                    if cover and cover.startswith("http://"): cover = cover.replace("http://", "https://")
                    all_books.append({
                        "id": it.get("id"), "title": vi.get("title") or "بدون عنوان",
                        "author": ", ".join(vi.get("authors", [])) if vi.get("authors") else "غير معروف",
                        "desc": vi.get("description") or "", "cover": cover or "", "cover_url": cover or "",
                        "source": "google", "rating": vi.get("averageRating"), "ratings_count": vi.get("ratingsCount"),
                    })
            elif source == "gutenberg":
                all_books = fetch_gutenberg_books(random.choice(RANDOM_TOPICS), page=page)[:per_page] or []
            elif source == "openlib":
                all_books = fetch_openlib_books(random.choice(RANDOM_TOPICS), limit=per_page, offset=(page-1)*per_page) or []
            elif source == "archive":
                all_books = fetch_archive_books(random.choice(RANDOM_TOPICS), limit=per_page) or []
            elif source == "itbook":
                all_books = fetch_itbook_books("programming", page=page, limit=per_page) or []
            
            return jsonify({'success': True, 'category': category_id, 'page': page, 'books': all_books})

        # Regular category search (Unified across all 5 sources)
        import random
        from ...routes.public import RANDOM_TOPICS
        search_term = random.choice(RANDOM_TOPICS) if category_id == 'all' else category_id

        def fetch_google():
            try:
                res = fetch_google_books(search_term, max_results=per_page, start_index=(page-1)*per_page)
                items = res[0] if isinstance(res, (tuple, list)) else res
                result = []
                for it in (items or []):
                    if not isinstance(it, dict): continue
                    vi = it.get("volumeInfo", {}) or {}
                    links = vi.get("imageLinks", {}) or {}
                    cover = links.get("thumbnail") or links.get("smallThumbnail")
                    if cover and cover.startswith("http://"): cover = cover.replace("http://", "https://")
                    result.append({
                        "id": it.get("id"),
                        "title": vi.get("title") or "بدون عنوان",
                        "author": ", ".join(vi.get("authors", [])) if vi.get("authors") else "غير معروف",
                        "desc": vi.get("description") or "",
                        "cover": cover or "", "cover_url": cover or "",
                        "source": "google",
                        "rating": vi.get("averageRating"),
                        "ratings_count": vi.get("ratingsCount"),
                    })
                return result
            except: return []

        def fetch_gut():
            try: return fetch_gutenberg_books(search_term, page=page)[:per_page] or []
            except: return []
        
        def fetch_ol():
            try: return fetch_openlib_books(search_term, limit=per_page, offset=(page-1)*per_page) or []
            except: return []

        def fetch_ia():
            try: return fetch_archive_books(search_term, limit=per_page) or []
            except: return []

        def fetch_it():
            try: return fetch_itbook_books(search_term, page=page, limit=per_page) or []
            except: return []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_google),
                executor.submit(fetch_gut),
                executor.submit(fetch_ol),
                executor.submit(fetch_ia),
                executor.submit(fetch_it)
            ]
            for future in futures:
                try:
                    res = future.result(timeout=12)
                    if res: all_books.extend(res)
                except: pass

        # Remove duplicates
        seen = set()
        unique = []
        for b in all_books:
            t = (b.get('title') or '').lower().strip()
            if t and t not in seen:
                seen.add(t)
                unique.append(b)

        return jsonify({
            'success': True,
            'category': category_id,
            'page': page,
            'books': unique
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@api_books_bp.route('/recommend-by-book', methods=['GET'])
def recommend_by_book():
    """
    توصيات كتب مشابهة بناءً على عنوان كتاب
    GET /api/books/recommend-by-book?title=Python&limit=24
    """
    title = request.args.get('title', '').strip()
    limit = request.args.get('limit', 24, type=int)
    
    if not title:
        return jsonify({'success': False, 'error': 'Missing title parameter'}), 400
    
    all_books = []
    
    def fetch_google():
        try:
            res = fetch_google_books(title, max_results=limit)
            items = res[0] if isinstance(res, tuple) else res
            return [{'id': b.get('id',''), 'title': b.get('title',''), 'author': b.get('author',''),
                     'desc': b.get('desc',''), 'cover_url': b.get('cover_url',''),
                     'rating': b.get('rating', 0), 'source': 'google',
                     'categories': b.get('categories', []),
                     'publisher': b.get('publisher',''), 'language': b.get('language','')}
                    for b in (items if isinstance(items, list) else [])]
        except: return []
    
    def fetch_gut():
        try: return fetch_gutenberg_books(title, page=1)[:limit] or []
        except: return []
    
    def fetch_ol():
        try: return fetch_openlib_books(title, limit=limit) or []
        except: return []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_google), executor.submit(fetch_gut), executor.submit(fetch_ol)]
        for future in futures:
            try:
                res = future.result(timeout=10)
                if res: all_books.extend(res)
            except: pass
    
    # Deduplicate
    seen = set()
    unique = []
    for b in all_books:
        t = (b.get('title') or '').lower().strip()
        if t and t not in seen and t.lower() != title.lower():
            seen.add(t)
            unique.append(b)
    
    return jsonify({'success': True, 'books': unique[:limit]})


@api_books_bp.route('/mood-recommendations', methods=['GET'])
def mood_recommendations():
    """
    توصيات كتب بناءً على المزاج
    GET /api/books/mood-recommendations?mood=happy&limit=12
    """
    mood = request.args.get('mood', '').strip()
    limit = request.args.get('limit', 12, type=int)
    
    mood_queries = {
        'happy': 'comedy feel good humor',
        'adventurous': 'adventure thriller action',
        'romantic': 'romance love poetry',
        'intellectual': 'philosophy science academic',
        'mysterious': 'mystery detective suspense',
        'relaxed': 'mindfulness meditation calm',
        'sad': 'drama emotional tragedy',
        'curious': 'science history biography discoveries',
        'calm': 'meditation mindfulness peace',
    }
    
    query = mood_queries.get(mood, mood if mood else 'popular books')
    
    all_books = []
    
    def fetch_google():
        try:
            res = fetch_google_books(query, max_results=limit)
            items = res[0] if isinstance(res, tuple) else res
            return [{'id': b.get('id',''), 'title': b.get('title',''), 'author': b.get('author',''),
                     'desc': b.get('desc',''), 'cover_url': b.get('cover_url',''),
                     'rating': b.get('rating', 0), 'source': 'google',
                     'categories': b.get('categories', []),
                     'publisher': b.get('publisher',''), 'language': b.get('language','')}
                    for b in (items if isinstance(items, list) else [])]
        except: return []
    
    def fetch_gut():
        try: return fetch_gutenberg_books(query, page=1)[:limit] or []
        except: return []
    
    def fetch_ol():
        try: return fetch_openlib_books(query, limit=limit) or []
        except: return []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_google), executor.submit(fetch_gut), executor.submit(fetch_ol)]
        for future in futures:
            try:
                res = future.result(timeout=10)
                if res: all_books.extend(res)
            except: pass
    
    # Deduplicate
    seen = set()
    unique = []
    for b in all_books:
        t = (b.get('title') or '').lower().strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(b)
    
    return jsonify({'success': True, 'mood': mood, 'books': unique[:limit]})


@api_books_bp.route('/<gid>/reviews', methods=['GET'])
def get_book_reviews(gid: str):
    """
    جلب مراجعات كتاب معين
    GET /api/books/<gid>/reviews
    """
    from flask_book_recommendation.models import BookReview, User
    
    reviews = (BookReview.query
               .filter_by(google_id=gid)
               .order_by(BookReview.created_at.desc())
               .limit(50)
               .all())
    
    result = []
    for r in reviews:
        user = User.query.get(r.user_id)
        result.append({
            'id': r.id,
            'user_id': r.user_id,
            'user_name': user.name if user else 'مستخدم',
            'google_id': r.google_id,
            'rating': r.rating,
            'review_text': r.review_text or '',
            'likes_count': r.likes_count or 0,
            'dislikes_count': r.dislikes_count or 0,
            'created_at': r.created_at.isoformat() if r.created_at else None,
        })
    
    return jsonify({
        'success': True,
        'total': len(result),
        'reviews': result
    })


@api_books_bp.route('/event', methods=['POST'])
@jwt_required()
def log_event():
    """
    POST /api/books/event
    تسجيل حدث تفاعل المستخدم مع كتاب (view, click, read, abandon, share, rate).
    """
    from flask_book_recommendation.models import UserEvent
    from flask_book_recommendation.extensions import db

    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    event_type = data.get('event_type')
    if not event_type or event_type not in ('view', 'click', 'read', 'abandon', 'share', 'rate'):
        return jsonify({'success': False, 'error': 'Invalid or missing event_type'}), 400

    book_google_id = data.get('book_google_id')
    if not book_google_id:
        return jsonify({'success': False, 'error': 'Missing book_google_id'}), 400

    try:
        import json
        metadata_json = None
        if data.get('metadata'):
            metadata_json = json.dumps(data['metadata'], ensure_ascii=False)

        event = UserEvent(
            user_id=user_id,
            event_type=event_type,
            book_google_id=book_google_id,
            session_id=data.get('session_id'),
            duration_seconds=data.get('duration_seconds'),
            scroll_depth=data.get('scroll_depth'),
            metadata_json=metadata_json,
        )
        db.session.add(event)
        db.session.commit()

        return jsonify({'success': True, 'event_id': event.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
