# routes/api/books.py
"""
API Books endpoints - البحث والتوصيات وتفاصيل الكتب
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, jwt_required
from ...recommender import (
    get_homepage_sections,
    get_trending,
    get_mood_based_recommendations,
    get_recommendations_by_title,
    MOOD_MAPPING
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
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'يرجى إدخال كلمة البحث (q)'
        }), 400
    
    # البحث في المصادر المتعددة بالتوازي
    all_books = []
    
    def fetch_google():
        try:
            return fetch_google_books(query, max_results=per_page, start_index=(page-1)*per_page) or []
        except:
            return []
    
    def fetch_gut():
        try:
            return fetch_gutenberg_books(query, page=page)[:per_page] if page == 1 else []
        except:
            return []
    
    def fetch_ol():
        try:
            return fetch_openlib_books(query, page=page)[:per_page] if page == 1 else []
        except:
            return []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fetch_google),
            executor.submit(fetch_gut),
            executor.submit(fetch_ol)
        ]
        
        for future in futures:
            try:
                result = future.result(timeout=10)
                if result:
                    all_books.extend(result)
            except:
                pass
    
    # إزالة التكرارات بناءً على العنوان
    seen_titles = set()
    unique_books = []
    for book in all_books:
        title = book.get('title', '').lower()
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
    التوصيات المخصصة للمستخدم
    GET /api/books/recommendations
    Headers: Authorization: Bearer <token>
    """
    user_id = int(get_jwt_identity())
    
    try:
        sections = get_homepage_sections(user_id=user_id)
        return jsonify({
            'success': True,
            'sections': sections
        })
    except Exception as e:
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
        {"id": "programming", "name": "برمجة", "name_en": "Programming"},
        {"id": "ai", "name": "ذكاء اصطناعي", "name_en": "Artificial Intelligence"},
        {"id": "fiction", "name": "روايات", "name_en": "Fiction"},
        {"id": "science", "name": "علوم", "name_en": "Science"},
        {"id": "history", "name": "تاريخ", "name_en": "History"},
        {"id": "psychology", "name": "علم نفس", "name_en": "Psychology"},
        {"id": "business", "name": "أعمال", "name_en": "Business"},
        {"id": "self-help", "name": "تطوير ذات", "name_en": "Self-Help"},
        {"id": "philosophy", "name": "فلسفة", "name_en": "Philosophy"},
        {"id": "biography", "name": "سير ذاتية", "name_en": "Biography"}
    ]
    
    return jsonify({
        'success': True,
        'categories': categories
    })


@api_books_bp.route('/category/<category_id>', methods=['GET'])
def get_books_by_category(category_id: str):
    """
    كتب حسب التصنيف
    GET /api/books/category/programming?page=1
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    try:
        books = fetch_google_books(
            category_id, 
            max_results=per_page, 
            start_index=(page-1)*per_page
        ) or []
        
        return jsonify({
            'success': True,
            'category': category_id,
            'page': page,
            'books': books
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'books': []
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
