# routes/main.py
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import current_user, login_required
import numpy as np
import pandas as pd
import requests
from ..models import BookStatus
import random
import time

logger = logging.getLogger(__name__)



from ..extensions import db, csrf, cache
import threading
import time
# في أعلى ملف main.py
from ..models import Book, UserRatingCF, BookEmbedding, UserPreference, SearchHistory, BookReview, BookQuote
# استيراد الدوال الموحدة
from ..utils import (
    fetch_openlib_detail, fetch_gutenberg_detail, fetch_archive_detail, fetch_itbook_detail, fetch_book_details,
    get_text_embedding, generate_book_embedding_if_missing,
    fetch_google_books, fetch_gutenberg_books, fetch_openlib_books, fetch_archive_books,
    fetch_itbook_books,
    translate_to_english_with_gemini,
    chat_with_ai  # مساعد AI للكتب
)
from ..recommender import (
    log_user_view, 
    get_deep_learning_recommendations,
    get_trending,
    get_top_rated,
    get_cf_similar,
    get_view_based_recommendations,
    get_behavior_based_recommendations
)
from ai_book_recommender.unified_pipeline import get_unified_engine




main_bp = Blueprint("main", __name__)


def _get_user_interests(user_id):
    """
    Get ALL user interests from both UserGenre and UserPreference.
    Returns a set of lowercase interest/topic strings.
    """
    if not user_id:
        return set()
    
    from ..models import UserGenre, Genre, UserPreference
    interests = set()
    
    # From UserGenre (explicit genre selections)
    user_genres = (
        db.session.query(Genre.name)
        .join(UserGenre)
        .filter(UserGenre.user_id == user_id)
        .all()
    )
    for (name,) in user_genres:
        interests.add(name.lower().strip())
    
    # From UserPreference (broader topic interests, behavioral)
    user_prefs = UserPreference.query.filter_by(user_id=user_id).filter(
        UserPreference.weight >= 5.0  # Only significant interests
    ).all()
    for pref in user_prefs:
        interests.add(pref.topic.lower().strip())
    
    return interests


def _get_book_uid(book):
    """
    Get a consistent unique ID for a book object or dictionary.
    Favors google_id string over local integer ID.
    """
    if isinstance(book, dict):
        gid = book.get("google_id") or book.get("id") or book.get("book_id")
        # If it's a digit string, it's likely a local ID, try title|author fallback if no gid
        if isinstance(gid, str) and gid.isdigit():
            return f"id_{gid}"
        return str(gid) if gid else f"title_{book.get('title')}_{book.get('author')}"
    else:
        # SQLAlchemy Model
        gid = getattr(book, 'google_id', None) or getattr(book, 'id', None)
        return str(gid)

def _strict_interest_filter(books, user_interests, limit=100):
    """
    🔒 STRICT INTEREST GATEKEEPER
    
    Filters a list of book recommendations to ONLY include books 
    that match at least one of the user's interests.
    
    Matching criteria:
    - Book's categories contain an interest keyword
    - Book's title contains an interest keyword
    
    This is a 100% strict filter — no random/unrelated books pass through.
    
    Args:
        books: List of book dicts with 'categories', 'title', etc.
        user_interests: Set of lowercase interest strings
        limit: Maximum number of books to return
    
    Returns:
        Filtered list of books matching user interests
    """
    if not user_interests:
        return books[:limit]  # No interests set = return as-is (cold start)
    
    filtered = []
    seen_ids = set()
    
    for book in books:
        if len(filtered) >= limit:
            break
            
        # 🧪 [DEDUPLICATION FIX] Skip if already seen in this filter pass
        uid = _get_book_uid(book)
        if uid in seen_ids:
            continue
        
        # Get book's categories
        cats_raw = book.get("categories", [])
        if isinstance(cats_raw, str):
            cats_raw = [c.strip() for c in cats_raw.split(",")]
        
        # Build searchable text from categories + title
        cat_text = " ".join(cats_raw).lower() if cats_raw else ""
        title_text = (book.get("title") or "").lower()
        author_text = (book.get("author") or "").lower()
        search_text = f"{cat_text} {title_text} {author_text}"
        
        # Check if ANY user interest matches
        matched = False
        for interest in user_interests:
            # Split multi-word interests for flexible matching
            interest_words = interest.split()
            if len(interest_words) > 1:
                # Multi-word: check if the full phrase appears
                if interest in search_text:
                    matched = True
                    break
            else:
                # Single word: check if it appears as a word
                if interest in search_text:
                    matched = True
                    break
        
        if matched:
            filtered.append(book)
            seen_ids.add(uid)
    
    return filtered


def _fetch_interest_books_from_api(interests, limit_per_interest=10):
    """
    Fetch books from Google Books API based on user interests.
    Used as fallback when local DB doesn't have enough matching books.
    
    Returns list of book dicts.
    """
    import os
    api_key = os.environ.get("GOOGLE_BOOKS_API_KEY")
    results = []
    seen_ids = set()
    
    for interest in list(interests)[:5]:  # Limit to 5 interests
        try:
            params = {
                "q": f"subject:{interest}",
                "maxResults": limit_per_interest,
                "orderBy": "relevance",
                "printType": "books",
            }
            if api_key:
                params["key"] = api_key
            
            import requests as _req
            resp = _req.get("https://www.googleapis.com/books/v1/volumes", 
                          params=params, timeout=10)
            if not resp.ok:
                continue
            
            for item in resp.json().get("items", []):
                gid = item.get("id")
                if not gid or gid in seen_ids:
                    continue
                seen_ids.add(gid)
                
                vi = item.get("volumeInfo", {})
                imgs = vi.get("imageLinks", {}) or {}
                cover = imgs.get("thumbnail") or imgs.get("smallThumbnail")
                if cover:
                    cover = cover.replace("http://", "https://").replace("&edge=curl", "")
                
                cats = vi.get("categories", [interest.title()])
                
                results.append({
                    "id": gid,
                    "title": vi.get("title", ""),
                    "author": ", ".join(vi.get("authors", ["Unknown"])),
                    "cover_url": cover,
                    "categories": cats,
                    "algorithm_tag": "INTEREST MATCH",
                    "algo_tag": "Interest-Based",
                    "confidence": 0.9,
                    "score": 0.85,
                    "rating": vi.get("averageRating", 4.5)
                })
        except Exception as e:
            logger.warning(f"[InterestAPI] Error fetching '{interest}': {e}")
            continue
    
    return results


@main_bp.route("/")
def home():
    """
    الصفحة الرئيسية — مع فلتر صارم بالاهتمامات لقسم Recommended for You.
    """
    from flask import make_response
    
    user_id = current_user.id if current_user.is_authenticated else None
    cache_key = f"home_full_{user_id or 'anon'}"
    
    # ⚡ محاولة جلب من الكاش أولاً (تسريع 90%+)
    cached_resp = cache.get(cache_key)
    if cached_resp:
        return cached_resp
    
    # ── Get user interests for strict filtering ──
    user_interests = _get_user_interests(user_id) if user_id else set()
    
    # جلب كتب متنوعة للأقسام المختلفة
    if user_id:
        featured = get_deep_learning_recommendations(user_id, limit=100)
    else:
        featured = get_trending(limit=100)
        
    top_rated = get_top_rated(limit=100)
    
    # Only use real top_rated books from users, no fallback.
    # We removed the trending fallback loop as per user request.
    most_viewed = get_trending(limit=100)
    
    if not featured or len(featured) < 3:
        if user_id and user_interests:
            # Fetch from Google Books based on interests instead of generic
            items_all = []
            for interest in list(user_interests)[:3]:
                items, _ = fetch_google_books(f"subject:{interest}", max_results=6)
                items_all.extend(items)
            featured = []
            for it in items_all:
                vi = it.get("volumeInfo", {})
                featured.append({
                    "id": it.get("id"),
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors", ["Unknown"])),
                    "cover": (vi.get("imageLinks", {})).get("thumbnail"),
                    "rating": vi.get("averageRating", 4.5)
                })
        else:
            items, _ = fetch_google_books("featured books architecture", max_results=12)
            featured = []
            for it in items:
                vi = it.get("volumeInfo", {})
                featured.append({
                    "id": it.get("id"),
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors", ["Unknown"])),
                    "cover": (vi.get("imageLinks", {})).get("thumbnail"),
                    "rating": vi.get("averageRating", 4.5)
                })
            
    if not most_viewed or len(most_viewed) < 3:
        most_viewed = featured[:10]
 
    # قائمة التصنيفات الأساسية
    categories = [
        "Programming", "Artificial Intelligence", "Science", 
        "History", "Philosophy", "Art", "Fiction",
        "Psychology", "Business", "Self-Help", "Travel", "Religion"
    ]
 
    # ═══════════════════════════════════════════════════════════════
    # 🔒 STRICT INTEREST-BASED RECOMMENDATIONS (ai_algo_books)
    # Only books matching user interests pass through
    # ═══════════════════════════════════════════════════════════════
    ai_algo_books = []
    mind_metrics_percentage = 68
    try:
        # 1. Get Neural Hybrid (Featured)
        engine = get_unified_engine()
        ctx = {"page": "home_showcase", "time": time.time()}
        neural_recs = engine.recommend_full_stack(user_id=user_id, top_k=100, context=ctx)
        
        # 2. Get DL and CF candidates
        dl_recs = get_deep_learning_recommendations(user_id, limit=100)
        cf_recs = get_cf_similar(user_id, top_n=100) if user_id else []
        
        # Calculate Dynamic Mind Metrics
        if user_id:
            mind_metrics_percentage = min(50 + len(dl_recs) + len(cf_recs), 98)
        else:
            import time as _time
            mind_metrics_percentage = 65 + (int(_time.time() / 3600) % 30)
            
        # Combine all candidates into ai_algo_books with tags
        seen_ids = set()
        
        # Featured (Neural)
        for b in (neural_recs or []):
            bid = b.get('id') or b.get('google_id')
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                ai_algo_books.append({
                    "id": bid,
                    "title": b.get('title'),
                    "author": b.get('author') or (b.get('authors') or ["Unknown"])[0],
                    "cover_url": b.get('cover_url') or b.get('cover'),
                    "categories": b.get('categories', []),
                    "algorithm_tag": b.get('algo_tag', 'NEURAL HYBRID'),
                    "algo_tag": b.get('algo_tag', 'Neural Full Stack'),
                    "confidence": b.get('confidence', 0),
                    "score": b.get('score', 0),
                    "rating": b.get('rating', 4.8)
                })
 
        # Supporters (DL, CF)
        candidates = []
        for b in dl_recs: candidates.append((b, "DEEP LEARNING"))
        for b in cf_recs: candidates.append((b, "COLLECTIVE"))
 
        for b, tag in candidates:
            if len(ai_algo_books) >= 200: break  # Gather more for filtering
            bid = b.get('id') if isinstance(b, dict) else (b.google_id or b.id)
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                if isinstance(b, dict):
                    ai_algo_books.append({
                        "id": bid,
                        "title": b.get('title'),
                        "author": b.get('author') or (b.get('authors') or ["Unknown"])[0],
                        "cover_url": b.get('cover_url') or b.get('cover'),
                        "categories": b.get('categories', []),
                        "algorithm_tag": tag,
                        "algo_tag": b.get('algo_tag', tag),
                        "confidence": b.get('confidence', 0),
                        "score": b.get('score', 0),
                        "rating": b.get('rating', 4.5)
                    })
                else:
                    ai_algo_books.append({
                        "id": bid,
                        "title": b.title,
                        "author": b.author or "Unknown",
                        "cover_url": b.cover_url,
                        "categories": b.categories.split(",") if b.categories else [],
                        "algorithm_tag": tag,
                        "algo_tag": tag,
                        "confidence": 0,
                        "score": 0,
                        "rating": 4.5
                    })
 
        # ── 🔒 STRICT INTEREST GATEKEEPER ──
        if user_id and user_interests:
            pre_filter_count = len(ai_algo_books)
            ai_algo_books = _strict_interest_filter(ai_algo_books, user_interests, limit=100)
            logger.debug(
                f"[InterestGatekeeper] Filtered {pre_filter_count} -> {len(ai_algo_books)} books "
                f"for user {user_id} with interests: {user_interests}"
            )
            
            # If strict filter left too few books, fetch from Google Books API
            if len(ai_algo_books) < 5:
                logger.debug(f"[InterestGatekeeper] Only {len(ai_algo_books)} books after filter, fetching from API...")
                api_books = _fetch_interest_books_from_api(user_interests, limit_per_interest=10)
                for ab in api_books:
                    if len(ai_algo_books) >= 100:
                        break
                    
                    # 🧪 [DEDUPLICATION FIX] Robust ID check
                    uid = _get_book_uid(ab)
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        ai_algo_books.append(ab)
 
    except Exception as e:
        import traceback
        logger.error(f"Error fetching AI recommendations: {str(e)}\n{traceback.format_exc()}")
        if not ai_algo_books and user_id and user_interests:
            # Fallback: fetch from API based on interests
            ai_algo_books = _fetch_interest_books_from_api(user_interests, limit_per_interest=8)
 
    # Get pipeline metadata for frontend visualization
    from ai_book_recommender.unified_pipeline import UnifiedRecommendationPipeline
    pipeline_meta = UnifiedRecommendationPipeline.get_pipeline_meta()
 
    resp = make_response(render_template(
        "home.html",
        featured_books=featured,
        categories=categories,
        unified_recommendations=[],
        algo_buckets={},
        top_rated_books_sorted=top_rated,
        most_viewed_books=most_viewed,
        trending_by_libraries=[],
        featured_book=featured[0] if featured else None,
        ai_algo_books=ai_algo_books,
        mind_metrics_percentage=mind_metrics_percentage,
        pipeline_meta=pipeline_meta,
        current_filters={'query': '', 'sort': 'ai_relevance', 'debug_ts': time.time()}
    ))
    # ⚡ Cache-Control: reduced to 1 minute for faster interest updates
    resp.headers["Cache-Control"] = "private, max-age=60"
    
    # ⚡ Store in server cache for 1 minute (was 3 min)
    cache.set(cache_key, resp, timeout=60)
    return resp

@main_bp.route("/api/pipeline-status")
def pipeline_status():
    """API endpoint: returns the last pipeline execution metadata for frontend visualization."""
    from flask import jsonify
    from ai_book_recommender.unified_pipeline import UnifiedRecommendationPipeline
    meta = UnifiedRecommendationPipeline.get_pipeline_meta()
    return jsonify(meta)

@main_bp.route("/feed/home")
def home_feed():
    """
    API endpoint — delivers homepage sections powered by the FULL Neural Stack.
    Each section calls a different strategy variant of the UnifiedRecommendationPipeline.
    """
    import time as _time
    from flask import jsonify, render_template, current_app
    import uuid

    user_id = current_user.id if current_user.is_authenticated else None
    
    # ⚡ Cache home_feed for 2 minutes
    feed_cache_key = f"home_feed_{user_id or 'anon'}"
    cached_feed = cache.get(feed_cache_key)
    if cached_feed:
        return cached_feed
    
    session_id = request.cookies.get("session", str(uuid.uuid4()))
    now_ts = _time.time()

    # ── Initialize Unified Engine (lazy, one-time) ──
    from ai_book_recommender.unified_pipeline import get_unified_engine
    engine = get_unified_engine()
    if engine.flask_app is None:
        engine.flask_app = current_app._get_current_object()

    # ── Build context ──
    ctx = {
        "page": "home",
        "device": "web",
        "time": now_ts,
        "session": session_id,
    }

    # ── Get user's top interest for display ──
    top_interest = _get_user_top_interest(user_id)

    # ── ⚡ FAST PATH: Compute "Because You Searched" section FIRST (fast DB query) ──
    dynamic_search_books = []
    last_search_term = "Self Growth"
    try:
        if user_id:
            last_search_obj = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).first()
            if last_search_obj and last_search_obj.query:
                last_search_term = last_search_obj.query.title()
                
        if not last_search_term or len(last_search_term) < 2:
            last_search_term = "Self Growth"

        dynamic_search_books = Book.query.filter(
            db.or_(
                Book.categories.ilike(f'%{last_search_term}%'),
                Book.title.ilike(f'%{last_search_term}%')
            )
        ).order_by(Book.id.desc()).limit(4).all()
        
        if not dynamic_search_books:
             dynamic_search_books = get_trending(limit=4)
             last_search_term = "Trending Now"
    except Exception as e:
        current_app.logger.error(f"[Feed] Search section failed: {str(e)}")
        if not dynamic_search_books:
            dynamic_search_books = get_trending(limit=4)
        last_search_term = "Hand-picked for You" if last_search_term == "Self Growth" else last_search_term

    # ═══════════════════════════════════════════════════════════════════
    # UNIFIED ASYNC EXECUTION FOR ALL RECOMMENDATION SECTIONS
    # ═══════════════════════════════════════════════════════════════════
    # UNIFIED ASYNC EXECUTION FOR ALL RECOMMENDATION SECTIONS
    # ═══════════════════════════════════════════════════════════════════

    from ..recommender import (
        get_top_rated, get_cf_similar,
        get_deep_learning_recommendations,
    )
    from ..recommender.exploration import UCB1Explorer
    from ..recommender.mood import get_mood_based_recommendations, MOOD_MAPPING
    
    import random
    
    # Pick a random mood for the user
    mood_keys = list(MOOD_MAPPING.keys())
    user_mood_key = random.choice(mood_keys)
    mood_info = MOOD_MAPPING[user_mood_key]

    try:
        from .explore import get_trending_by_libraries, get_most_viewed_books_custom
    except ImportError:
        get_trending_by_libraries = lambda limit: []
        get_most_viewed_books_custom = lambda limit: []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_safe(app_obj, func, *args, **kwargs):
        with app_obj.app_context():
            return func(*args, **kwargs)

    app_obj = current_app._get_current_object()
    cat_results = {}
    neural_sections = {}

    # ── FIX: Pre-warm the cache sequentially to prevent a cache stampede ──
    # DISABLED for faster home load - let parallel workers handle it
    # try:
    #     engine.recommend_full_stack(user_id=user_id, top_k=100, context=ctx)
    # except Exception as e:
    #     current_app.logger.warning(f"Cache pre-warm failed: {e}")

    with ThreadPoolExecutor(max_workers=10) as ex:  # ⚡ Increased for better parallelism
        futs = {
            # Basic AI - disabled randomization for consistency
            ex.submit(_run_safe, app_obj, get_deep_learning_recommendations, user_id, limit=100, randomize=False): ("cat", "deep_learning"),
            
            # Interactive / Community
            ex.submit(_run_safe, app_obj, get_mood_based_recommendations, mood_key=user_mood_key, limit=100): ("cat", "mood_ai"),
            ex.submit(_run_safe, app_obj, get_cf_similar, user_id=user_id, top_n=100): ("cat", "similar_minds"),
            
            # Stats for "Hot Right Now"
            ex.submit(_run_safe, app_obj, get_top_rated, limit=100): ("cat", "top_rated"),
            ex.submit(_run_safe, app_obj, get_most_viewed_books_custom, limit=100): ("cat", "most_viewed"),
            
            # Neural Engine (Now parallelized to prevent hanging)
            ex.submit(_run_safe, app_obj, engine.recommend_full_stack, user_id=user_id, top_k=100, context=ctx): ("neural", "recommended_for_you"),
            ex.submit(_run_safe, app_obj, engine.recommend_trending, user_id=user_id, top_k=100, context=ctx): ("neural", "trending_for_you"),
            ex.submit(_run_safe, app_obj, engine.recommend_because_you_read, user_id=user_id, top_k=100, context=ctx): ("neural", "because_you_read"),
            ex.submit(_run_safe, app_obj, engine.recommend_top_neural, user_id=user_id, top_k=100, context=ctx): ("neural", "top_neural_picks"),
            ex.submit(_run_safe, app_obj, engine.recommend_graph_discovery, user_id=user_id, top_k=100, context=ctx): ("neural", "graph_discovery"),
        }
        try:
            for f in as_completed(futs, timeout=8.0):  # ⚡ Reduced from 12s to 8s
                type_, name = futs[f]
                try:
                    res = f.result() or []
                    if type_ == "cat":
                        cat_results[name] = res
                    else:
                        neural_sections[name] = res
                except Exception as e:
                    current_app.logger.error(f"[Async] {name} failed: {e}")
                    if type_ == "cat":
                        cat_results[name] = []
                    else:
                        neural_sections[name] = []
        except Exception as e:
            current_app.logger.error(f"[Async] Timeout or Executor Error: {e}")
            pass
                        
    # Combine Top Rated and Most Viewed into "Hot Right Now"
    hot_now = []
    tr = cat_results.get("top_rated", [])
    mv = cat_results.get("most_viewed", [])
    
    # Simple weave: one from each
    for i in range(max(len(tr), len(mv))):
        if i < len(mv): hot_now.append(mv[i])
        if i < len(tr): hot_now.append(tr[i])
        
    # Remove duplicates
    seen = set()
    hot_now_unique = []
    for b in hot_now:
        bid = b.get("id") or b.get("google_id")
        if bid not in seen:
            seen.add(bid)
            hot_now_unique.append(b)
    cat_results["hot_right_now"] = hot_now_unique[:100]

    # ── Build Featured Lists ──
    featured_lists = _build_featured_lists()

    # ── 🔒 Apply Strict Interest Gatekeeper to neural sections ──
    user_interests = _get_user_interests(user_id) if user_id else set()
    if user_id and user_interests:
        # Filter "recommended_for_you" section strictly by interests
        if "recommended_for_you" in neural_sections:
            raw_recs = neural_sections["recommended_for_you"]
            neural_sections["recommended_for_you"] = _strict_interest_filter(
                raw_recs, user_interests, limit=100
            )
            logger.debug(
                f"[FeedGatekeeper] recommended_for_you: {len(raw_recs)} -> "
                f"{len(neural_sections['recommended_for_you'])}"
            )
            
            # If too few after filtering, supplement from Google Books API
            if len(neural_sections["recommended_for_you"]) < 3:
                api_supplement = _fetch_interest_books_from_api(user_interests, limit_per_interest=6)
                
                # 🧪 [DEDUPLICATION FIX] Check against existing books before extending
                rec_section = neural_sections["recommended_for_you"]
                existing_uids = { _get_book_uid(b) for b in rec_section }
                
                for ab in api_supplement:
                    uid = _get_book_uid(ab)
                    if uid not in existing_uids:
                        rec_section.append(ab)
                        existing_uids.add(uid)
        
        # Also filter deep_learning section
        dl_books = cat_results.get("deep_learning", [])
        if dl_books:
            cat_results["deep_learning"] = _strict_interest_filter(
                dl_books, user_interests, limit=100
            )

    # ── Render template ──
    html = render_template(
        "components/home_feed.html",
        neural_sections=neural_sections,
        top_interest=top_interest,
        dynamic_search_books=dynamic_search_books,
        last_search_term=last_search_term,
        
        # Elite Sections
        deep_learning_books=cat_results.get("deep_learning", []),
        mood_ai_books=cat_results.get("mood_ai", []),
        mood_info=mood_info,
        similar_minds=cat_results.get("similar_minds", []),
        hot_right_now=cat_results.get("hot_right_now", []),
        featured_lists=featured_lists,
    )
    resp = jsonify({"success": True, "html": html})
    resp.headers["Cache-Control"] = "private, max-age=60"
    
    # ⚡ Cache the feed response for 1 minute (reduced from 2 min for faster updates)
    cache.set(feed_cache_key, resp, timeout=60)
    return resp


def _get_user_top_interest(user_id):
    """Get the user's top interest topic for display."""
    from ..models import UserPreference, UserGenre, Genre
    top_interest = "AI & Discovery"
    if user_id:
        try:
            best_pref = UserPreference.query.filter_by(user_id=user_id).order_by(
                UserPreference.weight.desc()
            ).first()
            if best_pref:
                top_interest = best_pref.topic
            else:
                best_genre = (
                    db.session.query(Genre.name)
                    .join(UserGenre)
                    .filter(UserGenre.user_id == user_id)
                    .first()
                )
                if best_genre:
                    top_interest = best_genre[0]
        except Exception:
            pass
    return top_interest


def _build_featured_lists():
    """
    Build curated 'Featured Lists' for the homepage (Goodreads-style cards).
    Fetches books from Google Books API in parallel for high-quality covers.
    """
    from flask import current_app
    from ..extensions import cache
    from ..utils import fetch_google_books
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests as _requests
    
    # Using a fresh cache key
    cache_key = 'home_featured_lists_v9'
    try:
        cached_lists = cache.get(cache_key)
        if cached_lists:
            return cached_lists
    except Exception:
        pass

    lists = []
    colors = ['#b8a9e8', '#c8e6c9', '#ffe0b2', '#b3e5fc', '#f8bbd0', '#d1c4e9', '#c5cae9', '#dcedc8', '#a5d6a7', '#ffcc80', '#90caf9']

    category_configs = [
        {'title': 'Epic Fantasy', 'subject': 'epic fantasy', 'cat': 'Fantasy'},
        {'title': 'Astrophysics', 'subject': 'astrophysics', 'cat': 'Science'},
        {'title': 'Philosophy of Life', 'subject': 'philosophy', 'cat': 'Philosophy'},
        {'title': 'Architecture Design', 'subject': 'architecture design', 'cat': 'Art'},
        {'title': 'Future of AI', 'subject': 'artificial intelligence', 'cat': 'Technology'},
        {'title': 'Human Psychology', 'subject': 'psychology', 'cat': 'Psychology'},
        {'title': 'Arabic Literature', 'subject': 'arabic literature', 'cat': 'Fiction'},
        {'title': 'Ancient History', 'subject': 'ancient history', 'cat': 'History'},
        {'title': 'Entrepreneurship', 'subject': 'entrepreneurship', 'cat': 'Business'},
        {'title': 'Crime Thriller', 'subject': 'crime thriller', 'cat': 'Mystery'},
        {'title': 'Personal Growth', 'subject': 'personal growth', 'cat': 'Self-Help'},
        {'title': 'Travel Literature', 'subject': 'travel literature', 'cat': 'Travel'},
        {'title': 'Classic Literature', 'subject': 'classic literature', 'cat': 'Fiction'},
        {'title': 'Culinary Arts', 'subject': 'culinary arts', 'cat': 'Cooking'},
        {'title': 'Modern Architecture', 'subject': 'modern architecture', 'cat': 'Art'},
        {'title': 'Software Engineering', 'subject': 'software engineering', 'cat': 'Technology'},
        {'title': 'Holistic Health', 'subject': 'holistic health', 'cat': 'Health'},
        {'title': 'Space Exploration', 'subject': 'space exploration', 'cat': 'Science'},
        {'title': 'Emotional Intelligence', 'subject': 'emotional intelligence', 'cat': 'Psychology'},
        {'title': 'Mythology', 'subject': 'mythology', 'cat': 'History'},
    ]

    app_obj = current_app._get_current_object()
    GOOGLE_API_URL = "https://www.googleapis.com/books/v1/volumes"
    GOOGLE_API_KEY = app_obj.config.get('GOOGLE_BOOKS_API_KEY') or __import__('os').environ.get('GOOGLE_BOOKS_API_KEY')

    def _fetch_category(idx, cfg):
        covers = []
        books = []
        total_items = 0
        with app_obj.app_context():
            from ..models import Book
            from ..extensions import db
            try:
                # Broader search for thematic collections - search in title or general terms
                params = {
                    "q": cfg['subject'],
                    "maxResults": 40,
                    "orderBy": "relevance",
                    "printType": "books",
                }
                if GOOGLE_API_KEY:
                    params["key"] = GOOGLE_API_KEY

                r = _requests.get(GOOGLE_API_URL, params=params, timeout=10)
                if r.ok:
                    data = r.json()
                    items = data.get("items", [])
                    total_items = data.get("totalItems", 0)
                    for item in items:
                        vi = item.get("volumeInfo", {}) or {}
                        imgs = vi.get("imageLinks", {}) or {}
                        # Try multiple image sizes for highest quality
                        cover = (
                            imgs.get("medium")
                            or imgs.get("large")
                            or imgs.get("thumbnail")
                            or imgs.get("smallThumbnail")
                        )
                        if cover:
                            # Upgrade to https and zoom=1 (higher res), remove curl edge
                            if cover.startswith("http://"):
                                cover = "https://" + cover[7:]
                            cover = cover.replace("zoom=5", "zoom=1").replace("&edge=curl", "")
                            if cover not in covers:
                                covers.append(cover)
                                books.append({
                                    'id': item.get('id'),
                                    'title': vi.get('title', 'Unknown'),
                                    'author': ', '.join(vi.get('authors', ['Unknown'])),
                                    'cover': cover
                                })
                        if len(covers) >= 3:
                            break
            except Exception as e:
                current_app.logger.warning(f"[FeaturedLists] Google API error for {cfg['subject']}: {e}")

            # ── Fallback 1: Local DB ──
            if len(covers) < 3:
                try:
                    db_query = f"%{cfg['subject']}%"
                    db_books = Book.query.filter(
                        db.or_(
                            Book.categories.ilike(db_query),
                            Book.title.ilike(db_query)
                        ),
                        Book.cover_url.isnot(None),
                        Book.cover_url != ""
                    ).limit(20).all()
                    for b in db_books:
                        if b.cover_url and b.cover_url.startswith('http') and b.cover_url not in covers:
                            covers.append(b.cover_url)
                            books.append({
                                'id': b.google_id or b.id,
                                'title': b.title,
                                'author': b.author or 'Unknown',
                                'cover': b.cover_url
                            })
                        if len(covers) >= 3:
                            break
                except Exception:
                    pass

            # ── Fallback 2: General Relevant fallback if still empty ──
            if len(covers) < 3:
                try:
                    # Just search for the category title itself (often in Arabic) or general books
                    fallback_params = {"q": cfg['title'], "maxResults": 10}
                    if GOOGLE_API_KEY: fallback_params["key"] = GOOGLE_API_KEY
                    r2 = _requests.get(GOOGLE_API_URL, params=fallback_params, timeout=5)
                    if r2.ok:
                        for item in r2.json().get("items", []):
                            vi = item.get("volumeInfo", {}) or {}
                            c = (vi.get("imageLinks", {}) or {}).get("thumbnail")
                            if c and c not in covers:
                                c_clean = c.replace("http://", "https://").replace("&edge=curl", "")
                                covers.append(c_clean)
                                books.append({
                                    'id': item.get('id'),
                                    'title': vi.get('title', 'Unknown'),
                                    'author': ', '.join(vi.get('authors', ['Unknown'])),
                                    'cover': c_clean
                                })
                            if len(covers) >= 3: break
                except Exception:
                    pass

            if len(covers) >= 1:
                return {
                    'index': idx,
                    'title': cfg['title'],
                    'covers': covers,
                    'books': books,
                    'count': total_items if total_items > 0 else 20,
                    'url': f"/public/books?cat={cfg['cat']}",
                    'color': colors[idx % len(colors)]
                }
            return None

    try:
        results = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_fetch_category, i, cfg) for i, cfg in enumerate(category_configs)]
            for f in as_completed(futs, timeout=30):
                res = f.result()
                if res:
                    results.append(res)
        
        # Sort back to original order
        results.sort(key=lambda x: x['index'])
        for r in results:
            del r['index']
            lists.append(r)
            
        # Ensure we have at least 6 lists by adding generic ones if needed
        if len(lists) < 6:
            try:
                from ..models import Book
                from ..extensions import db
                import random
                # Add a "Popular Picks" list from DB
                pop_books = Book.query.filter(Book.cover_url.isnot(None), Book.cover_url != '').limit(50).all()
                while len(lists) < 6 and len(pop_books) >= 3:
                    chosen = random.sample(pop_books, 3)
                    lists.append({
                        'title': 'مختارات شائعة',
                        'covers': [b.cover_url for b in chosen],
                        'count': 100,
                        'url': '/public/books',
                        'color': '#6366f1'
                    })
            except Exception:
                pass

        # Save to cache
        if len(lists) > 0:
            try:
                cache.set(cache_key, lists, timeout=86400)
            except Exception:
                pass
                
    except Exception as e:
        current_app.logger.error(f"[FeaturedLists API] Error: {e}")

    logger.debug(f"[FeaturedLists] Built {len(lists)} lists")
    return lists





def _generate_home_data(user_id):
    """
    Legacy helper — kept for backward compatibility with background refresh.
    Delegates to the unified neural engine.
    """
    from flask import current_app
    from ai_book_recommender.unified_pipeline import get_unified_engine
    import uuid, time as _time

    engine = get_unified_engine()
    if engine.flask_app is None:
        engine.flask_app = current_app._get_current_object()

    ctx = {"page": "home", "device": "web", "time": _time.time(), "session": str(uuid.uuid4())}
    top_interest = _get_user_top_interest(user_id)

    recs = engine.recommend_full_stack(user_id=user_id, top_k=30, context=ctx)

    # Build simple algo_buckets from unified results for backward compat
    algo_buckets = {
        'search_history_results': [],
        'interest_results': [],
        'hybrid_results': recs[:10] if recs else [],
        'transformer_results': recs[10:20] if len(recs) > 10 else [],
        'collaborative_results': [],
        'graph_results': [],
        'vector_results': [],
        'reranker_results': recs[20:30] if len(recs) > 20 else [],
    }

    from ..recommender import get_top_rated
    try:
        from .explore import get_trending_by_libraries, get_most_viewed_books_custom
    except ImportError:
        get_trending_by_libraries = lambda limit: []
        get_most_viewed_books_custom = lambda limit: []

    return (
        recs,
        algo_buckets,
        get_top_rated(limit=20),
        get_most_viewed_books_custom(limit=20),
        get_trending_by_libraries(limit=20),
        top_interest,
    )

def _refresh_background(app, user_id):
    """Background task to refresh cache."""
    with app.app_context():
        try:
            data = _generate_home_data(user_id)
            cache_key = f"home_recs_{user_id}" if user_id else "home_anon"
            ttl = 90 if user_id else 300
            # Update cache with new timestamp
            cache.set(cache_key, (data, time.time()), timeout=ttl)
            logging.getLogger(__name__).info(f"Background refresh complete for user {user_id}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Background refresh failed: {e}")


@main_bp.route("/browse")
def browse():
    """
    Explore page for specific categories (See All).
    """
    category = request.args.get('category', 'unified')
    try: limit = int(request.args.get('limit', 100))
    except ValueError: limit = 100
    try: offset = int(request.args.get('offset', 0))
    except ValueError: offset = 0
    
    user_id = current_user.id if current_user.is_authenticated else None
    
    # Check cache
    cache_key = f"browse_{category}_{limit}_{offset}_{user_id or 'anon'}"
    cached_books = cache.get(cache_key)
    if cached_books is not None:
        books = cached_books
        title = "Browse Books"
        description = "Explore our collection"
        if category == 'top_rated':
            title = "Highest Rated by Community"
            description = "Books with the highest average ratings from our users."
        elif category == 'most_viewed':
            title = "Most Viewed This Week"
            description = "The most popular books currently being viewed by our community."
        elif category == 'trending_libs':
            title = "Trending in User Libraries"
            description = "Books that are frequently being added to user collections recently."
        elif category == 'unified':
            title = "Unified AI Picks"
            description = "Top recommendations curated by our specific AI ensemble for you."
    else:
        # Imports inside function to avoid circular dependency
        from ..recommender import (
            get_trending, get_top_rated, 
            get_deep_learning_recommendations, get_behavior_based_recommendations,
            get_cf_similar
        )
        from .explore import get_most_viewed_books_custom, get_trending_by_libraries
        from flask import current_app
        
        books = []
        title = "Browse Books"
        description = "Explore our collection"

    if category == 'top_rated':
        title = "Highest Rated by Community"
        description = "Books with the highest average ratings from our users."
        books = get_top_rated(limit=limit)
        
    elif category == 'most_viewed':
        title = "Most Viewed This Week"
        description = "The most popular books currently being viewed by our community."
        books = get_most_viewed_books_custom(limit=limit)
        
    elif category == 'trending_libs':
        title = "Trending in User Libraries"
        description = "Books that are frequently being added to user collections recently."
        books = get_trending_by_libraries(limit=limit)
        
    elif category == 'unified':
        title = "Unified AI Picks"
        description = "Top recommendations curated by our specific AI ensemble for you."
        if user_id:
            try:
                from concurrent.futures import ThreadPoolExecutor
                from ..recommender import get_topic_based # Import here


                # Helper to run safely
                def run_safe(app, func, *args, **kwargs):
                    try: 
                        with app.app_context():
                            res = func(*args, **kwargs)
                            # Ensure result is a list
                            # get_topic_based returns dict with 'books' key
                            if isinstance(res, dict) and 'books' in res:
                                return res['books']
                            return res if isinstance(res, list) else []
                    except Exception as e:
                        logger.error(f"Error in browse thread: {e}") 
                        return []
                
                app_obj = current_app._get_current_object()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    f1 = executor.submit(run_safe, app_obj, get_behavior_based_recommendations, user_id, limit=100, randomize=True)
                    f2 = executor.submit(run_safe, app_obj, get_deep_learning_recommendations, user_id, limit=100, randomize=True)
                    f3 = executor.submit(run_safe, app_obj, get_cf_similar, user_id, top_n=100, randomize=True)
                    f4 = executor.submit(run_safe, app_obj, get_topic_based, user_id, limit=100, randomize=True) # Added Interest Match
                    
                    res1 = f1.result(timeout=15) or []
                    res2 = f2.result(timeout=15) or []
                    res3 = f3.result(timeout=15) or []
                    res4 = f4.result(timeout=15) or []
                    
                    logger.info(f"Browse Debug: Hybrid={len(res1)}, DL={len(res2)}, CF={len(res3)}, Topic={len(res4)}")

                    # Combine and deduplicate
                    combined = res1 + res2 + res3 + res4
                    seen = set()
                    books = []
                    
                    # Helper to safely get ID/score from dict or object
                    def get_val(item, key, default=None):
                        if isinstance(item, dict):
                            return item.get(key, default)
                        else:
                            return getattr(item, key, default)

                    for b in combined:
                        if not b: continue
                        bid = get_val(b, 'id')
                        if bid and bid not in seen:
                            seen.add(bid)
                            books.append(b)
                    
                    # Sort by score/confidence safely with float conversion
                    def safe_score(x):
                        try:
                            val = get_val(x, 'score', 0) or get_val(x, 'confidence', 0)
                            return float(val)
                        except (ValueError, TypeError):
                            return 0.0

                    # 1. Sort by Score first to get quality
                    books.sort(key=safe_score, reverse=True)
                    
                    # Randomization logic: Shuffling disabled for static feel on most algos
                    # Only Interest Match (Topic) contributes dynamic content now
                    # (Keep the sorted order by AI quality)
                    books = books[:offset+limit+20] # Take sufficient buffer
                    
                    logger.info(f"Browse Debug: Post-Shuffle Count={len(books)}")

            except Exception as e:
                 logger.error(f"Browse Sort/Processing Error: {e}", exc_info=True)
                 # If sort fails, we still have 'books' populated (hopefully)
                 if not books:
                     books = []
            
            # Fallback if AI fails or returns nothing
            if not books:
                 logger.warning("Browse Debug: Triggering Fallback to Trending")
                 books = get_trending(limit=limit)
                 logger.info(f"Browse Debug: Fallback Count={len(books)}")
            
            # Additional Random Shuffle if results are small to force change
            if len(books) > 0 and len(books) < 20:
                 random.shuffle(books)
                 
        # Save to cache if books were found
        if books:
             cache.set(cache_key, books, timeout=600 if user_id else 1800)

        else:
             books = get_trending(limit=limit)

    return render_template("browse.html", books=books, title=title, description=description)


# ---------------------------------------------------------------------------
#                 خوارزمية Collaborative Filtering
# ---------------------------------------------------------------------------
def get_cf_recommendations(user_id: int, top_n: int = 8):
    try:
        ratings = UserRatingCF.query.all()
        if not ratings: return []

        rows = [{"user_id": r.user_id, "google_id": r.google_id, "rating": float(r.rating)} for r in ratings]
        df = pd.DataFrame(rows)

        if df.empty or len(df) < 2: return []

        pivot = df.pivot_table(index="user_id", columns="google_id", values="rating", aggfunc="mean").fillna(0.0)
        if user_id not in pivot.index: return []

        u_vec = pivot.loc[user_id].values.astype(np.float32)
        u_norm = np.linalg.norm(u_vec) + 1e-8
        all_mat = pivot.values.astype(np.float32)
        norms = np.linalg.norm(all_mat, axis=1) + 1e-8
        sims = (all_mat @ u_vec) / (norms * u_norm)

        sim_series = pd.Series(sims, index=pivot.index)
        sim_series = sim_series.drop(labels=[user_id], errors="ignore")
        sim_series = sim_series.sort_values(ascending=False).head(10)

        if sim_series.empty: return []

        sim_users = sim_series.index.values
        sim_scores = sim_series.values
        sim_matrix = pivot.loc[sim_users].values
        weighted = sim_matrix.T @ sim_scores
        scores = pd.Series(weighted, index=pivot.columns)
        
        user_rated = df[df["user_id"] == user_id]["google_id"].unique()
        scores = scores.drop(labels=list(user_rated), errors="ignore")
        
        scores = scores.sort_values(ascending=False).head(top_n)
        recommended_ids = list(scores.index)

        if not recommended_ids: return []

        recommended_books = []
        for gid in recommended_ids:
            book_sample = Book.query.filter_by(google_id=gid).first()
            if book_sample: recommended_books.append(book_sample)
                
        return recommended_books
    except Exception as e:
        print(f"CF Error: {e}")
        return []


# ---------------------------------------------------------------------------
#                           مكتبة المستخدم (كتبي)
# ---------------------------------------------------------------------------

# في ملف routes/main.py
# تأكد من استيراد دالة normalize_text إذا وضعتها في utils
# from ..utils import normalize_text 
# أو إذا وضعتها في نفس الملف كدالة عادية، اتركها كما هي.

@main_bp.get("/books")
@login_required
def books():
    # المدخلات من الـ GET
    q      = request.args.get("q", "").strip()
    sort   = request.args.get("sort", "")
    source = request.args.get("source", "")

    # 1) قاعدة البحث الأساسية (بدون فلترة العنوان هنا)
    query = Book.query.filter_by(owner_id=current_user.id)

    # ============ الفلترة حسب المصدر ============
    if source == "google":
        query = query.filter(Book.google_id.isnot(None))
    elif source == "local":
        query = query.filter(Book.google_id.is_(None))

    # ============ الفرز ============
    if sort == "new":
        query = query.order_by(Book.created_at.desc())
    elif sort == "alpha":
        query = query.order_by(Book.title.asc())
    elif sort == "rating":
        query = query.outerjoin(UserRatingCF, UserRatingCF.google_id == Book.google_id)
        query = query.group_by(Book.id)
        query = query.order_by(db.func.avg(UserRatingCF.rating).desc())
    
    # جلب جميع الكتب المطابقة للشروط السابقة
    my_books = query.all()

    # ============ 🔥 الإصلاح هنا: البحث الذكي داخل بايثون ============
    if q:
        # تعريف دالة التوحيد هنا إذا لم تستوردها من utils
        import re
        def normalize_local(text):
            if not text: return ""
            text = str(text).lower().strip()
            text = re.sub("[أإآ]", "ا", text)
            text = re.sub("ة", "ه", text)
            text = re.sub("ى", "ي", text)
            return text

        search_term = normalize_local(q)
        
        # تصفية القائمة يدوياً لضمان ظهور النتائج بغض النظر عن الهمزات
        filtered_books = []
        for book in my_books:
            book_title_norm = normalize_local(book.title)
            book_author_norm = normalize_local(book.author)
            
            # البحث في العنوان أو اسم المؤلف
            if search_term in book_title_norm or search_term in book_author_norm:
                filtered_books.append(book)
        
        my_books = filtered_books

    # ============ القوائم الثلاث (المفضلة وغيرها) ============
    statuses = BookStatus.query.filter_by(user_id=current_user.id).all()
    
    # 🆕 إنشاء قاموس للحالات والتقدم
    status_map = {}
    for s in statuses:
        status_map[s.book_id] = {
            'status': s.status,
            'progress': s.reading_progress or 0,
            'last_read': s.last_read_at,
            'created_at': s.created_at
        }

    favorites_ids = [s.book_id for s in statuses if s.status == "favorite"]
    later_ids     = [s.book_id for s in statuses if s.status == "later"]
    finished_ids  = [s.book_id for s in statuses if s.status == "finished"]
    reading_ids   = [s.book_id for s in statuses if s.status == "reading"]
    on_hold_ids   = [s.book_id for s in statuses if s.status == "on_hold"]
    dropped_ids   = [s.book_id for s in statuses if s.status == "dropped"]

    favorites = Book.query.filter(Book.id.in_(favorites_ids)).all() if favorites_ids else []
    later     = Book.query.filter(Book.id.in_(later_ids)).all()     if later_ids else []
    finished  = Book.query.filter(Book.id.in_(finished_ids)).all()  if finished_ids else []
    reading_books = Book.query.filter(Book.id.in_(reading_ids)).all() if reading_ids else []
    on_hold_books = Book.query.filter(Book.id.in_(on_hold_ids)).all() if on_hold_ids else []
    dropped_books = Book.query.filter(Book.id.in_(dropped_ids)).all() if dropped_ids else []
    
    # 🆕 إضافة بيانات الحالة والتقدم لكل كتاب
    for book in my_books:
        book_status_data = status_map.get(book.id, {})
        book.status = book_status_data.get('status')
        book.reading_progress = book_status_data.get('progress', 0)
        book.last_read_at = book_status_data.get('last_read')
        book.status_created_at = book_status_data.get('created_at')
        
        # حساب وقت القراءة المقدر (بافتراض 2 دقيقة لكل صفحة)
        if book.page_count:
            book.estimated_read_time = book.page_count * 2  # دقائق
        else:
            book.estimated_read_time = None
    
    # 🆕 إحصائيات القراءة الشاملة والمتقدمة
    
    # حساب الصفحات المقروءة فعلياً (الكتب المنتهية + نسبة التقدم في الباقي)
    pages_read = 0
    for b in my_books:
        pc = b.page_count or 0
        prog = status_map.get(b.id, {}).get('progress', 0)
        if status_map.get(b.id, {}).get('status') == 'finished':
            pages_read += pc
        elif prog > 0:
            pages_read += int(pc * prog / 100)
    
    # أسرع كتاب أنهيته (بناءً على started_at و finished_at)
    fastest_book = None
    fastest_days = None
    for s in statuses:
        if s.status == 'finished' and s.started_at and s.finished_at:
            days = (s.finished_at - s.started_at).days
            if days >= 0 and (fastest_days is None or days < fastest_days):
                fastest_days = days
                fb = Book.query.get(s.book_id)
                if fb:
                    fastest_book = {'title': fb.title, 'days': days, 'cover': fb.cover_url}
    
    # هدف القراءة السنوي
    reading_goal = current_user.reading_goal or 0
    goal_progress = 0
    if reading_goal > 0:
        goal_progress = min(100, round(len(finished_ids) / reading_goal * 100))
    
    reading_stats = {
        'total_books': len(my_books),
        'finished_count': len(finished_ids),
        'favorite_count': len(favorites_ids),
        'later_count': len(later_ids),
        'reading_count': len(reading_ids),
        'on_hold_count': len(on_hold_ids),
        'dropped_count': len(dropped_ids),
        'in_progress': len([b for b in my_books if status_map.get(b.id, {}).get('progress', 0) > 0 and status_map.get(b.id, {}).get('progress', 0) < 100]),
        'total_pages': sum(b.page_count or 0 for b in my_books),
        'pages_read': pages_read,
        'reading_hours': round(pages_read * 2 / 60, 1),  # 2 دقيقة/صفحة
        'avg_progress': round(sum(status_map.get(b.id, {}).get('progress', 0) for b in my_books) / max(len(my_books), 1), 1),
        'fastest_book': fastest_book,
        'reading_goal': reading_goal,
        'goal_progress': goal_progress,
    }
    
    # 🆕 جمع التصنيفات الفريدة للفلترة
    all_categories = set()
    all_languages = set()
    category_distribution = {}
    for book in my_books:
        if book.categories:
            for cat in book.categories.split(','):
                cat_clean = cat.strip()
                if cat_clean:
                    all_categories.add(cat_clean)
                    category_distribution[cat_clean] = category_distribution.get(cat_clean, 0) + 1
        if book.language:
            all_languages.add(book.language)

    # Get user ratings map for quick display
    user_ratings = {}
    if my_books:
        google_ids = [b.google_id for b in my_books if b.google_id]
        if google_ids:
            ratings = UserRatingCF.query.filter(
                UserRatingCF.user_id == current_user.id,
                UserRatingCF.google_id.in_(google_ids)
            ).all()
            for r in ratings:
                user_ratings[r.google_id] = r.rating

    from datetime import datetime as dt_now
    current_year = dt_now.now().year

    return render_template(
        "books.html",
        books=my_books,
        favorites=favorites,
        later=later,
        finished=finished,
        reading_books=reading_books,
        on_hold_books=on_hold_books,
        dropped_books=dropped_books,
        reading_stats=reading_stats,
        status_map=status_map,
        all_categories=sorted(all_categories),
        all_languages=sorted(all_languages),
        category_distribution=category_distribution,
        user_ratings=user_ratings,
        current_year=current_year,
    )


@main_bp.post("/books/update-reading-goal")
@login_required
def update_reading_goal():
    """Update the user's yearly reading goal."""
    goal = request.form.get("reading_goal", 0, type=int)
    if goal < 0:
        goal = 0
    if goal > 999:
        goal = 999
    current_user.reading_goal = goal
    db.session.commit()
    flash("Reading goal updated successfully! 🎯", "success")
    return redirect(url_for("main.books"))


def background_sync_book_details(app, book_id, google_id):
    """تحديث بيانات الكتاب في الخلفية لمنع تأخير عرض الصفحة"""
    with app.app_context():
        try:
            book = Book.query.get(book_id)
            if not book or not google_id: return
            
            from ..utils import fetch_book_details
            details = fetch_book_details(google_id)
            if not details: return
            
            changed = False
            if not book.published_date and details.get('publishedDate'):
                book.published_date = details.get('publishedDate')
                changed = True
            if not book.page_count and details.get('pageCount'):
                book.page_count = details.get('pageCount')
                changed = True
            if not book.categories and details.get('categories'):
                cats = details.get('categories')
                book.categories = ", ".join(cats) if isinstance(cats, list) else str(cats)
                changed = True
            if not book.publisher and details.get('publisher'):
                book.publisher = details.get('publisher')
                changed = True
            if not book.language and details.get('language'):
                book.language = details.get('language')
                changed = True
            
            if changed:
                db.session.commit()
                logger.debug(f"Background sync completed for book {book_id}")
        except Exception as e:
            logger.error(f"Background Sync Error: {e}")

@main_bp.route("/books/<int:book_id>")
@login_required
def book_detail(book_id):
    book = Book.query.get_or_404(book_id)

    # 🆕 تسجيل المشاهدة (Implicit Tracking)
    try:
        log_user_view(current_user.id, book)
    except Exception as e:
        logger.error(f"Failed to log view: {e}")
    
    # التحقق من الملكية (اختياري - حسب منطق التطبيق)
    # هنا نسمح برؤية أي كتاب، لكن التعديل مقيد
    
    # جلب تقييم المستخدم
    user_rating = UserRatingCF.query.filter_by(user_id=current_user.id, google_id=book.google_id).first()
    
    # جلب حالة الكتاب
    book_status_obj = BookStatus.query.filter_by(user_id=current_user.id, book_id=book.id).first()
    book_status = book_status_obj.status if book_status_obj else None
    
    # ---------------------------------------------------------
    # 🆕 التوصيات الهجينة (Hybrid Recommendations) - تم تعطيلها مؤقتاً لتسريع الصفحة
    # ---------------------------------------------------------
    similar = []
    author_books = []

    # جلب تفاصيل إضافية في الخلفية (لتسريع الاستجابة)
    if book.google_id:
        from flask import current_app
        import threading
        app_obj = current_app._get_current_object()
        threading.Thread(target=background_sync_book_details, args=(app_obj, book.id, book.google_id), daemon=True).start()
        
        # محاولة جلب التقييم العالمي من الكاش (لحظي إذا كان موجوداً)
        try:
            from ..utils import fetch_book_details
            details = fetch_book_details(book.google_id)
            if details:
                setattr(book, 'global_rating', details.get('rating'))
                setattr(book, 'global_ratings_count', details.get('ratingsCount'))
        except: pass

    # جلب المراجعات (للكتب المشتركة عبر Google ID أو المعرف المحلي)
    reviews = BookReview.query.filter(
        db.or_(
            BookReview.google_id == book.google_id if book.google_id else False,
            BookReview.google_id == str(book.id)
        )
    ).order_by(BookReview.created_at.desc()).limit(20).all()

    # التحقق من تفاعلات المستخدم الحالي (Likes/Dislikes)
    if current_user.is_authenticated:
        try:
            from ..models import ReviewReaction
            review_ids = [r.id for r in reviews]
            reactions = ReviewReaction.query.filter(
                ReviewReaction.review_id.in_(review_ids),
                ReviewReaction.user_id == current_user.id
            ).all()
            reaction_map = {rr.review_id: rr.reaction_type for rr in reactions}
            for r in reviews:
                r.user_reaction = reaction_map.get(r.id)
        except Exception as e:
            print(f"Error fetching review reactions: {e}")

    # جلب اقتباسات المستخدم
    quotes = BookQuote.query.filter_by(user_id=current_user.id, book_id=book.id).order_by(BookQuote.created_at.desc()).all()

    return render_template(
        "book_detail.html",
        book=book,
        user_rating=user_rating,
        book_status=book_status,
        status_entry=book_status_obj,
        similar=similar,
        author_books=author_books,
        reviews=reviews,
        quotes=quotes
    )


@main_bp.post("/books/<int:book_id>/notes")
@login_required
def save_notes(book_id):
    book = Book.query.get_or_404(book_id)
    if book.owner_id != current_user.id:
        # Check if user owns logic or return 403
        flash("غير مصرح لك بتعديل ملاحظات هذا الكتاب", "danger")
        return redirect(url_for("main.book_detail", book_id=book.id))
    
    notes = request.form.get("notes")
    book.notes = notes
    db.session.commit()
    flash("تم حفظ الملاحظات بنجاح ✨", "success")
    return redirect(url_for("main.book_detail", book_id=book.id))


@main_bp.post("/books/<int:book_id>/review")
@login_required
def add_review(book_id):
    book = Book.query.get_or_404(book_id)
    
    rating = request.form.get("rating", 5, type=int)
    content = request.form.get("content", "").strip()
    
    if not content:
        flash("يرجى كتابة مراجعة قبل الحفظ", "warning")
        return redirect(url_for("main.book_detail", book_id=book.id))
        
    # التحقق مما إذا كان هناك مراجعة سابقة
    review = BookReview.query.filter_by(user_id=current_user.id, book_id=book.id).first()
    
    if review:
        review.rating = rating
        review.review_text = content
        flash("تم تحديث مراجعتك بنجاح ✨", "success")
    else:
        review = BookReview(
            user_id=current_user.id,
            book_id=book.id,
            google_id=book.google_id,
            rating=rating,
            review_text=content
        )
        db.session.add(review)
        flash("تمت إضافة المراجعة بنجاح ✨", "success")
        
    db.session.commit()
    
    # تحديث اهتمامات المستخدم في الخلفية (اختياري)
    try:
        from .public import background_interest_update
        from flask import current_app
        import threading
        real_app = current_app._get_current_object()
        threading.Thread(target=background_interest_update, args=(
            real_app, 
            current_user.id, 
            book.title, 
            book.author, 
            content
        )).start()
    except: pass
    
    return redirect(url_for("main.book_detail", book_id=book.id))


@main_bp.route("/books/<int:book_id>/read")
@login_required
def book_read(book_id):
    book = Book.query.get_or_404(book_id)
    
    # 1. إذا كان هناك ملف محلي/رابط مباشر، نوجه المستخدم إليه
    if book.file_url:
        return redirect(book.file_url)
    
    # 2. إذا كان كتاب Google، نستخدم القارئ المدمج
    if book.google_id:
        # يمكننا إعادة توجيه المستخدم لصفحة القارئ العام
        # أو عرض نفس القالب هنا
        vi = {}
        target_link = ""
        try: 
            # محاولة جلب رابط المعاينة
            from ..utils import fetch_book_details
            d = fetch_book_details(book.google_id)
            if d:
                vi = d.get("volumeInfo", {})
                target_link = vi.get("previewLink") or vi.get("infoLink")
        except: pass
        
        if not target_link:
            target_link = f"https://books.google.com/books?id={book.google_id}"
        
        return render_template(
            "reader_frame.html", 
            book_title=book.title, 
            book_id=book.google_id,
            external_link=target_link
        )

    flash("لا يوجد ملف للقراءة لهذا الكتاب.", "warning")
    return redirect(url_for("main.book_detail", book_id=book.id))



@main_bp.post("/books/<int:book_id>/status/<status>")
@csrf.exempt
@login_required
def set_book_status(book_id, status):
    allowed_statuses = ['favorite', 'later', 'finished', 'reading', 'on_hold', 'dropped']
    if status not in allowed_statuses:
        flash("حالة غير معروفة", "danger")
        return redirect(url_for("main.book_detail", book_id=book_id))
        
    book = Book.query.get_or_404(book_id)
    
    # التقاط الوقت الحالي
    from datetime import datetime
    now = datetime.utcnow()

    # التحقق هل الحالة موجودة مسبقاً
    s = BookStatus.query.filter_by(user_id=current_user.id, book_id=book.id).first()
    
    if s:
        # إذا ضغط نفس الحالة -> حذف (Toggle)
        if s.status == status:
            db.session.delete(s)
            flash(f"تمت إزالة الكتاب من قائمة {status}", "info")
        else:
            # منطق تحديث التواريخ
            if status == 'reading' and s.status != 'reading':
                if not s.started_at:
                    s.started_at = now
            
            if status == 'finished' and s.status != 'finished':
                s.finished_at = now
                s.reading_progress = 100
            elif status != 'finished' and s.status == 'finished':
                s.finished_at = None # إعادة تعيين إذا خرج من المنتهية
                if status == 'reading':
                     s.reading_progress = s.reading_progress # Keep as is or reset? Usually keep.
                else:
                     pass

            # تغيير الحالة
            s.status = status
            flash(f"تم تغيير الحالة إلى {status}", "success")
    else:
        # إنشاء حالة جديدة
        s = BookStatus(user_id=current_user.id, book_id=book.id, status=status)
        
        if status == 'reading':
            s.started_at = now
        elif status == 'finished':
            s.finished_at = now
            s.reading_progress = 100
            
        db.session.add(s)
        flash(f"تمت الإضافة إلى قائمة {status}", "success")
        
    # --- 🆕 Online Learning Feedback Update ---
    try:
        from ai_book_recommender.engine import get_engine
        b_id_val = str(book.google_id or book.id)
        get_engine().record_feedback(
            user_id=current_user.id,
            item_id=b_id_val,
            feedback_type=status,
            value=1.0
        )
    except Exception as e_ol:
        import logging
        logging.getLogger(__name__).error(f"Online learning feedback error (status): {e_ol}")
    # ------------------------------------------
        
    db.session.commit()
    # return redirect(url_for("main.books"))
    return redirect(request.referrer or url_for("main.books"))


@main_bp.post("/books/<int:book_id>/progress")
@csrf.exempt
@login_required
def update_reading_progress(book_id):
    """تحديث نسبة تقدم القراءة للكتاب"""
    from flask import jsonify
    from datetime import datetime
    
    book = Book.query.get_or_404(book_id)
    
    try:
        progress = int(request.form.get("progress") or request.json.get("progress", 0))
    except (ValueError, TypeError):
        progress = 0
    
    # ضمان أن النسبة بين 0 و 100
    progress = max(0, min(100, progress))
    
    # البحث عن حالة الكتاب أو إنشاء واحدة جديدة
    status = BookStatus.query.filter_by(user_id=current_user.id, book_id=book.id).first()
    
    if not status:
        status = BookStatus(user_id=current_user.id, book_id=book.id, status="later")
        db.session.add(status)
    
    status.reading_progress = progress
    status.last_read_at = datetime.utcnow()
    
    # إذا وصل لـ 100% تلقائياً نحوله لـ finished
    if progress >= 100 and status.status != "finished":
        status.status = "finished"
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "progress": progress,
        "status": status.status
    })

@main_bp.post("/books/<int:book_id>/rate")
@login_required
def rate_book(book_id: int):
    book = Book.query.get_or_404(book_id)
    if not book.google_id:
        flash("لا يمكن تقييم كتاب محلي (بدون Google ID) لنظام التوصيات.", "warning")
        return redirect(url_for("main.book_detail", book_id=book.id))

    try: value = float(request.form.get("rating") or 0)
    except ValueError: value = 0.0
    if value < 1: value = 1
    if value > 5: value = 5

    r = UserRatingCF.query.filter_by(user_id=current_user.id, google_id=book.google_id).first()
    if r is None:
        r = UserRatingCF(user_id=current_user.id, google_id=book.google_id, rating=value)
        db.session.add(r)
        msg = "تم إضافة التقييم."
    else:
        r.rating = value
        msg = "تم تحديث التقييم."
        
    # --- 🆕 Online Learning Feedback Update ---
    try:
        from ai_book_recommender.engine import get_engine
        b_id_val = str(book.google_id or book.id)
        get_engine().record_feedback(
            user_id=current_user.id,
            item_id=b_id_val,
            feedback_type="rate",
            value=value
        )
    except Exception as e_ol:
        import logging
        logging.getLogger(__name__).error(f"Online learning feedback error (rate): {e_ol}")
    # ------------------------------------------
        
    db.session.commit()
    flash(msg, "success")
    return redirect(url_for("main.book_detail", book_id=book.id))


@main_bp.post("/books/create")
@login_required
@csrf.exempt
def create_book():
    google_id = request.form.get("google_id")
    action = request.form.get("action", "add")
    
    # Check if user already has this book by google_id
    if google_id:
        existing = Book.query.filter_by(owner_id=current_user.id, google_id=google_id).first()
        if existing:
            if action == 'favorite':
                from ..models import BookStatus
                status = BookStatus.query.filter_by(user_id=current_user.id, book_id=existing.id).first()
                if not status:
                    s = BookStatus(user_id=current_user.id, book_id=existing.id, status="favorite")
                    db.session.add(s)
                else:
                    status.status = "favorite"
                db.session.commit()
                if request.headers.get("HX-Request"):
                    return '<div class="bg-rose-500 text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg"><span class="material-symbols-outlined text-[28px]" style="font-variation-settings: \'FILL\' 1;">favorite</span></div>'
                flash("تمت إضافة الكتاب للمفضلة", "success")
                return redirect(url_for("main.books"))

            if request.headers.get("HX-Request"):
                return '<div class="bg-green-500 text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg"><span class="material-symbols-outlined text-[28px]" style="font-variation-settings: \'FILL\' 1;">check</span></div>'
            flash("هذا الكتاب موجود بالفعل في مكتبتك", "info")
            return redirect(url_for("main.books"))

    b = Book(
        title=request.form.get("title"), author=request.form.get("author"),
        description=request.form.get("description"), cover_url=request.form.get("cover_url") or None,
        google_id=google_id, owner_id=current_user.id
    )
    db.session.add(b)
    db.session.commit()
    
    if action == "favorite":
        from ..models import BookStatus
        s = BookStatus(user_id=current_user.id, book_id=b.id, status="favorite")
        db.session.add(s)
        db.session.commit()
        if request.headers.get("HX-Request"):
            return '<div class="bg-rose-500 text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg animate-pulse"><span class="material-symbols-outlined text-[28px]" style="font-variation-settings: \'FILL\' 1;">favorite</span></div>'
        flash("تمت إضافة الكتاب للمفضلة", "success")
        return redirect(url_for("main.books"))
        
    if request.headers.get("HX-Request"):
        return '<div class="bg-green-500 text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg animate-pulse"><span class="material-symbols-outlined text-[28px]" style="font-variation-settings: \'FILL\' 1;">check</span></div>'
        
    flash("تمت إضافة الكتاب", "success")
    return redirect(url_for("main.books"))


@main_bp.post("/books/<int:book_id>/notes")
@csrf.exempt
@login_required
def update_book_notes(book_id):
    """حفظ ملاحظات الكتاب عبر AJAX"""
    book = Book.query.get_or_404(book_id)
    if book.owner_id != current_user.id:
        return jsonify({"success": False, "error": "Unauthorized"}), 403
    
    notes = request.json.get("notes", "") if request.is_json else request.form.get("notes", "")
    book.notes = notes
    db.session.commit()
    return jsonify({"success": True, "notes": notes})

@main_bp.post("/books/<int:book_id>/update")
@login_required
def update_book(book_id: int):
    b = Book.query.get_or_404(book_id)
    if b.owner_id != current_user.id:
        flash("ليس لديك صلاحية", "danger"); return redirect(url_for("main.books"))
    b.title = request.form.get("u_title"); b.author = request.form.get("u_author")
    b.description = request.form.get("u_description"); b.cover_url = request.form.get("u_cover_url") or None
    b.file_url = request.form.get("u_file_url") or None
    db.session.commit(); flash("تم التحديث", "success")
    return redirect(url_for("main.books"))


@main_bp.post("/books/<int:book_id>/generate_cover")
@login_required
def generate_book_cover(book_id):
    """توليد غلاف للكتاب باستخدام AI"""
    book = Book.query.get_or_404(book_id)
    
    # التحقق من الملكية
    if book.owner_id != current_user.id:
        flash("غير مصرح لك بتعديل هذا الكتاب", "danger")
        return redirect(url_for("main.book_detail", book_id=book.id))
    
    # استدعاء دالة التوليد
    from ..utils import generate_ai_cover_url
    new_cover = generate_ai_cover_url(book.title, book.author)
    
    if new_cover:
        book.cover_url = new_cover
        db.session.commit()
        flash("تم توليد الغلاف بنجاح ✨", "success")
    else:
        flash("فشل توليد الغلاف", "error")
        
    return redirect(url_for("main.book_detail", book_id=book.id))


@main_bp.post("/books/<int:book_id>/delete")
@login_required
def delete_book(book_id: int):
    """
    حذف كتاب من مكتبة المستخدم نهائياً مع كافة البيانات المرتبطة به.
    """
    from ..models import (
        BookStatus, BookReview, BookQuote, UserBookView, 
        SearchHistory, BookEmbedding, BookGenre
    )
    
    # 1. العثور على الكتاب والتأكد من الملكية
    b = Book.query.get_or_404(book_id)
    if b.owner_id != current_user.id:
        flash("ليس لديك صلاحية لحذف هذا الكتاب.", "danger")
        return redirect(url_for("main.books"))
    
    try:
        # 2. حذف كافة السجلات المرتبطة (Cascading manual delete)
        # لتجنب أخطاء المفاتيح الخارجية (Foreign Key Constraints)
        
        # حذف الارتباط بالتصنيفات (Genres)
        BookGenre.query.filter_by(book_id=book_id).delete()
        
        # حذف الحالة (Status)
        BookStatus.query.filter_by(book_id=book_id).delete()
        
        # حذف المراجعات (Reviews)
        BookReview.query.filter_by(book_id=book_id).delete()
        
        # حذف الاقتباسات (Quotes)
        BookQuote.query.filter_by(book_id=book_id).delete()
        
        # حذف المشاهدات (Views)
        UserBookView.query.filter_by(book_id=book_id).delete()
        
        # حذف سجل البحث (Search History)
        db.session.query(SearchHistory).filter_by(book_id=book_id).delete()
        
        # حذف التضمينات (Embeddings)
        BookEmbedding.query.filter_by(book_id=book_id).delete()
        
        # 3. حذف سجل الكتاب نفسه
        db.session.delete(b)
        db.session.commit()
        
        flash(f"تم حذف كتاب '{b.title}' من مكتبتك بنجاح. ✨", "info")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting book {book_id}: {str(e)}", exc_info=True)
        flash(f"فشل حذف الكتاب بسبب خطأ في قاعدة البيانات: {str(e)}", "danger")
        
    return redirect(url_for("main.books"))


# ---------------------------------------------------------------------------
#                 إدارة الاقتباسات (Quotes)
# ---------------------------------------------------------------------------

@main_bp.post("/books/<int:book_id>/quotes")
@csrf.exempt
@login_required
def save_quote(book_id):
    """حفظ اقتباس جديد للكتاب"""
    from flask import jsonify
    
    book = Book.query.get_or_404(book_id)
    
    data = request.get_json() if request.is_json else request.form
    quote_text = data.get("quote_text", "").strip()
    
    if not quote_text:
        return jsonify({"success": False, "error": "الاقتباس فارغ"}), 400
    
    page_number = data.get("page_number")
    if page_number:
        try:
            page_number = int(page_number)
        except ValueError:
            page_number = None
    
    quote = BookQuote(
        user_id=current_user.id,
        book_id=book.id,
        google_id=book.google_id,
        quote_text=quote_text,
        page_number=page_number
    )
    db.session.add(quote)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "quote": {
            "id": quote.id,
            "text": quote.quote_text,
            "page": quote.page_number,
            "created_at": quote.created_at.strftime("%Y-%m-%d %H:%M")
        }
    })


@main_bp.delete("/quotes/<int:quote_id>")
@csrf.exempt
@login_required
def delete_quote(quote_id):
    """حذف اقتباس"""
    from flask import jsonify
    
    quote = BookQuote.query.get_or_404(quote_id)
    
    if quote.user_id != current_user.id:
        return jsonify({"success": False, "error": "غير مصرح"}), 403
    
    db.session.delete(quote)
    db.session.commit()
    
    return jsonify({"success": True})


@main_bp.get("/books/<int:book_id>/quotes")
@login_required
def get_quotes(book_id):
    """جلب اقتباسات الكتاب"""
    from flask import jsonify
    
    book = Book.query.get_or_404(book_id)
    
    quotes = BookQuote.query.filter_by(
        user_id=current_user.id,
        book_id=book.id
    ).order_by(BookQuote.created_at.desc()).all()
    
    return jsonify({
        "success": True,
        "quotes": [{
            "id": q.id,
            "text": q.quote_text,
            "page": q.page_number,
            "created_at": q.created_at.strftime("%Y-%m-%d %H:%M")
        } for q in quotes]
    })


# ... (باقي الكود كما هو) ...

# دالة الاستيراد المحدثة (مع الذكاء الاصطناعي)
@main_bp.post("/import/<gid>")
@csrf.exempt
@login_required
def import_book_generic(gid):
    # 1. جلب البيانات من المصدر المناسب
    data = None
    if gid.startswith("gut_"): data = fetch_gutenberg_detail(gid)
    elif gid.startswith("ia_"): data = fetch_archive_detail(gid)
    elif gid.startswith("ol_"): data = fetch_openlib_detail(gid)
    elif gid.isdigit() and len(gid) == 13: data = fetch_itbook_detail(gid)
    else: data = fetch_book_details(gid) # Google Books

    if not data:
        flash("فشل جلب بيانات الكتاب.", "danger")
        return redirect(url_for("explore.index")) # تم تعديل التوجيه لصفحة الاستكشاف

    # 2. التحقق من التكرار
    exists = Book.query.filter_by(owner_id=current_user.id, google_id=gid).first()
    if exists:
        flash("الكتاب موجود لديك مسبقاً.", "info")
        return redirect(url_for("main.books"))

    # 3. استخراج البيانات (التصحيح هنا) 🛠️
    # تهيئة المتغيرات
    title = data.get("title")
    author = data.get("author")
    desc = data.get("desc") or data.get("description")
    cover = data.get("cover")

    # معالجة خاصة لـ Google Books (لأن البيانات تكون داخل volumeInfo)
    if "volumeInfo" in data:
        vi = data["volumeInfo"]
        title = vi.get("title")
        author = ", ".join(vi.get("authors", [])) if vi.get("authors") else "Unknown"
        desc = vi.get("description")
        
        # استخراج الصورة من Google
        imgs = vi.get("imageLinks", {})
        cover = imgs.get("thumbnail") or imgs.get("smallThumbnail")

    # تحسين الروابط (تأكد أنها https)
    if cover and cover.startswith("http://"):
        cover = cover.replace("http://", "https://")

    # القيم الافتراضية إذا فشل كل شيء
    final_title = title or "Untitled"
    final_author = author or "Unknown"

    # إنشاء كائن الكتاب
    book = Book(
        title=final_title, 
        author=final_author, 
        description=desc, 
        cover_url=cover,
        owner_id=current_user.id, 
        google_id=gid 
    )

    db.session.add(book)
    db.session.commit()

    # 4. الذكاء الاصطناعي: حفظ البصمة (Embedding) تلقائياً
    try:
        generate_book_embedding_if_missing(book)
    except Exception as e:
        print(f"[AI Embedding] Non-critical error: {e}")

    flash("تمت الإضافة للمكتبة بنجاح.", "success")
    # توجيه المستخدم لمكتبته ليرى الكتاب الجديد
    return redirect(url_for("main.books"))


# ---------------------------------------------------------------------------
#                 (تم حذف البحث الذكي بناءً على الطلب)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#                 مساعد AI للكتب (Chatbot)
# ---------------------------------------------------------------------------

@main_bp.post("/ai/chat")
@csrf.exempt
def ai_chat():
    """
    API endpoint للمساعد الذكي
    يستقبل رسالة المستخدم ويرد بتوصيات كتب
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return {
                "reply": "مرحباً! أنا مكتبي، مساعدك الذكي للكتب 📚 كيف يمكنني مساعدتك؟",
                "books": []
            }
        
        # جمع سياق المستخدم
        user_context = None
        if current_user.is_authenticated:
            # جلب اهتمامات المستخدم
            prefs = UserPreference.query.filter_by(user_id=current_user.id).order_by(
                UserPreference.weight.desc()
            ).limit(5).all()
            
            # جلب آخر الكتب
            recent_books = Book.query.filter_by(owner_id=current_user.id).order_by(
                Book.created_at.desc()
            ).limit(3).all()
            
            user_context = {
                "interests": [p.topic for p in prefs],
                "recent_books": [b.title for b in recent_books]
            }
        
        # استدعاء AI
        result = chat_with_ai(user_message, user_context)
        
        return {
            "reply": result.get("reply", ""),
            "books": result.get("books", [])
        }
        
    except Exception as e:
        print(f"[AI Chat Route] Error: {e}")
        return {
            "reply": "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.",
            "books": []
        }, 500


