# -*- coding: utf-8 -*-
"""
Topic-based recommendations — from user interests, preferences, search history.
"""
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import (
    Book, UserPreference, SearchHistory, UserRatingCF, BookStatus
)
from ..extensions import db
from ..utils import (
    fetch_google_books, fetch_gutenberg_books,
    fetch_openlib_books, fetch_archive_books,
    fetch_itbook_books,
    translate_to_english_with_gemini
)
from .helpers import _extract_rating_with_fallback
from .trending import get_trending

logger = logging.getLogger(__name__)


def get_topic_based(user_id, limit=24, offset=0, prefs_limit=3, recent_query=None, randomize=False):
    """
    توصيات مبنية على اهتمامات المستخدم (Topic-Based Recommendations).
    """
    if not user_id:
        logger.warning(f"[Topic] No user_id provided, returning empty.")
        return {'books': [], 'interests_exhausted': True, 'total_interests': 0, 'current_page': 0}

    logger.info(f"[Topic] Getting topic-based recommendations for user {user_id}, limit={limit}, offset={offset}")

    topics = []
    seen_topics = set()
    potential_topics = []

    exclude_gids = set()
    try:
        user_books = Book.query.filter_by(owner_id=user_id).all()
        for b in user_books:
            if b.google_id: exclude_gids.add(b.google_id)
            
        statuses = BookStatus.query.filter_by(user_id=user_id).all()
        for s in statuses:
            if s.book and s.book.google_id:
                exclude_gids.add(s.book.google_id)
                
        ratings = UserRatingCF.query.filter_by(user_id=user_id).all()
        for r in ratings:
            if r.google_id: exclude_gids.add(r.google_id)
            
    except Exception as e:
        logger.error(f"[Topic] Error getting excluded books: {e}")

    if recent_query:
        potential_topics.append(recent_query)

    try:
        prefs = UserPreference.query.filter_by(user_id=user_id).order_by(UserPreference.weight.desc()).all()
        for p in prefs:
            if p.topic and not p.topic.startswith('special:'):
                potential_topics.append(p.topic)
    except Exception as e:
        logger.error(f"[Topic] prefs error: {e}", exc_info=True)

    try:
        last_search = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc(), SearchHistory.id.desc()).first()
        if last_search:
            potential_topics.append(last_search.query)
    except Exception as e:
        logger.error(f"[Topic] History error: {e}", exc_info=True)

    all_unique_topics = []
    for t in potential_topics:
        if not t or not t.strip():
            continue
        
        query_text = t.strip()
        topic_to_use = query_text
        
        if any("\u0600" <= c <= "\u06FF" for c in query_text):
            try:
                translated = translate_to_english_with_gemini(query_text)
                if translated and translated.strip():
                    topic_to_use = translated
                    logger.info(f"[Topic] Translated '{query_text}' to '{topic_to_use}'")
            except Exception as e:
                logger.warning(f"[Topic] Translation failed for '{query_text}': {e}")
        
        if topic_to_use.lower() not in seen_topics:
            all_unique_topics.append(topic_to_use)
            seen_topics.add(topic_to_use.lower())

    if randomize:
        discovery_pool = [
            "Best selling books 2024", "New York Times Best Sellers", "Man Booker Prize", 
            "Science Fiction Classics", "Must read biographies", "Self improvement trends",
            "Cyberpunk novels", "Psychological thrillers",
            "History of Science", "Modern Philosophy", "Artificial Intelligence Production"
        ]
        new_topics = random.sample(discovery_pool, 2)
        for t in new_topics:
            if t.lower() not in seen_topics:
                insert_pos = random.randint(1, len(all_unique_topics)) if len(all_unique_topics) > 0 else 0
                all_unique_topics.insert(insert_pos, t)
                seen_topics.add(t.lower())
                logger.info(f"[Topic] Injected discovery topic: '{t}'")

    if not all_unique_topics:
        logger.debug(f"[Topic] No topics found for user {user_id}")
        return []

    topics_per_page = 3
    
    if randomize:
        if len(all_unique_topics) > 3:
             start_shuffle = 1 if recent_query else 0
             pool_to_shuffle = all_unique_topics[start_shuffle:]
             random.shuffle(pool_to_shuffle)
             all_unique_topics = all_unique_topics[:start_shuffle] + pool_to_shuffle
             
    current_page = (offset // limit) if limit > 0 else 0
    start_topic_idx = current_page * topics_per_page
    
    topics = all_unique_topics[start_topic_idx:start_topic_idx + topics_per_page]
    
    next_page_start = (current_page + 1) * topics_per_page
    interests_exhausted = next_page_start >= len(all_unique_topics)
    
    if not topics:
        interests_exhausted = True
        start_topic_idx = max(0, len(all_unique_topics) - topics_per_page)
        topics = all_unique_topics[start_topic_idx:]
    
    logger.info(f"[Topic] Page {current_page + 1}: Using topics {start_topic_idx + 1}-{start_topic_idx + len(topics)} of {len(all_unique_topics)}: {topics} (exhausted: {interests_exhausted})")
    all_books = []
    seen_ids = set()
    
    per_topic_limit = max(4, int(limit / len(topics))) if topics else limit
    
    api_page = 1
    
    global_offset = 0
    if randomize:
        global_offset = random.randint(0, 200)
        api_page = random.randint(1, 10)
    
    def process_google_result(items, topic):
        books = []
        for it in items or []:
            if not isinstance(it, dict): continue
            gid = it.get("id")
            if not gid: continue
            vi = it.get("volumeInfo") or {}
            title = vi.get("title")
            if not title or not title.strip(): continue
            img = (vi.get("imageLinks") or {}).get("thumbnail")
            if img:
                if img.startswith("http://"): img = img.replace("http://", "https://")
                if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')
            
            rating = _extract_rating_with_fallback(vi)
            ratings_count = vi.get("ratingsCount")
            
            books.append({
                "id": gid, "title": title,
                "author": ", ".join(vi.get("authors") or []),
                "cover": img, "source": "Google Books",
                "reason": f"🎯 لأنك بحثت مؤخراً عن «{topic}»",
                "rating": rating, "ratings_count": ratings_count,
            })
        return books
    
    def fetch_all_sources_for_topic(topic, per_source, topic_offset, topic_page):
        results = []
        
        def fetch_google():
            try:
                gb_res = fetch_google_books(topic, max_results=per_source, start_index=topic_offset)
                items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
                return ("google", process_google_result(items, topic))
            except Exception as e:
                logger.error(f"[Topic] Google error for '{topic}': {e}")
                return ("google", [])
        
        def fetch_itbook():
            try:
                books = fetch_itbook_books(topic, limit=per_source, page=topic_page) or []
                return ("itbook", [{
                    "id": b.get("id"), "title": b.get("title"),
                    "author": b.get("author"), "cover": b.get("cover"),
                    "source": "IT Bookstore",
                    "reason": f"🎯 كتب تقنية: «{topic}»",
                } for b in books if b.get("id") and b.get("title")])
            except Exception as e:
                logger.error(f"[Topic] ITBook error for '{topic}': {e}")
                return ("itbook", [])
        
        def fetch_openlib():
            try:
                books = fetch_openlib_books(topic, limit=per_source, offset=topic_offset) or []
                return ("openlib", [{
                    "id": b.get("id"), "title": b.get("title"),
                    "author": b.get("author"), "cover": b.get("cover"),
                    "source": "OpenLibrary",
                    "reason": f"🎯 OpenLibrary: «{topic}»",
                } for b in books if b.get("id") and b.get("title")])
            except Exception as e:
                logger.error(f"[Topic] OpenLib error for '{topic}': {e}")
                return ("openlib", [])
        
        def fetch_archive():
            try:
                books = fetch_archive_books(topic, limit=per_source) or []
                return ("archive", [{
                    "id": b.get("id"), "title": b.get("title"),
                    "author": b.get("author"), "cover": b.get("cover"),
                    "source": "Internet Archive",
                    "reason": f"📚 من أرشيف الإنترنت: «{topic}»",
                } for b in books if b.get("id") and b.get("title")])
            except Exception as e:
                logger.error(f"[Topic] Archive error for '{topic}': {e}")
                return ("archive", [])
        
        def fetch_gutenberg():
            try:
                books = fetch_gutenberg_books(topic, limit=per_source, page=topic_page) or []
                return ("gutenberg", [{
                    "id": b.get("id"), "title": b.get("title"),
                    "author": b.get("author"), "cover": b.get("cover"),
                    "source": "Project Gutenberg",
                    "reason": f"📖 كلاسيكيات: «{topic}»",
                } for b in books if b.get("id") and b.get("title")])
            except Exception as e:
                logger.error(f"[Topic] Gutenberg error for '{topic}': {e}")
                return ("gutenberg", [])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_google),
                executor.submit(fetch_itbook),
                executor.submit(fetch_openlib),
                executor.submit(fetch_archive),
                executor.submit(fetch_gutenberg),
            ]
            
            try:
                for future in as_completed(futures, timeout=2.5):
                    try:
                        source, books = future.result(timeout=2.0)
                        results.extend(books)
                    except Exception as e:
                        logger.error(f"[Topic] Future error for '{topic}': {e}")
            except TimeoutError:
                logger.warning(f"[Topic] ⏱️ Timeout fetching sources for '{topic}', using partial results")
        
        return results
    
    def _fetch_topic_books(t, index):
        current_limit = per_topic_limit + 2 if index == 0 else per_topic_limit
        per_source = max(4, int(current_limit / 3))
        logger.debug(f"[Topic] Searching for '{t}' with limit {per_source}, offset {global_offset}, page {api_page}")
        return fetch_all_sources_for_topic(t, per_source, global_offset, api_page)

    with ThreadPoolExecutor(max_workers=len(topics)) as executor:
        topic_futures = [executor.submit(_fetch_topic_books, t, i) for i, t in enumerate(topics)]
        
        for future in as_completed(topic_futures, timeout=3.5):
            try:
                topic_books = future.result()
                for book in topic_books:
                    bid = book.get("id")
                    title = book.get("title")
                    
                    if title:
                        title = str(title).strip()
                        book["title"] = title
                    
                    if not bid or bid in seen_ids or bid in exclude_gids:
                        continue
                    
                    if title and (title.startswith('<') and '>' in title and 'object at' in title):
                         logger.warning(f"[Topic] Skipping corrupted title book: {bid} - {title}")
                         continue
        
                    if not title:
                        continue
                        
                    seen_ids.add(bid)
                    all_books.append(book)
                    
                    if len(all_books) >= limit:
                        break
            except Exception as e:
                logger.error(f"[Topic] Error processing topic future: {e}")
            if len(all_books) >= limit:
                break

    if randomize and len(all_books) > 0:
        random.shuffle(all_books)

    result = all_books[:limit]
    logger.info(f"[Topic] Returning {len(result)} books for user {user_id} (from {len(all_books)} total found)")
    if len(result) == 0:
        logger.warning(f"[Topic] No books found for user {user_id} with topics: {topics}")
    
    result_with_meta = {
        'books': result,
        'interests_exhausted': interests_exhausted,
        'total_interests': len(all_unique_topics),
        'current_page': current_page + 1
    }
    return result_with_meta


def get_personal_trending(user_id, limit=12):
    """
    يحصل على كتب رائجة مخصصة للمستخدم بناءً على اهتماماته.
    """
    if not user_id or user_id <= 0:
        return get_trending(limit)
    
    books_dicts = []
    seen_ids = set()
    topics_to_search = []
    
    try:
        last_search = (
            db.session.query(SearchHistory)
            .filter_by(user_id=user_id)
            .order_by(SearchHistory.created_at.desc(), SearchHistory.id.desc())
            .first()
        )
        if last_search and last_search.query:
            query_text = last_search.query.strip()
            if any("\u0600" <= c <= "\u06FF" for c in query_text):
                try:
                    translated = translate_to_english_with_gemini(query_text)
                    if translated and translated.strip():
                        topics_to_search.append(translated)
                        logger.info(f"[PersonalTrending] Using last search: '{query_text}' -> '{translated}'")
                    else:
                        topics_to_search.append(query_text)
                except Exception as e:
                    logger.warning(f"[PersonalTrending] Translation failed: {e}")
                    topics_to_search.append(query_text)
            else:
                topics_to_search.append(query_text)
    except Exception as e:
        logger.error(f"[PersonalTrending] Error getting last search: {e}", exc_info=True)
    
    try:
        prefs = (
            UserPreference.query
            .filter_by(user_id=user_id)
            .order_by(UserPreference.weight.desc())
            .limit(3)
            .all()
        )
        for pref in prefs:
            if pref.topic:
                if any("\u0600" <= c <= "\u06FF" for c in pref.topic):
                    try:
                        translated = translate_to_english_with_gemini(pref.topic)
                        if translated and translated.strip() and translated.lower() not in [t.lower() for t in topics_to_search]:
                            topics_to_search.append(translated)
                    except:
                        if pref.topic.lower() not in [t.lower() for t in topics_to_search]:
                            topics_to_search.append(pref.topic)
                else:
                    if pref.topic.lower() not in [t.lower() for t in topics_to_search]:
                        topics_to_search.append(pref.topic)
    except Exception as e:
        logger.error(f"[PersonalTrending] Error getting preferences: {e}", exc_info=True)
    
    if not topics_to_search:
        logger.info(f"[PersonalTrending] No personal topics found for user {user_id}, using general trending")
        return get_trending(limit)
    
    logger.info(f"[PersonalTrending] Searching for books in topics: {topics_to_search}")
    
    per_topic_limit = max(4, limit // len(topics_to_search))
    
    for topic in topics_to_search[:3]:
        try:
            gb_res = fetch_google_books(topic, max_results=per_topic_limit)
            items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
            
            for it in items or []:
                if not isinstance(it, dict): continue
                gid = it.get("id")
                if not gid or gid in seen_ids: continue
                seen_ids.add(gid)
                
                vi = it.get("volumeInfo") or {}
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img:
                    if img.startswith("http://"): img = img.replace("http://", "https://")
                    if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')
                
                books_dicts.append({
                    "id": gid, "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors") or []),
                    "cover": img, "source": "Google Books",
                    "reason": f"🔥 رائج في موضوع: {topic}",
                    "rating": _extract_rating_with_fallback(vi),
                    "ratings_count": vi.get("ratingsCount"),
                })
                
                if len(books_dicts) >= limit: break
        except Exception as e:
            logger.error(f"[PersonalTrending] Google Books error for '{topic}': {e}", exc_info=True)
        
        if len(books_dicts) >= limit: break
    
    if len(books_dicts) < limit:
        needed = limit - len(books_dicts)
        general_trending = get_trending(needed * 2)
        for book in general_trending:
            book_id = book.get("id")
            if book_id and book_id not in seen_ids:
                seen_ids.add(book_id)
                books_dicts.append(book)
                if len(books_dicts) >= limit: break
    
    random.shuffle(books_dicts)
    result = books_dicts[:limit]
    logger.info(f"[PersonalTrending] Returning {len(result)} personalized trending books for user {user_id}")
    return result


def get_last_search_recommendations(user_id, limit=12, randomize=False):
    """
    جلب توصيات بناءً على آخر عملية بحث قام بها المستخدم حصراً.
    """
    if not user_id:
        return None, None

    books_dicts = []
    seen_ids = set()
    
    try:
        last_search = (
            db.session.query(SearchHistory)
            .filter_by(user_id=user_id)
            .order_by(SearchHistory.created_at.desc(), SearchHistory.id.desc())
            .first()
        )
        
        if not last_search or not last_search.query:
            return None, None
            
        query_text = last_search.query.strip()
        display_query = query_text
        
        search_term = query_text
        if any("\u0600" <= c <= "\u06FF" for c in query_text):
            try:
                translated = translate_to_english_with_gemini(query_text)
                if translated and translated.strip():
                    search_term = translated
            except:
                pass
                
        start_index = 0
        if randomize:
            start_index = random.randint(0, 40)
            
        logger.info(f"[LastSearch] Fetching books for '{search_term}' (orig: '{query_text}') at index {start_index}")
        gb_res = fetch_google_books(search_term, max_results=(limit * 3 if randomize else limit), start_index=start_index)
        items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
        
        for it in items or []:
            if not isinstance(it, dict): continue
            gid = it.get("id")
            if not gid or gid in seen_ids: continue
            seen_ids.add(gid)

            vi = it.get("volumeInfo") or {}
            img = (vi.get("imageLinks") or {}).get("thumbnail")
            
            if not img: continue
            
            title = vi.get("title")
            if not title or len(title) < 4: continue
            
            authors = vi.get("authors")
            if not authors: continue

            if img:
                if img.startswith("http://"): img = img.replace("http://", "https://")
                if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')

            books_dicts.append({
                "id": gid, "title": title,
                "author": ", ".join(authors),
                "cover": img, "source": "Google Books",
                "reason": f"لأنك بحثت عن: {display_query}",
                "rating": vi.get("averageRating"),
                "ratings_count": vi.get("ratingsCount"),
                "algo_tag": "Search History"
            })
            
            if len(books_dicts) >= (limit * 3 if randomize else limit):
                break
        
        if randomize and len(books_dicts) > 0:
            random.shuffle(books_dicts)
        
        logger.info(f"[LastSearch] Found {len(books_dicts)} valid books for user {user_id}")
        return display_query, books_dicts[:limit]
        
    except Exception as e:
        logger.error(f"[LastSearch] Error: {e}", exc_info=True)
        return None, None


def get_archive_ai_recommendations(user_id, limit=16):
    """
    توصيات ذكية من Internet Archive بناءً على اهتمامات المستخدم.
    """
    if not user_id or user_id <= 0:
        return []
    
    books = []
    seen_ids = set()
    search_topics = []
    
    try:
        last_search = (
            db.session.query(SearchHistory)
            .filter_by(user_id=user_id)
            .order_by(SearchHistory.created_at.desc())
            .first()
        )
        if last_search and last_search.query:
            search_topics.append(last_search.query.strip())
            
        prefs = UserPreference.query.filter_by(user_id=user_id).limit(2).all()
        for p in prefs:
            if p.topic and p.topic not in search_topics:
                search_topics.append(p.topic)
    except Exception as e:
        logger.error(f"[ArchiveAI] Error getting user interests: {e}")
    
    if not search_topics:
        search_topics = ["programming", "science", "literature"]
    
    per_topic = max(4, limit // len(search_topics))
    
    for topic in search_topics[:3]:
        try:
            search_term = topic
            if any("\u0600" <= c <= "\u06FF" for c in topic):
                translated = translate_to_english_with_gemini(topic)
                if translated and translated.strip():
                    search_term = translated
                    logger.info(f"[ArchiveAI] Translated '{topic}' to '{search_term}'")
            
            ia_results = fetch_archive_books(search_term, limit=per_topic)
            
            for b in ia_results or []:
                bid = b.get("id")
                if not bid or bid in seen_ids: continue
                seen_ids.add(bid)
                
                books.append({
                    "id": bid, "title": b.get("title"),
                    "author": b.get("author"), "cover": b.get("cover"),
                    "source": "Internet Archive",
                    "reason": f"🤖 AI وجد هذا من اهتماماتك: «{topic}»",
                })
                
                if len(books) >= limit: break
        except Exception as e:
            logger.error(f"[ArchiveAI] Error for topic '{topic}': {e}")
        
        if len(books) >= limit: break
    
    logger.info(f"[ArchiveAI] Returning {len(books)} books for user {user_id}")
    return books
