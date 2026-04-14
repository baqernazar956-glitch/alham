# -*- coding: utf-8 -*-
"""
Mood-Based recommendations.
"""
import logging
import random

from flask_login import current_user

from ..models import SearchHistory
from ..utils import fetch_google_books
from ..extensions import db

logger = logging.getLogger(__name__)


MOOD_MAPPING = {
    "happy": {
        "title": "سعید",
        "emoji": "😃",
        "queries": ["Comedy", "Humor", "Feel-good", "Funny"],
        "color": "var(--warning)"
    },
    "sad": {
        "title": "حزین",
        "emoji": "😔",
        "queries": ["Drama", "Tragedy", "Emotional", "Sad"],
        "color": "var(--accent-purple)"
    },
    "adventurous": {
        "title": "متحمس",
        "emoji": "🚀",
        "queries": ["Adventure", "Action", "Science Fiction", "Thriller"],
        "color": "var(--accent-cyan)"
    },
    "calm": {
        "title": "هادئ",
        "emoji": "🧘",
        "queries": ["Meditation", "Philosophy", "Nature", "Calm"],
        "color": "var(--primary)"
    },
    "curious": {
        "title": "فضولي",
        "emoji": "🧐",
        "queries": ["Science", "Mystery", "History", "Nonfiction"],
        "color": "var(--accent-magenta)"
    },
    "romantic": {
        "title": "رومانسي",
        "emoji": "❤️",
        "queries": ["Romance", "Love", "Poetry"],
        "color": "var(--accent-pink, #f472b6)"
    }
}


def get_mood_based_recommendations(mood_key, limit=12):
    """
    جلب توصيات بناءً على مزاج المستخدم.
    """
    mood_info = MOOD_MAPPING.get(mood_key)
    if not mood_info:
        logger.warning(f"[Mood] Invalid mood key: {mood_key}")
        return []

    queries = mood_info.get("queries", [])
    if not queries: queries = ["Books"]
    
    random.shuffle(queries)
    
    personalization_reason = None
    if current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        try:
            last_search = db.session.query(SearchHistory).filter_by(user_id=current_user.id)\
                .order_by(SearchHistory.created_at.desc()).first()
            
            if last_search and last_search.query:
                mood_term = queries[0]
                personalized_query = f"{last_search.query} {mood_term}"
                queries.insert(0, personalized_query)
                personalization_reason = f"لأنك مهتم بـ '{last_search.query}' وتشعر بـ {mood_info['title']}"
                logger.info(f"[Mood] Personalized query added: {personalized_query}")
        except Exception as e:
            logger.warning(f"[Mood] Error adding personalization: {e}")

    all_books = []
    seen_ids = set()
    
    for query in queries:
        try:
            gb_res = fetch_google_books(query, max_results=limit)
            items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
            
            if not items:
                continue
            
            for it in items:
                if not isinstance(it, dict): continue
                gid = it.get("id")
                if not gid or gid in seen_ids: continue
                seen_ids.add(gid)
                
                vi = it.get("volumeInfo") or {}
                title = vi.get("title")
                if not title: continue
                
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img:
                    if img.startswith("http://"): img = img.replace("http://", "https://")
                    if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')
                
                reason_text = f"{mood_info['emoji']} لأنك تشعر بـ {mood_info['title']}"
                
                if personalization_reason and query == queries[0]:
                    reason_text = f"✨ {personalization_reason}"
                
                all_books.append({
                    "id": gid,
                    "title": title,
                    "author": ", ".join(vi.get("authors") or []),
                    "cover": img,
                    "source": "Mood API",
                    "reason": reason_text,
                    "rating": vi.get("averageRating"),
                    "ratings_count": vi.get("ratingsCount"),
                })
            
            if all_books:
                break
                
        except Exception as e:
            logger.warning(f"[Mood] Error with query '{query}': {e}")
            continue

    return all_books[:limit]
