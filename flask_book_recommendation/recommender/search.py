# -*- coding: utf-8 -*-
"""
Search — semantic search + reranking + recommendations by title.
"""
import logging
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Book, BookEmbedding, UserPreference
from ..utils import (
    fetch_google_books, get_text_embedding,
    analyze_search_intent_with_ai
)
from .helpers import _book_to_dict

logger = logging.getLogger(__name__)


def semantic_search(query: str, limit: int = 12, exclude_book_ids: list = None):
    """
    بحث دلالي: يحول الاستعلام إلى embedding ويقارنه مع embeddings الكتب.
    """
    if not query or not query.strip():
        return []
    
    exclude_book_ids = exclude_book_ids or []
    
    query_embedding = get_text_embedding(query)
    if not query_embedding:
        logger.warning(f"[SemanticSearch] Failed to get embedding for: {query}")
        return []
    
    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    all_embeds = BookEmbedding.query.all()
    if not all_embeds:
        logger.info("[SemanticSearch] No book embeddings found")
        return []
    
    book_ids = []
    vectors = []
    
    for row in all_embeds:
        if row.book_id in exclude_book_ids:
            continue
        if row.vector is None:
            continue
        
        # Defensive unpickling for raw bytes
        vec_data = __import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector
        v = np.array(vec_data, dtype=np.float32)
        if v.ndim == 1 and v.shape[0] == query_vec.shape[1]:
            book_ids.append(row.book_id)
            vectors.append(v)
    
    if not vectors:
        return []
    
    mat = np.vstack(vectors)
    try:
        similarities = cosine_similarity(query_vec, mat)[0]
    except Exception as e:
        logger.error(f"[SemanticSearch] Similarity error: {e}")
        return []
    
    ranked_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in ranked_indices[:limit * 2]:
        score = similarities[idx]
        if score < 0.3:
            continue
        
        book = Book.query.get(book_ids[idx])
        if not book:
            continue
        
        results.append(
            _book_to_dict(
                book,
                source="AI Search",
                reason=f"🔍 تشابه: {score:.0%}",
            )
        )
        
        if len(results) >= limit:
            break
    
    logger.info(f"[SemanticSearch] Found {len(results)} matches for '{query}'")
    return results


def rerank_search_results(user_id, books):
    """
    إعادة ترتيب نتائج البحث بناءً على اهتمامات المستخدم.
    """
    if not user_id or not books: return books
    
    try:
        prefs = UserPreference.query.filter_by(user_id=user_id).all()
        if not prefs: return books
        
        pref_map = {p.topic.lower(): p.weight for p in prefs}
        
        def calculate_score(book):
            score = 0.0
            title = (book.get("title") or "").lower()
            author = (book.get("author") or "").lower()
            
            for topic, weight in pref_map.items():
                if topic in title: score += weight * 1.5
                if topic in author: score += weight
            
            return score
            
        sorted_books = sorted(books, key=calculate_score, reverse=True)
        return sorted_books
    except Exception as e:
        logger.error(f"[Reranking] Error: {e}")
        return books


def get_recommendations_by_title(title, limit=24):
    """
    جلب توصيات بناءً على عنوان كتاب مشابه.
    """
    if not title: return []
    
    target_res, _ = fetch_google_books(title, max_results=1)
    if not target_res:
        return []
        
    target_book = target_res[0]
    vi = target_book.get("volumeInfo", {})
    categories = vi.get("categories", [])
    authors = vi.get("authors", [])
    
    queries = []
    
    if not categories or any(any('\u0600' <= c <= '\u06FF' for c in cat) for cat in categories):
        try:
            ai_info = analyze_search_intent_with_ai(vi.get("title", title))
            if ai_info and ai_info.get("query"):
                queries.append(ai_info["query"])
                logger.info(f"[Similar] AI Recommended query for technical book: {ai_info['query']}")
        except:
            pass

    if categories:
        cat = categories[0].split("/")[0].strip()
        queries.append(f"subject:{cat}")
    
    if authors:
        queries.append(f"inauthor:{authors[0]}")
        
    if not queries:
        queries.append(title)
        
    all_books = []
    seen_ids = set()
    target_id = target_book.get("id")
    seen_ids.add(target_id)
    
    for q in queries:
        try:
            res_items, _ = fetch_google_books(q, max_results=limit)
            if not res_items: continue
            
            for it in res_items:
                gid = it.get("id")
                if not gid or gid in seen_ids: continue
                seen_ids.add(gid)
                
                vi_it = it.get("volumeInfo") or {}
                
                img = (vi_it.get("imageLinks") or {}).get("thumbnail")
                if img:
                    if img.startswith("http://"): img = img.replace("http://", "https://")
                    if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')
                
                reason = "🔥 لأنك أحببت كتاباً مشابهاً"
                if q.startswith("subject:"):
                    reason = f"📚 من نفس التصنيف: {categories[0] if categories else 'مواضيع مشابهة'}"
                elif q.startswith("inauthor:"):
                    reason = f"✍️ لنفس المؤلف: {authors[0]}"
                elif q == queries[0] and len(queries) > 1:
                     reason = "🤖 مقترح ذكي لمجال الكتاب"
                    
                all_books.append({
                    "id": gid,
                    "title": vi_it.get("title"),
                    "author": ", ".join(vi_it.get("authors") or []),
                    "cover": img,
                    "source": "Similar API",
                    "reason": reason,
                    "rating": vi_it.get("averageRating"),
                    "ratings_count": vi_it.get("ratingsCount"),
                })
                
        except Exception as e:
            logger.error(f"[Similar] Error fetching similar books for {q}: {e}")
            
    random.shuffle(all_books)
    return all_books[:limit]
