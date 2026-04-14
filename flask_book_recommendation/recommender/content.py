# -*- coding: utf-8 -*-
"""
Content-Based recommendations — cosine similarity over embeddings.
"""
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Book, UserRatingCF, SearchHistory, BookEmbedding, UserBookView
from ..extensions import db
from .helpers import _book_to_dict

logger = logging.getLogger(__name__)


def get_content_similar(user_id, top_n=30, history_limit=20, randomize=False):
    """
    توصيات محتوى Content-Based باستخدام جدول BookEmbedding.
    """
    if not user_id or user_id <= 0:
        return []
    
    user_ratings = (
        UserRatingCF.query
        .filter_by(user_id=user_id)
        .order_by(UserRatingCF.created_at.desc())
        .limit(history_limit)
        .all()
    )
    rated_gids = [r.google_id for r in user_ratings if r.google_id]

    history_books_ids = []
    try:
        history_rows = (
            db.session.query(SearchHistory)
            .filter_by(user_id=user_id)
            .order_by(SearchHistory.created_at.desc())
            .limit(history_limit)
            .all()
        )
        for h in history_rows:
            if getattr(h, "book_id", None):
                history_books_ids.append(h.book_id)
    except Exception as e:
        logger.error(f"[Content] SearchHistory error: {e}", exc_info=True)

    rated_books = []
    if rated_gids:
        rated_books = (
            Book.query.filter(Book.google_id.in_(rated_gids)).all()
        )
    rated_book_ids = [b.id for b in rated_books]

    seed_book_ids = list({*rated_book_ids, *history_books_ids})
    if not seed_book_ids:
        return []

    seed_embeds = (
        BookEmbedding.query.filter(BookEmbedding.book_id.in_(seed_book_ids)).all()
    )
    if not seed_embeds:
        return []

    seed_vectors = []
    for row in seed_embeds:
        vec = row.vector
        if vec is None:
            continue
        # Unpickle explicitly if it's bytes
        vec = __import__("pickle").loads(vec) if isinstance(vec, bytes) else vec
        v = np.array(vec, dtype=np.float32)
        if v.ndim == 1:
            seed_vectors.append(v)
    if not seed_vectors:
        return []

    user_profile = np.mean(np.vstack(seed_vectors), axis=0).reshape(1, -1)

    all_embeds = BookEmbedding.query.all()
    book_ids = []
    vectors = []
    for row in all_embeds:
        vec = row.vector
        if vec is None:
            continue
        vec = __import__("pickle").loads(vec) if isinstance(vec, bytes) else vec
        v = np.array(vec, dtype=np.float32)
        if v.ndim == 1:
            book_ids.append(row.book_id)
            vectors.append(v)

    if not vectors:
        return []

    mat = np.vstack(vectors)
    try:
        sims = cosine_similarity(user_profile, mat)[0]
    except Exception as e:
        logger.error(f"[Content] cosine_similarity error: {e}", exc_info=True)
        return []

    exclude_ids = set(seed_book_ids)
    ranked_indices = np.argsort(sims)[::-1]

    if randomize:
        pool_size = max(top_n * 4, 100)
        potential = ranked_indices[:pool_size]
        np.random.shuffle(potential)
        ranked_indices = potential

    recs = []
    for idx in ranked_indices:
        score = sims[idx]
        if score <= 0:
            continue
        b_id = book_ids[idx]
        if b_id in exclude_ids:
            continue

        book = Book.query.get(b_id)
        if not book:
            continue

        recs.append(
            _book_to_dict(
                book,
                source="Content",
                reason="📖 لأنك قرأت كتباً مشابهة",
            )
        )
        if len(recs) >= top_n:
            break

    return recs


def get_view_based_recommendations(user_id, top_n=12, history_limit=10, randomize=False):
    """
    توصيات ذكية بناءً على سجل المشاهدات (UserBookView) باستخدام AI Embeddings.
    """
    if not user_id or user_id <= 0:
        return []

    try:
        recent_views = (
            UserBookView.query
            .filter_by(user_id=user_id)
            .order_by(UserBookView.last_viewed_at.desc())
            .limit(history_limit)
            .all()
        )
        
        if not recent_views:
            return []

        viewed_book_ids = []
        viewed_google_ids = []
        for v in recent_views:
            if v.book_id: viewed_book_ids.append(v.book_id)
            if v.google_id: viewed_google_ids.append(v.google_id)
            
        if viewed_google_ids:
            g_books = Book.query.filter(Book.google_id.in_(viewed_google_ids)).all()
            for b in g_books:
                viewed_book_ids.append(b.id)
                
        viewed_book_ids = list(set(viewed_book_ids))
        if not viewed_book_ids:
            return []

        seed_embeds = (
            BookEmbedding.query.filter(BookEmbedding.book_id.in_(viewed_book_ids)).all()
        )
        
        seed_vectors = []
        for row in seed_embeds:
            if row.vector is not None:
                vec = __import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector
                v = np.array(vec, dtype=np.float32)
                if v.ndim == 1:
                    seed_vectors.append(v)
                    
        if not seed_vectors:
            return []

        interest_profile = np.mean(np.vstack(seed_vectors), axis=0).reshape(1, -1)

        all_embeds = BookEmbedding.query.all()
        candidate_ids = []
        candidate_vectors = []
        
        for row in all_embeds:
            if row.book_id in viewed_book_ids:
                continue
                
            if row.vector is not None:
                vec = __import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector
                v = np.array(vec, dtype=np.float32)
                if v.ndim == 1:
                    candidate_ids.append(row.book_id)
                    candidate_vectors.append(v)
                    
        if not candidate_vectors:
            return []
            
        mat = np.vstack(candidate_vectors)
        sims = cosine_similarity(interest_profile, mat)[0]
        
        ranked_indices = np.argsort(sims)[::-1]
        
        if randomize:
            pool_size = max(top_n * 4, 40)
            potential = ranked_indices[:pool_size]
            np.random.shuffle(potential)
            ranked_indices = potential
            
        recs = []
        for idx in ranked_indices:
            score = sims[idx]
            if score < 0.4:
                continue
                
            b_id = candidate_ids[idx]
            book = Book.query.get(b_id)
            if not book:
                continue
                
            recs.append(
                _book_to_dict(
                    book,
                    source="AI Views",
                    reason=f"👀 🤖 ماتش ذكي: {int(score*100)}%",
                )
            )
            
            if len(recs) >= top_n:
                break
                
        logger.info(f"[ViewAI] Generated {len(recs)} recommendations based on {len(viewed_book_ids)} viewed books")
        return recs
        
    except Exception as e:
        logger.error(f"[ViewAI] Error: {e}", exc_info=True)
        return []
