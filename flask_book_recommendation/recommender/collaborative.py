# -*- coding: utf-8 -*-
"""
Collaborative Filtering — User-User CF recommendations.
"""
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Book, UserRatingCF
from ..extensions import db
from .helpers import _book_to_dict

logger = logging.getLogger(__name__)


def get_cf_similar(user_id, top_n=30, min_users=2, offset=0, randomize=False):
    """
    Get recommendations based on similar users (User-User Collaborative Filtering)
    """
    if not user_id:
        return []

    try:
        ratings = UserRatingCF.query.all()
        if not ratings:
            logger.debug(f"[CF] No ratings found for user {user_id}")
            return []

        user_ratings = [r for r in ratings if r.user_id == user_id]
        if len(user_ratings) == 0:
            logger.debug(f"[CF] User {user_id} has no ratings")
            return []

        user_ids = sorted({r.user_id for r in ratings})
        item_gids = sorted({r.google_id for r in ratings if r.google_id})

        if len(user_ids) < min_users or len(item_gids) == 0:
            logger.debug(f"[CF] Not enough users ({len(user_ids)}) or items ({len(item_gids)})")
            return []

        user_index = {u_id: idx for idx, u_id in enumerate(user_ids)}
        item_index = {gid: idx for idx, gid in enumerate(item_gids)}

        mat = np.zeros((len(user_ids), len(item_gids)), dtype=np.float32)
        for r in ratings:
            if not r.google_id:
                continue
            ui = user_index[r.user_id]
            ii = item_index[r.google_id]
            mat[ui, ii] = float(r.rating or 0.0)

        if user_id not in user_index:
            logger.warning(f"[CF] User {user_id} not found in user_index")
            return []

        u_idx = user_index[user_id]
        user_vec = mat[u_idx].reshape(1, -1)

        if np.count_nonzero(user_vec) == 0:
            logger.debug(f"[CF] User {user_id} has all zero ratings")
            return []

        sims = cosine_similarity(user_vec, mat)[0]
        sims[u_idx] = 0.0

        sim_matrix = sims.reshape(-1, 1)
        weighted_sum = (sim_matrix * mat).sum(axis=0)
        sim_sum = (sim_matrix * (mat > 0)).sum(axis=0) + 1e-8
        scores = weighted_sum / sim_sum

        user_rated_mask = mat[u_idx] > 0
        scores[user_rated_mask] = -1.0

        top_indices = np.argsort(scores)[::-1]
        
        if offset >= len(top_indices):
            return []
            
        if randomize:
            pool_size = max(top_n * 3, 100)
            potential_indices = top_indices[offset : offset + pool_size]
            np.random.shuffle(potential_indices)
            top_indices = potential_indices
        else:
            top_indices = top_indices[offset:]
        
        # 🚀 [OPTIMIZATION] Bulk-fetch books for CF
        recs = []
        target_gids = [item_gids[idx] for idx in top_indices if scores[idx] > 0]
        if not target_gids:
            return []
            
        found_books = Book.query.filter(Book.google_id.in_(target_gids)).all()
        book_map = {b.google_id: b for b in found_books}
        
        seen_ids = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            gid = item_gids[idx]
            book = book_map.get(gid)
            if not book:
                continue
            
            book_id_key = book.google_id or f"local_{book.id}"
            if book_id_key in seen_ids:
                continue
            seen_ids.add(book_id_key)
            
            recs.append(
                _book_to_dict(
                    book,
                    source="CF",
                    reason=f"✨ تقارب أذواق بنسبة {int(scores[idx]*100)}%",
                )
            )
            if len(recs) >= top_n:
                break

        logger.info(f"[CF] Generated {len(recs)} recommendations for user {user_id}")
        return recs
        
    except Exception as e:
        logger.error(f"[CF] Error in get_cf_similar for user {user_id}: {e}", exc_info=True)
        return []


def _get_cf_recommendations(user_id, limit=6, offset=0):
    """
    توصيات Collaborative Filtering - مستخدمون مشابهون.
    """
    try:
        cf_books = get_cf_similar(user_id, top_n=limit, offset=offset)
        
        for book in cf_books:
            book["score"] = 0.8
            book["rec_type"] = "collaborative"
            if "reason" not in book or not book["reason"]:
                book["reason"] = "👥 أعجب مستخدمين بذوق مشابه"
        
        logger.info(f"[CF] Found {len(cf_books)} CF recommendations for user {user_id}")
        return cf_books
        
    except Exception as e:
        logger.error(f"[CF-V2] Error: {e}", exc_info=True)
        return []
