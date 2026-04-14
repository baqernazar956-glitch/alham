# -*- coding: utf-8 -*-
"""
Session-Adaptive Recommendations
=================================
Generates real-time recommendations based on the user's *current session*
interactions (views, favorites, ratings, searches) — Instagram-style.
"""
import logging
import random
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Book, BookEmbedding, UserEvent, UserBookView
from ..extensions import db
from ..utils import fetch_google_books

logger = logging.getLogger(__name__)


def get_session_adaptive_recommendations(user_id, session_events=None, limit=12):
    """
    Generate recommendations adapted to the user's *current session*.

    Parameters
    ----------
    user_id : int
        Logged-in user ID.
    session_events : list[dict] | None
        Recent interaction dicts from the client, each with at least
        ``{"event_type", "book_id"}``.  If None the function will query
        the last 10 UserEvent rows for the user.
    limit : int
        Max books to return.

    Returns
    -------
    list[dict]
        Book dicts ready for ``render_carousel_section`` / ``render_book_card``.
    """
    try:
        # ── 1. Gather session interactions ────────────────────────────
        if session_events is None:
            recent = (
                UserEvent.query
                .filter_by(user_id=user_id)
                .order_by(UserEvent.created_at.desc())
                .limit(10)
                .all()
            )
            google_ids = [e.book_google_id for e in recent if e.book_google_id]
        else:
            google_ids = [
                e.get("book_id") for e in session_events if e.get("book_id")
            ]

        if not google_ids:
            return []

        # ── 2. Extract dominant categories from interacted books ──────
        books = Book.query.filter(Book.google_id.in_(google_ids)).all()
        if not books:
            return []

        category_counter = Counter()
        book_db_ids = []
        seen_google_ids = set(google_ids)

        for b in books:
            book_db_ids.append(b.id)
            if b.categories:
                for cat in b.categories.split(","):
                    cat = cat.strip()
                    if cat:
                        category_counter[cat] += 1

        # Pick top 2 categories
        top_categories = [c for c, _ in category_counter.most_common(2)]

        # ── 3. Embedding-based similarity for quick results ───────────
        embedding_recs = []
        if book_db_ids:
            try:
                embs = BookEmbedding.query.filter(
                    BookEmbedding.book_id.in_(book_db_ids)
                ).all()
                vecs = [
                    np.array(__import__("pickle").loads(e.vector) if isinstance(e.vector, bytes) else e.vector, dtype=np.float32)
                    for e in embs
                    if e.vector is not None
                ]
                if vecs:
                    centroid = np.mean(np.vstack(vecs), axis=0).reshape(1, -1)

                    # Compare against the full embedding matrix
                    all_embs = BookEmbedding.query.filter(
                        ~BookEmbedding.book_id.in_(book_db_ids)
                    ).limit(500).all()

                    if all_embs:
                        cand_ids = [e.book_id for e in all_embs if e.vector is not None]
                        cand_vecs = np.vstack([
                            np.array(__import__("pickle").loads(e.vector) if isinstance(e.vector, bytes) else e.vector, dtype=np.float32)
                            for e in all_embs
                            if e.vector is not None
                        ])
                        if cand_vecs.shape[0] > 0 and cand_vecs.shape[1] == centroid.shape[1]:
                            sims = cosine_similarity(centroid, cand_vecs)[0]
                            top_indices = np.argsort(sims)[::-1][:limit]

                            for idx in top_indices:
                                if sims[idx] < 0.15:
                                    continue
                                cand_book = Book.query.get(cand_ids[idx])
                                if cand_book and cand_book.google_id not in seen_google_ids:
                                    seen_google_ids.add(cand_book.google_id)
                                    embedding_recs.append({
                                        "id": cand_book.google_id or f"local_{cand_book.id}",
                                        "title": cand_book.title,
                                        "author": cand_book.author or "Unknown",
                                        "cover": cand_book.cover_url,
                                        "rating": None,
                                        "source": "Session AI",
                                        "reason": f"🔄 مشابه لما شاهدته ({int(sims[idx]*100)}%)",
                                        "score": float(sims[idx]),
                                        "rec_type": "session_adaptive",
                                    })
            except Exception as e:
                logger.error(f"[SessionAdaptive] Embedding fallback: {e}")

        # ── 4. Google Books category boost ────────────────────────────
        google_recs = []
        if top_categories:
            search_topic = top_categories[0]
            try:
                result = fetch_google_books(search_topic, max_results=limit)
                items = result[0] if isinstance(result, tuple) else result
                for it in items or []:
                    if not isinstance(it, dict):
                        continue
                    gid = it.get("id")
                    if not gid or gid in seen_google_ids:
                        continue
                    seen_google_ids.add(gid)
                    vi = it.get("volumeInfo") or {}
                    img = (vi.get("imageLinks") or {}).get("thumbnail", "")
                    if img.startswith("http://"):
                        img = img.replace("http://", "https://")
                    google_recs.append({
                        "id": gid,
                        "title": vi.get("title"),
                        "author": ", ".join(vi.get("authors") or []),
                        "cover": img,
                        "rating": vi.get("averageRating"),
                        "source": "Session Discovery",
                        "reason": f"🔄 بناءً على اهتمامك بـ {search_topic}",
                        "score": 0.5,
                        "rec_type": "session_adaptive",
                    })
            except Exception as e:
                logger.error(f"[SessionAdaptive] Google fetch error: {e}")

        # ── 5. Merge & shuffle ────────────────────────────────────────
        all_recs = embedding_recs + google_recs
        # Put embedding matches first, then category, with light shuffle
        random.shuffle(all_recs[len(embedding_recs):])
        # Interleave for variety
        merged = []
        e_idx, g_idx = 0, 0
        while len(merged) < limit and (e_idx < len(embedding_recs) or g_idx < len(google_recs)):
            if e_idx < len(embedding_recs):
                merged.append(embedding_recs[e_idx])
                e_idx += 1
            if g_idx < len(google_recs) and len(merged) < limit:
                merged.append(google_recs[g_idx])
                g_idx += 1

        logger.info(
            f"[SessionAdaptive] user={user_id}, "
            f"session_books={len(google_ids)}, "
            f"top_cats={top_categories}, "
            f"results={len(merged)}"
        )
        return merged[:limit]

    except Exception as e:
        logger.error(f"[SessionAdaptive] Fatal error: {e}", exc_info=True)
        return []
