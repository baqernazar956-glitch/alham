# -*- coding: utf-8 -*-
"""
recommender package — backward-compatible re-export of all public names.

Usage (unchanged from before):
    from flask_book_recommendation.recommender import get_homepage_sections
    from ..recommender import get_trending, log_user_view
"""
import time as _time
import logging as _logging

# ── helpers ──────────────────────────────────────────────────────────
from .helpers import (
    _book_to_dict,
    _extract_rating_with_fallback,
    _deduplicate_dicts,
    _apply_mmr_diversity,
    run_in_context,
    get_dl_engine,
)

# ── embedding cache ─────────────────────────────────────────────────
from .embedding_cache import (
    _GLOBAL_EMBEDDING_CACHE,
    _get_embeddings_matrix,
)

# ── collaborative ───────────────────────────────────────────────────
from .collaborative import (
    get_cf_similar,
    _get_cf_recommendations,
)

# ── content ─────────────────────────────────────────────────────────
from .content import (
    get_content_similar,
    get_view_based_recommendations,
)

# ── topic ───────────────────────────────────────────────────────────
from .topic import (
    get_topic_based,
    get_personal_trending,
    get_last_search_recommendations,
    get_archive_ai_recommendations,
)

# ── trending ────────────────────────────────────────────────────────
from .trending import (
    get_trending,
    get_trending_by_period,
)

# ── homepage ────────────────────────────────────────────────────────
from .homepage import (
    get_homepage_sections,
    get_discovery_picks,
    get_all_libraries_showcase,
)

# ── pipeline ────────────────────────────────────────────────────────
from .pipeline import (
    _get_ai_embedding_recommendations,
    get_deep_learning_recommendations,
    _fetch_behavior_hybrid_candidates,
    get_behavior_based_recommendations,
    _get_behavior_based_recommendations_legacy,
)

# ── events ──────────────────────────────────────────────────────────
from .events import (
    log_user_view,
    analyze_user_profile_with_ai,
)

# ── search ──────────────────────────────────────────────────────────
from .search import (
    semantic_search,
    rerank_search_results,
    get_recommendations_by_title,
)

# ── mood ────────────────────────────────────────────────────────────
from .mood import (
    MOOD_MAPPING,
    get_mood_based_recommendations,
)

# ── session adaptive ────────────────────────────────────────────────
from .session_adaptive import (
    get_session_adaptive_recommendations,
)

# ── hybrid ──────────────────────────────────────────────────────────
from .hybrid import (
    get_hybrid_recommendations,
    get_author_books,
    get_top_rated,
    get_because_you_read,
    get_similar_users_favorites,
    get_genre_explorer,
)

_logger = _logging.getLogger(__name__)


# ── Task 5: Baseline measurement ────────────────────────────────────
def measure_baseline() -> dict:
    """
    Returns baseline metrics for monitoring recommendation system performance.
    """
    from ..models import User, Book, UserRatingCF, BookEmbedding
    from ..extensions import db
    from sqlalchemy import func

    try:
        total_users = db.session.query(func.count(User.id)).scalar() or 0
        total_books = db.session.query(func.count(Book.id)).scalar() or 0
        total_ratings = db.session.query(func.count(UserRatingCF.id)).scalar() or 0
        books_with_embeddings = db.session.query(func.count(BookEmbedding.id)).scalar() or 0

        avg_ratings_per_user = 0.0
        if total_users > 0:
            avg_ratings_per_user = total_ratings / total_users

        # Measure recommendation latency (average over 10 calls)
        latencies = []
        for _ in range(10):
            start = _time.perf_counter()
            get_trending(limit=5)
            elapsed_ms = (_time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_users": total_users,
            "total_books": total_books,
            "total_ratings": total_ratings,
            "books_with_embeddings": books_with_embeddings,
            "avg_ratings_per_user": round(avg_ratings_per_user, 2),
            "recommendation_latency_ms": round(avg_latency, 2),
        }
    except Exception as e:
        _logger.error(f"[Baseline] Error: {e}")
        return {"error": str(e)}
