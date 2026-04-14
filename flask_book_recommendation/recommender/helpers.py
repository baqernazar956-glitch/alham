# -*- coding: utf-8 -*-
"""
Shared helpers used across all recommender sub-modules.
"""
import logging
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func
from flask import current_app
from flask_login import current_user

from ..models import (
    Book, UserRatingCF, SearchHistory,
    UserPreference, BookEmbedding, BookReview, UserBookView,
    BookStatus, UserGenre, Genre, PublicRating
)
from ..utils import (
    fetch_google_books, fetch_gutenberg_books,
    fetch_openlib_books, fetch_archive_books,
    fetch_itbook_books,
    translate_to_english_with_gemini,
    get_text_embedding,
    fetch_openlib_rating,
    analyze_search_intent_with_ai
)
from ..extensions import db, cache
from ..advanced_recommender import DLInferenceEngine
from ..ai_client import ai_client

logger = logging.getLogger(__name__)

# Initialize DL Engine lazily to avoid blocking startup or script imports
_dl_engine = None

def get_dl_engine():
    global _dl_engine
    if _dl_engine is None:
        _dl_engine = DLInferenceEngine()
    return _dl_engine


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _book_to_dict(book, source="Local", reason=None, extra_meta=None):
    """
    يحوّل كائن Book من الـ ORM إلى قاموس جاهز للتمبليت.
    """
    if book is None:
        return None

    cover_url = getattr(book, "cover_url", None)
    if cover_url:
        if cover_url.startswith("http://"):
            cover_url = "https://" + cover_url[7:]
        if 'books.google.com' in cover_url and '&edge=curl' in cover_url:
            cover_url = cover_url.replace('&edge=curl', '').replace('&edge=curl&', '&')
        if 'via.placeholder.com' in cover_url:
            cover_url = cover_url.replace('via.placeholder.com', 'placehold.co')

    data = {
        "id": getattr(book, "google_id", None) or f"local_{book.id}",
        "title": getattr(book, "title", None),
        "author": getattr(book, "author", None),
        "desc": getattr(book, "description", None),
        "cover": cover_url,
        "source": source,
        "reason": reason,
        "rating": getattr(book, "average_rating", None) or getattr(book, "rating", None),
        "pageCount": getattr(book, "page_count", None),
        "publishedDate": getattr(book, "published_date", None),
        "isbn": getattr(book, "isbn", None),
        "language": getattr(book, "language", None),
        "categories": getattr(book, "categories", None).split(",") if (getattr(book, "categories", None) and isinstance(getattr(book, "categories", None), str)) else getattr(book, "categories", []),
    }
    
    # Add AI Metadata if provided
    if extra_meta:
        data.update(extra_meta)
        
    return data


def _extract_rating_with_fallback(vi):
    """
    استخراج التقييم من بيانات Google Books مع محاولة Fallback إلى OpenLibrary.
    """
    rating = vi.get("averageRating")
    if rating:
        return rating
    
    # محاولة الحصول على ISBN للبحث في OpenLibrary
    isbns = vi.get("industryIdentifiers") or []
    isbn_13 = next((i["identifier"] for i in isbns if i["type"] == "ISBN_13"), None)
    isbn_10 = next((i["identifier"] for i in isbns if i["type"] == "ISBN_10"), None)
    isbn = isbn_13 or isbn_10
    
    if isbn:
        try:
            return fetch_openlib_rating(isbn=isbn)
        except:
            pass
            
    return None


def _deduplicate_dicts(items, key="id"):
    """
    يزيل التكرارات من قائمة القواميس بناءً على مفتاح محدد.
    """
    seen = set()
    out = []
    for it in items:
        k = it.get(key)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _apply_mmr_diversity(books, lambda_param=0.5, max_per_category=2):
    """
    تطبيق خوارزمية MMR (Maximal Marginal Relevance) لضمان التنوع.
    """
    if not books or len(books) <= 3:
        return books
    
    selected = []
    remaining = books.copy()
    category_counts = {}
    author_counts = {}
    
    while remaining and len(selected) < len(books):
        best_score = -1
        best_idx = 0
        
        for idx, book in enumerate(remaining):
            category = book.get("category", "unknown")
            author = book.get("author", "unknown")
            
            cat_count = category_counts.get(category, 0)
            auth_count = author_counts.get(author, 0)
            
            diversity_penalty = 0
            if cat_count >= max_per_category:
                diversity_penalty += 0.5
            if auth_count >= max_per_category:
                diversity_penalty += 0.3
            
            original_score = book.get("score", 1.0)
            final_score = original_score * (1 - lambda_param * diversity_penalty)
            
            if final_score > best_score:
                best_score = final_score
                best_idx = idx
        
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        
        category = chosen.get("category", "unknown")
        author = chosen.get("author", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        author_counts[author] = author_counts.get(author, 0) + 1
    
    return selected


def run_in_context(app, func, *args, **kwargs):
    """Helper to run function within app context"""
    with app.app_context():
        return func(*args, **kwargs)
