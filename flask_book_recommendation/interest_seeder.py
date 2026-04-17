# -*- coding: utf-8 -*-
"""
🌱 Interest Book Seeder
========================

Proactively fetches books from Google Books API based on user interests
and stores them in the local database with embeddings.

This ensures that when a new user selects interests, there are always
relevant books available for the recommendation engine to work with.
"""

import logging
import threading
import os
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

GOOGLE_API_URL = "https://www.googleapis.com/books/v1/volumes"

# Map interest names to optimized search queries for better results
INTEREST_SEARCH_QUERIES = {
    "fiction": ["best fiction novels", "award winning fiction", "modern fiction"],
    "sci-fi": ["best science fiction books", "sci-fi novels", "space opera books"],
    "mystery": ["best mystery novels", "detective fiction", "thriller mystery"],
    "thriller": ["best thriller books", "suspense novels", "psychological thriller"],
    "romance": ["best romance novels", "love story books", "contemporary romance"],
    "non-fiction": ["best non-fiction books", "popular non-fiction", "narrative non-fiction"],
    "biography": ["best biographies", "autobiography books", "memoir books"],
    "history": ["best history books", "world history", "historical narrative"],
    "science": ["popular science books", "science explained", "physics chemistry biology"],
    "technology": ["technology books", "artificial intelligence books", "programming books"],
    "philosophy": ["philosophy books", "great philosophers", "modern philosophy"],
    "art": ["art books", "art history", "contemporary art"],
    "business": ["best business books", "entrepreneurship", "business strategy"],
    "self-help": ["self improvement books", "personal development", "self help bestsellers"],
    "travel": ["travel books", "travel literature", "adventure travel"],
    "horror": ["best horror novels", "horror fiction", "gothic horror"],
    "poetry": ["poetry collections", "modern poetry", "classic poetry"],
    "children": ["children books", "kids literature", "young adult fiction"],
    "psychology": ["psychology books", "behavioral psychology", "cognitive psychology"],
    "cooking": ["cookbook bestseller", "culinary arts", "cooking techniques"],
    "religion": ["religious books", "spirituality", "world religions"],
    "programming": ["programming books", "software engineering", "coding tutorials"],
    "artificial intelligence": ["AI books", "machine learning books", "deep learning"],
    "deep learning": ["deep learning books", "neural networks", "AI fundamentals"],
}


def seed_books_for_interests(user_id, interests, app=None):
    """
    Seed books from Google Books API for a list of user interests.
    Runs in a background thread to avoid blocking the UI.
    
    Args:
        user_id: The user's ID
        interests: List of interest strings (e.g., ["Fiction", "Science", "History"])
        app: Flask app instance (required for app context in background thread)
    """
    if not app:
        try:
            from flask import current_app
            app = current_app._get_current_object()
        except RuntimeError:
            logger.error("[Seeder] No Flask app available for background seeding")
            return

    def _background_seed():
        with app.app_context():
            total_seeded = 0
            for interest in interests:
                try:
                    count = _seed_single_interest(interest)
                    total_seeded += count
                except Exception as e:
                    logger.error(f"[Seeder] Error seeding '{interest}': {e}")
            
            logger.info(f"[Seeder] ✅ Seeded {total_seeded} total books for user {user_id} across {len(interests)} interests")

    # Run in background thread
    thread = threading.Thread(target=_background_seed, daemon=True)
    thread.start()
    logger.info(f"[Seeder] 🚀 Started background seeding for user {user_id} with {len(interests)} interests")


def _seed_single_interest(interest_name):
    """
    Fetch and store books for a single interest from Google Books API.
    
    Returns:
        Number of new books seeded
    """
    from .extensions import db
    from .models import Book, BookGenre, Genre
    from .utils import get_text_embedding

    api_key = os.environ.get("GOOGLE_BOOKS_API_KEY")
    interest_lower = interest_name.strip().lower()
    
    # Get optimized search queries for this interest
    queries = INTEREST_SEARCH_QUERIES.get(interest_lower, [interest_name])
    
    seeded_count = 0
    seen_google_ids = set()
    
    for query in queries:
        try:
            params = {
                "q": f"subject:{query}",
                "maxResults": 20,
                "orderBy": "relevance",
                "printType": "books",
                "langRestrict": "en",
            }
            if api_key:
                params["key"] = api_key

            import time
            max_retries = 3
            resp = None
            
            for attempt in range(max_retries):
                try:
                    resp = requests.get(GOOGLE_API_URL, params=params, timeout=15)
                    if resp.ok:
                        break
                    logger.warning(f"[Seeder] Google API error for '{query}': {resp.status_code} (Attempt {attempt+1}/{max_retries})")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"[Seeder] Request error for '{query}': {e} (Attempt {attempt+1}/{max_retries})")
                    
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
            
            if not resp or not resp.ok:
                logger.error(f"[Seeder] Failed to fetch '{query}' after {max_retries} attempts.")
                continue

            data = resp.json()
            items = data.get("items", [])
            
            for item in items:
                try:
                    google_id = item.get("id")
                    if not google_id or google_id in seen_google_ids:
                        continue
                    seen_google_ids.add(google_id)
                    
                    # Check if book already exists
                    existing = Book.query.filter_by(google_id=google_id).first()
                    if existing:
                        # Just ensure it has the right category
                        _ensure_book_genre(existing, interest_name)
                        continue
                    
                    vi = item.get("volumeInfo", {})
                    
                    # Extract data
                    title = vi.get("title", "").strip()
                    if not title:
                        continue
                    
                    authors = vi.get("authors", ["Unknown"])
                    author = ", ".join(authors) if authors else "Unknown"
                    description = vi.get("description", "")
                    
                    # Get best cover image
                    imgs = vi.get("imageLinks", {}) or {}
                    cover_url = (
                        imgs.get("medium") or 
                        imgs.get("large") or 
                        imgs.get("thumbnail") or 
                        imgs.get("smallThumbnail")
                    )
                    if cover_url:
                        cover_url = cover_url.replace("http://", "https://").replace("&edge=curl", "")
                    
                    # Categories
                    cats = vi.get("categories", [])
                    categories_str = ", ".join(cats) if cats else interest_name.title()
                    
                    # Create book record
                    book = Book(
                        title=title,
                        author=author,
                        description=description,
                        cover_url=cover_url,
                        google_id=google_id,
                        publisher=vi.get("publisher"),
                        published_date=vi.get("publishedDate"),
                        page_count=vi.get("pageCount"),
                        isbn=_extract_isbn(vi.get("industryIdentifiers", [])),
                        language=vi.get("language", "en"),
                        categories=categories_str,
                    )
                    db.session.add(book)
                    db.session.flush()  # Get the book.id
                    
                    # Link to genre
                    _ensure_book_genre(book, interest_name)
                    
                    # Build embedding for the new book
                    try:
                        from .models import BookEmbedding
                        import pickle
                        embed_text = f"{title} {author} {description[:200]}"
                        vec = get_text_embedding(embed_text)
                        if vec:
                            book_emb = BookEmbedding(
                                book_id=book.id,
                                vector=pickle.dumps(vec)
                            )
                            db.session.add(book_emb)
                    except Exception:
                        pass  # Non-critical
                    
                    seeded_count += 1
                    
                except Exception as e:
                    logger.debug(f"[Seeder] Skipping item: {e}")
                    continue
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"[Seeder] Error fetching '{query}': {e}")
            db.session.rollback()
    
    logger.info(f"[Seeder] Seeded {seeded_count} new books for interest '{interest_name}'")
    return seeded_count


def _ensure_book_genre(book, interest_name):
    """Ensure a book is linked to the appropriate genre."""
    from .extensions import db
    from .models import Genre, BookGenre
    
    genre = Genre.query.filter(Genre.name.ilike(interest_name.strip())).first()
    if genre:
        existing_link = BookGenre.query.filter_by(book_id=book.id, genre_id=genre.id).first()
        if not existing_link:
            try:
                db.session.add(BookGenre(book_id=book.id, genre_id=genre.id))
            except Exception:
                pass  # May fail on unique constraint


def _extract_isbn(identifiers):
    """Extract ISBN from industry identifiers."""
    if not identifiers:
        return None
    for ident in identifiers:
        if ident.get("type") in ("ISBN_13", "ISBN_10"):
            return ident.get("identifier")
    return None
