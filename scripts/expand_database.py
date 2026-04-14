
import os
import sys
import sqlite3
import logging
import requests
import json
import time
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.getcwd())

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ExpandDB")

DB_PATH = "instance/app.db"
ST_MODEL_NAME = "all-MiniLM-L6-v2"
BOOKS_PER_CATEGORY = 20

# Very distinct, niche categories to force the AI to learn differences
NEW_CATEGORIES = [
    "Programming", "Artificial Intelligence", "Cybersecurity", # Persona 1: Tech
    "Romance", "Poetry", "Love Stories",                        # Persona 2: Romance
    "World History", "Biography", "Political Science"          # Persona 3: History
]

def fetch_diverse_books():
    logger.info("📚 Fetching new diverse books from Google Books API...")
    app = create_app()
    with app.app_context():
        # Get existing Google IDs to avoid duplicates
        from flask_book_recommendation.models import Book
        existing_gids = {b.google_id for b in Book.query.with_entities(Book.google_id).all() if b.google_id}
        
        unique_books = {}
        for cat in NEW_CATEGORIES:
            logger.info(f"  -> Fetching Category: {cat}")
            url = f"https://www.googleapis.com/books/v1/volumes"
            params = {
                "q": f"subject:{cat}",
                "maxResults": BOOKS_PER_CATEGORY,
                "langRestrict": "en",
                "printType": "books"
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    vi = item.get("volumeInfo", {})
                    gid = item.get("id")
                    
                    if gid in existing_gids or gid in unique_books:
                        continue
                        
                    title = vi.get("title", "Unknown")
                    authors = ", ".join(vi.get("authors", ["Unknown Author"]))
                    description = vi.get("description", "")
                    cover = vi.get("imageLinks", {}).get("thumbnail", "")
                    
                    # Force the category to be the one we searched for better mapping later
                    category = cat 
                    
                    unique_books[gid] = {
                        "title": title, "author": authors, "description": description,
                        "cover_url": cover, "google_id": gid, "categories": category
                    }
            except Exception as e:
                logger.error(f"Error fetching {cat}: {e}")
            
            time.sleep(1) # Respect API limits

        logger.info(f"✅ Found {len(unique_books)} new unique books. Saving to DB...")
        
        added = 0
        for gid, info in unique_books.items():
            b = Book(**info)
            db.session.add(b)
            added += 1
        
        db.session.commit()
        logger.info(f"✨ Successfully added {added} new books to the database.")
        return added

def generate_new_embeddings():
    logger.info(f"🤖 Generating embeddings for new books using {ST_MODEL_NAME}...")
    try:
        model = SentenceTransformer(ST_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return
        
    app = create_app()
    with app.app_context():
        from flask_book_recommendation.models import Book, BookEmbedding
        import pickle
        
        # Find books without embeddings
        query = db.session.query(Book).outerjoin(BookEmbedding).filter(BookEmbedding.book_id == None)
        books_to_embed = query.all()
        
        if not books_to_embed:
            logger.info("✅ All books already have embeddings.")
            return

        logger.info(f"  -> Processing {len(books_to_embed)} books...")
        
        for b in books_to_embed:
            text = f"{b.title}. {b.description or ''}"[:1000]
            emb = model.encode(text)
            blob = pickle.dumps(emb)
            
            be = BookEmbedding(book_id=b.id, vector=blob)
            db.session.add(be)
            
        db.session.commit()
        logger.info(f"✅ Generated and saved {len(books_to_embed)} new embeddings.")

if __name__ == "__main__":
    fetch_diverse_books()
    generate_new_embeddings()
    logger.info("🎉 Database expansion complete.")
