import logging
import random
import time
from typing import List, Dict, Optional
import requests
from threading import Thread, Lock
import json
import os

logger = logging.getLogger(__name__)

class InterestFetcherService:
    """
    🌍 External Interest Fetcher Service
    
    Fetches trending topics, categories, and books from the internet 
    (Google Books, OpenLibrary) to enrich the recommendation pipeline 
    with "fresh" signals.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InterestFetcherService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._cache = {
            "trending_books": [],
            "trending_categories": [],
            "last_updated": 0,
            "hot_keywords": []
        }
        self.update_interval = 3600  # 1 hour
        self.is_updating = False
        
        # Redis Connection
        self.redis = None
        try:
            import redis
            # Use short timeout for initial check
            client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"), 
                decode_responses=True,
                socket_connect_timeout=1
            )
            if client.ping():
                self.redis = client
                logger.info("✅ [InterestFetcher] Redis connected")
            else:
                logger.warning("⚠️ [InterestFetcher] Redis ping failed, operating in memory-only mode")
        except Exception as e:
            logger.debug(f"⚠️ [InterestFetcher] Redis not available (skipping): {e}")

        # Initial background update
        self.start_background_update()

    def start_background_update(self):
        """Start a background thread to update interests."""
        if self.is_updating:
            return
            
        def _update():
            self.is_updating = True
            logger.info("🌍 [InterestFetcher] Starting background update...")
            try:
                self.refresh_trending_data()
            except Exception as e:
                logger.error(f"🌍 [InterestFetcher] Update failed: {e}")
            finally:
                self.is_updating = False
                
        Thread(target=_update, daemon=True).start()

    def refresh_trending_data(self):
        """Fetch fresh data from APIs."""
        # 1. Fetch Trending Categories
        categories = self._fetch_trending_categories()
        
        # 2. Fetch Hot Keywords (Simulated or API)
        keywords = ["Artificial Intelligence", "Climate Change", "Mental Health", "Space Exploration", "Startups"]
        
        # 3. Fetch Trending Books (from a random subset of categories)
        trending_books = self._fetch_trending_books(categories)
        
        # Update Cache safely
        with self._lock:
            self._cache["trending_categories"] = categories
            self._cache["trending_books"] = trending_books
            self._cache["hot_keywords"] = keywords
            self._cache["last_updated"] = time.time()
            
            # Sync to Redis if available
            if self.redis:
                try:
                    self.redis.set("interest:trending_categories", json.dumps(categories), ex=7200)
                    self.redis.set("interest:trending_books", json.dumps(trending_books), ex=7200)
                    self.redis.set("interest:hot_keywords", json.dumps(keywords), ex=7200)
                    self.redis.set("interest:last_updated", str(time.time()), ex=7200)
                except Exception as e:
                    logger.error(f"⚠️ [InterestFetcher] Redis sync failed: {e}")
            
        logger.info(f"🌍 [InterestFetcher] Updated: {len(categories)} cats, {len(trending_books)} books")

    def get_trending_interests(self) -> Dict[str, List]:
        """Get cached trending data."""
        # Check if cache is stale
        if self.redis:
            # Try fetching from Redis first
            try:
                cats = self.redis.get("interest:trending_categories")
                books = self.redis.get("interest:trending_books")
                if cats and books:
                    return {
                        "categories": json.loads(cats),
                        "books": json.loads(books),
                        "source": "redis"
                    }
            except: pass

        if time.time() - self._cache["last_updated"] > self.update_interval:
            self.start_background_update()
            
        return {
            "categories": self._cache["trending_categories"],
            "books": self._cache["trending_books"],
            "source": "memory"
        }

    def _fetch_trending_categories(self, limit=20) -> List[Dict]:
        """Fetch trending categories using Google Books API."""
        # We simulate "trending" by querying for popular tags or diverse subjects
        queries = [
            "bestsellers 2024", "trending fiction", "science popular", 
            "technology trends", "history new releases", "self help bestsellers"
        ]
        
        found_categories = {}
        
        for q in queries:
            try:
                url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=10&printType=books"
                resp = requests.get(url, timeout=5)
                if resp.ok:
                    data = resp.json()
                    for item in data.get("items", []):
                        cats = item.get("volumeInfo", {}).get("categories", [])
                        for cat in cats:
                            cat_clean = cat.lower().strip()
                            if cat_clean and len(cat_clean) > 2:
                                found_categories[cat] = found_categories.get(cat, 0) + 1
            except Exception:
                continue
                
        # Sort by popularity
        sorted_cats = sorted(found_categories.items(), key=lambda x: x[1], reverse=True)
        
        return [{"name": name, "score": count} for name, count in sorted_cats[:limit]]

    def _fetch_trending_books(self, categories: List[Dict], limit=30) -> List[Dict]:
        """Fetch popular books from top categories."""
        if not categories:
            return []
            
        # Pick 3 random categories to keep it fresh
        target_cats = [c["name"] for c in random.sample(categories, min(3, len(categories)))]
        all_books = []
        
        for cat in target_cats:
            try:
                url = f"https://www.googleapis.com/books/v1/volumes?q=subject:{cat}&orderBy=newest&maxResults=10"
                resp = requests.get(url, timeout=5)
                if resp.ok:
                    data = resp.json()
                    for item in data.get("items", []):
                        info = item.get("volumeInfo", {})
                        all_books.append({
                            "id": item.get("id"),
                            "title": info.get("title"),
                            "author": ", ".join(info.get("authors", [])),
                            "cover": info.get("imageLinks", {}).get("thumbnail"),
                            "description": info.get("description", "")[:200],
                            "category": cat,
                            "source": "trending_web"
                        })
            except Exception:
                continue
                
        random.shuffle(all_books)
        return all_books[:limit]

# Singleton instance
interest_service = InterestFetcherService()
