# -*- coding: utf-8 -*-
"""
📊 Feature Store
=================

Centralized feature extraction and caching for users and books.
Provides consistent, reusable features across all models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class UserFeatures:
    """Extracted features for a user."""
    
    user_id: int
    
    # Embedding representations
    history_embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    interest_embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    dynamic_embedding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    
    # Behavioral signals
    view_count: int = 0
    rating_count: int = 0
    avg_rating: float = 0.0
    save_count: int = 0
    
    # Temporal features
    days_since_registration: int = 0
    days_since_last_activity: int = 0
    session_count: int = 0
    avg_session_duration: float = 0.0
    
    # Interest profile
    preferred_categories: List[str] = field(default_factory=list)
    preferred_authors: List[str] = field(default_factory=list)
    interest_diversity: float = 0.0
    
    # Engagement metrics
    click_through_rate: float = 0.0
    completion_rate: float = 0.0
    
    # Context
    is_cold_start: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """Convert features to a single vector for model input."""
        numerical = np.array([
            self.view_count / 100.0,  # Normalize
            self.rating_count / 50.0,
            self.avg_rating / 5.0,
            self.save_count / 20.0,
            min(self.days_since_registration / 365.0, 1.0),
            min(self.days_since_last_activity / 30.0, 1.0),
            self.session_count / 100.0,
            self.avg_session_duration / 3600.0,
            self.interest_diversity,
            self.click_through_rate,
            self.completion_rate,
            float(self.is_cold_start),
        ])
        return np.concatenate([
            self.history_embedding,
            self.interest_embedding,
            numerical
        ])


@dataclass
class BookFeatures:
    """Extracted features for a book."""
    
    book_id: str
    
    # Content embedding
    text_embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    title_embedding: np.ndarray = field(default_factory=lambda: np.zeros(384))
    
    # Metadata
    title: str = ""
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    publish_year: int = 0
    page_count: int = 0
    language: str = "en"
    
    # Popularity signals
    avg_rating: float = 0.0
    rating_count: int = 0
    view_count: int = 0
    save_count: int = 0
    popularity_score: float = 0.0
    
    # Quality signals
    description_length: int = 0
    has_cover: bool = False
    
    # Derived features
    recency_score: float = 0.0
    category_popularity: Dict[str, float] = field(default_factory=dict)
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector for model input."""
        numerical = np.array([
            self.avg_rating / 5.0,
            min(self.rating_count / 1000.0, 1.0),
            min(self.view_count / 10000.0, 1.0),
            min(self.save_count / 500.0, 1.0),
            self.popularity_score,
            min(self.page_count / 1000.0, 1.0),
            self.recency_score,
            float(self.has_cover),
            min(self.description_length / 2000.0, 1.0),
        ])
        return np.concatenate([self.text_embedding, numerical])


class FeatureStore:
    """
    Centralized feature extraction and caching.
    
    Responsibilities:
    1. Extract user features from database
    2. Extract book features from database
    3. Cache computed features
    4. Provide temporal decay for features
    """
    
    def __init__(self, cache_backend: str = "memory", ttl: int = 300):
        """
        Initialize feature store.
        
        Args:
            cache_backend: "memory" or "redis"
            ttl: Time-to-live for cached features in seconds
        """
        self.cache_backend = cache_backend
        self.ttl = ttl
        
        # In-memory cache
        self._user_cache: Dict[int, Tuple[UserFeatures, datetime]] = {}
        self._book_cache: Dict[str, Tuple[BookFeatures, datetime]] = {}
        
        # Embedding model (lazy load)
        self._embedding_model = None
        
        logger.info(f"FeatureStore initialized with {cache_backend} cache, TTL={ttl}s")
    
    @property
    def embedding_model(self):
        """Lazy load sentence transformer."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence Transformer loaded successfully")
            except ImportError:
                logger.warning("SentenceTransformer not available")
                self._embedding_model = None
        return self._embedding_model
    
    def _is_cache_valid(self, cached_time: datetime) -> bool:
        """Check if cached data is still valid."""
        return (datetime.now() - cached_time).total_seconds() < self.ttl
    
    def _compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        if not text or self.embedding_model is None:
            return np.zeros(384)
        
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(384)
    
    def get_user_features(
        self,
        user_id: int,
        db_session: Optional[Any] = None,
        force_refresh: bool = False
    ) -> UserFeatures:
        """
        Get or compute features for a user.
        
        Args:
            user_id: User ID
            db_session: Optional SQLAlchemy session
            force_refresh: Force recomputation
            
        Returns:
            UserFeatures object
        """
        # Check cache
        if not force_refresh and user_id in self._user_cache:
            features, cached_time = self._user_cache[user_id]
            if self._is_cache_valid(cached_time):
                return features
        
        # Compute features
        features = self._extract_user_features(user_id, db_session)
        
        # Cache
        self._user_cache[user_id] = (features, datetime.now())
        
        return features
    
    def _extract_user_features(
        self,
        user_id: int,
        db_session: Optional[Any] = None
    ) -> UserFeatures:
        """Extract features from database for a user."""
        features = UserFeatures(user_id=user_id)
        
        if db_session is None:
            return features
        
        try:
            # Import models dynamically to avoid circular imports
            from flask_book_recommendation.models import (
                User, UserBookView, BookRating, UserBookStatus,
                Book, BookEmbedding, UserPreference, UserEmbedding
            )
            
            # Get user
            user = db_session.query(User).get(user_id)
            if not user:
                return features
            
            # Basic info
            if user.created_at:
                features.days_since_registration = (datetime.now() - user.created_at).days
            
            # View history
            views = db_session.query(UserBookView).filter_by(user_id=user_id).all()
            features.view_count = len(views)
            
            if views:
                last_view = max(v.viewed_at for v in views if v.viewed_at)
                if last_view:
                    features.days_since_last_activity = (datetime.now() - last_view).days
            
            # Ratings
            ratings = db_session.query(BookRating).filter_by(user_id=user_id).all()
            features.rating_count = len(ratings)
            if ratings:
                features.avg_rating = sum(r.rating for r in ratings) / len(ratings)
            
            # Saved books
            saved = db_session.query(UserBookStatus).filter_by(
                user_id=user_id, status="saved"
            ).count()
            features.save_count = saved
            
            # Cold start check
            total_interactions = features.view_count + features.rating_count
            features.is_cold_start = total_interactions < 5
            
            # 🆕 Use pre-calculated UserEmbedding if available (Phase 2)
            user_emb = db_session.query(UserEmbedding).filter_by(user_id=user_id).first()
            if user_emb and user_emb.vector is not None:
                features.history_embedding = np.array(user_emb.vector)
                logger.debug(f"Loaded pre-calculated embedding for user {user_id}")
            elif features.view_count > 0:
                # Fallback to on-the-fly calculation
                viewed_book_ids = [v.book_id for v in views[-20:]]  # Last 20
                embeddings = db_session.query(BookEmbedding).filter(
                    BookEmbedding.book_id.in_(viewed_book_ids)
                ).all()
                
                if embeddings:
                    # Apply temporal decay
                    weights = []
                    vectors = []
                    for emb in embeddings:
                        if emb.vector is not None:
                            vectors.append(np.array(emb.vector))
                            # More recent = higher weight
                            idx = viewed_book_ids.index(emb.book_id) if emb.book_id in viewed_book_ids else 0
                            weight = 0.95 ** (len(viewed_book_ids) - idx - 1)
                            weights.append(weight)
                    
                    if vectors:
                        weights = np.array(weights) / sum(weights)
                        features.history_embedding = np.average(vectors, axis=0, weights=weights)
            
            # Interest embedding from preferences
            prefs = db_session.query(UserPreference).filter_by(user_id=user_id).first()
            if prefs and prefs.selected_interests:
                interest_text = " ".join(prefs.selected_interests)
                features.interest_embedding = self._compute_text_embedding(interest_text)
                features.preferred_categories = prefs.selected_interests[:10]
            
            # Preferred authors from highly rated books
            if features.rating_count > 0:
                high_rated = db_session.query(BookRating).filter(
                    BookRating.user_id == user_id,
                    BookRating.rating >= 4
                ).limit(10).all()
                
                if high_rated:
                    book_ids = [r.book_id for r in high_rated]
                    books = db_session.query(Book).filter(Book.id.in_(book_ids)).all()
                    authors = [b.author for b in books if b.author]
                    features.preferred_authors = list(set(authors))[:5]
            
            # Interest diversity
            unique_categories = set()
            for v in views:
                book = db_session.query(Book).get(v.book_id)
                if book and book.genre:
                    unique_categories.add(book.genre)
            features.interest_diversity = min(len(unique_categories) / 10.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
        
        features.last_updated = datetime.now()
        return features
    
    def get_book_features(
        self,
        book_id: str,
        db_session: Optional[Any] = None,
        force_refresh: bool = False
    ) -> BookFeatures:
        """
        Get or compute features for a book.
        
        Args:
            book_id: Book ID
            db_session: Optional SQLAlchemy session
            force_refresh: Force recomputation
            
        Returns:
            BookFeatures object
        """
        # Check cache
        if not force_refresh and book_id in self._book_cache:
            features, cached_time = self._book_cache[book_id]
            if self._is_cache_valid(cached_time):
                return features
        
        # Compute features
        features = self._extract_book_features(book_id, db_session)
        
        # Cache
        self._book_cache[book_id] = (features, datetime.now())
        
        return features
    
    def _extract_book_features(
        self,
        book_id: str,
        db_session: Optional[Any] = None
    ) -> BookFeatures:
        """Extract features from database for a book."""
        features = BookFeatures(book_id=book_id)
        
        if db_session is None:
            return features
        
        try:
            from flask_book_recommendation.models import (
                Book, BookEmbedding, BookRating, UserBookView, UserBookStatus
            )
            from sqlalchemy import func
            
            # Get book
            book = db_session.query(Book).get(book_id)
            if not book:
                return features
            
            # Basic metadata
            features.title = book.title or ""
            features.authors = [book.author] if book.author else []
            features.categories = [book.genre] if book.genre else []
            features.language = book.language or "en"
            features.page_count = book.page_count or 0
            
            # Try to extract year from published_date
            if book.published_date:
                try:
                    if len(book.published_date) >= 4:
                        features.publish_year = int(book.published_date[:4])
                except (ValueError, TypeError):
                    pass
            
            # Description
            features.description_length = len(book.description or "")
            features.has_cover = bool(book.thumbnail)
            
            # Get embedding
            embedding = db_session.query(BookEmbedding).filter_by(book_id=book_id).first()
            if embedding and embedding.vector:
                features.text_embedding = np.array(embedding.vector)
            
            # Ratings
            rating_stats = db_session.query(
                func.avg(BookRating.rating).label("avg"),
                func.count(BookRating.rating).label("count")
            ).filter_by(book_id=book_id).first()
            
            if rating_stats:
                features.avg_rating = float(rating_stats.avg or book.average_rating or 0)
                features.rating_count = int(rating_stats.count or 0)
            else:
                features.avg_rating = float(book.average_rating or 0)
            
            # View count
            features.view_count = db_session.query(UserBookView).filter_by(
                book_id=book_id
            ).count()
            
            # Save count
            features.save_count = db_session.query(UserBookStatus).filter_by(
                book_id=book_id, status="saved"
            ).count()
            
            # Popularity score (normalized combination)
            features.popularity_score = self._compute_popularity(
                features.view_count,
                features.rating_count,
                features.avg_rating,
                features.save_count
            )
            
            # Recency score
            if features.publish_year > 0:
                years_old = datetime.now().year - features.publish_year
                features.recency_score = max(0, 1 - (years_old / 50.0))
            
        except Exception as e:
            logger.error(f"Error extracting book features: {e}")
        
        features.last_updated = datetime.now()
        return features
    
    def _compute_popularity(
        self,
        views: int,
        ratings: int,
        avg_rating: float,
        saves: int
    ) -> float:
        """
        Compute popularity score with log dampening.
        
        Formula:
        popularity = log(1 + views) * 0.3 + log(1 + ratings) * 0.3 +
                     avg_rating/5 * 0.25 + log(1 + saves) * 0.15
        """
        import math
        
        view_score = math.log1p(views) / math.log1p(10000)  # Normalize
        rating_score = math.log1p(ratings) / math.log1p(1000)
        quality_score = avg_rating / 5.0
        save_score = math.log1p(saves) / math.log1p(500)
        
        return min(1.0, (
            0.30 * view_score +
            0.30 * rating_score +
            0.25 * quality_score +
            0.15 * save_score
        ))
    
    def get_batch_user_features(
        self,
        user_ids: List[int],
        db_session: Optional[Any] = None
    ) -> Dict[int, UserFeatures]:
        """Get features for multiple users efficiently."""
        return {
            uid: self.get_user_features(uid, db_session)
            for uid in user_ids
        }
    
    def get_batch_book_features(
        self,
        book_ids: List[str],
        db_session: Optional[Any] = None
    ) -> Dict[str, BookFeatures]:
        """Get features for multiple books efficiently."""
        return {
            bid: self.get_book_features(bid, db_session)
            for bid in book_ids
        }
    
    def invalidate_user_cache(self, user_id: int) -> None:
        """Invalidate cached features for a user."""
        if user_id in self._user_cache:
            del self._user_cache[user_id]
    
    def invalidate_book_cache(self, book_id: str) -> None:
        """Invalidate cached features for a book."""
        if book_id in self._book_cache:
            del self._book_cache[book_id]
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        self._user_cache.clear()
        self._book_cache.clear()
        logger.info("Feature store cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "user_cache_size": len(self._user_cache),
            "book_cache_size": len(self._book_cache),
        }


# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get or create global feature store."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
