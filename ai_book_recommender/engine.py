# -*- coding: utf-8 -*-
"""
🚀 AI Book Recommender Engine
==============================

Main recommendation engine orchestrating all components.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

from .config import Config, get_config
from .feature_store import FeatureStore, UserFeatures, BookFeatures
from .models import (
    TransformerEncoder,
    TwoTowerV2,
    GraphRecommender,
    EnsembleRanker,
    NeuralReranker,
    ContextAwareRanker,
)
from .retrieval import (
    VectorIndexService,
    HybridRetriever,
    CacheManager,
    get_cache,
)
from .user_intelligence import (
    UserProfiler,
    DynamicUserModel,
    OnlineLearner,
)
from .explainability import RecommendationExplainer, ExplanationResult
from .evaluation import RecommendationMetrics, MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    """Request for recommendations."""
    
    user_id: int
    num_recommendations: int = 10
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Filters
    exclude_ids: List[str] = field(default_factory=list)
    category_filter: Optional[str] = None
    
    # Options
    include_explanations: bool = True
    diversity_factor: float = 0.2
    exploration_rate: float = 0.1


@dataclass
class RecommendationResponse:
    """Response with recommendations."""
    
    user_id: int
    recommendations: List[Dict[str, Any]]
    explanations: List[ExplanationResult] = field(default_factory=list)
    
    # Metadata
    request_id: str = ""
    latency_ms: float = 0.0
    model_version: str = "1.0.0"
    
    # Debug info
    retrieval_count: int = 0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """
    🚀 Main Recommendation Engine
    
    Orchestrates all AI components for end-to-end recommendations.
    
    Pipeline:
    1. Feature extraction (user + context)
    2. Candidate retrieval (hybrid search)
    3. Neural ranking (Two-Tower + Graph + CF)
    4. Re-ranking (context-aware, neural)
    5. Ensemble scoring
    6. Diversity/exploration adjustment
    7. Explanation generation
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()
        
        # Core components
        self.feature_store = FeatureStore()
        self.user_profiler = UserProfiler()
        
        # Models (lazy loaded)
        self._two_tower: Optional[TwoTowerV2] = None
        self._graph_rec: Optional[GraphRecommender] = None
        self._reranker: Optional[NeuralReranker] = None
        self._context_ranker: Optional[ContextAwareRanker] = None
        
        # Retrieval
        self.vector_service = VectorIndexService(
            index_dir=str(self.config.index_dir)
        )
        self.hybrid_retriever = HybridRetriever()
        
        # Ensemble
        self.ensemble = EnsembleRanker()
        
        # Online learning
        self.online_learner = OnlineLearner(
            exploration_rate=self.config.online_learning.epsilon
        )
        
        # Explainability
        self.explainer = RecommendationExplainer()
        
        # Metrics
        self.metrics = RecommendationMetrics(k=10)
        self.metrics_tracker = MetricsTracker()
        
        # Cache
        self.cache = get_cache(backend=self.config.cache.backend)
        
        self._initialized = False
        
        logger.info("RecommendationEngine created")
    
    def initialize(self) -> None:
        """Initialize all components (load models, indexes, etc.)."""
        if self._initialized:
            return
        
        logger.info("Initializing RecommendationEngine...")
        
        # Load vector index
        try:
            index = self.vector_service.get_index("books")
            self.hybrid_retriever.set_vector_index(index)
            logger.info(f"Loaded book index with {index.size} vectors")
        except Exception as e:
            logger.warning(f"Could not load book index: {e}")
        
        self._initialized = True
        logger.info("RecommendationEngine initialized")
    
    def recommend(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Generate recommendations for a user.
        
        Args:
            request: Recommendation request
            
        Returns:
            RecommendationResponse with recommendations
        """
        start_time = time.time()
        
        # Ensure initialized
        if not self._initialized:
            self.initialize()
        
        # Check cache
        cache_key = self.cache.make_key(
            request.user_id,
            request.num_recommendations,
            request.category_filter
        )
        cached = self.cache.get(f"recs:{cache_key}")
        if cached:
            logger.debug(f"Cache hit for user {request.user_id}")
            return RecommendationResponse(**cached)
        
        # 1. Extract user features
        user_features = self.feature_store.get_user_features(request.user_id)
        user_profile = self.user_profiler.get_profile(request.user_id)
        
        # Check cold start
        is_cold_start = self.user_profiler.is_cold_start(request.user_id)
        
        # 2. Candidate retrieval
        candidates = self._retrieve_candidates(
            user_features=user_features,
            request=request,
            is_cold_start=is_cold_start
        )
        
        if not candidates:
            return self._empty_response(request, start_time)
        
        # 3. Score candidates
        scored = self._score_candidates(
            candidates=candidates,
            user_features=user_features,
            context=request.context
        )
        
        # 4. Apply exploration if enabled
        if request.exploration_rate > 0:
            scored = self._apply_exploration(
                scored,
                request.exploration_rate
            )
        
        # 5. Select final recommendations
        final = scored[:request.num_recommendations]
        
        # 6. Generate explanations
        explanations = []
        if request.include_explanations:
            explanations = self._generate_explanations(final, user_features)
        
        # Build response
        recommendations = [
            {
                "book_id": item["id"],
                "score": item["score"],
                "rank": idx + 1,
                "sources": item.get("sources", {}),
            }
            for idx, item in enumerate(final)
        ]
        
        latency = (time.time() - start_time) * 1000
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            explanations=explanations,
            latency_ms=latency,
            retrieval_count=len(candidates),
        )
        
        # Track metrics
        self.metrics_tracker.record("latency_ms", latency)
        self.metrics_tracker.record("num_candidates", len(candidates))
        
        # Cache response
        self.cache.set(f"recs:{cache_key}", response.__dict__, ttl=300)
        
        return response
    
    def _retrieve_candidates(
        self,
        user_features: UserFeatures,
        request: RecommendationRequest,
        is_cold_start: bool
    ) -> List[Dict[str, Any]]:
        """Retrieve candidate items for ranking."""
        k = request.num_recommendations * 10  # Retrieve more for reranking
        
        # Get query vector from user features
        query_vector = user_features.history_embedding
        if query_vector is None:
            query_vector = user_features.interest_embedding
        
        if query_vector is None and is_cold_start:
            # Cold start: use popular items
            logger.info(f"Cold start for user {request.user_id}")
            return self._get_popular_items(k)
        
        # Hybrid retrieval
        results = self.hybrid_retriever.search(
            query_vector=query_vector,
            query_text=None,  # Could use user interests as text
            k=k,
            filter_ids=set(request.exclude_ids)
        )
        
        return [
            {
                "id": r.item_id,
                "score": r.score,
                "source": r.source,
                "sources": r.metadata,
            }
            for r in results
        ]
    
    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_features: UserFeatures,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score candidates using ensemble of models."""
        if not candidates:
            return []
        
        # Collect scores from each source
        item_ids = [c["id"] for c in candidates]
        scores = {}
        
        # Get retrieval scores
        scores["semantic"] = np.array([c["score"] for c in candidates])
        
        # Get popularity scores
        scores["popularity"] = self._get_popularity_scores(item_ids)
        
        # Ensemble combination
        combined = self.ensemble.combine(
            scores=scores,
            item_ids=item_ids,
            user_history=None  # Previously user_features.history_book_ids but the attribute doesn't exist
        )
        
        # Update candidates with combined scores
        scored = []
        for item_id, score, breakdown in combined:
            scored.append({
                "id": item_id,
                "score": score,
                "sources": breakdown
            })
        
        return scored
    
    def _apply_exploration(
        self,
        candidates: List[Dict[str, Any]],
        exploration_rate: float
    ) -> List[Dict[str, Any]]:
        """Apply exploration/exploitation using online learner."""
        item_ids = [c["id"] for c in candidates]
        scores = np.array([c["score"] for c in candidates])
        
        selected_ids = self.online_learner.select_items(
            candidate_ids=item_ids,
            scores=scores,
            k=len(candidates),
            strategy="epsilon_greedy"
        )
        
        # Reorder candidates by selection
        id_to_candidate = {c["id"]: c for c in candidates}
        return [id_to_candidate[id_] for id_ in selected_ids if id_ in id_to_candidate]
    
    def _generate_explanations(
        self,
        recommendations: List[Dict[str, Any]],
        user_features: UserFeatures
    ) -> List[ExplanationResult]:
        """Generate explanations for recommendations."""
        explanations = []
        
        for rec in recommendations:
            exp = self.explainer.explain(
                item_id=rec["id"],
                score_breakdown=rec.get("sources", {}),
                user_history=None  # Could pass history here
            )
            explanations.append(exp)
        
        return explanations
    
    def _get_popular_items(self, k: int) -> List[Dict[str, Any]]:
        """Get popular items for cold start from the database."""
        try:
            from flask import current_app
            from flask_book_recommendation.models import Book, UserBookView
            from flask_book_recommendation.extensions import db
            from sqlalchemy import func, desc
            
            # Ensure we have an application context
            def _query():
                results = (
                    db.session.query(
                        Book, 
                        func.sum(UserBookView.view_count).label('total_views')
                    )
                    .join(UserBookView, Book.id == UserBookView.book_id)
                    .group_by(Book.id)
                    .order_by(desc('total_views'))
                    .limit(k)
                    .all()
                )
                
                popular = []
                for book, views in results:
                    popular.append({
                        "id": book.google_id or str(book.id),
                        "score": 1.0,
                        "source": "popularity",
                        "metadata": {"views": views}
                    })
                
                if not popular:
                    books = Book.query.order_by(Book.average_rating.desc()).limit(k).all()
                    for book in books:
                        popular.append({
                            "id": book.google_id or str(book.id),
                            "score": 0.8,
                            "source": "top_rated"
                        })
                return popular
            
            try:
                # Try using existing context first
                current_app._get_current_object()
                return _query()
            except RuntimeError:
                # No app context — get it from the db engine's app
                app = db.get_app() if hasattr(db, 'get_app') else None
                if app:
                    with app.app_context():
                        return _query()
                return []
                    
        except Exception as e:
            logger.error(f"Error in _get_popular_items: {e}")
            return []
    
    def _get_popularity_scores(self, item_ids: List[str]) -> np.ndarray:
        """Get popularity scores for items based on interaction counts."""
        try:
            from flask import current_app
            from flask_book_recommendation.models import UserBookView
            from flask_book_recommendation.extensions import db
            from sqlalchemy import func
            
            # Extract local IDs if possible (handling 'local_X' or integer strings)
            local_ids = []
            for oid in item_ids:
                if oid.startswith("local_"):
                    local_ids.append(int(oid.replace("local_", "")))
                elif oid.isdigit():
                    local_ids.append(int(oid))
            
            if not local_ids:
                return np.ones(len(item_ids)) * 0.5
            
            def _query():
                view_counts = (
                    db.session.query(UserBookView.book_id, func.sum(UserBookView.view_count))
                    .filter(UserBookView.book_id.in_(local_ids))
                    .group_by(UserBookView.book_id)
                    .all()
                )
                
                counts_map = {bid: count for bid, count in view_counts}
                max_count = max(counts_map.values()) if counts_map else 1
                
                scores = []
                for oid in item_ids:
                    bid = None
                    if oid.startswith("local_"): bid = int(oid.replace("local_", ""))
                    elif oid.isdigit(): bid = int(oid)
                    
                    count = counts_map.get(bid, 0)
                    score = 0.1 + (0.9 * (count / max_count))
                    scores.append(score)
                    
                return np.array(scores)
            
            try:
                current_app._get_current_object()
                return _query()
            except RuntimeError:
                # No app context — get it from the db engine's app
                app = db.get_app() if hasattr(db, 'get_app') else None
                if app:
                    with app.app_context():
                        return _query()
                return np.ones(len(item_ids)) * 0.5
                
        except Exception as e:
            logger.error(f"Error in _get_popularity_scores: {e}")
            return np.ones(len(item_ids)) * 0.5
    
    def _empty_response(
        self,
        request: RecommendationRequest,
        start_time: float
    ) -> RecommendationResponse:
        """Create empty response when no candidates found."""
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[],
            latency_ms=(time.time() - start_time) * 1000
        )
    
    # ==================== Feedback ====================
    
    def record_feedback(
        self,
        user_id: int,
        item_id: str,
        feedback_type: str,
        value: float = 1.0
    ) -> None:
        """
        Record user feedback for online learning.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            feedback_type: Type (click, view, rate, skip)
            value: Feedback value
        """
        self.online_learner.record_feedback(
            user_id=user_id,
            item_id=item_id,
            feedback_type=feedback_type,
            value=value
        )
        
        # Update user profile
        # Would also update feature store here
        
        logger.debug(f"Recorded {feedback_type} feedback for user {user_id} on {item_id}")
    
    # ==================== Management ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "initialized": self._initialized,
            "metrics": self.metrics_tracker.get_all_summaries(),
            "exploration": self.online_learner.get_exploration_stats(),
            "cache_stats": self.cache._cache.stats() if hasattr(self.cache._cache, "stats") else {}
        }
    
    def rebuild_index(self, books_data: List[Dict[str, Any]]) -> int:
        """
        Rebuild the vector index from book data.
        
        Args:
            books_data: List of book dictionaries with embeddings
            
        Returns:
            Number of books indexed
        """
        vectors = []
        ids = []
        
        for book in books_data:
            if "embedding" in book:
                vectors.append(book["embedding"])
                ids.append(book["id"])
        
        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            count = self.vector_service.build_index("books", vectors_np, ids)
            logger.info(f"Rebuilt book index with {count} vectors")
            return count
        
        return 0


# Singleton engine instance
_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    """Get or create the global recommendation engine."""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine
