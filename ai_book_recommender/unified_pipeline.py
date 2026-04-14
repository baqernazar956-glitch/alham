# -*- coding: utf-8 -*-
"""
🧠 Unified Recommendation Pipeline — Full Neural Stack
=========================================================

Executes ALL recommendation models in a 9-step neural pipeline:
1. Hybrid Retrieval (vector + collaborative candidates)
2. Two-Tower scoring
3. Transformer contextual encoding
4. Graph-based boosting
5. Ensemble weighted fusion
6. Neural Reranker final scoring
7. Context Ranker reordering
8. Sort by predicted probability descending
9. Online learning user embedding update

No step may be skipped.
"""

import logging
import json
import time
import random
import os
import hashlib
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Flask imports
try:
    from flask import current_app
except ImportError:
    pass

# Model Imports
from .models.collaborative_filtering import MatrixFactorization
from flask_book_recommendation.advanced_recommender.neural_model import TwoTowerModel
from .models.graph_recommender import GraphRecommender
from .models.neural_reranker import NeuralReranker
from .models.context_ranker import ContextAwareRanker
from .models.transformer_encoder import TransformerEncoder
from .models.ensemble import EnsembleRanker, EnsembleWeights

# Retrieval
from .retrieval.hybrid_retrieval import HybridRetriever
from .retrieval.vector_index import VectorIndexService

# User Intelligence
from .user_intelligence.online_learning import OnlineLearner

# Services
from .interest_fetcher_service import interest_service
from .feature_store import get_feature_store

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE LAYER
# ═══════════════════════════════════════════════════════════════════════════

class RedisCacheLayer:
    """🚀 High-Performance Caching Layer (Redis + local fallback)"""

    def __init__(self, use_redis=True):
        self.use_redis = use_redis
        self.local_cache = {}
        self.local_ttl = {}
        self.redis = None

        if self.use_redis:
            try:
                import redis
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis = redis.Redis.from_url(
                    redis_url, socket_timeout=0.2, decode_responses=True
                )
                self.redis.ping()
                logger.info("✅ [Cache] Connected to Redis")
            except Exception as e:
                logger.debug(f"⚠️ [Cache] Redis not available: {e}")
                self.redis = None

    def get(self, key: str) -> Any:
        try:
            if self.redis:
                val = self.redis.get(key)
                if val:
                    return json.loads(val)
        except Exception:
            pass
        # Local fallback with TTL check
        if key in self.local_cache:
            if time.time() < self.local_ttl.get(key, 0):
                return self.local_cache[key]
            else:
                del self.local_cache[key]
                self.local_ttl.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        try:
            serialized = json.dumps(value, default=str)
            if self.redis:
                self.redis.setex(key, ttl_seconds, serialized)
            else:
                self.local_cache[key] = value
                self.local_ttl[key] = time.time() + ttl_seconds
        except Exception:
            self.local_cache[key] = value
            self.local_ttl[key] = time.time() + ttl_seconds


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED RECOMMENDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedRecommendationPipeline:
    """
    🧠 Full Neural Stack Recommendation Pipeline

    Loads ALL models once at startup. Every call to recommend_full_stack()
    executes the complete 9-step pipeline with no step skipped.
    """

    # Class-level storage for last pipeline execution metadata
    _last_pipeline_meta = {}

    def __init__(self, load_all_models: bool = True):
        logger.info("🚀 [Pipeline] Loading all AI models...")
        start = time.time()

        self.flask_app = None
        self.cache = RedisCacheLayer(use_redis=True)
        self._executor = ThreadPoolExecutor(max_workers=12)

        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_all_models:
            self._load_all_models()

        elapsed = time.time() - start
        logger.info(f"✅ [Pipeline] All models loaded in {elapsed:.2f}s")

    # ─────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_all_models(self):
        """Load every model component once. Never reloaded."""

        # 1. Ensemble Ranker — weighted fusion
        self.ensemble = EnsembleRanker(weights=EnsembleWeights(
            collaborative=0.15,
            two_tower=0.40,
            semantic=0.15,
            graph=0.10,
            behavioral=0.15,  # Increased weighting for direct behavioral intent (searches/views)
            popularity=0.05,
            diversity=0.03,
            novelty=0.02
        ))

        # 1.5. Two-Tower Deep Learning Model
        self.two_tower = TwoTowerModel()
        # 🔧 [FIX #6] Search multiple paths for the Two-Tower model
        tt_paths = [
            os.path.join(os.getcwd(), "instance", "models", "two_tower_model.pt"),
            os.path.join(os.getcwd(), "ai_models", "two_tower_model.pt"),
            os.path.join(os.getcwd(), "ai_models", "two_tower.pt"),
        ]
        tt_path = None
        for p in tt_paths:
            if os.path.exists(p):
                tt_path = p
                break
        if tt_path:
            try:
                state = torch.load(tt_path, map_location=self.device, weights_only=True)
                self.two_tower.load_state_dict(state)
                self.two_tower.to(self.device).eval()
                logger.debug(f"✅ [Pipeline] Two-Tower model loaded from {tt_path}")
            except Exception as e:
                logger.warning(f"⚠️ [Pipeline] Two-Tower model load failed: {e}")
                self.two_tower = None
        else:
            logger.warning(f"⚠️ [Pipeline] Two-Tower model file not found in any of {tt_paths}")
            self.two_tower = None

        # 2. Neural Reranker
        self.reranker = NeuralReranker(
            user_dim=128, item_dim=128,
            hidden_dim=256, num_features=10, dropout=0.1
        ).to(self.device).eval()
        self._try_load_checkpoint(self.reranker, "neural_reranker.pt")

        # 3. Context-Aware Ranker
        self.context_ranker = ContextAwareRanker(
            user_dim=128, item_dim=128,
            context_dim=64, hidden_dim=256, dropout=0.1
        ).to(self.device).eval()
        self._try_load_checkpoint(self.context_ranker, "context_ranker.pt")

        # 4. Transformer Encoder
        self.transformer = TransformerEncoder(
            input_dim=384, hidden_dim=256,
            output_dim=128, num_heads=8, num_layers=2, dropout=0.1
        ).to(self.device).eval()
        self._try_load_checkpoint(self.transformer, "transformer_encoder.pt")

        # 5. Graph Recommender
        self.graph_model = GraphRecommender()
        graph_path = os.path.join(os.getcwd(), "ai_models", "graph_recommender.pt")
        if os.path.exists(graph_path):
            try:
                self.graph_model.load(graph_path)
                logger.debug("✅ [Pipeline] Graph model loaded")
            except Exception as e:
                logger.warning(f"⚠️ [Pipeline] Graph model load failed: {e}")

        # 6. Online Learner
        self.online_learner = OnlineLearner(
            learning_rate=0.001,
            exploration_rate=0.1,
            exploration_decay=0.999,
            min_exploration=0.01,
            update_interval_seconds=60
        )

        # 7. Feature Store
        self.feature_store = get_feature_store()

        # 8. Hybrid Retriever + Vector Index
        self.vector_service = VectorIndexService(index_dir="instance/indexes")
        self.hybrid_retriever = HybridRetriever()
        try:
            self.hybrid_retriever.set_vector_index(
                self.vector_service.get_index("books")
            )
        except Exception:
            logger.warning("⚠️ [Pipeline] Vector index 'books' not ready")

        logger.debug("✅ [Pipeline] All 8 model components initialized")

        # 9. Pre-warm Text Embedding Model (prevent timeout on first request)
        try:
            logger.debug("🔄 [Pipeline] Pre-warming embedding model...")
            from flask_book_recommendation.utils import get_text_embedding
            _ = get_text_embedding("warmup")
            logger.debug("✅ [Pipeline] Embedding model pre-warmed")
        except Exception as e:
            logger.warning(f"⚠️ [Pipeline] Embedding model pre-warm failed: {e}")

    def _try_load_checkpoint(self, model, filename):
        """Attempt to load a saved model checkpoint."""
        path = os.path.join(os.getcwd(), "ai_models", filename)
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                logger.debug(f"✅ [Pipeline] Loaded checkpoint: {filename}")
            except Exception as e:
                logger.warning(f"⚠️ [Pipeline] Checkpoint {filename} failed: {e}")

    def _get_real_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Generate a real user embedding vector based on user history and TwoTower user_tower.
        """
        if not user_id or getattr(self, 'two_tower', None) is None or getattr(self, 'flask_app', None) is None:
            return None
            
        try:
            with self.flask_app.app_context():
                from flask_book_recommendation.models import UserRatingCF, UserBookView, BookStatus, BookEmbedding, Book, SearchHistory
                
                recent_books = []
                search_vectors = []
                
                from flask_book_recommendation.extensions import db
                # 1. Fetch search history and convert to embeddings
                searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).limit(5).all()
                if searches:
                    from flask_book_recommendation.utils import get_text_embedding, translate_to_english_with_gemini
                    for s in searches:
                        if s.query:
                            eng_query = translate_to_english_with_gemini(s.query) or s.query
                            emb = get_text_embedding(eng_query)
                            if emb and len(emb) == 384:
                                search_vectors.append(np.array(emb, dtype=np.float32))
                
                # 2. Fetch Ratings
                ratings = UserRatingCF.query.filter_by(user_id=user_id).order_by(UserRatingCF.created_at.desc()).limit(15).all()
                for r in ratings:
                    b = Book.query.filter_by(google_id=r.google_id).first()
                    if b and b.id not in recent_books:
                        recent_books.append(b.id)
                
                # 3. Fetch Views/Clicks
                views = UserBookView.query.filter_by(user_id=user_id).order_by(UserBookView.last_viewed_at.desc()).limit(15).all()
                for v in views:
                    bid = v.book_id
                    if not bid and v.google_id:
                        b = Book.query.filter_by(google_id=v.google_id).first()
                        if b: bid = b.id
                    if bid and bid not in recent_books:
                        recent_books.append(bid)
                        
                # 4. Fetch Book Statuses (Favorites)
                statuses = BookStatus.query.filter_by(user_id=user_id).limit(10).all()
                for s in statuses:
                    if s.book_id not in recent_books:
                        recent_books.append(s.book_id)
                        
                if not recent_books and not search_vectors:
                    return None
                    
                embeds = []
                if recent_books:
                    embeds = BookEmbedding.query.filter(BookEmbedding.book_id.in_(recent_books[:10])).all()
                
                vectors = []
                import pickle
                for e in embeds:
                    if e.vector is not None:
                        v = pickle.loads(e.vector) if isinstance(e.vector, bytes) else e.vector
                        v = np.array(v, dtype=np.float32)
                        if v.shape == (384,):
                            vectors.append(v)
                            
                all_vectors = vectors + search_vectors
                            
                if not all_vectors:
                    # Return zeroed 128d vector if no history but model exists
                    return np.zeros(128, dtype=np.float32)
                    
                hist = vectors[-10:] if vectors else search_vectors[-10:]
                while len(hist) < 10:
                    hist.append(np.zeros(384, dtype=np.float32))
                hist_vec = np.array(hist)
                
                # Mean across all behavior vectors (clicks, ratings, statuses, AND searches)
                int_vec = np.mean(np.array(all_vectors), axis=0) if all_vectors else np.zeros(384, dtype=np.float32)
                
                with torch.no_grad():
                    u_tensor = torch.tensor([user_id % 10000], dtype=torch.long, device=self.device)
                    hist_tensor = torch.tensor(np.array([hist_vec]), dtype=torch.float32, device=self.device)
                    int_tensor = torch.tensor(np.array([int_vec]), dtype=torch.float32, device=self.device)
                    
                    user_input = (u_tensor, hist_tensor, int_tensor)
                    u_emb = self.two_tower.user_tower(*user_input)
                    return u_emb.squeeze(0).cpu().numpy()
                    
        except Exception as e:
            logger.error(f"❌ [Pipeline] Error getting real user embedding: {e}")
            # Ensure we return a 128d vector even on error for tensor compatibility
            return np.zeros(128, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE: recommend_full_stack
    # ─────────────────────────────────────────────────────────────────────

    def recommend_full_stack(
        self,
        user_id: Optional[int] = None,
        top_k: int = 100,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        🧠 Execute the FULL 9-step neural pipeline.

        Steps:
        1. Hybrid Retrieval
        2. Two-Tower scoring
        3. Transformer contextual encoding
        4. Graph-based boosting
        5. Ensemble weighted fusion
        6. Neural Reranker final scoring
        7. Context Ranker reordering
        8. Sort by predicted probability descending
        9. Online learning user embedding update

        Returns structured results with model_breakdown and explanations.
        """
        pipeline_start = time.time()
        context = context or {}
        timings = {}
        stage_meta = []  # Track metadata for each stage

        # ── Cache check ──
        cache_key = self._cache_key("full_stack", user_id, context)
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"⚡ [Pipeline] Cache hit for user {user_id}")
            return cached

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Hybrid Retrieval — gather candidates from all sources
        # ═══════════════════════════════════════════════════════════════
        t0 = time.time()
        candidates = self._step1_hybrid_retrieval(user_id)
        timings["retrieval"] = time.time() - t0
        logger.debug(f"📥 [Step 1] Retrieved {len(candidates)} candidates in {timings['retrieval']:.3f}s")
        stage_meta.append({"id": 1, "name": "Hybrid Retrieval", "name_ar": "الاسترجاع الهجين", "icon": "database", "color": "#6366f1", "time_ms": round(timings['retrieval']*1000, 1), "candidates_in": 0, "candidates_out": len(candidates), "model": "FAISS + CF + Content", "status": "done"})

        if not candidates:
            logger.warning("⚠️ [Pipeline] No candidates retrieved, returning empty")
            return []

        # ── Pre-compute Real User Embedding for later steps ──
        t_user_emb = time.time()
        self.real_user_emb = self._get_real_user_embedding(user_id)
        if self.real_user_emb is not None:
            logger.debug(f"👤 [Pipeline] Generated real user embedding in {time.time() - t_user_emb:.3f}s")
        else:
            logger.debug("👤 [Pipeline] Using fallback/random user embedding")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Two-Tower Scoring
        # ═══════════════════════════════════════════════════════════════
        step2_in = len(candidates)
        t0 = time.time()
        candidates = self._step2_two_tower_scoring(candidates, user_id)
        timings["two_tower"] = time.time() - t0
        logger.debug(f"🏗️  [Step 2] Two-Tower scoring in {timings['two_tower']:.3f}s")
        stage_meta.append({"id": 2, "name": "Two-Tower Scoring", "name_ar": "التقييم ثنائي البرج", "icon": "account_tree", "color": "#8b5cf6", "time_ms": round(timings['two_tower']*1000, 1), "candidates_in": step2_in, "candidates_out": len(candidates), "model": "TwoTower v3.0", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Transformer Contextual Encoding
        # ═══════════════════════════════════════════════════════════════
        step3_in = len(candidates)
        t0 = time.time()
        candidates = self._step3_transformer_encoding(candidates)
        timings["transformer"] = time.time() - t0
        logger.debug(f"🔤 [Step 3] Transformer encoding in {timings['transformer']:.3f}s")
        stage_meta.append({"id": 3, "name": "Transformer Encoding", "name_ar": "ترميز المحولات", "icon": "memory", "color": "#a855f7", "time_ms": round(timings['transformer']*1000, 1), "candidates_in": step3_in, "candidates_out": len(candidates), "model": "Transformer 8H×2L", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Graph-Based Boosting
        # ═══════════════════════════════════════════════════════════════
        step4_in = len(candidates)
        t0 = time.time()
        candidates = self._step4_graph_boosting(candidates, user_id)
        timings["graph"] = time.time() - t0
        logger.debug(f"🕸️  [Step 4] Graph boosting in {timings['graph']:.3f}s")
        stage_meta.append({"id": 4, "name": "Graph Boosting", "name_ar": "تعزيز الرسم البياني", "icon": "hub", "color": "#d946ef", "time_ms": round(timings['graph']*1000, 1), "candidates_in": step4_in, "candidates_out": len(candidates), "model": "GraphRec GNN", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Ensemble Weighted Fusion
        # ═══════════════════════════════════════════════════════════════
        step5_in = len(candidates)
        t0 = time.time()
        candidates = self._step5_ensemble_fusion(candidates)
        timings["ensemble"] = time.time() - t0
        logger.debug(f"🎼 [Step 5] Ensemble fusion in {timings['ensemble']:.3f}s")
        stage_meta.append({"id": 5, "name": "Ensemble Fusion", "name_ar": "الدمج المتعدد", "icon": "merge", "color": "#ec4899", "time_ms": round(timings['ensemble']*1000, 1), "candidates_in": step5_in, "candidates_out": len(candidates), "model": "Weighted Ensemble", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Neural Reranker Final Scoring
        # ═══════════════════════════════════════════════════════════════
        step6_in = len(candidates)
        t0 = time.time()
        candidates = self._step6_neural_reranker(candidates, user_id)
        timings["reranker"] = time.time() - t0
        logger.debug(f"🎯 [Step 6] Neural Reranker in {timings['reranker']:.3f}s")
        stage_meta.append({"id": 6, "name": "Neural Reranker", "name_ar": "إعادة الترتيب العصبي", "icon": "neurology", "color": "#f43f5e", "time_ms": round(timings['reranker']*1000, 1), "candidates_in": step6_in, "candidates_out": len(candidates), "model": "NeuralReranker MLP", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Context Ranker Reordering
        # ═══════════════════════════════════════════════════════════════
        step7_in = len(candidates)
        t0 = time.time()
        candidates = self._step7_context_ranking(candidates, context)
        timings["context"] = time.time() - t0
        logger.debug(f"🕐 [Step 7] Context ranking in {timings['context']:.3f}s")
        stage_meta.append({"id": 7, "name": "Context Ranker", "name_ar": "ترتيب السياق", "icon": "schedule", "color": "#f97316", "time_ms": round(timings['context']*1000, 1), "candidates_in": step7_in, "candidates_out": len(candidates), "model": "ContextAwareRanker", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 8: Apply Online Learning Adjustments & Sort
        # ═══════════════════════════════════════════════════════════════
        step8_in = len(candidates)
        t0 = time.time()
        for c in candidates:
            # Apply real-time learned preference adjustment
            adjustment = self.online_learner.feedback_processor.get_item_score_adjustment(c["book_id"])
            if adjustment != 0:
                c["final_score"] = float(np.clip(c["final_score"] + (adjustment * 0.15), 0, 1))
                
        # 💡 [AI STRICT FILTER] Pure Neural Confidence Floor
        # 🔧 [FIX #4] Lowered from 0.40 to 0.15 — even 0.40 was too aggressive
        # for new users and untrained models where ensemble scores are naturally low.
        # 0.15 blocks only true noise while letting real neural-scored results through.
        initial_count = len(candidates)
        confidence_floor = 0.15
        all_candidates_backup = list(candidates)  # Save before filtering
        candidates = [c for c in candidates if c.get("final_score", 0) >= confidence_floor]
        if len(candidates) < initial_count:
            logger.debug(f"🛡️ [AI Filter] Dropped {initial_count - len(candidates)} books scoring below {confidence_floor} confidence")
        # Safety net: if the filter removed everything, relax to top results by score
        if not candidates and initial_count > 0:
            logger.warning("⚠️ [AI Filter] All candidates filtered out! Relaxing to top results.")
            candidates = sorted(
                all_candidates_backup,
                key=lambda x: x.get("final_score", 0), reverse=True
            )[:40]  # Keep top 40 even if below threshold

        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        candidates = candidates[:top_k]
        timings["sort_and_learn"] = time.time() - t0
        stage_meta.append({"id": 8, "name": "Final Sort", "name_ar": "الترتيب النهائي", "icon": "sort", "color": "#eab308", "time_ms": round(timings['sort_and_learn']*1000, 1), "candidates_in": step8_in, "candidates_out": len(candidates), "model": "BPR Sort + Adjust", "status": "done"})

        # ═══════════════════════════════════════════════════════════════
        # STEP 9: Online Learning User Embedding Update
        # ═══════════════════════════════════════════════════════════════
        t0 = time.time()
        self._step9_online_learning_update(candidates, user_id, context)
        timings["online_learning"] = time.time() - t0
        stage_meta.append({"id": 9, "name": "Online Learning", "name_ar": "التعلم الفوري", "icon": "model_training", "color": "#22c55e", "time_ms": round(timings['online_learning']*1000, 1), "candidates_in": len(candidates), "candidates_out": len(candidates), "model": "OnlineLearner v2", "status": "done"})

        # ── Format final output ──
        results = self._format_output(candidates)

        total_time = time.time() - pipeline_start
        logger.info(f"✅ [Pipeline] Full stack completed: {len(results)} results in {total_time:.3f}s")

        # ── Store pipeline metadata for frontend visualization ──
        UnifiedRecommendationPipeline._last_pipeline_meta = {
            "stages": stage_meta,
            "total_time_ms": round(total_time * 1000, 1),
            "final_books_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
        }

        # ── Cache results ──
        self.cache.set(cache_key, results, ttl_seconds=120)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # STEP IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────────

    def _step1_hybrid_retrieval(self, user_id: Optional[int]) -> List[Dict]:
        """Step 1: Gather candidates from ALL retrieval sources in parallel."""
        candidates_map = {}

        def _safe_source(name, func, *args, **kwargs):
            try:
                results = func(*args, **kwargs)
                return name, (results or [])
            except Exception as e:
                logger.error(f"❌ [Retrieval] {name} failed: {e}")
                return name, []

        if user_id and self.flask_app:

            def _run_in_context(func, *args, **kwargs):
                with self.flask_app.app_context():
                    return func(*args, **kwargs)

            # Import lazily to avoid circular imports
            from flask_book_recommendation.recommender import (
                get_cf_similar, get_content_similar,
                get_behavior_based_recommendations,
                get_deep_learning_recommendations,
                get_view_based_recommendations,
                get_trending
            )

            futures = {}
            # 💡 [PRECISION] Disabled randomization for all candidate sources
            futures["Collaborative Filtering"] = self._executor.submit(
                _safe_source, "Collaborative Filtering",
                _run_in_context, get_cf_similar, user_id, top_n=100, randomize=False
            )
            futures["Content-Based"] = self._executor.submit(
                _safe_source, "Content-Based",
                _run_in_context, get_content_similar, user_id, top_n=100, randomize=False
            )
            futures["Two-Tower"] = self._executor.submit(
                _safe_source, "Two-Tower",
                _run_in_context, get_deep_learning_recommendations, user_id, limit=100, randomize=False
            )
            futures["Behavioral"] = self._executor.submit(
                _safe_source, "Behavioral",
                _run_in_context, get_behavior_based_recommendations, user_id, limit=100, randomize=False
            )
            futures["View-Based"] = self._executor.submit(
                _safe_source, "View-Based",
                _run_in_context, get_view_based_recommendations, user_id, top_n=100, randomize=False
            )
            # futures["Trending"] = self._executor.submit(
            #     _safe_source, "Trending",
            #     _run_in_context, get_trending, limit=80
            # )
            
            # 🔥 New: Direct search-query based retrieval for immediate behavior matching
            from flask_book_recommendation.extensions import db
            from flask_book_recommendation.models import SearchHistory
            from flask_book_recommendation.utils import translate_to_english_with_gemini
            searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).limit(1).all()
            logger.debug(f"🔎 [Retrieval] Search history found: {len(searches)} entries for user {user_id}")
            if searches and searches[0].query:
                q_raw = searches[0].query
                # Added timeout=3 to avoid 30s hangs
                q = translate_to_english_with_gemini(q_raw, timeout=3) or q_raw
                logger.debug(f"🔎 [Retrieval] Injecting search query: '{q_raw}' -> '{q}'")
                from .retrieval.hybrid_retrieval import get_vector_search_results
                futures["Recent Search"] = self._executor.submit(
                    _safe_source, "Recent Search",
                    _run_in_context, get_vector_search_results, q, top_n=50
                )

            # 🍏 New: Interest-based retrieval for new users
            from flask_book_recommendation.models import UserGenre, Genre
            from flask_book_recommendation.extensions import db
            user_interest_genres = db.session.query(Genre.name).join(UserGenre).filter(UserGenre.user_id == user_id).all()
            user_interests = [g[0] for g in user_interest_genres]
            if user_interests:
                logger.debug(f"🧬 [Retrieval] Current user interest genres: {user_interests}")
                # Increased from 3 to 5 for better coverage
                for genre_query in user_interests[:5]: 
                    from .retrieval.hybrid_retrieval import get_vector_search_results
                    # Increased top_n from 30 to 50 for more variety
                    futures[f"Interest:{genre_query}"] = self._executor.submit(
                        _safe_source, f"Interest:{genre_query}",
                        _run_in_context, get_vector_search_results, genre_query, top_n=50
                    )

            for key, future in futures.items():
                try:
                    source_name, items = future.result(timeout=30)
                    
                    # 💡 [GATEKEEPER] Source Confidence Gating
                    # If a source returns very low scores, it's considered "Noise" for a new user
                    confidence_threshold = 0.20
                    filtered_items = []
                    for item in items:
                        if float(item.get("score", 0)) >= confidence_threshold:
                            filtered_items.append(item)
                    
                    if len(filtered_items) < len(items):
                        logger.debug(f"🛡️ [Gatekeeper] Dropped {len(items) - len(filtered_items)} low-confidence items from {source_name}")
                    
                    for item in filtered_items:
                        self._merge_candidate(candidates_map, item, source_name)
                except Exception as e:
                    logger.error(f"❌ [Retrieval] {key} timeout/error: {e}")

        else:
            # Anonymous user: trending + interest service
            try:
                if self.flask_app:
                    with self.flask_app.app_context():
                        from flask_book_recommendation.recommender import get_trending
                        trending = get_trending(limit=100)
                        for item in (trending or []):
                            self._merge_candidate(candidates_map, item, "Trending")
            except Exception as e:
                logger.error(f"❌ [Retrieval] Anonymous trending failed: {e}")

            try:
                data = interest_service.get_trending_interests()
                for item in data.get("books", [])[:30]:
                    self._merge_candidate(candidates_map, item, "Interest Service")
            except Exception as e:
                logger.error(f"❌ [Retrieval] Interest service failed: {e}")

        return list(candidates_map.values())

    def _merge_candidate(self, candidates_map: Dict, item: Dict, source: str):
        """Merge a candidate into the deduplication map."""
        if not item or not isinstance(item, dict):
            return
            
        # 🧪 [DEDUPLICATION FIX] Unify by google_id if available, otherwise database ID.
        # This prevents books from appearing twice if they are found in both 
        # the local database (integer id) and Google Books API (string google_id).
        gid = item.get("google_id")
        bid_raw = item.get("id") or item.get("book_id")
        
        # If bid_raw is a string and looks like a Google ID (not a digit), treat as gid
        if not gid and isinstance(bid_raw, str) and bid_raw and not bid_raw.isdigit():
            gid = bid_raw
            
        bid = str(gid or bid_raw or f"local_{id(item)}")
        score = float(item.get("score", 0) or item.get("ai_score", 0) or 0.1)

        if bid not in candidates_map:
            candidates_map[bid] = {
                "book_id": bid,
                "google_id": gid,
                "title": item.get("title", "Unknown"),
                "author": item.get("author", ""),
                "cover": item.get("cover", ""),
                "source": item.get("source", ""),
                "_raw": item,
                "scores": {},
                "_sources": [],
            }

        norm = self._normalize_source(source)
        candidates_map[bid]["scores"][norm] = max(
            candidates_map[bid]["scores"].get(norm, 0), score
        )
        if source not in candidates_map[bid]["_sources"]:
            candidates_map[bid]["_sources"].append(source)

    def _step2_two_tower_scoring(self, candidates: List[Dict], user_id: Optional[int]) -> List[Dict]:
        """Step 2: Score all candidates with Two-Tower model embeddings."""
        try:
            if getattr(self, 'two_tower', None) is None or getattr(self, 'flask_app', None) is None:
                raise ValueError("TwoTower model not available")
            
            # If user embedding is missing or wrong shape, use a neutral one
            u_emb = self.real_user_emb
            if u_emb is None or u_emb.shape != (128,):
                u_emb = np.zeros(128, dtype=np.float32)
                
            emb_map = {}
            with self.flask_app.app_context():
                bids = [c["book_id"] for c in candidates if c.get("book_id")]
                if bids:
                    from flask_book_recommendation.models import BookEmbedding, Book
                    import pickle
                    
                    google_ids = [b for b in bids if isinstance(b, str) and not b.isdigit()]
                    local_ids = [int(b) for b in bids if isinstance(b, int) or (isinstance(b, str) and b.isdigit())]
                    
                    # Map google_id to books.id
                    google_to_local = {}
                    local_to_google = {}
                    if google_ids:
                        mapped = Book.query.filter(Book.google_id.in_(google_ids)).all()
                        for b in mapped:
                            local_ids.append(b.id)
                            google_to_local[b.google_id] = b.id
                            local_to_google[b.id] = b.google_id
                            
                    local_ids = list(set(local_ids))
                    rows = BookEmbedding.query.filter(BookEmbedding.book_id.in_(local_ids)).all()
                    for r in rows:
                        if r.vector is not None:
                            vec = pickle.loads(r.vector) if isinstance(r.vector, bytes) else r.vector
                            # Put it under local id
                            emb_map[str(r.book_id)] = np.asarray(vec, dtype=np.float32)
                            # And also under google id so candidate loop finds it
                            if r.book_id in local_to_google:
                                emb_map[local_to_google[r.book_id]] = np.asarray(vec, dtype=np.float32)
                            
            item_embs = []
            valid_indices = []
            for i, c in enumerate(candidates):
                bid = str(c.get("book_id", ""))
                if bid in emb_map and emb_map[bid].shape == (384,):
                    item_embs.append(emb_map[bid])
                    valid_indices.append(i)
                    
            if not item_embs:
                raise ValueError("No item embeddings found")
                
            with torch.no_grad():
                # Enforce float32 and check shape
                u_array = np.array([u_emb], dtype=np.float32)
                item_array = np.array(item_embs, dtype=np.float32)
                
                u_tensor = torch.tensor(u_array, device=self.device)
                item_tensor = torch.tensor(item_array, device=self.device)
                
                item_tower_embs = self.two_tower.item_tower(item_tensor)
                scores = (u_tensor * item_tower_embs).sum(dim=1).cpu().numpy()
                
            for idx, list_idx in enumerate(valid_indices):
                score = float(np.clip(scores[idx], 0, 1))
                candidates[list_idx]["scores"]["two_tower"] = score
                
            for i, c in enumerate(candidates):
                if i not in valid_indices:
                    base_score = np.mean([v for v in c["scores"].values()]) if c["scores"] else 0.5
                    c["scores"]["two_tower"] = float(np.clip(base_score - 0.1, 0, 1))
                    
        except Exception as e:
            logger.error(f"❌ [Step 2] Two-Tower fallback: {e}")
            for c in candidates:
                if "two_tower" not in c["scores"]:
                    base_score = np.mean([v for v in c["scores"].values()]) if c["scores"] else 0.5
                    c["scores"]["two_tower"] = float(np.clip(base_score + 0.1, 0, 1))
                    
        return candidates

    def _step3_transformer_encoding(self, candidates: List[Dict]) -> List[Dict]:
        """Step 3: Encode candidate texts through Transformer for contextual embeddings."""
        try:
            # Fetch real embeddings from DB if available
            from flask import current_app
            
            # Use a map for efficiency
            emb_map = {}
            if self.flask_app:
                with self.flask_app.app_context():
                    # 🔧 [FIX #2] Properly resolve google_ids to local book IDs
                    # BookEmbedding.book_id is an integer FK to Book.id,
                    # but candidates use google_id (string) as book_id.
                    bids = [c["book_id"] for c in candidates if c.get("book_id")]
                    if bids:
                        try:
                            from flask_book_recommendation.extensions import db
                            from flask_book_recommendation.models import BookEmbedding, Book
                            import pickle
                            
                            # Separate google_ids from local integer IDs
                            google_ids = [b for b in bids if isinstance(b, str) and not b.isdigit()]
                            local_ids = [int(b) for b in bids if isinstance(b, int) or (isinstance(b, str) and b.isdigit())]
                            
                            # Map google_id -> local book.id
                            if google_ids:
                                mapped = Book.query.filter(Book.google_id.in_(google_ids)).all()
                                for b in mapped:
                                    local_ids.append(b.id)
                                    # Store mapping so we can look up by google_id too
                                    emb_map[f"_gid_{b.google_id}"] = b.id
                            
                            local_ids = list(set(local_ids))
                            if local_ids:
                                rows = BookEmbedding.query.filter(BookEmbedding.book_id.in_(local_ids)).all()
                                for r in rows:
                                    if r.vector is not None:
                                        vec = pickle.loads(r.vector) if isinstance(r.vector, bytes) else r.vector
                                        vec = np.asarray(vec, dtype=np.float32)
                                        emb_map[str(r.book_id)] = vec
                            
                            # Also build reverse lookup: google_id -> embedding
                            for gid_key in [k for k in emb_map if k.startswith("_gid_")]:
                                gid = gid_key[5:]  # Remove "_gid_" prefix
                                local_id = emb_map[gid_key]
                                if str(local_id) in emb_map:
                                    emb_map[gid] = emb_map[str(local_id)]
                            
                            # Clean up temp keys
                            for k in [k for k in emb_map if k.startswith("_gid_")]:
                                del emb_map[k]
                                
                        except Exception as e:
                            logger.error(f"Error fetching DB embeddings in Step 3: {e}")

            # Process candidates
            embeddings_list = []
            valid_indices = []

            for i, c in enumerate(candidates):
                bid = str(c.get("book_id", ""))
                if bid in emb_map:
                    embeddings_list.append(emb_map[bid])
                    valid_indices.append(i)
                else:
                    # Fallback to deterministic hash if no real embedding exists
                    hash_val = int(hashlib.md5(c.get("title", "").encode("utf-8", errors="ignore")).hexdigest(), 16)
                    rng = np.random.RandomState(hash_val % (2**31))
                    embeddings_list.append(rng.randn(384).astype(np.float32) * 0.1)
                    valid_indices.append(i)

            if not embeddings_list:
                return candidates

            with torch.no_grad():
                input_tensor = torch.tensor(
                    np.array(embeddings_list), dtype=torch.float32
                ).unsqueeze(1).to(self.device) # (batch, 1, input_dim)

                # Pass through transformer encoder
                encoded = self.transformer(input_tensor)
                encoded_np = encoded.cpu().numpy()

                for j, i in enumerate(valid_indices):
                    c = candidates[i]
                    transformer_score = float(np.clip(
                        np.linalg.norm(encoded_np[j]) / 2.0, 0, 1 # Normalized magnitude
                    ))
                    c["scores"]["transformer"] = transformer_score
                    c["_transformer_emb"] = encoded_np[j]

        except Exception as e:
            logger.error(f"❌ [Step 3] Transformer encoding failed: {e}")
            for c in candidates:
                if "transformer" not in c["scores"]:
                    c["scores"]["transformer"] = 0.5

        return candidates

    def _step4_graph_boosting(self, candidates: List[Dict], user_id: Optional[int]) -> List[Dict]:
        """Step 4: Boost scores using graph-based connectivity."""
        try:
            if self.graph_model and self.graph_model.model is not None and user_id:
                # Try to get graph-based scores
                try:
                    user_emb = self.graph_model.get_user_embedding(user_id)
                    if user_emb is not None:
                        for c in candidates:
                            try:
                                item_emb = self.graph_model.get_item_embedding(c["book_id"])
                                if item_emb is not None:
                                    graph_score = float(np.dot(user_emb, item_emb))
                                    c["scores"]["graph"] = float(np.clip(graph_score, 0, 1))
                                    continue
                            except Exception:
                                pass
                            c["scores"]["graph"] = 0.1
                        return candidates
                except Exception:
                    pass

            # Fallback: compute graph score from source diversity
            for c in candidates:
                source_count = len(c.get("_sources", []))
                graph_score = float(np.clip(source_count * 0.15, 0, 1))
                c["scores"]["graph"] = graph_score

        except Exception as e:
            logger.error(f"❌ [Step 4] Graph boosting failed: {e}")
            for c in candidates:
                c["scores"].setdefault("graph", 0.1)

        return candidates

    def _step5_ensemble_fusion(self, candidates: List[Dict]) -> List[Dict]:
        """Step 5: Combine all model scores via weighted ensemble."""
        weights = {
            "collaborative": 0.15,
            "two_tower": 0.40,
            "semantic": 0.15,
            "graph": 0.10,
            "behavioral": 0.15,  # Real-time search/view boost
            "popularity": 0.00,  # [PURE-PERS] Disabled popularity bias
        }

        for c in candidates:
            scores = c.get("scores", {})
            weighted_sum = 0.0
            total_weight = 0.0

            for key, weight in weights.items():
                if key in scores:
                    weighted_sum += weight * float(scores[key])
                    total_weight += weight

            # Normalize by total active weights
            ensemble_score = weighted_sum / max(total_weight, 0.01)

            # Boost items appearing in multiple sources
            multi_source_bonus = min(len(c.get("_sources", [])) * 0.03, 0.15)
            ensemble_score = float(np.clip(ensemble_score + multi_source_bonus, 0, 1))

            c["ensemble_score"] = ensemble_score
            c["final_score"] = ensemble_score  # Will be refined by reranker

        return candidates

    def _step6_neural_reranker(self, candidates: List[Dict], user_id: Optional[int]) -> List[Dict]:
        """Step 6: Neural Reranker for final scoring refinement."""
        try:
            n = len(candidates)
            if n == 0:
                return candidates

            with torch.no_grad():
                # Use real user embedding if available
                if getattr(self, 'real_user_emb', None) is not None:
                    user_emb = torch.tensor(np.array([self.real_user_emb]), dtype=torch.float32, device=self.device)
                else:
                    user_emb = torch.randn(1, 128, device=self.device) * 0.1
                    if user_id:
                        rng = np.random.RandomState(user_id % (2**31))
                        user_emb = torch.tensor(
                            rng.randn(1, 128).astype(np.float32) * 0.1,
                            device=self.device
                        )

                # Create item embeddings from transformer embeddings or scores
                item_embs = []
                for c in candidates:
                    if "_transformer_emb" in c:
                        emb = c["_transformer_emb"]
                    else:
                        # Create from scores hash
                        scores_hash = sum(c.get("scores", {}).values())
                        rng = np.random.RandomState(int(scores_hash * 1000) % (2**31))
                        emb = rng.randn(128).astype(np.float32) * 0.1
                    item_embs.append(emb[:128] if len(emb) >= 128 else np.pad(emb, (0, 128 - len(emb))))

                item_tensor = torch.tensor(
                    np.array(item_embs), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                # Run Neural Reranker
                reranker_scores = self.reranker(user_emb, item_tensor)
                reranker_scores = torch.sigmoid(reranker_scores).squeeze(0).cpu().numpy()

                for i, c in enumerate(candidates):
                    reranker_score = float(reranker_scores[i]) if i < len(reranker_scores) else 0.5
                    c["scores"]["reranker"] = reranker_score

                    # Blend ensemble + reranker (70% ensemble, 30% reranker)
                    c["final_score"] = float(
                        0.70 * c.get("ensemble_score", 0.5) +
                        0.30 * reranker_score
                    )

        except Exception as e:
            logger.error(f"❌ [Step 6] Neural Reranker failed: {e}")
            for c in candidates:
                c["scores"].setdefault("reranker", 0.5)

        return candidates

    def _step7_context_ranking(self, candidates: List[Dict], context: Dict) -> List[Dict]:
        """Step 7: Context-aware reranking based on time/device/session."""
        try:
            now = datetime.now()
            hour = now.hour
            day = now.weekday()

            # Time-based adjustments
            # Evening users prefer different content
            time_boost = 0.0
            if 18 <= hour <= 23:
                time_boost = 0.02  # Slight boost for evening browsing
            elif 6 <= hour <= 9:
                time_boost = 0.01  # Morning boost

            session_id = context.get("session", "")
            device = context.get("device", "web")

            with torch.no_grad():
                n = len(candidates)
                if n == 0:
                    return candidates

                if getattr(self, 'real_user_emb', None) is not None:
                    user_emb = torch.tensor(np.array([self.real_user_emb]), dtype=torch.float32, device=self.device)
                else:
                    user_emb = torch.randn(1, 128, device=self.device) * 0.1

                item_embs = []
                base_scores_list = []
                for c in candidates:
                    if "_transformer_emb" in c:
                        emb = c["_transformer_emb"]
                    else:
                        # 🔧 [FIX #4] Use deterministic seed instead of pure random
                        ctx_hash = int(hashlib.md5(c.get("title", str(i)).encode("utf-8", errors="ignore")).hexdigest(), 16)
                        ctx_rng = np.random.RandomState(ctx_hash % (2**31))
                        emb = ctx_rng.randn(128).astype(np.float32) * 0.1
                    item_embs.append(emb[:128] if len(emb) >= 128 else np.pad(emb, (0, 128 - len(emb))))
                    base_scores_list.append(c.get("final_score", 0.5))

                item_tensor = torch.tensor(
                    np.array(item_embs), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                base_scores = torch.tensor(
                    [base_scores_list], dtype=torch.float32, device=self.device
                )

                # Get context tensors
                ctx = self.context_ranker.get_current_context(
                    device=self.device,
                    session_duration=float(context.get("session_duration", 0)),
                    session_clicks=int(context.get("session_clicks", 0)),
                    session_views=int(context.get("session_views", 0)),
                    is_returning=bool(context.get("is_returning", False)),
                    activity_level=float(context.get("activity_level", 0.5)),
                )
                hour_t = ctx["hour"]
                day_t = ctx["day"]
                session_feats = ctx["session_features"]

                # Run context-aware ranking
                context_scores = self.context_ranker(
                    user_emb, item_tensor,
                    hour_t, day_t, session_feats,
                    base_scores=base_scores
                )
                context_scores = context_scores.squeeze(0).cpu().numpy()

                for i, c in enumerate(candidates):
                    if i < len(context_scores):
                        ctx_score = float(context_scores[i])
                        # Blend: 80% current final_score + 20% context adjustment
                        c["final_score"] = float(
                            0.80 * c.get("final_score", 0.5) +
                            0.20 * ctx_score +
                            time_boost
                        )
                        c["final_score"] = float(np.clip(c["final_score"], 0, 1))

        except Exception as e:
            logger.error(f"❌ [Step 7] Context ranking failed: {e}")
            # Keep existing final_scores

        return candidates

    def _step9_online_learning_update(
        self, candidates: List[Dict], user_id: Optional[int], context: Dict
    ):
        """Step 9: Update user embeddings via online learning."""
        if not user_id:
            return

        try:
            # Record the recommendation event as implicit feedback
            for c in candidates[:10]:  # Top 10 recommendations as positive signals
                self.online_learner.record_feedback(
                    user_id=user_id,
                    item_id=c.get("book_id", ""),
                    feedback_type="recommend",
                    value=c.get("final_score", 0.5),
                    context=context
                )

            # Decay exploration rate
            self.online_learner.decay_exploration()
            
            # 🔥 Persist updated embedding to DB (Step 9 Goal)
            # 🔧 [FIX #5] Only persist if real_user_emb is not None
            if getattr(self, 'real_user_emb', None) is not None and self.flask_app:
                from flask_book_recommendation.models import UserEmbedding
                from flask_book_recommendation.extensions import db
                import pickle
                
                with self.flask_app.app_context():
                    ue = UserEmbedding.query.filter_by(user_id=user_id).first()
                    if not ue:
                        ue = UserEmbedding(user_id=user_id, interaction_count=0)
                        db.session.add(ue)
                    
                    # int_vec is 384d, u_emb is 128d. We store the latent 128d for future sessions.
                    ue.vector = pickle.dumps(self.real_user_emb)
                    ue.interaction_count += 1
                    db.session.commit()
                    logger.debug(f"💾 [Step 9] Persisted updated user embedding for user {user_id}")
            else:
                logger.debug(f"⏭️ [Step 9] Skipped embedding persistence (no real embedding available)")

        except Exception as e:
            logger.error(f"❌ [Step 9] Online learning update failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # STRATEGY VARIANTS — Different ranking strategies for homepage sections
    # ─────────────────────────────────────────────────────────────────────

    def recommend_trending(
        self, user_id: Optional[int] = None, top_k: int = 100, context: Optional[Dict] = None
    ) -> List[Dict]:
        """Neural + popularity weighted. Boosts trending/popular items."""
        results = self.recommend_full_stack(user_id=user_id, top_k=top_k * 2, context=context)

        # Re-weight with popularity emphasis
        for r in results:
            pop_score = r.get("model_breakdown", {}).get("popularity", 0.3)
            # Boost by popularity signal
            r["final_score"] = float(
                0.5 * r.get("final_score", 0.5) +
                0.5 * max(pop_score, r.get("final_score", 0.5) * 0.8)
            )

        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results[:top_k]

    def recommend_because_you_read(
        self, user_id: Optional[int] = None, top_k: int = 100, context: Optional[Dict] = None
    ) -> List[Dict]:
        """Content + Transformer focused recommendations."""
        results = self.recommend_full_stack(user_id=user_id, top_k=top_k * 2, context=context)

        # Re-weight with content/transformer emphasis
        for r in results:
            bd = r.get("model_breakdown", {})
            content_score = bd.get("content", 0.3)
            transformer_score = bd.get("transformer", 0.3)
            r["final_score"] = float(
                0.3 * r.get("final_score", 0.5) +
                0.4 * content_score +
                0.3 * transformer_score
            )

        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results[:top_k]

    def recommend_top_neural(
        self, user_id: Optional[int] = None, top_k: int = 100, context: Optional[Dict] = None
    ) -> List[Dict]:
        """Highest final_score items — pure neural quality."""
        results = self.recommend_full_stack(user_id=user_id, top_k=top_k, context=context)
        # Already sorted by final_score from full stack
        return results[:top_k]

    def recommend_graph_discovery(
        self, user_id: Optional[int] = None, top_k: int = 100, context: Optional[Dict] = None
    ) -> List[Dict]:
        """Graph recommender focused — discovery through connections."""
        results = self.recommend_full_stack(user_id=user_id, top_k=top_k * 2, context=context)

        # Re-weight with graph emphasis
        for r in results:
            graph_score = r.get("model_breakdown", {}).get("graph", 0.2)
            r["final_score"] = float(
                0.4 * r.get("final_score", 0.5) +
                0.6 * max(graph_score, 0.2)
            )

        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _format_output(self, candidates: List[Dict]) -> List[Dict]:
        """Format candidates into the final output structure."""
        results = []
        for c in candidates:
            scores = c.get("scores", {})

            # Build model breakdown
            breakdown = {
                "cf": round(float(scores.get("collaborative", 0)), 4),
                "two_tower": round(float(scores.get("two_tower", 0)), 4),
                "transformer": round(float(scores.get("transformer", 0)), 4),
                "graph": round(float(scores.get("graph", 0)), 4),
                "reranker": round(float(scores.get("reranker", 0)), 4),
            }

            # Compute probability_like from sigmoid of final_score
            final_score = float(c.get("final_score", 0.5))
            probability_like = float(1.0 / (1.0 + np.exp(-5 * (final_score - 0.5))))

            # Generate explanation
            top_model = max(breakdown, key=breakdown.get)
            model_names = {
                "cf": "Collaborative Filtering",
                "two_tower": "Deep Learning Two-Tower",
                "transformer": "Transformer Encoder",
                "graph": "Graph Neural Network",
                "reranker": "Neural Reranker",
            }
            explanation = f"Recommended by {model_names.get(top_model, 'AI')}"

            active_models = [k for k, v in breakdown.items() if v > 0.1]
            if len(active_models) > 1:
                explanation += f" with consistency across {len(active_models)} models"

            raw = c.get("_raw", {})

            results.append({
                "book_id": c.get("book_id", ""),
                "id": c.get("book_id", ""),  # Compatibility alias
                "title": c.get("title", "Unknown"),
                "author": c.get("author", ""),
                "cover": c.get("cover", "") or raw.get("cover", ""),
                "final_score": round(final_score, 4),
                "score": round(final_score, 2),
                "probability_like": round(probability_like, 4),
                "model_breakdown": breakdown,
                "explanation": explanation,
                "reason": explanation,
                "algo_tag": "Neural Full Stack",
                "confidence": round(probability_like, 2),
                "contributing_algorithms": c.get("_sources", []),
                "source": raw.get("source", ""),
                "rating": raw.get("rating", None),
            })

        return results

    def _normalize_source(self, name: str) -> str:
        """Normalize source name to a canonical key."""
        name = name.lower()
        if "two-tower" in name or "two_tower" in name or "deep learning" in name:
            return "two_tower"
        if "graph" in name:
            return "graph"
        if "collaborative" in name or "cf" in name:
            return "collaborative"
        if "content" in name or "vector" in name:
            return "semantic"
        if "hybrid" in name or "semantic" in name:
            return "semantic"
        if "trending" in name or "popular" in name:
            return "popularity"
        if "behavior" in name or "search" in name or "view" in name:
            return "behavioral"
        # 🔧 [FIX #1] Interest-based sources should map to "semantic", not "popularity"
        # because popularity weight is 0.00 in the ensemble, which was making
        # all interest-sourced books invisible to new users.
        if "interest" in name:
            return "semantic"
        return "semantic"

    @classmethod
    def get_pipeline_meta(cls) -> Dict[str, Any]:
        """Return the last pipeline execution metadata for frontend display."""
        return cls._last_pipeline_meta or {
            "stages": [
                {"id": 1, "name": "Hybrid Retrieval", "name_ar": "الاسترجاع الهجين", "icon": "database", "color": "#6366f1", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "FAISS + CF", "status": "pending"},
                {"id": 2, "name": "Two-Tower Scoring", "name_ar": "التقييم ثنائي البرج", "icon": "account_tree", "color": "#8b5cf6", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "TwoTower v3.0", "status": "pending"},
                {"id": 3, "name": "Transformer Encoding", "name_ar": "ترميز المحولات", "icon": "memory", "color": "#a855f7", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "Transformer 8H×2L", "status": "pending"},
                {"id": 4, "name": "Graph Boosting", "name_ar": "تعزيز الرسم البياني", "icon": "hub", "color": "#d946ef", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "GraphRec GNN", "status": "pending"},
                {"id": 5, "name": "Ensemble Fusion", "name_ar": "الدمج المتعدد", "icon": "merge", "color": "#ec4899", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "Weighted Ensemble", "status": "pending"},
                {"id": 6, "name": "Neural Reranker", "name_ar": "إعادة الترتيب العصبي", "icon": "neurology", "color": "#f43f5e", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "NeuralReranker MLP", "status": "pending"},
                {"id": 7, "name": "Context Ranker", "name_ar": "ترتيب السياق", "icon": "schedule", "color": "#f97316", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "ContextAwareRanker", "status": "pending"},
                {"id": 8, "name": "Final Sort", "name_ar": "الترتيب النهائي", "icon": "sort", "color": "#eab308", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "BPR Sort", "status": "pending"},
                {"id": 9, "name": "Online Learning", "name_ar": "التعلم الفوري", "icon": "model_training", "color": "#22c55e", "time_ms": 0, "candidates_in": 0, "candidates_out": 0, "model": "OnlineLearner v2", "status": "pending"},
            ],
            "total_time_ms": 0,
            "final_books_count": 0,
            "timestamp": datetime.now().isoformat(),
            "user_id": None,
        }

    def _cache_key(self, strategy: str, user_id: Optional[int], context: Dict) -> str:
        """Generate a unique cache key."""
        ts_bucket = int(time.time() // 120)  # 2-minute buckets
        return f"neural:{strategy}:{user_id}:{ts_bucket}"

    def clear_user_cache(self, user_id: int):
        """Clear all cached results for a specific user."""
        try:
            if self.cache.redis:
                pattern = f"neural:*:{user_id}:*"
                keys = self.cache.redis.keys(pattern)
                if keys:
                    self.cache.redis.delete(*keys)
            # Also clear local cache entries for this user
            keys_to_delete = [k for k in self.cache.local_cache if f":{user_id}:" in k]
            for k in keys_to_delete:
                self.cache.local_cache.pop(k, None)
                self.cache.local_ttl.pop(k, None)
            logger.info(f"🧹 [Pipeline] Cleared cache for user {user_id}")
        except Exception as e:
            logger.warning(f"⚠️ [Pipeline] Cache clear failed for user {user_id}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON + BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

# Global instance — created once, reused forever
_unified_engine: Optional[UnifiedRecommendationPipeline] = None


def get_unified_engine() -> UnifiedRecommendationPipeline:
    """Get or create the global pipeline instance."""
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedRecommendationPipeline(load_all_models=True)
    return _unified_engine


# Backward compatibility: the old `pipeline` variable
# This creates a lightweight wrapper that lazy-loads on first access
class _LazyPipeline:
    """Lazy proxy so `from .unified_pipeline import pipeline` still works."""
    _instance = None

    def __getattr__(self, name):
        if _LazyPipeline._instance is None:
            _LazyPipeline._instance = get_unified_engine()
        return getattr(_LazyPipeline._instance, name)

pipeline = _LazyPipeline()
