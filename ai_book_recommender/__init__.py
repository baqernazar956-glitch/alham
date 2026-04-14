# -*- coding: utf-8 -*-
"""
🧠 AI Book Recommender - Production-Grade Recommendation Engine
================================================================

A hybrid AI recommendation platform combining:
- Deep Learning (Two-Tower, Graph Neural Networks)
- Transformers (Sentence Embeddings)
- Vector Search (FAISS)
- Collaborative Filtering
- Reinforcement Learning
- Explainable AI

Architecture:
- models/          Neural network architectures
- retrieval/       Vector search and hybrid retrieval
- user_intelligence/ Behavior and interest modeling
- explainability/  XAI explanations
- evaluation/      Metrics and evaluation

Author: AI Recommendation System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Recommendation System"

from .config import Config, get_config
from .feature_store import FeatureStore, UserFeatures, BookFeatures
from .engine import (
    RecommendationEngine,
    RecommendationRequest,
    RecommendationResponse,
    get_engine,
)

__all__ = [
    "__version__",
    "__author__",
    # Config
    "Config",
    "get_config",
    # Feature Store
    "FeatureStore",
    "UserFeatures",
    "BookFeatures",
    # Engine
    "RecommendationEngine",
    "RecommendationRequest",
    "RecommendationResponse",
    "get_engine",
]

