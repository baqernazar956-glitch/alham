# -*- coding: utf-8 -*-
"""
⚙️ Configuration Management
============================

Centralized configuration for the AI recommendation engine.
Uses Pydantic for validation and environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Neural model hyperparameters."""
    
    # Embedding dimensions
    embedding_dim: int = 384
    hidden_dim: int = 256
    output_dim: int = 128
    
    # User Tower
    user_embedding_dim: int = 128
    max_users: int = 100000
    history_seq_len: int = 20
    
    # Item Tower
    book_embedding_dim: int = 384
    
    # Graph Model
    num_gnn_layers: int = 3
    gnn_hidden_dim: int = 256
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 50
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5
    
    # Transformer
    num_attention_heads: int = 8
    num_transformer_layers: int = 2


@dataclass
class RetrievalConfig:
    """Vector search and retrieval settings."""
    
    # FAISS settings
    index_type: str = "IVF"  # "Flat", "IVF", "HNSW"
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    use_gpu: bool = False
    
    # Search settings
    default_k: int = 50
    rerank_k: int = 100  # Retrieve more for reranking
    
    # Hybrid retrieval weights
    semantic_weight: float = 0.5
    keyword_weight: float = 0.3
    popularity_weight: float = 0.2


@dataclass
class EnsembleWeights:
    """Weights for combining different recommendation signals."""
    
    two_tower: float = 0.30
    graph: float = 0.20
    collaborative: float = 0.15
    semantic: float = 0.15
    popularity: float = 0.10
    diversity: float = 0.05
    novelty: float = 0.05
    
    # Bias correction
    position_bias_penalty: float = 0.10
    
    def normalize(self) -> Dict[str, float]:
        """Ensure weights sum to 1.0."""
        total = (
            self.two_tower + self.graph + self.collaborative +
            self.semantic + self.popularity + self.diversity + self.novelty
        )
        return {
            "two_tower": self.two_tower / total,
            "graph": self.graph / total,
            "collaborative": self.collaborative / total,
            "semantic": self.semantic / total,
            "popularity": self.popularity / total,
            "diversity": self.diversity / total,
            "novelty": self.novelty / total,
        }


@dataclass
class OnlineLearningConfig:
    """Real-time learning settings."""
    
    # Multi-Armed Bandit
    epsilon: float = 0.1  # Exploration rate
    ucb_c: float = 2.0   # UCB exploration parameter
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    
    # Feedback processing
    click_weight: float = 1.0
    rating_weight: float = 2.0
    view_duration_weight: float = 0.5
    save_weight: float = 1.5
    
    # Learning rate for online updates
    online_lr: float = 0.01
    
    # Decay factors
    temporal_decay_rate: float = 0.95
    interest_decay_days: int = 30


@dataclass
class ExplainabilityConfig:
    """XAI settings."""
    
    # Number of reasons to show
    max_reasons: int = 3
    min_confidence: float = 0.3
    
    # Templates
    enable_natural_language: bool = True
    language: str = "en"  # "en" or "ar"


@dataclass
class AdvancedConfig:
    """Advanced features configuration."""
    
    # Cold Start
    cold_start_threshold: int = 5  # Min interactions before personalization
    onboarding_genres_count: int = 5
    
    # Diversity (MMR)
    mmr_lambda: float = 0.5  # Balance relevance vs diversity
    max_per_author: int = 2
    max_per_category: int = 3
    
    # Novelty & Serendipity
    novelty_boost: float = 0.1
    serendipity_rate: float = 0.05
    
    # Debiasing
    popularity_debiasing: bool = True
    position_bias_correction: bool = True
    fairness_constraints: bool = True


@dataclass 
class CacheConfig:
    """Caching settings."""
    
    enabled: bool = True
    backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379"
    
    # TTL in seconds
    user_embedding_ttl: int = 300
    recommendation_ttl: int = 60
    search_ttl: int = 120


@dataclass
class Config:
    """Main configuration class."""
    
    # App info
    app_name: str = "AI Book Recommender"
    version: str = "1.0.0"
    debug: bool = False
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    model_dir: Path = field(default_factory=lambda: Path("instance/models"))
    index_dir: Path = field(default_factory=lambda: Path("instance/indexes"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ensemble: EnsembleWeights = field(default_factory=EnsembleWeights)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    
    # Sentence Transformer model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.model_dir, self.index_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG", "").lower() == "true"
        
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL")
        
        if os.getenv("API_PORT"):
            config.api_port = int(os.getenv("API_PORT"))
        
        if os.getenv("REDIS_URL"):
            config.cache.backend = "redis"
            config.cache.redis_url = os.getenv("REDIS_URL")
        
        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
