# -*- coding: utf-8 -*-
"""
🧑 Dynamic User Model
======================

Real-time user representation with:
- Interest evolution tracking
- Multi-interest representation
- Temporal preference decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with multi-dimensional interests."""
    
    user_id: int
    
    # Core embeddings
    static_embedding: Optional[np.ndarray] = None  # Learned traits
    dynamic_embedding: Optional[np.ndarray] = None  # Current interests
    
    # Multi-interest representation
    interest_clusters: List[np.ndarray] = field(default_factory=list)
    interest_weights: List[float] = field(default_factory=list)
    
    # Behavioral stats
    total_interactions: int = 0
    recent_interactions: int = 0
    categories_viewed: Dict[str, int] = field(default_factory=dict)
    authors_viewed: Dict[str, int] = field(default_factory=dict)
    
    # Temporal info
    last_active: Optional[datetime] = None
    session_count: int = 0
    
    # Engagement metrics
    avg_dwell_time: float = 0.0
    click_through_rate: float = 0.0
    rating_tendency: float = 3.0  # Average rating given
    
    def get_combined_embedding(self) -> Optional[np.ndarray]:
        """Get combined user embedding."""
        if self.static_embedding is None and self.dynamic_embedding is None:
            return None
        
        if self.static_embedding is None:
            return self.dynamic_embedding
        
        if self.dynamic_embedding is None:
            return self.static_embedding
        
        # Weighted combination
        return 0.3 * self.static_embedding + 0.7 * self.dynamic_embedding


class UserProfiler:
    """
    🧑 User Profiler
    
    Builds and maintains user profiles from behavior data.
    
    Features:
    - Profile aggregation
    - Interest clustering
    - Temporal decay
    - Cold-start handling
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_interest_clusters: int = 5,
        decay_rate: float = 0.95,
        decay_period_days: int = 7
    ):
        self.embedding_dim = embedding_dim
        self.num_interest_clusters = num_interest_clusters
        self.decay_rate = decay_rate
        self.decay_period_days = decay_period_days
        
        # Cache
        self._profiles: Dict[int, UserProfile] = {}
        
        logger.info(
            f"UserProfiler: dim={embedding_dim}, clusters={num_interest_clusters}"
        )
    
    def get_profile(self, user_id: int) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]
    
    def update_from_interaction(
        self,
        user_id: int,
        item_embedding: np.ndarray,
        interaction_type: str = "view",  # view, click, rate, purchase
        item_category: Optional[str] = None,
        item_author: Optional[str] = None,
        dwell_time: Optional[float] = None,
        rating: Optional[float] = None
    ) -> UserProfile:
        """
        Update user profile from interaction.
        
        Args:
            user_id: User identifier
            item_embedding: Embedding of interacted item
            interaction_type: Type of interaction
            item_category: Category of item
            item_author: Author of item
            dwell_time: Time spent (seconds)
            rating: Rating given (if any)
            
        Returns:
            Updated profile
        """
        profile = self.get_profile(user_id)
        
        # Update interaction counts
        profile.total_interactions += 1
        profile.recent_interactions += 1
        profile.last_active = datetime.now()
        
        # Update category/author preferences
        if item_category:
            profile.categories_viewed[item_category] = \
                profile.categories_viewed.get(item_category, 0) + 1
        
        if item_author:
            profile.authors_viewed[item_author] = \
                profile.authors_viewed.get(item_author, 0) + 1
        
        # Update dwell time average
        if dwell_time:
            n = profile.total_interactions
            profile.avg_dwell_time = (
                profile.avg_dwell_time * (n - 1) + dwell_time
            ) / n
        
        # Update rating tendency
        if rating is not None:
            n = profile.total_interactions
            profile.rating_tendency = (
                profile.rating_tendency * (n - 1) + rating
            ) / n
        
        # Update dynamic embedding using exponential moving average
        weight = self._get_interaction_weight(interaction_type)
        self._update_dynamic_embedding(profile, item_embedding, weight)
        
        # Update interest clusters
        if len(profile.interest_clusters) < self.num_interest_clusters:
            self._add_interest_cluster(profile, item_embedding)
        else:
            self._update_interest_clusters(profile, item_embedding)
        
        return profile
    
    def _get_interaction_weight(self, interaction_type: str) -> float:
        """Get weight for interaction type."""
        weights = {
            "view": 0.1,
            "click": 0.2,
            "rate": 0.4,
            "purchase": 0.5,
            "save": 0.3,
            "share": 0.3
        }
        return weights.get(interaction_type, 0.1)
    
    def _update_dynamic_embedding(
        self,
        profile: UserProfile,
        item_embedding: np.ndarray,
        weight: float
    ) -> None:
        """Update dynamic embedding with EMA."""
        if profile.dynamic_embedding is None:
            profile.dynamic_embedding = item_embedding.copy()
        else:
            # Exponential moving average
            alpha = weight / (1 + profile.total_interactions * 0.01)
            profile.dynamic_embedding = (
                (1 - alpha) * profile.dynamic_embedding +
                alpha * item_embedding
            )
            # Normalize
            norm = np.linalg.norm(profile.dynamic_embedding)
            if norm > 0:
                profile.dynamic_embedding /= norm
    
    def _add_interest_cluster(
        self,
        profile: UserProfile,
        item_embedding: np.ndarray
    ) -> None:
        """Add new interest cluster."""
        profile.interest_clusters.append(item_embedding.copy())
        profile.interest_weights.append(1.0)
    
    def _update_interest_clusters(
        self,
        profile: UserProfile,
        item_embedding: np.ndarray
    ) -> None:
        """Update nearest interest cluster."""
        # Find nearest cluster
        similarities = [
            np.dot(item_embedding, cluster)
            for cluster in profile.interest_clusters
        ]
        nearest_idx = np.argmax(similarities)
        
        # Update cluster with EMA
        alpha = 0.1
        profile.interest_clusters[nearest_idx] = (
            (1 - alpha) * profile.interest_clusters[nearest_idx] +
            alpha * item_embedding
        )
        
        # Normalize
        norm = np.linalg.norm(profile.interest_clusters[nearest_idx])
        if norm > 0:
            profile.interest_clusters[nearest_idx] /= norm
        
        # Increase weight
        profile.interest_weights[nearest_idx] += 0.1
    
    def apply_temporal_decay(self, user_id: int) -> None:
        """Apply temporal decay to user interests."""
        if user_id not in self._profiles:
            return
        
        profile = self._profiles[user_id]
        
        if profile.last_active is None:
            return
        
        days_inactive = (datetime.now() - profile.last_active).days
        decay_periods = days_inactive // self.decay_period_days
        
        if decay_periods > 0:
            decay = self.decay_rate ** decay_periods
            profile.recent_interactions = int(profile.recent_interactions * decay)
            
            # Decay interest weights
            profile.interest_weights = [
                w * decay for w in profile.interest_weights
            ]
    
    def get_top_categories(self, user_id: int, k: int = 5) -> List[Tuple[str, int]]:
        """Get user's top categories."""
        profile = self.get_profile(user_id)
        sorted_cats = sorted(
            profile.categories_viewed.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_cats[:k]
    
    def get_top_authors(self, user_id: int, k: int = 5) -> List[Tuple[str, int]]:
        """Get user's top authors."""
        profile = self.get_profile(user_id)
        sorted_authors = sorted(
            profile.authors_viewed.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_authors[:k]
    
    def is_cold_start(self, user_id: int, threshold: int = 5) -> bool:
        """Check if user is in cold-start phase."""
        profile = self.get_profile(user_id)
        return profile.total_interactions < threshold


class DynamicUserModel(nn.Module):
    """
    🧠 Dynamic User Embedding Model
    
    Neural network for generating user embeddings from behavior.
    
    Architecture:
    - Sequence encoder for history
    - Interest attention mechanism
    - Temporal position encoding
    - Multi-interest output heads
    """
    
    def __init__(
        self,
        item_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_interests: int = 4,
        max_history: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_interests = num_interests
        
        # Item projection
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        
        # Temporal position encoding
        self.position_embedding = nn.Embedding(max_history, hidden_dim)
        
        # Sequence encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Interest extraction heads
        self.interest_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_interests)
        ])
        
        # Aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(output_dim * num_interests, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
        
        logger.info(
            f"DynamicUserModel: item_dim={item_dim}, hidden={hidden_dim}, "
            f"output={output_dim}, interests={num_interests}"
        )
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        history_items: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate user embedding from history.
        
        Args:
            history_items: Item embeddings (batch, seq_len, item_dim)
            history_mask: Mask for valid positions (batch, seq_len)
            
        Returns:
            Tuple of (aggregated_embedding, [interest_embeddings])
        """
        batch_size, seq_len, _ = history_items.shape
        device = history_items.device
        
        # Project items
        items_proj = self.item_proj(history_items)  # (batch, seq, hidden)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)
        items_proj = items_proj + pos_emb.unsqueeze(0)
        
        # Encode sequence
        if history_mask is not None:
            src_key_padding_mask = ~history_mask
        else:
            src_key_padding_mask = None
        
        encoded = self.sequence_encoder(
            items_proj,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pool (mean over valid positions)
        if history_mask is not None:
            mask_expanded = history_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)
        
        # Extract interests
        interests = [head(pooled) for head in self.interest_heads]
        
        # Normalize interests
        interests = [F.normalize(interest, p=2, dim=1) for interest in interests]
        
        # Aggregate
        combined = torch.cat(interests, dim=1)
        aggregated = self.aggregator(combined)
        aggregated = F.normalize(aggregated, p=2, dim=1)
        
        return aggregated, interests
    
    def get_multi_interests(
        self,
        history_items: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Get only the multi-interest representations."""
        _, interests = self.forward(history_items, history_mask)
        return interests
