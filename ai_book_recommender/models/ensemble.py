# -*- coding: utf-8 -*-
"""
🎼 Ensemble Ranker
===================

Combines multiple recommendation signals using:
- Weighted combination
- Learn-to-rank fusion
- Dynamic weight adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeights:
    """Default weights for ensemble components."""
    two_tower: float = 0.45    # 🚀 Increased: Strongest personalization signal
    graph: float = 0.05        # 📉 Decreased: Can sometimes pull broad matches
    collaborative: float = 0.10 # 📉 Decreased: Unreliable for cold-start users
    semantic: float = 0.35     # 🚀 Increased: Best for interest-based matching
    behavioral: float = 0.05   # 📉 Decreased: Handled partially by Two-Tower
    popularity: float = 0.00   # ❌ Disabled: Purely personalized
    diversity: float = 0.03
    novelty: float = 0.02


class EnsembleRanker:
    """
    🎼 Weighted Ensemble Ranker
    
    Combines multiple recommendation signals using configurable weights.
    
    Score formula:
        FinalScore = Σ (weight_i × normalize(score_i))
    
    With optional adjustments:
        - Position bias correction
        - Diversity penalty
        - Novelty boost
    """
    
    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        normalize: bool = True,
        position_bias_correction: bool = True
    ):
        self.weights = weights or EnsembleWeights()
        self.normalize = normalize
        self.position_bias_correction = position_bias_correction
        
        logger.info(f"EnsembleRanker initialized with weights: {self.weights}")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)
    
    def combine(
        self,
        scores: Dict[str, np.ndarray],
        item_ids: List[str],
        seen_categories: Optional[List[str]] = None,
        user_history: Optional[List[str]] = None
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Combine scores from multiple sources.
        
        Args:
            scores: Dictionary of score arrays by source name
                Keys: "two_tower", "graph", "collaborative", "semantic", "popularity"
            item_ids: List of item IDs corresponding to scores
            seen_categories: Categories already shown (for diversity)
            user_history: User's interaction history (for novelty)
            
        Returns:
            List of (item_id, final_score, breakdown) tuples, sorted by score
        """
        n = len(item_ids)
        if n == 0:
            return []
        
        # Initialize final scores
        final_scores = np.zeros(n)
        breakdown = [{} for _ in range(n)]
        
        # Weight map
        weight_map = {
            "two_tower": self.weights.two_tower,
            "graph": self.weights.graph,
            "collaborative": self.weights.collaborative,
            "semantic": self.weights.semantic,
            "behavioral": self.weights.behavioral,
            "popularity": self.weights.popularity,
        }
        
        # Combine weighted scores
        total_weight = 0.0
        
        for source, source_scores in scores.items():
            if source not in weight_map:
                continue
            
            weight = weight_map[source]
            if weight <= 0:
                continue
            
            # Ensure same length
            if len(source_scores) != n:
                logger.warning(f"Score length mismatch for {source}: {len(source_scores)} vs {n}")
                continue
            
            # Normalize
            if self.normalize:
                source_scores = self._normalize_scores(np.array(source_scores))
            else:
                source_scores = np.array(source_scores)
            
            # Add weighted contribution
            weighted = weight * source_scores
            final_scores += weighted
            total_weight += weight
            
            # Track breakdown
            for i in range(n):
                breakdown[i][source] = float(source_scores[i])
        
        # Normalize by total weight
        if total_weight > 0:
            final_scores /= total_weight
        
        # Apply diversity penalty
        if seen_categories:
            diversity_penalty = self._compute_diversity_penalty(
                item_ids, seen_categories
            )
            final_scores = final_scores * (1 - self.weights.diversity * diversity_penalty)
            for i in range(n):
                breakdown[i]["diversity_penalty"] = float(diversity_penalty[i])
        
        # Apply novelty boost
        if user_history:
            novelty_boost = self._compute_novelty_boost(item_ids, user_history)
            final_scores = final_scores * (1 + self.weights.novelty * novelty_boost)
            for i in range(n):
                breakdown[i]["novelty_boost"] = float(novelty_boost[i])
        
        # Position bias correction
        if self.position_bias_correction:
            # Items at similar scores but different original positions
            # should have position bias removed
            pass  # Handled at a higher level typically
        
        # Create results
        results = [
            (item_ids[i], float(final_scores[i]), breakdown[i])
            for i in range(n)
        ]
        
        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _compute_diversity_penalty(
        self,
        item_ids: List[str],
        seen_categories: List[str]
    ) -> np.ndarray:
        """Compute penalty for items from already-shown categories."""
        # This would need item metadata - placeholder
        return np.zeros(len(item_ids))
    
    def _compute_novelty_boost(
        self,
        item_ids: List[str],
        user_history: List[str]
    ) -> np.ndarray:
        """Compute boost for novel items."""
        history_set = set(user_history)
        # Items in history get no boost, others get boost
        return np.array([
            0.0 if iid in history_set else 1.0
            for iid in item_ids
        ])
    
    def update_weights(self, **kwargs) -> None:
        """Update ensemble weights dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
        
        logger.info(f"Updated weights: {self.weights}")


class LearnToRankEnsemble(nn.Module):
    """
    🎓 Learn-to-Rank Ensemble
    
    Neural network that learns optimal combination of signals.
    Uses LambdaRank-style training.
    
    Input features per item:
    - All source scores
    - Item features
    - User-item interaction features
    
    Output: Ranking score
    """
    
    def __init__(
        self,
        num_score_sources: int = 5,
        item_feature_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input: scores + item features
        input_dim = num_score_sources + item_feature_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learned weights for interpretability
        self.score_weights = nn.Parameter(torch.ones(num_score_sources) / num_score_sources)
        
        self._init_weights()
        
        logger.info(
            f"LearnToRankEnsemble: {num_score_sources} sources, "
            f"item_features={item_feature_dim}"
        )
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        source_scores: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ensemble scores.
        
        Args:
            source_scores: Scores from each source (batch, num_items, num_sources)
            item_features: Optional item features (batch, num_items, feature_dim)
            
        Returns:
            Final scores (batch, num_items)
        """
        batch_size, num_items, num_sources = source_scores.shape
        
        # Apply learned weights to scores first (for interpretability)
        weights = F.softmax(self.score_weights, dim=0)
        weighted_sum = (source_scores * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        # Combine with item features if provided
        if item_features is not None:
            features = torch.cat([source_scores, item_features], dim=-1)
        else:
            features = source_scores
        
        # Neural refinement
        neural_scores = self.net(features).squeeze(-1)
        
        # Combine weighted sum with neural scores
        final_scores = 0.5 * weighted_sum + 0.5 * neural_scores
        
        return final_scores
    
    def get_weights(self) -> Dict[str, float]:
        """Get current learned weights."""
        weights = F.softmax(self.score_weights, dim=0).detach().cpu().numpy()
        sources = ["two_tower", "graph", "collaborative", "semantic", "popularity"]
        return {s: float(w) for s, w in zip(sources, weights)}
    
    def lambda_loss(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """
        LambdaRank-style loss.
        
        Approximates NDCG optimization by weighting pairwise comparisons
        by their impact on NDCG.
        
        Args:
            scores: Predicted scores (batch, num_items)
            relevance: Ground truth relevance (batch, num_items)
            k: Cutoff for NDCG calculation
            
        Returns:
            Loss value
        """
        batch_size, num_items = scores.shape
        device = scores.device
        
        # Compute pairwise differences
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (batch, n, n)
        rel_diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)
        
        # Target: positive if i should rank higher than j
        target = (rel_diff > 0).float()
        
        # Compute DCG gains
        gains = 2 ** relevance - 1
        
        # Compute discount factors
        positions = torch.arange(1, num_items + 1, device=device).float()
        discounts = 1 / torch.log2(positions + 1)
        
        # Approximate delta NDCG for each pair swap
        # Simplified: weight by gain difference
        gain_diff = torch.abs(gains.unsqueeze(2) - gains.unsqueeze(1))
        
        # Cross-entropy weighted by delta NDCG
        loss = F.binary_cross_entropy_with_logits(
            score_diff,
            target,
            weight=gain_diff
        )
        
        return loss
