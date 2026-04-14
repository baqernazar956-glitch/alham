# -*- coding: utf-8 -*-
"""
🎯 Neural Re-ranking Model
===========================

Neural re-ranker for improving initial retrieval results.
Uses cross-attention between user and candidate items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class CrossAttentionReranker(nn.Module):
    """
    🎯 Cross-Attention Re-ranker
    
    Re-ranks candidates using cross-attention between:
    - User representation (query)
    - Candidate item representations (documents)
    
    Architecture:
    1. User embedding → Query projection
    2. Candidates → Key/Value projection
    3. Cross-attention → Attended representation
    4. Score MLP → Final ranking scores
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Query projection (user)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        
        # Key/Value projection (items)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Score computation
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
        logger.info(
            f"CrossAttentionReranker: input={input_dim}, hidden={hidden_dim}, "
            f"heads={num_heads}, layers={num_layers}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        user_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute re-ranking scores.
        
        Args:
            user_emb: User embedding (batch, input_dim)
            candidate_embs: Candidate embeddings (batch, num_candidates, input_dim)
            candidate_mask: Optional mask (batch, num_candidates)
            
        Returns:
            Scores (batch, num_candidates)
        """
        batch_size, num_candidates, _ = candidate_embs.shape
        
        # Project user query
        query = self.query_proj(user_emb)  # (batch, hidden)
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        
        # Project candidates
        keys = self.key_proj(candidate_embs)  # (batch, num_cand, hidden)
        values = self.value_proj(candidate_embs)
        
        # Apply attention layers
        attended = query
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn(
                attended, keys, values,
                key_padding_mask=~candidate_mask if candidate_mask is not None else None
            )
            attended = norm(attended + attn_out)
        
        # Expand attended for each candidate
        attended = attended.expand(-1, num_candidates, -1)  # (batch, num_cand, hidden)
        
        # Combine attended query with candidate values
        combined = torch.cat([attended, values], dim=-1)  # (batch, num_cand, hidden*2)
        
        # Compute scores
        scores = self.score_mlp(combined).squeeze(-1)  # (batch, num_candidates)
        
        return scores
    
    def rerank(
        self,
        user_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        initial_scores: Optional[torch.Tensor] = None,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Re-rank candidates, optionally combining with initial scores.
        
        Args:
            user_emb: User embedding (batch, dim) or (1, dim)
            candidate_embs: Candidates (batch, num_cand, dim) or (num_cand, dim)
            initial_scores: Optional initial retrieval scores
            alpha: Weight for neural scores vs initial scores
            
        Returns:
            Re-ranked scores
        """
        # Ensure proper dimensions
        if candidate_embs.dim() == 2:
            candidate_embs = candidate_embs.unsqueeze(0)
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        
        # Get neural scores
        with torch.no_grad():
            neural_scores = self.forward(user_emb, candidate_embs)
            neural_scores = torch.sigmoid(neural_scores)  # Normalize to [0, 1]
        
        # Combine with initial scores
        if initial_scores is not None:
            if initial_scores.dim() == 1:
                initial_scores = initial_scores.unsqueeze(0)
            
            # Normalize initial scores to [0, 1]
            initial_min = initial_scores.min(dim=-1, keepdim=True)[0]
            initial_max = initial_scores.max(dim=-1, keepdim=True)[0]
            initial_norm = (initial_scores - initial_min) / (initial_max - initial_min + 1e-8)
            
            final_scores = alpha * neural_scores + (1 - alpha) * initial_norm
        else:
            final_scores = neural_scores
        
        return final_scores.squeeze(0)


class NeuralReranker(nn.Module):
    """
    🎯 Neural Re-ranker
    
    Simpler re-ranker using direct user-item interaction modeling.
    
    Features:
    - Position bias correction
    - Listwise ranking
    - Feature fusion
    """
    
    def __init__(
        self,
        user_dim: int = 128,
        item_dim: int = 128,
        hidden_dim: int = 256,
        num_features: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature fusion
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        
        # Additional features (popularity, recency, etc.)
        self.feature_proj = nn.Linear(num_features, hidden_dim // 2)
        
        # Score network
        total_dim = hidden_dim * 2 + hidden_dim // 2
        
        self.score_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Position bias (learned)
        self.position_bias = nn.Embedding(100, 1)  # Up to 100 positions
        
        logger.info(
            f"NeuralReranker: user_dim={user_dim}, item_dim={item_dim}, "
            f"hidden={hidden_dim}"
        )
    
    def forward(
        self,
        user_emb: torch.Tensor,
        item_embs: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute re-ranking scores.
        
        Args:
            user_emb: (batch, user_dim)
            item_embs: (batch, num_items, item_dim)
            features: Optional (batch, num_items, num_features)
            positions: Optional original positions (batch, num_items)
            
        Returns:
            Scores (batch, num_items)
        """
        batch_size, num_items, _ = item_embs.shape
        
        # Project embeddings
        user_proj = self.user_proj(user_emb)  # (batch, hidden)
        item_proj = self.item_proj(item_embs)  # (batch, num_items, hidden)
        
        # Expand user for each item
        user_expanded = user_proj.unsqueeze(1).expand(-1, num_items, -1)
        
        # Process additional features
        if features is not None:
            feat_proj = self.feature_proj(features)
        else:
            feat_proj = torch.zeros(
                batch_size, num_items, self.feature_proj.out_features,
                device=item_embs.device
            )
        
        # Combine
        combined = torch.cat([user_expanded, item_proj, feat_proj], dim=-1)
        
        # Score
        scores = self.score_net(combined).squeeze(-1)  # (batch, num_items)
        
        # Position bias correction
        if positions is not None:
            positions = positions.clamp(0, 99)
            pos_bias = self.position_bias(positions).squeeze(-1)
            scores = scores - pos_bias  # Subtract bias to correct
        
        return scores
    
    def listwise_loss(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Listwise ranking loss (ListNet).
        
        Args:
            scores: Predicted scores (batch, num_items)
            relevance: Ground truth relevance (batch, num_items)
            temperature: Softmax temperature
            
        Returns:
            Loss value
        """
        # Convert to probabilities
        pred_probs = F.softmax(scores / temperature, dim=-1)
        true_probs = F.softmax(relevance / temperature, dim=-1)
        
        # Cross-entropy
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=-1)
        
        return loss.mean()
    
    def pairwise_loss(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """
        Pairwise margin ranking loss.
        
        Args:
            scores: Predicted scores (batch, num_items)
            relevance: Ground truth relevance (batch, num_items)
            margin: Margin for ranking
            
        Returns:
            Loss value
        """
        batch_size, num_items = scores.shape
        
        # Compare all pairs
        scores_i = scores.unsqueeze(2)  # (batch, num_items, 1)
        scores_j = scores.unsqueeze(1)  # (batch, 1, num_items)
        
        rel_i = relevance.unsqueeze(2)
        rel_j = relevance.unsqueeze(1)
        
        # Target: +1 if i should rank higher, -1 if j should
        target = torch.sign(rel_i - rel_j)
        
        # Margin ranking loss
        loss = torch.clamp(margin - target * (scores_i - scores_j), min=0)
        
        # Mask diagonal
        mask = ~torch.eye(num_items, device=scores.device, dtype=torch.bool)
        loss = loss[:, mask].mean()
        
        return loss
