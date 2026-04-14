# -*- coding: utf-8 -*-
"""
📊 Behavior Sequence Model
===========================

Models user behavior sequences for:
- Session-aware recommendations
- Sequential pattern recognition
- Interest evolution tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Data for a single user session."""
    
    session_id: str
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Interactions in order
    item_ids: List[str] = field(default_factory=list)
    item_embeddings: List[np.ndarray] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)  # view, click, rate
    timestamps: List[datetime] = field(default_factory=list)
    
    # Computed features
    session_embedding: Optional[np.ndarray] = None
    
    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        if self.end_time is None or not self.timestamps:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 60
    
    @property
    def num_interactions(self) -> int:
        """Get number of interactions."""
        return len(self.item_ids)


class SessionEncoder(nn.Module):
    """
    🔄 Session Encoder
    
    Encodes a session of interactions into a fixed embedding.
    Uses attention over time-aware item representations.
    """
    
    def __init__(
        self,
        item_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Item projection
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        
        # Action embedding
        self.action_embedding = nn.Embedding(5, 16)  # view, click, rate, save, purchase
        
        # Time encoding (relative position in session)
        self.time_proj = nn.Linear(1, 16)
        
        # Combined projection
        self.combined_proj = nn.Linear(hidden_dim + 16 + 16, hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"SessionEncoder: hidden={hidden_dim}, output={output_dim}")
    
    def forward(
        self,
        item_embeddings: torch.Tensor,
        action_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode session.
        
        Args:
            item_embeddings: (batch, seq_len, item_dim)
            action_ids: (batch, seq_len) action type indices
            time_deltas: (batch, seq_len, 1) relative times (0 to 1)
            mask: (batch, seq_len) valid positions
            
        Returns:
            Session embedding (batch, output_dim)
        """
        # Project items
        items = self.item_proj(item_embeddings)  # (batch, seq, hidden)
        
        # Action embeddings
        actions = self.action_embedding(action_ids)  # (batch, seq, 16)
        
        # Time encoding
        times = self.time_proj(time_deltas)  # (batch, seq, 16)
        
        # Combine
        combined = torch.cat([items, actions, times], dim=-1)
        combined = self.combined_proj(combined)  # (batch, seq, hidden)
        
        # Self-attention
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        
        attended, _ = self.attention(
            combined, combined, combined,
            key_padding_mask=key_padding_mask
        )
        
        # Mean pooling over valid positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = attended.mean(dim=1)
        
        # Output
        output = self.output_proj(pooled)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class BehaviorSequenceModel(nn.Module):
    """
    📊 Behavior Sequence Model
    
    Models sequential patterns in user behavior for:
    - Next-item prediction
    - Session-aware ranking
    - Interest transitions
    
    Architecture:
    - Item embedding + Position encoding
    - Causal Transformer for sequential modeling
    - Prediction head for next item
    """
    
    def __init__(
        self,
        item_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        
        # Item projection
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Session encoder (for short-term context)
        self.session_encoder = SessionEncoder(
            item_dim=item_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=output_dim
        )
        
        self._init_weights()
        
        logger.info(
            f"BehaviorSequenceModel: hidden={hidden_dim}, layers={num_layers}, "
            f"max_seq={max_seq_len}"
        )
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate causal (lower triangular) attention mask."""
        # True = ignore this position
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask
    
    def forward(
        self,
        item_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for next-item prediction.
        
        Args:
            item_embeddings: (batch, seq_len, item_dim)
            mask: (batch, seq_len) valid positions
            
        Returns:
            Predictions (batch, seq_len, output_dim) for each position
        """
        batch_size, seq_len, _ = item_embeddings.shape
        device = item_embeddings.device
        
        # Project items
        items = self.item_proj(item_embeddings)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)
        items = items + pos_emb.unsqueeze(0)
        
        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Key padding mask
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        
        # Transform
        encoded = self.transformer(
            items,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )
        
        # Predict next item at each position
        predictions = self.predictor(encoded)
        predictions = F.normalize(predictions, p=2, dim=-1)
        
        return predictions
    
    def predict_next(
        self,
        history_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.
        
        Args:
            history_embeddings: User history (1, seq_len, item_dim)
            candidate_embeddings: Candidates (num_candidates, item_dim)
            
        Returns:
            Scores (num_candidates,)
        """
        # Get prediction for last position
        predictions = self.forward(history_embeddings)
        last_pred = predictions[0, -1, :]  # (output_dim,)
        
        # Score candidates
        candidate_proj = F.normalize(candidate_embeddings, p=2, dim=-1)
        scores = torch.matmul(candidate_proj, last_pred)
        
        return scores
    
    def get_session_embedding(
        self,
        item_embeddings: torch.Tensor,
        action_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get embedding for current session."""
        return self.session_encoder(item_embeddings, action_ids, time_deltas, mask)
    
    def sequence_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sequence prediction loss.
        
        Uses contrastive loss where target is the actual next item.
        
        Args:
            predictions: (batch, seq_len, dim)
            targets: (batch, seq_len, dim) actual next items
            mask: (batch, seq_len) valid prediction positions
            
        Returns:
            Loss value
        """
        batch_size, seq_len, dim = predictions.shape
        
        # Flatten for batch processing
        pred_flat = predictions.reshape(-1, dim)
        target_flat = targets.reshape(-1, dim)
        
        # Cosine similarity
        similarity = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        
        # Negative sampling within batch
        neg_indices = torch.randperm(pred_flat.size(0))
        neg_similarity = F.cosine_similarity(pred_flat, target_flat[neg_indices], dim=-1)
        
        # Margin ranking loss
        margin = 0.5
        loss = F.relu(margin - similarity + neg_similarity)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum()
        else:
            loss = loss.mean()
        
        return loss
