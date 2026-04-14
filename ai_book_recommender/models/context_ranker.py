# -*- coding: utf-8 -*-
"""
🕐 Context-Aware Ranker
========================

Ranking model that incorporates contextual signals:
- Time of day
- Day of week
- Session context
- Device type
- User activity level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """Encode contextual features into embeddings."""
    
    def __init__(
        self,
        hour_dim: int = 16,
        day_dim: int = 8,
        session_dim: int = 32,
        output_dim: int = 64
    ):
        super().__init__()
        
        # Time embeddings
        self.hour_embedding = nn.Embedding(24, hour_dim)
        self.day_embedding = nn.Embedding(7, day_dim)
        
        # Session features
        self.session_proj = nn.Linear(5, session_dim)  # duration, clicks, etc.
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hour_dim + day_dim + session_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        hour: torch.Tensor,
        day: torch.Tensor,
        session_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context.
        
        Args:
            hour: Hour of day (batch,) values 0-23
            day: Day of week (batch,) values 0-6
            session_features: Session stats (batch, 5)
            
        Returns:
            Context embedding (batch, output_dim)
        """
        hour_emb = self.hour_embedding(hour)
        day_emb = self.day_embedding(day)
        session_emb = self.session_proj(session_features)
        
        combined = torch.cat([hour_emb, day_emb, session_emb], dim=-1)
        return self.fusion(combined)


class ContextAwareRanker(nn.Module):
    """
    🕐 Context-Aware Ranking Model
    
    Modulates recommendation scores based on:
    1. Temporal context (time, day)
    2. Session context (activity level, duration)
    3. User state (engagement, recency)
    
    Architecture:
    - Base score from user-item matching
    - Context encoding
    - Score modulation via gating
    """
    
    def __init__(
        self,
        user_dim: int = 128,
        item_dim: int = 128,
        context_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.user_dim = user_dim
        self.item_dim = item_dim
        
        # Context encoder
        self.context_encoder = ContextEncoder(output_dim=context_dim)
        
        # User embedding with context
        self.user_context_fusion = nn.Sequential(
            nn.Linear(user_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, user_dim)
        )
        
        # Item embedding with context (for category/time-sensitive items)
        self.item_context_fusion = nn.Sequential(
            nn.Linear(item_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, item_dim)
        )
        
        # Score computation
        self.score_net = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Context-aware gate (modulates final score)
        self.context_gate = nn.Sequential(
            nn.Linear(context_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        logger.info(
            f"ContextAwareRanker: user_dim={user_dim}, item_dim={item_dim}, "
            f"context_dim={context_dim}"
        )
    
    def forward(
        self,
        user_emb: torch.Tensor,
        item_embs: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        session_features: torch.Tensor,
        base_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute context-aware scores.
        
        Args:
            user_emb: User embedding (batch, user_dim)
            item_embs: Item embeddings (batch, num_items, item_dim)
            hour: Hour of day (batch,)
            day: Day of week (batch,)
            session_features: Session features (batch, 5)
            base_scores: Optional base scores to modulate (batch, num_items)
            
        Returns:
            Context-aware scores (batch, num_items)
        """
        batch_size, num_items, _ = item_embs.shape
        device = user_emb.device
        
        # Encode context
        context = self.context_encoder(hour, day, session_features)  # (batch, context_dim)
        
        # Fuse user with context
        user_context = torch.cat([user_emb, context], dim=-1)
        user_adapted = self.user_context_fusion(user_context)  # (batch, user_dim)
        
        # Expand context for items
        context_expanded = context.unsqueeze(1).expand(-1, num_items, -1)
        
        # Fuse items with context
        item_context = torch.cat([item_embs, context_expanded], dim=-1)
        items_adapted = self.item_context_fusion(item_context)  # (batch, num_items, item_dim)
        
        # Expand user for scoring
        user_expanded = user_adapted.unsqueeze(1).expand(-1, num_items, -1)
        
        # Score
        combined = torch.cat([user_expanded, items_adapted], dim=-1)
        scores = self.score_net(combined).squeeze(-1)  # (batch, num_items)
        
        # Apply context gate if base scores provided
        if base_scores is not None:
            gate_input = torch.cat([
                context_expanded,
                base_scores.unsqueeze(-1)
            ], dim=-1)
            gate = self.context_gate(gate_input).squeeze(-1)  # (batch, num_items)
            
            # Combine base and context-aware scores
            scores = gate * scores + (1 - gate) * base_scores
        
        return scores
    
    @staticmethod
    def get_current_context(
        device: torch.device,
        session_duration: float = 0.0,
        session_clicks: int = 0,
        session_views: int = 0,
        is_returning: bool = False,
        activity_level: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Get current context tensors.
        
        Args:
            device: Target device
            session_duration: Duration in minutes
            session_clicks: Number of clicks this session
            session_views: Number of views this session
            is_returning: Whether returning user
            activity_level: User activity level (0-1)
            
        Returns:
            Dictionary of context tensors
        """
        now = datetime.now()
        
        return {
            "hour": torch.tensor([now.hour], device=device),
            "day": torch.tensor([now.weekday()], device=device),
            "session_features": torch.tensor([[
                min(session_duration / 60.0, 1.0),  # Normalize
                min(session_clicks / 20.0, 1.0),
                min(session_views / 50.0, 1.0),
                float(is_returning),
                activity_level
            ]], device=device)
        }
    
    def rank_with_context(
        self,
        user_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        base_scores: torch.Tensor,
        session_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Convenience method for ranking with current context.
        
        Args:
            user_emb: User embedding (dim,) or (1, dim)
            candidate_embs: Candidates (num_cand, dim) or (1, num_cand, dim)
            base_scores: Initial scores (num_cand,) or (1, num_cand)
            session_info: Optional session metadata
            
        Returns:
            Re-ranked scores
        """
        device = user_emb.device
        
        # Ensure proper dimensions
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        if candidate_embs.dim() == 2:
            candidate_embs = candidate_embs.unsqueeze(0)
        if base_scores.dim() == 1:
            base_scores = base_scores.unsqueeze(0)
        
        # Get context
        if session_info is None:
            session_info = {}
        
        context = self.get_current_context(
            device,
            session_duration=session_info.get("duration", 0),
            session_clicks=session_info.get("clicks", 0),
            session_views=session_info.get("views", 0),
            is_returning=session_info.get("is_returning", False),
            activity_level=session_info.get("activity_level", 0.5)
        )
        
        with torch.no_grad():
            scores = self.forward(
                user_emb,
                candidate_embs,
                context["hour"],
                context["day"],
                context["session_features"],
                base_scores
            )
        
        return scores.squeeze(0)
