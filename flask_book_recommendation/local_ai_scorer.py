# -*- coding: utf-8 -*-
"""
🤖 Local AI Scorer
==================

Uses the trained recommendation model locally without external API.
Integrates with the existing recommendation system.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleScoringModel(nn.Module):
    """نموذج بسيط لحساب التوافق بين المستخدم والكتاب"""
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, user_embedding, item_embedding):
        combined = torch.cat([user_embedding, item_embedding], dim=-1)
        return self.scorer(combined).squeeze(-1)


class LocalAIScorer:
    """
    🤖 Local AI Scorer
    
    Uses pre-trained model to score book recommendations locally.
    Provides faster inference without network overhead.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        embedding_dim: int = 768
    ):
        """
        Initialize scorer.
        
        Args:
            checkpoint_path: Path to model checkpoint, or None to auto-detect
            embedding_dim: Dimension of embeddings
        """
        self.model = None
        self.embedding_dim = embedding_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
        
        # Auto-detect checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"LocalAIScorer: device={self.device}")
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint in default locations."""
        possible_paths = [
            "instance/checkpoints/simple_recommender/model_best.pt",
            "instance/checkpoints/simple_recommender/model_final.pt",
            "checkpoints/simple_recommender/model_best.pt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found checkpoint: {path}")
                return path
        
        return None
    
    def initialize(self):
        """Load model from checkpoint."""
        if self._initialized:
            return
        
        if self.checkpoint_path is None or not os.path.exists(self.checkpoint_path):
            logger.warning("No checkpoint found, using random model")
            self.model = SimpleScoringModel(self.embedding_dim)
            self.model.to(self.device)
            self._initialized = True
            return
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Detect embedding dim from saved model
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            first_layer_key = [k for k in state_dict.keys() if "weight" in k][0]
            first_weight = state_dict[first_layer_key]
            detected_dim = first_weight.shape[1] // 2
            
            self.embedding_dim = detected_dim
            self.model = SimpleScoringModel(detected_dim)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model from {self.checkpoint_path}, dim={detected_dim}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = SimpleScoringModel(self.embedding_dim)
            self.model.to(self.device)
            self._initialized = True
    
    @torch.no_grad()
    def score(
        self,
        user_embedding: np.ndarray,
        item_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Score items for a user.
        
        Args:
            user_embedding: User embedding vector
            item_embeddings: List of item embedding vectors
            
        Returns:
            List of scores for each item
        """
        if not self._initialized:
            self.initialize()
        
        if len(item_embeddings) == 0:
            return []
        
        # Prepare tensors
        user_tensor = torch.tensor(user_embedding, dtype=torch.float32).to(self.device)
        user_batch = user_tensor.unsqueeze(0).expand(len(item_embeddings), -1)
        
        items_tensor = torch.tensor(
            np.array(item_embeddings), dtype=torch.float32
        ).to(self.device)
        
        # Score
        scores = self.model(user_batch, items_tensor)
        
        return torch.sigmoid(scores).cpu().numpy().tolist()
    
    def rank_items(
        self,
        user_embedding: np.ndarray,
        item_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rank items by predicted score.
        
        Args:
            user_embedding: User embedding
            item_embeddings: Dict of item_id -> embedding
            top_k: Number of top items to return
            
        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        if not item_embeddings:
            return []
        
        item_ids = list(item_embeddings.keys())
        embeddings = [item_embeddings[i] for i in item_ids]
        
        scores = self.score(user_embedding, embeddings)
        
        # Sort by score
        ranked = sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def get_health(self) -> Dict:
        """Get scorer health status."""
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "model_loaded": self.model is not None,
            "checkpoint": self.checkpoint_path,
            "embedding_dim": self.embedding_dim,
            "device": self.device
        }


# Singleton instance
_local_scorer = None

def get_local_scorer() -> LocalAIScorer:
    """Get or create the local scorer singleton."""
    global _local_scorer
    if _local_scorer is None:
        _local_scorer = LocalAIScorer()
    return _local_scorer
