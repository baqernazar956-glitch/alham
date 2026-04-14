# -*- coding: utf-8 -*-
"""
🕸️ Graph-Based Recommender
===========================

Graph Neural Network for recommendation using:
- LightGCN (lightweight graph convolution)
- User-Item bipartite graph
- Neighbor aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class LightGCNConv(nn.Module):
    """
    Light Graph Convolution layer (LightGCN).
    
    Simplified graph convolution without feature transformation and nonlinearity.
    Only performs neighborhood aggregation.
    
    Formula:
        e_u^(k+1) = Σ (1/√|N_u||N_i|) * e_i^(k)  for all neighbors i of u
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node embeddings (num_nodes, dim)
            edge_index: Edge indices (2, num_edges) - [source, target]
            edge_weight: Optional edge weights (num_edges,)
            
        Returns:
            Updated embeddings (num_nodes, dim)
        """
        row, col = edge_index
        num_nodes = x.size(0)
        
        # Compute normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalized weights
        if edge_weight is None:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight
        
        # Aggregate
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row] * norm.unsqueeze(1))
        
        return out


class LightGCN(nn.Module):
    """
    🕸️ LightGCN Model
    
    Implements LightGCN for collaborative filtering.
    
    Key features:
    - No feature transformation
    - No activation function
    - Only neighborhood aggregation
    - Layer combination with equal weights
    
    Reference:
    He et al. "LightGCN: Simplifying and Powering Graph Convolution Network 
    for Recommendation" SIGIR 2020
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
        logger.info(
            f"LightGCN: users={num_users}, items={num_items}, "
            f"dim={embedding_dim}, layers={num_layers}"
        )
    
    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def get_ego_embeddings(self) -> torch.Tensor:
        """Get concatenated user and item embeddings."""
        return torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all GCN layers.
        
        Args:
            edge_index: Edge indices (2, num_edges)
            edge_weight: Optional edge weights
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Initial embeddings
        all_emb = self.get_ego_embeddings()
        emb_list = [all_emb]
        
        # Propagate through layers
        for conv in self.convs:
            all_emb = conv(all_emb, edge_index, edge_weight)
            all_emb = self.dropout(all_emb)
            emb_list.append(all_emb)
        
        # Layer combination (mean over all layers)
        final_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        
        # Split user and item embeddings
        user_emb, item_emb = torch.split(
            final_emb, [self.num_users, self.num_items]
        )
        
        return user_emb, item_emb
    
    def compute_scores(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scores for user-item pairs.
        
        Args:
            user_ids: User indices (batch,)
            item_ids: Item indices (batch,) or (batch, num_items)
            user_emb: All user embeddings (num_users, dim)
            item_emb: All item embeddings (num_items, dim)
            
        Returns:
            Scores (batch,) or (batch, num_items)
        """
        u_emb = user_emb[user_ids]  # (batch, dim)
        
        if item_ids.dim() == 1:
            i_emb = item_emb[item_ids]  # (batch, dim)
            scores = (u_emb * i_emb).sum(dim=1)
        else:
            # Multiple items per user
            i_emb = item_emb[item_ids]  # (batch, num_items, dim)
            scores = torch.bmm(i_emb, u_emb.unsqueeze(2)).squeeze(2)
        
        return scores
    
    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BPR (Bayesian Personalized Ranking) loss.
        
        Args:
            user_ids: User indices (batch,)
            pos_item_ids: Positive item indices (batch,)
            neg_item_ids: Negative item indices (batch,)
            user_emb: User embeddings
            item_emb: Item embeddings
            
        Returns:
            BPR loss value
        """
        u_emb = user_emb[user_ids]
        pos_emb = item_emb[pos_item_ids]
        neg_emb = item_emb[neg_item_ids]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization on embeddings
        reg_loss = (
            u_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / (2 * len(user_ids))
        
        return loss + 1e-5 * reg_loss


class GraphRecommender(nn.Module):
    """
    📊 Graph-Based Recommendation Engine
    
    Higher-level wrapper for graph-based recommendations.
    
    Features:
    - Build graph from interactions
    - Train LightGCN model
    - Efficient inference
    - Score caching
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Model (created when graph is built)
        self.model: Optional[LightGCN] = None
        
        # Graph data
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        
        # Mappings
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}
        
        # Cached embeddings
        self._user_emb: Optional[torch.Tensor] = None
        self._item_emb: Optional[torch.Tensor] = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_graph(
        self,
        interactions: List[Tuple[int, str, float]]
    ) -> None:
        """
        Build user-item graph from interactions.
        
        Args:
            interactions: List of (user_id, item_id, weight) tuples
        """
        if not interactions:
            logger.warning("No interactions provided for graph building")
            return
        
        # Build mappings
        users = set()
        items = set()
        
        for user_id, item_id, _ in interactions:
            users.add(user_id)
            items.add(item_id)
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(sorted(users))}
        self.item_id_map = {iid: idx for idx, iid in enumerate(sorted(items))}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        
        # Build edge index
        edges = []
        weights = []
        
        for user_id, item_id, weight in interactions:
            u_idx = self.user_id_map[user_id]
            i_idx = self.item_id_map[item_id] + num_users  # Offset item indices
            
            # Bidirectional edges
            edges.append([u_idx, i_idx])
            edges.append([i_idx, u_idx])
            weights.extend([weight, weight])
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).T.to(self.device)
        self.edge_weight = torch.tensor(weights, dtype=torch.float).to(self.device)
        
        # Create model
        self.model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Clear cache
        self._user_emb = None
        self._item_emb = None
        
        logger.info(
            f"Graph built: {num_users} users, {num_items} items, "
            f"{len(interactions)} interactions"
        )
    
    def train_step(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            user_ids: User indices in mapped space
            pos_item_ids: Positive item indices
            neg_item_ids: Negative item indices
            
        Returns:
            Loss value
        """
        if self.model is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        self.model.train()
        
        # Forward pass
        user_emb, item_emb = self.model(self.edge_index, self.edge_weight)
        
        # BPR loss
        loss = self.model.bpr_loss(
            user_ids, pos_item_ids, neg_item_ids,
            user_emb, item_emb
        )
        
        # Clear cache
        self._user_emb = None
        self._item_emb = None
        
        return loss.item()
    
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached embeddings or compute them."""
        if self._user_emb is None or self._item_emb is None:
            if self.model is None:
                raise ValueError("Graph not built. Call build_graph() first.")
            
            self.model.eval()
            with torch.no_grad():
                self._user_emb, self._item_emb = self.model(
                    self.edge_index, self.edge_weight
                )
        
        return self._user_emb, self._item_emb
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: Original user ID
            k: Number of recommendations
            exclude_items: Items to exclude (already interacted)
            
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not in graph")
            return []
        
        user_idx = self.user_id_map[user_id]
        user_emb, item_emb = self.get_embeddings()
        
        # Compute scores for all items
        u_emb = user_emb[user_idx]  # (dim,)
        scores = torch.matmul(item_emb, u_emb)  # (num_items,)
        
        # Exclude items
        exclude_indices = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_id_map:
                    exclude_indices.add(self.item_id_map[iid])
        
        # Get top-k
        scores_np = scores.cpu().numpy()
        
        results = []
        sorted_indices = np.argsort(-scores_np)
        
        for idx in sorted_indices:
            if idx in exclude_indices:
                continue
            
            item_id = self.reverse_item_map.get(idx)
            if item_id:
                results.append((item_id, float(scores_np[idx])))
            
            if len(results) >= k:
                break
        
        return results
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific item."""
        if item_id not in self.item_id_map:
            return None
        
        _, item_emb = self.get_embeddings()
        idx = self.item_id_map[item_id]
        
        return item_emb[idx].cpu().numpy()
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific user."""
        if user_id not in self.user_id_map:
            return None
        
        user_emb, _ = self.get_embeddings()
        idx = self.user_id_map[user_id]
        
        return user_emb[idx].cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model and mappings."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        torch.save({
            "model_state": self.model.state_dict(),
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
            "config": {
                "embedding_dim": self.embedding_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout
            }
        }, path)
        
        logger.info(f"Graph recommender saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model and mappings."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.user_id_map = checkpoint["user_id_map"]
        self.item_id_map = checkpoint["item_id_map"]
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        self.edge_index = checkpoint["edge_index"].to(self.device)
        self.edge_weight = checkpoint["edge_weight"].to(self.device)
        
        config = checkpoint["config"]
        self.embedding_dim = config["embedding_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        
        # Recreate model
        self.model = LightGCN(
            num_users=len(self.user_id_map),
            num_items=len(self.item_id_map),
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Clear cache
        self._user_emb = None
        self._item_emb = None
        
        logger.info(f"Graph recommender loaded from {path}")
