# -*- coding: utf-8 -*-
"""
🔢 Collaborative Filtering Models
==================================

Matrix factorization and implicit feedback models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class MatrixFactorization(nn.Module):
    """
    🔢 Neural Matrix Factorization
    
    Classic matrix factorization enhanced with neural network layers.
    
    Components:
    - User/Item embeddings
    - Bias terms
    - Optional MLP for non-linear interactions
    
    Suitable for explicit feedback (ratings).
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        use_bias: bool = True,
        use_mlp: bool = True,
        mlp_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        self.use_mlp = use_mlp
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        
        # MLP for non-linear interactions
        if use_mlp:
            layers = []
            input_dim = embedding_dim * 2
            
            for dim in mlp_dims:
                layers.extend([
                    nn.Linear(input_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = dim
            
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
        
        logger.info(
            f"MatrixFactorization: users={num_users}, items={num_items}, "
            f"dim={embedding_dim}, bias={use_bias}, mlp={use_mlp}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict ratings.
        
        Args:
            user_ids: User indices (batch,)
            item_ids: Item indices (batch,)
            
        Returns:
            Predicted ratings (batch,)
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        
        # Dot product
        mf_output = (u_emb * i_emb).sum(dim=1)
        
        # MLP
        if self.use_mlp:
            mlp_input = torch.cat([u_emb, i_emb], dim=1)
            mlp_output = self.mlp(mlp_input).squeeze(1)
            output = mf_output + mlp_output
        else:
            output = mf_output
        
        # Bias
        if self.use_bias:
            u_bias = self.user_bias(user_ids).squeeze(1)
            i_bias = self.item_bias(item_ids).squeeze(1)
            output = output + u_bias + i_bias + self.global_bias
        
        return output
    
    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ratings: torch.Tensor,
        reg_lambda: float = 1e-5
    ) -> torch.Tensor:
        """
        Compute MSE loss with L2 regularization.
        """
        predictions = self.forward(user_ids, item_ids)
        mse_loss = F.mse_loss(predictions, ratings)
        
        # L2 regularization
        reg_loss = reg_lambda * (
            self.user_embedding.weight.norm(2) +
            self.item_embedding.weight.norm(2)
        )
        
        return mse_loss + reg_loss
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user.
        """
        self.eval()
        device = self.user_embedding.weight.device
        
        with torch.no_grad():
            u_emb = self.user_embedding(torch.tensor([user_id], device=device))
            
            # Score all items
            all_items = torch.arange(self.num_items, device=device)
            i_emb = self.item_embedding(all_items)
            
            scores = (u_emb * i_emb).sum(dim=1)
            
            if self.use_bias:
                u_bias = self.user_bias(torch.tensor([user_id], device=device))
                i_bias = self.item_bias(all_items).squeeze(1)
                scores = scores + u_bias.squeeze() + i_bias + self.global_bias
        
        scores = scores.cpu().numpy()
        
        # Exclude items
        if exclude_items:
            for idx in exclude_items:
                if 0 <= idx < len(scores):
                    scores[idx] = float("-inf")
        
        # Top-k
        top_indices = np.argsort(-scores)[:k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class ALSModel:
    """
    ⚡ Alternating Least Squares
    
    Efficient ALS implementation for implicit feedback.
    Uses pure NumPy for compatibility and speed on CPU.
    
    Formula:
        Loss = Σ c_ui (p_ui - x_u^T y_i)^2 + λ(||x_u||^2 + ||y_i||^2)
    
    where:
        c_ui = 1 + α * r_ui (confidence)
        p_ui = 1 if r_ui > 0 else 0 (preference)
    """
    
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
        # Mappings
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}
        
        logger.info(
            f"ALSModel: factors={factors}, reg={regularization}, "
            f"alpha={alpha}, iters={iterations}"
        )
    
    def fit(
        self,
        interactions: List[Tuple[int, str, float]],
        show_progress: bool = True
    ) -> None:
        """
        Fit ALS model on interactions.
        
        Args:
            interactions: List of (user_id, item_id, value) tuples
            show_progress: Whether to show progress
        """
        from scipy.sparse import csr_matrix
        
        # Build mappings
        users = sorted(set(uid for uid, _, _ in interactions))
        items = sorted(set(iid for _, iid, _ in interactions))
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(items)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        num_users = len(users)
        num_items = len(items)
        
        # Build sparse matrix
        row_ind = [self.user_id_map[uid] for uid, _, _ in interactions]
        col_ind = [self.item_id_map[iid] for _, iid, _ in interactions]
        data = [val for _, _, val in interactions]
        
        R = csr_matrix((data, (row_ind, col_ind)), shape=(num_users, num_items))
        
        # Confidence matrix: C = 1 + alpha * R
        C = 1 + self.alpha * R
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (num_users, self.factors))
        self.item_factors = np.random.normal(0, 0.01, (num_items, self.factors))
        
        # ALS iterations
        for iteration in range(self.iterations):
            # Update user factors
            self._als_step(C, self.user_factors, self.item_factors, True)
            
            # Update item factors
            self._als_step(C.T.tocsr(), self.item_factors, self.user_factors, False)
            
            if show_progress:
                logger.info(f"ALS iteration {iteration + 1}/{self.iterations}")
        
        logger.info(
            f"ALS training complete: {num_users} users, {num_items} items"
        )
    
    def _als_step(
        self,
        C: "csr_matrix",
        X: np.ndarray,
        Y: np.ndarray,
        is_user: bool
    ) -> None:
        """
        Single ALS step.
        
        Solves: x_u = (Y^T C_u Y + λI)^-1 Y^T C_u p_u
        """
        YtY = Y.T @ Y
        lambda_eye = self.regularization * np.eye(self.factors)
        
        for i in range(X.shape[0]):
            # Get confidence values for this user/item
            row = C.getrow(i)
            indices = row.indices
            confidence = row.data
            
            if len(indices) == 0:
                continue
            
            # C_u diagonal for this user
            Y_u = Y[indices]  # Items user interacted with
            
            # A = Y^T C_u Y + λI
            CuI = np.diag(confidence - 1)  # C_u - I
            A = YtY + Y_u.T @ CuI @ Y_u + lambda_eye
            
            # b = Y^T C_u p_u (p_u = 1 for all observed)
            b = Y_u.T @ confidence
            
            # Solve
            X[i] = np.linalg.solve(A, b)
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user.
        """
        if user_id not in self.user_id_map:
            return []
        
        if self.user_factors is None or self.item_factors is None:
            return []
        
        u_idx = self.user_id_map[user_id]
        u_vec = self.user_factors[u_idx]
        
        # Compute scores
        scores = self.item_factors @ u_vec
        
        # Exclude items
        exclude_indices = set()
        if exclude_items:
            for iid in exclude_items:
                if iid in self.item_id_map:
                    exclude_indices.add(self.item_id_map[iid])
        
        # Top-k
        results = []
        sorted_indices = np.argsort(-scores)
        
        for idx in sorted_indices:
            if idx in exclude_indices:
                continue
            
            item_id = self.reverse_item_map.get(idx)
            if item_id:
                results.append((item_id, float(scores[idx])))
            
            if len(results) >= k:
                break
        
        return results
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get embedding for a user."""
        if user_id not in self.user_id_map or self.user_factors is None:
            return None
        return self.user_factors[self.user_id_map[user_id]]
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for an item."""
        if item_id not in self.item_id_map or self.item_factors is None:
            return None
        return self.item_factors[self.item_id_map[item_id]]
    
    def save(self, path: str) -> None:
        """Save model."""
        np.savez(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_id_map=np.array(list(self.user_id_map.items()), dtype=object),
            item_id_map=np.array(list(self.item_id_map.items()), dtype=object),
            config=np.array([
                self.factors, self.regularization, self.alpha, self.iterations
            ])
        )
        logger.info(f"ALS model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        data = np.load(path, allow_pickle=True)
        
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        
        self.user_id_map = dict(data["user_id_map"])
        self.item_id_map = dict(data["item_id_map"])
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        config = data["config"]
        self.factors = int(config[0])
        self.regularization = float(config[1])
        self.alpha = float(config[2])
        self.iterations = int(config[3])
        
        logger.info(f"ALS model loaded from {path}")
