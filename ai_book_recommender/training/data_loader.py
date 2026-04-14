# -*- coding: utf-8 -*-
"""
📚 Training Data Loaders
=========================

Data loading and preprocessing for recommendation training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InteractionSample:
    """Single user-item interaction sample."""
    
    user_id: int
    item_id: str
    label: float  # 1.0 for positive, 0.0 for negative
    
    # Optional features
    user_embedding: Optional[np.ndarray] = None
    item_embedding: Optional[np.ndarray] = None
    context: Optional[Dict] = None
    timestamp: Optional[float] = None


class RecommendationDataset(Dataset):
    """
    📚 Recommendation Dataset
    
    Supports:
    - Point-wise training (user, item, label)
    - Pair-wise training (user, pos_item, neg_item)
    - List-wise training (user, item_list, label_list)
    """
    
    def __init__(
        self,
        interactions: List[InteractionSample],
        user_embeddings: Optional[Dict[int, np.ndarray]] = None,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        negative_sampling: bool = True,
        negative_ratio: int = 4,
        item_pool: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            interactions: List of interaction samples
            user_embeddings: Pre-computed user embeddings
            item_embeddings: Pre-computed item embeddings
            negative_sampling: Whether to sample negatives
            negative_ratio: Number of negatives per positive
            item_pool: Pool of items for negative sampling
        """
        self.interactions = interactions
        self.user_embeddings = user_embeddings or {}
        self.item_embeddings = item_embeddings or {}
        self.negative_sampling = negative_sampling
        self.negative_ratio = negative_ratio
        self.item_pool = item_pool or list(set(i.item_id for i in interactions))
        
        # Build positive set per user for negative sampling
        self._user_positives: Dict[int, set] = {}
        for inter in interactions:
            if inter.label > 0.5:
                if inter.user_id not in self._user_positives:
                    self._user_positives[inter.user_id] = set()
                self._user_positives[inter.user_id].add(inter.item_id)
        
        logger.info(
            f"RecommendationDataset: {len(interactions)} interactions, "
            f"{len(self.item_pool)} items"
        )
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample = self.interactions[idx]
        
        # Get embeddings
        user_emb = self.user_embeddings.get(sample.user_id)
        item_emb = self.item_embeddings.get(sample.item_id)
        
        result = {
            "user_id": torch.tensor(sample.user_id, dtype=torch.long),
            "item_id": sample.item_id,
            "label": torch.tensor(sample.label, dtype=torch.float32),
        }
        
        if user_emb is not None:
            result["user_embedding"] = torch.tensor(user_emb, dtype=torch.float32)
        
        if item_emb is not None:
            result["item_embedding"] = torch.tensor(item_emb, dtype=torch.float32)
        
        # Add negative samples if needed
        if self.negative_sampling and sample.label > 0.5:
            negatives = self._sample_negatives(sample.user_id)
            result["negative_items"] = negatives
        
        return result
    
    def _sample_negatives(self, user_id: int) -> List[str]:
        """Sample negative items for a user."""
        positives = self._user_positives.get(user_id, set())
        negatives = []
        
        attempts = 0
        while len(negatives) < self.negative_ratio and attempts < 100:
            candidate = np.random.choice(self.item_pool)
            if candidate not in positives:
                negatives.append(candidate)
            attempts += 1
        
        return negatives


class SequenceDataset(Dataset):
    """
    🔗 Sequence Dataset for Sequential Recommendation
    
    Creates sequences from user interaction history.
    """
    
    def __init__(
        self,
        user_sequences: Dict[int, List[str]],
        item_embeddings: Dict[str, np.ndarray],
        max_seq_len: int = 50,
        prediction_len: int = 1
    ):
        """
        Initialize sequence dataset.
        
        Args:
            user_sequences: User ID -> list of item IDs (chronological)
            item_embeddings: Item embeddings
            max_seq_len: Maximum sequence length
            prediction_len: How many items to predict
        """
        self.item_embeddings = item_embeddings
        self.max_seq_len = max_seq_len
        self.prediction_len = prediction_len
        
        # Create samples (user_id, input_seq, target_seq)
        self.samples = []
        
        for user_id, seq in user_sequences.items():
            if len(seq) > max_seq_len + prediction_len:
                # Sliding window
                for i in range(len(seq) - max_seq_len - prediction_len + 1):
                    input_seq = seq[i:i + max_seq_len]
                    target_seq = seq[i + max_seq_len:i + max_seq_len + prediction_len]
                    self.samples.append((user_id, input_seq, target_seq))
            elif len(seq) > prediction_len:
                # Use what we have
                input_seq = seq[:-prediction_len]
                target_seq = seq[-prediction_len:]
                self.samples.append((user_id, input_seq, target_seq))
        
        logger.info(f"SequenceDataset: {len(self.samples)} samples from {len(user_sequences)} users")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, input_seq, target_seq = self.samples[idx]
        
        # Get embeddings
        input_embs = [
            self.item_embeddings.get(item_id, np.zeros(128))
            for item_id in input_seq
        ]
        target_embs = [
            self.item_embeddings.get(item_id, np.zeros(128))
            for item_id in target_seq
        ]
        
        # Pad if needed
        seq_len = len(input_embs)
        if seq_len < self.max_seq_len:
            padding = [np.zeros_like(input_embs[0])] * (self.max_seq_len - seq_len)
            input_embs = padding + input_embs
        
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "input_embeddings": torch.tensor(np.array(input_embs), dtype=torch.float32),
            "target_embeddings": torch.tensor(np.array(target_embs), dtype=torch.float32),
            "mask": torch.tensor(
                [0] * (self.max_seq_len - seq_len) + [1] * seq_len,
                dtype=torch.bool
            ),
            "target_ids": target_seq,
        }


def create_data_loaders(
    train_data: List[InteractionSample],
    val_data: Optional[List[InteractionSample]] = None,
    user_embeddings: Optional[Dict] = None,
    item_embeddings: Optional[Dict] = None,
    batch_size: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation data loaders.
    
    Args:
        train_data: Training interaction samples
        val_data: Validation samples (optional)
        user_embeddings: User embedding dict
        item_embeddings: Item embedding dict
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = RecommendationDataset(
        interactions=train_data,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        negative_sampling=False  # Disable for simple training
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_data:
        val_dataset = RecommendationDataset(
            interactions=val_data,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            negative_sampling=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return train_loader, val_loader
