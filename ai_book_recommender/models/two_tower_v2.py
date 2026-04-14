# -*- coding: utf-8 -*-
"""
🏛️ Two-Tower Model V2
=======================

Enhanced Two-Tower architecture for book recommendation with:
- Cross-attention for user history
- Temporal position encoding
- Interest evolution layer
- Multi-modal feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class TemporalPositionEncoder(nn.Module):
    """Encode temporal positions for sequence modeling."""
    
    def __init__(self, hidden_dim: int, max_len: int = 100):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        
        # Learnable time decay
        self.time_decay = nn.Parameter(torch.tensor(0.95))
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional embeddings with temporal decay.
        
        Args:
            seq_len: Sequence length
            device: Target device
            
        Returns:
            Position embeddings (seq_len, hidden_dim)
        """
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)
        
        # Apply exponential decay (more recent = higher weight)
        decay_weights = self.time_decay ** torch.arange(
            seq_len - 1, -1, -1, device=device
        ).float()
        
        return pos_emb * decay_weights.unsqueeze(-1)


class CrossAttentionLayer(nn.Module):
    """Cross-attention between user history and candidate items."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            query: Query tensor (batch, query_len, hidden)
            key_value: Key/Value tensor (batch, kv_len, hidden)
            mask: Optional attention mask
            
        Returns:
            Attended output (batch, query_len, hidden)
        """
        # Cross-attention
        attn_out, _ = self.attention(
            query, key_value, key_value,
            key_padding_mask=mask
        )
        query = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        output = self.norm2(query + ffn_out)
        
        return output


class InterestEvolutionLayer(nn.Module):
    """
    Model evolution of user interests over time.
    
    Uses GRU with attention to capture both short-term and long-term interests.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention for interest extraction
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Short-term vs long-term interest separation
        self.short_term_proj = nn.Linear(hidden_dim, hidden_dim)
        self.long_term_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        history: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract evolving interests from history.
        
        Args:
            history: Sequence of book embeddings (batch, seq, input_dim)
            mask: Optional sequence mask (batch, seq)
            
        Returns:
            Tuple of (fused_interest, short_term, long_term)
        """
        batch_size, seq_len, _ = history.shape
        
        # GRU encoding
        outputs, hidden = self.gru(history)  # outputs: (batch, seq, hidden)
        
        # Compute attention weights
        attn_scores = self.attention(outputs).squeeze(-1)  # (batch, seq)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq)
        
        # Long-term: weighted sum of all outputs
        long_term = (outputs * attn_weights.unsqueeze(-1)).sum(dim=1)
        long_term = self.long_term_proj(long_term)
        
        # Short-term: last hidden state
        short_term = hidden[-1]  # (batch, hidden)
        short_term = self.short_term_proj(short_term)
        
        # Fuse short-term and long-term
        combined = torch.cat([short_term, long_term], dim=1)
        gate = self.fusion_gate(combined)
        
        fused = gate * short_term + (1 - gate) * long_term
        
        return fused, short_term, long_term


class UserTowerV2(nn.Module):
    """
    🧑 Enhanced User Tower
    
    Encodes user features including:
    - User ID embedding (latent traits)
    - History encoding with attention
    - Explicit interest processing
    - Temporal interest evolution
    - Multi-interest representation
    
    Architecture:
    1. User ID → Embedding
    2. History → Temporal Pos Enc → GRU → Attention → Interest Evolution
    3. Explicit Interests → Projection
    4. Fusion → MLP → Output
    """
    
    def __init__(
        self,
        num_users: int = 100000,
        user_embedding_dim: int = 128,
        history_input_dim: int = 384,
        interest_input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_interests: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_interests = num_interests
        
        # 1. User ID Embedding
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # 2. History Processing
        self.history_proj = nn.Linear(history_input_dim, hidden_dim)
        self.temporal_encoder = TemporalPositionEncoder(hidden_dim)
        self.interest_evolution = InterestEvolutionLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 3. Explicit Interest Processing
        self.interest_proj = nn.Linear(interest_input_dim, hidden_dim)
        
        # 4. Multi-interest extraction (for diversity)
        self.interest_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            for _ in range(num_interests)
        ])
        
        # 5. Fusion Layer
        # Input: user_emb(128) + evolved_interest(256) + explicit(256) + short_term(256)
        fusion_dim = user_embedding_dim + hidden_dim * 3
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
        
        logger.info(
            f"UserTowerV2: users={num_users}, history_dim={history_input_dim}, "
            f"hidden={hidden_dim}, output={output_dim}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        history_vectors: torch.Tensor,
        interest_vectors: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: User IDs (batch,)
            history_vectors: History book embeddings (batch, seq_len, 384)
            interest_vectors: Explicit interest embeddings (batch, 384)
            history_mask: Optional mask for history (batch, seq_len)
            
        Returns:
            User embedding (batch, output_dim)
        """
        batch_size = user_ids.size(0)
        device = user_ids.device
        
        # 1. User embedding
        user_emb = self.user_embedding(user_ids)  # (batch, user_emb_dim)
        
        # 2. History processing
        seq_len = history_vectors.size(1)
        
        # Project history
        hist_proj = self.history_proj(history_vectors)  # (batch, seq, hidden)
        
        # Add temporal positions
        pos_emb = self.temporal_encoder(seq_len, device)
        hist_proj = hist_proj + pos_emb.unsqueeze(0)
        
        # Interest evolution
        evolved_int, short_term, long_term = self.interest_evolution(
            hist_proj, history_mask
        )
        
        # 3. Explicit interests
        int_emb = F.relu(self.interest_proj(interest_vectors))  # (batch, hidden)
        
        # 4. Fusion
        combined = torch.cat([
            user_emb,
            evolved_int,
            int_emb,
            short_term
        ], dim=1)
        
        output = self.fusion(combined)
        output = self.output_norm(output)
        
        # Normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output
    
    def get_multi_interests(
        self,
        history_vectors: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Extract multiple interest vectors for diversity.
        
        Args:
            history_vectors: User history (batch, seq, dim)
            
        Returns:
            List of interest vectors, each (batch, hidden)
        """
        hist_proj = self.history_proj(history_vectors)
        pooled = hist_proj.mean(dim=1)  # Simple pooling
        
        return [head(pooled) for head in self.interest_heads]


class ItemTowerV2(nn.Module):
    """
    📚 Enhanced Item Tower
    
    Encodes book features including:
    - Text embedding (from transformer)
    - Metadata features (category, author, year)
    - Popularity features
    
    Architecture:
    1. Text → Projection → LayerNorm
    2. Metadata → Embedding → Concat
    3. Popularity → MLP
    4. Fusion → Output
    """
    
    def __init__(
        self,
        text_input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_categories: int = 100,
        category_embedding_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 1. Text processing
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Category embedding
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        
        # 3. Metadata processing
        # Numerical features: year, page_count, avg_rating, popularity
        numerical_dim = 8
        self.numerical_proj = nn.Sequential(
            nn.Linear(numerical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 4. Fusion
        fusion_dim = hidden_dim + category_embedding_dim + 32
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
        
        logger.info(
            f"ItemTowerV2: text_dim={text_input_dim}, hidden={hidden_dim}, "
            f"output={output_dim}"
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
        text_vectors: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_vectors: Pre-computed text embeddings (batch, text_dim)
            category_ids: Category IDs (batch,) - optional
            numerical_features: Numerical features (batch, 8) - optional
            
        Returns:
            Item embedding (batch, output_dim)
        """
        batch_size = text_vectors.size(0)
        device = text_vectors.device
        
        # 1. Text processing
        text_emb = self.text_proj(text_vectors)  # (batch, hidden)
        
        # 2. Category embedding
        if category_ids is not None:
            cat_emb = self.category_embedding(category_ids)
        else:
            cat_emb = torch.zeros(
                batch_size, self.category_embedding.embedding_dim,
                device=device
            )
        
        # 3. Numerical features
        if numerical_features is not None:
            num_emb = self.numerical_proj(numerical_features)
        else:
            num_emb = torch.zeros(batch_size, 32, device=device)
        
        # 4. Fusion
        combined = torch.cat([text_emb, cat_emb, num_emb], dim=1)
        output = self.fusion(combined)
        output = self.output_norm(output)
        
        # Normalize
        output = F.normalize(output, p=2, dim=1)
        
        return output


class TwoTowerV2(nn.Module):
    """
    🏛️ Two-Tower Model V2
    
    Production-grade Two-Tower architecture combining:
    - Enhanced User Tower with interest evolution
    - Enhanced Item Tower with multi-modal features
    - Efficient batch scoring
    - Temperature-scaled softmax
    
    Training:
    - Contrastive loss (in-batch negatives)
    - Hard negative mining
    - Temperature annealing
    """
    
    def __init__(
        self,
        num_users: int = 100000,
        user_embedding_dim: int = 128,
        history_input_dim: int = 384,
        text_input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_categories: int = 100,
        temperature: float = 0.07,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.temperature = temperature
        self.output_dim = output_dim
        
        # Towers
        self.user_tower = UserTowerV2(
            num_users=num_users,
            user_embedding_dim=user_embedding_dim,
            history_input_dim=history_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.item_tower = ItemTowerV2(
            text_input_dim=text_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_categories=num_categories,
            dropout=dropout
        )
        
        logger.info(f"TwoTowerV2 initialized: output_dim={output_dim}, temp={temperature}")
    
    def forward(
        self,
        user_ids: torch.Tensor,
        history_vectors: torch.Tensor,
        interest_vectors: torch.Tensor,
        book_vectors: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        category_ids: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both embeddings.
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        user_emb = self.user_tower(
            user_ids, history_vectors, interest_vectors, history_mask
        )
        
        item_emb = self.item_tower(
            book_vectors, category_ids, numerical_features
        )
        
        return user_emb, item_emb
    
    def compute_scores(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores.
        
        Args:
            user_emb: User embeddings (batch, dim)
            item_emb: Item embeddings (num_items, dim)
            
        Returns:
            Scores (batch, num_items)
        """
        # Cosine similarity (vectors are normalized)
        scores = torch.matmul(user_emb, item_emb.T)
        
        # Temperature scaling
        scores = scores / self.temperature
        
        return scores
    
    def contrastive_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss with in-batch negatives.
        
        Args:
            user_emb: User embeddings (batch, dim)
            pos_item_emb: Positive item embeddings (batch, dim)
            neg_item_emb: Optional hard negatives (batch, num_neg, dim)
            
        Returns:
            Loss value
        """
        batch_size = user_emb.size(0)
        device = user_emb.device
        
        # Compute all pairwise scores (in-batch negatives)
        scores = self.compute_scores(user_emb, pos_item_emb)  # (batch, batch)
        
        # Add hard negatives if provided
        if neg_item_emb is not None:
            num_neg = neg_item_emb.size(1)
            neg_scores = torch.bmm(
                user_emb.unsqueeze(1),
                neg_item_emb.transpose(1, 2)
            ).squeeze(1)  # (batch, num_neg)
            neg_scores = neg_scores / self.temperature
            
            # Concatenate
            scores = torch.cat([scores, neg_scores], dim=1)
        
        # Labels: positive is on diagonal
        labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(scores, labels)
        
        return loss
    
    def predict(
        self,
        user_ids: torch.Tensor,
        history_vectors: torch.Tensor,
        interest_vectors: torch.Tensor,
        candidate_book_vectors: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.
        
        Args:
            user_ids: User IDs (batch,) or (1,)
            history_vectors: History (batch, seq, dim) or (1, seq, dim)
            interest_vectors: Interests (batch, dim) or (1, dim)
            candidate_book_vectors: Candidates (num_candidates, dim)
            
        Returns:
            Scores (batch, num_candidates)
        """
        with torch.no_grad():
            user_emb = self.user_tower(
                user_ids, history_vectors, interest_vectors, history_mask
            )
            
            item_emb = self.item_tower(candidate_book_vectors)
            
            scores = self.compute_scores(user_emb, item_emb)
        
        return scores
