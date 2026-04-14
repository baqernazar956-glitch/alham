# -*- coding: utf-8 -*-
"""
🔤 Transformer Text Encoder
============================

Advanced transformer-based text encoding for books.
Uses multi-head attention with various pooling strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    🔤 Transformer-based Text Encoder
    
    Encodes book title + description into dense embeddings.
    Supports multiple pooling strategies:
    - cls: Use [CLS] token representation
    - mean: Mean pooling over all tokens
    - max: Max pooling over all tokens
    - attention: Attention-weighted pooling
    
    Architecture:
    1. Input Projection (if needed)
    2. Positional Encoding
    3. Transformer Encoder Layers
    4. Pooling Layer
    5. Output Projection
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: Literal["cls", "mean", "max", "attention"] = "mean",
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling (if used)
        if pooling == "attention":
            self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info(
            f"TransformerEncoder initialized: {input_dim}→{hidden_dim}→{output_dim}, "
            f"heads={num_heads}, layers={num_layers}, pooling={pooling}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask (batch, seq_len)
            
        Returns:
            Encoded representation of shape (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq, hidden)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, hidden)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq, hidden)
        
        # Apply transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        # Pooling
        pooled = self._pool(x, mask)  # (batch, hidden)
        
        # Output projection
        output = self.output_proj(pooled)
        output = self.layer_norm(output)
        
        # Normalize for cosine similarity
        output = F.normalize(output, p=2, dim=1)
        
        return output
    
    def _pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply pooling strategy.
        
        Args:
            x: Transformer output (batch, seq, hidden)
            mask: Optional mask (batch, seq)
            
        Returns:
            Pooled representation (batch, hidden)
        """
        if self.pooling == "cls":
            return x[:, 0, :]
        
        elif self.pooling == "mean":
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                x = x * mask
                return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return x.mean(dim=1)
        
        elif self.pooling == "max":
            if mask is not None:
                mask = mask.unsqueeze(-1)
                x = x.masked_fill(~mask, float("-inf"))
            return x.max(dim=1)[0]
        
        elif self.pooling == "attention":
            # Compute attention weights
            attn_scores = self.attention_weights(x).squeeze(-1)  # (batch, seq)
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq)
            
            # Weighted sum
            return (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def encode_texts(
        self,
        texts: List[str],
        sentence_transformer: Optional[object] = None
    ) -> torch.Tensor:
        """
        Encode list of texts using sentence transformer + this encoder.
        
        Args:
            texts: List of text strings
            sentence_transformer: SentenceTransformer model instance
            
        Returns:
            Encoded representations (num_texts, output_dim)
        """
        if sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.error("SentenceTransformer not available")
                return torch.zeros(len(texts), self.output_dim)
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = sentence_transformer.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Add sequence dimension (treat as single-token sequences)
            if base_embeddings.dim() == 2:
                base_embeddings = base_embeddings.unsqueeze(1)
            
            # Pass through transformer
            return self.forward(base_embeddings)


class BookTextEncoder(nn.Module):
    """
    📚 Specialized encoder for book text (title + description).
    
    Encodes title and description separately, then fuses them.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Separate encoders for title and description
        self.title_encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=1,
            dropout=dropout,
            pooling="attention"
        )
        
        self.desc_encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pooling="mean"
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Weights for combining (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Title weight
    
    def forward(
        self,
        title_emb: torch.Tensor,
        desc_emb: torch.Tensor,
        title_mask: Optional[torch.Tensor] = None,
        desc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode book text.
        
        Args:
            title_emb: Title embeddings (batch, title_len, dim)
            desc_emb: Description embeddings (batch, desc_len, dim)
            
        Returns:
            Book embedding (batch, output_dim)
        """
        # Encode separately
        title_vec = self.title_encoder(title_emb, title_mask)
        desc_vec = self.desc_encoder(desc_emb, desc_mask)
        
        # Fuse
        combined = torch.cat([title_vec, desc_vec], dim=1)
        output = self.fusion(combined)
        
        # Normalize
        return F.normalize(output, p=2, dim=1)
