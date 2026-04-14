# -*- coding: utf-8 -*-
"""
🔀 Hybrid Retrieval
====================

Combines multiple retrieval strategies:
- Semantic search (vector similarity)
- Keyword search (BM25)
- Popularity-based retrieval
- Reciprocal Rank Fusion (RRF) for combining results
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with source tracking."""
    
    item_id: str
    score: float
    source: str  # "semantic", "keyword", "popularity", "hybrid"
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """
    📝 BM25 Keyword Search
    
    Simple BM25 implementation for keyword matching.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Document data
        self.docs: Dict[str, List[str]] = {}  # id -> tokens
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        
        # Inverted index
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.doc_freqs: Dict[str, int] = {}  # token -> num docs containing
        
        self.n_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation, split on whitespace
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: Dict of {id: text}
        """
        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self.docs[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            
            # Update inverted index
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.inverted_index[token].add(doc_id)
        
        # Update statistics
        self.n_docs = len(self.docs)
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, self.n_docs)
        
        # Update document frequencies
        self.doc_freqs = {
            token: len(doc_ids)
            for token, doc_ids in self.inverted_index.items()
        }
        
        logger.info(f"BM25 index built with {self.n_docs} documents")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            # IDF
            df = self.doc_freqs.get(token, 0)
            idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # Score each document containing this token
            for doc_id in self.inverted_index[token]:
                doc_tokens = self.docs[doc_id]
                tf = doc_tokens.count(token)
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                
                scores[doc_id] += idf * numerator / denominator
        
        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


class HybridRetriever:
    """
    🔀 Hybrid Retrieval System
    
    Combines multiple retrieval methods:
    1. Semantic search (FAISS vector similarity)
    2. Keyword search (BM25)
    3. Popularity-based retrieval
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    
    RRF formula:
        score(d) = Σ 1 / (k + rank_i(d))
        
    where k is a constant (default 60) and rank_i(d) is the rank
    of document d in result list i.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.3,
        popularity_weight: float = 0.2,
        rrf_k: int = 60,
        use_rrf: bool = True
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_weight: Weight for semantic search (if not using RRF)
            keyword_weight: Weight for keyword search
            popularity_weight: Weight for popularity
            rrf_k: RRF constant
            use_rrf: Whether to use RRF (else weighted sum)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.popularity_weight = popularity_weight
        self.rrf_k = rrf_k
        self.use_rrf = use_rrf
        
        # Components
        self.vector_index = None
        self.bm25_index = BM25Index()
        self.popularity_scores: Dict[str, float] = {}
        
        logger.info(
            f"HybridRetriever: semantic={semantic_weight}, keyword={keyword_weight}, "
            f"popularity={popularity_weight}, rrf={use_rrf}"
        )
    
    def set_vector_index(self, index) -> None:
        """Set the vector index for semantic search."""
        self.vector_index = index
    
    def index_documents(
        self,
        documents: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Index documents for keyword search.
        
        Args:
            documents: Dict of {id: {"title": ..., "description": ..., "popularity": ...}}
        """
        # Build BM25 index from combined text
        texts = {}
        for doc_id, doc in documents.items():
            text = f"{doc.get('title', '')} {doc.get('description', '')}"
            texts[doc_id] = text
            
            # Store popularity
            self.popularity_scores[doc_id] = doc.get("popularity", 0.0)
        
        self.bm25_index.add_documents(texts)
    
    def set_popularity_scores(self, scores: Dict[str, float]) -> None:
        """Set popularity scores for items."""
        self.popularity_scores = scores
    
    def search(
        self,
        query_vector: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        rerank_k: int = 100,
        filter_ids: Optional[Set[str]] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search.
        
        Args:
            query_vector: Query embedding for semantic search
            query_text: Query text for keyword search
            k: Number of final results
            rerank_k: Number of candidates to retrieve from each source
            filter_ids: Optional set of IDs to exclude
            
        Returns:
            List of RetrievalResult objects
        """
        candidates: Dict[str, Dict[str, Any]] = {}  # id -> {scores, ranks}
        
        # 1. Semantic search
        if query_vector is not None and self.vector_index is not None:
            semantic_results = self.vector_index.search(query_vector, k=rerank_k)
            
            for rank, (item_id, score) in enumerate(semantic_results):
                if filter_ids and item_id in filter_ids:
                    continue
                
                if item_id not in candidates:
                    candidates[item_id] = {"scores": {}, "ranks": {}}
                
                candidates[item_id]["scores"]["semantic"] = score
                candidates[item_id]["ranks"]["semantic"] = rank + 1
        
        # 2. Keyword search
        if query_text and self.bm25_index.n_docs > 0:
            keyword_results = self.bm25_index.search(query_text, k=rerank_k)
            
            for rank, (item_id, score) in enumerate(keyword_results):
                if filter_ids and item_id in filter_ids:
                    continue
                
                if item_id not in candidates:
                    candidates[item_id] = {"scores": {}, "ranks": {}}
                
                candidates[item_id]["scores"]["keyword"] = score
                candidates[item_id]["ranks"]["keyword"] = rank + 1
        
        # 3. Popularity
        if self.popularity_scores:
            # Sort by popularity
            sorted_pop = sorted(
                self.popularity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:rerank_k]
            
            for rank, (item_id, score) in enumerate(sorted_pop):
                if filter_ids and item_id in filter_ids:
                    continue
                
                if item_id not in candidates:
                    candidates[item_id] = {"scores": {}, "ranks": {}}
                
                candidates[item_id]["scores"]["popularity"] = score
                candidates[item_id]["ranks"]["popularity"] = rank + 1
        
        # 4. Combine scores
        final_scores = self._combine_scores(candidates)
        
        # 5. Sort and return top-k
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (item_id, score) in enumerate(sorted_items[:k]):
            # Determine primary source
            item_data = candidates.get(item_id, {"ranks": {}})
            source = self._get_primary_source(item_data["ranks"])
            
            results.append(RetrievalResult(
                item_id=item_id,
                score=score,
                source=source,
                rank=rank + 1,
                metadata=item_data.get("scores", {})
            ))
        
        return results
    
    def _combine_scores(
        self,
        candidates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Combine scores from multiple sources."""
        final_scores = {}
        
        if self.use_rrf:
            # Reciprocal Rank Fusion
            for item_id, data in candidates.items():
                rrf_score = 0.0
                
                for source, rank in data.get("ranks", {}).items():
                    rrf_score += 1.0 / (self.rrf_k + rank)
                
                final_scores[item_id] = rrf_score
        else:
            # Weighted sum
            weight_map = {
                "semantic": self.semantic_weight,
                "keyword": self.keyword_weight,
                "popularity": self.popularity_weight
            }
            
            # Normalize scores per source
            source_scores: Dict[str, List[float]] = defaultdict(list)
            for item_id, data in candidates.items():
                for source, score in data.get("scores", {}).items():
                    source_scores[source].append(score)
            
            source_max = {
                source: max(scores) if scores else 1.0
                for source, scores in source_scores.items()
            }
            source_min = {
                source: min(scores) if scores else 0.0
                for source, scores in source_scores.items()
            }
            
            for item_id, data in candidates.items():
                weighted_sum = 0.0
                total_weight = 0.0
                
                for source, score in data.get("scores", {}).items():
                    weight = weight_map.get(source, 0.0)
                    if weight <= 0:
                        continue
                    
                    # Normalize score
                    s_min = source_min.get(source, 0.0)
                    s_max = source_max.get(source, 1.0)
                    if s_max - s_min > 1e-8:
                        normalized = (score - s_min) / (s_max - s_min)
                    else:
                        normalized = 0.5
                    
                    weighted_sum += weight * normalized
                    total_weight += weight
                
                if total_weight > 0:
                    final_scores[item_id] = weighted_sum / total_weight
                else:
                    final_scores[item_id] = 0.0
        
        return final_scores
    
    def _get_primary_source(self, ranks: Dict[str, int]) -> str:
        """Determine primary source based on best rank."""
        if not ranks:
            return "hybrid"
        
        best_source = min(ranks.items(), key=lambda x: x[1])
        return best_source[0] if best_source[1] <= 10 else "hybrid"


def get_vector_search_results(query: str, top_n: int = 50) -> List[Dict]:
    """
    Standalone helper for vectorized search.
    Used by the unified pipeline for immediate behavioral retrieval.
    """
    from .vector_index import VectorIndexService
    from flask_book_recommendation.utils import get_text_embedding
    from flask_book_recommendation.models import Book
    import os
    
    # 1. Get embedding for the query
    emb = get_text_embedding(query)
    if emb is None:
        return []
    
    # 2. Search FAISS index
    index_service = VectorIndexService(index_dir=os.path.join("instance", "indexes"))
    results = index_service.search(np.array(emb, dtype=np.float32), k=top_n, index_name="books")
    
    if not results:
        return []
        
    # 3. Resolve IDs to book metadata
    book_candidates = []
    ids = [r[0] for r in results]
    
    # Check if they are google_ids or local ids
    google_ids = [bid for bid in ids if not str(bid).isdigit()]
    local_ids = [int(bid) for bid in ids if str(bid).isdigit()]
    
    books_map = {}
    from flask_book_recommendation.extensions import db
    from flask import current_app
    
    def _fetch_books():
        if google_ids:
            found = Book.query.filter(Book.google_id.in_(google_ids)).all()
            for b in found: books_map[b.google_id] = b
        if local_ids:
            found = Book.query.filter(Book.id.in_(local_ids)).all()
            for b in found: books_map[str(b.id)] = b
            
    if current_app:
        with current_app.app_context():
            _fetch_books()
    else:
        # Fallback if no app context (rare in production)
        _fetch_books()
        
    for bid, score in results:
        book = books_map.get(str(bid))
        if book:
            book_candidates.append({
                "id": str(bid),
                "google_id": book.google_id,
                "title": book.title,
                "author": book.author,
                "cover": book.cover_url,
                "score": float(score),
                "source": "Recent Search"
            })
            
    return book_candidates

