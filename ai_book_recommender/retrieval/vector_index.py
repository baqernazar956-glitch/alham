# -*- coding: utf-8 -*-
"""
📊 FAISS Vector Index
======================

High-performance vector similarity search using FAISS.
Supports multiple index types for different scale/accuracy tradeoffs.
"""

import numpy as np
import os
import pickle
from typing import Optional, List, Tuple, Dict, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Lazy import FAISS
_faiss = None

def get_faiss():
    """Lazy load FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            raise ImportError("FAISS is required for vector search")
    return _faiss


class FAISSIndex:
    """
    📊 FAISS Vector Index
    
    Wrapper around FAISS for vector similarity search.
    
    Supported index types:
    - Flat: Exact search, best accuracy, O(n) per query
    - IVF: Inverted file, good balance, O(n/nlist) per query
    - HNSW: Graph-based, fast but memory intensive
    - IVF+PQ: Compressed, for very large datasets
    
    All indexes use Inner Product (cosine similarity for normalized vectors).
    """
    
    def __init__(
        self,
        dim: int = 384,
        index_type: Literal["Flat", "IVF", "HNSW", "IVFPQ"] = "IVF",
        nlist: int = 100,
        nprobe: int = 10,
        m_hnsw: int = 32,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS index.
        
        Args:
            dim: Vector dimension
            index_type: Type of index
            nlist: Number of clusters (for IVF)
            nprobe: Number of clusters to search (for IVF)
            m_hnsw: Number of connections per layer (for HNSW)
            use_gpu: Whether to use GPU (if available)
        """
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m_hnsw = m_hnsw
        self.use_gpu = use_gpu
        
        # Index and metadata
        self.index = None
        self.id_map: Dict[int, str] = {}  # FAISS idx -> original ID
        self.reverse_map: Dict[str, int] = {}  # original ID -> FAISS idx
        
        self._is_trained = False
        
        logger.info(
            f"FAISSIndex: dim={dim}, type={index_type}, nlist={nlist}, nprobe={nprobe}"
        )
    
    def _create_index(self, num_vectors: int = 0) -> None:
        """Create the FAISS index structure."""
        faiss = get_faiss()
        
        if self.index_type == "Flat":
            # Exact search
            self.index = faiss.IndexFlatIP(self.dim)
            self._is_trained = True
            
        elif self.index_type == "IVF":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(self.nlist, max(1, num_vectors // 10))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.nprobe
            
        elif self.index_type == "HNSW":
            # HNSW graph-based
            self.index = faiss.IndexHNSWFlat(self.dim, self.m_hnsw, faiss.METRIC_INNER_PRODUCT)
            self._is_trained = True
            
        elif self.index_type == "IVFPQ":
            # IVF with Product Quantization for compression
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(self.nlist, max(1, num_vectors // 10))
            m = 8  # Number of subquantizers
            bits = 8
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.nprobe
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move to GPU: {e}")
    
    def build(
        self,
        vectors: np.ndarray,
        ids: List[str],
        normalize: bool = True
    ) -> int:
        """
        Build index from vectors.
        
        Args:
            vectors: Embeddings (N, dim)
            ids: Corresponding IDs
            normalize: Whether to L2 normalize vectors
            
        Returns:
            Number of vectors indexed
        """
        if len(vectors) == 0:
            logger.warning("No vectors to index")
            return 0
        
        if len(vectors) != len(ids):
            raise ValueError(f"Vectors ({len(vectors)}) and IDs ({len(ids)}) must match")
        
        # Ensure float32
        vectors = np.array(vectors, dtype=np.float32)
        
        # Normalize for cosine similarity
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            vectors = vectors / norms
        
        # Create index
        self._create_index(len(vectors))
        
        # Train if needed
        if not self._is_trained:
            logger.info(f"Training index with {len(vectors)} vectors...")
            self.index.train(vectors)
            self._is_trained = True
        
        # Add vectors
        self.index.add(vectors)
        
        # Build ID mappings
        self.id_map = {i: str(ids[i]) for i in range(len(ids))}
        self.reverse_map = {str(ids[i]): i for i in range(len(ids))}
        
        logger.info(f"Index built with {len(vectors)} vectors")
        return len(vectors)
    
    def add(
        self,
        vectors: np.ndarray,
        ids: List[str],
        normalize: bool = True
    ) -> int:
        """
        Add vectors to existing index.
        
        Args:
            vectors: New embeddings (N, dim)
            ids: Corresponding IDs
            normalize: Whether to normalize
            
        Returns:
            Number of vectors added
        """
        if self.index is None:
            return self.build(vectors, ids, normalize)
        
        vectors = np.array(vectors, dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            vectors = vectors / norms
        
        # Get current count
        current_count = self.index.ntotal
        
        # Add to index
        self.index.add(vectors)
        
        # Update mappings
        for i, id_ in enumerate(ids):
            idx = current_count + i
            self.id_map[idx] = str(id_)
            self.reverse_map[str(id_)] = idx
        
        return len(vectors)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        normalize: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector (dim,) or (1, dim)
            k: Number of results
            normalize: Whether to normalize query
            
        Returns:
            List of (id, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure 2D
        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize
        if normalize:
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # Map back to IDs
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
        
        return results
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        normalize: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Query vectors (N, dim)
            k: Results per query
            normalize: Whether to normalize
            
        Returns:
            List of result lists
        """
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(queries))]
        
        queries = np.array(queries, dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            queries = queries / norms
        
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(queries, k)
        
        results = []
        for query_indices, query_scores in zip(indices, distances):
            query_results = []
            for idx, score in zip(query_indices, query_scores):
                if idx >= 0 and idx in self.id_map:
                    query_results.append((self.id_map[idx], float(score)))
            results.append(query_results)
        
        return results
    
    def get_vector(self, id_: str) -> Optional[np.ndarray]:
        """Get vector by ID."""
        if id_ not in self.reverse_map:
            return None
        
        idx = self.reverse_map[id_]
        return self.index.reconstruct(idx)
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        faiss = get_faiss()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            # Convert back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path))
        else:
            faiss.write_index(self.index, str(path))
        
        # Save metadata
        meta_path = str(path) + ".meta"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "reverse_map": self.reverse_map,
                "dim": self.dim,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
            }, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        faiss = get_faiss()
        
        # Load FAISS index
        self.index = faiss.read_index(str(path))
        
        # Load metadata
        meta_path = str(path) + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.id_map = meta["id_map"]
                self.reverse_map = meta["reverse_map"]
                self.dim = meta.get("dim", self.dim)
                self.index_type = meta.get("index_type", self.index_type)
        
        self._is_trained = True
        
        # Set nprobe if IVF index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        
        logger.info(f"Index loaded from {path}, {self.index.ntotal} vectors")
    
    @property
    def size(self) -> int:
        """Get number of indexed vectors."""
        return self.index.ntotal if self.index else 0


class VectorIndexService:
    """
    🔍 Vector Index Service
    
    High-level service for managing vector indexes.
    Supports multiple indexes and automatic reloading.
    """
    
    def __init__(
        self,
        index_dir: str = "instance/indexes",
        default_dim: int = 384,
        default_type: str = "IVF"
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_dim = default_dim
        self.default_type = default_type
        
        # Index cache
        self._indexes: Dict[str, FAISSIndex] = {}
        
        logger.info(f"VectorIndexService: dir={index_dir}")
    
    def get_index(self, name: str = "default") -> FAISSIndex:
        """Get or create an index by name."""
        if name not in self._indexes:
            index = FAISSIndex(dim=self.default_dim, index_type=self.default_type)
            
            # Try to load from disk
            path = self.index_dir / f"{name}.index"
            if path.exists():
                index.load(str(path))
            
            self._indexes[name] = index
        
        return self._indexes[name]
    
    def build_index(
        self,
        name: str,
        vectors: np.ndarray,
        ids: List[str],
        save: bool = True
    ) -> int:
        """Build and optionally save an index."""
        index = FAISSIndex(dim=vectors.shape[1], index_type=self.default_type)
        count = index.build(vectors, ids)
        
        self._indexes[name] = index
        
        if save:
            path = self.index_dir / f"{name}.index"
            index.save(str(path))
        
        return count
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        index_name: str = "default"
    ) -> List[Tuple[str, float]]:
        """Search an index."""
        index = self.get_index(index_name)
        return index.search(query, k)
    
    def reload_index(self, name: str) -> None:
        """Force reload an index from disk."""
        if name in self._indexes:
            del self._indexes[name]
        self.get_index(name)
    
    def list_indexes(self) -> List[str]:
        """List available indexes."""
        indexes = []
        for path in self.index_dir.glob("*.index"):
            indexes.append(path.stem)
        return indexes
