# -*- coding: utf-8 -*-
"""
Global embedding cache — stores deserialized embeddings in memory.
"""
import logging
import numpy as np

from ..models import BookEmbedding

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global Performance Caches
# ------------------------------------------------------------------
# This cache stores deserialized embeddings in memory to avoid heavy DB queries
# on every page refresh or recommendation request.
_GLOBAL_EMBEDDING_CACHE = {
    'matrix': None,      # numpy matrix (N, D)
    'book_ids': [],      # List of book IDs corresponding to rows
    'last_updated': 0,   # Timestamp
    'lock': False        # Simple flag for atomic-ish updates
}

def _get_embeddings_matrix(ttl=3600):
    """
    Helper to get the embeddings matrix from memory, loading it from DB if needed.
    TTL defaults to 1 hour to account for new books.
    """
    import time
    now = time.time()
    
    # 1. Check if cache is valid
    if (_GLOBAL_EMBEDDING_CACHE['matrix'] is not None and 
        (now - _GLOBAL_EMBEDDING_CACHE['last_updated'] < ttl)):
        return _GLOBAL_EMBEDDING_CACHE['matrix'], _GLOBAL_EMBEDDING_CACHE['book_ids']

    # 2. Loading from DB
    try:
        logger.info("[Embedding-Cache] Loading embeddings matrix from database...")
        start_time = time.perf_counter()
        
        # Pull all embeddings
        all_rows = BookEmbedding.query.all()
        if not all_rows:
            return None, []

        ids = []
        vectors = []
        target_dim = None
        
        for row in all_rows:
            if row.vector is not None:
                v = np.array(__import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector, dtype=np.float32)
                if v.ndim == 1:
                    # Initialize target_dim from the first valid vector
                    if target_dim is None:
                         target_dim = v.shape[0]
                         
                    # Only include vectors with consistent dimension
                    if v.shape[0] == target_dim:
                        ids.append(row.book_id)
                        vectors.append(v)
        
        if not vectors:
            return None, []

        # Update Cache
        _GLOBAL_EMBEDDING_CACHE['matrix'] = np.vstack(vectors)
        _GLOBAL_EMBEDDING_CACHE['book_ids'] = ids
        _GLOBAL_EMBEDDING_CACHE['last_updated'] = now
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"[Embedding-Cache] Matrix loaded: {len(ids)} vectors in {elapsed:.2f}ms")
        
        return _GLOBAL_EMBEDDING_CACHE['matrix'], _GLOBAL_EMBEDDING_CACHE['book_ids']
        
    except Exception as e:
        logger.error(f"[Embedding-Cache] Error loading matrix: {e}", exc_info=True)
        return None, []
