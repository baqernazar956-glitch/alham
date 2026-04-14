# -*- coding: utf-8 -*-
"""
💾 Cache Manager
=================

Caching layer for recommendations and embeddings.
Supports in-memory and Redis backends.
"""

import time
import hashlib
import json
from typing import Optional, Any, Dict, Callable
from dataclasses import dataclass
from functools import wraps
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""
    value: Any
    expires_at: float  # Unix timestamp


class MemoryCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value with TTL in seconds."""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_expired()
            
            # If still full, remove oldest
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        self._cache[key] = CacheEntry(
            value=value,
            expires_at=time.time() + ttl
        )
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
    
    def _evict_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now > v.expires_at]
        for k in expired:
            del self._cache[k]
        return len(expired)
    
    def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        now = time.time()
        valid = sum(1 for v in self._cache.values() if now <= v.expires_at)
        return {
            "total": len(self._cache),
            "valid": valid,
            "expired": len(self._cache) - valid
        }


class RedisCache:
    """Redis-backed cache."""
    
    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "ai_rec"):
        self.url = url
        self.prefix = prefix
        self._client = None
    
    @property
    def client(self):
        """Lazy redis client initialization."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.url)
                self._client.ping()
                logger.info(f"Connected to Redis at {self.url}")
            except ImportError:
                logger.error("redis package not installed")
                raise ImportError("redis package required for Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client
    
    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value."""
        try:
            data = self.client.get(self._key(key))
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value with TTL."""
        try:
            self.client.setex(
                self._key(key),
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            return self.client.delete(self._key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all prefixed keys."""
        try:
            pattern = f"{self.prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class CacheManager:
    """
    💾 Unified Cache Manager
    
    Provides a consistent interface for caching with:
    - Configurable backends (memory or redis)
    - TTL management
    - Key generation utilities
    - Cache decorator
    """
    
    def __init__(
        self,
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300,
        max_memory_size: int = 10000
    ):
        """
        Initialize cache manager.
        
        Args:
            backend: "memory" or "redis"
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            max_memory_size: Max entries for memory cache
        """
        self.backend = backend
        self.default_ttl = default_ttl
        
        self.backend = backend
        self.default_ttl = default_ttl
        
        if backend == "redis":
            try:
                import redis
                # Quick connection check
                test_client = redis.from_url(redis_url, socket_connect_timeout=1)
                test_client.ping()
                self._cache = RedisCache(url=redis_url)
                logger.info(f"CacheManager initialized with redis backend at {redis_url}")
            except Exception as e:
                logger.warning(f"⚠️ CacheManager: Redis connection failed ({e}). Falling back to memory backend.")
                self.backend = "memory"
                self._cache = MemoryCache(max_size=max_memory_size)
        else:
            self._cache = MemoryCache(max_size=max_memory_size)
            logger.info("CacheManager initialized with memory backend")
    
    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Serialize arguments
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        # Hash for consistent length
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        self._cache.set(key, value, ttl or self.default_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        return self._cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = ""
    ) -> Callable:
        """
        Decorator for caching function results.
        
        Usage:
            @cache_manager.cached(ttl=60, key_prefix="recommendations")
            def get_recommendations(user_id):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate key
                key = f"{key_prefix}:{func.__name__}:{self.make_key(*args, **kwargs)}"
                
                # Check cache
                cached_value = self.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {key}")
                    return cached_value
                
                # Compute value
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, ttl)
                logger.debug(f"Cache set: {key}")
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching a pattern.
        Only supported for memory cache.
        """
        if isinstance(self._cache, MemoryCache):
            count = 0
            keys_to_delete = [
                k for k in self._cache._cache.keys()
                if pattern in k
            ]
            for k in keys_to_delete:
                self._cache.delete(k)
                count += 1
            return count
        else:
            logger.warning("Pattern invalidation not fully supported for Redis")
            return 0
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Optional TTL override
            
        Returns:
            Cached or computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        
        value = compute_fn()
        self.set(key, value, ttl)
        return value


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache(
    backend: str = "memory",
    redis_url: str = "redis://localhost:6379"
) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(backend=backend, redis_url=redis_url)
    return _cache_manager


def reset_cache() -> None:
    """Reset the global cache manager."""
    global _cache_manager
    if _cache_manager:
        _cache_manager.clear()
    _cache_manager = None
