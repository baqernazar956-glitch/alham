# -*- coding: utf-8 -*-
"""
📊 Recommendation Metrics
==========================

Comprehensive evaluation metrics for recommendation systems.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class RecommendationMetrics:
    """
    📊 Recommendation System Metrics
    
    Implements standard recommendation evaluation metrics:
    
    Ranking Metrics:
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - MAP (Mean Average Precision)
    - MRR (Mean Reciprocal Rank)
    - Hit Rate @ K
    
    Diversity Metrics:
    - Coverage
    - Intra-List Diversity (ILD)
    - Novelty
    - Serendipity
    
    Engagement Metrics:
    - CTR (Click-Through Rate)
    - Conversion Rate
    - Dwell Time
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize metrics calculator.
        
        Args:
            k: Default cutoff for @K metrics
        """
        self.k = k
        
        logger.info(f"RecommendationMetrics initialized with k={k}")
    
    # ==================== Ranking Metrics ====================
    
    def ndcg_at_k(
        self,
        recommendations: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            recommendations: Ordered list of recommended items
            relevant: Set of relevant items
            relevance_scores: Optional relevance scores (else binary)
            k: Cutoff (default: self.k)
            
        Returns:
            NDCG@K value in [0, 1]
        """
        k = k or self.k
        recommendations = recommendations[:k]
        
        if not relevant:
            return 0.0
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(recommendations):
            if item in relevant:
                rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute Ideal DCG
        if relevance_scores:
            ideal_rels = sorted(
                [relevance_scores.get(r, 1.0) for r in relevant],
                reverse=True
            )[:k]
        else:
            ideal_rels = [1.0] * min(len(relevant), k)
        
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def precision_at_k(
        self,
        recommendations: List[str],
        relevant: Set[str],
        k: Optional[int] = None
    ) -> float:
        """Calculate Precision@K."""
        k = k or self.k
        recommendations = recommendations[:k]
        
        if not recommendations:
            return 0.0
        
        hits = sum(1 for item in recommendations if item in relevant)
        return hits / len(recommendations)
    
    def recall_at_k(
        self,
        recommendations: List[str],
        relevant: Set[str],
        k: Optional[int] = None
    ) -> float:
        """Calculate Recall@K."""
        k = k or self.k
        recommendations = recommendations[:k]
        
        if not relevant:
            return 0.0
        
        hits = sum(1 for item in recommendations if item in relevant)
        return hits / len(relevant)
    
    def average_precision(
        self,
        recommendations: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate Average Precision for a single query."""
        if not relevant:
            return 0.0
        
        hits = 0
        precision_sum = 0.0
        
        for i, item in enumerate(recommendations):
            if item in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant)
    
    def mean_average_precision(
        self,
        all_recommendations: List[List[str]],
        all_relevant: List[Set[str]]
    ) -> float:
        """Calculate Mean Average Precision across multiple queries."""
        if not all_recommendations:
            return 0.0
        
        aps = [
            self.average_precision(recs, rel)
            for recs, rel in zip(all_recommendations, all_relevant)
        ]
        
        return np.mean(aps)
    
    def reciprocal_rank(
        self,
        recommendations: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate Reciprocal Rank (position of first relevant item)."""
        for i, item in enumerate(recommendations):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def mean_reciprocal_rank(
        self,
        all_recommendations: List[List[str]],
        all_relevant: List[Set[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not all_recommendations:
            return 0.0
        
        rrs = [
            self.reciprocal_rank(recs, rel)
            for recs, rel in zip(all_recommendations, all_relevant)
        ]
        
        return np.mean(rrs)
    
    def hit_rate_at_k(
        self,
        recommendations: List[str],
        relevant: Set[str],
        k: Optional[int] = None
    ) -> float:
        """Check if any relevant item appears in top-K."""
        k = k or self.k
        recommendations = recommendations[:k]
        
        return 1.0 if any(item in relevant for item in recommendations) else 0.0
    
    # ==================== Diversity Metrics ====================
    
    def coverage(
        self,
        all_recommendations: List[List[str]],
        catalog_size: int
    ) -> float:
        """
        Calculate catalog coverage.
        
        What fraction of items ever get recommended?
        """
        if catalog_size == 0:
            return 0.0
        
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        
        return len(unique_recommended) / catalog_size
    
    def intra_list_diversity(
        self,
        recommendations: List[str],
        similarity_fn: callable
    ) -> float:
        """
        Calculate Intra-List Diversity (ILD).
        
        Average pairwise distance between recommended items.
        Higher = more diverse.
        
        Args:
            recommendations: List of recommended items
            similarity_fn: Function(item1, item2) -> float in [0, 1]
            
        Returns:
            ILD score (1 - average similarity)
        """
        n = len(recommendations)
        if n < 2:
            return 0.0
        
        total_distance = 0.0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_fn(recommendations[i], recommendations[j])
                total_distance += (1 - similarity)
                pairs += 1
        
        return total_distance / pairs
    
    def novelty(
        self,
        recommendations: List[str],
        item_popularity: Dict[str, float]
    ) -> float:
        """
        Calculate novelty as inverse popularity.
        
        Higher = recommending less popular (long-tail) items.
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 0.01)
            novelty_scores.append(-np.log2(popularity + 1e-10))
        
        return np.mean(novelty_scores)
    
    def serendipity(
        self,
        recommendations: List[str],
        expected: Set[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate serendipity.
        
        Items that are relevant but unexpected (not in baseline).
        """
        if not recommendations:
            return 0.0
        
        serendipitous = [
            item for item in recommendations
            if item in relevant and item not in expected
        ]
        
        return len(serendipitous) / len(recommendations)
    
    # ==================== Engagement Metrics ====================
    
    def click_through_rate(
        self,
        impressions: List[Dict],  # {"item_id": str, "clicked": bool}
    ) -> float:
        """Calculate CTR from impression data."""
        if not impressions:
            return 0.0
        
        clicks = sum(1 for imp in impressions if imp.get("clicked", False))
        return clicks / len(impressions)
    
    def position_weighted_ctr(
        self,
        impressions: List[Dict]  # {"item_id": str, "position": int, "clicked": bool}
    ) -> float:
        """
        CTR weighted by position (higher weight for lower positions).
        """
        if not impressions:
            return 0.0
        
        weighted_clicks = 0.0
        total_weight = 0.0
        
        for imp in impressions:
            position = imp.get("position", 1)
            weight = 1.0 / np.log2(position + 1)
            total_weight += weight
            
            if imp.get("clicked", False):
                weighted_clicks += weight
        
        return weighted_clicks / total_weight if total_weight > 0 else 0.0
    
    # ==================== Batch Evaluation ====================
    
    def evaluate(
        self,
        recommendations: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run all ranking metrics.
        
        Returns dictionary with all metric values.
        """
        k = k or self.k
        
        return {
            f"ndcg@{k}": self.ndcg_at_k(recommendations, relevant, relevance_scores, k),
            f"precision@{k}": self.precision_at_k(recommendations, relevant, k),
            f"recall@{k}": self.recall_at_k(recommendations, relevant, k),
            f"hit_rate@{k}": self.hit_rate_at_k(recommendations, relevant, k),
            "mrr": self.reciprocal_rank(recommendations, relevant),
            "ap": self.average_precision(recommendations, relevant),
        }


class MetricsTracker:
    """
    📈 Metrics Tracker
    
    Tracks metrics over time for monitoring and A/B testing.
    """
    
    def __init__(self):
        self._history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._aggregates: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")}
        )
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        now = datetime.now()
        self._history[metric_name].append((now, value))
        
        agg = self._aggregates[metric_name]
        agg["sum"] += value
        agg["count"] += 1
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
    
    def get_average(self, metric_name: str) -> float:
        """Get running average for a metric."""
        agg = self._aggregates.get(metric_name)
        if not agg or agg["count"] == 0:
            return 0.0
        return agg["sum"] / agg["count"]
    
    def get_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        agg = self._aggregates.get(metric_name)
        if not agg or agg["count"] == 0:
            return {"mean": 0, "min": 0, "max": 0, "count": 0}
        
        return {
            "mean": agg["sum"] / agg["count"],
            "min": agg["min"] if agg["min"] != float("inf") else 0,
            "max": agg["max"] if agg["max"] != float("-inf") else 0,
            "count": agg["count"]
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all tracked metrics."""
        return {name: self.get_summary(name) for name in self._aggregates}
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._history.clear()
        self._aggregates.clear()
