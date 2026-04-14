# -*- coding: utf-8 -*-
"""
🔍 Recommendation Explainer
============================

Generates explanations for why items were recommended.
Uses multiple signals:
- Content similarity
- User behavior patterns
- Collaborative filtering signals
- Feature importance
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations."""
    CONTENT_SIMILARITY = "content_similarity"
    BEHAVIORAL = "behavioral"
    COLLABORATIVE = "collaborative"
    POPULARITY = "popularity"
    AUTHOR_MATCH = "author_match"
    CATEGORY_MATCH = "category_match"
    TOPIC_OVERLAP = "topic_overlap"
    TEMPORAL = "temporal"
    DIVERSITY = "diversity"


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    
    item_id: str
    primary_reason: str  # Human-readable main reason
    secondary_reasons: List[str] = field(default_factory=list)
    
    # Detailed breakdown
    signals: Dict[str, float] = field(default_factory=dict)
    matched_items: List[str] = field(default_factory=list)  # Items this was similar to
    matched_features: List[str] = field(default_factory=list)  # Features that matched
    
    # Confidence
    confidence_score: float = 0.0
    
    # For UI display
    explanation_type: ExplanationType = ExplanationType.CONTENT_SIMILARITY
    icon: str = "📚"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "item_id": self.item_id,
            "primary_reason": self.primary_reason,
            "secondary_reasons": self.secondary_reasons,
            "signals": self.signals,
            "matched_items": self.matched_items,
            "matched_features": self.matched_features,
            "confidence": self.confidence_score,
            "type": self.explanation_type.value,
            "icon": self.icon
        }


class RecommendationExplainer:
    """
    🔍 Recommendation Explainer
    
    Analyzes recommendation scores and generates
    human-readable explanations.
    
    Explanation strategies:
    1. Content-based: "Similar to books you've read"
    2. Behavioral: "Based on your recent activity"
    3. Collaborative: "Users like you also read this"
    4. Author/Category: "From an author/category you like"
    5. Trending: "Popular in your interests"
    """
    
    def __init__(
        self,
        min_confidence: float = 0.3,
        max_explanations: int = 3
    ):
        self.min_confidence = min_confidence
        self.max_explanations = max_explanations
        
        # Templates for explanations
        self.templates = {
            ExplanationType.CONTENT_SIMILARITY: [
                "Similar to '{book}' which you enjoyed",
                "Matches themes from books you've read",
                "Related to your reading history",
            ],
            ExplanationType.BEHAVIORAL: [
                "Based on your recent browsing",
                "Matches your current reading interests",
                "From your active reading session",
            ],
            ExplanationType.COLLABORATIVE: [
                "Readers with similar taste loved this",
                "Popular among readers like you",
                "Recommended by readers who share your interests",
            ],
            ExplanationType.AUTHOR_MATCH: [
                "From {author}, an author you follow",
                "By {author}, whose work you've enjoyed",
                "Another book by {author}",
            ],
            ExplanationType.CATEGORY_MATCH: [
                "In {category}, a genre you love",
                "Matches your interest in {category}",
                "From your favorite genre: {category}",
            ],
            ExplanationType.TOPIC_OVERLAP: [
                "Shares topics with books you've rated highly",
                "Contains themes you've shown interest in",
                "Has overlapping topics with your favorites",
            ],
            ExplanationType.POPULARITY: [
                "Trending in your areas of interest",
                "Highly rated in {category}",
                "Popular among readers this week",
            ],
            ExplanationType.TEMPORAL: [
                "New release matching your preferences",
                "Recently added to match your tastes",
            ],
            ExplanationType.DIVERSITY: [
                "Something different you might enjoy",
                "Expand your reading horizons",
            ],
        }
        
        # Icons for each type
        self.icons = {
            ExplanationType.CONTENT_SIMILARITY: "📚",
            ExplanationType.BEHAVIORAL: "👁️",
            ExplanationType.COLLABORATIVE: "👥",
            ExplanationType.AUTHOR_MATCH: "✍️",
            ExplanationType.CATEGORY_MATCH: "📂",
            ExplanationType.TOPIC_OVERLAP: "🏷️",
            ExplanationType.POPULARITY: "🔥",
            ExplanationType.TEMPORAL: "🆕",
            ExplanationType.DIVERSITY: "🌈",
        }
        
        logger.info("RecommendationExplainer initialized")
    
    def explain(
        self,
        item_id: str,
        score_breakdown: Dict[str, float],
        user_history: Optional[List[Dict]] = None,
        item_metadata: Optional[Dict] = None,
        similar_items: Optional[List[Dict]] = None
    ) -> ExplanationResult:
        """
        Generate explanation for a recommendation.
        
        Args:
            item_id: ID of recommended item
            score_breakdown: Scores from each signal source
            user_history: User's interaction history
            item_metadata: Metadata about the recommended item
            similar_items: Items this was similar to
            
        Returns:
            ExplanationResult with human-readable explanations
        """
        # Analyze which signals contributed most
        signals = self._normalize_signals(score_breakdown)
        
        # Determine primary explanation type
        primary_type = self._determine_primary_type(signals)
        
        # Generate primary reason
        primary_reason = self._generate_primary_reason(
            primary_type, signals, user_history, item_metadata, similar_items
        )
        
        # Generate secondary reasons
        secondary_reasons = self._generate_secondary_reasons(
            signals, primary_type, user_history, item_metadata
        )
        
        # Extract matched items/features
        matched_items = []
        matched_features = []
        
        if similar_items:
            matched_items = [item.get("id", "") for item in similar_items[:3]]
        
        if item_metadata:
            if item_metadata.get("categories"):
                matched_features.extend(item_metadata["categories"][:2])
            if item_metadata.get("topics"):
                matched_features.extend(item_metadata["topics"][:2])
        
        # Confidence based on score strength
        confidence = self._compute_confidence(signals)
        
        return ExplanationResult(
            item_id=item_id,
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons[:self.max_explanations - 1],
            signals=signals,
            matched_items=matched_items,
            matched_features=matched_features,
            confidence_score=confidence,
            explanation_type=primary_type,
            icon=self.icons.get(primary_type, "📚")
        )
    
    def _normalize_signals(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        
        max_score = max(abs(v) for v in scores.values()) or 1.0
        return {k: v / max_score for k, v in scores.items()}
    
    def _determine_primary_type(self, signals: Dict[str, float]) -> ExplanationType:
        """Determine the primary explanation type."""
        # Map signal names to explanation types
        signal_type_map = {
            "semantic": ExplanationType.CONTENT_SIMILARITY,
            "two_tower": ExplanationType.CONTENT_SIMILARITY,
            "collaborative": ExplanationType.COLLABORATIVE,
            "graph": ExplanationType.COLLABORATIVE,
            "behavioral": ExplanationType.BEHAVIORAL,
            "author": ExplanationType.AUTHOR_MATCH,
            "category": ExplanationType.CATEGORY_MATCH,
            "topic": ExplanationType.TOPIC_OVERLAP,
            "popularity": ExplanationType.POPULARITY,
            "temporal": ExplanationType.TEMPORAL,
            "diversity": ExplanationType.DIVERSITY,
        }
        
        # Find highest scoring signal
        best_signal = max(signals.items(), key=lambda x: x[1], default=("semantic", 0.5))
        signal_name = best_signal[0].lower()
        
        for key, exp_type in signal_type_map.items():
            if key in signal_name:
                return exp_type
        
        return ExplanationType.CONTENT_SIMILARITY
    
    def _generate_primary_reason(
        self,
        exp_type: ExplanationType,
        signals: Dict[str, float],
        user_history: Optional[List[Dict]],
        item_metadata: Optional[Dict],
        similar_items: Optional[List[Dict]]
    ) -> str:
        """Generate the primary explanation reason."""
        templates = self.templates.get(exp_type, ["Recommended for you"])
        template = np.random.choice(templates)
        
        # Fill in placeholders
        if "{book}" in template and similar_items:
            book_title = similar_items[0].get("title", "a book you liked")
            template = template.replace("{book}", book_title)
        
        if "{author}" in template and item_metadata:
            author = item_metadata.get("author", "this author")
            template = template.replace("{author}", author)
        
        if "{category}" in template and item_metadata:
            categories = item_metadata.get("categories", [])
            category = categories[0] if categories else "your interests"
            template = template.replace("{category}", category)
        
        return template
    
    def _generate_secondary_reasons(
        self,
        signals: Dict[str, float],
        primary_type: ExplanationType,
        user_history: Optional[List[Dict]],
        item_metadata: Optional[Dict]
    ) -> List[str]:
        """Generate secondary explanation reasons."""
        reasons = []
        
        # Sort signals by strength
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        
        for signal_name, strength in sorted_signals[1:4]:  # Skip primary
            if strength < self.min_confidence:
                continue
            
            # Generate reason based on signal
            if "popular" in signal_name.lower():
                reasons.append("Highly rated by readers")
            elif "collab" in signal_name.lower() or "graph" in signal_name.lower():
                reasons.append("Enjoyed by similar readers")
            elif "semantic" in signal_name.lower():
                reasons.append("Matches your reading taste")
            elif "behavior" in signal_name.lower():
                reasons.append("Fits your recent activity")
        
        return reasons
    
    def _compute_confidence(self, signals: Dict[str, float]) -> float:
        """Compute overall explanation confidence."""
        if not signals:
            return 0.5
        
        values = list(signals.values())
        
        # Confidence based on signal agreement
        max_signal = max(values)
        mean_signal = np.mean(values)
        
        # Higher if signals agree
        agreement = 1 - np.std(values)
        
        return 0.5 * max_signal + 0.3 * mean_signal + 0.2 * agreement
    
    def batch_explain(
        self,
        items: List[Tuple[str, Dict[str, float]]],
        user_history: Optional[List[Dict]] = None,
        item_metadata_map: Optional[Dict[str, Dict]] = None,
        similarity_map: Optional[Dict[str, List[Dict]]] = None
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple items.
        
        Args:
            items: List of (item_id, score_breakdown) tuples
            user_history: User's history
            item_metadata_map: Metadata keyed by item_id
            similarity_map: Similar items keyed by item_id
            
        Returns:
            List of ExplanationResults
        """
        results = []
        
        for item_id, score_breakdown in items:
            metadata = item_metadata_map.get(item_id) if item_metadata_map else None
            similar = similarity_map.get(item_id) if similarity_map else None
            
            result = self.explain(
                item_id=item_id,
                score_breakdown=score_breakdown,
                user_history=user_history,
                item_metadata=metadata,
                similar_items=similar
            )
            results.append(result)
        
        return results
