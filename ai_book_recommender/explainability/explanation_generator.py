# -*- coding: utf-8 -*-
"""
💬 Explanation Generator
=========================

Generates natural language explanations using templates and LLM.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    💬 Natural Language Explanation Generator
    
    Creates user-friendly explanations combining:
    - Template-based generation
    - Feature attribution
    - Comparison-based explanations
    """
    
    def __init__(self, style: str = "conversational"):
        """
        Initialize generator.
        
        Args:
            style: "conversational", "concise", or "detailed"
        """
        self.style = style
        
        # Explanation templates by category
        self.templates = self._load_templates()
        
        logger.info(f"ExplanationGenerator initialized with style: {style}")
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load explanation templates."""
        return {
            "similar_book": [
                "Because you enjoyed \"{book}\", we think you'll love this one too.",
                "Based on your interest in \"{book}\", this seems like a great match.",
                "Readers who loved \"{book}\" often enjoy this.",
            ],
            "similar_author": [
                "You've enjoyed books by {author} before.",
                "Since you like {author}'s writing style, try this.",
                "Another great read from {author}.",
            ],
            "category_match": [
                "This fits your love of {category} books.",
                "A {category} book matching your taste.",
                "Since you enjoy {category}, check this out.",
            ],
            "topic_match": [
                "Contains themes of {topic} that interest you.",
                "Explores {topic}, which you've shown interest in.",
                "If you like reading about {topic}, this is for you.",
            ],
            "popularity": [
                "Highly rated in the {category} genre.",
                "A reader favorite this month.",
                "Top pick among readers like you.",
            ],
            "behavioral": [
                "Based on your recent browsing activity.",
                "Matches your current reading mood.",
                "Suggested from your session today.",
            ],
            "collaborative": [
                "Readers with similar taste recommend this.",
                "Popular with people who share your interests.",
                "A community favorite among similar readers.",
            ],
            "diversity": [
                "Something a bit different for you.",
                "Expand your horizons with this pick.",
                "A refreshing change from your usual reads.",
            ],
            "default": [
                "We think you'll enjoy this book.",
                "A personalized recommendation for you.",
                "Picked just for your reading taste.",
            ]
        }
    
    def generate(
        self,
        item_metadata: Dict[str, Any],
        explanation_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a natural language explanation.
        
        Args:
            item_metadata: Metadata about the recommended item
            explanation_type: Type of explanation to generate
            context: Additional context (matched book, author, etc.)
            
        Returns:
            Human-readable explanation string
        """
        templates = self.templates.get(explanation_type, self.templates["default"])
        template = np.random.choice(templates)
        
        # Fill in placeholders
        context = context or {}
        
        # Standard replacements
        replacements = {
            "{book}": context.get("matched_book", "a book you liked"),
            "{author}": context.get("author", item_metadata.get("author", "this author")),
            "{category}": self._get_category(item_metadata, context),
            "{topic}": context.get("topic", "topics you enjoy"),
        }
        
        explanation = template
        for key, value in replacements.items():
            explanation = explanation.replace(key, str(value))
        
        # Apply style adjustments
        if self.style == "concise":
            explanation = self._make_concise(explanation)
        elif self.style == "detailed":
            explanation = self._make_detailed(explanation, item_metadata, context)
        
        return explanation
    
    def _get_category(
        self,
        item_metadata: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Extract category from metadata or context."""
        if "category" in context:
            return context["category"]
        
        categories = item_metadata.get("categories", [])
        if categories:
            return categories[0]
        
        return "your favorite genre"
    
    def _make_concise(self, explanation: str) -> str:
        """Shorten explanation for concise style."""
        # Remove filler words
        concise = explanation.replace("we think ", "")
        concise = concise.replace("Based on ", "From ")
        
        # Limit length
        if len(concise) > 50:
            concise = concise[:47] + "..."
        
        return concise
    
    def _make_detailed(
        self,
        explanation: str,
        item_metadata: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Expand explanation for detailed style."""
        details = []
        
        # Add rating info
        if "rating" in item_metadata:
            rating = item_metadata["rating"]
            details.append(f"Rated {rating:.1f}/5 by readers")
        
        # Add similarity score
        if "similarity_score" in context:
            score = context["similarity_score"] * 100
            details.append(f"{score:.0f}% match with your taste")
        
        # Add page count
        if "page_count" in item_metadata:
            pages = item_metadata["page_count"]
            details.append(f"{pages} pages")
        
        if details:
            explanation += f" ({', '.join(details)})"
        
        return explanation
    
    def generate_multi_reason(
        self,
        reasons: List[Dict[str, Any]],
        max_reasons: int = 3
    ) -> str:
        """
        Generate explanation combining multiple reasons.
        
        Args:
            reasons: List of reason dictionaries with type and context
            max_reasons: Maximum reasons to include
            
        Returns:
            Combined explanation
        """
        if not reasons:
            return np.random.choice(self.templates["default"])
        
        explanations = []
        for reason in reasons[:max_reasons]:
            exp = self.generate(
                item_metadata=reason.get("item_metadata", {}),
                explanation_type=reason.get("type", "default"),
                context=reason.get("context", {})
            )
            explanations.append(exp)
        
        # Combine with connectors
        if len(explanations) == 1:
            return explanations[0]
        elif len(explanations) == 2:
            return f"{explanations[0]} Also, {explanations[1].lower()}"
        else:
            main = explanations[0]
            others = " • ".join(exp.capitalize() for exp in explanations[1:])
            return f"{main} • {others}"
    
    def format_for_ui(
        self,
        explanation: str,
        format_type: str = "card"
    ) -> Dict[str, Any]:
        """
        Format explanation for different UI contexts.
        
        Args:
            explanation: The explanation text
            format_type: "card", "tooltip", "list", or "full"
            
        Returns:
            Formatted explanation dict for UI
        """
        if format_type == "tooltip":
            return {
                "text": explanation[:100] + "..." if len(explanation) > 100 else explanation,
                "truncated": len(explanation) > 100
            }
        
        elif format_type == "card":
            return {
                "primary": explanation.split(".")[0] + "." if "." in explanation else explanation,
                "full": explanation
            }
        
        elif format_type == "list":
            # Split into bullet points
            points = explanation.split(". ")
            return {
                "bullets": [p.strip() for p in points if p.strip()]
            }
        
        else:  # full
            return {
                "text": explanation,
                "html": f"<p class='explanation'>{explanation}</p>"
            }
