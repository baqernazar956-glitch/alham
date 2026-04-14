# -*- coding: utf-8 -*-
"""
🌐 FastAPI Recommendation Service
==================================

Production REST API for AI book recommendations.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from datetime import datetime

# Import engine
from ..engine import (
    RecommendationEngine,
    RecommendationRequest,
    RecommendationResponse,
    get_engine,
)

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class RecommendRequest(BaseModel):
    """API request for recommendations."""
    
    user_id: int = Field(..., description="User identifier")
    num_recommendations: int = Field(10, ge=1, le=100)
    exclude_ids: List[str] = Field(default_factory=list)
    category_filter: Optional[str] = None
    include_explanations: bool = True
    diversity_factor: float = Field(0.2, ge=0, le=1)
    exploration_rate: float = Field(0.1, ge=0, le=0.5)


class BookScore(BaseModel):
    """Single book recommendation."""
    
    book_id: str
    score: float
    rank: int
    explanation: Optional[str] = None
    explanation_type: Optional[str] = None
    sources: Dict[str, float] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    """API response with recommendations."""
    
    user_id: int
    recommendations: List[BookScore]
    latency_ms: float
    request_id: str = ""
    model_version: str = "1.0.0"


class FeedbackRequest(BaseModel):
    """Request to record feedback."""
    
    user_id: int
    item_id: str
    feedback_type: str = Field(..., pattern="^(click|view|rate|skip|save|purchase|search|recommend|dwell|favorite|later|finished)$")
    value: float = 1.0


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: str


class StatsResponse(BaseModel):
    """Engine statistics response."""
    
    initialized: bool
    metrics: Dict[str, Any]
    exploration: Dict[str, Any]


# ==================== FastAPI App ====================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="AI Book Recommender API",
        description="Production-grade hybrid AI book recommendation service",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==================== Endpoints ====================
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check service health."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
    async def get_recommendations(request: RecommendRequest):
        """
        Get personalized book recommendations for a user.
        
        The recommendation pipeline includes:
        1. User feature extraction
        2. Candidate retrieval (hybrid search)
        3. Neural ranking
        4. Re-ranking with context
        5. Diversity/exploration adjustment
        6. Explanation generation
        """
        try:
            engine = get_engine()
            
            # Convert to internal request
            internal_request = RecommendationRequest(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                exclude_ids=request.exclude_ids,
                category_filter=request.category_filter,
                include_explanations=request.include_explanations,
                diversity_factor=request.diversity_factor,
                exploration_rate=request.exploration_rate,
            )
            
            # Get recommendations
            response = engine.recommend(internal_request)
            
            # Convert to API response
            recommendations = []
            for i, rec in enumerate(response.recommendations):
                exp = response.explanations[i] if i < len(response.explanations) else None
                
                recommendations.append(BookScore(
                    book_id=rec["book_id"],
                    score=rec["score"],
                    rank=rec["rank"],
                    explanation=exp.primary_reason if exp else None,
                    explanation_type=exp.explanation_type.value if exp else None,
                    sources=rec.get("sources", {}),
                ))
            
            return RecommendResponse(
                user_id=response.user_id,
                recommendations=recommendations,
                latency_ms=response.latency_ms,
                request_id=response.request_id,
                model_version=response.model_version,
            )
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["Recommendations"])
    async def get_recommendations_simple(
        user_id: int,
        k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
        include_explanations: bool = Query(True)
    ):
        """Simple GET endpoint for recommendations."""
        request = RecommendRequest(
            user_id=user_id,
            num_recommendations=k,
            include_explanations=include_explanations
        )
        return await get_recommendations(request)
    
    @app.post("/feedback", tags=["Feedback"])
    async def record_feedback(request: FeedbackRequest):
        """
        Record user feedback for online learning.
        
        Feedback types:
        - click: User clicked on item
        - view: User viewed item details
        - rate: User rated the item (value = rating)
        - skip: User skipped/dismissed the item
        - save: User saved/bookmarked the item
        - purchase: User purchased/borrowed the item
        """
        try:
            engine = get_engine()
            engine.record_feedback(
                user_id=request.user_id,
                item_id=request.item_id,
                feedback_type=request.feedback_type,
                value=request.value
            )
            return {"status": "recorded", "item_id": request.item_id}
            
        except Exception as e:
            logger.error(f"Feedback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats", response_model=StatsResponse, tags=["Admin"])
    async def get_stats():
        """Get engine statistics and metrics."""
        engine = get_engine()
        stats = engine.get_stats()
        return StatsResponse(**stats)
    
    @app.post("/initialize", tags=["Admin"])
    async def initialize_engine():
        """Initialize or reinitialize the recommendation engine."""
        try:
            engine = get_engine()
            engine.initialize()
            return {"status": "initialized"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default app instance for dev/testing
if __name__ == "__main__":
    app = create_app()
else:
    # We provide a factory or the user can call create_app()
    # But for unified_server.py, we might still need a module-level 'app'
    # if it's imported as 'engine_app'.
    # To keep it safe, we'll only create it if NOT in a reload/import loop that we want to avoid.
    app = create_app() 


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
