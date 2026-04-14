# flask_book_recommendation/recommendation_logger.py
"""
[REC] Recommendation Pipeline Logger
=================================
Comprehensive logging system for tracking AI recommendation execution.
Logs each stage of the pipeline with timing, results, and diagnostic info.
"""

import logging
import time
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure recommendation-specific logger
rec_logger = logging.getLogger("recommendation_pipeline")
rec_logger.setLevel(logging.DEBUG)
rec_logger.propagate = False  # Don't send to root logger (console)

# Console handler with colored output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show errors/warnings in console; details go to log file
console_format = logging.Formatter(
    '\033[36m[REC]\033[0m %(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_format)

# File handler for detailed logs
file_handler = logging.FileHandler(LOGS_DIR / "recommendations.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    '[%(asctime)s] %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_format)

rec_logger.addHandler(console_handler)
rec_logger.addHandler(file_handler)


@dataclass
class RecommendationTrace:
    """
    Tracks metadata for each recommendation result.
    Returned alongside book data for transparency.
    """
    algorithm: str = "Unknown"
    model_version: str = "v1.0"
    score: float = 0.0
    rank: int = 0
    features_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    is_fallback: bool = False
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineExecutionLog:
    """
    Complete execution log for a recommendation request.
    """
    request_id: str = ""
    user_id: int = 0
    timestamp: str = ""
    
    # Stage execution flags
    transformer_invoked: bool = False
    neural_invoked: bool = False
    behavioral_invoked: bool = False
    hybrid_merge_performed: bool = False
    
    # Timing (milliseconds)
    transformer_time_ms: float = 0.0
    neural_time_ms: float = 0.0
    behavioral_time_ms: float = 0.0
    hybrid_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Result counts
    transformer_results: int = 0
    neural_results: int = 0
    behavioral_results: int = 0
    final_results: int = 0
    
    # Merge weights
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Fallback info
    used_fallback: bool = False
    fallback_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def log_summary(self) -> str:
        """Generate formatted log summary."""
        lines = [
            f"===========================================================",
            f"[REC] RECOMMENDATION REQUEST: {self.request_id}",
            f"   User ID: {self.user_id} | Time: {self.timestamp}",
            f"-----------------------------------------------------------",
        ]
        
        # Transformer
        status = "Y" if self.transformer_invoked else "N"
        lines.append(
            f"   |-- [TRANSFORMER]  Invoked: {status} | "
            f"Time: {self.transformer_time_ms:.0f}ms | "
            f"Results: {self.transformer_results}"
        )
        
        # Neural
        status = "Y" if self.neural_invoked else "N"
        lines.append(
            f"   |-- [NEURAL]       Invoked: {status} | "
            f"Time: {self.neural_time_ms:.0f}ms | "
            f"Results: {self.neural_results}"
        )
        
        # Behavioral
        status = "Y" if self.behavioral_invoked else "N"
        lines.append(
            f"   |-- [BEHAVIORAL]   Invoked: {status} | "
            f"Time: {self.behavioral_time_ms:.0f}ms | "
            f"Results: {self.behavioral_results}"
        )
        
        # Hybrid merge
        if self.hybrid_merge_performed:
            weights_str = ", ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
            lines.append(
                f"   |-- [HYBRID]       Merge: Y | "
                f"Time: {self.hybrid_time_ms:.0f}ms | "
                f"Weights: [{weights_str}]"
            )
        
        # Fallback warning
        if self.used_fallback:
            lines.append(f"   [WARN] FALLBACK USED: {self.fallback_reason}")
        
        # Errors
        for err in self.errors:
            lines.append(f"   [ERROR] ERROR: {err}")
        
        # Summary
        lines.append(f"-----------------------------------------------------------")
        lines.append(
            f"   |__ [TOTAL] {self.total_time_ms:.0f}ms | "
            f"Final Results: {self.final_results} books"
        )
        lines.append(f"===========================================================")
        
        return "\n".join(lines)


class RecommendationPipelineLogger:
    """
    Context manager for logging recommendation pipeline execution.
    """
    
    _current_log: Optional[PipelineExecutionLog] = None
    _request_counter: int = 0
    
    def __init__(self, user_id: int):
        RecommendationPipelineLogger._request_counter += 1
        self.log = PipelineExecutionLog(
            request_id=f"REQ-{RecommendationPipelineLogger._request_counter:06d}",
            user_id=user_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self._start_time = None
        
    def __enter__(self):
        RecommendationPipelineLogger._current_log = self.log
        self._start_time = time.perf_counter()
        rec_logger.info(f"[REC] Starting recommendation for user {self.log.user_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.total_time_ms = (time.perf_counter() - self._start_time) * 1000
        
        # Log summary
        summary = self.log.log_summary()
        rec_logger.info(f"\n{summary}")
        
        # Also log to file as JSON for analysis
        rec_logger.debug(f"EXECUTION_LOG_JSON: {json.dumps(self.log.to_dict())}")
        
        RecommendationPipelineLogger._current_log = None
        return False  # Don't suppress exceptions
    
    @classmethod
    def get_current(cls) -> Optional[PipelineExecutionLog]:
        """Get the current active execution log."""
        return cls._current_log
    
    def log_stage(self, stage: str, **kwargs):
        """Log a specific stage execution."""
        if stage == "transformer":
            self.log.transformer_invoked = True
            self.log.transformer_time_ms = kwargs.get("time_ms", 0)
            self.log.transformer_results = kwargs.get("results", 0)
        elif stage == "neural":
            self.log.neural_invoked = True
            self.log.neural_time_ms = kwargs.get("time_ms", 0)
            self.log.neural_results = kwargs.get("results", 0)
        elif stage == "behavioral":
            self.log.behavioral_invoked = True
            self.log.behavioral_time_ms = kwargs.get("time_ms", 0)
            self.log.behavioral_results = kwargs.get("results", 0)
        elif stage == "hybrid":
            self.log.hybrid_merge_performed = True
            self.log.hybrid_time_ms = kwargs.get("time_ms", 0)
            self.log.weights = kwargs.get("weights", {})
    
    def log_fallback(self, reason: str):
        """Log that fallback was used."""
        self.log.used_fallback = True
        self.log.fallback_reason = reason
        rec_logger.warning(f"[WARN] Fallback triggered: {reason}")
    
    def log_error(self, error: str):
        """Log an error."""
        self.log.errors.append(error)
        rec_logger.error(f"[ERROR] {error}")
    
    def set_final_count(self, count: int):
        """Set final result count."""
        self.log.final_results = count


def timed_stage(stage_name: str):
    """
    Decorator to time and log a recommendation stage.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Log to current pipeline if active
            current_log = RecommendationPipelineLogger.get_current()
            if current_log:
                result_count = len(result) if isinstance(result, (list, tuple)) else 0
                
                if stage_name == "transformer":
                    current_log.transformer_invoked = True
                    current_log.transformer_time_ms = elapsed_ms
                    current_log.transformer_results = result_count
                elif stage_name == "neural":
                    current_log.neural_invoked = True
                    current_log.neural_time_ms = elapsed_ms
                    current_log.neural_results = result_count
                elif stage_name == "behavioral":
                    current_log.behavioral_invoked = True
                    current_log.behavioral_time_ms = elapsed_ms
                    current_log.behavioral_results = result_count
            
            rec_logger.debug(
                f"Stage [{stage_name}] completed in {elapsed_ms:.1f}ms with {result_count if isinstance(result, (list, tuple)) else 'N/A'} results"
            )
            
            return result
        return wrapper
    return decorator


# Validation helpers
class RecommendationValidationError(Exception):
    """Raised when recommendation validation fails."""
    pass


def validate_embedding(vector, context: str = ""):
    """Validate that an embedding vector is valid."""
    if vector is None:
        raise RecommendationValidationError(f"Embedding vector is None {context}")
    if hasattr(vector, '__len__') and len(vector) == 0:
        raise RecommendationValidationError(f"Embedding vector is empty {context}")
    rec_logger.debug(f"✓ Embedding validated: dim={len(vector) if hasattr(vector, '__len__') else 'scalar'} {context}")


def validate_similarity_score(score: float, context: str = ""):
    """Validate that a similarity score is valid."""
    if score is None:
        raise RecommendationValidationError(f"Similarity score is None {context}")
    if score < 0:
        rec_logger.warning(f"Similarity score is negative: {score} {context}")
    rec_logger.debug(f"Similarity score validated: {score:.4f} {context}")


def validate_user_features(features: dict, context: str = ""):
    """Validate that user features are present."""
    if not features:
        raise RecommendationValidationError(f"User features are empty {context}")
    rec_logger.debug(f"✓ User features validated: {list(features.keys())} {context}")


# Export all
__all__ = [
    'rec_logger',
    'RecommendationTrace',
    'PipelineExecutionLog', 
    'RecommendationPipelineLogger',
    'timed_stage',
    'validate_embedding',
    'validate_similarity_score',
    'validate_user_features',
    'RecommendationValidationError',
]
