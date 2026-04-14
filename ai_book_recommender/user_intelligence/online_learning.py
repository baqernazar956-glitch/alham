# -*- coding: utf-8 -*-
"""
🔄 Online Learning
===================

Real-time model updates from user feedback.
Implements multi-armed bandit for exploration/exploitation.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Single feedback event."""
    
    user_id: int
    item_id: str
    feedback_type: str  # click, rate, skip, dwell
    value: float  # 1.0 for click, rating value, 0 for skip, seconds for dwell
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class FeedbackProcessor:
    """
    📬 Feedback Processor
    
    Processes and aggregates user feedback for model updates.
    
    Features:
    - Implicit feedback inference
    - Decay-weighted aggregation
    - Batch processing for efficiency
    """
    
    def __init__(
        self,
        positive_threshold: float = 0.5,
        dwell_threshold_seconds: float = 30.0,
        batch_size: int = 100,
        decay_rate: float = 0.99
    ):
        self.positive_threshold = positive_threshold
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        
        # Feedback buffers
        self._event_buffer: List[FeedbackEvent] = []
        self._user_item_feedback: Dict[Tuple[int, str], List[FeedbackEvent]] = defaultdict(list)
        
        # Aggregated signals
        self._item_signals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "total": 0}
        )
        
        self._lock = threading.Lock()
        
        logger.info(
            f"FeedbackProcessor: pos_threshold={positive_threshold}, "
            f"dwell_threshold={dwell_threshold_seconds}s"
        )
    
    def add_feedback(self, event: FeedbackEvent) -> None:
        """Add a feedback event."""
        with self._lock:
            self._event_buffer.append(event)
            self._user_item_feedback[(event.user_id, event.item_id)].append(event)
            
            # Update item signals
            signal = self._compute_signal(event)
            self._item_signals[event.item_id]["total"] += 1
            if signal > 0:
                self._item_signals[event.item_id]["positive"] += signal
            else:
                self._item_signals[event.item_id]["negative"] += abs(signal)
            
            # Process batch if needed
            if len(self._event_buffer) >= self.batch_size:
                self._process_batch()
    
    def _compute_signal(self, event: FeedbackEvent) -> float:
        """Convert event to positive/negative signal."""
        if event.feedback_type == "click":
            return 1.0
        
        elif event.feedback_type == "rate":
            # Normalize rating to [-1, 1]
            return (event.value - 3.0) / 2.0
        
        elif event.feedback_type == "dwell":
            # Positive if above threshold
            if event.value >= self.dwell_threshold_seconds:
                return min(event.value / self.dwell_threshold_seconds, 2.0) - 1.0
            else:
                return -0.5
        
        elif event.feedback_type == "skip":
            return -1.0
            
        elif event.feedback_type == "view":
            return 0.5
            
        elif event.feedback_type == "recommend":
            return 0.05
        
        elif event.feedback_type == "search":
            return 0.2  # Higher than recommend, as it reflects explicit intent
        
        elif event.feedback_type in ["save", "favorite", "later"]:
            return 1.5
            
        elif event.feedback_type == "finished":
            return 2.0
        
        elif event.feedback_type == "purchase":
            return 2.0
        
        return 0.0
    
    def _process_batch(self) -> None:
        """Process buffered events."""
        # Clear buffer
        events = self._event_buffer
        self._event_buffer = []
        
        # Here you would trigger model updates
        logger.debug(f"Processed batch of {len(events)} feedback events")
    
    def get_item_score_adjustment(self, item_id: str) -> float:
        """
        Get score adjustment based on feedback.
        
        Returns value in [-1, 1] that can be used to adjust scores.
        """
        with self._lock:
            signals = self._item_signals.get(item_id)
            if not signals or signals["total"] == 0:
                return 0.0
            
            pos = signals["positive"]
            neg = signals["negative"]
            total = signals["total"]
            
            # Wilson score lower bound for positive rate
            if pos + neg == 0:
                return 0.0
            
            n = pos + neg
            phat = pos / n
            z = 1.96  # 95% confidence
            
            # Wilson score
            score = (phat + z*z/(2*n) - z*np.sqrt((phat*(1-phat) + z*z/(4*n))/n)) / (1 + z*z/n)
            
            # Map to [-1, 1]
            return (score - 0.5) * 2
    
    def get_training_pairs(
        self,
        min_events: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Get training data from feedback.
        
        Returns list of (user_id, item_id, label) tuples.
        """
        pairs = []
        
        with self._lock:
            for (user_id, item_id), events in self._user_item_feedback.items():
                if len(events) < min_events:
                    continue
                
                # Aggregate signals
                total_signal = sum(self._compute_signal(e) for e in events)
                label = 1.0 if total_signal > self.positive_threshold else 0.0
                
                pairs.append((user_id, item_id, label))
        
        return pairs


class OnlineLearner:
    """
    🔄 Online Learning System
    
    Implements online model updates with:
    - Learning rate scheduling
    - Gradient accumulation
    - Exploration/exploitation via epsilon-greedy
    
    Also implements multi-armed bandit for cold-start exploration.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.999,
        min_exploration: float = 0.01,
        update_interval_seconds: int = 60
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.update_interval = update_interval_seconds
        
        # Feedback processor
        self.feedback_processor = FeedbackProcessor()
        
        # Bandit arms (item_id -> (trials, successes))
        self._arms: Dict[str, Tuple[int, float]] = {}
        
        # Update tracking
        self._last_update = datetime.now()
        self._update_count = 0
        
        logger.info(
            f"OnlineLearner: lr={learning_rate}, explore={exploration_rate}"
        )
    
    def record_feedback(
        self,
        user_id: int,
        item_id: str,
        feedback_type: str,
        value: float,
        context: Optional[Dict] = None
    ) -> None:
        """Record user feedback."""
        event = FeedbackEvent(
            user_id=user_id,
            item_id=item_id,
            feedback_type=feedback_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.feedback_processor.add_feedback(event)
        
        # Update bandit
        self._update_bandit(item_id, feedback_type, value)
    
    def _update_bandit(
        self,
        item_id: str,
        feedback_type: str,
        value: float
    ) -> None:
        """Update multi-armed bandit arms."""
        if item_id not in self._arms:
            self._arms[item_id] = (0, 0.0)
        
        trials, successes = self._arms[item_id]
        
        # We don't want a background "recommend" or "search" event to count as a full user trial failure
        if feedback_type in ["recommend", "search"]:
            return
            
        trials += 1
        
        # Convert feedback to reward
        reward = 0.0
        if feedback_type == "click" or feedback_type == "view":
            reward = 1.0
        elif feedback_type == "rate" and value >= 4.0:
            reward = 1.0
        elif feedback_type == "dwell" and value >= 30:
            reward = 0.5
        elif feedback_type in ["save", "purchase", "favorite", "finished", "later"]:
            reward = 1.0
        
        successes += reward
        self._arms[item_id] = (trials, successes)
    
    def select_items(
        self,
        candidate_ids: List[str],
        scores: np.ndarray,
        k: int = 10,
        strategy: str = "epsilon_greedy"
    ) -> List[str]:
        """
        Select items with exploration/exploitation.
        
        Args:
            candidate_ids: List of candidate item IDs
            scores: Predicted scores for candidates
            k: Number of items to select
            strategy: "epsilon_greedy", "thompson", or "ucb"
            
        Returns:
            Selected item IDs
        """
        if strategy == "epsilon_greedy":
            return self._epsilon_greedy_select(candidate_ids, scores, k)
        elif strategy == "thompson":
            return self._thompson_select(candidate_ids, k)
        elif strategy == "ucb":
            return self._ucb_select(candidate_ids, scores, k)
        else:
            # Default to pure exploitation
            top_indices = np.argsort(-scores)[:k]
            return [candidate_ids[i] for i in top_indices]
    
    def _epsilon_greedy_select(
        self,
        candidate_ids: List[str],
        scores: np.ndarray,
        k: int
    ) -> List[str]:
        """Epsilon-greedy selection."""
        selected = []
        available = list(range(len(candidate_ids)))
        
        for _ in range(min(k, len(available))):
            if np.random.random() < self.exploration_rate:
                # Explore: random selection
                idx = np.random.choice(available)
            else:
                # Exploit: best score
                best_idx = max(available, key=lambda i: scores[i])
                idx = best_idx
            
            selected.append(candidate_ids[idx])
            available.remove(idx)
        
        return selected
    
    def _thompson_select(
        self,
        candidate_ids: List[str],
        k: int
    ) -> List[str]:
        """Thompson sampling selection."""
        samples = []
        
        for item_id in candidate_ids:
            if item_id in self._arms:
                trials, successes = self._arms[item_id]
                # Beta distribution sampling
                alpha = successes + 1
                beta = trials - successes + 1
                sample = np.random.beta(alpha, beta)
            else:
                # No data - use uniform prior
                sample = np.random.beta(1, 1)
            
            samples.append((item_id, sample))
        
        # Sort by sample value
        samples.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in samples[:k]]
    
    def _ucb_select(
        self,
        candidate_ids: List[str],
        scores: np.ndarray,
        k: int
    ) -> List[str]:
        """Upper Confidence Bound selection."""
        total_trials = sum(t for t, _ in self._arms.values()) + 1
        
        ucb_scores = []
        
        for i, item_id in enumerate(candidate_ids):
            if item_id in self._arms:
                trials, successes = self._arms[item_id]
                mean = successes / max(trials, 1)
                confidence = np.sqrt(2 * np.log(total_trials) / max(trials, 1))
                ucb = mean + confidence
            else:
                # Uninitiated item gets high UCB
                ucb = 1.0 + np.random.random() * 0.1
            
            # Combine with predicted score
            combined = 0.7 * scores[i] + 0.3 * ucb
            ucb_scores.append(combined)
        
        top_indices = np.argsort(-np.array(ucb_scores))[:k]
        return [candidate_ids[i] for i in top_indices]
    
    def decay_exploration(self) -> None:
        """Decay exploration rate."""
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
    
    def should_update_model(self) -> bool:
        """Check if it's time for a model update."""
        elapsed = (datetime.now() - self._last_update).total_seconds()
        return elapsed >= self.update_interval
    
    def mark_update_complete(self) -> None:
        """Mark that a model update was completed."""
        self._last_update = datetime.now()
        self._update_count += 1
        self.decay_exploration()
        
        logger.info(
            f"Model update #{self._update_count} complete. "
            f"Exploration rate: {self.exploration_rate:.4f}"
        )
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        total_arms = len(self._arms)
        explored_arms = sum(1 for t, _ in self._arms.values() if t > 0)
        avg_trials = np.mean([t for t, _ in self._arms.values()]) if self._arms else 0
        
        return {
            "exploration_rate": self.exploration_rate,
            "total_arms": total_arms,
            "explored_arms": explored_arms,
            "avg_trials_per_arm": avg_trials,
            "update_count": self._update_count
        }
