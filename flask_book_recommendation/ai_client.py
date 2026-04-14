import requests
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AIClient:
    def __init__(self, base_url="http://localhost:5000/api/engine"):
        self.base_url = base_url
        self.session = requests.Session()
        self._local_scorer = None
    
    @property
    def local_scorer(self):
        """Lazy-load local scorer."""
        if self._local_scorer is None:
            try:
                from .local_ai_scorer import get_local_scorer
                self._local_scorer = get_local_scorer()
            except Exception as e:
                logger.warning(f"Could not load local scorer: {e}")
        return self._local_scorer

    def get_recommendations(self, user_id, history_texts=None, interest_texts=None, k=10):
        """
        Call the Two-Tower Neural Recommender.
        Falls back to local scorer if external API is unavailable.
        """
        payload = {
            "user_id": user_id,
            "history_texts": history_texts or [],
            "interest_texts": interest_texts or [],
            "k": k
        }
        
        try:
            resp = self.session.post(f"{self.base_url}/recommend", json=payload, timeout=2.0)
            resp.raise_for_status()
            return resp.json().get("recommendations", [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"AI Engine unreachable (Recommend): {e}")
            return None # Indicate fallback needed
    
    def score_with_local_model(self, user_embedding, item_embeddings):
        """
        Score items using local trained model.
        
        Args:
            user_embedding: User embedding (numpy array)
            item_embeddings: Dict of item_id -> embedding
            
        Returns:
            List of (item_id, score) tuples sorted by score
        """
        if self.local_scorer is None:
            return []
        
        try:
            return self.local_scorer.rank_items(user_embedding, item_embeddings)
        except Exception as e:
            logger.error(f"Local scoring failed: {e}")
            return []

    def semantic_search(self, query, k=10):
        """
        Call Semantic Search.
        """
        try:
            resp = self.session.post(f"{self.base_url}/search", json={"query": query, "k": k}, timeout=2.0)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.warning(f"AI Engine unreachable (Search): {e}")
            return None

    def send_feedback(self, user_id, book_id, event_type, value):
        """
        Fire-and-forget RL feedback.
        """
        payload = {
            "user_id": user_id,
            "item_id": str(book_id),
            "feedback_type": event_type,
            "value": value
        }
        try:
            # Short timeout, we don't want to block UI for logging
            self.session.post(f"{self.base_url}/feedback", json=payload, timeout=0.5)
        except:
            pass # Ignore errors for feedback logging

    def get_health(self):
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=1.0)
            result = resp.json() if resp.ok else {"status": "error"}
        except:
            result = {"status": "offline"}
        
        # Add local scorer status
        if self.local_scorer is not None:
            result["local_scorer"] = self.local_scorer.get_health()
        else:
            result["local_scorer"] = {"status": "not_loaded"}
        
        return result

    def get_stats(self):
        try:
            resp = self.session.get(f"{self.base_url}/stats", timeout=1.0)
            return resp.json() if resp.ok else {}
        except:
            return {}

    def trigger_index_rebuild(self):
        try:
            resp = self.session.post(f"{self.base_url}/admin/build_index", timeout=1.0)
            return resp.json() if resp.ok else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Singleton
ai_client = AIClient()

