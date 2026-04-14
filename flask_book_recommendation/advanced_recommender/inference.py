import torch
import numpy as np
import os
from .neural_model import TwoTowerModel

class DLInferenceEngine:
    def __init__(self, model_path="instance/models/two_tower_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel()
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.is_ready = True
            except Exception as e:
                print(f"[DL-Inference] Error loading model: {e}")
                self.is_ready = False
        else:
            print(f"[DL-Inference] Warning: Model file {model_path} not found.")
            self.is_ready = False
            
    def get_hybrid_score(self, two_tower_score, semantic_score, popularity_score, diversity_score):
        """
        Implements the User's requested Hybrid Formula:
         0.55 × TwoTowerSimilarity
        + 0.25 × SemanticEmbeddingSimilarity
        + 0.10 × Popularity
        + 0.10 × Diversity
        """
        # Ensure only positive boost for saved items (Online Loop check required externally or passed here)
        return (0.55 * two_tower_score) + \
               (0.25 * semantic_score) + \
               (0.10 * popularity_score) + \
               (0.10 * diversity_score)

    def predict(self, user_id, history_vectors, interest_vector, candidate_books_features):
        """
        user_id: int
        history_vectors: np.array (10, 384)
        interest_vector: np.array (384,)
        candidate_books_features: dict {book_id: vector (384,)}
        """
        if not self.is_ready:
            return {}
            
        # Prepare User Input
        safe_user_id = user_id if user_id is not None else 0
        u_id_tensor = torch.tensor([safe_user_id], dtype=torch.long).to(self.device)
        hist_tensor = torch.tensor([history_vectors], dtype=torch.float32).to(self.device)
        int_tensor = torch.tensor([interest_vector], dtype=torch.float32).to(self.device)
        
        # Prepare Candidates (Batch processing)
        if not candidate_books_features:
            return {}
            
        book_ids = list(candidate_books_features.keys())
        vectors = np.array(list(candidate_books_features.values()))
        book_tensor = torch.tensor(vectors, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # 1. Get User Embedding once
            user_input = (u_id_tensor, hist_tensor, int_tensor)
            # We call user_tower directly to avoid re-computing it for every item
            user_emb = self.model.user_tower(u_id_tensor, hist_tensor, int_tensor) # (1, emb_dim)
            
            # 2. Get Item Embeddings
            item_embs = self.model.item_tower(book_tensor) # (N, emb_dim)
            
            # 3. Calculate Scores (Dot Product)
            # Because vectors are normalized in the model, this is Cosine Similarity
            scores = (user_emb @ item_embs.T).squeeze(0) # (N,)
            scores = scores.cpu().numpy()
            
        return {bid: score for bid, score in zip(book_ids, scores)}

    def generate_recommendations(self, user_id, user_data, all_books_data, top_k=10):
        """
        Full pipeline wrapper.
        user_data = {'history': ..., 'interests': ...}
        all_books_data = [{'id': 1, 'vector': ..., 'popularity': 0.5}, ...]
        """
        # 1. Two Tower Scores
        candidate_vectors = {b['id']: b['vector'] for b in all_books_data if b.get('vector') is not None}
        dl_scores = self.predict(
            user_id, 
            user_data.get('history', np.zeros((10, 384))), 
            user_data.get('interests', np.zeros(384)),
            candidate_vectors
        )
        
        # 2. Hybrid Ranking
        ranked_books = []
        for book in all_books_data:
            book_id = book['id']
            tt_score = dl_scores.get(book_id, 0.0)
            sem_score = book.get('semantic_score', 0.0) # Pre-calculated or separate
            pop_score = book.get('popularity', 0.0)
            div_score = book.get('diversity_score', 0.0) # Calculated relative to list so far? Or static?
            
            final_score = self.get_hybrid_score(tt_score, sem_score, pop_score, div_score)
            
            # Boost/Penalize (Feedback Loop)
            if book.get('is_saved'):
                final_score *= 1.2
            if book.get('is_ignored'):
                final_score *= 0.5
                
            ranked_books.append({
                'id': book_id,
                'final_score': final_score,
                'reasons': {
                    'AI Match': f"{tt_score:.2f}", 
                    'Hybrid': f"{final_score:.2f}"
                }
            })
            
        # Sort
        ranked_books.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked_books[:top_k]
