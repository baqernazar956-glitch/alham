import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from datetime import datetime, timedelta

class BookInteractionDataset(Dataset):
    def __init__(self, interactions, user_features, book_features, neg_ratio=4):
        """
        interactions: List of (user_id, book_id, timestamp, weight)
        user_features: Dict mapping user_id -> (history_vectors, interest_matrix)
        book_features: Dict mapping book_id -> vector (768 dim)
        neg_ratio: Number of negative samples per positive sample
        """
        self.interactions = interactions
        self.user_features = user_features
        self.book_features = book_features
        self.neg_ratio = neg_ratio
        self.all_book_ids = list(book_features.keys())
        
        # Pre-filter interactions to apply Temporal Weighting logic during data prep
        # (Though simpler to do it in the loop or loss, we structure it here)
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, pos_book_id, timestamp, base_weight = self.interactions[idx]
        
        # 1. Temporal Weighting Calculation
        # 70% value for recent (last 30 days), 30% for older
        days_diff = (datetime.now() - timestamp).days
        time_weight = 1.0
        if days_diff <= 30:
            time_weight = 1.0 # High relevance
        else:
            time_weight = 0.3 + (0.7 * np.exp(-days_diff / 365)) # Decay
            
        final_weight = base_weight * time_weight
        
        # 2. User Features
        # Ensure we have valid tensors
        if user_id in self.user_features:
            hist_vecs, int_vecs = self.user_features[user_id]
        else:
            # Fallback for empty/new user
            hist_vecs = np.zeros((10, 768), dtype=np.float32) # Max Seq Length 10
            int_vecs = np.zeros((768,), dtype=np.float32)
            
        # 3. Positive Item Features
        pos_vec = self.book_features.get(pos_book_id, np.zeros((768,), dtype=np.float32))
        
        # 4. Negative Sampling
        # Random book not in interactions (simplified for efficiency)
        neg_book_id = random.choice(self.all_book_ids)
        while neg_book_id == pos_book_id: # Retry if same
            neg_book_id = random.choice(self.all_book_ids)
            
        neg_vec = self.book_features.get(neg_book_id, np.zeros((768,), dtype=np.float32))
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'history': torch.tensor(hist_vecs, dtype=torch.float32),
            'interests': torch.tensor(int_vecs, dtype=torch.float32),
            'pos_item': torch.tensor(pos_vec, dtype=torch.float32),
            'neg_item': torch.tensor(neg_vec, dtype=torch.float32),
            'weight': torch.tensor(final_weight, dtype=torch.float32)
        }

def get_dummy_data_loader(num_users=100, num_books=500):
    """
    Generate synthetic data for executing/testing the code immediately.
    """
    user_ids = range(1, num_users + 1)
    book_ids = range(1, num_books + 1)
    
    # Mock Embeddings (BERT style)
    book_features = {bid: np.random.randn(384).astype(np.float32) for bid in book_ids}
    
    # Mock User Interactions
    interactions = []
    user_features = {}
    
    for uid in user_ids:
        # Mock History: Sequence of 10 book vectors
        hist = np.random.randn(10, 384).astype(np.float32)
        # Mock Interests
        interests = np.random.randn(384).astype(np.float32)
        user_features[uid] = (hist, interests)
        
        # Create 20 interactions per user
        for _ in range(20):
            bid = random.choice(book_ids)
            # Random timestamp within last year
            days_ago = random.randint(0, 365)
            ts = datetime.now() - timedelta(days=days_ago)
            weight = 1.0 # Base interaction weight
            interactions.append((uid, bid, ts, weight))
            
    dataset = BookInteractionDataset(interactions, user_features, book_features)
    return DataLoader(dataset, batch_size=32, shuffle=True)
