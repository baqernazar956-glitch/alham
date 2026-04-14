import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(basedir)

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import UserRatingCF, BookStatus, UserBookView, BookEmbedding, Book
from flask_book_recommendation.advanced_recommender.neural_model import TwoTowerModel

def get_db_data():
    app = create_app()
    with app.app_context():
        print("Fetching embeddings...")
        embeddings = {}
        for row in BookEmbedding.query.filter(BookEmbedding.vector.isnot(None)).all():
            vec_data = __import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector
            if vec_data is not None:
                embeddings[row.book_id] = np.array(vec_data, dtype=np.float32)
                
        # Mapping from google_id to book_id
        google_to_local = {b.google_id: b.id for b in Book.query.with_entities(Book.google_id, Book.id).all() if b.google_id}
        
        print("Fetching user interactions...")
        # user_interactions: user_id -> {book_id: weight}
        user_interactions = {}
        
        # 1. UserRatingCF
        for r in UserRatingCF.query.all():
            bid = google_to_local.get(r.google_id)
            if not bid: continue
            weight = float(r.rating) # weight=rating value
            user_dict = user_interactions.setdefault(r.user_id, {})
            user_dict[bid] = max(user_dict.get(bid, 0), weight)
            
        # 2. BookStatus
        for s in BookStatus.query.all():
            bid = s.book_id
            if s.status == 'finished': w = 5.0
            elif s.status == 'favorite': w = 4.0
            elif s.status == 'later': w = 1.5
            else: w = 1.0
            user_dict = user_interactions.setdefault(s.user_id, {})
            user_dict[bid] = max(user_dict.get(bid, 0), w)
            
        # 3. UserBookView
        for v in UserBookView.query.all():
            bid = v.book_id or google_to_local.get(v.google_id)
            if not bid: continue
            w = 2.0 if (v.view_count and v.view_count > 3) else 1.0
            user_dict = user_interactions.setdefault(v.user_id, {})
            user_dict[bid] = max(user_dict.get(bid, 0), w)
            
        return user_interactions, embeddings

class BPRDataset(Dataset):
    def __init__(self, user_interactions, embeddings, num_negative=4):
        self.samples = []
        self.embeddings = embeddings
        self.num_negative = num_negative
        
        self.all_book_ids = list(embeddings.keys())
        
        print("Building Dataset Samples...")
        self.user_profiles = {}
        for uid, interactions in user_interactions.items():
            books = list(interactions.keys())
            b_embs = [self.embeddings[b] for b in books if b in self.embeddings]
            if not b_embs: continue
            
            # History vector (pad to 10 max recent logic is simplified here)
            hist = b_embs[-10:]
            while len(hist) < 10:
                hist.append(np.zeros(384, dtype=np.float32))
            hist_vec = np.array(hist)
            
            # Interest vector (mean of available)
            int_vec = np.mean(np.array(b_embs), axis=0)
            
            self.user_profiles[uid] = (hist_vec, int_vec)
            
            for bid, weight in interactions.items():
                if bid in self.embeddings:
                    self.samples.append((uid, bid, weight))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        uid, pos_bid, weight = self.samples[idx]
        hist_vec, int_vec = self.user_profiles[uid]
        
        pos_emb = self.embeddings[pos_bid]
        
        # Sample negative items
        neg_embs = []
        for _ in range(self.num_negative):
            neg_bid = random.choice(self.all_book_ids)
            # basic ensure not positive
            while neg_bid == pos_bid:
                neg_bid = random.choice(self.all_book_ids)
            neg_embs.append(self.embeddings[neg_bid])
            
        neg_embs = np.array(neg_embs)
        
        return {
            'user_id': uid,
            'history': hist_vec,
            'interest': int_vec,
            'pos_item': pos_emb,
            'neg_items': neg_embs,
            'weight': weight
        }

def collate_fn(batch):
    uids = torch.tensor([b['user_id'] for b in batch], dtype=torch.long)
    hists = torch.tensor(np.array([b['history'] for b in batch]), dtype=torch.float32)
    ints = torch.tensor(np.array([b['interest'] for b in batch]), dtype=torch.float32)
    pos_items = torch.tensor(np.array([b['pos_item'] for b in batch]), dtype=torch.float32)
    neg_items = torch.tensor(np.array([b['neg_items'] for b in batch]), dtype=torch.float32)
    weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)
    
    return uids, hists, ints, pos_items, neg_items, weights

def train():
    user_interactions, embeddings = get_db_data()
    
    if not user_interactions or not embeddings:
        print("Not enough data to train!")
        return
        
    dataset = BPRDataset(user_interactions, embeddings)
    total = len(dataset)
    print(f"Total training samples: {total}")
    
    tr_len = int(0.8 * total)
    val_len = int(0.1 * total)
    test_len = total - tr_len - val_len
    
    from torch.utils.data import random_split
    tr_ds, val_ds, test_ds = random_split(dataset, [tr_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
    
    tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    
    model = TwoTowerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model.to(device)
    
    epochs = 10
    
    os.makedirs('instance/models', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for uids, hists, ints, pos_items, neg_items, weights in tqdm(tr_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move to device optionally, clamp uids to dict size
            uids = torch.clamp(uids, 0, 9999).to(device)
            hists = hists.to(device)
            ints = ints.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            
            user_input = (uids, hists, ints)
            u_emb = model.user_tower(*user_input)
            pos_emb = model.item_tower(pos_items)
            
            B, num_neg, dim = neg_items.shape
            neg_items_flat = neg_items.view(-1, dim)
            neg_emb_flat = model.item_tower(neg_items_flat)
            neg_emb = neg_emb_flat.view(B, num_neg, -1)
            
            pos_score = (u_emb * pos_emb).sum(dim=1)
            u_emb_expanded = u_emb.unsqueeze(1)
            neg_score = (u_emb_expanded * neg_emb).sum(dim=2)
            
            pos_score_expanded = pos_score.unsqueeze(1)
            loss_per_neg = -F.logsigmoid(pos_score_expanded - neg_score)
            
            loss = (loss_per_neg.mean(dim=1) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(tr_dl):.4f}")
        torch.save(model.state_dict(), f"instance/models/twotower_v{epoch+1}.pt")
    
    # Save final model state
    torch.save(model.state_dict(), "instance/models/two_tower_model.pt")
    
    # Save test dataset for evaluate_model.py
    torch.save({
        'test_indices': test_ds.indices,
        'user_interactions': user_interactions,
    }, "instance/models/test_set_metadata.pt")
    print("Training complete!")
    
if __name__ == "__main__":
    train()
