import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(basedir)

from flask_book_recommendation.app import create_app
from flask_book_recommendation.advanced_recommender.neural_model import TwoTowerModel
from scripts.train_from_database import get_db_data, BPRDataset

def evaluate(model, dataset, test_indices, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    hits = 0
    ndcg = 0.0
    mrr = 0.0
    total_queries = 0
    
    all_books = dataset.all_book_ids
    
    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Evaluating Metrics"):
            sample = dataset[idx]
            uid = sample['user_id']
            pos_emb = sample['pos_item']
            hist_vec = sample['history']
            int_vec = sample['interest']
            
            # Evaluate by ranking positive item against 99 random negatives
            neg_bids = random.sample(all_books, min(99, len(all_books)))
            neg_embs = [dataset.embeddings[b] for b in neg_bids]
            
            candidates = [pos_emb] + neg_embs
            cand_tensor = torch.tensor(np.array(candidates), dtype=torch.float32).to(device)
            
            u_id_t = torch.tensor([uid], dtype=torch.long).clamp(0, 9999).to(device)
            hist_t = torch.tensor([hist_vec], dtype=torch.float32).to(device)
            int_t = torch.tensor([int_vec], dtype=torch.float32).to(device)
            
            u_emb = model.user_tower(u_id_t, hist_t, int_t) # (1, 128)
            i_embs = model.item_tower(cand_tensor) # (N, 128)
            
            scores = (u_emb * i_embs).sum(dim=1).cpu().numpy()
            
            # The target (pos item) is at index 0
            ranked_indices = np.argsort(scores)[::-1]
            target_rank = np.where(ranked_indices == 0)[0][0]
            
            if target_rank < k:
                hits += 1
                ndcg += 1.0 / np.log2(target_rank + 2)
            
            mrr += 1.0 / (target_rank + 1)
            total_queries += 1
            
    hit_rate = hits / total_queries
    ndcg /= total_queries
    mrr /= total_queries
    
    print(f"Hit Rate@{k}: {hit_rate:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"MRR: {mrr:.4f}")
    return hit_rate, ndcg, mrr

def compute_coverage(model, dataset, test_indices, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_books = dataset.all_book_ids
    all_embs = torch.tensor(np.array([dataset.embeddings[b] for b in all_books]), dtype=torch.float32).to(device)
    
    recommended_items = set()
    
    # Subsample users for speed
    unique_test_users = list(set([dataset[i]['user_id'] for i in test_indices]))
    sample_users = unique_test_users[:min(200, len(unique_test_users))]
    
    with torch.no_grad():
        i_embs = model.item_tower(all_embs)
        
        for uid in tqdm(sample_users, desc="Evaluating Coverage"):
            hist_vec, int_vec = dataset.user_profiles[uid]
            
            u_id_t = torch.tensor([uid], dtype=torch.long).clamp(0, 9999).to(device)
            hist_t = torch.tensor([hist_vec], dtype=torch.float32).to(device)
            int_t = torch.tensor([int_vec], dtype=torch.float32).to(device)
            
            u_emb = model.user_tower(u_id_t, hist_t, int_t)
            
            scores = (u_emb * i_embs).sum(dim=1).cpu().numpy()
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_k_indices:
                recommended_items.add(all_books[idx])
                
    coverage = len(recommended_items) / len(all_books) * 100.0
    print(f"Coverage@{k}: {coverage:.2f}% ({len(recommended_items)} unique items recommended)")

def main():
    metadata_path = "instance/models/test_set_metadata.pt"
    if not os.path.exists(metadata_path):
        print("No test metadata found. Run train_from_database.py first to generate it.")
        return
        
    meta = torch.load(metadata_path, map_location='cpu')
    test_indices = meta['test_indices']
    
    user_inter, embeddings = get_db_data()
    dataset = BPRDataset(user_inter, embeddings, num_negative=1)
    
    print("\n" + "="*40)
    print("--- Evaluating Untrained Baseline ---")
    model_untrained = TwoTowerModel()
    evaluate(model_untrained, dataset, test_indices)
    compute_coverage(model_untrained, dataset, test_indices)
    
    trained_model_path = "instance/models/two_tower_model.pt"
    if os.path.exists(trained_model_path):
        print("\n" + "="*40)
        print("--- Evaluating Trained Model ---")
        model_trained = TwoTowerModel()
        model_trained.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
        evaluate(model_trained, dataset, test_indices)
        compute_coverage(model_trained, dataset, test_indices)
    else:
        print("\nTrained model not found. Run train_from_database.py first.")

if __name__ == "__main__":
    main()
