import torch
import torch.nn as nn
import torch.optim as optim
import os
from .neural_model import TwoTowerModel
from .data_loader import get_dummy_data_loader

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_sim, neg_sim, weights):
        # We want pos_sim to be 1, neg_sim to be -1 (or distance based)
        # Using Hinge Loss variant for Similarity
        # Maximize (pos_sim - neg_sim)
        
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return (loss * weights).mean()

class RecommenderTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = ContrastiveLoss(margin=0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            user_ids = batch['user_id'].to(self.device)
            history = batch['history'].to(self.device)
            interests = batch['interests'].to(self.device)
            pos_item = batch['pos_item'].to(self.device)
            neg_item = batch['neg_item'].to(self.device)
            weights = batch['weight'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward Pass
            user_emb, pos_item_emb = self.model((user_ids, history, interests), pos_item)
            _, neg_item_emb = self.model((user_ids, history, interests), neg_item)
            
            # Calculate Similarity
            pos_sim = (user_emb * pos_item_emb).sum(dim=1)
            neg_sim = (user_emb * neg_item_emb).sum(dim=1)
            
            loss = self.criterion(pos_sim, neg_sim, weights)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def save_model(self, path="instance/models/two_tower_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# --- Execution Entry Point (for manual run) ---
if __name__ == "__main__":
    print("Initializing Two-Tower Logic...")
    model = TwoTowerModel()
    trainer = RecommenderTrainer(model)
    dataloader = get_dummy_data_loader()
    
    print("Starting Training (1 Epoch Simulation)...")
    loss = trainer.train_epoch(dataloader)
    print(f"Epoch 1 Loss: {loss:.4f}")
    
    trainer.save_model()
