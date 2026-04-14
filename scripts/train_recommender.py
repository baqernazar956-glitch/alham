# -*- coding: utf-8 -*-
"""
🎓 AI Book Recommender - Training Demonstration
==============================================

This script demonstrates how to train the Two-Tower recommendation model
using the custom training pipeline.

Usage:
    python scripts/train_recommender.py
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_book_recommender.models.two_tower_v2 import TwoTowerV2
from ai_book_recommender.training.data_loader import InteractionSample, create_data_loaders
from ai_book_recommender.training.train import Trainer, TrainingConfig

def generate_dummy_data(num_users=100, num_items=500, num_interactions=2000):
    """Generate dummy data for demonstration."""
    print(f"Generating {num_interactions} dummy interactions...")
    interactions = []
    
    for _ in range(num_interactions):
        user_id = np.random.randint(0, num_users)
        item_id = f"book_{np.random.randint(0, num_items)}"
        label = 1.0 if np.random.random() > 0.5 else 0.0
        
        interactions.append(InteractionSample(
            user_id=user_id,
            item_id=item_id,
            label=label
        ))
    
    # Pre-computed embeddings (simulated)
    user_embeddings = {i: np.random.randn(128).astype(np.float32) for i in range(num_users)}
    item_embeddings = {f"book_{i}": np.random.randn(128).astype(np.float32) for i in range(num_items)}
    
    return interactions, user_embeddings, item_embeddings

def main():
    # 1. Setup Configuration
    # You can customize any training parameter here
    config = TrainingConfig(
        model_name="two_tower_v2_demo",
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir="instance/checkpoints",
        device="cpu" # Use "cuda" if you have a GPU
    )
    
    # 2. Prepare Data
    # In a real scenario, you would load this from your database
    interactions, user_embs, item_embs = generate_dummy_data()
    
    # Split into train/val
    split_idx = int(0.8 * len(interactions))
    train_samples = interactions[:split_idx]
    val_samples = interactions[split_idx:]
    
    # Create DataLoaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_samples,
        val_data=val_samples,
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        batch_size=config.batch_size
    )
    
    # 3. Initialize Model
    # Important: The dimensions must match your embeddings
    model = TwoTowerV2(
        user_id_dim=128,
        item_id_dim=128,
        embedding_dim=64, # The dimension we want to project to
        hidden_dim=128
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 5. Start Training
    print(f"\n--- Starting Training at {datetime.now().strftime('%H:%M:%S')} ---")
    results = trainer.train()
    print(f"--- Training Finished at {datetime.now().strftime('%H:%M:%S')} ---")
    
    # 6. Results Summary
    print("\nTraining Results:")
    print(f"Best Metric Score: {results['best_metric']:.4f}")
    print(f"Best Epoch: {results['best_epoch'] + 1}")
    print(f"Checkpoints saved in: {config.checkpoint_dir}/{config.model_name}")

if __name__ == "__main__":
    main()
