
# -*- coding: utf-8 -*-
import os
import sys
import logging
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle

# Add project root to path
sys.path.append(os.getcwd())

# We reuse the core bootstrap functions for retraining
from scripts.bootstrap_training import (
    train_neural_models, 
    build_faiss_index,
    DB_PATH,
    MODEL_DIR,
    INDEX_DIR,
    DEVICE,
    LATENT_DIM,
    EMBEDDING_DIM
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RetrainModels")

def execute_retraining():
    logger.info("🚀 Starting Full Pipeline Retraining...")
    
    # 1. Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if we have interactions to train on
    cursor.execute("SELECT COUNT(*) FROM user_ratings_cf")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count < 10:
        logger.error(f"❌ Not enough interactions ({count}) to effectively train models.")
        return
        
    logger.info(f"✅ Found {count} ratings to train on.")
    
    # 2. Run the Neural Training Pipeline
    # This trains Transformer, Reranker, Context, and Graph
    try:
        train_neural_models()
        logger.info("✅ All neural models retrained successfully.")
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}", exc_info=True)
        return
        
    # 3. Rebuild the FAISS index with ALL books (including the 100+ new ones)
    try:
        build_faiss_index()
        logger.info("✅ FAISS Vector Index rebuilt successfully.")
    except Exception as e:
        logger.error(f"❌ Error during FAISS index build: {e}", exc_info=True)
        return

    logger.info("🎉 SUCCESS: Models are now retrained and aware of the new distinct personas & books.")

if __name__ == "__main__":
    execute_retraining()
