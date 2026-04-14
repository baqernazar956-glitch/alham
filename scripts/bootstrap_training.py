# -*- coding: utf-8 -*-
"""
🚀 ALHAM Bootstrap Training — The "Deep End" Solution
=====================================================

1. Fetches 500+ books from Google Books API (Diverse categories)
2. Generates real Sentence-Transformer embeddings
3. Creates synthetic users & interactions (Ratings, Views)
4. Trains all 8 neural models (Two-Tower, Transformer, Graph, etc.)
5. Saves all checkpoints and FAISS index
"""

import os
import sys
import time
import json
import random
import logging
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Bootstrap")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
DB_PATH = "flask_book_recommendation/app.db"
MODEL_DIR = "ai_models"
INDEX_DIR = "instance/indexes"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ST_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
LATENT_DIM = 128

CATEGORIES = [
    "Fiction", "Mystery", "Science Fiction", "Biography", "History",
    "Science", "Technology", "Business", "Philosophy", "Psychology",
    "Cooking", "Art", "Travel", "Self-Help", "Poetry", "Religion",
    "Health", "Finance", "Fantasy", "Thriller"
]

BOOKS_PER_CATEGORY = 30

# Ensuring directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH BOOKS FROM GOOGLE BOOKS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_google_books():
    logger.info("📚 Fetching books from Google Books API...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    unique_books = {}
    
    for cat in CATEGORIES:
        logger.info(f"  -> Category: {cat}")
        url = f"https://www.googleapis.com/books/v1/volumes"
        params = {
            "q": f"subject:{cat}",
            "maxResults": BOOKS_PER_CATEGORY,
            "langRestrict": "en",
            "printType": "books"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            items = data.get("items", [])
            for item in items:
                vi = item.get("volumeInfo", {})
                gid = item.get("id")
                title = vi.get("title", "Unknown")
                authors = ", ".join(vi.get("authors", ["Unknown Author"]))
                description = vi.get("description", "")
                cover = vi.get("imageLinks", {}).get("thumbnail", "")
                category = ", ".join(vi.get("categories", [cat]))
                
                if gid not in unique_books:
                    unique_books[gid] = (title, authors, description, cover, category)
        except Exception as e:
            logger.error(f"Error fetching {cat}: {e}")

    logger.info(f"✅ Fetched {len(unique_books)} unique books. Saving to DB...")
    
    # Save to 'books' table (id, title, author, description, cover, google_id)
    # Note: 'owner_id' is nullable in models.py
    added = 0
    for gid, info in unique_books.items():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO books (title, author, description, cover_url, google_id, categories, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (info[0], info[1], info[2], info[3], gid, info[4]))
            added += cursor.rowcount
        except Exception as e:
            pass
            
    conn.commit()
    conn.close()
    logger.info(f"✨ Added {added} new books to database.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: GENERATE REAL EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings():
    logger.info(f"🤖 Generating real embeddings using {ST_MODEL_NAME}...")
    model = SentenceTransformer(ST_MODEL_NAME)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get books that don't have embeddings yet
    cursor.execute("""
        SELECT b.id, b.title, b.description 
        FROM books b
        LEFT JOIN book_embeddings be ON b.id = be.book_id
        WHERE be.book_id IS NULL
    """)
    books = cursor.fetchall()
    
    if not books:
        logger.info("✅ All books already have embeddings.")
        conn.close()
        return

    logger.info(f"  -> Processing {len(books)} books...")
    
    for bid, title, desc in books:
        text = f"{title}. {desc}"[:1000] # Cap text length
        emb = model.encode(text)
        
        # Save to book_embeddings (PickleType stores as binary)
        import pickle
        blob = pickle.dumps(emb)
        cursor.execute("INSERT INTO book_embeddings (book_id, vector) VALUES (?, ?)", (bid, blob))
        
    conn.commit()
    conn.close()
    logger.info("✅ Embeddings generated and saved.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_interactions():
    logger.info("🧪 Generating synthetic user interactions...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create 50 synthetic users if they don't exist
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    if user_count < 50:
        logger.info(f"  -> Creating {50 - user_count} dummy users...")
        for i in range(50 - user_count):
            name = f"User_{i+100}"
            email = f"user{i+100}@example.com"
            cursor.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, 'dummy')", (name, email))
    
    conn.commit()
    
    # Fetch all user IDs and book IDs
    cursor.execute("SELECT id FROM users")
    uids = [r[0] for r in cursor.fetchall()]
    cursor.execute("SELECT id, google_id FROM books WHERE google_id IS NOT NULL")
    binfo = cursor.fetchall()
    bids = [r[0] for r in binfo]
    
    logger.info(f"  -> Generating interactions for {len(uids)} users and {len(binfo)} books...")
    
    # For each user, pick favourite books and generate interactions
    for uid in uids:
        n_fav = min(10, len(bids))
        if n_fav == 0:
            continue
        fav_books = random.sample(bids, n_fav)
        for bid in fav_books:
            gid = next((x[1] for x in binfo if x[0] == bid), None)
            if not gid:
                continue
            # Views (user_book_views uses google_id, not book_id)
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO user_book_views (user_id, google_id, view_count, last_viewed_at) VALUES (?, ?, ?, datetime('now'))",
                    (uid, gid, random.randint(1, 5))
                )
            except Exception:
                pass
            # Ratings (CF)
            rating = random.randint(3, 5)
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO user_ratings_cf (user_id, google_id, rating, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (uid, gid, rating)
                )
            except Exception:
                pass
    
    conn.commit()
    conn.close()
    logger.info("Synthetic interactions generated.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_models():
    logger.info("Training all neural models...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # ── Load all book embeddings from DB ──
    import pickle
    cursor.execute("SELECT book_id, vector FROM book_embeddings")
    emb_rows = cursor.fetchall()
    book_emb_map = {}
    for bid, blob in emb_rows:
        try:
            if blob is not None:
                v = pickle.loads(blob)
                if hasattr(v, 'shape') and v.shape == (384,):
                    book_emb_map[bid] = v
        except Exception:
            pass
    
    if not book_emb_map:
        logger.error("No embeddings found! Run generate_embeddings first.")
        conn.close()
        return
    
    logger.info(f"  Loaded {len(book_emb_map)} book embeddings")
    
    # ── Load interactions ──
    cursor.execute("SELECT user_id, google_id, rating FROM user_ratings_cf")
    ratings = cursor.fetchall()
    
    # Map google_id -> book_id
    cursor.execute("SELECT id, google_id FROM books WHERE google_id IS NOT NULL")
    gid_to_bid = {r[1]: r[0] for r in cursor.fetchall()}
    
    interactions = []  # (user_id, google_id, weight)
    for uid, gid, rating in ratings:
        interactions.append((uid, gid, float(rating) / 5.0))
    
    logger.info(f"  Loaded {len(interactions)} user-book interactions")
    conn.close()
    
    # ======================================================================
    # 4.1: TRANSFORMER ENCODER — Contrastive autoencoder training
    # ======================================================================
    logger.info("  [1/4] Training Transformer Encoder...")
    from ai_book_recommender.models.transformer_encoder import TransformerEncoder
    
    transformer = TransformerEncoder(
        input_dim=EMBEDDING_DIM, hidden_dim=256, output_dim=LATENT_DIM,
        num_heads=8, num_layers=2, dropout=0.1
    ).to(DEVICE)
    transformer.train()
    
    # Training: reconstruct embeddings through bottleneck
    all_embs = np.array(list(book_emb_map.values()), dtype=np.float32)
    
    # Simple self-supervised: input -> transformer -> output should preserve similarity
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Project output back to input dim for reconstruction
    decoder = nn.Linear(LATENT_DIM, EMBEDDING_DIM).to(DEVICE)
    all_params = list(transformer.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=0.001)
    
    n_samples = len(all_embs)
    batch_size = min(32, n_samples)
    n_epochs = 30
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            batch = torch.tensor(all_embs[batch_idx], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            encoded = transformer(batch)  # (B, LATENT_DIM)
            reconstructed = decoder(encoded)  # (B, EMBEDDING_DIM)
            
            loss = criterion(reconstructed, batch.squeeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_batches:.6f}")
    
    transformer.eval()
    torch.save(transformer.state_dict(), os.path.join(MODEL_DIR, "transformer_encoder.pt"))
    logger.info("    Transformer checkpoint saved.")
    
    # ======================================================================
    # 4.2: NEURAL RERANKER — Pairwise ranking on user-book pairs
    # ======================================================================
    logger.info("  [2/4] Training Neural Reranker...")
    from ai_book_recommender.models.neural_reranker import NeuralReranker
    
    reranker = NeuralReranker(
        user_dim=LATENT_DIM, item_dim=LATENT_DIM,
        hidden_dim=256, num_features=10, dropout=0.1
    ).to(DEVICE)
    reranker.train()
    
    optimizer_rr = optim.Adam(reranker.parameters(), lr=0.001)
    
    # Generate user embeddings (average of their rated book embeddings)
    user_emb_map = {}
    user_books = {}
    for uid, gid, w in interactions:
        bid = gid_to_bid.get(gid)
        if bid and bid in book_emb_map:
            user_books.setdefault(uid, []).append((bid, w))
    
    for uid, book_list in user_books.items():
        embs = [book_emb_map[bid] for bid, _ in book_list]
        user_emb_map[uid] = np.mean(embs, axis=0)
    
    # Generate encoded book embeddings through trained transformer
    transformer.eval()
    encoded_books = {}
    with torch.no_grad():
        for bid, emb in book_emb_map.items():
            inp = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            enc = transformer(inp).squeeze(0).cpu().numpy()  # shape: (128,)
            encoded_books[bid] = enc
    
    # Save user embeddings to DB
    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    for uid, emb in user_emb_map.items():
        blob = pickle.dumps(emb)
        cur2.execute("INSERT OR REPLACE INTO user_embeddings (user_id, vector, interaction_count) VALUES (?, ?, ?)",
                     (uid, blob, len(user_books.get(uid, []))))
    conn2.commit()
    conn2.close()
    logger.info(f"    Saved {len(user_emb_map)} user embeddings to DB")
    
    # Train reranker with pairwise loss
    # Reranker.forward expects: user_emb (batch, user_dim), item_embs (batch, num_items, item_dim)
    all_bids = list(encoded_books.keys())
    n_epochs_rr = 20
    
    for epoch in range(n_epochs_rr):
        total_loss = 0.0
        n_pairs = 0
        
        for uid, book_list in user_books.items():
            if uid not in user_emb_map:
                continue
            
            u_emb_raw = user_emb_map[uid]
            # Encode user embedding through transformer
            with torch.no_grad():
                u_inp = torch.tensor(u_emb_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                u_enc = transformer(u_inp)  # (1, LATENT_DIM)
            
            for pos_bid, weight in book_list[:5]:
                if pos_bid not in encoded_books:
                    continue
                
                neg_bid = random.choice(all_bids)
                while neg_bid == pos_bid:
                    neg_bid = random.choice(all_bids)
                
                # Shape: (1, 1, LATENT_DIM) — batch=1, num_items=1
                pos_emb = torch.tensor(encoded_books[pos_bid], dtype=torch.float32).view(1, 1, -1).to(DEVICE)
                neg_emb = torch.tensor(encoded_books[neg_bid], dtype=torch.float32).view(1, 1, -1).to(DEVICE)
                
                pos_score = reranker(u_enc, pos_emb)  # (1, 1)
                neg_score = reranker(u_enc, neg_emb)  # (1, 1)
                
                # BPR loss
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                
                optimizer_rr.zero_grad()
                loss.backward()
                optimizer_rr.step()
                
                total_loss += loss.item()
                n_pairs += 1
        
        if (epoch + 1) % 5 == 0 and n_pairs > 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs_rr}, BPR Loss: {total_loss/n_pairs:.6f}")
    
    reranker.eval()
    torch.save(reranker.state_dict(), os.path.join(MODEL_DIR, "neural_reranker.pt"))
    logger.info("    Reranker checkpoint saved.")
    
    # ======================================================================
    # 4.3: CONTEXT-AWARE RANKER — Simple supervised training
    # ======================================================================
    logger.info("  [3/4] Training Context-Aware Ranker...")
    from ai_book_recommender.models.context_ranker import ContextAwareRanker
    
    context_ranker = ContextAwareRanker(
        user_dim=LATENT_DIM, item_dim=LATENT_DIM,
        context_dim=64, hidden_dim=256, dropout=0.1
    ).to(DEVICE)
    context_ranker.train()
    
    optimizer_cr = optim.Adam(context_ranker.parameters(), lr=0.001)
    
    # Train to predict interaction weight
    n_epochs_cr = 15
    for epoch in range(n_epochs_cr):
        total_loss = 0.0
        n_samples_cr = 0
        
        for uid, book_list in user_books.items():
            if uid not in user_emb_map:
                continue
            
            with torch.no_grad():
                u_inp = torch.tensor(user_emb_map[uid], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                u_enc = transformer(u_inp)  # (1, LATENT_DIM)
            
            for bid, weight in book_list[:3]:
                if bid not in encoded_books:
                    continue
                
                # item_emb: (1, 1, LATENT_DIM)
                item_emb = torch.tensor(encoded_books[bid], dtype=torch.float32).view(1, 1, -1).to(DEVICE)
                
                ctx = context_ranker.get_current_context(device=DEVICE)
                hour_t = ctx["hour"]
                day_t = ctx["day"]
                session_feats = ctx["session_features"]
                
                base_score = torch.tensor([[weight]], dtype=torch.float32, device=DEVICE)
                
                pred = context_ranker(u_enc, item_emb, hour_t, day_t, session_feats, base_scores=base_score)
                target = torch.tensor([[weight]], dtype=torch.float32, device=DEVICE)
                
                loss = nn.MSELoss()(pred, target)
                
                optimizer_cr.zero_grad()
                loss.backward()
                optimizer_cr.step()
                
                total_loss += loss.item()
                n_samples_cr += 1
        
        if (epoch + 1) % 5 == 0 and n_samples_cr > 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs_cr}, MSE Loss: {total_loss/n_samples_cr:.6f}")
    
    context_ranker.eval()
    torch.save(context_ranker.state_dict(), os.path.join(MODEL_DIR, "context_ranker.pt"))
    logger.info("    Context Ranker checkpoint saved.")

    
    # ======================================================================
    # 4.4: GRAPH RECOMMENDER (LightGCN) — Build graph and save
    # ======================================================================
    logger.info("  [4/4] Building Graph Recommender...")
    from ai_book_recommender.models.graph_recommender import GraphRecommender
    
    graph = GraphRecommender(embedding_dim=64, num_layers=3)
    
    if interactions:
        graph.build_graph(interactions)
        
        # Simple BPR training for graph embeddings
        if graph.model is not None:
            graph.model.train()
            optimizer_g = optim.Adam(graph.model.parameters(), lr=0.01)
            
            user_ids = list(set(uid for uid, _, _ in interactions))
            item_ids = list(set(gid for _, gid, _ in interactions))
            
            user_items = {}
            for uid, gid, w in interactions:
                user_items.setdefault(uid, set()).add(gid)
            
            n_epochs_g = 20
            for epoch in range(n_epochs_g):
                total_loss = 0.0
                n_pairs_g = 0
                
                user_emb_g, item_emb_g = graph.model(graph.edge_index, graph.edge_weight)
                num_users_g = len(graph.user_id_map)
                
                for uid in random.sample(user_ids, min(30, len(user_ids))):
                    if uid not in graph.user_id_map:
                        continue
                    u_idx = graph.user_id_map[uid]
                    
                    pos_items = user_items.get(uid, set())
                    if not pos_items:
                        continue
                    
                    pos_gid = random.choice(list(pos_items))
                    neg_gid = random.choice(item_ids)
                    while neg_gid in pos_items and len(item_ids) > len(pos_items):
                        neg_gid = random.choice(item_ids)
                    
                    if pos_gid not in graph.item_id_map or neg_gid not in graph.item_id_map:
                        continue
                    
                    pos_idx = graph.item_id_map[pos_gid] + num_users_g
                    neg_idx = graph.item_id_map[neg_gid] + num_users_g
                    
                    u_vec = user_emb_g[u_idx]
                    pos_vec = item_emb_g[pos_idx - num_users_g]
                    neg_vec = item_emb_g[neg_idx - num_users_g]
                    
                    pos_score = (u_vec * pos_vec).sum()
                    neg_score = (u_vec * neg_vec).sum()
                    
                    loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
                    
                    optimizer_g.zero_grad()
                    loss.backward()
                    optimizer_g.step()
                    
                    total_loss += loss.item()
                    n_pairs_g += 1
                    
                    # Recompute embeddings after update
                    user_emb_g, item_emb_g = graph.model(graph.edge_index, graph.edge_weight)
                
                if (epoch + 1) % 5 == 0 and n_pairs_g > 0:
                    logger.info(f"    Epoch {epoch+1}/{n_epochs_g}, BPR Loss: {total_loss/n_pairs_g:.6f}")
            
            graph.model.eval()
        
        graph.save(os.path.join(MODEL_DIR, "graph_recommender.pt"))
        logger.info("    Graph Recommender checkpoint saved.")
    else:
        logger.warning("    No interactions for graph — skipping.")
    
    logger.info("All models trained and saved to ai_models/")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: FAISS INDEX BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_faiss_index():
    logger.info("Building FAISS Vector Index...")
    from ai_book_recommender.retrieval.vector_index import FAISSIndex
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT book_id, vector FROM book_embeddings")
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning("No embeddings found to index!")
        conn.close()
        return
    
    import pickle
    vectors = []
    ids = []
    for bid, blob in rows:
        try:
            if blob is not None:
                v = pickle.loads(blob)
                if hasattr(v, 'shape') and v.shape == (384,):
                    vectors.append(v)
                    ids.append(str(bid))
        except Exception:
            pass
        
    vectors = np.array(vectors).astype('float32')
    
    index = FAISSIndex(dim=EMBEDDING_DIM, index_type="Flat")
    index.build(vectors, ids)
    
    path = os.path.join(INDEX_DIR, "books.index")
    index.save(path)
    conn.close()
    logger.info(f"FAISS index saved. Size: {index.size} vectors")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("🚀 Starting Bootstrap Training Flow")
    
    try:
        fetch_google_books()
        generate_embeddings()
        generate_synthetic_interactions()
        train_neural_models()
        build_faiss_index()
        logger.info("🎉 BOOTSTRAP COMPLETE! You now have a working AI-powered site.")
    except Exception as e:
        logger.critical(f"❌ Bootstrap Failed: {e}", exc_info=True)
        sys.exit(1)
