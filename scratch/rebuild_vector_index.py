import os
import sys
import pickle
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import Book, BookEmbedding
from ai_book_recommender.retrieval.vector_index import VectorIndexService

def rebuild_index():
    print("Initializing Flask context...")
    app = create_app()
    with app.app_context():
        print("Fetching bindings from DB...")
        embeddings = BookEmbedding.query.all()
        print(f"Found {len(embeddings)} embeddings in DB.")
        
        if not embeddings:
            print("No embeddings to index.")
            return

        vectors = []
        ids = []
        
        for emb in embeddings:
            if emb.vector is not None:
                try:
                    # Deserialize vector
                    v = pickle.loads(emb.vector) if isinstance(emb.vector, bytes) else emb.vector
                    v_arr = np.array(v, dtype=np.float32)
                    
                    # Ensure shape is correct (384 for all-MiniLM-L6-v2)
                    if v_arr.shape == (384,):
                        vectors.append(v_arr)
                        # We use 'local_' prefix if Google ID is missing to match DB expectations,
                        # but in the unified pipeline we saw `book["id"]` being used. 
                        # Let's align with _get_popularity_scores in engine.py which expects string numbers or local_X
                        b_id = str(emb.book_id)
                        ids.append(b_id)
                except Exception as e:
                    print(f"Error processing embedding for book {emb.book_id}: {e}")
        
        print(f"Successfully processed {len(vectors)} valid vector arrays.")
        
        if vectors:
            print("Building FAISS index...")
            vectors_np = np.array(vectors, dtype=np.float32)
            vector_service = VectorIndexService(index_dir="instance/indexes")
            
            # The name used in unified_pipeline.py is "books"
            count = vector_service.build_index("books", vectors_np, ids, save=True)
            print(f"✅ Successfully built and saved index with {count} vectors.")
        else:
            print("No valid vectors found to index.")

if __name__ == "__main__":
    rebuild_index()
