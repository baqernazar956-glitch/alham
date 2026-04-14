import numpy as np
import logging
from datetime import datetime
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import UserEmbedding, BookEmbedding, Book

logger = logging.getLogger(__name__)

class UserEmbeddingManager:
    """
    🚀 Manages real-time User Embeddings.
    Computes a running mean of book embeddings for a user.
    """
    
    @staticmethod
    def update_user_embedding(user_id, book_id=None, google_id=None):
        """
        Updates the user's interest vector based on a new interaction.
        Formula: new_mean = (old_mean * n + new_vector) / (n + 1)
        """
        try:
            # 1. Get Book Vector
            if book_id:
                book_emb = BookEmbedding.query.filter_by(book_id=book_id).first()
            elif google_id:
                # Find book first to get internal ID
                book = Book.query.filter_by(google_id=google_id).first()
                if not book:
                    logger.warning(f"Book with google_id {google_id} not found in DB. Skipping embedding update.")
                    return False
                book_emb = BookEmbedding.query.filter_by(book_id=book.id).first()
            else:
                return False

            if not book_emb or book_emb.vector is None:
                logger.debug(f"No vector found for book {book_id or google_id}")
                return False

            import pickle
            
            # De-serialize the vector
            vec_data = book_emb.vector
            if isinstance(vec_data, bytes):
                vec_data = pickle.loads(vec_data)
            elif isinstance(vec_data, str):
                import json
                try:
                    vec_data = json.loads(vec_data)
                except:
                    vec_data = eval(vec_data)
                    
            new_vector = np.array(vec_data, dtype=np.float32)

            # 2. Get/Create User Embedding
            user_emb = UserEmbedding.query.filter_by(user_id=user_id).first()
            if not user_emb:
                user_emb = UserEmbedding(
                    user_id=user_id,
                    vector=pickle.dumps(new_vector.tolist()),
                    interaction_count=1
                )
                db.session.add(user_emb)
            else:
                # update using running mean
                n = user_emb.interaction_count
                
                u_vec_data = user_emb.vector
                if isinstance(u_vec_data, bytes):
                    u_vec_data = pickle.loads(u_vec_data)
                elif isinstance(u_vec_data, str):
                    import json
                    try:
                        u_vec_data = json.loads(u_vec_data)
                    except:
                        u_vec_data = eval(u_vec_data)
                
                old_vector = np.array(u_vec_data, dtype=np.float32)
                
                # Check for shape mismatch (e.g. model changed from 128 to 384 dims)
                if old_vector.shape != new_vector.shape:
                    logger.debug(f"Shape mismatch for user {user_id}: {old_vector.shape} vs {new_vector.shape}. Resetting embedding.")
                    user_emb.vector = pickle.dumps(new_vector.tolist())
                    user_emb.interaction_count = 1
                else:
                    # running mean formula
                    updated_vector = (old_vector * n + new_vector) / (n + 1)
                    user_emb.vector = pickle.dumps(updated_vector.tolist())
                    user_emb.interaction_count = n + 1
            
            db.session.commit()
            logger.info(f"✅ Updated UserEmbedding for user {user_id} (count: {user_emb.interaction_count})")
            return True

        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error updating UserEmbedding: {e}")
            return False

    @staticmethod
    def initialize_from_interests(user_id, interests):
        """
        Creates an initial UserEmbedding vector based on a list of interest strings.
        Useful for cold-start personalization.
        """
        from flask_book_recommendation.utils import get_text_embedding
        
        if not interests:
            return False
            
        try:
            vectors = []
            for interest in interests:
                vec = get_text_embedding(interest)
                if vec:
                    vectors.append(np.array(vec))
            
            if not vectors:
                logger.warning(f"Could not generate vectors for interests: {interests}")
                return False
                
            # Average the vectors
            initial_vector = np.mean(vectors, axis=0)
            
            # Save or Update
            user_emb = UserEmbedding.query.filter_by(user_id=user_id).first()
            if not user_emb:
                user_emb = UserEmbedding(
                    user_id=user_id,
                    vector=initial_vector,
                    interaction_count=len(interests)
                )
                db.session.add(user_emb)
            else:
                user_emb.vector = initial_vector
                user_emb.interaction_count = len(interests)
                
            db.session.commit()
            logger.info(f"✅ Initialized UserEmbedding for user {user_id} with {len(interests)} interests")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error initializing UserEmbedding: {e}")
            return False

# Singleton instance
user_embedding_manager = UserEmbeddingManager()
