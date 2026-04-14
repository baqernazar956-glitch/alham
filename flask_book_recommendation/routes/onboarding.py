
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from ..extensions import db
from ..models import Genre, UserGenre, UserPreference
from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager

onboarding_bp = Blueprint("onboarding", __name__, url_prefix="/api/onboarding")

@onboarding_bp.route("/interests", methods=["POST"])
@login_required
def save_interests():
    """
    Save user interests during onboarding.
    Expected JSON: {"interests": ["Sci-Fi", "History", "Coding", ...]}
    """
    data = request.get_json()
    interests = data.get("interests", [])
    
    if not interests:
        return jsonify({"error": "No interests provided"}), 400
        
    try:
        # 1. Clear previous preferences to be safe (or we could append)
        # For onboarding, we usually want a fresh start
        UserGenre.query.filter_by(user_id=current_user.id).delete()
        UserPreference.query.filter_by(user_id=current_user.id).delete()
        
        # 2. Process each interest
        all_genres = {g.name.lower(): g.id for g in Genre.query.all()}
        
        for interest in interests:
            term = interest.strip()
            if not term:
                continue
                
            term_lower = term.lower()
            
            # Check if it matches a known Genre
            if term_lower in all_genres:
                # Save as UserGenre
                genre_id = all_genres[term_lower]
                db.session.add(UserGenre(user_id=current_user.id, genre_id=genre_id))
            else:
                # Save as generic UserPreference (Topic)
                # Give high weight (150.0) for explicit selection
                db.session.add(UserPreference(user_id=current_user.id, topic=term, weight=150.0))
        
        # 3. Mark onboarding as completed
        current_user.onboarding_completed = True
        db.session.commit()
        
        # 4. Initialize User Embedding in background (or inline for now)
        try:
            user_embedding_manager.initialize_from_interests(current_user.id, interests)
        except Exception as e:
            logger.error(f"Error initializing user embedding: {e}")

        return jsonify({"status": "success", "message": "Interests saved"}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
