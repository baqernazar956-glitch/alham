import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from ..extensions import db, cache
from ..models import Genre, UserGenre, UserPreference

logger = logging.getLogger(__name__)

prefs_bp = Blueprint("prefs", __name__, url_prefix="/prefs")

@prefs_bp.route("/preferences", methods=["GET","POST"])
@login_required
def form():
    genres = Genre.query.order_by(Genre.name.asc()).all()
    if request.method == "POST":
        # Form sends name="interests" with text values like "Technology", "Fiction"
        selected_interests = request.form.getlist("interests")
        
        if len(selected_interests) < 3:
            flash("Please select at least 3 interests", "error")
            return redirect(url_for("prefs.form"))
        
        # ── 1. Clear previous selections ──
        UserGenre.query.filter_by(user_id=current_user.id).delete()
        UserPreference.query.filter_by(user_id=current_user.id).delete()
        
        # ── 2. Save to both UserGenre AND UserPreference ──
        # Build a lookup of genre names to genre objects
        all_genres = {g.name.lower(): g for g in Genre.query.all()}
        selected_genre_names = []
        
        for interest_name in selected_interests:
            interest_name = interest_name.strip()
            if not interest_name:
                continue
            
            selected_genre_names.append(interest_name)
            
            # Look up genre by name (case-insensitive)
            genre = all_genres.get(interest_name.lower())
            if not genre:
                genre = Genre(name=interest_name)
                db.session.add(genre)
                db.session.flush() # get the newly created genre ID
                all_genres[interest_name.lower()] = genre
                
            db.session.add(UserGenre(user_id=current_user.id, genre_id=genre.id))
            
            # Always save as UserPreference with high weight
            db.session.add(UserPreference(
                user_id=current_user.id,
                topic=interest_name,
                weight=150.0  # Explicit selection = highest priority
            ))
        
        # ── 3. Mark onboarding as completed ──
        current_user.onboarding_completed = True
        db.session.commit()
        
        # ── 4. Build User Embedding from interests (cold-start) ──
        try:
            from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
            user_embedding_manager.initialize_from_interests(current_user.id, selected_genre_names)
            logger.info(f"[Prefs] ✅ Built User Embedding for user {current_user.id}")
        except Exception as e:
            logger.error(f"[Prefs] Error building user embedding: {e}")
        
        # ── 5. Seed books from Google Books in background ──
        try:
            from ..interest_seeder import seed_books_for_interests
            app = current_app._get_current_object()
            seed_books_for_interests(current_user.id, selected_genre_names, app=app)
        except Exception as e:
            logger.error(f"[Prefs] Error seeding books: {e}")
        
        # ── 6. Clear all caches for immediate recommendation refresh ──
        try:
            cache.clear()
        except Exception:
            pass
        try:
            from ai_book_recommender.unified_pipeline import get_unified_engine
            engine = get_unified_engine()
            engine.clear_user_cache(current_user.id)
        except Exception:
            pass
        
        flash("Your interests have been saved! 🎉", "success")
        return redirect(url_for("explore.index"))
    
    chosen = {x.genre_id for x in UserGenre.query.filter_by(user_id=current_user.id).all()}
    return render_template("preferences.html", genres=genres, chosen=chosen)

