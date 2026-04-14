import os
import sys
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import UserPreference, UserGenre

def update_all_embeddings(app):
    """
    Background job to update user embeddings based on latest interests and interactions.
    """
    with app.app_context():
        app.logger.info("[Scheduler] Starting 24h user embedding update job...")
        try:
            from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
            
            # Fetch all distinct user IDs that have preferences or genres
            pref_users = set(u[0] for u in db.session.query(UserPreference.user_id).distinct().all())
            genre_users = set(u[0] for u in db.session.query(UserGenre.user_id).distinct().all())
            
            all_users = pref_users.union(genre_users)
            
            for user_id in all_users:
                # Re-initialize or update their vector based on explicit interests
                # (You could also fetch recent interactions from the log here)
                interests = []
                prefs = UserPreference.query.filter_by(user_id=user_id).order_by(UserPreference.weight.desc()).limit(5).all()
                genres = db.session.query(UserGenre.genre_id).filter_by(user_id=user_id).all()
                
                interests.extend([p.topic for p in prefs])
                # Note: To get genre names, we'd need to join, but simple topics are usually enough
                if interests:
                    try:
                        # Re-calculate the base vector
                        user_embedding_manager.initialize_from_interests(user_id, interests)
                    except Exception as e:
                        app.logger.error(f"Failed to update embedding for user {user_id}: {e}")
                        
            app.logger.info("[Scheduler] Finished updating user embeddings.")
            
        except Exception as e:
            app.logger.error(f"[Scheduler] Error during embedding update: {e}")

def run_automated_training(app):
    """
    Background job to run the model retraining scripts automatically.
    """
    with app.app_context():
        app.logger.info("[Scheduler] Starting automated model training job...")
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            
            app.logger.info("[Scheduler] Running retrain_all_models.py...")
            subprocess.run([sys.executable, "scripts/retrain_all_models.py"], cwd=project_root, check=True)
            
            app.logger.info("[Scheduler] Running train_from_database.py...")
            subprocess.run([sys.executable, "scripts/train_from_database.py"], cwd=project_root, check=True)
            
            app.logger.info("[Scheduler] Automated model training finished successfully.")
        except subprocess.CalledProcessError as e:
            app.logger.error(f"[Scheduler] Error during automated training script execution: {e}")
        except Exception as e:
            app.logger.error(f"[Scheduler] Unexpected error during automated training: {e}")


def start_scheduler(app):
    """
    Initializes and starts the APScheduler.
    """
    # Check if we're running in the main Werkzeug process (avoids double execution in debug mode)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true" and app.debug:
        return
        
    # Guard against uvicorn reloader if it's active
    if os.environ.get("RELOAD") == "true" and os.environ.get("UVICORN_RELOADER_RUN") != "true":
        return

    scheduler = BackgroundScheduler(daemon=True)
    
    # Run every 24 hours
    scheduler.add_job(
        func=update_all_embeddings,
        trigger="interval",
        hours=24,
        args=[app],
        id="update_embeddings_job",
        name="Update all user embeddings daily",
        replace_existing=True
    )
    
    # Run model training every 1 hour (or any desired interval)
    scheduler.add_job(
        func=run_automated_training,
        trigger="interval",
        hours=1,
        args=[app],
        id="automated_training_job",
        name="Automated model retraining",
        replace_existing=True
    )
    
    scheduler.start()
    app.logger.debug("[Scheduler] APScheduler started successfully.")
