import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    try:
        # Create new table
        db.create_all()
        print("Tables created (if not existed).")
        
        # Add columns to book_reviews if they don't exist
        with db.engine.connect() as conn:
            # Check for likes_count
            try:
                conn.execute(text("ALTER TABLE book_reviews ADD COLUMN likes_count INTEGER DEFAULT 0"))
                print("Added likes_count to book_reviews")
            except Exception as e:
                print(f"likes_count likely already exists: {e}")
                
            try:
                conn.execute(text("ALTER TABLE book_reviews ADD COLUMN dislikes_count INTEGER DEFAULT 0"))
                print("Added dislikes_count to book_reviews")
            except Exception as e:
                print(f"dislikes_count likely already exists: {e}")
            
            conn.commit()
        print("Migration completed.")
    except Exception as e:
        print(f"Migration error: {e}")
