"""
Migration script to add reading_progress and last_read_at columns to book_status table.
Run this script once to update the database schema.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from sqlalchemy import text

def migrate():
    app = create_app()
    
    with app.app_context():
        # Check if columns already exist
        try:
            result = db.session.execute(text("SELECT reading_progress FROM book_status LIMIT 1"))
            print("[OK] Column 'reading_progress' already exists")
        except Exception:
            # Add reading_progress column
            try:
                db.session.execute(text("ALTER TABLE book_status ADD COLUMN reading_progress INTEGER DEFAULT 0"))
                db.session.commit()
                print("[OK] Added column 'reading_progress' to book_status")
            except Exception as e:
                print(f"[ERR] Error adding reading_progress: {e}")
                db.session.rollback()
        
        try:
            result = db.session.execute(text("SELECT last_read_at FROM book_status LIMIT 1"))
            print("[OK] Column 'last_read_at' already exists")
        except Exception:
            # Add last_read_at column
            try:
                db.session.execute(text("ALTER TABLE book_status ADD COLUMN last_read_at DATETIME"))
                db.session.commit()
                print("[OK] Added column 'last_read_at' to book_status")
            except Exception as e:
                print(f"[ERR] Error adding last_read_at: {e}")
                db.session.rollback()
        
        print("\n[DONE] Migration completed successfully!")

if __name__ == "__main__":
    migrate()
