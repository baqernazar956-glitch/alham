"""
Script to add profile_picture column to users table.
Run this script to fix the 'no such column: users.profile_picture' error.
"""
import sys
import os
from sqlalchemy import text

sys.path.append(os.getcwd())
from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db

app = create_app()

def add_profile_picture_column():
    with app.app_context():
        print("Adding profile_picture column to users table...")
        with db.engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN profile_picture VARCHAR(500)"))
                conn.commit()
                print("SUCCESS: profile_picture column added successfully!")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print("Column already exists, no action needed.")
                else:
                    raise e

if __name__ == "__main__":
    try:
        add_profile_picture_column()
    except Exception as e:
        print(f"FAILED: {e}")
