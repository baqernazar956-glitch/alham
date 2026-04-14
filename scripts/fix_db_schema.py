
import sys
import os
from sqlalchemy import text

# Add project root to path
sys.path.append(os.getcwd())

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db

app = create_app()

def fix_schema():
    with app.app_context():
        print("Attempting to add 'query' column to search_history table...")
        try:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE search_history ADD COLUMN query VARCHAR(255)"))
                conn.commit()
            print("SUCCESS: Column 'query' added.")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("Column 'query' already exists.")
            else:
                print(f"FAILED: {e}")

if __name__ == "__main__":
    fix_schema()
