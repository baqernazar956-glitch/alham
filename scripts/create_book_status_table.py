"""
Script to create the book_status table if it doesn't exist.
"""
import sys
import os
from sqlalchemy import text

sys.path.append(os.getcwd())
from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db

app = create_app()

def create_book_status_table():
    with app.app_context():
        print("Checking/Creating book_status table...")
        with db.engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='book_status'"))
            exists = result.fetchone()
            
            if exists:
                print("Table 'book_status' already exists.")
            else:
                print("Creating 'book_status' table...")
                conn.execute(text("""
                    CREATE TABLE book_status (
                        id INTEGER NOT NULL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        book_id INTEGER NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        created_at DATETIME,
                        FOREIGN KEY(user_id) REFERENCES users (id),
                        FOREIGN KEY(book_id) REFERENCES books (id),
                        UNIQUE (user_id, book_id)
                    )
                """))
                conn.execute(text("CREATE INDEX idx_book_status_user_id ON book_status (user_id)"))
                conn.execute(text("CREATE INDEX idx_book_status_book_id ON book_status (book_id)"))
                conn.execute(text("CREATE INDEX idx_book_status_status ON book_status (status)"))
                conn.execute(text("CREATE INDEX idx_user_status ON book_status (user_id, status)"))
                conn.commit()
                print("SUCCESS: book_status table created!")

if __name__ == "__main__":
    try:
        create_book_status_table()
    except Exception as e:
        print(f"FAILED: {e}")
