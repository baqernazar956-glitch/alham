import sqlite3
import os

db_path = r'c:\Users\al6md\Desktop\project alham\flask_book_recommendation_starter\flask_book_recommendation\app.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add columns to book_reviews
try:
    cursor.execute("ALTER TABLE book_reviews ADD COLUMN likes_count INTEGER DEFAULT 0")
    print("Added likes_count to book_reviews")
except Exception as e:
    print(f"likes_count likely exists: {e}")

try:
    cursor.execute("ALTER TABLE book_reviews ADD COLUMN dislikes_count INTEGER DEFAULT 0")
    print("Added dislikes_count to book_reviews")
except Exception as e:
    print(f"dislikes_count likely exists: {e}")

# Create review_reactions table
try:
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_reactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        review_id INTEGER NOT NULL,
        reaction_type VARCHAR(20) NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(review_id) REFERENCES book_reviews(id),
        UNIQUE(user_id, review_id)
    )
    """)
    print("Created review_reactions table")
except Exception as e:
    print(f"Error creating table: {e}")

conn.commit()
conn.close()
print("Done.")
