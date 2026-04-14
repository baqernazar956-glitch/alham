
import sys
import os
from sqlalchemy import text

sys.path.append(os.getcwd())
from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db

app = create_app()

def fix_constraint():
    with app.app_context():
        print("Starting schema fix for search_history...")
        with db.engine.connect() as conn:
            # 1. Rename old table
            conn.execute(text("ALTER TABLE search_history RENAME TO search_history_old"))
            
            # 2. Create new table
            conn.execute(text("""
                CREATE TABLE search_history (
                    id INTEGER NOT NULL, 
                    user_id INTEGER NOT NULL, 
                    book_id INTEGER, 
                    "query" VARCHAR(255), 
                    created_at DATETIME, 
                    PRIMARY KEY (id), 
                    FOREIGN KEY(user_id) REFERENCES users (id), 
                    FOREIGN KEY(book_id) REFERENCES books (id)
                )
            """))
            
            # 3. Copy data
            # Note: We must handle the case where old data (if any) fits into new schema
            conn.execute(text("""
                INSERT INTO search_history (id, user_id, book_id, "query", created_at)
                SELECT id, user_id, book_id, "query", created_at FROM search_history_old
            """))
            
            # 4. Drop old table
            conn.execute(text("DROP TABLE search_history_old"))
            
            # 5. Recreate Indices
            conn.execute(text("CREATE INDEX idx_search_history_created_at ON search_history (created_at)"))
            conn.execute(text("CREATE INDEX ix_search_history_book_id ON search_history (book_id)"))
            conn.execute(text("CREATE INDEX ix_search_history_user_id ON search_history (user_id)"))
            conn.execute(text("CREATE INDEX idx_user_created ON search_history (user_id, created_at)"))

            conn.commit()
            print("SUCCESS: Schema fixed. book_id is now nullable.")

if __name__ == "__main__":
    try:
        fix_constraint()
    except Exception as e:
        print(f"FAILED: {e}")
