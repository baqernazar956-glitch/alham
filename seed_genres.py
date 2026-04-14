import sys
import os

# Add the current directory to sys.path to allow imports
sys.path.append(os.getcwd())

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import Genre

def seed_genres():
    app = create_app()
    with app.app_context():
        genres = [
            "Fiction", "Sci-Fi", "Mystery", "Thriller", "Romance", 
            "Non-Fiction", "Biography", "History", "Science", 
            "Technology", "Philosophy", "Art", "Business", 
            "Self-Help", "Travel", "Horror", "Poetry", 
            "Children", "Psychology", "Cooking", "Religion",
            "Programming", "Artificial Intelligence", "Deep Learning"
        ]
        
        count = 0
        for genre_name in genres:
            existing = Genre.query.filter_by(name=genre_name).first()
            if not existing:
                db.session.add(Genre(name=genre_name))
                count += 1
        
        if count > 0:
            db.session.commit()
            print(f"✅ Seeded {count} new genres.")
        else:
            print("ℹ️ Genres already seeded.")

if __name__ == "__main__":
    seed_genres()
