
import sys
import os
import time

sys.path.append(os.getcwd())

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import Book, BookEmbedding, UserBookView, User
from flask_book_recommendation.utils import get_book_embedding

app = create_app()
ctx = app.app_context()
ctx.push()

user = User.query.filter_by(email="almagd1020@gmail.com").first()
views = UserBookView.query.filter_by(user_id=user.id).all()
book_ids = [v.book_id for v in views if v.book_id]

print(f"Generating embeddings for {len(book_ids)} viewed books...")

for book_id in book_ids:
    book = Book.query.get(book_id)
    if not book: continue
    
    # Check if exists
    if BookEmbedding.query.filter_by(book_id=book.id).first():
        print(f" - Skipping {book.title} (already exists)")
        continue

    print(f" - Generating for: {book.title}...")
    embedding = get_book_embedding(book)
    
    if embedding:
        db.session.add(BookEmbedding(book_id=book.id, vector=embedding))
        db.session.commit()
        print("   ✅ Done")
    else:
        print("   ❌ Failed")
    
    time.sleep(1)

# Also generate for 5 random other books to have candidates
print("\nGenerating candidates...")
candidates = Book.query.filter(Book.id.notin_(book_ids)).limit(5).all()
for book in candidates:
    if BookEmbedding.query.filter_by(book_id=book.id).first():
        continue
        
    print(f" - Candidate: {book.title}...")
    embedding = get_book_embedding(book)
    if embedding:
        db.session.add(BookEmbedding(book_id=book.id, vector=embedding))
        db.session.commit()
    time.sleep(1)

print("\nDone!")
