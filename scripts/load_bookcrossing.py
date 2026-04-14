import os
import sys
import pandas as pd
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup path
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(basedir)

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import Book, User, UserRatingCF, BookEmbedding

BOOKS_URL = 'https://raw.githubusercontent.com/ajaykuma/Datasets_For_Work/main/BX-Books.csv'
RATINGS_URL = 'https://raw.githubusercontent.com/rochitasundar/Collaborative-Filtering-Book-Recommendation-System/master/BX-Book-Ratings.csv'

def download_csv(url, filename, extract_to='bx_data'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    filepath = os.path.join(extract_to, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Saved to {filepath}")
        else:
            print(f"Error: Failed to download {filename} (Status: {r.status_code})")
    else:
        print(f"{filename} already exists.")
    return filepath

def load_bookcrossing_data():
    app = create_app()
    with app.app_context():
        data_dir = 'bx_data'
        download_csv(BOOKS_URL, 'BX-Books.csv')
        download_csv(RATINGS_URL, 'BX-Book-Ratings.csv')

        print("Loading CSV files...")
        # Mirror schemas are inconsistent: 
        # BX-Books.csv (GitHub) is comma-separated, lower_case
        # BX-Book-Ratings.csv (GitHub) is semicolon-separated, Title-Case
        
        try:
            books = pd.read_csv(os.path.join(data_dir, 'BX-Books.csv'), sep=',', encoding='latin-1', on_bad_lines='skip', low_memory=False)
            if 'isbn' in books.columns:
                books.columns = [c.upper() for c in books.columns]
                books = books.rename(columns={'BOOK_TITLE': 'Title', 'BOOK_AUTHOR': 'Author', 'YEAR_OF_PUBLICATION': 'Year', 'PUBLISHER': 'Publisher'})
        except Exception as e:
            print(f"CSV Books Load Error: {e}")
            books = pd.read_csv(os.path.join(data_dir, 'BX-Books.csv'), sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)

        try:
            ratings = pd.read_csv(os.path.join(data_dir, 'BX-Book-Ratings.csv'), sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
            # Remove quotes if they exist in column names
            ratings.columns = [c.strip('"').replace('-', '_').upper() for c in ratings.columns]
            ratings = ratings.rename(columns={'USER_ID': 'User_ID', 'BOOK_RATING': 'Rating'})
        except Exception as e:
            print(f"CSV Ratings Load Error: {e}")
            ratings = pd.read_csv(os.path.join(data_dir, 'BX-Book-Ratings.csv'), sep=',', encoding='latin-1', on_bad_lines='skip', low_memory=False)

        print(f"Original Ratings: {len(ratings)}")
        
        # Limit to a sample for testing pipeline stability
        ratings = ratings.head(200000)
        
        # Ensure 'Rating' is numeric
        ratings['Rating'] = pd.to_numeric(ratings['Rating'], errors='coerce').fillna(0)
        ratings = ratings[ratings['Rating'] > 0]
        print(f"Sampling {len(ratings)} ratings.")

        # Filter users >= 2 ratings
        user_counts = ratings['User_ID'].value_counts()
        valid_users = user_counts[user_counts >= 2].index
        ratings = ratings[ratings['User_ID'].isin(valid_users)]
        
        # Filter books >= 2 ratings
        book_counts = ratings['ISBN'].value_counts()
        valid_books = book_counts[book_counts >= 2].index
        ratings = ratings[ratings['ISBN'].isin(valid_books)]
        
        print(f"Filtered Ratings: {len(ratings)}")

        # Keep only books that exist in the ratings
        valid_isbns = ratings['ISBN'].unique()
        books = books[books['ISBN'].isin(valid_isbns)]
        
        print(f"Filtered Books: {len(books)}")

        # Users to create
        valid_user_ids = ratings['User_ID'].unique()
        
        # 1. Create Users
        print(f"Creating/Checking {len(valid_user_ids)} Users...")
        existing_users = {u.id for u in User.query.with_entities(User.id).all()}
        new_users = []
        for uid in valid_user_ids:
            try:
                uid_int = int(str(uid).strip('"'))
                if uid_int not in existing_users:
                    new_users.append(User(
                        id=uid_int, 
                        name=f"bx_user_{uid_int}", 
                        email=f"bx_{uid_int}@bookcrossing.com", 
                        password_hash="dummy"
                    ))
                    existing_users.add(uid_int)
                    if len(new_users) >= 1000:
                        db.session.bulk_save_objects(new_users)
                        db.session.commit()
                        new_users = []
            except:
                continue
        if new_users:
            db.session.bulk_save_objects(new_users)
            db.session.commit()

        # 2. Create Books
        print(f"Creating/Checking {len(books)} Books...")
        existing_books = {b.google_id: b.id for b in Book.query.with_entities(Book.google_id, Book.id).all()}
        new_books = []
        for _, row in books.iterrows():
            isbn = str(row['ISBN']).strip('"')
            if isbn not in existing_books:
                new_books.append(Book(
                    google_id=isbn,
                    title=str(row['Title'])[:250],
                    author=str(row['Author'])[:250] if 'Author' in row else None,
                    cover_url=str(row['IMAGE_URL_L'])[:250] if 'IMAGE_URL_L' in row and pd.notna(row['IMAGE_URL_L']) else None,
                    published_date=str(row['Year'])[:50] if 'Year' in row else None,
                    categories=str(row['Publisher'])[:100] if 'Publisher' in row else None
                ))
                if len(new_books) >= 1000:
                    db.session.bulk_save_objects(new_books)
                    db.session.commit()
                    new_books = []
        if new_books:
            db.session.bulk_save_objects(new_books)
            db.session.commit()
            
        # Refresh existing_books dict after inserts
        existing_books = {b.google_id: b.id for b in Book.query.with_entities(Book.google_id, Book.id).all()}

        # 3. Create BookEmbeddings
        print("Generating Embeddings for Books...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Find books without embeddings
        books_with_emb = {e.book_id for e in BookEmbedding.query.with_entities(BookEmbedding.book_id).all()}
        books_to_embed = []
        for _, row in books.iterrows():
            isbn = str(row['ISBN']).strip('"')
            book_id = existing_books.get(isbn)
            if book_id and book_id not in books_with_emb:
                books_to_embed.append({'book_id': book_id, 'title': str(row['Title'])})
                
        if books_to_embed:
            print(f"Embedding {len(books_to_embed)} books...")
            titles = [b['title'] for b in books_to_embed]
            vectors = model.encode(titles, batch_size=64, show_progress_bar=True)
            
            new_embs = []
            for item, vector in zip(books_to_embed, vectors):
                new_embs.append(BookEmbedding(
                    book_id=item['book_id'],
                    vector=vector.tolist()
                ))
                if len(new_embs) >= 1000:
                    db.session.bulk_save_objects(new_embs)
                    db.session.commit()
                    new_embs = []
            if new_embs:
                db.session.bulk_save_objects(new_embs)
                db.session.commit()

        # 4. Insert Ratings
        print("Inserting user ratings...")
        existing_cf = {(r.user_id, r.google_id) for r in UserRatingCF.query.with_entities(UserRatingCF.user_id, UserRatingCF.google_id).all()}
        new_ratings = []
        
        for _, row in tqdm(ratings.iterrows(), total=len(ratings)):
            try:
                uid = int(str(row['User_ID']).strip('"'))
                isbn = str(row['ISBN']).strip('"')
                rating_val = float(row['Rating'])
                
                rating_mapped = rating_val / 2.0
                
                if (uid, isbn) not in existing_cf:
                    new_ratings.append(UserRatingCF(
                        user_id=uid,
                        google_id=isbn,
                        rating=rating_mapped
                    ))
                    existing_cf.add((uid, isbn))
                    
                    if len(new_ratings) >= 2000:
                        db.session.bulk_save_objects(new_ratings)
                        db.session.commit()
                        new_ratings = []
            except:
                continue
                    
        if new_ratings:
            db.session.bulk_save_objects(new_ratings)
            db.session.commit()
            
        print("Data Loading Complete!")

if __name__ == "__main__":
    load_bookcrossing_data()
