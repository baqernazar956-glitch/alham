from flask import Blueprint, render_template, request, make_response, redirect, url_for, jsonify
from flask_login import current_user
import threading
import random
from datetime import datetime, timedelta
from ..models import UserPreference, UserBookView, Book, SearchHistory, UserRatingCF, PublicRating, BookEmbedding, UserGenre
from ..extensions import db, cache
from sqlalchemy import func, desc

# Import Recommender Functions
from ..recommender import (
    get_trending,
    get_top_rated,
    analyze_user_profile_with_ai,
    get_cf_similar,
    get_view_based_recommendations,
    get_behavior_based_recommendations,
    get_deep_learning_recommendations,
    get_content_similar,
    get_last_search_recommendations
)

explore_bp = Blueprint("explore", __name__, url_prefix="/explore")





def _book_to_dict(book, source="Local", reason=None, extra_meta=None):
    """تحويل كائن Book إلى قاموس"""
    if book is None:
        return None
    cover_url = getattr(book, "cover_url", None)
    if cover_url and cover_url.startswith("http://"):
        cover_url = "https://" + cover_url[7:]
        
    data = {
        "id": getattr(book, "google_id", None) or f"local_{book.id}",
        "local_id": book.id,
        "title": getattr(book, "title", None),
        "author": getattr(book, "author", None),
        "cover": cover_url,
        "source": source,
        "reason": reason,
        "rating": getattr(book, "average_rating", None) or getattr(book, "rating", None),
        "categories": getattr(book, "categories", "General"),
        "view_count": 0 # Default
    }
    if extra_meta:
        data.update(extra_meta)
    return data


def get_trending_by_libraries(limit=10):
    """
    الكتب الأكثر إضافة إلى مكتبات المستخدمين (Books in user libraries)
    """
    try:
        # Count occurrences of google_id in user libraries (books with owner_id)
        # Group by google_id to aggregate same books added by different users
        results = (
            db.session.query(
                Book.google_id, 
                func.count(Book.id).label('add_count')
            )
            .filter(Book.owner_id.isnot(None))
            .filter(Book.google_id.isnot(None))
            .group_by(Book.google_id)
            .order_by(desc('add_count'))
            .limit(limit)
            .all()
        )
        
        books_list = []
        for google_id, count in results:
            if not google_id: continue
            book = Book.query.filter_by(google_id=google_id).first()
            if book:
                b_dict = _book_to_dict(book, source="Libraries", reason=f"📚 Added by {count} users")
                b_dict['library_count'] = count
                books_list.append(b_dict)
                
        return books_list
    except Exception as e:
        print(f"Error in get_trending_by_libraries: {e}")
        return []

def get_most_viewed_books_custom(limit=12):
    """
    الكتب الأكثر مشاهدة (Most Viewed)
    """
    try:
        results = (
            db.session.query(
                UserBookView.google_id,
                UserBookView.book_id,
                func.sum(UserBookView.view_count).label('total_views')
            )
            .group_by(UserBookView.google_id, UserBookView.book_id)
            .order_by(desc('total_views'))
            .limit(limit)
            .all()
        )

        books_list = []
        seen = set()
        for gid, bid, views in results:
            if gid and gid in seen: continue
            if bid and f"local_{bid}" in seen: continue
            
            book = None
            if bid: book = Book.query.get(bid)
            elif gid: book = Book.query.filter_by(google_id=gid).first()
            
            if book:
                key = gid or f"local_{book.id}"
                if key in seen: continue
                seen.add(key)
                
                b_dict = _book_to_dict(book, source="Most Viewed", reason=f"👁️ {views} Views")
                b_dict['total_views'] = views
                books_list.append(b_dict)

        return books_list
    except Exception as e:
        print(f"Error in get_most_viewed_books_custom: {e}")
        return []


@explore_bp.get("/", endpoint="index")
def index():
    """
    Redirect /explore to / (Home) as the new main entry point.
    """
    return redirect(url_for("main.home"))

