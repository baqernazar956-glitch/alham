# -*- coding: utf-8 -*-
"""
Trending recommendations — popular books.
"""
import logging
import random
from sqlalchemy import func

from ..models import Book, BookStatus, UserBookView, BookReview
from ..extensions import db, cache
from ..utils import fetch_google_books
from .helpers import _book_to_dict, _deduplicate_dicts

logger = logging.getLogger(__name__)


def get_trending(limit=12):
    """
    يحصل على الكتب الرائجة مع fallback ذكي في حال كانت قاعدة البيانات فارغة.
    """
    books_dicts = []
    seen_ids = set()

    try:
        user_books = (
            Book.query
            .filter(Book.owner_id.isnot(None))
            .order_by(Book.created_at.desc())
            .limit(limit * 3)
            .all()
        )
        
        if len(user_books) < limit:
            # ⭐ Deterministic fallback: Recent additions instead of random
            more_books = Book.query.order_by(Book.id.desc()).limit(limit * 3).all()
            user_books.extend(more_books)
            
        random.shuffle(user_books)
        
        for b in user_books:
            book_id = f"local_{b.id}" if not b.google_id else b.google_id
            if book_id in seen_ids:
                continue
            seen_ids.add(book_id)
            
            if not b.title or b.title in ['Untitled', 'Unknown']:
                continue

            owner_name = "مستخدم"
            owner_id = None
            if getattr(b, "owner", None):
                owner_id = b.owner.id
                if b.owner.name:
                    owner_name = b.owner.name
                    
            book_dict = _book_to_dict(
                b,
                source="المكتبة",
                reason=f"👤 أضافه: {owner_name}" if getattr(b, "owner", None) else "🔥 شائع محلياً",
            )
            
            if book_dict:
                book_dict['owner_name'] = owner_name
                book_dict['owner_id'] = owner_id
                books_dicts.append(book_dict)
            
            if len(books_dicts) >= limit:
                break
                
        if len(books_dicts) < limit:
            try:
                # ⭐ Deterministic fallback: High-quality curated topic
                fallback_query = "best selling books"
                items, _ = fetch_google_books(fallback_query, max_results=limit - len(books_dicts))
                for item in items:
                    v = item.get("volumeInfo", {})
                    cover = v.get("imageLinks", {}).get("thumbnail")
                    if cover and cover.startswith("http://"): cover = "https" + cover[4:]
                    fallback_dict = {
                        "id": item.get("id"),
                        "title": v.get("title", "رائج الان"),
                        "author": v.get("authors", ["غير معروف"])[0] if v.get("authors") else "غير معروف",
                        "cover": cover,
                        "source": "Google Books",
                        "reason": "🔥 شائع عالمياً",
                        "rating": v.get("averageRating")
                    }
                    books_dicts.append(fallback_dict)
            except Exception as e:
                logger.error(f"[Trending] Internet fallback error: {e}", exc_info=True)
                
    except Exception as e:
        logger.error(f"[Trending] Error: {e}", exc_info=True)

    random.shuffle(books_dicts)
    books_dicts = _deduplicate_dicts(books_dicts)
    result = books_dicts[:limit]
    logger.info(f"[Trending] Returning {len(result)} trending books")
    return result


def get_trending_by_period(period='week', limit=12):
    """
    جلب الكتب الرائجة بناءً على فترة زمنية محددة.
    """
    from datetime import datetime, timedelta
    
    try:
        now = datetime.utcnow()
        if period == 'day':
            start_date = now - timedelta(days=1)
            period_label = "اليوم"
        elif period == 'week':
            start_date = now - timedelta(weeks=1)
            period_label = "هذا الأسبوع"
        elif period == 'month':
            start_date = now - timedelta(days=30)
            period_label = "هذا الشهر"
        else:
            start_date = None
            period_label = "كل الأوقات"
        
        query = db.session.query(
            BookStatus.book_id,
            func.count(BookStatus.id).label('count')
        ).filter(
            BookStatus.status.in_(['favorite', 'finished'])
        )
        
        if start_date:
            query = query.filter(BookStatus.created_at >= start_date)
        
        popular_books = query.group_by(BookStatus.book_id).order_by(
            func.count(BookStatus.id).desc()
        ).limit(limit * 2).all()
        
        books = []
        seen_ids = set()
        
        for book_id, count in popular_books:
            if len(books) >= limit:
                break
            book = Book.query.get(book_id)
            if book and book.google_id not in seen_ids:
                seen_ids.add(book.google_id)
                books.append(_book_to_dict(
                    book, 
                    source="Trending",
                    reason=f"🔥 رائج {period_label} ({count} قارئ)"
                ))
        
        if len(books) < limit:
            view_query = db.session.query(
                UserBookView.book_id,
                func.sum(UserBookView.view_count).label('views')
            )
            
            if start_date:
                view_query = view_query.filter(UserBookView.last_viewed_at >= start_date)
            
            popular_views = view_query.group_by(UserBookView.book_id).order_by(
                func.sum(UserBookView.view_count).desc()
            ).limit(limit).all()
            
            for book_id, views in popular_views:
                if len(books) >= limit:
                    break
                book = Book.query.get(book_id)
                if book and book.google_id not in seen_ids:
                    seen_ids.add(book.google_id)
                    books.append(_book_to_dict(
                        book,
                        source="Trending",
                        reason=f"👀 الأكثر مشاهدة {period_label}"
                    ))
        
        logger.info(f"[Trending] Found {len(books)} books for period '{period}'")
        return books
        
    except Exception as e:
        logger.error(f"[Trending by Period] Error: {e}")
        return []
