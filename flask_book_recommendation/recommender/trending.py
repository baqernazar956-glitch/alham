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


def get_trending(limit=100):
    """
    يحصل على الكتب الأكثر تقييماً وتفاعلاً من قبل المجتمع (حقيقية 100%).
    """
    books_dicts = []
    seen_ids = set()

    try:
        # 1. جلب الكتب الأكثر تقييماً من المجتمع (التي تحتوي على مراجعات فعلياً)
        top_reviewed = (
            db.session.query(
                BookReview.google_id,
                BookReview.book_id,
                func.avg(BookReview.rating).label('avg_rating'),
                func.count(BookReview.id).label('count')
            )
            .group_by(BookReview.google_id, BookReview.book_id)
            .order_by(func.count(BookReview.id).desc(), func.avg(BookReview.rating).desc())
            .limit(limit)
            .all()
        )

        for google_id, book_id, avg_rating, count in top_reviewed:
            b = None
            if book_id: b = Book.query.get(book_id)
            elif google_id: b = Book.query.filter_by(google_id=google_id).first()
            
            if b:
                bid = b.google_id if b.google_id else f"local_{b.id}"
                if bid in seen_ids: continue
                seen_ids.add(bid)
                rating_val = float(avg_rating) if avg_rating is not None else 0.0
                d = _book_to_dict(b, source="Community", reason=f"⭐ تقييم المجتمع: {rating_val:.1f} ({count})", extra_meta={"rating": rating_val})
                if d: books_dicts.append(d)

        # 2. إكمال العدد بكتب حقيقية مضافة للمكتبة من قبل المستخدمين (ليست وهمية)
        if len(books_dicts) < limit:
            recent = Book.query.filter(Book.owner_id.isnot(None)).order_by(Book.created_at.desc()).limit(limit - len(books_dicts)).all()
            for b in recent:
                bid = b.google_id if b.google_id else f"local_{b.id}"
                if bid in seen_ids: continue
                seen_ids.add(bid)
                avg_rating = db.session.query(func.avg(BookReview.rating)).filter_by(google_id=b.google_id).scalar()
                rating_val = float(avg_rating) if avg_rating is not None else 0.0
                d = _book_to_dict(b, source="Library", reason="📚 مضاف حديثاً للمجتمع", extra_meta={"rating": rating_val})
                if d: books_dicts.append(d)

        # 3. Fallback: Google Books (If still empty or too few)
        if len(books_dicts) < 5:
            from ..utils import fetch_google_books
            items, _ = fetch_google_books("bestsellers", max_results=limit - len(books_dicts))
            for b in items:
                bid = b.get('id')
                if bid in seen_ids: continue
                seen_ids.add(bid)
                vi = b.get('volumeInfo', {})
                img = vi.get('imageLinks', {}).get('thumbnail')
                if img and img.startswith('http://'): img = img.replace('http://', 'https://')
                books_dicts.append({
                    "id": bid,
                    "title": vi.get('title'),
                    "author": ", ".join(vi.get('authors', ['Unknown'])),
                    "cover_url": img,
                    "rating": vi.get('averageRating', 4.5),
                    "source": "Global",
                    "reason": "🔥 الأكثر مبيعاً عالمياً"
                })

    except Exception as e:
        logger.error(f"[Trending] Error: {e}")

    return [b for b in books_dicts if b.get('title')][:limit]



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
