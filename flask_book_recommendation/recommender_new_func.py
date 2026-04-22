
# ------------------------------------------------------------------
# Top Rated – الأعلى تقييماً
# ------------------------------------------------------------------

@cache.memoize(timeout=300)  # Cache for 5 minutes
def get_top_rated(limit=10):
    """
    Get top rated books based on user reviews (BookReview).
    Returns a list of dicts.
    """
    try:
        # 1. Aggregate ratings: Average Rating & Count
        # Filter for books with at least 1 review
        # Order by Average DESC, then Count DESC
        results = (
            db.session.query(
                BookReview.google_id,
                func.avg(BookReview.rating).label('avg_rating'),
                func.count(BookReview.id).label('review_count')
            )
            .group_by(BookReview.google_id)
            .having(func.count(BookReview.id) >= 1) # At least 1 review
            .order_by(func.avg(BookReview.rating).desc(), func.count(BookReview.id).desc())
            .limit(limit)
            .all()
        )
        
        books_dicts = []
        for row in results:
            gid = row.google_id
            avg = float(row.avg_rating)
            count = int(row.review_count)
            
            # Fetch latest review comment for this book
            latest_review = BookReview.query.filter_by(google_id=gid).filter(BookReview.review_text != '').order_by(BookReview.created_at.desc()).first()
            review_text = latest_review.review_text if latest_review else None
            reviewer_name = latest_review.user.name if latest_review and latest_review.user else "Reader"
            
            # 2. Get Book Details
            # Try local DB first
            book = Book.query.filter_by(google_id=gid).first()
            
            book_dict = None
            if book:
                book_dict = _book_to_dict(book, source="Community", reason=f"⭐ {avg:.1f} ({count})")
            else:
                # Fallback to API/Utils if not in DB (slower but necessary)
                from .utils import fetch_book_details
                details = fetch_book_details(gid)
                if details:
                    cover = details.get("cover")
                    if cover and cover.startswith("http://"): cover = "https://" + cover[7:]
                    
                    book_dict = {
                        "id": gid,
                        "title": details.get("title"),
                        "author": details.get("author"),
                        "cover": cover,
                        "source": "Community",
                        "reason": f"⭐ {avg:.1f} ({count})",
                    }
                    
            if book_dict:
                book_dict['rating'] = avg # Explicit rating for UI
                book_dict['review_count'] = count
                if review_text:
                    book_dict['review_text'] = review_text
                    book_dict['reviewer_name'] = reviewer_name
                books_dicts.append(book_dict)
        
        return books_dicts

    except Exception as e:
        logger.error(f"[TopRated] Error: {e}", exc_info=True)
        return []
