
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
            
            # 2. Get Book Details
            # Try local DB first
            book = Book.query.filter_by(google_id=gid).first()
            if book:
                d = _book_to_dict(book, source="Community", reason=f"⭐ {avg:.1f} ({count})")
                d['rating'] = avg # Explicit rating for UI
                books_dicts.append(d)
            else:
                # Fallback to API/Utils if not in DB (slower but necessary)
                # We can use fetch_book_details from utils (imported)
                from .utils import fetch_book_details
                details = fetch_book_details(gid)
                if details:
                    cover = details.get("cover")
                    if cover and cover.startswith("http://"): cover = "https://" + cover[7:]
                    
                    books_dicts.append({
                        "id": gid,
                        "title": details.get("title"),
                        "author": details.get("author"),
                        "cover": cover,
                        "source": "Community",
                        "reason": f"⭐ {avg:.1f} ({count})",
                        "rating": avg
                    })
        
        return books_dicts

    except Exception as e:
        logger.error(f"[TopRated] Error: {e}", exc_info=True)
        return []
