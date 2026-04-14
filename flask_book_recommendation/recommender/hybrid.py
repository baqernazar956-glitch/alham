# -*- coding: utf-8 -*-
"""
Hybrid recommendations — get_hybrid_recommendations, get_author_books,
get_top_rated, get_because_you_read, get_similar_users_favorites,
get_hidden_gems, get_genre_explorer.
"""
import logging
import random
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func

from ..models import (
    Book, UserRatingCF, BookEmbedding,
    BookStatus, BookReview, UserBookView
)
from ..extensions import db, cache
from ..utils import fetch_google_books
from .helpers import _book_to_dict

logger = logging.getLogger(__name__)


@cache.memoize(timeout=3600)
def get_hybrid_recommendations(user_id, book, limit=12):
    """
    توصيات هجينة للكتاب الحالي.
    """
    if not book: return []
    
    recs = []
    seen_ids = {book.google_id} if book.google_id else {f"local_{book.id}"}
    
    # --- 1. Collaborative Filtering (Item-based) ---
    try:
        if book.google_id:
            fans = UserRatingCF.query.filter(
                UserRatingCF.google_id == book.google_id, 
                UserRatingCF.rating >= 4
            ).limit(20).all()
            
            fan_ids = [f.user_id for f in fans if f.user_id != user_id]
            if fan_ids:
                suggested_ratings = UserRatingCF.query.filter(
                    UserRatingCF.user_id.in_(fan_ids),
                    UserRatingCF.rating >= 4,
                    UserRatingCF.google_id != book.google_id
                ).limit(limit * 2).all()
                
                gids = [r.google_id for r in suggested_ratings]
                common_gids = [gid for gid, count in Counter(gids).most_common(limit)]
                
                for gid in common_gids:
                    if gid in seen_ids: continue
                    b = Book.query.filter_by(google_id=gid).first()
                    if b:
                        recs.append(_book_to_dict(b, source="Community", reason="👥 أحبه قراء آخرون لهم نفس ذوقك"))
                        seen_ids.add(gid)
    except Exception as e:
        logger.error(f"[Hybrid] CF error: {e}")

    # --- 2. More by Same Author ---
    if len(recs) < limit and book.author and book.author not in ['Unknown', 'غير معروف']:
        try:
            author_recs = get_author_books(book.author, exclude_book_id=book.google_id, limit=limit//2)
            for r in author_recs:
                if r['id'] not in seen_ids:
                    r['reason'] = f"✍️ للمؤلف {book.author}"
                    recs.append(r)
                    seen_ids.add(r['id'])
        except Exception as e:
            logger.error(f"[Hybrid] Author fallback error: {e}")

    # --- 3. Content-Based (AI Embeddings) ---
    if len(recs) < limit:
        try:
            current_embedding = None
            if hasattr(book, 'id') and book.id:
                emb_entry = BookEmbedding.query.filter_by(book_id=book.id).first()
                if emb_entry and emb_entry.vector:
                    # Defensive unpickling
                    vec_data = __import__("pickle").loads(emb_entry.vector) if isinstance(emb_entry.vector, bytes) else emb_entry.vector
                    current_embedding = np.array(vec_data, dtype=np.float32).reshape(1, -1)
            
            if current_embedding is not None:
                all_embeds = BookEmbedding.query.all()
                vectors = []
                b_ids = []
                for row in all_embeds:
                    if hasattr(book, 'id') and row.book_id == book.id: continue
                    if row.vector:
                        # Defensive unpickling
                        vec_data = __import__("pickle").loads(row.vector) if isinstance(row.vector, bytes) else row.vector
                        vectors.append(np.array(vec_data, dtype=np.float32))
                        b_ids.append(row.book_id)
                
                if vectors:
                    mat = np.vstack(vectors)
                    sims = cosine_similarity(current_embedding, mat)[0]
                    indices = np.argsort(sims)[::-1][:limit]
                    
                    for idx in indices:
                        score = sims[idx]
                        if score < 0.6: continue
                        target_book = Book.query.get(b_ids[idx])
                        if not target_book: continue
                        bid = target_book.google_id or f"local_{target_book.id}"
                        if bid in seen_ids: continue
                        recs.append(_book_to_dict(target_book, source="AI Similarity", reason=f"🤖 محتوى مشابه ({score:.0%})"))
                        seen_ids.add(bid)
                        if len(recs) >= limit: break
        except Exception as e:
            logger.error(f"[Hybrid] Content-based error: {e}")

    # --- 4. Search Fallback ---
    if len(recs) < limit:
        try:
            query = book.title
            gb_res = fetch_google_books(query, max_results=limit)
            items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
            for it in items or []:
                gid = it.get("id")
                if not gid or gid in seen_ids: continue
                vi = it.get("volumeInfo") or {}
                if gid == book.google_id or vi.get("title") == book.title: continue
                
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img:
                    if img.startswith("http://"): img = img.replace("http://", "https://")
                    if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')

                recs.append({
                    "id": gid,
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors") or []),
                    "cover": img,
                    "source": "Google Books",
                    "reason": "📚 كتب ذات صلة",
                    "rating": vi.get("averageRating"),
                })
                seen_ids.add(gid)
                if len(recs) >= limit: break
        except Exception as e:
            logger.error(f"[Hybrid] Metadata fallback error: {e}")

    return recs[:limit]


@cache.memoize(timeout=43200)
def get_author_books(author_name, exclude_book_id=None, limit=8):
    """
    جلب كتب أخرى لنفس المؤلف.
    """
    if not author_name or author_name.lower() in ['unknown', 'غير معروف']:
        return []

    books_dicts = []
    seen_ids = set()
    if exclude_book_id:
        seen_ids.add(exclude_book_id)

    try:
        local_books = Book.query.filter(
            Book.author.ilike(f"%{author_name}%"),
            Book.google_id != exclude_book_id
        ).limit(5).all()
        
        for b in local_books:
            bid = b.google_id or f"local_{b.id}"
            if bid in seen_ids: continue
            books_dicts.append(_book_to_dict(b, source="Local", reason=f"✍️ للمؤلف {author_name}"))
            seen_ids.add(bid)
    except Exception as e:
        logger.error(f"[AuthorBooks] Local search error: {e}")

    try:
        query = f'inauthor:"{author_name}"'
        gb_res = fetch_google_books(query, max_results=limit)
        items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
        
        for it in items or []:
            if not isinstance(it, dict): continue
            gid = it.get("id")
            if not gid or gid in seen_ids: continue
            
            vi = it.get("volumeInfo") or {}
            authors = vi.get("authors", [])
            if not any(author_name.lower() in a.lower() for a in authors):
                continue

            img = (vi.get("imageLinks") or {}).get("thumbnail")
            if img:
                if img.startswith("http://"): img = img.replace("http://", "https://")
                if '&edge=curl' in img: img = img.replace('&edge=curl', '').replace('&edge=curl&', '&')

            books_dicts.append({
                "id": gid,
                "title": vi.get("title"),
                "author": ", ".join(authors),
                "cover": img,
                "source": "Google Books",
                "reason": f"✍️ للمؤلف {author_name}",
                "rating": vi.get("averageRating"),
            })
            seen_ids.add(gid)
            if len(books_dicts) >= limit: break
            
    except Exception as e:
        logger.error(f"[AuthorBooks] API error: {e}")

    return books_dicts[:limit]


@cache.memoize(timeout=3600)
def get_top_rated(limit=10):
    """
    Get top rated books based on user reviews (BookReview).
    """
    try:
        results = (
            db.session.query(
                BookReview.google_id,
                func.avg(BookReview.rating).label('avg_rating'),
                func.count(BookReview.id).label('review_count')
            )
            .group_by(BookReview.google_id)
            .having(func.count(BookReview.id) >= 1)
            .order_by(func.avg(BookReview.rating).desc(), func.count(BookReview.id).desc())
            .limit(limit)
            .all()
        )
        
        books_dicts = []
        for row in results:
            gid = row.google_id
            avg = float(row.avg_rating)
            count = int(row.review_count)
            
            book = Book.query.filter_by(google_id=gid).first()
            if book:
                d = _book_to_dict(book, source="Community", reason=f"⭐ {avg:.1f} ({count})")
                d['rating'] = avg
                books_dicts.append(d)
            else:
                from ..utils import fetch_book_details
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


@cache.memoize(timeout=600)
def get_because_you_read(user_id, limit=12):
    """
    توصيات بناءً على كتاب قرأه المستخدم مؤخراً.
    """
    if not user_id:
        return {'source_book': None, 'recommendations': []}
    
    try:
        user_books = BookStatus.query.filter(
            BookStatus.user_id == user_id,
            BookStatus.status.in_(['favorite', 'finished'])
        ).order_by(func.random()).limit(5).all()
        
        if not user_books:
            return {'source_book': None, 'recommendations': []}
        
        status_entry = random.choice(user_books)
        source_book = Book.query.get(status_entry.book_id)
        
        if not source_book:
            return {'source_book': None, 'recommendations': []}
        
        recs = get_hybrid_recommendations(user_id, source_book, limit=limit)
        
        for rec in recs:
            rec['reason'] = f"📖 لأنك قرأت: {source_book.title[:30]}..."
        
        source_dict = _book_to_dict(source_book, source="Reference")
        
        logger.info(f"[BecauseYouRead] Generated {len(recs)} recs based on '{source_book.title}'")
        return {
            'source_book': source_dict,
            'recommendations': recs
        }
        
    except Exception as e:
        logger.error(f"[BecauseYouRead] Error: {e}")
        return {'source_book': None, 'recommendations': []}


@cache.memoize(timeout=1800)
def get_similar_users_favorites(user_id, limit=12):
    """
    جلب الكتب المفضلة لدى مستخدمين لهم ذوق مشابه.
    """
    if not user_id:
        return []
    
    try:
        user_favorites = BookStatus.query.filter(
            BookStatus.user_id == user_id,
            BookStatus.status == 'favorite'
        ).all()
        
        user_book_ids = {s.book_id for s in user_favorites}
        
        if not user_book_ids:
            return []
        
        similar_users = db.session.query(
            BookStatus.user_id,
            func.count(BookStatus.id).label('overlap')
        ).filter(
            BookStatus.book_id.in_(user_book_ids),
            BookStatus.status == 'favorite',
            BookStatus.user_id != user_id
        ).group_by(BookStatus.user_id).order_by(
            func.count(BookStatus.id).desc()
        ).limit(10).all()
        
        if not similar_users:
            return []
        
        similar_user_ids = [u[0] for u in similar_users]
        
        their_favorites = BookStatus.query.filter(
            BookStatus.user_id.in_(similar_user_ids),
            BookStatus.status == 'favorite',
            ~BookStatus.book_id.in_(user_book_ids)
        ).all()
        
        book_counts = Counter(s.book_id for s in their_favorites)
        top_book_ids = [bid for bid, _ in book_counts.most_common(limit)]
        
        books = []
        for book_id in top_book_ids:
            book = Book.query.get(book_id)
            if book:
                count = book_counts[book_id]
                books.append(_book_to_dict(
                    book,
                    source="Similar Users",
                    reason=f"❤️ أحبه {count} قارئ يشبهونك في الذوق"
                ))
        
        logger.info(f"[SimilarUsers] Found {len(books)} favorites from similar users")
        return books
        
    except Exception as e:
        logger.error(f"[SimilarUsers] Error: {e}")
        return []





def get_genre_explorer(user_id, limit=12):
    """
    Suggestions for new genres.
    """
    return None
