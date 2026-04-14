# -*- coding: utf-8 -*-
"""
User events — log_user_view + analyze_user_profile_with_ai.
"""
import logging
from datetime import datetime

from ..models import (
    Book, UserRatingCF, SearchHistory,
    UserPreference, UserBookView
)
from ..extensions import db

logger = logging.getLogger(__name__)


def log_user_view(user_id, book):
    """
    تسجيل مشاهدة المستخدم للكتاب.
    يتم استدعاؤها عند فتح صفحة التفاصيل.
    Now also creates a UserEvent record for richer analytics.
    """
    try:
        if not user_id: return
        
        b_id = getattr(book, 'id', None)
        g_id = getattr(book, 'google_id', None)
        
        criteria = {'user_id': user_id}
        if g_id:
            criteria['google_id'] = g_id
        elif b_id:
            criteria['book_id'] = b_id
        else:
            return

        view = None
        if g_id:
            view = UserBookView.query.filter_by(user_id=user_id, google_id=g_id).first()
        if not view and b_id:
            view = UserBookView.query.filter_by(user_id=user_id, book_id=b_id).first()
            
        if view:
            view.view_count += 1
            view.last_viewed_at = datetime.utcnow()
        else:
            view = UserBookView(
                user_id=user_id,
                book_id=b_id if hasattr(book, 'id') and isinstance(book.id, int) else None,
                google_id=g_id,
                view_count=1
            )
            db.session.add(view)

        # Create UserEvent for richer analytics
        try:
            from ..models import UserEvent
            event = UserEvent(
                user_id=user_id,
                event_type='view',
                book_google_id=g_id,
            )
            db.session.add(event)

        except Exception:
            pass  # UserEvent table may not exist yet (pre-migration)

        db.session.commit()
        
        # --- 🆕 User Embedding Update (Phase 2) ---
        try:
            from ai_book_recommender.feature_store.user_embeddings import user_embedding_manager
            user_embedding_manager.update_user_embedding(user_id, book_id=b_id, google_id=g_id)
        except Exception as e_emb:
            logger.error(f"Embedding update error: {e_emb}")
            
        # --- 🆕 Online Learning Feedback Update ---
        try:
            from ai_book_recommender.engine import get_engine
            b_id_val = str(g_id or b_id or "")
            if b_id_val:
                get_engine().record_feedback(
                    user_id=user_id,
                    item_id=b_id_val,
                    feedback_type="view",
                    value=1.0
                )
        except Exception as e_ol:
            logger.error(f"Online learning feedback error (view): {e_ol}")
        # ------------------------------------------

    except Exception as e:
        logger.error(f"[LogView] Error: {e}")


def analyze_user_profile_with_ai(user_id):
    """
    تحليل سلوك المستخدم باستخدام Generative AI (Gemini).
    يقرأ: المشاهدات, التقييمات, المفضلة, البحث.
    يكتب: تحديث UserPreference.
    """
    import os
    import json
    import requests
    from datetime import timedelta
    
    try:
        views = UserBookView.query.filter_by(user_id=user_id).order_by(UserBookView.last_viewed_at.desc()).limit(15).all()
        viewed_books = []
        for v in views:
            title = "Unknown"
            if v.book: title = v.book.title
            elif v.google_id:
                 b = Book.query.filter_by(google_id=v.google_id).first()
                 if b: title = b.title
            
            if title != "Unknown":
                viewed_books.append(title)
        
        ratings = UserRatingCF.query.filter_by(user_id=user_id).filter(UserRatingCF.rating >= 4).limit(10).all()

        searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).limit(10).all()
        search_terms = [s.query for s in searches if s.query]

        if not viewed_books and not search_terms:
            return

        prompt = f"""
        Analyze this users reading behavior and suggest interests.
        
        Viewed Books: {", ".join(viewed_books)}
        Search Terms: {", ".join(search_terms)}
        
        Task:
        1. Identify 5 core topics/genres this user is interested in.
        2. Format as JSON list of objects: {{"topic": "topic_name", "weight": float_1_to_3, "reason_en": "reason", "reason_ar": "reason_in_arabic"}}
        3. Topics should be broad enough for book search (e.g. "Science Fiction", "Python Programming").
        """

        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key: return

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            },
            timeout=10
        )
        
        if response.ok:
            data = response.json()
            text_resp = data['candidates'][0]['content']['parts'][0]['text']
            suggestions = json.loads(text_resp)
            
            for item in suggestions:
                topic = item.get("topic")
                weight = item.get("weight", 1.0)
                
                if not topic: continue
                
                pref = UserPreference.query.filter_by(user_id=user_id, topic=topic).first()
                if pref:
                    pref.weight = min(5.0, (pref.weight + weight) / 2 + 1)
                else:
                    new_pref = UserPreference(user_id=user_id, topic=topic, weight=weight)
                    db.session.add(new_pref)
            
            db.session.commit()
            logger.info(f"[AI Analysis] Updated preferences for user {user_id}")

    except Exception as e:
        logger.error(f"[AI Analysis] Error: {e}")

def update_user_model_online(user_id: int, event):
    """
    🔄 تحديث نظام التوصيات للمستخدم مباشرة في الوقت الفعلي بناءً على الحدث المدخل.
    
    Enhanced behavioral weight system:
    - Click/View: +0.3 (initial interest signal)
    - Long View (2+ min): +0.8 (deep interest)
    - Search: +1.5 (strongest explicit intent)
    - Rate 4+: +1.0 (positive endorsement)
    - Rate 3-: -0.3 (negative signal)
    - Favorite/Later: +0.8 (save intent)
    - Finished reading: +2.0 (strongest behavioral signal)
    - Abandon: -0.5 (disinterest)
    
    Also handles:
    - Dynamic interest creation: new categories -> interests after 3+ views
    - Auto-trigger AI analysis every 10 interactions
    - Cache invalidation after updates
    """
    if not user_id or not event:
        return

    # استخراج التصنيف/الموضوع للكتاب المعني
    topics = []
    if event.book_google_id:
        book = Book.query.filter_by(google_id=event.book_google_id).first()
        if book and book.categories:
            # Parse all categories
            cats = [c.strip() for c in book.categories.split(",")]
            topics = [c for c in cats if c]
    
    if not topics:
        topics = ['general']

    # ── Calculate weight change based on event type ──
    weight_change = 0.0
    
    if event.event_type == 'finish' or event.event_type == 'finished':
        weight_change = 2.0  # Strongest signal: completed a book
    elif event.event_type == 'view' or event.event_type == 'click':
        duration = getattr(event, 'duration_seconds', 0) or 0
        weight_change = 0.3  # Base: clicking/viewing = initial interest
        if duration > 120:  # More than 2 minutes = deep interest
            weight_change = 0.8
        elif duration > 60:  # More than 1 minute
            weight_change = 0.5
    elif event.event_type == 'search':
        weight_change = 1.5  # Search = highest explicit intent
    elif event.event_type == 'rate':
        rating = getattr(event, 'metadata_json', None)
        try:
            import json
            if rating:
                meta = json.loads(str(rating)) if isinstance(rating, str) else rating
                rating_val = float(meta.get('rating', 3))
            else:
                rating_val = 3.0
        except Exception:
            rating_val = 3.0
        
        if rating_val >= 4:
            weight_change = 1.0  # Positive endorsement
        elif rating_val <= 2:
            weight_change = -0.3  # Negative signal
    elif event.event_type in ('favorite', 'later', 'save'):
        weight_change = 0.8  # Save intent
    elif event.event_type == 'abandon':
        scroll = getattr(event, 'scroll_depth', 0.0) or 0.0
        if scroll < 0.2:
            weight_change = -0.5  # Quick abandon = disinterest
        elif scroll < 0.5:
            weight_change = -0.2

    if weight_change == 0.0:
        return

    # ── Update UserPreference for each topic ──
    updated_topics = []
    for topic in topics:
        preference = UserPreference.query.filter_by(user_id=user_id, topic=topic).first()
        if not preference:
            # Only create new preferences for positive signals
            if weight_change > 0:
                preference = UserPreference(user_id=user_id, topic=topic, weight=weight_change)
                db.session.add(preference)
                updated_topics.append(topic)
        else:
            preference.weight = max(0.0, preference.weight + weight_change)
            updated_topics.append(topic)
    
    try:
        db.session.commit()
        if updated_topics:
            logger.info(
                f"[BehaviorUpdate] user={user_id} event={event.event_type} "
                f"topics={updated_topics} change={weight_change:+.1f}"
            )
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating online model for user {user_id}: {e}")
        return

    # ── 🧠 Dynamic Interest Creation ──
    # If a user has viewed books in a new category 3+ times, auto-add as interest
    if weight_change > 0:
        try:
            for topic in topics:
                if topic == 'general':
                    continue
                pref = UserPreference.query.filter_by(user_id=user_id, topic=topic).first()
                if pref and pref.weight >= 3.0:
                    # Check if this is already a high-weight interest
                    from ..models import UserGenre, Genre
                    genre = Genre.query.filter(Genre.name.ilike(topic)).first()
                    if genre:
                        existing = UserGenre.query.filter_by(
                            user_id=user_id, genre_id=genre.id
                        ).first()
                        if not existing:
                            db.session.add(UserGenre(user_id=user_id, genre_id=genre.id))
                            db.session.commit()
                            logger.info(
                                f"[DynamicInterest] Auto-added genre '{topic}' "
                                f"for user {user_id} (weight={pref.weight:.1f})"
                            )
        except Exception as e:
            logger.error(f"[DynamicInterest] Error: {e}")

    # ── 🤖 Auto-trigger AI Profile Analysis every 10 interactions ──
    try:
        from ..models import UserEvent
        interaction_count = UserEvent.query.filter_by(user_id=user_id).count()
        if interaction_count > 0 and interaction_count % 10 == 0:
            import threading
            from flask import current_app
            app = current_app._get_current_object()
            
            def _bg_analyze():
                with app.app_context():
                    try:
                        analyze_user_profile_with_ai(user_id)
                        logger.info(f"[AutoAnalysis] AI profile analysis triggered for user {user_id} (interaction #{interaction_count})")
                    except Exception as e:
                        logger.error(f"[AutoAnalysis] Error: {e}")
            
            thread = threading.Thread(target=_bg_analyze, daemon=True)
            thread.start()
    except Exception:
        pass

    # ── 🗑️ Cache Invalidation for immediate recommendation refresh ──
    try:
        from ..extensions import cache
        cache.delete(f"home_full_{user_id}")
        cache.delete(f"home_feed_{user_id}")
    except Exception:
        pass


def decay_preferences():
    """
    تتلاشى التفضيلات تدريجياً مع مرور الوقت للتركيز على الاهتمامات الأحدث.
    هذه الدالة يُفترض أن تُشغل عبر مهمة مجدولة (Celery Beat مثلاً) أسبوعياً كـ: `كل أحد 2 صباحاً`
    
    Note: Explicit onboarding interests (weight >= 100) decay slower (0.98)
    while behavioral interests (weight < 100) decay faster (0.90) to be responsive.
    """
    try:
        # Explicit interests decay slowly
        db.session.execute(
            'UPDATE user_preferences SET weight = weight * 0.98 WHERE weight >= 100'
        )
        # Behavioral interests decay faster
        db.session.execute(
            'UPDATE user_preferences SET weight = weight * 0.90 WHERE weight < 100'
        )
        db.session.commit()
        logger.info("Executed weekly preference weight decay (explicit: 0.98, behavioral: 0.90).")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error decaying preferences: {e}")
