# -*- coding: utf-8 -*-
"""
Real-time API — Instagram-style dynamic feed updates.

Endpoints
---------
POST /api/track          — Log an interaction (view, favorite, rate, search)
GET  /api/realtime_feed  — Return an updated "For You" section based on session
"""
import logging
import uuid
from datetime import datetime

from flask import Blueprint, request, jsonify, render_template, current_app
from flask_login import current_user

from ..extensions import db, csrf
from ..models import UserEvent, Book

logger = logging.getLogger(__name__)

realtime_bp = Blueprint("realtime", __name__)


# ═══════════════════════════════════════════════════════════════════════
# POST /api/track — Lightweight interaction logger
# ═══════════════════════════════════════════════════════════════════════
@realtime_bp.route("/api/track", methods=["POST"])
@csrf.exempt
def track_interaction():
    """
    Receive a user interaction from the frontend and persist it.

    Expected JSON body::

        {
            "event_type": "view" | "favorite" | "rate" | "search" | "click",
            "book_id": "google_id_string",
            "session_id": "client-generated-uuid",
            "metadata": {}           // optional extra data
        }

    Returns ``{"ok": true, "interaction_count": N}`` where *N* is the
    number of interactions in this session so far (so the client knows
    when to refresh the feed).
    """
    if not current_user.is_authenticated:
        return jsonify(ok=True, interaction_count=0)

    try:
        data = request.get_json(silent=True) or {}
        event_type = data.get("event_type", "view")
        book_id = data.get("book_id")
        session_id = data.get("session_id", "")
        extra_meta = data.get("metadata")

        if not book_id:
            return jsonify(ok=False, error="missing book_id"), 400

        # Persist event
        event = UserEvent(
            user_id=current_user.id,
            event_type=event_type,
            book_google_id=str(book_id),
            session_id=session_id or None,
            duration_seconds=data.get("duration"),
            scroll_depth=data.get("scroll_depth"),
            metadata_json=str(extra_meta) if extra_meta else None,
        )
        db.session.add(event)
        db.session.commit()

        # Update user model online (non-blocking best-effort)
        try:
            from ..recommender.events import update_user_model_online
            update_user_model_online(current_user.id, event)
        except Exception:
            pass

        # Count interactions in this session
        if session_id:
            count = (
                UserEvent.query
                .filter_by(user_id=current_user.id, session_id=session_id)
                .count()
            )
        else:
            count = 1

        # ── 🗑️ Cache Invalidation — force fresh recommendations ──
        try:
            from ..extensions import cache
            cache.delete(f"home_full_{current_user.id}")
            cache.delete(f"home_feed_{current_user.id}")
        except Exception:
            pass

        # ── 🤖 Auto-trigger AI Analysis at 5+ session interactions ──
        if count >= 5 and count % 5 == 0:
            try:
                import threading
                from flask import current_app
                app = current_app._get_current_object()
                uid = current_user.id
                
                def _bg_analysis():
                    with app.app_context():
                        try:
                            from ..recommender.events import analyze_user_profile_with_ai
                            analyze_user_profile_with_ai(uid)
                            logger.info(f"[RealtimeTrack] AI analysis triggered for user {uid} (session interaction #{count})")
                        except Exception as e:
                            logger.error(f"[RealtimeTrack] AI analysis error: {e}")
                
                thread = threading.Thread(target=_bg_analysis, daemon=True)
                thread.start()
            except Exception:
                pass

        return jsonify(ok=True, interaction_count=count)

    except Exception as e:
        logger.error(f"[RealtimeTrack] Error: {e}", exc_info=True)
        db.session.rollback()
        return jsonify(ok=False, error=str(e)), 500


# ═══════════════════════════════════════════════════════════════════════
# GET /api/realtime_feed — Session-adaptive recommendations
# ═══════════════════════════════════════════════════════════════════════
@realtime_bp.route("/api/realtime_feed")
def realtime_feed():
    """
    Return an HTML snippet containing a "Updated for You" carousel
    section based on the user's current session interactions.

    Query params
    ------------
    session_id : str   — The client's session UUID to scope recent events.
    """
    if not current_user.is_authenticated:
        return jsonify(success=False, html="")

    try:
        session_id = request.args.get("session_id", "")

        # Fetch recent session events
        query = UserEvent.query.filter_by(user_id=current_user.id)
        if session_id:
            query = query.filter_by(session_id=session_id)
        recent_events = (
            query.order_by(UserEvent.created_at.desc()).limit(10).all()
        )

        if not recent_events:
            return jsonify(success=False, html="")

        # Build session_events list for the adaptive recommender
        session_events = [
            {"event_type": e.event_type, "book_id": e.book_google_id}
            for e in recent_events
            if e.book_google_id
        ]

        from ..recommender.session_adaptive import get_session_adaptive_recommendations

        recs = get_session_adaptive_recommendations(
            user_id=current_user.id,
            session_events=session_events,
            limit=12,
        )

        if not recs:
            return jsonify(success=False, html="")

        # Determine what the user was browsing for the subtitle
        book_ids = [e.book_google_id for e in recent_events if e.book_google_id]
        top_topic = _extract_session_topic(book_ids)

        html = render_template(
            "components/realtime_section.html",
            realtime_books=recs,
            session_topic=top_topic,
        )

        return jsonify(success=True, html=html)

    except Exception as e:
        logger.error(f"[RealtimeFeed] Error: {e}", exc_info=True)
        return jsonify(success=False, html="")


def _extract_session_topic(google_ids):
    """Return the most common category from a list of google_ids."""
    from collections import Counter

    if not google_ids:
        return "اهتماماتك"

    books = Book.query.filter(Book.google_id.in_(google_ids)).all()
    cats = Counter()
    for b in books:
        if b.categories:
            for c in b.categories.split(","):
                c = c.strip()
                if c:
                    cats[c] += 1

    if cats:
        return cats.most_common(1)[0][0]
    return "اهتماماتك"
