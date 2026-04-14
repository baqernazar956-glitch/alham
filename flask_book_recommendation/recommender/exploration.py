# -*- coding: utf-8 -*-
"""
Content Exploration System (UCB1 Bandit)
========================================
Surfaces newer or less-established books using Upper Confidence Bound algorithms.
"""
import math
import logging
import random
from datetime import datetime, timedelta
from sqlalchemy import func
from flask import current_app

from ..models import Book, UserEvent, UserBookView, db
from ..utils import fetch_google_books

logger = logging.getLogger(__name__)

class UCB1Explorer:
    """
    مستكشف UCB1 (Upper Confidence Bound).
    يوازن بين Exploration (تجارب كتب جديدة) و Exploitation (عرض كتب ذات CTR عالٍ).
    """
    
    @staticmethod
    def _compute_ucb(book_stats, total_impressions):
        """
        حساب UCB1 Score.
        """
        impressions = book_stats.get('impressions', 0)
        clicks = book_stats.get('clicks', 0)
        
        # إذا لم يُعرض الكتيب أبدًا، نعطيه أولوية قصوى لضمان ظهوره الأولي (Exploration)
        if impressions == 0:
            return 99.0
            
        ctr = clicks / impressions
        # عامل الاستكشاف (Exploration Term): 
        # يتناسب طردياً مع إجمالي المشاهدات وعكسياً مع مرات ظهور الكتاب المعني
        exploration_term = math.sqrt(2 * math.log(max(1, total_impressions)) / impressions)
        
        return ctr + exploration_term

    @classmethod
    def get_exploration_pool(cls, limit=10):
        """
        جلب قائمة بالكتب المرشحة من خلال محرك البحث المكتشف.
        نستهدف تبديل نسبة الـ 10% من المحتوى من هنا.
        """
        try:
            # فلترة الكتب الأحدث (أضيف في آخر 30 يوماً كـ مثال)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_books = Book.query.filter(Book.created_at >= thirty_days_ago).limit(100).all()
            
            if not recent_books:
                # Fallback إن لم تكن هناك كتب حديثة
                recent_books = Book.query.order_by(db.func.random()).limit(50).all()
                if not recent_books:
                    return []

            # 1. جمع الإحصائيات (Impressions & Clicks) لكل كتاب
            book_ids = [b.id for b in recent_books]
            google_ids = [b.google_id for b in recent_books if b.google_id]
            
            # (تقريب) الانطباعات من UserBookView والضغطات/القراءات من UserEvent
            views_query = db.session.query(
                UserBookView.google_id, 
                func.sum(UserBookView.view_count).label('impressions')
            ).filter(UserBookView.google_id.in_(google_ids)).group_by(UserBookView.google_id).all()
            
            clicks_query = db.session.query(
                UserEvent.book_google_id, 
                func.count(UserEvent.id).label('clicks')
            ).filter(UserEvent.book_google_id.in_(google_ids), UserEvent.event_type == 'click').group_by(UserEvent.book_google_id).all()
            
            views_dict = {row.google_id: row.impressions for row in views_query}
            clicks_dict = {row.book_google_id: row.clicks for row in clicks_query}
            
            # حساب الإجمالي لتناغم المعادلات
            total_impressions = sum(views_dict.values()) + 1 # تجنب القسمة على صفر
            
            # 2. حساب رصيد UCB
            scored_books = []
            for b in recent_books:
                if not b.google_id: continue
                
                stats = {
                    'impressions': views_dict.get(b.google_id, 0),
                    'clicks': clicks_dict.get(b.google_id, 0)
                }
                
                score = cls._compute_ucb(stats, total_impressions)
                scored_books.append((score, b))
                
            # 3. ترتيب الكتب حسب UCB تنازلياً
            scored_books.sort(key=lambda x: x[0], reverse=True)
            
            # تعيين البيانات للطباعة
            results = []
            for score, b in scored_books[:limit]:
                results.append({
                    "id": b.google_id,
                    "title": b.title,
                    "author": b.author or "Unknown",
                    "cover": b.cover_url,
                    "reason": "✨ اكتشاف جديد (UCB Bandit)",
                    "source": "Exploration",
                    "score": score
                })
                
            return results
            
        except Exception as e:
            logger.error(f"[Exploration] Error generating UCB pool: {e}")
            return []

    @classmethod
    def inject_exploration(cls, existing_books, limit_injection=None):
        """
        حقن (Inject) نسبة 10% (أو الحد المعطى) من الكتب المكتشفة في قائمة التوصيات الأولية.
        """
        if not existing_books:
            return cls.get_exploration_pool(limit=4)
            
        injection_count = limit_injection if limit_injection else max(1, len(existing_books) // 10)
        
        # استدعاء الروتين لاستخلاص الـ Pool
        exploration_books = cls.get_exploration_pool(limit=injection_count)
        if not exploration_books: return existing_books
        
        existing_gids = {str(b.get('id')): True for b in existing_books}
        
        # التصفية والمزج
        for eb in exploration_books:
            if str(eb.get('id')) not in existing_gids:
                # استبدال عشوائي لعنصر بأسفل القائمة (أو إدراج)
                idx = random.randint(min(3, len(existing_books)-1), len(existing_books)-1)
                existing_books.insert(idx, eb)
                # حفاظ على الطول
                if len(existing_books) > len(existing_books) - injection_count + len(exploration_books): # check simplified logic normally
                    existing_books.pop()
                    
        return existing_books
