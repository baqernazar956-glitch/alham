#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
سكربت توليد Embeddings لجميع الكتب
====================================
يقوم بتوليد وحفظ embeddings لجميع الكتب الموجودة في قاعدة البيانات
التي لا تملك embeddings حالياً.

الاستخدام:
    python generate_embeddings.py
    
    أو مع تحديد batch size:
    python generate_embeddings.py --batch 10
"""

import sys
import time
import argparse
from datetime import datetime

# إضافة المسار للمشروع
sys.path.insert(0, '.')

from flask_book_recommendation.app import create_app
from flask_book_recommendation.extensions import db
from flask_book_recommendation.models import Book, BookEmbedding
from flask_book_recommendation.utils import get_book_embedding


def generate_all_embeddings(batch_size=20, delay_between_batches=2):
    """
    توليد embeddings لجميع الكتب التي لا تملك embedding.
    
    Args:
        batch_size: عدد الكتب في كل دفعة
        delay_between_batches: الانتظار بين الدفعات (بالثواني)
    """
    app = create_app()
    
    with app.app_context():
        print("=" * 60)
        print("🚀 بدء توليد Embeddings للكتب")
        print("=" * 60)
        
        # إحصائيات
        total_books = Book.query.count()
        books_with_embedding = db.session.query(BookEmbedding.book_id).count()
        books_without_embedding = total_books - books_with_embedding
        
        print(f"📚 إجمالي الكتب: {total_books}")
        print(f"✅ كتب لديها embedding: {books_with_embedding}")
        print(f"❌ كتب بدون embedding: {books_without_embedding}")
        print("-" * 60)
        
        if books_without_embedding == 0:
            print("🎉 جميع الكتب لديها embeddings!")
            return
        
        # جلب الكتب التي لا تملك embedding
        existing_book_ids = db.session.query(BookEmbedding.book_id).all()
        existing_ids = {row[0] for row in existing_book_ids}
        
        books_to_process = Book.query.filter(~Book.id.in_(existing_ids)).all()
        
        total_to_process = len(books_to_process)
        processed = 0
        success = 0
        failed = 0
        
        start_time = datetime.now()
        
        print(f"⏳ سيتم معالجة {total_to_process} كتاب...")
        print()
        
        # معالجة على دفعات
        for i in range(0, total_to_process, batch_size):
            batch = books_to_process[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_to_process + batch_size - 1) // batch_size
            
            print(f"📦 الدفعة {batch_num}/{total_batches}")
            
            for book in batch:
                processed += 1
                
                # شريط التقدم
                progress = (processed / total_to_process) * 100
                bar_length = 30
                filled = int(bar_length * processed / total_to_process)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"\r   [{bar}] {progress:.1f}% - {book.title[:40]}...", end="", flush=True)
                
                # توليد embedding
                embedding = get_book_embedding(book)
                
                if embedding:
                    try:
                        new_embed = BookEmbedding(book_id=book.id, vector=embedding)
                        db.session.add(new_embed)
                        db.session.commit()
                        success += 1
                    except Exception as e:
                        db.session.rollback()
                        failed += 1
                        print(f"\n   ❌ خطأ في الحفظ: {e}")
                else:
                    failed += 1
                
                # تأخير صغير لتجنب rate limiting
                time.sleep(0.3)
            
            print()  # سطر جديد بعد كل دفعة
            
            # تأخير بين الدفعات
            if i + batch_size < total_to_process:
                print(f"   ⏸️ انتظار {delay_between_batches}s قبل الدفعة التالية...")
                time.sleep(delay_between_batches)
        
        # النتائج النهائية
        elapsed = datetime.now() - start_time
        print()
        print("=" * 60)
        print("📊 النتائج النهائية")
        print("=" * 60)
        print(f"✅ نجاح: {success}")
        print(f"❌ فشل: {failed}")
        print(f"⏱️ الوقت الإجمالي: {elapsed}")
        print(f"📈 معدل: {success/max(elapsed.total_seconds(),1):.2f} كتاب/ثانية")
        print("=" * 60)


def test_single_embedding():
    """اختبار توليد embedding لكتاب واحد"""
    app = create_app()
    
    with app.app_context():
        book = Book.query.first()
        if not book:
            print("❌ لا يوجد كتب في قاعدة البيانات!")
            return
        
        print(f"🧪 اختبار على كتاب: {book.title}")
        print(f"   المؤلف: {book.author}")
        print(f"   الوصف: {(book.description or 'لا يوجد')[:100]}...")
        print()
        
        embedding = get_book_embedding(book)
        
        if embedding:
            print(f"✅ نجح التوليد!")
            print(f"   أبعاد الـ vector: {len(embedding)}")
            print(f"   أول 5 قيم: {embedding[:5]}")
        else:
            print("❌ فشل التوليد!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="توليد embeddings للكتب")
    parser.add_argument("--batch", type=int, default=20, help="حجم الدفعة (افتراضي: 20)")
    parser.add_argument("--delay", type=float, default=2, help="التأخير بين الدفعات بالثواني")
    parser.add_argument("--test", action="store_true", help="اختبار على كتاب واحد فقط")
    
    args = parser.parse_args()
    
    if args.test:
        test_single_embedding()
    else:
        generate_all_embeddings(batch_size=args.batch, delay_between_batches=args.delay)
