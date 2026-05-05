import re
import os

file_path = r"c:\Users\al6md\Desktop\project alham\flask_book_recommendation_starter\flask_book_recommendation\routes\public.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update generate_ai_summary
pattern_summary = re.compile(r'def generate_ai_summary\(gid\):.*?try:.*?# جلب معلومات الكتاب.*?if not book_data:', re.DOTALL)
replacement_summary = """def generate_ai_summary(gid):
    \"\"\"توليد ملخص ذكي للكتاب باستخدام AI\"\"\"
    from ..utils import generate_book_summary
    from flask import jsonify
    
    logger.debug(f"[AI Summary] Requested for gid: {gid}")
    try:
        # جلب معلومات الكتاب باستخدام الدالة الموحدة الذكية
        book_data = fetch_book_details(gid)
        
        # Fallback للمعلومات المحلية إذا لم نجدها في الـ APIs
        if not book_data:
            from ..models import Book
            local_id = int(gid) if gid.isdigit() else None
            book = Book.query.get(local_id) if local_id else Book.query.filter_by(google_id=gid).first()
            if book:
                from ..recommender.helpers import _book_to_dict
                book_data = _book_to_dict(book)
        
        if not book_data:"""

content = re.sub(r'def generate_ai_summary\(gid\):.*?if not book_data:', replacement_summary, content, flags=re.DOTALL, count=1)

# 2. Update generate_why_like
replacement_why = """def generate_why_like(gid):
    \"\"\"تحليل لماذا قد يعجب الكتاب المستخدم\"\"\"
    from ..utils import generate_why_you_like
    from flask import jsonify
    
    logger.debug(f"[WhyLike] Requested for gid: {gid}")
    try:
        # جلب معلومات الكتاب باستخدام الدالة الموحدة الذكية
        book_data = fetch_book_details(gid)
        
        # Fallback للمعلومات المحلية
        if not book_data:
            from ..models import Book
            local_id = int(gid) if gid.isdigit() else None
            book = Book.query.get(local_id) if local_id else Book.query.filter_by(google_id=gid).first()
            if book:
                from ..recommender.helpers import _book_to_dict
                book_data = _book_to_dict(book)
        
        if not book_data:"""

content = re.sub(r'def generate_why_like\(gid\):.*?if not book_data:', replacement_why, content, flags=re.DOTALL, count=1)

# 3. Update generate_plan_route
replacement_plan = """def generate_plan_route(gid):
    \"\"\"توليد خطة قراءة للكتاب\"\"\"
    from ..utils import generate_reading_plan
    from flask import jsonify
    
    logger.debug(f"[ReadingPlan] Requested for gid: {gid}")
    try:
        # جلب معلومات الكتاب باستخدام الدالة الموحدة الذكية
        book_data = fetch_book_details(gid)
        
        # Fallback للمعلومات المحلية
        if not book_data:
            from ..models import Book
            local_id = int(gid) if gid.isdigit() else None
            book = Book.query.get(local_id) if local_id else Book.query.filter_by(google_id=gid).first()
            if book:
                from ..recommender.helpers import _book_to_dict
                book_data = _book_to_dict(book)
        
        if not book_data:"""

content = re.sub(r'def generate_plan_route\(gid\):.*?if not book_data:', replacement_plan, content, flags=re.DOTALL, count=1)

# 4. Update chat_with_book_route
replacement_chat = """def chat_with_book_route(gid):
    \"\"\"الدردشة مع سياق الكتاب\"\"\"
    from ..utils import chat_with_book_context
    from flask import jsonify
    
    logger.debug(f"[Chat] Requested for gid: {gid}")
    try:
        message = request.json.get("message", "")
        history = request.json.get("history", [])
        
        # جلب معلومات الكتاب باستخدام الدالة الموحدة الذكية
        book_data = fetch_book_details(gid)
        
        # Fallback للمعلومات المحلية
        if not book_data:
            from ..models import Book
            local_id = int(gid) if gid.isdigit() else None
            book = Book.query.get(local_id) if local_id else Book.query.filter_by(google_id=gid).first()
            if book:
                from ..recommender.helpers import _book_to_dict
                book_data = _book_to_dict(book)
        
        if not book_data:"""

content = re.sub(r'def chat_with_book_route\(gid\):.*?if not book_data:', replacement_chat, content, flags=re.DOTALL, count=1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched public.py")
