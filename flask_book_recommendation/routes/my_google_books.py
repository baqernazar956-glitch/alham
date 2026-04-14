# routes/google_books.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from ..extensions import db
from ..models import Book
# استيراد الأداة الموحدة للبحث
from ..utils import fetch_google_books

google_bp = Blueprint("google", __name__, url_prefix="/google")

def _item(v):
    """تنسيق البيانات للعرض في القالب"""
    vi = v.get("volumeInfo", {}) or {}
    imgs = vi.get("imageLinks", {}) or {}
    
    # اختيار أفضل صورة متاحة
    cover = (
        imgs.get("extraLarge") 
        or imgs.get("large") 
        or imgs.get("medium")
        or imgs.get("thumbnail") 
        or imgs.get("smallThumbnail")
    )
    
    # إصلاح روابط الصور
    if cover and cover.startswith("http://"):
        cover = "https://" + cover[len("http://"):]
        
    return dict(
        google_id=v.get("id"),
        title=vi.get("title") or "Untitled",
        author=", ".join(vi.get("authors", [])) if vi.get("authors") else None,
        description=vi.get("description"),
        cover_url=cover
    )

@google_bp.get("/search")
def search():
    q = (request.args.get("q") or "").strip() or "python"
    
    # --- التغيير هنا: استخدام الدالة الموحدة ---
    # نطلب 20 نتيجة كما كان في الكود الأصلي
    items_raw, _ = fetch_google_books(q, max_results=20)
    
    # تحويل النتائج الخام إلى الشكل المطلوب للقالب
    items = [_item(v) for v in items_raw]
    
    return render_template("google_search.html", q=q, items=items)

@google_bp.post("/import")
def import_book():
    # هذا الجزء يبقى كما هو لأنه يتعامل مع قاعدة البيانات فقط
    gid   = request.form.get("google_id")
    title = request.form.get("title")
    author= request.form.get("author")
    desc  = request.form.get("description")
    cover = request.form.get("cover_url")

    if not gid:
        flash("Missing Google ID", "danger")
        return redirect(url_for("google.search"))

    # التحقق من التكرار قبل الإضافة
    if Book.query.filter_by(google_id=gid).first():
        flash("Book already exists", "info")
        return redirect(url_for("main.books"))

    b = Book(
        google_id=gid, 
        title=title, 
        author=author, 
        description=desc, 
        cover_url=cover
    )
    db.session.add(b)
    db.session.commit()
    
    flash("Book imported", "success")
    return redirect(url_for("main.books"))