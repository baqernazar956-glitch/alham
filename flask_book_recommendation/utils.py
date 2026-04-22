import requests
import json
import os
import re
import logging
from dotenv import load_dotenv

# تحميل ملف .env لضمان تحميل مفاتيح API
from .extensions import cache

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# -----------------------------------------------------------
# 1. Google Books (تم الإصلاح هنا)
# -----------------------------------------------------------
API_URL = "https://www.googleapis.com/books/v1/volumes"
load_dotenv(dotenv_path, override=True)

def clean_book_title(title):
    """تنظيف وتنسيق عناوين الكتب الطويلة والمعقدة"""
    if not title: return ""
    
    # تفكيك العنوان إذا كان يحتوي على نقطتين (غالباً ما يكون الباقي هو العنوان الفرعي)
    if ":" in title:
        parts = title.split(":", 1)
        main_title = parts[0].strip()
        # إذا كان العنوان الرئيسي قصيراً جداً (مثلاً "A: The Story of..."), نأخذ العنوان كاملاً
        if len(main_title) > 8:
            title = main_title
            
    # إزالة الأقواس والمعلومات الإضافية المتكررة في مصادر مثل Gutenberg
    title = title.split("(", 1)[0].strip()
    title = title.split("[", 1)[0].strip()
    
    # تقليص الطول إذا كان لا يزال طويلاً جداً (مثلاً أكثر من 80 حرف)
    if len(title) > 85:
        title = title[:82] + "..."
        
    return title

# @cache.memoize(timeout=1800) # DISABLED: We need completely fresh books for FreshInject
@cache.memoize(timeout=86400)
def fetch_google_books(query, max_results=12, start_index=0, order_by="relevance"):
    # Google Books API supports max 40 results per request
    safe_max = min(max_results, 40)
    
    params = {
        "q": query, "maxResults": safe_max,
        "startIndex": start_index, "orderBy": order_by, "printType": "books",
        "key": os.environ.get("GOOGLE_BOOKS_API_KEY")
    }
    headers = {"User-Agent": "Elham-Platform/1.0 (Book Discovery)"}
    
    try:
        _utils_logger = logging.getLogger('flask_book_recommendation.utils')
        _utils_logger.debug(f"[Google] Searching for: '{query}'")
        r = requests.get(API_URL, params=params, headers=headers, timeout=5.0) # Increased to 5s
        if r.ok:
            data = r.json()
            items = data.get("items", [])
            total = data.get("totalItems", 0)
            _utils_logger = logging.getLogger('flask_book_recommendation.utils')
            _utils_logger.debug(f"[Google] ✅ Found {len(items)} results (Total: {total})")
            return items, total
        elif r.status_code in (403, 429):
            _utils_logger = logging.getLogger('flask_book_recommendation.utils')
            _utils_logger.debug(f"[Google] API Error: {r.status_code} - Falling back to OpenLibrary")
            return fetch_openlibrary_fallback(query, max_results)
        else:
            _utils_logger = logging.getLogger('flask_book_recommendation.utils')
            _utils_logger.debug(f"[Google] API Error: {r.status_code}")

    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        pass
    
    return [], 0

def fetch_openlibrary_fallback(query, max_results=12):
    try:
        url = "https://openlibrary.org/search.json"
        params = {"q": query, "limit": max_results}
        r = requests.get(url, params=params, timeout=5.0)
        if r.ok:
            data = r.json()
            docs = data.get("docs", [])
            items = []
            for doc in docs:
                item = {
                    "id": doc.get("key", "").replace("/works/", ""),
                    "source": "openlibrary",
                    "volumeInfo": {
                        "title": doc.get("title", ""),
                        "authors": doc.get("author_name", []),
                        "description": "",
                        "publishedDate": str(doc.get("first_publish_year", "")),
                        "pageCount": doc.get("number_of_pages_median", 0),
                        "categories": doc.get("subject", [])[:3] if doc.get("subject") else [],
                        "imageLinks": {},
                        "industryIdentifiers": []
                    }
                }
                if doc.get("cover_i"):
                    item["volumeInfo"]["imageLinks"]["thumbnail"] = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg"
                if doc.get("isbn"):
                    item["volumeInfo"]["industryIdentifiers"].append({"type": "ISBN_13", "identifier": doc["isbn"][0]})
                
                items.append(item)
            
            _utils_logger = logging.getLogger('flask_book_recommendation.utils')
            _utils_logger.debug(f"[OpenLib] ✅ Fallback returned {len(items)} results")
            return items, data.get("numFound", len(items))
    except Exception as e:
        pass
        
    return [], 0

@cache.memoize(timeout=172800)
def fetch_book_details(book_id, source="google"):
    """
    جلب تفاصيل الكتاب بناءً على المصدر
    """
    if source == "gutenberg":
        return fetch_gutenberg_detail(book_id)
    elif source == "archive":
        return fetch_archive_detail(book_id)
    elif source == "openlibrary":
        return fetch_openlib_detail(book_id)
    elif source == "itbook":
        return fetch_itbook_detail(book_id)
        
    # Default: Google Books
    try:
        api_key = os.environ.get("GOOGLE_BOOKS_API_KEY")
        url = f"https://www.googleapis.com/books/v1/volumes/{book_id}"
        params = {"key": api_key} if api_key else {}
        r = requests.get(url, params=params, timeout=5)
        if r.ok:
            data = r.json()
            vol = data.get("volumeInfo", {})
            rating = vol.get("averageRating")
            
            # Fallback: Try OpenLibrary if no Google rating
            if not rating:
                isbns = vol.get("industryIdentifiers", [])
                isbn = next((i["identifier"] for i in isbns if i["type"] in ["ISBN_13", "ISBN_10"]), None)
                if isbn:
                    rating = fetch_openlib_rating(isbn=isbn)

            # Get ISBN
            isbns = vol.get("industryIdentifiers", [])
            isbn = next((i["identifier"] for i in isbns if i["type"] in ["ISBN_13", "ISBN_10"]), None)

            return {
                "id": data["id"],
                "title": clean_book_title(vol.get("title", "No Title")),
                "author": vol.get("authors", ["Unknown"])[0],
                "description": vol.get("description"),
                "cover": vol.get("imageLinks", {}).get("thumbnail"),
                "preview": vol.get("previewLink"),
                "pageCount": vol.get("pageCount"),
                "rating": rating,
                "ratings_count": vol.get("ratingsCount"),
                "publishedDate": vol.get("publishedDate"),
                "publisher": vol.get("publisher"),
                "language": vol.get("language"),
                "categories": vol.get("categories", []),
                "isbn": isbn,
                "source": "google"
            }
    except Exception as e:
        print(f"Error fetching Google book: {e}")
        
    return None


@cache.memoize(timeout=86400) # cache for 24h
def generate_ai_description(title: str, author: str = "") -> str:
    """
    توليد وصف قصير للكتاب باستخدام AI عندما لا يتوفر وصف.
    محسّن للسرعة - يستخدم نماذج سريعة مع timeout قصير.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return None
    
    if not title or title in ["عنوان غير متوفر", "No Title"]:
        return None
    
    # Prompt مختصر للسرعة
    prompt = f"""اكتب وصفاً قصيراً (40-60 كلمة) لكتاب "{title}" للمؤلف {author or 'غير محدد'}. 
    اكتب الوصف مباشرة بدون مقدمات."""

    # Try Groq first (أسرع بكثير!)
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",  # نموذج أسرع!
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 100
                },
                timeout=4  # timeout قصير
            )
            
            if response.ok:
                data = response.json()
                desc = data["choices"][0]["message"]["content"].strip()
                print(f"[AI Desc] ✅ Generated for: {title[:25]}...")
                return desc
        except requests.exceptions.Timeout:
            print(f"[AI Desc] ⏱️ Groq timeout")
        except Exception as e:
            print(f"[AI Desc] Groq error: {e}")
    
    # Fallback to Gemini (أبطأ قليلاً)
    if gemini_key:
        try:
            print(f"DEBUG: Calling Gemini API for {title}...")
            # Switched to gemini-1.5-flash to avoid 404
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=10
            )
            
            if response.ok:
                data = response.json()
                desc = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if desc:
                    print(f"[AI Desc] ✅ Generated (Gemini) for: {title[:25]}...")
                    return desc.strip()
            else:
                print(f"[AI Desc] Gemini Failed: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            print(f"[AI Desc] ⏱️ Gemini timeout")
        except Exception as e:
            print(f"[AI Desc] Gemini error: {e}")
    
    return None

# -----------------------------------------------------------
# 2. Project Gutenberg
# -----------------------------------------------------------
@cache.memoize(timeout=86400)
def fetch_gutenberg_books(query, page=1, limit=12, **kwargs):
    api_url = "https://gutendex.com/books"
    params = {"search": query, "page": page}
    headers = {"User-Agent": "Elham-Platform/1.0 (Library Integration)"}
    try:
        pass  # Silenced: [Gutenberg] Searching
        r = requests.get(api_url, params=params, headers=headers, timeout=7) # Increased to 7s
        if r.ok:
            results = r.json().get("results", [])
            books = []
            seen = set()
            for b in results:
                title = b.get("title", "")
                if title[:20].lower() in seen: continue
                seen.add(title[:20].lower())
                authors = ", ".join([a.get("name") for a in b.get("authors", [])])
                books.append({
                    "id": f"gut_{b.get('id')}", "title": clean_book_title(title), "author": authors,
                    "cover": b.get("formats", {}).get("image/jpeg"), "source": "gutenberg"
                })
            pass  # Silenced: [Gutenberg] Found
            return books[:limit]
        else:
            pass
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        pass
    return []

@cache.memoize(timeout=172800)
def fetch_gutenberg_detail(gut_id):
    clean_id = gut_id.replace("gut_", "")
    try:
        r = requests.get(f"https://gutendex.com/books/{clean_id}", timeout=4)
        if r.ok:
            b = r.json()
            formats = b.get("formats", {})
            return {
                "id": gut_id, "title": clean_book_title(b.get("title")), 
                "author": ", ".join([a.get("name") for a in b.get("authors", [])]),
                "desc": "كلاسيكيات عالمية (Public Domain).",
                "cover": formats.get("image/jpeg"),
                "preview": formats.get("text/html") or formats.get("text/plain"),
                "source": "gutenberg",
                "publishedDate": str(b.get("authors")[0].get("birth_year")) if b.get("authors") else "N/A" # Fallback to author birth if nothing else
            }
    except: pass
    return None

# -----------------------------------------------------------
# 3. OpenLibrary
# -----------------------------------------------------------
def fetch_openlib_rating(isbn=None, olid=None, title=None):
    """جلب التقييم من OpenLibrary"""
    try:
        url = None
        if olid:
            url = f"https://openlibrary.org/works/{olid}/ratings.json"
        elif isbn:
            url = f"https://openlibrary.org/isbn/{isbn}/ratings.json"
        
        if url:
            r = requests.get(url, timeout=3)
            if r.ok:
                data = r.json()
                summary = data.get("summary", {})
                return summary.get("average")
    except: pass
    return None

@cache.memoize(timeout=3600)
def fetch_openlib_books(query, limit=12, offset=0, **kwargs):
    """جلب كتب من OpenLibrary مع تحسين جلب الأغلفة"""
    headers = {"User-Agent": "Elham-Platform/1.0 (Book Search Engine)"}
    try:
        pass  # Silenced: [OpenLib] Searching
        r = requests.get("https://openlibrary.org/search.json",
                 params={"q": query, "limit": limit, "offset": offset}, headers=headers, timeout=7) # Increased to 7s

        if r.ok:
            docs = r.json().get("docs", [])
            books = []
            for doc in docs:
                # ... same logic ...
                key = doc.get("key", "").replace("/works/", "")
                if not key: continue
                cover = None
                if doc.get("cover_i"):
                    cover = f"https://covers.openlibrary.org/b/id/{doc.get('cover_i')}-M.jpg"
                elif doc.get("isbn"):
                    isbn_list = doc.get("isbn", [])
                    if isbn_list:
                        cover = f"https://covers.openlibrary.org/b/isbn/{isbn_list[0]}-M.jpg"
                elif doc.get("cover_edition_key"):
                    cover = f"https://covers.openlibrary.org/b/olid/{doc.get('cover_edition_key')}-M.jpg"
                
                author = doc.get("author_name")
                if isinstance(author, list): author = ", ".join(author[:2])
                
                title = doc.get("title")
                if not title: continue
                if not cover: cover = "https://placehold.co/300x450/stone/white?text=OpenLibrary"
                
                rating = doc.get("ratings_average")
                if rating: rating = round(float(rating), 1)

                books.append({
                    "id": f"ol_{key}", "title": title, "author": author or "مؤلف غير معروف",
                    "cover": cover, "rating": rating, "source": "openlibrary"
                })
            
            pass  # Silenced: [OpenLib] Found
            return books
        else:
            pass
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        pass
    return []

@cache.memoize(timeout=172800)
def fetch_openlib_detail(ol_id):
    clean_id = ol_id.replace("ol_", "")
    try:
        r = requests.get(f"https://openlibrary.org/works/{clean_id}.json", timeout=5)
        if r.ok:
            data = r.json()
            desc = data.get("description")
            if isinstance(desc, dict): desc = desc.get("value")
            cover = f"https://covers.openlibrary.org/b/id/{data['covers'][0]}-M.jpg" if data.get("covers") else None

            # إذا لم يوجد غلاف في العمل الرئيسي، نبحث في الطبعات (editions)
            if not cover:
                try:
                    ed_r = requests.get(f"https://openlibrary.org/works/{clean_id}/editions.json", timeout=3)
                    if ed_r.ok:
                        entries = ed_r.json().get("entries", [])
                        for entry in entries:
                            if entry.get("covers"):
                                cover = f"https://covers.openlibrary.org/b/id/{entry['covers'][0]}-M.jpg"
                                break
                except:
                    pass
            
            # Fetch Rating explicitly
            rating = fetch_openlib_rating(olid=clean_id)

            return {
                "id": ol_id, "title": clean_book_title(data.get("title")), "author": "OpenLibrary Author",
                "desc": desc or "No description.", "cover": cover,
                "preview": f"https://openlibrary.org/works/{clean_id}", 
                "rating": rating,
                "pageCount": data.get("number_of_pages"),
                "publishedDate": data.get("publish_date"),
                "source": "openlibrary"
            }
    except: pass
    return None

# -----------------------------------------------------------
# 📚 Open Library Covers API (مجاني وعالي الجودة)
# -----------------------------------------------------------

def fetch_cover_from_openlibrary(isbn=None, title=None, author=None, size="L"):
    """
    جلب غلاف كتاب من Open Library Covers API
    
    Args:
        isbn: رقم ISBN للكتاب (أفضل طريقة)
        title: عنوان الكتاب (للبحث إذا لم يتوفر ISBN)
        author: اسم المؤلف (اختياري - يحسن نتائج البحث)
        size: حجم الصورة (S=صغير, M=متوسط, L=كبير)
    
    Returns:
        رابط الغلاف أو None
    """
    
    # 1. أولاً: البحث بـ ISBN (الأدق)
    if isbn:
        # تنظيف ISBN من الشرطات
        clean_isbn = str(isbn).replace("-", "").replace(" ", "").strip()
        if len(clean_isbn) in [10, 13]:
            cover_url = f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-{size}.jpg"
            # التحقق من وجود الصورة
            if _verify_cover_exists(cover_url):
                pass
                return cover_url
    
    # 2. ثانياً: البحث في OpenLibrary بالعنوان والمؤلف
    if title:
        try:
            search_query = title
            if author:
                search_query += f" {author}"
            
            params = {"q": search_query, "limit": 1}
            r = requests.get("https://openlibrary.org/search.json", params=params, timeout=5)
            
            if r.ok:
                docs = r.json().get("docs", [])
                if docs:
                    doc = docs[0]
                    
                    # محاولة 1: cover_i (ID الغلاف المباشر)
                    if doc.get("cover_i"):
                        cover_url = f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-{size}.jpg"
                        pass
                        return cover_url
                    
                    # محاولة 2: ISBN من نتائج البحث
                    if doc.get("isbn"):
                        isbn_list = doc.get("isbn", [])
                        for isbn_try in isbn_list[:3]:  # نحاول أول 3 ISBNs
                            cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn_try}-{size}.jpg"
                            if _verify_cover_exists(cover_url):
                                pass
                                return cover_url
                    
                    # محاولة 3: OLID (Open Library ID)
                    if doc.get("cover_edition_key"):
                        cover_url = f"https://covers.openlibrary.org/b/olid/{doc['cover_edition_key']}-{size}.jpg"
                        pass
                        return cover_url
                        
        except Exception as e:
            pass
    
    return None


def _verify_cover_exists(url):
    """التحقق من وجود الغلاف (لتجنب الصور الفارغة)"""
    try:
        r = requests.head(url, timeout=3)
        # OpenLibrary ترجع 200 دائماً، لكن الصورة الفارغة حجمها < 1KB
        if r.ok:
            content_length = r.headers.get("Content-Length", 0)
            return int(content_length) > 1000  # أكثر من 1KB = صورة حقيقية
    except:
        pass
    return True  # نفترض أنها موجودة إذا فشل التحقق


@cache.memoize(timeout=86400)
def get_book_cover_smart(title, author=None, isbn=None, source=None):
    """
    جلب غلاف كتاب بطريقة ذكية من مصادر متعددة
    
    يحاول من:
    1. Open Library (بـ ISBN أو بالبحث)
    2. Google Books
    3. توليد غلاف AI كـ fallback
    
    Args:
        title: عنوان الكتاب (مطلوب)
        author: اسم المؤلف (اختياري)
        isbn: رقم ISBN (اختياري)
        source: مصدر الكتاب الأصلي (اختياري)
    
    Returns:
        dict: {"cover_url": str, "source": str}
    """
    
    # 1. Open Library (الأفضل للجودة)
    ol_cover = fetch_cover_from_openlibrary(isbn=isbn, title=title, author=author)
    if ol_cover:
        return {"cover_url": ol_cover, "source": "openlibrary"}
    
    # 2. Google Books
    try:
        search_query = title
        if author:
            search_query += f" {author}"
        
        items, _ = fetch_google_books(search_query, max_results=1)
        if items:
            vi = items[0].get("volumeInfo", {}) or {}
            imgs = vi.get("imageLinks", {}) or {}
            cover = imgs.get("large") or imgs.get("medium") or imgs.get("thumbnail")
            if cover:
                if cover.startswith("http://"):
                    cover = "https://" + cover[7:]
                return {"cover_url": cover, "source": "google"}
    except Exception as e:
        print(f"[Smart Cover] Google error: {e}")
    
    # 3. AI Cover (Pollinations)
    import urllib.parse
    import hashlib
    
    # تحسين الـ Prompt ليكون "abstract" و "minimalist"
    prompt = f"minimalist book cover for '{title}'"
    if author:
        prompt += f" by {author}"
    
    # إضافة كلمات مفتاحية تضمن الجودة وتبتعد عن الروبوتات
    prompt += ", abstract art, elegant design, high quality, 4k"
    
    # تخصيص حسب العنوان
    lower_title = title.lower()
    if any(w in lower_title for w in ['tech', 'data', 'code', 'algorithm']):
        prompt += ", geometric shapes, data visualization style, NO ROBOTS, NO FACES"
    elif any(w in lower_title for w in ['history', 'ancient']):
        prompt += ", vintage style, paper texture"
    else:
        prompt += ", modern typography"

    encoded_prompt = urllib.parse.quote(prompt)
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 10000
    
    ai_cover = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=400&height=600&nologo=true&seed={seed}&model=flux"
    
    return {"cover_url": ai_cover, "source": "ai_generated"}


def generate_ai_cover_url(title, author=None):
    """
    توليد رابط غلاف AI مباشرة (بدون بحث)
    """
    import urllib.parse
    import hashlib

    # تحسين الـ Prompt ليكون "abstract" و "minimalist"
    prompt = f"minimalist book cover for '{title}'"
    if author:
        prompt += f" by {author}"
    
    # إضافة كلمات مفتاحية تضمن الجودة وتبتعد عن الروبوتات
    prompt += ", abstract art, elegant design, high quality, 4k"
    
    # تخصيص حسب العنوان
    lower_title = title.lower()
    if any(w in lower_title for w in ['tech', 'data', 'code', 'algorithm']):
        prompt += ", geometric shapes, data visualization style, NO ROBOTS, NO FACES"
    elif any(w in lower_title for w in ['history', 'ancient']):
        prompt += ", vintage style, paper texture"
    else:
        prompt += ", modern typography"

    encoded_prompt = urllib.parse.quote(prompt)
    # استخدام hash للعنوان لضمان ثبات الصورة لنفس الكتاب
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 10000
    
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=400&height=600&nologo=true&seed={seed}&model=flux"


# -----------------------------------------------------------
# 4. IT Bookstore
# -----------------------------------------------------------
@cache.memoize(timeout=172800)
def fetch_itbook_detail(isbn):
    try:
        r = requests.get(f"https://api.itbook.store/1.0/books/{isbn}", timeout=5)
        if r.ok:
            data = r.json()
            return {
                "id": data.get("isbn13"), "title": clean_book_title(data.get("title")), "author": data.get("authors"),
                "desc": data.get("desc"), "cover": data.get("image"), "preview": data.get("url"), 
                "pageCount": data.get("pages"),
                "publishedDate": data.get("year"),
                "source": "itbook"
            }
    except: pass
    return None

# utils.py

# ... (باقي الكود في الأعلى كما هو) ...

@cache.memoize(timeout=86400)
def fetch_itbook_books(query, page=1, limit=8, **kwargs):
    headers = {"User-Agent": "Elham-Platform/1.0 (Tech Books)"}
    try:
        url = f"https://api.itbook.store/1.0/search/{query}/{page}"
        print(f"[ITBook] Searching for: '{query}'")
        r = requests.get(url, headers=headers, timeout=7) # Increased to 7s
        if not r.ok: 
            print(f"[ITBook] ❌ API Error: {r.status_code}")
            return []

        data = r.json()
        books_raw = data.get("books", [])[:limit]
        
        books = []
        for b in books_raw:
            isbn13 = b.get("isbn13")
            if not isbn13: continue
            author = b.get("subtitle") or "IT Book"
            books.append({
                "id": isbn13, "title": clean_book_title(b.get("title") or "Untitled"), "author": author,
                "cover": b.get("image"), "source": "itbook"
            })
        print(f"[ITBook] ✅ Found {len(books)} tech books")
        return books
    except requests.exceptions.Timeout:
        print(f"[ITBook] ⏱️ Timeout for '{query}'")
    except Exception as e:
        print(f"[ITBook] ⚠️ Error: {e}")
    return []

# ... (باقي الكود في الأسفل كما هو) ...
# -----------------------------------------------------------
# 5. Archive.org
# -----------------------------------------------------------
@cache.memoize(timeout=172800)
def fetch_archive_detail(archive_id, max_results=1): # تم تعديل التوقيع ليتوافق
    # إذا تم تمرير ID كنص عادي (للبحث عن التفاصيل)
    if isinstance(archive_id, str) and not archive_id.startswith("http"):
        clean_id = archive_id.replace("arch_", "")
        url = f"https://archive.org/metadata/{clean_id}"
        try:
            r = requests.get(url, timeout=4)
            if r.ok:
                data = r.json()
                meta = data.get("metadata", {})
                if meta and meta.get("title"):
                    desc = meta.get("description", "No description available.")
                    if isinstance(desc, list):
                        desc = " ".join(desc)
                    return {
                        "id": archive_id, "title": clean_book_title(meta.get("title")),
                        "author": meta.get("creator") if isinstance(meta.get("creator"), str) else ", ".join(meta.get("creator", [])) if meta.get("creator") else "Unknown Author",
                        "desc": desc,
                        "cover": f"https://archive.org/services/img/{clean_id}",
                        "preview": f"https://archive.org/details/{clean_id}",
                        "publishedDate": meta.get("date"),
                        "source": "archive"
                    }
        except Exception as e:
            print(f"[Archive Detail] Error: {e}")
        
        # Fallback: إرجاع بيانات افتراضية
        return {
            "id": archive_id, 
            "title": clean_id.replace("_", " ").replace("00", " ").title(),
            "author": "Internet Archive",
            "desc": "هذا الكتاب متوفر على Internet Archive. اضغط على معاينة للقراءة المجانية.",
            "cover": f"https://archive.org/services/img/{clean_id}",
            "preview": f"https://archive.org/details/{clean_id}",
            "source": "archive"
        }
    
    # إذا تم استخدامها للبحث (كما في book_detail سابقاً)
    return fetch_archive_books(archive_id, limit=max_results), 0

@cache.memoize(timeout=86400)
def fetch_archive_books(query, limit=12, **kwargs):
    """جلب كتب من Internet Archive مع معالجة أفضل للأخطاء"""
    base_url = "https://archive.org/advancedsearch.php"
    
    # تنظيف الاستعلام
    clean_query = query.strip()
    if not clean_query:
        clean_query = "books"
    
    # استعلام بسيط
    search_query = f"{clean_query} mediatype:texts"
    params = {
        "q": search_query, 
        "rows": limit, 
        "output": "json",
        "fl": "identifier,title,creator"
    }
    
    try:
        pass  # Silenced: [Archive] Searching
        r = requests.get(base_url, params=params, timeout=4)  # Accelerated timeout
        if r.ok:
            data = r.json()
            docs = data.get("response", {}).get("docs", [])
            books = []
            for doc in docs:
                identifier = doc.get("identifier")
                if not identifier:
                    continue
                title = doc.get("title")
                if not title:
                    continue
                creator = doc.get("creator", "Unknown Author")
                if isinstance(creator, list): 
                    creator = ", ".join(creator)
                books.append({
                    "id": f"arch_{identifier}", 
                    "title": title, 
                    "author": creator,
                    "cover": f"https://archive.org/services/img/{identifier}", 
                    "source": "archive"
                })
            pass  # Silenced: [Archive] Found
            if books:
                return books
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        pass
    
    # Fallback: لا نعرض كتب عشوائية - أفضل عدم عرض شيء من عرض كتب غير متعلقة
    pass
    return []

# -----------------------------------------------------------
from functools import lru_cache

@lru_cache(maxsize=500)
def analyze_search_intent_with_ai(text: str, timeout: int = 10) -> dict:
    """
    تحليل نية البحث باستخدام الذكاء الاصطناعي.
    Input: "أبي روايات بوليسية تشبه شارلوك هولمز"
    Output: {"query": "sherlock holmes detective novels", "is_tech": False}
    """
    if not text or len(text.split()) < 2:
        return {"query": text, "is_tech": False}

    import json
    api_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    # Prompt محسن لاستخراج الكلمات المفتاحية
    prompt = f"""
    You are a search query optimizer for a book library.
    Convert this natural language request into a precise search query for Google Books/OpenLibrary APIs.
    
    User Request: "{text}"
    
    Rules:
    1. Extract core keywords (genre, author, topic).
    2. Keep English terms if they are better for search.
    3. Remove fluff ("I want", "books about", "please").
    4. Detect if it's a technical/programming query.
    
    Return JSON ONLY: {{"query": "keywords_here", "is_tech": boolean}}
    """
    
    try:
        # 1. محاولة استخدام Groq (أسرع)
        if groq_key:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                },
                timeout=5
            )
            if response.ok:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return json.loads(content)

        # 2. Fallback إلى Gemini
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=timeout
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp)
                
    except Exception as e:
        print(f"[Search Analysis] Error: {e}")
    
    # في حال الفشل، نعيد النص كما هو
    return {"query": text, "is_tech": False}

def translate_to_english_with_gemini(text: str, timeout: int = 10) -> str:
    """Wrapper for backward compatibility"""
    res = analyze_search_intent_with_ai(text, timeout=timeout)
    return res.get("query", text)

def generate_reading_plan_with_ai(book_title, pages, days):
    """
    إنشاء خطة قراءة ذكية باستخدام AI
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = f"""
    Create a reading plan for book: "{book_title}" ({pages} pages) to finish in {days} days.
    
    Return JSON ONLY:
    {{
        "daily_quota": "number of pages",
        "strategy": "brief strategy advice (1 sentence)",
        "schedule": [
            {{"day": 1, "focus": "pages x-y", "tip": "brief tip"}},
            ... (for each day)
        ]
    }}
    """
    
    try:
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=8
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp)
    except Exception as e:
        print(f"Plan error: {e}")
        
    # Fallback plan
    quota =  int(int(pages) / int(days))
    return {
        "daily_quota": quota,
        "strategy": "Consistent daily reading is key.",
        "advice": "Consistent daily reading is key.", # Matching JS 'data.advice'
        "schedule": [
            {
                "day": i+1, 
                "focus": f"Read {quota} pages", 
                "task": f"Read up to page {quota * (i+1)}", # Matching JS 'm.task'
                "tip": "Keep going!"
            } 
            for i in range(int(days))
        ]
    }

def extract_quotes_with_ai(title, author):
    """
    استخراج اقتباسات ملهمة من الكتاب باستخدام AI
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = f"""
    Extract 4 short, inspiring, and beautiful quotes from the book "{title}" by {author}.
    If the exact text is not available, generate 4 quotes that capture the essence and style of the book perfectly.
    
    Return JSON ONLY:
    {{
        "quotes": [
            "Quote 1 text...",
            "Quote 2 text...",
            "Quote 3 text...",
            "Quote 4 text..."
        ]
    }}
    """
    
    try:
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=8
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp)
    except Exception as e:
        print(f"Quote error: {e}")
        
    return {
        "quotes": [
            f"The love of books is a love which requires neither justification, apology, nor defense. - {author}",
            f"A room without books is like a body without a soul. - {title}",
            "So many books, so little time.",
            "I have always imagined that Paradise will be a kind of library."
        ]
    }

def analyze_book_mood_with_ai(title, description):
    """
    تحليل مزاج الكتاب لتفعيل وضع الانغماس
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = f"""
    Analyze the mood/atmosphere of the book "{title}": {description[:200]}...
    
    Classify into ONE of these:
    - dark (Horror, Thriller, Mystery)
    - happy (Comedy, Romance, Kids)
    - calm (Philosophy, Nature, Self-help)
    - epic (Fantasy, History, Sci-Fi)
    
    Return JSON ONLY: {{"mood": "dark|happy|calm|epic"}}
    """
    
    try:
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=5
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp)
    except Exception as e:
        print(f"Mood error: {e}")
        
    return {"mood": "calm"}

def generate_quiz_with_ai(title, author):
    """
    Generate 5 MCQ questions about the book using AI
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = f"""
    Create a fun quiz for the book "{title}" by {author}.
    Generate 5 multiple-choice questions (MCQs).
    
    Format JSON ONLY:
    {{
        "questions": [
            {{
                "question": "Question text?",
                "options": ["A", "B", "C"],
                "correct_index": 0
            }}
        ]
    }}
    
    Make questions testing plot key points.
    Language: Arabic (Translate if needed).
    """
    
    try:
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=8
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp)
    except Exception as e:
        print(f"Quiz error: {e}")
        
    # Fallback
    return {
        "questions": [
            {
                "question": f"من هو مؤلف كتاب {title}؟",
                "options": [author, "نجيب محفوظ", "طه حسين"],
                "correct_index": 0
            },
            {
                "question": "ما هو نوع هذا الكتاب؟",
                "options": ["رواية", "شعر", "سيرة ذاتية"],
                "correct_index": 0
            }
        ]
    }


def extract_interests_from_text_ai(title: str, author: str, review_text: str = "") -> list:
    """
    Extract 3-5 key topics/genres from a book and its review using AI.
    Used to update user preferences based on high-rated books.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    prompt = f"""
    Analyze this book and optional review to extract 3-5 distinct, search-friendly topics or genres.
    Book: "{title}" by {author}.
    Review: "{review_text}"
    
    Return JSON ONLY:
    {{
        "topics": ["Genre/Topic 1", "Genre/Topic 2", "Genre/Topic 3"]
    }}
    
    Rules:
    - Topics should be general enough for recommendation (e.g., "Science Fiction", "Self Help", "History", "Dragons").
    - If the review mentions specific aspects (e.g., "loved the world building"), include related topics (e.g., "World Building", "Fantasy").
    - English is preferred for better matching, but Arabic is okay if the input is Arabic.
    """
    
    try:
        # 1. Try Groq (Fastest)
        if groq_key:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"}
                },
                timeout=5
            )
            if response.ok:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return json.loads(content).get("topics", [])

        # 2. Fallback to Gemini
        if api_key:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"}
                },
                timeout=5
            )
            if response.ok:
                data = response.json()
                text_resp = data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_resp).get("topics", [])
                
    except Exception as e:
        print(f"[Interest Extraction] Error: {e}")
    
    # Simple Fallback if AI fails: just use keywords from title
    return [w for w in title.split() if len(w) > 4][:2]

# ... (دالة get_text_embedding اتركها كما هي) ...

import time
from functools import lru_cache
import threading

# Global cache for the model
_embedding_model = None
_embedding_model_lock = threading.Lock()
_local_model_failed = False

@lru_cache(maxsize=1000)
def get_text_embedding(text, max_retries=3):
    """
    تحويل النص إلى embedding vector.
    يحاول أولاً استخدام نموذج محلي (SentenceTransformers) لضمان الاستمرارية،
    ثم يحاول استخدام Gemini API كبديل.
    """
    # 1. محاولة استخدام النموذج المحلي (أسرع وأكثر اعتمادية)
    global _embedding_model, _local_model_failed
    if not _local_model_failed:
        try:
            if _embedding_model is None:
                with _embedding_model_lock:
                    # Double-check after acquiring lock
                    if _embedding_model is None and not _local_model_failed:
                        import os
                        # Set timeout environment variable for huggingface to fail fast if network is bad
                        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
                        
                        from sentence_transformers import SentenceTransformer
                        # Suppress verbose model load reports
                        import logging as _lg
                        for _n in ['safetensors', 'transformers', 'sentence_transformers']:
                            _lg.getLogger(_n).setLevel(_lg.WARNING)
                        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if _embedding_model is not None:
                embedding = _embedding_model.encode(text)
                return embedding.tolist()
        except Exception as e:
            _local_model_failed = True
            print(f"[Local-Embedding] ⚠️ Error loading local model: {e}. Falling back to Gemini permanently for this session.")

    # 2. محاولة استخدام Gemini API كبديل
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    
    if not text or not text.strip():
        return None
    
    clean_text = text.strip()[:2000]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    payload = {
        "model": "models/embedding-001", 
        "content": {"parts": [{"text": clean_text}]}
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                embedding = data.get('embedding', {}).get('values')
                if embedding:
                    return embedding
            time.sleep(1)
        except:
            pass
    
    return None


def get_book_embedding(book):
    """
    توليد embedding لكتاب بناءً على عنوانه ومؤلفه ووصفه.
    
    Args:
        book: كائن Book من قاعدة البيانات
        
    Returns:
        embedding vector أو None
    """
    if not book:
        return None
    
    # جمع المعلومات المتاحة عن الكتاب
    parts = []
    
    if book.title:
        parts.append(f"Title: {book.title}")
    
    if book.author:
        parts.append(f"Author: {book.author}")
    
    if book.description:
        # نأخذ أول 500 حرف من الوصف
        desc = book.description[:500]
        parts.append(f"Description: {desc}")

    # 🆕 إضافة التصنيفات للمحتوى (مهم جداً للتوصيات)
    if book.categories:
        parts.append(f"Genre: {book.categories}")
    
    if not parts:
        return None
    
    text = ". ".join(parts)
    return get_text_embedding(text)


def generate_book_embedding_if_missing(book):
    """
    توليد وحفظ embedding للكتاب إذا لم يكن موجوداً.
    
    Args:
        book: كائن Book
        
    Returns:
        True إذا تم التوليد بنجاح، False خلاف ذلك
    """
    from .models import BookEmbedding
    from .extensions import db
    
    if not book or not book.id:
        return False
    
    # تحقق إذا كان موجوداً مسبقاً
    existing = BookEmbedding.query.filter_by(book_id=book.id).first()
    if existing and existing.vector:
        return True  # موجود مسبقاً
    
    # توليد embedding جديد
    embedding = get_book_embedding(book)
    if not embedding:
        return False
    
    try:
        if existing:
            existing.vector = embedding
        else:
            new_embed = BookEmbedding(book_id=book.id, vector=embedding)
            db.session.add(new_embed)
        
        db.session.commit()
        print(f"[Embedding] ✅ Generated for book: {book.title[:30]}...")
        return True
        
    except Exception as e:
        db.session.rollback()
        print(f"[Embedding] ❌ Failed to save: {e}")
        return False


# utils.py - أضف هذا في النهاية
import re

def normalize_text(text):
    if not text: return ""
    # تحويل النص إلى string وضمان الأحرف الصغيرة
    text = str(text).lower().strip()
    # توحيد الألف (أ إ آ -> ا)
    text = re.sub("[أإآ]", "ا", text)
    # توحيد التاء المربوطة والهاء (ة -> ه)
    text = re.sub("ة", "ه", text)
    # توحيد الياء (ى -> ي)
    text = re.sub("ى", "ي", text)
    # إزالة التشكيل (اختياري)
    text = re.sub("[\u064B-\u065F]", "", text)
    return text


# -----------------------------------------------------------
# 7. AI Chatbot للكتب (Groq + Gemini Fallback)
# -----------------------------------------------------------

def chat_with_ai(user_message: str, user_context: dict = None) -> dict:
    """
    مساعد AI ذكي للكتب يجيب على أسئلة المستخدمين ويقدم توصيات.
    يستخدم Groq API (مجاني وسريع) كأولوية أولى.
    
    Args:
        user_message: رسالة المستخدم
        user_context: سياق إضافي (اهتمامات، كتب سابقة، إلخ)
        
    Returns:
        قاموس يحتوي على رد AI وتوصيات الكتب
    """
    
    if not user_message or not user_message.strip():
        return {
            "reply": "مرحباً! كيف يمكنني مساعدتك في اختيار كتاب؟",
            "books": [],
            "search_query": None
        }

    # Ensure keys are loaded
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        print("DEBUG: Reloading .env in chat_with_ai...")
        load_dotenv(override=True)
    
    # بناء السياق
    context_info = ""
    if user_context:
        if user_context.get("interests"):
            context_info += f"\nاهتمامات المستخدم: {', '.join(user_context['interests'])}"
        if user_context.get("recent_books"):
            context_info += f"\nآخر الكتب التي اطلع عليها: {', '.join(user_context['recent_books'])}"
        if user_context.get("book_title"):
            context_info += f"\nالكتاب الحالي: {user_context['book_title']} للمؤلف {user_context.get('book_author', 'غير معروف')}"
        if user_context.get("book_desc"):
            context_info += f"\nوصف الكتاب: {user_context['book_desc']}"
        
        history = user_context.get("history", [])
        if history:
            context_info += "\n\nسجل المحادثة السابقة:"
            for msg in history[-5:]:  # Limit to last 5 messages
                role = "المستخدم" if msg.get("role") == "user" else "المساعد"
                context_info += f"\n{role}: {msg.get('content')}"
    
    # بناء الـ prompt
    system_prompt = """أنت مساعد ذكي متخصص في الكتب واسمك "مكتبي". مهمتك:
1. فهم ما يبحث عنه المستخدم من كتب
2. تقديم توصيات مفيدة ومحددة
3. الإجابة على أسئلة عن الكتب والقراءة

قواعد مهمة:
- كن ودوداً ومختصراً (جملتين إلى 3 جمل كحد أقصى)
- إذا طلب المستخدم كتاباً، استخرج الموضوع الرئيسي للبحث
- رد بالعربية دائماً
- أضف إيموجي واحد مناسب

في نهاية ردك، اكتب في سطر جديد:
SEARCH_QUERY: [كلمات البحث بالإنجليزية للموضوع المطلوب]

مثال:
المستخدم: أريد كتاب عن الذكاء الاصطناعي
الرد: مجال رائع! 🤖 الذكاء الاصطناعي من أهم مجالات العصر. سأبحث لك عن أفضل الكتب.
SEARCH_QUERY: Artificial Intelligence books"""

    full_prompt = f"{system_prompt}{context_info}\n\nالمستخدم: {user_message}"
    
    # محاولة 1: Groq API (مجاني وسريع)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        result = _call_groq_api(groq_key, full_prompt)
        if result:
            return _process_ai_response(result)
    
    # محاولة 2: Gemini API (fallback)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        result = _call_gemini_api(gemini_key, full_prompt)
        if result:
            return _process_ai_response(result)
    
    # لا يوجد مفتاح متاح أو فشل في الاتصال
    error_msg = "عذراً، المساعد غير متاح حالياً."
    if not gemini_key and not groq_key:
        error_msg += " يرجى إضافة مفتاح API في الإعدادات."
    else:
        error_msg += " نواجه ضغطاً على الخوادم، يرجى المحاولة لاحقاً."

    return {
        "reply": error_msg,
        "books": [],
        "search_query": None
    }


def _call_groq_api(api_key: str, prompt: str) -> str:
    """استدعاء Groq API - مجاني وسريع جداً"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # قائمة النماذج المدعومة للمحاولة
    models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    
    payload = {
        "model": "llama-3.3-70b-versatile",  # النموذج الأحدث والأسرع
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        print("[AI Chat] Using Groq API...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            try:
                ai_text = data['choices'][0]['message']['content'].strip()
                print(f"[AI Chat] Groq success!")
                return ai_text
            except (KeyError, IndexError) as e:
                print(f"[AI Chat] Groq parsing error: {e}")
                return None
                
        elif response.status_code == 429:
            print("[AI Chat] Groq rate limited, trying fallback...")
            return None
        else:
            print(f"[AI Chat] Groq error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[AI Chat] Groq exception: {e}")
        return None


def _call_gemini_api(api_key: str, prompt: str) -> str:
    """استدعاء Gemini API كـ fallback"""
    # استخدام 1.5-flash المتوفر والمستقر
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200
        }
    }
    
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"[AI Chat] Using Gemini API (fallback)... Attempt {attempt+1}")
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                try:
                    # التحقق الآمن من وجود المرشحات
                    candidates = data.get('candidates', [])
                    if not candidates:
                        print("[AI Chat] Gemini returned no candidates (Safety filter?)")
                        return None
                        
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if not parts:
                        return None
                        
                    ai_text = parts[0].get('text', '').strip()
                    print(f"[AI Chat] Gemini success!")
                    return ai_text
                except (KeyError, IndexError, AttributeError) as e:
                    print(f"[AI Chat] Gemini parsing error: {e}")
                    return None
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 2
                print(f"[AI Chat] Gemini rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"[AI Chat] Gemini error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"[AI Chat] Gemini exception: {e}")
            return None
            
    return None


def _process_ai_response(ai_text: str) -> dict:
    """معالجة رد AI واستخراج الكتب"""
    # استخراج query البحث
    search_query = None
    reply = ai_text
    
    if "SEARCH_QUERY:" in ai_text:
        parts = ai_text.split("SEARCH_QUERY:")
        reply = parts[0].strip()
        if len(parts) > 1:
            search_query = parts[1].strip()
    
    # جلب كتب إذا وجد query
    books = []
    if search_query:
        try:
            items, _ = fetch_google_books(search_query, max_results=6)
            for item in items[:6]:
                vi = item.get("volumeInfo", {}) or {}
                imgs = vi.get("imageLinks", {}) or {}
                cover = imgs.get("thumbnail", "")
                if cover.startswith("http://"):
                    cover = "https://" + cover[7:]
                
                books.append({
                    "id": item.get("id"),
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors", [])) if vi.get("authors") else "",
                    "cover": cover,
                    "source": "google"
                })
        except Exception as e:
            print(f"[AI Chat] Book fetch error: {e}")
    
    return {
        "reply": reply,
        "books": books,
        "search_query": search_query
    }


# -----------------------------------------------------------
# 📝 ملخص AI للكتب
# -----------------------------------------------------------
def generate_book_summary(book_info: dict) -> dict:
    """
    توليد ملخص ذكي للكتاب باستخدام Groq (أو Gemini كاحتياطي)
    
    Args:
        book_info: قاموس يحتوي معلومات الكتاب (title, author, description, categories)
    
    Returns:
        قاموس يحتوي {"success": bool, "summary": str, "error": str}
    """
    import json
    
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "summary": "", "error": "لا يوجد مفتاح API متاح"}
    
    # تجميع معلومات الكتاب
    title = book_info.get("title", "غير معروف")
    author = book_info.get("author", "غير معروف")
    description = book_info.get("description", "")
    categories = book_info.get("categories", "")
    
    prompt = f"""أنت ناقد أدبي محترف. قم بكتابة ملخص شامل وجذاب لهذا الكتاب:

📚 عنوان الكتاب: {title}
✍️ المؤلف: {author}
📂 التصنيف: {categories}
📝 الوصف الأصلي: {description[:500] if description else 'غير متوفر'}

اكتب ملخصاً باللغة العربية يتضمن:
1. موضوع الكتاب الرئيسي (2-3 جمل)
2. الأفكار الرئيسية المتوقعة (3-4 نقاط)
3. الفئة المستهدفة من القراء

اجعل الملخص جذاباً ومختصراً (150-200 كلمة كحد أقصى).
لا تذكر أنك ذكاء اصطناعي، اكتب كأنك خبير في الكتب."""

    # محاولة Groq أولاً
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                summary = data["choices"][0]["message"]["content"].strip()
                return {"success": True, "summary": summary, "error": ""}
            else:
                print(f"[AI Summary] Groq error: {response.status_code}")
        except Exception as e:
            print(f"[AI Summary] Groq exception: {e}")
    
    # Fallback إلى Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                summary = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if summary:
                    return {"success": True, "summary": summary.strip(), "error": ""}
        except Exception as e:
            print(f"[AI Summary] Gemini exception: {e}")
    
    # Final Fallback - Basic Summary based on metadata
    summary = f"كتاب عنوانه '{title}' للمؤلف {author}. "
    if categories:
        summary += f"يندرج هذا العمل تحت تصنيف {categories}. "
    if description:
        summary += f"\n\nوصف سريع:\n{description[:300]}..."
    else:
        summary += "لا يتوفر وصف تفصيلي حالياً، لكنه يعتبر من العناوين المميزة في فئته."
        
    return {"success": True, "summary": summary, "error": ""}


# -----------------------------------------------------------
# 🎯 لماذا قد يعجبك هذا الكتاب
# -----------------------------------------------------------
def generate_why_you_like(book_info: dict, user_context: dict) -> dict:
    """
    تحليل لماذا قد يعجب هذا الكتاب المستخدم بناءً على اهتماماته
    
    Args:
        book_info: معلومات الكتاب
        user_context: سياق المستخدم (interests, recent_books, favorite_genres)
    
    Returns:
        قاموس يحتوي {"success": bool, "analysis": str, "error": str}
    """
    import json
    
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "analysis": "", "error": "لا يوجد مفتاح API متاح"}
    
    # معلومات الكتاب
    title = book_info.get("title", "غير معروف")
    author = book_info.get("author", "غير معروف")
    description = book_info.get("description", "")
    categories = book_info.get("categories", "")
    
    # سياق المستخدم
    interests = user_context.get("interests", [])
    recent_books = user_context.get("recent_books", [])
    favorite_genres = user_context.get("favorite_genres", [])
    
    prompt = f"""أنت مستشار قراءة شخصي ذكي. حلل هذا الكتاب واشرح للقارئ لماذا قد يناسبه:

📚 الكتاب:
- العنوان: {title}
- المؤلف: {author}
- التصنيف: {categories}
- الوصف: {description[:300] if description else 'غير متوفر'}

👤 ملف القارئ:
- الاهتمامات: {', '.join(interests) if interests else 'متنوعة'}
- الكتب الأخيرة: {', '.join(recent_books[:5]) if recent_books else 'لم يُحدد'}
- الأنواع المفضلة: {', '.join(favorite_genres) if favorite_genres else 'متنوعة'}

اكتب تحليلاً شخصياً باللغة العربية يوضح:
1. 🎯 نقاط التوافق بين الكتاب واهتمامات القارئ
2. ✨ ما الذي سيستفيده القارئ من هذا الكتاب
3. 💡 لماذا هذا الوقت مناسب لقراءته

اجعل التحليل شخصياً وحميمياً كأنك صديق يقترح كتاباً (100-150 كلمة).
ابدأ مباشرة بالتحليل بدون مقدمات."""

    # محاولة Groq أولاً
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 400
                },
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                analysis = data["choices"][0]["message"]["content"].strip()
                return {"success": True, "analysis": analysis, "error": ""}
            else:
                print(f"[AI WhyLike] Groq error: {response.status_code}")
        except Exception as e:
            print(f"[AI WhyLike] Groq exception: {e}")
    
    # Fallback إلى Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                analysis = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if analysis:
                    return {"success": True, "analysis": analysis.strip(), "error": ""}
        except Exception as e:
            print(f"[AI WhyLike] Gemini exception: {e}")
    
    return {"success": False, "analysis": "", "error": "تعذر توليد التحليل"}


# -----------------------------------------------------------
# 📅 خطة القراءة الذكية
# -----------------------------------------------------------
def generate_reading_plan(book_info: dict, days: int = 7) -> dict:
    """
    توليد خطة قراءة ذكية للكتاب
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "plan": "", "error": "لا يوجد مفتاح API متاح"}
        
    title = book_info.get("title", "كتاب")
    pages = book_info.get("pageCount", 0)
    
    if not pages or pages == 0:
        pages = "غير محدد (افترض متوسط 300 صفحة)"
    
    prompt = f"""قم بإنشاء خطة قراءة لمدة {days} أيام لهذا الكتاب:
    
- الكتاب: {title}
- عدد الصفحات: {pages}

المطلوب: جدول markdown بسيط يوضح ماذا أقرأ كل يوم.
اجعل الخطة مشجعة وعملية.
Format:
| اليوم | الصفحات | الهدف |
|-------|---------|-------|
...
"""

    # ... (نفس منطق الاستدعاء لـ Groq/Gemini مثل الدوال السابقة)
    # للاختصار سأستخدم دالة مساعدة داخلية لو كان ممكناً، لكن سأكرر الكود للأسف لضمان الاستقلالية
    import requests
    
    # Try Groq
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5
                },
                timeout=30
            )
            if response.ok:
                return {"success": True, "plan": response.json()["choices"][0]["message"]["content"], "error": ""}
        except: pass
        
    # Try Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            if response.ok:
                text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text: return {"success": True, "plan": text, "error": ""}
        except: pass

    return {"success": False, "error": "AI unavailable"}


# -----------------------------------------------------------
# 🗣️ التحدث مع الكتاب
# -----------------------------------------------------------
def chat_with_book_context(book_info: dict, user_msg: str, history: list = None) -> dict:
    """
    الدردشة مع سياق الكتاب (يتقمص الـ AI دور الكتاب/المؤلف)
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "reply": "عذراً، خدمة الذكاء الاصطناعي غير متوفرة.", "error": "No API Key"}
        
    title = book_info.get("title", "")
    author = book_info.get("author", "")
    desc = book_info.get("description", "")
    
    system_prompt = f"""أنت الآن تتقمص شخصية هذا الكتاب أو مؤلفه:
العنوان: {title}
المؤلف: {author}
الوصف: {desc[:500]}

تعليمات:
1. أجب عن أسئلة المستخدم بصيغة المتكلم (أنا الكتاب/المؤلف).
2. استخدم المعلومات المتوفرة عن الكتاب للإجابة.
3. كن ودوداً وعميقاً في إجاباتك.
4. تحدث باللغة العربية.
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # إضافة السجل السابق (آخر 4 رسائل)
    if history:
        for msg in history[-4:]:
            role = "user" if msg.get("is_user") else "assistant"
            messages.append({"role": role, "content": msg.get("text", "")})
            
    messages.append({"role": "user", "content": user_msg})
    
    import requests
    
    # Groq First
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.7
                }, 
                timeout=30
            )
            if response.ok:
                return {"success": True, "reply": response.json()["choices"][0]["message"]["content"], "error": ""}
        except Exception as e: print(f"Book Chat Groq Error: {e}")

    # Gemini Fallback (Simplified, no history in same format easily, just prompt)
    if gemini_key:
        try:
            # Combine for Gemini (since it's stateless here effectively unless we construct chat structure)
            full_prompt = system_prompt + "\n\n"
            if history:
                for h in history[-4:]:
                    role = "User" if h.get("is_user") else "Book"
                    full_prompt += f"{role}: {h.get('text')}\n"
            full_prompt += f"User: {user_msg}\nBook:"
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": full_prompt}]}]},
                timeout=30
            )
            if response.ok:
                if text: return {"success": True, "reply": text, "error": ""}
        except Exception as e: print(f"Book Chat Gemini Error: {e}")

    return {"success": False, "reply": "عذراً، حدث خطأ أثناء الاتصال بالكتاب.", "error": "Failed"}


# -----------------------------------------------------------
# 🧠 مسابقة الكتاب
# -----------------------------------------------------------
def generate_book_quiz(book_info: dict) -> dict:
    """
    توليد أسئلة اختبار قصيرة من محتوى الكتاب
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "quiz": [], "error": "No API Key"}
        
    title = book_info.get("title", "")
    desc = book_info.get("description", "")
    
    prompt = f"""Generate a short 3-question quiz (JSON format) about this book concept/genre:
Title: {title}
Description: {desc[:800]}

Format Requirements:
- Output ONLY valid JSON list.
- Each item: {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "The correct option text"}}
- Language: Arabic.
- Questions should be general enough to be answerable from the distinct description or general knowledge about this famous book (if famous). If obscure, base it strictly on description.
"""
    
    import requests
    import json
    import re

    def parse_quiz_json(text):
        try:
            # Extract JSON array
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except:
            return []

    # Try Groq
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5
                },
                timeout=30
            )
            if response.ok:
                content = response.json()["choices"][0]["message"]["content"]
                quiz = parse_quiz_json(content)
                if quiz: return {"success": True, "quiz": quiz, "error": ""}
        except Exception as e: print(f"Quiz Groq Error: {e}")

    # Try Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            if response.ok:
                text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                quiz = parse_quiz_json(text)
                if quiz: return {"success": True, "quiz": quiz, "error": ""}
        except: pass

    return {"success": False, "quiz": [], "error": "Failed to generate"}


# -----------------------------------------------------------
# 📜 اقتباسات ذكية
# -----------------------------------------------------------
def extract_book_quotes(book_info: dict) -> dict:
    """
    استخراج اقتباسات ملهمة من الكتاب
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "quotes": [], "error": "No API Key"}
        
    title = book_info.get("title", "")
    author = book_info.get("author", "")
    
    prompt = f"""Extract or generate 3 inspiring quotes (Arabic) attributed to the book "{title}" by {author}.
Format: JSON list of strings ["Quote 1", "Quote 2", "Quote 3"].
If the book is not famous, generate quotes reflecting its likely themes based on title.
"""

    import requests
    import json
    import re
    
    def parse_quotes_json(text):
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match: return json.loads(match.group())
            return []
        except: return []

    # Try Groq
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6
                },
                timeout=30
            )
            if response.ok:
                content = response.json()["choices"][0]["message"]["content"]
                quotes = parse_quotes_json(content)
                if quotes: return {"success": True, "quotes": quotes}
        except: pass
        
    # Try Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            if response.ok:
                text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                quotes = parse_quotes_json(text)
                if quotes: return {"success": True, "quotes": quotes}
        except: pass

    return {"success": False, "quotes": [], "error": "Failed"}


# -----------------------------------------------------------
# 📊 تحليل عادات القراءة
# -----------------------------------------------------------
def analyze_reading_habits(user_id: int) -> dict:
    """
    تحليل شامل لعادات القراءة للمستخدم مع نصائح AI
    """
    from .models import BookStatus, UserRatingCF, BookReview, Book, SearchHistory
    from .extensions import db
    from sqlalchemy import func
    from datetime import datetime, timedelta
    
    stats = {
        "total_books": 0,
        "finished_books": 0,
        "favorite_books": 0,
        "later_books": 0,
        "average_rating": 0,
        "total_reviews": 0,
        "recent_searches": 0,
        "top_genres": [],
        "monthly_activity": [],
        "ai_tips": []
    }
    
    try:
        # إحصائيات الكتب
        stats["total_books"] = Book.query.filter_by(owner_id=user_id).count()
        
        # حالات الكتب
        statuses = BookStatus.query.filter_by(user_id=user_id).all()
        genre_counter = {}
        
        for s in statuses:
            if s.status == "finished": stats["finished_books"] += 1
            elif s.status == "favorite": stats["favorite_books"] += 1
            elif s.status == "later": stats["later_books"] += 1
            
            # استخراج التصنيفات (محاكاة لأننا لا نملك جدول تصنيفات دقيق محلياً بعد)
            if s.book:
                # محاولة استنتاج التصنيف من العنوان أو استخدام تصنيف افتراضي
                # في التطبيق الحقيقي سنحتاج لجدول BookGenre
                title_lower = s.book.title.lower()
                genre = "General"
                if "python" in title_lower or "code" in title_lower or "programming" in title_lower: genre = "Programming"
                elif "fiction" in title_lower or "novel" in title_lower or "story" in title_lower: genre = "Fiction"
                elif "science" in title_lower or "physics" in title_lower: genre = "Science"
                elif "history" in title_lower: genre = "History"
                elif "art" in title_lower or "design" in title_lower: genre = "Art"
                
                genre_counter[genre] = genre_counter.get(genre, 0) + 1

        # ترتيب التصنيفات
        stats["top_genres"] = sorted([{"name": k, "count": v} for k, v in genre_counter.items()], key=lambda x: x["count"], reverse=True)[:5]
        
        # استنتاج المزاج (Mood)
        # خوارزمية بسيطة: الكتاب العلمي = فضولي، الرواية = حالم، البرمجة = مركز
        mood_scores = {"curious": 0, "dreamy": 0, "focused": 0, "adventurous": 0}
        for g in stats["top_genres"]:
            name = g["name"]
            if name in ["Science", "History"]: mood_scores["curious"] += g["count"]
            elif name in ["Fiction", "Art"]: mood_scores["dreamy"] += g["count"]
            elif name in ["Programming"]: mood_scores["focused"] += g["count"]
            else: mood_scores["adventurous"] += g["count"]
            
        stats["current_mood"] = max(mood_scores, key=mood_scores.get) if any(mood_scores.values()) else "Explorer"

        # متوسط التقييمات
        avg_rating = db.session.query(func.avg(UserRatingCF.rating)).filter_by(user_id=user_id).scalar()
        stats["average_rating"] = round(avg_rating, 1) if avg_rating else 0
        
        # عدد المراجعات
        stats["total_reviews"] = BookReview.query.filter_by(user_id=user_id).count()
        
        # عمليات البحث الأخيرة (آخر 30 يوم)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        stats["recent_searches"] = db.session.query(SearchHistory).filter(
            SearchHistory.user_id == user_id,
            SearchHistory.created_at >= thirty_days_ago
        ).count()
        
        # النشاط الشهري (آخر 6 أشهر)
        for i in range(6):
            month_start = datetime.utcnow().replace(day=1) - timedelta(days=30*i)
            month_end = month_start + timedelta(days=30)
            count = BookStatus.query.filter(
                BookStatus.user_id == user_id,
                BookStatus.status == "finished",
                BookStatus.created_at >= month_start,
                BookStatus.created_at < month_end
            ).count()
            stats["monthly_activity"].append({
                "month": month_start.strftime("%b"),
                "count": count
            })
        stats["monthly_activity"].reverse()
        
        # توليد نصائح AI
        stats["ai_tips"] = _generate_reading_tips(stats)
        
    except Exception as e:
        print(f"Reading Analytics Error: {e}")
    
    return {"success": True, "stats": stats}


def _generate_reading_tips(stats: dict) -> list:
    """توليد نصائح مخصصة بناءً على الإحصائيات"""
    tips = []
    
    # نصيحة التنوع (Smart Recommendation)
    top_genre = stats["top_genres"][0]["name"] if stats["top_genres"] else None
    if top_genre == "Programming":
        tips.append("🚀 لقد ركزت كثيراً على البرمجة مؤخراً، ما رأيك في قليل من الأدب الخيالي (Fiction) لتجديد نشاطك الإبداعي؟")
    elif top_genre == "Fiction":
        tips.append("📖 أنت غارق في العوالم الخيالية!، جرب قراءة كتاب في التاريخ أو العلوم لربط الخيال بالواقع.")
    elif top_genre == "Science":
        tips.append("🔬 العقل العلمي يحتاج للراحة أحياناً، جرب قراءة سيرة ذاتية ملهمة.")
    else:
        tips.append("💡 حاول تنويع قراءتك بين الكتب العلمية والأدبية لتوسيع آفاقك.")

    if stats["finished_books"] == 0:
        tips.append("📚 ابدأ بإنهاء كتاب واحد هذا الأسبوع!")
    elif stats["finished_books"] < 5:
        tips.append("🎯 أنت على المسار الصحيح! حاول إنهاء كتاب إضافي هذا الشهر.")
    
    if stats["later_books"] > 10:
        tips.append("📖 لديك قائمة انتظار طويلة! حاول تحديد الأولويات.")
    
    return tips


# -----------------------------------------------------------
# 🎨 توليد غلاف AI
# -----------------------------------------------------------
def generate_ai_cover(book_info: dict) -> dict:
    """
    توليد غلاف فني للكتاب باستخدام Pollinations.ai (مجاني)
    """
    import urllib.parse
    
    title = book_info.get("title", "Book")
    author = book_info.get("author", "")
    description = book_info.get("description", "")[:200]
    
    # بناء prompt للصورة - تجريدي وبسيط
    prompt = f"minimalist book cover design for '{title}'"
    if author:
        prompt += f" by {author}"
    prompt += ". Professional, elegant, abstract art, high quality"
    
    # إضافة سياق من الوصف مع تجنب الروبوتات
    lower_desc = description.lower()
    if "fiction" in lower_desc or "novel" in lower_desc:
        prompt += ", fantasy elements, dramatic lighting"
    elif "science" in lower_desc or "programming" in lower_desc:
        # تجنب الروبوتات في الكتب التقنية
        prompt += ", geometric shapes, modern tech aesthetic, abstract digital art, NO ROBOTS"
    elif "history" in lower_desc:
        prompt += ", historical elements, vintage style, paper texture"
    else:
        prompt += ", elegant typography, minimalist"
    
    # Pollinations.ai URL (الرابط الجديد)
    encoded_prompt = urllib.parse.quote(prompt)
    import hashlib
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 10000
    
    # استخدام الرابط الجديد: https://pollinations.ai/p/
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=400&height=600&nologo=true&seed={seed}&model=flux"
    
    return {
        "success": True,
        "cover_url": image_url,
        "source": "pollinations_ai"
    }



# -----------------------------------------------------------
# 🧠 تحليل سلوك المستخدم (User Behavior Analysis)
# -----------------------------------------------------------
def get_user_behavior_profile(user_id: int) -> dict:
    """
    تحليل شامل لسلوك المستخدم لفهم اهتماماته.
    
    يجمع البيانات من:
    - BookStatus (المفضلة، المنتهية، للقراءة لاحقاً)
    - SearchHistory (عمليات البحث)
    - UserRatingCF (التقييمات)
    - UserBookView (المشاهدات)
    - UserPreference (الاهتمامات المسجلة)
    
    Returns:
        dict: ملف سلوك المستخدم الشامل
    """
    from .models import (
        BookStatus, SearchHistory, UserRatingCF, UserBookView, 
        UserPreference, Book, BookReview
    )
    from .extensions import db
    from sqlalchemy import func
    from datetime import datetime, timedelta
    from collections import Counter
    
    profile = {
        "favorite_genres": [],
        "favorite_authors": [],
        "recent_interests": [],
        "reading_patterns": {},
        "behavior_summary": "",
        "raw_data": {}
    }
    
    try:
        # 1. جمع الكتب المفضلة والمنتهية
        favorite_books = []
        finished_books = []
        
        statuses = BookStatus.query.filter_by(user_id=user_id).all()
        for s in statuses:
            if s.book:
                book_info = {
                    "title": s.book.title,
                    "author": s.book.author,
                    "google_id": s.book.google_id
                }
                if s.status == "favorite":
                    favorite_books.append(book_info)
                elif s.status == "finished":
                    finished_books.append(book_info)
        
        profile["raw_data"]["favorites"] = favorite_books[:10]
        profile["raw_data"]["finished"] = finished_books[:10]
        
        # 2. استخراج المؤلفين المفضلين
        all_books = favorite_books + finished_books
        authors = [b["author"] for b in all_books if b.get("author")]
        author_counts = Counter(authors)
        profile["favorite_authors"] = [a[0] for a in author_counts.most_common(5)]
        
        # 3. جمع عمليات البحث الأخيرة (آخر 30 يوم)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        searches = db.session.query(SearchHistory).filter(
            SearchHistory.user_id == user_id,
            SearchHistory.created_at >= thirty_days_ago
        ).order_by(SearchHistory.created_at.desc()).limit(20).all()
        
        search_queries = [s.query for s in searches if s.query]
        profile["recent_interests"] = list(dict.fromkeys(search_queries))[:10]  # unique
        
        # 4. جمع التقييمات العالية (4-5 نجوم)
        high_ratings = UserRatingCF.query.filter(
            UserRatingCF.user_id == user_id,
            UserRatingCF.rating >= 4
        ).limit(20).all()
        
        rated_google_ids = [r.google_id for r in high_ratings]
        profile["raw_data"]["high_rated_ids"] = rated_google_ids[:10]
        
        # 5. جمع اهتمامات المستخدم المسجلة
        prefs = UserPreference.query.filter_by(user_id=user_id)\
            .order_by(UserPreference.weight.desc()).limit(10).all()
        registered_interests = [p.topic for p in prefs if p.topic and not p.topic.startswith("special:")]
        profile["favorite_genres"] = registered_interests[:5]
        
        # 6. أنماط القراءة
        total_ratings = UserRatingCF.query.filter_by(user_id=user_id).count()
        avg_rating = db.session.query(func.avg(UserRatingCF.rating))\
            .filter(UserRatingCF.user_id == user_id).scalar()
        
        profile["reading_patterns"] = {
            "total_rated": total_ratings,
            "avg_rating": round(float(avg_rating), 1) if avg_rating else 0,
            "favorites_count": len(favorite_books),
            "finished_count": len(finished_books)
        }
        
        # 7. توليد ملخص السلوك
        profile["behavior_summary"] = _generate_behavior_summary(profile)
        
    except Exception as e:
        print(f"[BehaviorProfile] Error: {e}")
        import traceback
        traceback.print_exc()
    
    return profile


def _generate_behavior_summary(profile: dict) -> str:
    """توليد ملخص نصي لسلوك المستخدم"""
    parts = []
    
    if profile.get("favorite_genres"):
        parts.append(f"مهتم بـ: {', '.join(profile['favorite_genres'][:3])}")
    
    if profile.get("favorite_authors"):
        parts.append(f"يفضل كتابات: {', '.join(profile['favorite_authors'][:2])}")
    
    if profile.get("recent_interests"):
        parts.append(f"بحث مؤخراً عن: {', '.join(profile['recent_interests'][:2])}")
    
    patterns = profile.get("reading_patterns", {})
    if patterns.get("finished_count", 0) > 5:
        parts.append("قارئ نشط")
    elif patterns.get("favorites_count", 0) > 10:
        parts.append("يحفظ الكثير من الكتب")
    
    return " | ".join(parts) if parts else "مستخدم جديد"


# -----------------------------------------------------------
# 🤖 التوصيات الذكية بالـ AI
# -----------------------------------------------------------
def get_ai_personalized_recommendations(user_id: int, limit: int = 12) -> dict:
    """
    توصيات مخصصة باستخدام AI.
    
    1. يجلب ملف سلوك المستخدم
    2. يرسل للـ AI لتحليل الاهتمامات واقتراح مواضيع
    3. يبحث عن كتب في المواضيع المقترحة
    4. يضيف أسباب ذكية لكل توصية
    
    Returns:
        dict: {"books": [...], "ai_analysis": "...", "suggested_topics": [...]}
    """
    import json
    
    result = {
        "books": [],
        "ai_analysis": "",
        "suggested_topics": [],
        "success": False
    }
    
    try:
        # 1. جلب ملف السلوك
        profile = get_user_behavior_profile(user_id)
        
        if not profile.get("favorite_genres") and not profile.get("recent_interests"):
            # مستخدم جديد (Cold Start) - نعرض مزيج ترحيبي
            default_topics = ["علوم الفضاء", "الذكاء الاصطناعي", "روايات عالمية", "تطوير الذات"]
            result["ai_analysis"] = "أهلاً بك! بما أننا لا نعرف ذوقك بعد، جهزنا لك مختارات متنوعة لتبدأ رحلتك 🚀"
            result["suggested_topics"] = default_topics
            result["books"] = _fetch_books_for_topics(default_topics, limit=limit, user_profile=profile)
            result["success"] = True
            return result
        
        # 2. طلب تحليل من AI
        ai_suggestions = _get_ai_topic_suggestions(profile)
        
        if ai_suggestions.get("success"):
            result["ai_analysis"] = ai_suggestions.get("analysis", "")
            result["suggested_topics"] = ai_suggestions.get("topics", [])
            
            # 3. البحث عن كتب في المواضيع المقترحة
            books = _fetch_books_for_topics(
                ai_suggestions.get("topics", []),
                limit=limit,
                user_profile=profile
            )
            result["books"] = books
            result["success"] = True
        else:
            # Fallback: استخدام الاهتمامات المسجلة مباشرة
            topics = profile.get("favorite_genres", [])[:3] or profile.get("recent_interests", [])[:3]
            if topics:
                result["suggested_topics"] = topics
                result["books"] = _fetch_books_for_topics(topics, limit=limit, user_profile=profile)
                result["ai_analysis"] = f"توصيات بناءً على اهتماماتك في: {', '.join(topics)}"
                result["success"] = True
                
    except Exception as e:
        print(f"[AI Recommendations] Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def _get_ai_topic_suggestions(profile: dict) -> dict:
    """
    استخدام AI لتحليل سلوك المستخدم واقتراح مواضيع كتب.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if not groq_key and not gemini_key:
        return {"success": False, "error": "No API key"}
    
    # بناء البرومبت
    prompt = f"""أنت خبير توصيات كتب. حلل ملف القارئ التالي واقترح 3 مواضيع كتب جديدة ستعجبه:

📊 ملف القارئ:
- الاهتمامات المسجلة: {', '.join(profile.get('favorite_genres', [])[:5]) or 'غير محدد'}
- المؤلفين المفضلين: {', '.join(profile.get('favorite_authors', [])[:3]) or 'غير محدد'}
- عمليات البحث الأخيرة: {', '.join(profile.get('recent_interests', [])[:5]) or 'لا يوجد'}
- ملخص السلوك: {profile.get('behavior_summary', 'مستخدم جديد')}

المطلوب:
1. اكتب تحليلاً قصيراً (جملة واحدة) عن ذوق القارئ
2. اقترح 3 مواضيع كتب جديدة باللغة الإنجليزية (للبحث في APIs)

أجب بصيغة JSON فقط:
{{"analysis": "تحليل قصير بالعربي", "topics": ["topic1", "topic2", "topic3"]}}
"""

    import json
    import re
    
    def parse_json_response(text):
        try:
            # محاولة استخراج JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        except:
            return None
    
    # محاولة Groq أولاً
    if groq_key:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=15
            )
            if response.ok:
                content = response.json()["choices"][0]["message"]["content"]
                parsed = parse_json_response(content)
                if parsed:
                    return {"success": True, **parsed}
        except Exception as e:
            print(f"[AI Topics] Groq error: {e}")
    
    # Fallback إلى Gemini
    if gemini_key:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15
            )
            if response.ok:
                text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                parsed = parse_json_response(text)
                if parsed:
                    return {"success": True, **parsed}
        except Exception as e:
            print(f"[AI Topics] Gemini error: {e}")
    
    return {"success": False, "error": "AI call failed"}


def _fetch_books_for_topics(topics: list, limit: int = 12, user_profile: dict = None) -> list:
    """
    جلب كتب من Google Books للمواضيع المقترحة مع أسباب ذكية.
    """
    books = []
    seen_ids = set()
    per_topic = max(4, limit // len(topics)) if topics else limit
    
    for topic in topics[:3]:
        try:
            items, _ = fetch_google_books(topic, max_results=per_topic)
            for item in items or []:
                gid = item.get("id")
                if not gid or gid in seen_ids:
                    continue
                seen_ids.add(gid)
                
                vi = item.get("volumeInfo", {})
                imgs = vi.get("imageLinks", {}) or {}
                cover = imgs.get("thumbnail", "")
                if cover.startswith("http://"):
                    cover = "https://" + cover[7:]
                
                # توليد سبب ذكي للتوصية
                reason = _generate_smart_reason(topic, vi, user_profile)
                
                books.append({
                    "id": gid,
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors", [])) or "غير معروف",
                    "cover": cover,
                    "source": "AI Personalized",
                    "reason": reason,
                    "rating": vi.get("averageRating"),
                    "ratings_count": vi.get("ratingsCount")
                })
                
                if len(books) >= limit:
                    break
                    
        except Exception as e:
            print(f"[FetchBooks] Error for topic '{topic}': {e}")
        
        if len(books) >= limit:
            break
    
    return books


def _generate_smart_reason(topic: str, book_info: dict, user_profile: dict = None) -> str:
    """
    توليد سبب ذكي ومخصص للتوصية.
    """
    reasons = []
    
    # التحقق من المؤلف المفضل
    if user_profile and user_profile.get("favorite_authors"):
        book_authors = book_info.get("authors", [])
        for author in book_authors:
            if author in user_profile["favorite_authors"]:
                return f"✍️ من كاتبك المفضل: {author}"
    
    # التحقق من التصنيف
    categories = book_info.get("categories", [])
    if user_profile and user_profile.get("favorite_genres"):
        for cat in categories:
            for genre in user_profile["favorite_genres"]:
                if genre.lower() in cat.lower():
                    return f"📚 يتوافق مع اهتمامك بـ {genre}"
    
    # سبب عام بناءً على الموضوع
    return f"🤖 مقترح لأنك مهتم بـ {topic}"


# -----------------------------------------------------------
# 🔄 تحديث التفضيلات تلقائياً
# -----------------------------------------------------------
def update_user_preferences_from_behavior(user_id: int, action: str, book_info: dict):
    """
    تحديث أوزان UserPreference بناءً على سلوك المستخدم.
    
    Args:
        user_id: معرف المستخدم
        action: "view", "favorite", "finished", "rated_high", "search"
        book_info: معلومات الكتاب (categories, author, title, etc.)
        
    الأوزان:
        - view: +1
        - favorite: +10
        - finished: +15
        - rated_high (4-5): +20
        - search: +5
    """
    from .models import UserPreference
    from .extensions import db
    
    weight_map = {
        "view": 15,
        "favorite": 40,
        "finished": 60,
        "rated_high": 120,
        "search": 30
    }
    
    weight = weight_map.get(action, 1)
    
    # استخراج المواضيع من الكتاب
    topics = []
    
    # من التصنيفات
    categories = book_info.get("categories", [])
    if isinstance(categories, str):
        categories = [categories]
    topics.extend(categories[:2])
    
    # من العنوان (كلمات رئيسية)
    title = book_info.get("title", "")
    if title:
        # استخراج كلمات مهمة (أكثر من 4 حروف)
        import re
        words = re.findall(r'\b[A-Za-z\u0600-\u06FF]{5,}\b', title)
        topics.extend(words[:2])
    
    # تحديث الأوزان
    for topic in topics:
        if not topic or len(topic) < 3:
            continue
            
        try:
            pref = UserPreference.query.filter_by(user_id=user_id, topic=topic).first()
            if pref:
                pref.weight = (pref.weight or 0) + weight
            else:
                pref = UserPreference(user_id=user_id, topic=topic, weight=weight)
                db.session.add(pref)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[UpdatePrefs] Error: {e}")
