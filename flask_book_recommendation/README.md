# نظام توصية الكتب - Flask Book Recommendation System

نظام توصية كتب متقدم مبني على Flask يجمع بين عدة تقنيات للذكاء الاصطناعي والتعلم الآلي.

## ✨ الميزات الرئيسية

- 📚 **أنظمة توصية متعددة**:
  - Collaborative Filtering (التوصية التعاونية)
  - Content-Based Recommendations (التوصية القائمة على المحتوى)
  - Topic-Based Recommendations (التوصية القائمة على المواضيع)
  - Trending Books (الكتب الرائجة)

- 🔍 **بحث ذكي**:
  - دعم اللغة العربية مع ترجمة تلقائية (Gemini AI)
  - بحث في 5 مصادر مختلفة (Google Books, Gutenberg, OpenLibrary, Archive, IT Bookstore)
  - Semantic Search باستخدام Embeddings

- 👤 **ميزات المستخدم**:
  - نظام تسجيل دخول آمن
  - مكتبة شخصية
  - نظام تقييمات
  - تفضيلات المستخدم
  - سجل البحث
  - حالات الكتب (مفضلة، لاحقاً، منتهية)

- 🛡️ **الأمان والأداء**:
  - CSRF Protection
  - نظام Logging منظم
  - Caching للنتائج المكلفة
  - Database Indexes محسّنة

## 🚀 البدء السريع

### 1. التثبيت

```bash
# إنشاء بيئة افتراضية
py -m venv venv

# تفعيل البيئة
venv\Scripts\activate  # Windows
# أو
source venv/bin/activate  # Linux/Mac

# تثبيت المتطلبات
pip install -r requirements.txt
```

### 2. الإعداد

#### إنشاء ملف `.env`:
```env
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key
DB_URL=  # اتركه فارغاً لاستخدام SQLite محلي
LOG_LEVEL=INFO
```

#### تهيئة قاعدة البيانات:
```bash
# تهيئة Migrations (مرة واحدة فقط)
flask db init

# إنشاء Migration أولي
flask db migrate -m "Initial migration"

# تطبيق التغييرات
flask db upgrade
```

### 3. التشغيل

```bash
# طريقة 1: استخدام run.py
python run.py

# طريقة 2: استخدام Flask CLI
flask run

# طريقة 3: استخدام start_bookrec.bat (Windows)
start_bookrec.bat
```

افتح المتصفح على: `http://127.0.0.1:5000`

## 📁 هيكل المشروع

```
flask_book_recommendation/
├── app.py                 # تطبيق Flask الرئيسي
├── config.py              # إعدادات التطبيق
├── extensions.py          # Flask Extensions
├── models.py              # نماذج قاعدة البيانات
├── recommender.py         # أنظمة التوصية
├── utils.py               # دوال مساعدة (APIs, AI)
├── routes/                # Blueprints
│   ├── auth.py           # تسجيل الدخول
│   ├── main.py           # الصفحات الرئيسية
│   ├── explore.py        # صفحة الاستكشاف
│   ├── public.py         # الكتب العامة
│   └── preferences.py    # التفضيلات
└── templates/            # قوالب HTML
```

## 🔧 المتطلبات

- Python 3.8+
- Flask 3.0.3
- SQLAlchemy
- NumPy, Pandas, Scikit-learn
- Flask-Migrate (لإدارة قاعدة البيانات)
- Flask-WTF (لحماية CSRF)
- Flask-Caching (للأداء)

راجع `requirements.txt` للقائمة الكاملة.

## 📊 قاعدة البيانات

### النماذج الرئيسية:
- `User`: المستخدمون
- `Book`: الكتب
- `UserRatingCF`: التقييمات (للتوصية التعاونية)
- `BookEmbedding`: Embeddings للكتب (للتوصية القائمة على المحتوى)
- `UserPreference`: تفضيلات المستخدم
- `SearchHistory`: سجل البحث
- `BookStatus`: حالات الكتب

## 🎯 استخدام أنظمة التوصية

### في الكود:

```python
from flask_book_recommendation.recommender import (
    get_trending,
    get_cf_similar,
    get_content_similar,
    get_topic_based,
    get_homepage_sections
)

# الكتب الرائجة
trending = get_trending(limit=12)

# توصيات CF
cf_recs = get_cf_similar(user_id=1, top_n=30)

# توصيات Content-Based
content_recs = get_content_similar(user_id=1, top_n=30)

# توصيات Topic-Based
topic_recs = get_topic_based(user_id=1, limit=24)

# جميع الأقسام لصفحة الاستكشاف
sections = get_homepage_sections(user_id=1)
```

## 🔐 الأمان

- ✅ تشفير كلمات المرور (Werkzeug)
- ✅ CSRF Protection
- ✅ حماية المسارات بـ `@login_required`
- ✅ استخدام `SECRET_KEY` من متغيرات البيئة

## 📝 Logging

الملفات تُسجل في `flask_book_recommendation/app.log`:
- معلومات التطبيق
- تحذيرات
- أخطاء مع stack trace

## 🚀 النشر

### للإنتاج:

1. **تغيير إعدادات Caching**:
```python
# في config.py
CACHE_TYPE = "RedisCache"
CACHE_REDIS_URL = "redis://localhost:6379/0"
```

2. **تغيير Log Level**:
```env
LOG_LEVEL=WARNING
```

3. **استخدام MySQL**:
```env
DB_URL=mysql+pymysql://user:password@host:port/database
```

4. **تأكد من**:
   - `SECRET_KEY` قوي وآمن
   - `GEMINI_API_KEY` موجود
   - قاعدة البيانات محمية
   - HTTPS مفعّل

## 📚 التوثيق

- `PROJECT_EVALUATION.md`: تقييم شامل للمشروع
- `IMPROVEMENTS_SUMMARY.md`: ملخص التحسينات المطبقة
- `MIGRATION_GUIDE.md`: دليل استخدام Flask-Migrate
- `PREFERENCES_INTEGRATION.md`: دليل تكامل التفضيلات

## 🐛 استكشاف الأخطاء

### مشكلة: "Import flask_migrate could not be resolved"
**الحل**: تأكد من تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

### مشكلة: "CSRF token missing"
**الحل**: أضف `{{ csrf_token() }}` في جميع النماذج

### مشكلة: قاعدة البيانات لا تعمل
**الحل**: 
```bash
flask db upgrade
```

## 🤝 المساهمة

نرحب بالمساهمات! يرجى:
1. Fork المشروع
2. إنشاء branch جديد
3. Commit التغييرات
4. Push للـ branch
5. فتح Pull Request

## 📄 الترخيص

هذا المشروع مفتوح المصدر.

## 👨‍💻 المطور

تم تطويره باستخدام:
- Flask
- SQLAlchemy
- Scikit-learn
- Gemini AI
- Bootstrap (RTL)

---

**آخر تحديث**: 2024
**الإصدار**: 2.0 (مع التحسينات)

