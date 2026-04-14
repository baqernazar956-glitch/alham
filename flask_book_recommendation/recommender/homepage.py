# -*- coding: utf-8 -*-
"""
Homepage sections — get_homepage_sections, get_discovery_picks, get_all_libraries_showcase.
"""
import logging
import random

from ..extensions import cache, db
from ..utils import (
    fetch_google_books, fetch_gutenberg_books,
    fetch_openlib_books, fetch_archive_books,
    fetch_itbook_books,
    translate_to_english_with_gemini
)
from .helpers import _extract_rating_with_fallback
from .collaborative import get_cf_similar
from .content import get_content_similar
from .topic import get_topic_based, get_last_search_recommendations, get_archive_ai_recommendations
from .trending import get_trending, get_trending_by_period
from .hybrid import (
    get_top_rated, get_genre_explorer,
    get_because_you_read, get_similar_users_favorites
)
from .mood import get_mood_based_recommendations

def get_user_context(user_id) -> dict:
    from datetime import datetime
    try:
        from hijri_converter import convert
    except ImportError:
        convert = None
    from ..models import BookStatus, SearchHistory, UserGenre, Genre

    now = datetime.now()
    hour = now.hour
    is_weekend = now.weekday() in [4, 5]  # Friday (4) and Saturday (5)
    
    is_ramadan = False
    if convert:
        hijri_date = convert.Gregorian(now.year, now.month, now.day).to_hijri()
        is_ramadan = (hijri_date.month == 9)

    streak_days = 0
    has_unfinished = False
    last_genre = None
    pace = "normal"

    if user_id:
        try:
            statuses = BookStatus.query.filter_by(user_id=user_id).all()
            reads = [s for s in statuses if s.reading_progress and s.reading_progress > 0]
            streak_days = min(len(reads), 10) # rough approximation for now
            
            unfinished = [s for s in statuses if s.reading_progress and 0 < s.reading_progress < 100]
            has_unfinished = len(unfinished) > 0

            last_search = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(SearchHistory.created_at.desc()).first()
            if last_search and last_search.query:
                last_genre = last_search.query
            else:
                user_genre = UserGenre.query.filter_by(user_id=user_id).first()
                if user_genre:
                    g = Genre.query.get(user_genre.genre_id)
                    if g: last_genre = g.name
        except Exception as e:
            logger.error(f"[Context Engine] Context error: {e}")

    return {
        "hour": hour,
        "is_weekend": is_weekend,
        "is_ramadan": is_ramadan,
        "streak_days": streak_days,
        "last_genre": last_genre,
        "pace": pace,
        "has_unfinished": has_unfinished
    }

def auto_detect_mood(context) -> str:
    if context.get('hour', 12) < 6: return 'calm'
    if context.get('is_ramadan'): return 'spiritual'
    if context.get('is_weekend') and context.get('hour', 0) > 20: return 'adventure'
    if context.get('streak_days', 0) > 7: return 'challenge'
    return 'curious'


logger = logging.getLogger(__name__)


@cache.memoize(timeout=3600)
def _get_cached_ai_section(user_id, limit=12):
    """Cached AI personalized section — once per hour (Task 4: Gemini caching fix)."""
    from ..utils import get_ai_personalized_recommendations
    return get_ai_personalized_recommendations(user_id, limit=limit)


def get_discovery_picks(limit=12):
    """
    حلقة اكتشاف عشوائية: كتب من موضوعات متنوعة لا علاقة لها بتاريخ المستخدم.
    """
    discovery_topics = [
        "Award winning novels", "Best poetry collections",
        "Modern architecture", "Space exploration",
        "Artificial intelligence ethics", "World mythology",
        "Independent cinema", "Classical music history",
        "Marine biology", "Quantum physics",
        "Street art movement", "Culinary science"
    ]
    
    selected = random.sample(discovery_topics, min(3, len(discovery_topics)))
    books = []
    seen = set()
    
    for topic in selected:
        try:
            gb_res = fetch_google_books(topic, max_results=limit // 3 + 1)
            items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
            for it in items or []:
                if not isinstance(it, dict): continue
                gid = it.get("id")
                if not gid or gid in seen: continue
                seen.add(gid)
                vi = it.get("volumeInfo") or {}
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img and img.startswith("http://"): img = img.replace("http://", "https://")
                books.append({
                    "id": gid,
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors") or []),
                    "cover": img,
                    "source": "Discovery",
                    "reason": f"💡 اكتشاف: {topic}",
                    "rating": vi.get("averageRating"),
                })
        except Exception as e:
            logger.error(f"[Discovery] Error for '{topic}': {e}")
    
    random.shuffle(books)
    return books[:limit]


def get_homepage_sections(user_id, recent_query=None):
    """
    ترجع قائمة أقسام لصفحة /explore مع توصيات متنوعة.
    """
    sections = []
    
    # Context & Mood Engine
    context = get_user_context(user_id)
    mood = auto_detect_mood(context)
    
    # Call existing MOOD_MAPPING safely
    safe_mood = mood
    if safe_mood == 'spiritual': safe_mood = 'calm'
    if safe_mood == 'adventure': safe_mood = 'adventurous'
    if safe_mood == 'challenge': safe_mood = 'curious'
    
    try:
        mood_recs = get_mood_based_recommendations(safe_mood, limit=12)
        if mood_recs:
            sections.append({
                "title": f"🎭 مزاجك الآن: {safe_mood}",
                "subtitle": "توصيات متناغمة مع سياقك الحالي (تلقائي)",
                "books": mood_recs,
                "style": "primary",
                "icon": "face-smile",
                "query": f"special:mood-{safe_mood}"
            })
    except Exception as e:
        logger.error(f"[Homepage] Auto Mood error: {e}")

    # 🤖 AI section (cached for 1 hour via _get_cached_ai_section)
    try:
        ai_recs = _get_cached_ai_section(user_id, limit=12)
        if ai_recs.get("success") and ai_recs.get("books"):
            ai_analysis = ai_recs.get("ai_analysis", "")
            subtitle = ai_analysis if ai_analysis else "توصيات مخصصة بناءً على سلوكك وتفضيلاتك"
            sections.append({
                "title": "🤖 مخصص لك بالذكاء الاصطناعي",
                "subtitle": subtitle,
                "books": ai_recs["books"],
                "style": "gradient",
                "icon": "robot",
                "query": "special:ai-personalized",
                "ai_topics": ai_recs.get("suggested_topics", [])
            })
    except Exception as e:
        logger.error(f"[Homepage] AI recommendations error: {e}")

    # 💎 Discovery Picks
    try:
        discovery = get_discovery_picks(limit=12)
        if discovery:
            sections.append({
                "title": "✨ اكتشافات اليوم",
                "subtitle": "عناوين متنوعة اخترناها لك لتجربة قراءة مختلفة",
                "books": discovery,
                "style": "info",
                "icon": "compass",
                "query": "special:discovery"
            })
    except Exception as e:
        logger.error(f"[Homepage] Discovery error: {e}")

    # 0) لأنك بحثت عن...
    last_query_text, last_search_books = get_last_search_recommendations(user_id, limit=20)
    if last_search_books:
        sections.append({
            "title": f"🔍 لأنك بحثت عن «{last_query_text}»",
            "subtitle": "نتائج خاصة بآخر اهتماماتك البحثية",
            "books": last_search_books,
            "style": "danger",
            "icon": "magnifying-glass",
            "query": last_query_text
        })

    # A) CF
    cf_raw = get_cf_similar(user_id, top_n=40)
    if cf_raw:
        sections.append({
            "title": "✨ مختارات لك",
            "subtitle": "باستخدام التوصية التعاونية (مستخدمون يشبهونك في الذوق)",
            "books": cf_raw[:20],
            "style": "primary",
            "icon": "sparkle",
            "query": "special:cf"
        })

    # B) Content-Based
    content_raw = get_content_similar(user_id, top_n=40)
    if content_raw:
        sections.append({
            "title": "📖 لأنك قرأت كتباً معينة",
            "subtitle": "كتب مشابهة في المحتوى والموضوع",
            "books": content_raw[:20],
            "style": "success",
            "icon": "book-open",
            "query": "special:content"
        })

    # C) Topic-based
    topics_result = get_topic_based(user_id, limit=60, recent_query=recent_query)
    if isinstance(topics_result, dict):
        topics_raw = topics_result.get('books', [])
    else:
        topics_raw = topics_result if topics_result else []
    
    if topics_raw:
        sections.append({
            "title": "🎯 من اهتماماتك العامة",
            "subtitle": "بناءً على سجل اهتماماتك الطويل (القديم والجديد)",
            "books": topics_raw,
            "style": "info",
            "icon": "target",
            "query": "special:interests"
        })

    # D) Trending
    community_trend = get_trending(limit=24)
    if community_trend:
        sections.append({
            "title": "🔥 الرائج في مجتمع القرّاء",
            "subtitle": "كتب يقرأها ويضيفها أصدقاؤك في المنصة",
            "books": community_trend,
            "style": "warning",
            "icon": "fire",
            "query": "special:trending"
        })


    # F) Genre Explorer
    try:
        genre_explorer = get_genre_explorer(user_id, limit=12)
        if genre_explorer:
            sections.append({
                "title": "🧭 استكشف تصنيفاً جديداً",
                "subtitle": "وسّع آفاقك مع تصنيفات لم تجربها من قبل",
                "books": genre_explorer,
                "style": "accent",
                "icon": "compass",
                "query": "special:genre-explorer"
            })
    except Exception as e:
        logger.error(f"[Homepage] Genre Explorer error: {e}")

    # G) Because You Read
    try:
        because_result = get_because_you_read(user_id, limit=12)
        if because_result and because_result.get('recommendations'):
            source_book = because_result.get('source_book', {})
            source_title = source_book.get('title', 'كتاب')
            sections.append({
                "title": f"📚 لأنك قرأت «{source_title[:30]}»",
                "subtitle": "كتب مشابهة لما أحببته مؤخراً",
                "books": because_result['recommendations'],
                "style": "success",
                "icon": "heart",
                "query": "special:because-you-read"
            })
    except Exception as e:
        logger.error(f"[Homepage] Because You Read error: {e}")

    # H) Similar Users Favorites
    try:
        similar_users = get_similar_users_favorites(user_id, limit=12)
        if similar_users:
            sections.append({
                "title": "👥 مفضلات قراء مشابهين",
                "subtitle": "كتب يحبها مستخدمون لديهم ذوق مشابه لذوقك",
                "books": similar_users,
                "style": "primary",
                "icon": "users-three",
                "query": "special:similar-users"
            })
    except Exception as e:
        logger.error(f"[Homepage] Similar Users error: {e}")

    # I) Trending by Period
    try:
        weekly_trending = get_trending_by_period('week', limit=12)
        if weekly_trending:
            sections.append({
                "title": "📈 رائج هذا الأسبوع",
                "subtitle": "أكثر الكتب شعبية في الأيام السبعة الماضية",
                "books": weekly_trending,
                "style": "danger",
                "icon": "trend-up",
                "query": "special:weekly-trending"
            })
    except Exception as e:
        logger.error(f"[Homepage] Weekly Trending error: {e}")

    # J) Top Rated
    try:
        top_rated = get_top_rated(limit=12)
        if top_rated:
            sections.append({
                "title": "⭐ الأعلى تقييماً",
                "subtitle": "أفضل الكتب حسب تقييمات المجتمع",
                "books": top_rated,
                "style": "gold",
                "icon": "star",
                "query": "special:top-rated"
            })
    except Exception as e:
        logger.error(f"[Homepage] Top Rated error: {e}")

    # K) Archive AI
    try:
        archive_recs = get_archive_ai_recommendations(user_id, limit=12)
        if archive_recs:
            sections.append({
                "title": "🌐 كنوز من أرشيف الإنترنت",
                "subtitle": "كتب نادرة ومجانية من Internet Archive",
                "books": archive_recs,
                "style": "info",
                "icon": "globe",
                "query": "special:archive"
            })
    except Exception as e:
        logger.error(f"[Homepage] Archive AI error: {e}")

    # L) Cold Start
    if len(sections) < 2:
        try:
            default_topics = ["programming", "science fiction", "history", "psychology"]
            cold_start_books = []
            
            for topic in default_topics[:2]:
                topic_result = fetch_google_books(topic, max_results=8)
                items = topic_result[0] if isinstance(topic_result, tuple) else topic_result
                
                for it in items or []:
                    if not isinstance(it, dict): continue
                    gid = it.get("id")
                    if not gid: continue
                    
                    vi = it.get("volumeInfo") or {}
                    img = (vi.get("imageLinks") or {}).get("thumbnail")
                    if img:
                        if img.startswith("http://"): img = img.replace("http://", "https://")
                    
                    cold_start_books.append({
                        "id": gid,
                        "title": vi.get("title"),
                        "author": ", ".join(vi.get("authors") or []),
                        "cover": img,
                        "source": "Google Books",
                        "reason": f"🌟 موصى به في {topic}",
                        "rating": vi.get("averageRating"),
                    })
            
            if cold_start_books:
                sections.insert(0, {
                    "title": "🌟 اكتشف كتباً رائعة",
                    "subtitle": "ابدأ رحلتك في القراءة مع هذه الاقتراحات",
                    "books": cold_start_books[:16],
                    "style": "gradient",
                    "icon": "compass",
                    "query": "special:discover"
                })
                # omitted fallback addition logic for brevity if sections empty
        except Exception as e:
            logger.error(f"[Homepage] Cold start error: {e}")

    # --- Task 2 & 3: Context-Aware Dynamic Ordering & Renaming ---
    for sec in sections:
        if "مختارات لك" in sec["title"]:
            if context.get('hour', 12) < 7:
                sec["title"] = "🌅 ابدأ يومك بهذا"
            elif context.get('hour', 12) > 21:
                sec["title"] = "🌙 قراءة ليلية مريحة"
            elif context.get('is_weekend'):
                sec["title"] = "🎉 لديك وقت — كتاب يستحق"
            elif context.get('streak_days', 0) > 7:
                sec["title"] = "🔥 أسبوع متواصل — استمر"

    if context.get('hour', 12) < 6:
        # Generate 'قصيرة الآن'
        try:
            short_res = fetch_google_books("short stories", max_results=8)
            items = short_res[0] if isinstance(short_res, tuple) else short_res
            short_books = []
            for it in items or []:
                if not isinstance(it, dict): continue
                vi = it.get("volumeInfo") or {}
                if vi.get("pageCount", 999) < 150:
                    gid = it.get("id")
                    img = (vi.get("imageLinks") or {}).get("thumbnail")
                    if img and img.startswith("http://"): img = img.replace("http://", "https://")
                    short_books.append({
                        "id": gid, 
                        "title": vi.get("title"), 
                        "author": ", ".join(vi.get("authors") or []), 
                        "cover": img, 
                        "rating": vi.get("averageRating"),
                        "reason": "⏳ أقل من 150 صفحة"
                    })
            if short_books:
                sections.insert(0, {
                    "title": "☕ قصيرة الآن",
                    "subtitle": "قراءات سريعة تناسب فجر اليوم",
                    "books": short_books,
                    "style": "info",
                    "icon": "clock",
                    "query": "special:short"
                })
        except: pass
        
    elif context.get('is_ramadan'):
        try:
            ram_res = fetch_google_books("islamic self-development", max_results=8)
            items = ram_res[0] if isinstance(ram_res, tuple) else ram_res
            ram_books = []
            for it in items or []:
                if not isinstance(it, dict): continue
                vi = it.get("volumeInfo") or {}
                gid = it.get("id")
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img and img.startswith("http://"): img = img.replace("http://", "https://")
                ram_books.append({
                    "id": gid, 
                    "title": vi.get("title"), 
                    "author": ", ".join(vi.get("authors") or []), 
                    "cover": img, 
                    "rating": vi.get("averageRating"),
                    "reason": "🌙 أجواء روحانية"
                })
            if ram_books:
                sections.insert(0, {
                    "title": "🌙 التطوير الذاتي + الإسلاميات",
                    "subtitle": "تناسب أجواء رمضان",
                    "books": ram_books,
                    "style": "success",
                    "icon": "moon",
                    "query": "special:ramadan"
                })
        except: pass
        
    elif context.get('is_weekend') and context.get('hour', 12) > 20:
        try:
            week_res = fetch_google_books("epic fantasy adventure", max_results=12)
            items = week_res[0] if isinstance(week_res, tuple) else week_res
            week_books = []
            for it in items or []:
                if not isinstance(it, dict): continue
                vi = it.get("volumeInfo") or {}
                if vi.get("pageCount", 0) > 300:
                    gid = it.get("id")
                    img = (vi.get("imageLinks") or {}).get("thumbnail")
                    if img and img.startswith("http://"): img = img.replace("http://", "https://")
                    week_books.append({
                        "id": gid, 
                        "title": vi.get("title"), 
                        "author": ", ".join(vi.get("authors") or []), 
                        "cover": img, 
                        "rating": vi.get("averageRating"),
                        "reason": "🏕️ مغامرة مشوقة للحظات الفراغ"
                    })
            if week_books:
                sections.insert(0, {
                    "title": "🏕️ روايات طويلة + مغامرات",
                    "subtitle": "استمتع بعطلة نهاية الأسبوع",
                    "books": week_books,
                    "style": "danger",
                    "icon": "fire",
                    "query": "special:weekend-adventure"
                })
        except: pass

    # Always ensure the "continue reading" is first if available (mocking it if they have it)
    if context.get('has_unfinished'):
        sections.insert(0, {
            "title": "🔖 استكمل قراءة",
            "subtitle": "الكتب التي لم تنهها بعد",
            "books": [], # Handled by the real frontend or specialized API; placeholder for structure
            "style": "warning",
            "icon": "book-open",
            "query": "special:continue-reading"
        })

    # --- Task 4: UCB1 Bandit Injection (Only for General sections) ---
    try:
        from .exploration import UCB1Explorer
        for sec in sections:
            # Skip for high-intent personalized sections
            title = sec.get('title', '')
            if any(x in title for x in ["مخصص لك", "لأنك بحثت", "لأنك قرأت", "مفضلات", "استكمل قراءة"]):
                continue
                
            if sec.get('books') and len(sec['books']) > 0:
                sec['books'] = UCB1Explorer.inject_exploration(sec['books'])
    except Exception as e:
        logger.error(f"[Homepage] UCB1 Injection error: {e}")

    return sections


@cache.memoize(timeout=43200)
def get_all_libraries_showcase(query="books", limit_per_source=6):
    """
    جلب كتب من جميع المصادر الخمسة لعرضها معاً.
    """
    sections = []
    
    search_query = query
    if any("\u0600" <= c <= "\u06FF" for c in query):
        try:
            translated = translate_to_english_with_gemini(query)
            if translated and translated.strip():
                search_query = translated
        except:
            pass
    
    # 1. Google Books
    try:
        gb_res = fetch_google_books(search_query, max_results=limit_per_source)
        items = gb_res[0] if isinstance(gb_res, tuple) else gb_res
        if items:
            google_books = []
            for it in items or []:
                if not isinstance(it, dict): continue
                gid = it.get("id")
                if not gid: continue
                vi = it.get("volumeInfo") or {}
                img = (vi.get("imageLinks") or {}).get("thumbnail")
                if img and img.startswith("http://"):
                    img = img.replace("http://", "https://")
                google_books.append({
                    "id": gid, "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors") or []),
                    "cover": img, "source": "Google Books",
                    "rating": vi.get("averageRating"),
                    "ratings_count": vi.get("ratingsCount"),
                })
            if google_books:
                sections.append({
                    "title": "🔵 Google Books",
                    "subtitle": "أكبر مكتبة رقمية في العالم",
                    "books": google_books,
                    "style": "google", "icon": "google-logo",
                })
    except Exception as e:
        logger.error(f"[AllLibs] Google error: {e}")
    
    # 2. Internet Archive
    try:
        ia_books = fetch_archive_books(search_query, limit=limit_per_source)
        if ia_books:
            sections.append({
                "title": "🟡 Internet Archive",
                "subtitle": "ملايين الكتب المجانية",
                "books": ia_books, "style": "archive", "icon": "archive",
            })
    except Exception as e:
        logger.error(f"[AllLibs] Archive error: {e}")
    
    # 3. Project Gutenberg
    try:
        gut_books = fetch_gutenberg_books(search_query, limit=limit_per_source)
        if gut_books:
            sections.append({
                "title": "🟢 Project Gutenberg",
                "subtitle": "كلاسيكيات الأدب العالمي",
                "books": gut_books, "style": "gutenberg", "icon": "book-open-text",
            })
    except Exception as e:
        logger.error(f"[AllLibs] Gutenberg error: {e}")
    
    # 4. OpenLibrary
    try:
        ol_books = fetch_openlib_books(search_query, limit=limit_per_source)
        if ol_books:
            sections.append({
                "title": "🔴 OpenLibrary",
                "subtitle": "مكتبة مفتوحة المصدر",
                "books": ol_books, "style": "openlib", "icon": "books",
            })
    except Exception as e:
        logger.error(f"[AllLibs] OpenLib error: {e}")
    
    # 5. IT Bookstore
    try:
        it_books = fetch_itbook_books(search_query, limit=limit_per_source)
        if it_books:
            sections.append({
                "title": "💙 IT Bookstore",
                "subtitle": "كتب البرمجة والتقنية",
                "books": it_books, "style": "itbook", "icon": "code",
            })
    except Exception as e:
        logger.error(f"[AllLibs] ITBook error: {e}")
    
    return sections
