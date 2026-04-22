from flask_login import UserMixin
from .extensions import db
from datetime import datetime


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    onboarding_completed = db.Column(db.Boolean, default=False)  # هل اختار اهتماماته؟
    profile_picture = db.Column(db.String(500), nullable=True)  # مسار صورة الملف الشخصي
    bio = db.Column(db.Text, nullable=True)  # نبذة عن المستخدم
    reading_goal = db.Column(db.Integer, default=0)  # هدف القراءة السنوي
    rank = db.Column(db.String(50), default="Novice Reader")  # رتبة المستخدم (مثلاً: قارئ مبتدئ، محترف، إلخ)
    last_active_date = db.Column(db.Date, nullable=True)  # آخر تاريخ نشاط للمستخدم
    current_streak = db.Column(db.Integer, default=0)  # سلسلة النشاط الحالية بالايام

class Book(db.Model):
    __tablename__ = "books"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    author = db.Column(db.String(300))
    description = db.Column(db.Text)
    cover_url = db.Column(db.String(1000))
    
    # تفاصيل إضافية (تمت إضافتها حديثاً)
    publisher = db.Column(db.String(255))
    published_date = db.Column(db.String(50))
    page_count = db.Column(db.Integer)
    isbn = db.Column(db.String(20))
    language = db.Column(db.String(10))
    categories = db.Column(db.Text)  # JSON or comma-separated
    notes = db.Column(db.Text)       # ملاحظات المستخدم الخاصة
    
    # --- التعديل هنا: حذفنا unique=True ---
    google_id = db.Column(db.String(128), nullable=True, index=True)
    
    file_url    = db.Column(db.String(1024))
    owner_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    owner = db.relationship("User", backref=db.backref("books", lazy="dynamic"))

    # --- إضافة شرط مركب: يمنع المستخدم نفسه من إضافة الكتاب مرتين، لكن يسمح لغيره ---
    __table_args__ = (
        db.UniqueConstraint('owner_id', 'google_id', name='uq_owner_book'),
        db.Index('idx_book_owner_created', 'owner_id', 'created_at'),  # ⚡ Phase 3: faster library queries
        db.Index('idx_book_categories', 'categories'),  # ⚡ Phase 3: faster category search
    )

class Genre(db.Model):
    __tablename__ = "genres"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)

class UserGenre(db.Model):
    __tablename__ = "user_genres"
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), primary_key=True)
    genre_id = db.Column(db.Integer, db.ForeignKey("genres.id"), primary_key=True)

class BookGenre(db.Model):
    __tablename__ = "book_genres"
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), primary_key=True)
    genre_id = db.Column(db.Integer, db.ForeignKey("genres.id"), primary_key=True)

class SearchEvent(db.Model):
    __tablename__ = "search_events"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True, nullable=True)
    query = db.Column(db.String(255), nullable=False)
    topics = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class UserPreference(db.Model):
    __tablename__ = "user_preferences"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True, nullable=False)
    topic = db.Column(db.String(80), index=True, nullable=False)
    weight = db.Column(db.Float, default=1.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'topic', name='uq_user_topic'),)

class PublicRating(db.Model):
    __tablename__ = "public_ratings"
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    stars = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class UserRatingCF(db.Model):
    __tablename__ = "user_ratings_cf"

    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    google_id = db.Column(db.String(128), nullable=False, index=True) 
    rating    = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship("User", backref=db.backref("ratings_cf", lazy="dynamic"))

    __table_args__ = (
        db.UniqueConstraint("user_id", "google_id", name="uq_user_google_cf"),
        db.Index("idx_user_rating", "user_id", "rating"),
        db.Index("idx_google_rating", "google_id", "rating"),
    )

    def __repr__(self) -> str:
        return f"<UserRatingCF user={self.user_id} google_id={self.google_id} rating={self.rating}>"

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=True, index=True)
    query = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship('User', backref='search_history')
    book = db.relationship('Book')
    
    __table_args__ = (
        db.Index("idx_user_created", "user_id", "created_at"),
    )

class BookEmbedding(db.Model):
    __tablename__ = "book_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    
    # المفتاح الصحيح
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=False, unique=True, index=True)

    vector = db.Column(db.PickleType)

class UserEmbedding(db.Model):
    __tablename__ = "user_embeddings"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # Stores the numpy embedding (running mean)
    vector = db.Column(db.PickleType)
    
    # Number of interactions used to compute this vector
    interaction_count = db.Column(db.Integer, default=0)
    
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class BookStatus(db.Model):
    __tablename__ = "book_status"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user = db.relationship("User", backref="book_statuses", lazy=True)

    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=False, index=True)
    book = db.relationship("Book", backref="status_entries", lazy=True)

    # one of: favorite / later / finished
    status = db.Column(db.String(20), nullable=False, index=True)

    # 🆕 تتبع تقدم القراءة
    reading_progress = db.Column(db.Integer, default=0)  # نسبة مئوية 0-100
    last_read_at = db.Column(db.DateTime, nullable=True)  # آخر وقت قراءة

    # 🆕 تواريخ تتبع القراءة
    started_at = db.Column(db.DateTime, nullable=True)   # تاريخ بدء القراءة
    finished_at = db.Column(db.DateTime, nullable=True)  # تاريخ إتمام القراءة
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint("user_id", "book_id", name="uq_user_book_status"),
        db.Index("idx_user_status", "user_id", "status"),
    )


class BookReview(db.Model):
    """نموذج مراجعات الكتب - يسمح للمستخدمين بتقييم الكتب وكتابة مراجعات"""
    __tablename__ = "book_reviews"

    id = db.Column(db.Integer, primary_key=True)
    
    # المستخدم الذي كتب المراجعة
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user = db.relationship("User", backref=db.backref("reviews", lazy="dynamic"))
    
    # الكتاب (يمكن أن يكون google_id للكتب الخارجية أو book_id للمحلية)
    google_id = db.Column(db.String(128), nullable=True, index=True)
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=True, index=True)
    book = db.relationship("Book", backref=db.backref("reviews", lazy="dynamic"))
    
    # التقييم والمراجعة
    rating = db.Column(db.Integer, nullable=False)  # 1-5 نجوم
    review_text = db.Column(db.Text, nullable=True)  # نص المراجعة (اختياري)
    
    # التفاعلات (Persistent Likes/Dislikes)
    likes_count = db.Column(db.Integer, default=0)
    dislikes_count = db.Column(db.Integer, default=0)
    
    # التواريخ
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        # مستخدم واحد يمكنه كتابة مراجعة واحدة لكل كتاب
        db.UniqueConstraint("user_id", "google_id", name="uq_user_review_google"),
        db.UniqueConstraint("user_id", "book_id", name="uq_user_review_book"),
        db.Index("idx_google_rating", "google_id", "rating"),
    )

    def __repr__(self):
        return f"<BookReview user={self.user_id} rating={self.rating}>"

class ReviewReaction(db.Model):
    """تتبع إعجابات وعدم إعجاب المستخدمين بالمراجعات"""
    __tablename__ = "review_reactions"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    review_id = db.Column(db.Integer, db.ForeignKey("book_reviews.id"), nullable=False, index=True)
    
    # 'like' or 'dislike'
    reaction_type = db.Column(db.String(20), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    review = db.relationship("BookReview", backref=db.backref("reactions", lazy="dynamic"))
    user = db.relationship("User", backref="review_reactions")
    
    __table_args__ = (
        db.UniqueConstraint("user_id", "review_id", name="uq_user_review_reaction"),
    )

class UserBookView(db.Model):
    """
    نموذج لتتبع مشاهدات الكتب من قبل المستخدمين.
    يساعد في فهم سلوك المستخدم لتحسين التوصيات.
    """
    __tablename__ = "user_book_views"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    
    # يمكن أن يكون الكتاب محلياً (book_id) أو من Google Books (google_id)
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=True, index=True)
    google_id = db.Column(db.String(128), nullable=True, index=True)
    
    # عدد مرات المشاهدة
    view_count = db.Column(db.Integer, default=1)
    
    # توقيت آخر مشاهدة
    last_viewed_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    user = db.relationship("User", backref=db.backref("book_views", lazy="dynamic"))
    book = db.relationship("Book", backref=db.backref("views", lazy="dynamic"))

    __table_args__ = (
        db.Index("idx_user_view_book", "user_id", "book_id"),
        db.Index("idx_user_view_google", "user_id", "google_id"),
        db.Index("idx_view_count_desc", "view_count"),  # ⚡ Phase 3: faster trending queries
        db.Index("idx_view_last_viewed", "last_viewed_at"),  # ⚡ Phase 3: faster recent views
    )

    def __repr__(self):
        return f"<UserBookView user={self.user_id} book={self.book_id or self.google_id} count={self.view_count}>"


class BookQuote(db.Model):
    """نموذج اقتباسات الكتب - يسمح للمستخدمين بحفظ اقتباسات من الكتب"""
    __tablename__ = "book_quotes"

    id = db.Column(db.Integer, primary_key=True)
    
    # المستخدم الذي أضاف الاقتباس
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    user = db.relationship("User", backref=db.backref("quotes", lazy="dynamic"))
    
    # الكتاب (يمكن أن يكون book_id للمحلية أو google_id للخارجية)
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=True, index=True)
    google_id = db.Column(db.String(128), nullable=True, index=True)
    book = db.relationship("Book", backref=db.backref("quotes", lazy="dynamic"))
    
    # نص الاقتباس ورقم الصفحة
    quote_text = db.Column(db.Text, nullable=False)
    page_number = db.Column(db.Integer, nullable=True)
    
    # التواريخ
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.Index("idx_user_book_quote", "user_id", "book_id"),
        db.Index("idx_user_google_quote", "user_id", "google_id"),
    )

    def __repr__(self):
        return f"<BookQuote user={self.user_id} book={self.book_id or self.google_id}>"


class UserEvent(db.Model):
    """
    نموذج تتبع أحداث المستخدم بدقة أعلى من UserBookView.
    يدعم أنواع أحداث متعددة مع بيانات سلوكية إضافية.
    """
    __tablename__ = "user_events"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # نوع الحدث: view / click / read / abandon / share / rate
    event_type = db.Column(db.String(20), nullable=False, index=True)

    # معرف الكتاب (Google Books ID)
    book_google_id = db.Column(db.String(128), nullable=True, index=True)

    # معرف الجلسة (لربط أحداث متعددة ضمن نفس الزيارة)
    session_id = db.Column(db.String(64), nullable=True, index=True)

    # مدة التفاعل بالثواني (مثلاً: مدة قراءة صفحة التفاصيل)
    duration_seconds = db.Column(db.Integer, nullable=True)

    # عمق التمرير (0.0 = أعلى الصفحة, 1.0 = أسفل الصفحة)
    scroll_depth = db.Column(db.Float, nullable=True)

    # بيانات إضافية بصيغة JSON نصية
    metadata_json = db.Column(db.Text, nullable=True)

    # وقت الحدث
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # العلاقات
    user = db.relationship("User", backref=db.backref("events", lazy="dynamic"))

    def __repr__(self):
        return f"<UserEvent user={self.user_id} type={self.event_type} book={self.book_google_id}>"

class Experiment(db.Model):
    """
    نموذج لاختبارات A/B (Experiments).
    يحتوي على إعدادات التجربة مثل نسبة التقسيم وحالتها.
    """
    __tablename__ = "experiments"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    
    traffic_split = db.Column(db.Float, default=0.5) # e.g., 0.2 means 20% in treatment
    status = db.Column(db.String(20), default='active') # 'active', 'paused', 'completed'
    
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=True)
    
    winning_variant = db.Column(db.String(50), nullable=True)

class ExperimentAssignment(db.Model):
    """
    تعيينات المستخدمين لتجارب A/B بشكل ثابت (Deterministic).
    """
    __tablename__ = "experiment_assignments"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True)
    
    variant = db.Column(db.String(50), nullable=False) # 'control' or 'treatment'
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint("user_id", "experiment_id", name="uq_user_experiment"),
    )

class ExperimentMetric(db.Model):
    """
    سجل المقاييس (Metrics) المرتبطة باختبارات A/B.
    """
    __tablename__ = "experiment_metrics"
    
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=False, index=True)
    variant = db.Column(db.String(50), nullable=False, index=True)
    
    metric_name = db.Column(db.String(100), nullable=False, index=True)
    metric_value = db.Column(db.Float, nullable=False)
    
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
