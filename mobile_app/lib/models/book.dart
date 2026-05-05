class Book {
  final int? id;
  final String? gid;
  final String title;
  final String author;
  final List<String> authors;
  final String description;
  final String coverUrl;
  final String publisher;
  final String publishedDate;
  final int pageCount;
  final String language;
  final double averageRating;
  final int ratingsCount;
  final List<String> categories;
  final String source;
  final String? algorithmTag;
  final String? previewLink;
  final String? infoLink;
  final bool canRead;
  final bool epubAvailable;
  final bool pdfAvailable;
  final String? status; // favorite, reading, to_read, read
  final int readingProgress;
  final String? recommendationReason;

  Book({
    this.id,
    this.gid,
    required this.title,
    required this.author,
    this.authors = const [],
    this.description = '',
    this.coverUrl = '',
    this.publisher = '',
    this.publishedDate = '',
    this.pageCount = 0,
    this.language = '',
    this.averageRating = 0.0,
    this.ratingsCount = 0,
    this.categories = const [],
    this.source = 'google',
    this.algorithmTag,
    this.previewLink,
    this.infoLink,
    this.canRead = false,
    this.epubAvailable = false,
    this.pdfAvailable = false,
    this.status,
    this.readingProgress = 0,
    this.recommendationReason,
  });

  /// Unique identifier — prefers gid, falls back to int id as string
  String get uniqueId {
    if (gid != null && gid!.isNotEmpty) {
      return gid!;
    }
    if (id != null) {
      return id!.toString();
    }
    return '';
  }

  factory Book.fromJson(Map<String, dynamic> json) {
    List<String> authorsList = [];
    if (json['authors'] != null) {
      authorsList = List<String>.from(json['authors']);
    }

    List<String> categoriesList = [];
    if (json['categories'] != null) {
      if (json['categories'] is String) {
        categoriesList = [json['categories']];
      } else {
        categoriesList = List<String>.from(json['categories']);
      }
    }

    return Book(
      id: json['id'] is int ? json['id'] : null,
      gid: json['gid'] ??
          json['google_id'] ??
          (json['id'] is String ? json['id'] : null),
      title: json['title'] ?? 'Untitled',
      author: json['author'] ??
          (authorsList.isNotEmpty ? authorsList.join(', ') : 'Unknown'),
      authors: authorsList,
      description: json['description'] ?? json['desc'] ?? '',
      coverUrl: (json['cover_url'] ?? json['cover'] ?? '').toString().replaceAll('http://', 'https://'),
      publisher: json['publisher'] ?? '',
      publishedDate: json['published_date'] ?? json['publishedDate'] ?? '',
      pageCount: json['page_count'] ?? json['pageCount'] ?? 0,
      language: json['language'] ?? '',
      averageRating:
          (json['average_rating'] ?? json['rating'] ?? 0.0).toDouble(),
      ratingsCount: json['ratings_count'] ?? json['ratingsCount'] ?? 0,
      categories: categoriesList,
      source: json['source'] ?? 'google',
      algorithmTag: json['algorithm_tag'],
      previewLink: json['preview_link'],
      infoLink: json['info_link'],
      canRead: json['can_read'] ?? false,
      epubAvailable: json['epub_available'] ?? false,
      pdfAvailable: json['pdf_available'] ?? false,
      status: json['status'],
      readingProgress: json['reading_progress'] ?? 0,
      recommendationReason: json['reason'] ?? json['recommendation_reason'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'gid': gid,
      'title': title,
      'author': author,
      'authors': authors,
      'description': description,
      'cover_url': coverUrl,
      'publisher': publisher,
      'published_date': publishedDate,
      'page_count': pageCount,
      'language': language,
      'average_rating': averageRating,
      'ratings_count': ratingsCount,
      'categories': categories,
      'source': source,
      'algorithm_tag': algorithmTag,
      'reason': recommendationReason,
    };
  }
}
