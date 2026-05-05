import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/book.dart';
import '../models/review.dart';
import 'auth_service.dart';
import 'user_service.dart';

/// Books service — pure API client, all logic runs on the server.
class BooksService {


  // ─── Helper: retry with backoff ───
  static Future<http.Response> _getWithRetry(String url, {int maxRetries = 3, Duration timeout = const Duration(seconds: 30), Map<String, String>? headers}) async {
    for (int attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        final response = await http.get(Uri.parse(url), headers: headers).timeout(timeout);
        if (response.statusCode == 200) return response;
        // Server error, retry
        if (attempt < maxRetries) {
          await Future.delayed(Duration(milliseconds: 500 * attempt));
          continue;
        }
        return response;
      } catch (e) {
        print('Attempt $attempt/$maxRetries failed for $url: $e');
        if (attempt < maxRetries) {
          await Future.delayed(Duration(milliseconds: 500 * attempt));
          continue;
        }
        rethrow;
      }
    }
    throw Exception('All $maxRetries attempts failed');
  }

  // ─── Search (5 libraries in parallel via backend) ───
  static Future<List<Book>> search(String query, {int page = 1, bool logSearch = true}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/search?q=${Uri.encodeComponent(query)}&page=$page&per_page=40&log=$logSearch';
      // Send auth headers so backend can log search to SearchHistory (if log=true)
      final authHeaders = await AuthService.authHeaders;
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 30), headers: authHeaders);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final items = data['books'] as List<dynamic>? ?? [];
        return items.map((item) => _parseServerBook(item as Map<String, dynamic>)).toList();
      }
      return [];
    } catch (e) {
      print('Search error: $e');
      return [];
    }
  }

  // ─── Categories ───
  static Future<List<dynamic>> getCategories() async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/categories';
      final response = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final cats = data['categories'] as List<dynamic>? ?? [];
        // Add "All" at the beginning
        return [
          {'id': 'all', 'name': 'All Books'},
          ...cats.map((c) => {
            'id': c['id'],
            'name': c['name_en'] ?? c['name'],
            'emoji': null,
          })
        ];
      }
    } catch (e) {
      print('Categories fetch error: $e');
    }
    
    // Fallback
    return [
      {'id': 'all', 'name': 'All Books'},
      {'id': 'fiction', 'name': 'Fiction'},
      {'id': 'science', 'name': 'Science'},
      {'id': 'history', 'name': 'History'},
      {'id': 'philosophy', 'name': 'Philosophy'},
      {'id': 'psychology', 'name': 'Psychology'},
      {'id': 'business', 'name': 'Business'},
      {'id': 'programming', 'name': 'Programming'},
      {'id': 'ai', 'name': 'Artificial Intelligence'},
    ];
  }

  static String _getEmojiForCategory(String id) {
    switch (id.toLowerCase()) {
      case 'programming': return '💻';
      case 'ai': return '🤖';
      case 'fiction': return '📖';
      case 'science': return '🔬';
      case 'history': return '🏛️';
      case 'philosophy': return '🤔';
      case 'psychology': return '🧠';
      case 'technology': return '📱';
      case 'business': return '💼';
      case 'self-help': return '🌱';
      default: return '📚';
    }
  }

  // ─── Books by Category (5 libraries in parallel via backend) ───
  static Future<List<Book>> getBooksByCategory(String categoryId,
      {int page = 1}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/category/$categoryId?page=$page&per_page=40';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final items = data['books'] as List<dynamic>? ?? [];
        return items.map((item) => _parseServerBook(item as Map<String, dynamic>)).toList();
      }
      return [];
    } catch (e) {
      print('Category fetch error: $e');
      return [];
    }
  }

  // ─── Trending (via server) ───
  static Future<List<Book>> getTrending({int limit = 12}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/trending?limit=$limit';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final items = data['books'] as List<dynamic>? ?? [];
        return items.map((item) => _parseServerBook(item)).toList();
      }
      
      // Fallback: use server search instead of direct Google API
      return _getTrendingFallback(limit: limit);
    } catch (e) {
      print('Error fetching trending: $e');
      return _getTrendingFallback(limit: limit);
    }
  }

  static Future<List<Book>> _getTrendingFallback({int limit = 12}) async {
    try {
      // Use server's multi-library search as fallback
      return await search('bestseller', page: 1);
    } catch (_) { return []; }
  }

  // ─── Personalized Recommendations (Server Unified Neural Pipeline) ───
  // ALL recommendation logic runs on the server — Flutter is just the UI
  static Future<List<Map<String, dynamic>>> getPersonalizedRecommendations() async {
    try {
      final token = await AuthService.getToken();
      if (token == null) return [];

      final url = '${AppConfig.serverBaseUrl}/api/books/recommendations';
      final response = await _getWithRetry(
        url,
        timeout: const Duration(seconds: 45),
        headers: {'Authorization': 'Bearer $token', 'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          final sections = data['sections'] as List<dynamic>? ?? [];
          final result = <Map<String, dynamic>>[];
          
          for (final section in sections) {
             final secMap = section as Map<String, dynamic>;
             final rawBooks = secMap['books'] as List<dynamic>? ?? [];
             
             final books = rawBooks.map((b) => _parseServerBook(b as Map<String, dynamic>)).toList();
             
             result.add({
               'title': secMap['title'] ?? '',
               'subtitle': secMap['subtitle'] ?? '',
               'icon': secMap['icon'] ?? '📚',
               'books': books.map((b) => b.toJson()).toList(),
             });
          }
          return result;
        }
      }
      return [];
    } catch (e) {
      print('Error fetching personalized recommendations: $e');
      return [];
    }
  }

  // ─── Top Rated (Trending Now section) ───
  static Future<List<Book>> getTopRated({int limit = 15}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/top-rated?limit=$limit';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final items = data['books'] as List<dynamic>? ?? [];
        return items.map((item) => _parseServerBook(item)).toList();
      }
      return [];
    } catch (e) {
      print('Error fetching top rated: $e');
      return [];
    }
  }

  // ─── Curated Collections (via server's multi-library search) ───
  static Future<List<Map<String, dynamic>>> getCuratedCollections() async {
    final collections = [
      {'title': 'Epic Fantasy', 'query': 'epic fantasy', 'color': 0xFFb8a9e8},
      {'title': 'Astrophysics', 'query': 'astrophysics', 'color': 0xFFc8e6c9},
      {'title': 'Philosophy of Life', 'query': 'philosophy', 'color': 0xFFffe0b2},
      {'title': 'Future of AI', 'query': 'artificial intelligence', 'color': 0xFFb3e5fc},
      {'title': 'Human Psychology', 'query': 'psychology', 'color': 0xFFf8bbd0},
      {'title': 'Arabic Literature', 'query': 'arabic literature', 'color': 0xFFd1c4e9},
      {'title': 'Ancient History', 'query': 'ancient history', 'color': 0xFFc5cae9},
      {'title': 'Entrepreneurship', 'query': 'entrepreneurship', 'color': 0xFFdcedc8},
      {'title': 'Crime Thriller', 'query': 'crime thriller', 'color': 0xFFa5d6a7},
      {'title': 'Personal Growth', 'query': 'personal growth', 'color': 0xFFffcc80},
      {'title': 'Classic Literature', 'query': 'classic literature', 'color': 0xFF90caf9},
      {'title': 'Space Exploration', 'query': 'space exploration', 'color': 0xFFCE93D8},
    ];

    final results = <Map<String, dynamic>>[];

    for (final col in collections) {
      try {
        // Use server's multi-library search instead of direct Google API
        final url = '${AppConfig.serverBaseUrl}/api/books/search?q=${Uri.encodeComponent(col['query'] as String)}&per_page=6';
        final response = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 15));

        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          final items = data['books'] as List<dynamic>? ?? [];
          final books = <Book>[];
          final covers = <String>[];

          for (final item in items) {
            final book = _parseServerBook(item as Map<String, dynamic>);
            if (book.coverUrl.isNotEmpty) {
              books.add(book);
              covers.add(book.coverUrl);
            }
            if (covers.length >= 3) break;
          }

          if (covers.isNotEmpty) {
            results.add({
              'title': col['title'],
              'query': col['query'],
              'color': col['color'],
              'covers': covers,
              'books': books,
              'count': items.length,
            });
          }
        }
      } catch (_) {}

      if (results.length >= 6) break;
    }

    return results;
  }

  // ─── Mood Recommendations (via server) ───
  static Future<List<Book>> getMoodRecommendations(String mood,
      {int limit = 12}) async {
    try {
      // Use server's mood-recommendations endpoint
      final url = '${AppConfig.serverBaseUrl}/api/books/mood-recommendations?mood=$mood&limit=$limit';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          final items = data['books'] as List<dynamic>? ?? [];
          return items.map((item) => _parseServerBook(item as Map<String, dynamic>)).toList();
        }
      }
      // Fallback: use server search with mood query
      return await search(mood, page: 1);
    } catch (e) {
      print('Error fetching mood recommendations: $e');
      return [];
    }
  }

  // ─── Recommend by Book (via server) ───
  static Future<List<Book>> getRecommendByBook(String title,
      {int limit = 24}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/recommend-by-book?title=${Uri.encodeComponent(title)}&limit=$limit';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          final items = data['books'] as List<dynamic>? ?? [];
          return items.map((item) => _parseServerBook(item as Map<String, dynamic>)).toList();
        }
      }
      // Fallback: search via server
      return await search(title, page: 1);
    } catch (e) {
      return [];
    }
  }

  // ─── Book Detail (via server) ───
  static Future<Book?> getBookDetail(String gid) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/$gid';
      final response = await _getWithRetry(url, timeout: const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true && data['book'] != null) {
          return _parseServerBook(data['book'] as Map<String, dynamic>);
        }
      }
      return null;
    } catch (e) {
      return null;
    }
  }


  // ─── Reviews (Server-based) ───
  static Future<List<Review>> getBookReviews(String gid) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/books/$gid/reviews';
      final response = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final List<dynamic> reviewsData = data['reviews'] ?? [];
        return reviewsData.map((r) => Review(
          id: r['id'] as int,
          userId: r['user_id'] as int,
          userName: r['user_name'] as String? ?? 'User',
          googleId: r['google_id'] as String?,
          rating: r['rating'] as int,
          reviewText: r['review_text'] as String? ?? '',
          createdAt: DateTime.parse(r['created_at'] as String? ?? DateTime.now().toIso8601String()),
        )).toList();
      }
      return [];
    } catch (e) {
      print('Error fetching reviews: $e');
      return [];
    }
  }

  static Future<bool> submitReview(
      String gid, int rating, String reviewText) async {
    // Reuse UserService.rateBook which is already server-synced
    return await UserService.rateBook(gid, rating.toDouble(), reviewText);
  }

  static Future<Map<String, dynamic>> reactToReview(
      int reviewId, String type) async {
    return {'success': true}; // simplified for local
  }

  // ─── Mood Meta ───
  static Future<Map<String, dynamic>> getMoodMeta() async {
    return {
      'happy': {'emoji': '😊', 'label': 'Happy', 'color': '#FFC107'},
      'sad': {'emoji': '😔', 'label': 'Sad', 'color': '#9C27B0'},
      'adventurous': {'emoji': '🗺️', 'label': 'Adventurous', 'color': '#00BCD4'},
      'calm': {'emoji': '🧘', 'label': 'Calm', 'color': '#4CAF50'},
      'curious': {'emoji': '🧐', 'label': 'Curious', 'color': '#E91E63'},
      'romantic': {'emoji': '❤️', 'label': 'Romantic', 'color': '#F06292'},
    };
  }

  // ─── Event Logging (Server-based) ───
  static Future<void> logEvent(String eventType, String bookGid,
      {Map<String, dynamic>? metadata}) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/books/event';

      await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({
          'event_type': eventType,
          'book_google_id': bookGid,
          'metadata': metadata ?? {},
        }),
      ).timeout(const Duration(seconds: 5));
    } catch (e) {
      print('Error logging event to server: $e');
    }
  }

  // ─── Parse Server Book response ───
  static Book _parseServerBook(Map<String, dynamic> item) {
    String cover = item['cover_url'] ?? item['cover'] ?? '';
    if (cover.isNotEmpty && cover.startsWith('http://')) {
      cover = cover.replaceFirst('http://', 'https://');
    }

    return Book(
      gid: item['id']?.toString() ?? item['gid']?.toString() ?? '',
      title: item['title'] ?? 'Untitled',
      author: item['author'] ?? 'Unknown',
      authors: [item['author'] ?? 'Unknown'],
      description: item['desc'] ?? item['description'] ?? '',
      coverUrl: cover,
      publisher: item['publisher'] ?? '',
      publishedDate: item['publishedDate'] ?? '',
      pageCount: item['pageCount'] is int ? item['pageCount'] : 0,
      language: item['language'] ?? '',
      averageRating: (item['average_rating'] ?? item['rating'] ?? 0.0).toDouble(),
      ratingsCount: (item['ratings_count'] ?? 0).toInt(),
      categories: item['categories'] is List ? List<String>.from(item['categories']) : (item['categories'] is String && item['categories'].toString().isNotEmpty ? [item['categories'].toString()] : []),
      source: item['source'] ?? 'server',
      previewLink: item['preview_link'],
      infoLink: item['info_link'],
      recommendationReason: item['reason'],
      algorithmTag: item['algorithm_tag'],
    );
  }

  // ─── Parse Google Books API response ───
  static Book _parseGoogleBook(Map<String, dynamic> item) {
    final vi = item['volumeInfo'] as Map<String, dynamic>? ?? {};
    final gid = item['id'] as String? ?? '';

    // Get cover URL - try to get higher resolution if available
    String coverUrl = '';
    final imageLinks = vi['imageLinks'] as Map<String, dynamic>?;
    if (imageLinks != null) {
      coverUrl = (imageLinks['medium'] as String?) ??
          (imageLinks['small'] as String?) ??
          (imageLinks['thumbnail'] as String?) ??
          (imageLinks['smallThumbnail'] as String?) ??
          '';
      
      // Fix HTTP to HTTPS and ensure high quality parameters
      if (coverUrl.isNotEmpty) {
        if (coverUrl.contains('http://')) {
          coverUrl = coverUrl.replaceAll('http://', 'https://');
        }
        // Remove edge=curl which sometimes causes issues
        coverUrl = coverUrl.replaceAll('&edge=curl', '');
      }
    }

    // Get categories
    List<String> categories = [];
    if (vi['categories'] != null) {
      categories = List<String>.from(vi['categories']);
    }

    // Get authors
    List<String> authors = [];
    if (vi['authors'] != null) {
      authors = List<String>.from(vi['authors']);
    }

    return Book(
      gid: gid,
      title: vi['title'] as String? ?? 'Untitled',
      author: authors.isNotEmpty ? authors.join(', ') : 'Unknown',
      authors: authors,
      description: vi['description'] as String? ?? '',
      coverUrl: coverUrl,
      publisher: vi['publisher'] as String? ?? '',
      publishedDate: vi['publishedDate'] as String? ?? '',
      pageCount: vi['pageCount'] as int? ?? 0,
      language: vi['language'] as String? ?? '',
      averageRating: (vi['averageRating'] ?? 0.0).toDouble(),
      ratingsCount: vi['ratingsCount'] as int? ?? 0,
      categories: categories,
      source: 'google',
      previewLink: vi['previewLink'] as String?,
      infoLink: vi['infoLink'] as String?,
    );
  }
}
