import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../config/app_config.dart';
import '../models/book.dart';
import 'auth_service.dart';

/// User service — all data synced with the Flask server.
class UserService {
  static const String _searchHistoryKey = 'search_history';

  // ─── Search History ───
  static Future<List<String>> getSearchHistory() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_searchHistoryKey) ?? [];
  }

  static Future<void> saveSearchQuery(String query) async {
    // 1. Save locally
    final prefs = await SharedPreferences.getInstance();
    List<String> history = prefs.getStringList(_searchHistoryKey) ?? [];
    
    if (history.contains(query)) {
      history.remove(query);
    }
    history.insert(0, query);
    
    if (history.length > 10) {
      history = history.sublist(0, 10);
    }
    await prefs.setStringList(_searchHistoryKey, history);

    // 2. Send search to server to update recommendations
    await logSearchToServer(query);
  }

  /// Send search operation to server to save in SearchHistory and update recommendations
  static Future<void> logSearchToServer(String query) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/log-search';
      
      await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'query': query}),
      ).timeout(const Duration(seconds: 5));
    } catch (e) {
      print('Error logging search to server: $e');
    }
  }

  /// Fetch search history from server (instead of local only)
  static Future<List<String>> getServerSearchHistory({int limit = 10}) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/search-history?limit=$limit';
      
      final response = await http.get(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          return List<String>.from(data['queries'] ?? []);
        }
      }
    } catch (e) {
      print('Error fetching server search history: $e');
    }
    // Fallback to local
    return getSearchHistory();
  }


  // ─── Library ───
  
  static Future<List<Map<String, dynamic>>> getLibrary({String? status}) async {
    try {
      final headers = await AuthService.authHeaders;
      var url = '${AppConfig.serverBaseUrl}/api/user/library';
      if (status != null) {
        url += '?status=$status';
      }

      final response = await http.get(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final List<dynamic> books = data['books'] ?? [];
        return books.cast<Map<String, dynamic>>();
      }
      return [];
    } catch (e) {
      print('Error fetching library: $e');
      return [];
    }
  }

  static Future<bool> addToLibrary(String gid, String status, {Book? book}) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/library/$gid';
      
      final response = await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'status': status}),
      ).timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print('Error adding to library: $e');
      return false;
    }
  }

  static Future<bool> removeFromLibrary(String gid) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/library/$gid';
      
      final response = await http.delete(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print('Error removing from library: $e');
      return false;
    }
  }

  static Future<Map<String, dynamic>?> getBookStatus(String gid) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/library/$gid/status';
      
      final response = await http.get(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      print('Error getting book status: $e');
      return null;
    }
  }

  // ─── Rate/Review ───
  
  static Future<bool> rateBook(String gid, double rating, String review) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/rate/$gid';
      
      final response = await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({
          'rating': rating.toInt(),
          'review': review,
        }),
      ).timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print('Error rating book: $e');
      return false;
    }
  }

  // ─── Stats ───
  
  static Future<Map<String, dynamic>> getStats() async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/stats';
      
      final response = await http.get(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['stats'] ?? {};
      }
      return {};
    } catch (e) {
      print('Error fetching stats: $e');
      return {};
    }
  }

  // ─── Notes ───
  
  static Future<String> getNote(String gid) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/notes/$gid';
      
      final response = await http.get(
        Uri.parse(url),
        headers: headers,
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['note_text'] ?? '';
      }
      return '';
    } catch (e) {
      print('Error fetching note: $e');
      return '';
    }
  }

  static Future<bool> saveNote(String gid, String noteText) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/notes/$gid';
      
      final response = await http.put(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'note_text': noteText}),
      ).timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print('Error saving note: $e');
      return false;
    }
  }

  // ─── Log View ───
  
  static Future<void> logBookView(String gid) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/book-view';
      
      await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'google_id': gid}),
      ).timeout(const Duration(seconds: 5));
    } catch (e) {
      print('Error logging book view: $e');
    }
  }

  // ─── Profile Update ───
  
  static Future<Map<String, dynamic>> updateProfile({
    String? name,
    String? bio,
    int? readingGoal,
  }) async {
    return await AuthService.updateProfile(
      name: name,
      bio: bio,
      readingGoal: readingGoal,
    );
  }

  // ─── Update Reading Progress ───
  
  static Future<bool> updateReadingProgress(String gid, int progress) async {
    try {
      final headers = await AuthService.authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/library/$gid/progress';
      
      final response = await http.put(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'progress': progress}),
      ).timeout(const Duration(seconds: 10));

      return response.statusCode == 200;
    } catch (e) {
      print('Error updating reading progress: $e');
      return false;
    }
  }
}
