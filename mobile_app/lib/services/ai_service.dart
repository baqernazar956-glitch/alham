
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import 'auth_service.dart';

/// AI Service — routes all AI calls through the backend server.
class AiService {

  /// Chat with AI about a book or general.
  static Future<Map<String, dynamic>> chat(String message, String? gid,
      {String? bookTitle, String? bookAuthor}) async {
    try {
      final url = gid != null && gid.isNotEmpty
          ? '${AppConfig.serverBaseUrl}/api/ai/book/$gid/chat'
          : '${AppConfig.serverBaseUrl}/api/ai/chat';

      final authHeaders = await AuthService.authHeaders;

      final response = await http.post(
        Uri.parse(url),
        headers: authHeaders,
        body: jsonEncode({
          'message': message,
          'history': [],
        }),
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'success': data['success'] ?? true,
          'response': data['reply'] ?? data['response'] ?? 'No response',
        };
      } else {
        return {
          'success': false,
          'error': 'Server error: ${response.statusCode}',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Connection Error: $e',
      };
    }
  }

  /// Get AI-generated book summary from server.
  static Future<Map<String, dynamic>> getSummary(String gid,
      {String? bookTitle, String? bookAuthor, String? description}) async {
    try {
      // Updated to use the unified API endpoint (GET)
      final url = '${AppConfig.serverBaseUrl}/api/ai/book/$gid/summary';
      final authHeaders = await AuthService.authHeaders;
      
      final response = await http.get(
        Uri.parse(url),
        headers: authHeaders,
      ).timeout(const Duration(seconds: 45));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'success': data['success'] ?? true,
          'summary': data['summary'] ?? data['error'] ?? 'I could not generate a summary.',
        };
      } else {
        return {
          'success': false,
          'error': 'Server error: ${response.statusCode}',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Connection Error: $e',
      };
    }
  }

  /// Get AI-powered book recommendations via server.
  static Future<List<String>> getSmartRecommendations({
    required List<String> interests,
    required List<String> readBooks,
    required List<String> likedCategories,
    List<String> recentSearches = const [],
    String? mood,
    int count = 10,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('${AppConfig.serverBaseUrl}/api/ai-recommendations'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'interests': interests,
          'read_books': readBooks,
          'liked_categories': likedCategories,
          'recent_searches': recentSearches,
          'mood': mood,
          'count': count,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true && data['recommendations'] != null) {
          final List<dynamic> recs = data['recommendations'] as List<dynamic>;
          return recs.map((r) => r['title'] as String).toList();
        }
      }
      return [];
    } catch (e) {
      return [];
    }
  }
}
