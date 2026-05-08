
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:google_generative_ai/google_generative_ai.dart';
import '../config/app_config.dart';

/// AI Service — calls Gemini API directly from the mobile app.
class AiService {
  static GenerativeModel? _model;

  static GenerativeModel get _gemini {
    _model ??= GenerativeModel(
      model: AppConfig.geminiModel,
      apiKey: AppConfig.geminiApiKey,
    );
    return _model!;
  }

  /// Chat with AI about a book or general.
  static Future<Map<String, dynamic>> chat(String message, String? gid,
      {String? bookTitle, String? bookAuthor}) async {
    try {
      final url = gid != null && gid.isNotEmpty
          ? '${AppConfig.serverBaseUrl}/books/$gid/chat'
          : '${AppConfig.serverBaseUrl}/api/ai-chat';

      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': message,
          'history': [],
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'success': data['success'] ?? true,
          'response': data['reply'] ?? 'No response',
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

  /// Get AI-generated book summary.
  /// Get AI-generated book summary from server.
  static Future<Map<String, dynamic>> getSummary(String gid,
      {String? bookTitle, String? bookAuthor, String? description}) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/books/$gid/ai-summary';
      
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
      );

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

  /// Get AI-powered book recommendations based on user profile.
  /// Uses Deep Learning analysis of user behavior, interests, and search patterns.
  static Future<List<String>> getSmartRecommendations({
    required List<String> interests,
    required List<String> readBooks,
    required List<String> likedCategories,
    List<String> recentSearches = const [],
    String? mood,
    int count = 10,
  }) async {
    try {
      final searchSection = recentSearches.isNotEmpty
          ? '- Recently searched for: ${recentSearches.take(5).join(', ')} ⚡ (High priority — suggest books directly related to recent searches)'
          : '';
      
      final prompt = '''You are a smart book recommendation system that uses Deep Learning and behavioral pattern analysis.

User Profile:
- Interests: ${interests.join(', ')}
- Books Read: ${readBooks.take(10).join(', ')}
- Favorite Categories: ${likedCategories.join(', ')}
$searchSection
${mood != null ? '- Current Mood: $mood' : ''}

Important Instructions:
1. If the user searched for something recently, most recommendations should be directly related to that.
2. Suggest real, well-known books that can be found in Google Books.
3. Diversify recommendations between recent searches and general interests.

Suggest $count suitable books.

The response must be a JSON array only in this format (no extra text):
[{"title": "Book Title", "author": "Author", "reason": "One sentence reason for recommendation"}]''';

      final response = await _gemini.generateContent([Content.text(prompt)]);
      final text = response.text ?? '[]';

      // Extract JSON from response
      final jsonMatch = RegExp(r'\[[\s\S]*\]').firstMatch(text);
      if (jsonMatch != null) {
        final List<dynamic> books = jsonDecode(jsonMatch.group(0)!);
        return books.map((b) => b['title'] as String).toList();
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  /// Generate embeddings using Gemini Embedding API.
  static Future<List<double>?> generateEmbedding(String text) async {
    try {
      final model = GenerativeModel(
        model: 'text-embedding-004',
        apiKey: AppConfig.geminiApiKey,
      );

      final result = await model.embedContent(Content.text(text));
      return result.embedding.values;
    } catch (e) {
      return null;
    }
  }
}
