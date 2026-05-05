
import 'dart:convert';
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

  /// Chat with AI about a book.
  static Future<Map<String, dynamic>> chat(String message, String? gid,
      {String? bookTitle, String? bookAuthor}) async {
    try {
      final context = bookTitle != null
          ? 'You are a smart book assistant. The user is asking about the book "$bookTitle" '
              '${bookAuthor != null ? "by $bookAuthor" : ""}. '
              'Answer in English in a helpful and concise way. If you suggest other books, put the book title between double square brackets like [[Book Title]].'
          : 'You are a smart book assistant. Answer in English in a helpful and concise way. '
              'If you suggest books, always put the book title between double square brackets like [[Book Title]].';

      final response = await _gemini.generateContent([
        Content.text('$context\n\nUser Question: $message'),
      ]);

      return {
        'success': true,
        'response': response.text ?? 'I could not answer that.',
      };
    } catch (e) {
      return {
        'success': false,
        'error': 'AI Connection Error: $e',
      };
    }
  }

  /// Get AI-generated book summary.
  static Future<Map<String, dynamic>> getSummary(String gid,
      {String? bookTitle, String? bookAuthor, String? description}) async {
    try {
      final prompt = '''You are a professional literary critic. Provide a comprehensive summary and critical review for this book:

Title: ${bookTitle ?? 'Unknown'}
Author: ${bookAuthor ?? 'Unknown'}
${description != null && description.isNotEmpty ? 'Description: $description' : ''}

Provide the summary in the following format:
📖 **Book Summary:** (3-4 sentences)
⭐ **Why it is worth reading:** (two points)
🎯 **Suitable for:** (one sentence)
📊 **Rating:** (out of 5)''';

      final response = await _gemini.generateContent([Content.text(prompt)]);

      return {
        'success': true,
        'summary': response.text ?? 'I could not generate a summary.',
      };
    } catch (e) {
      return {
        'success': false,
        'error': 'Error: $e',
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
