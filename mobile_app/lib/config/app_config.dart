/// Application configuration for standalone mode.
/// No external server needed — all APIs called directly.
class AppConfig {
  // ─── Gemini AI ───
  static const String geminiApiKey = 'AIzaSyCJZJ6-23vF-i-3szRWvevs6yon8o2Inz4';
  static const String geminiBaseUrl =
      'https://generativelanguage.googleapis.com/v1beta';
  static const String geminiModel = 'gemini-flash-latest';
  static const String embeddingModel = 'embedding-001';

  // ─── Backend Server ───
  static const String serverBaseUrl = 'http://192.168.100.25:2953';

  // ─── Google Books ───
  static const String googleBooksBaseUrl =
      'https://www.googleapis.com/books/v1/volumes';

  // ─── App Settings ───
  static const String appName = 'Elham';
  static const String appVersion = '2.0.0';
  static const int recommendationCacheTtlMinutes = 30;
  static const int maxRecommendations = 20;
  static const int maxSearchResults = 40;
}
