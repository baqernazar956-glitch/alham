/// Application configuration.
/// All AI calls are routed through the backend server.
class AppConfig {
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
