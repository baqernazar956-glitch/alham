import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/user.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Server-based authentication via JWT — shares database with Flask web app.
class AuthService {
  static const String _tokenKey = 'jwt_token';
  static const String _userIdKey = 'current_user_id';

  // ─── Token Management ───

  /// Get the stored JWT token.
  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  /// Save JWT token and user ID.
  static Future<void> _saveSession(String token, int userId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
    await prefs.setInt(_userIdKey, userId);
  }

  /// Remove saved session.
  static Future<void> removeSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userIdKey);
  }

  /// Get saved user ID (for quick checks without network).
  static Future<int?> getCurrentUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_userIdKey);
  }

  /// Build auth headers with JWT token.
  static Future<Map<String, String>> get authHeaders async {
    final token = await getToken();
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  // ─── Login ───

  static Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/auth/login';
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'email': email.trim().toLowerCase(),
          'password': password,
        }),
      ).timeout(const Duration(seconds: 15));

      final data = jsonDecode(response.body);

      if (response.statusCode == 200 && data['success'] == true) {
        final token = data['token'] as String;
        final userData = data['user'] as Map<String, dynamic>;
        final userId = userData['id'] as int;

        await _saveSession(token, userId);

        return {
          'success': true,
          'user': User.fromJson(userData),
        };
      } else {
        return {
          'success': false,
          'error': data['error'] ?? 'Login failed',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Could not connect to server: $e',
      };
    }
  }

  // ─── Register ───

  static Future<Map<String, dynamic>> register(
      String name, String email, String password) async {
    try {
      final url = '${AppConfig.serverBaseUrl}/api/auth/register';
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'name': name.trim(),
          'email': email.trim().toLowerCase(),
          'password': password,
        }),
      ).timeout(const Duration(seconds: 15));

      final data = jsonDecode(response.body);

      if ((response.statusCode == 200 || response.statusCode == 201) &&
          data['success'] == true) {
        final token = data['token'] as String;
        final userData = data['user'] as Map<String, dynamic>;
        final userId = userData['id'] as int;

        await _saveSession(token, userId);

        return {
          'success': true,
          'user': User.fromJson(userData),
        };
      } else {
        return {
          'success': false,
          'error': data['error'] ?? 'Registration failed',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Could not connect to server: $e',
      };
    }
  }

  // ─── Get Current User ───

  static Future<User?> getCurrentUser() async {
    try {
      final token = await getToken();
      if (token == null) return null;

      final url = '${AppConfig.serverBaseUrl}/api/auth/me';
      final response = await http.get(
        Uri.parse(url),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true && data['user'] != null) {
          return User.fromJson(data['user']);
        }
      }

      // Token expired or invalid — clear session
      if (response.statusCode == 401 || response.statusCode == 422) {
        await removeSession();
      }
      return null;
    } catch (e) {
      // Network error — try to return cached user ID at least
      return null;
    }
  }

  // ─── Logout ───

  static Future<bool> logout() async {
    try {
      final headers = await authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/auth/logout';
      await http.post(Uri.parse(url), headers: headers)
          .timeout(const Duration(seconds: 5));
    } catch (_) {
      // Even if server call fails, clear local session
    }
    await removeSession();
    return true;
  }

  // ─── Onboarding ───

  static Future<Map<String, dynamic>> completeOnboarding(
      List<String> interests) async {
    try {
      final headers = await authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/auth/onboarding';
      final response = await http.post(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode({'interests': interests}),
      ).timeout(const Duration(seconds: 10));

      final data = jsonDecode(response.body);
      return {
        'success': data['success'] ?? false,
        'error': data['error'],
      };
    } catch (e) {
      return {'success': false, 'error': 'Could not connect to server: $e'};
    }
  }

  // ─── Update Profile ───

  static Future<Map<String, dynamic>> updateProfile({
    String? name,
    String? bio,
    int? readingGoal,
  }) async {
    try {
      final headers = await authHeaders;
      final url = '${AppConfig.serverBaseUrl}/api/user/profile';

      final body = <String, dynamic>{};
      if (name != null) body['name'] = name;
      if (bio != null) body['bio'] = bio;
      if (readingGoal != null) body['reading_goal'] = readingGoal;

      final response = await http.put(
        Uri.parse(url),
        headers: headers,
        body: jsonEncode(body),
      ).timeout(const Duration(seconds: 10));

      final data = jsonDecode(response.body);
      return {
        'success': data['success'] ?? false,
        'error': data['error'],
      };
    } catch (e) {
      return {'success': false, 'error': '$e'};
    }
  }
}
