import 'package:flutter/material.dart';
import '../models/user.dart';
import '../services/auth_service.dart';

class AuthProvider with ChangeNotifier {
  User? _currentUser;
  bool _isLoading = true;
  String? _error;

  User? get currentUser => _currentUser;
  bool get isAuthenticated => _currentUser != null;
  bool get isLoading => _isLoading;
  String? get error => _error;

  AuthProvider() {
    _initAuth();
  }

  Future<void> _initAuth() async {
    final userId = await AuthService.getCurrentUserId();
    if (userId != null) {
      _currentUser = await AuthService.getCurrentUser();
    }
    _isLoading = false;
    notifyListeners();
  }

  Future<bool> login(String email, String password) async {
    _setLoading(true);
    final result = await AuthService.login(email, password);
    if (result['success']) {
      _currentUser = result['user'];
      _error = null;
      _setLoading(false);
      return true;
    } else {
      _error = result['error'];
      _setLoading(false);
      return false;
    }
  }

  Future<bool> register(String name, String email, String password) async {
    _setLoading(true);
    final result = await AuthService.register(name, email, password);
    if (result['success']) {
      _currentUser = result['user'];
      _error = null;
      _setLoading(false);
      return true;
    } else {
      _error = result['error'];
      _setLoading(false);
      return false;
    }
  }

  Future<void> completeOnboarding(List<String> interests) async {
    final result = await AuthService.completeOnboarding(interests);
    if (result['success'] && _currentUser != null) {
      _currentUser = User(
        id: _currentUser!.id,
        name: _currentUser!.name,
        email: _currentUser!.email,
        onboardingCompleted: true,
        bio: _currentUser!.bio,
        readingGoal: _currentUser!.readingGoal,
        interests: interests,
      );
      notifyListeners();
    }
  }

  Future<void> logout() async {
    await AuthService.logout();
    _currentUser = null;
    notifyListeners();
  }

  void _setLoading(bool value) {
    _isLoading = value;
    notifyListeners();
  }
}
