import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LocaleProvider with ChangeNotifier {
  Locale _locale = const Locale('ar');

  Locale get locale => _locale;

  LocaleProvider() {
    _loadLocale();
  }

  Future<void> _loadLocale() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final langCode = prefs.getString('language_code') ?? 'ar';
      _locale = Locale(langCode);
      notifyListeners();
    } catch (_) {
      // Fallback if shared_preferences fails
    }
  }

  Future<void> setLocale(Locale locale) async {
    if (!['ar', 'en'].contains(locale.languageCode)) return;
    _locale = locale;
    notifyListeners();
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('language_code', locale.languageCode);
    } catch (_) {}
  }

  Future<void> toggleLocale() async {
    final nextLang = _locale.languageCode == 'ar' ? 'en' : 'ar';
    await setLocale(Locale(nextLang));
  }
}
