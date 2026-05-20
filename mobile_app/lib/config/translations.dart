import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/locale_provider.dart';

class AppTranslations {
  static const Map<String, Map<String, String>> _values = {
    'en': {
      'discover': 'Discover',
      'public_library': 'Public Library',
      'my_book': 'My Book',
      'about_us': 'About Us',
      'assistant': 'Assistant',
      'profile': 'Profile',
      'login': 'Login',
      'register': 'Register',
      'logout': 'Logout',
      'settings': 'Account Settings',
      'display_name': 'Display Name',
      'bio': 'Bio',
      'reading_goal': 'Reading Goal',
      'save_changes': 'Save Changes',
      'in_library': 'In Library',
      'completed': 'Completed',
      'reviews': 'Reviews',
      'explored': 'Explored',
      'library_breakdown': 'Library Breakdown',
      'currently_reading': 'Currently Reading',
      'want_to_read': 'Want to Read',
      'favorites': 'Favorites',
      'reading_journey': 'Reading Journey',
      'books': 'Books',
      'avg_rating': 'Avg Rating',
      'days_member': 'days',
      'rank': 'Rank',
      'streak': 'Streak',
      'language': 'Language',
      'select_language': 'Select Language',
      'profile_updated': 'Profile updated!',
      'books_year': 'books/year',
    },
    'ar': {
      'discover': 'اكتشف',
      'public_library': 'المكتبة العامة',
      'my_book': 'كتبي',
      'about_us': 'من نحن',
      'assistant': 'المساعد',
      'profile': 'الشخصي',
      'login': 'تسجيل الدخول',
      'register': 'إنشاء حساب',
      'logout': 'تسجيل الخروج',
      'settings': 'إعدادات الحساب',
      'display_name': 'الاسم المستعار',
      'bio': 'السيرة الذاتية',
      'reading_goal': 'هدف القراءة',
      'save_changes': 'حفظ التغييرات',
      'in_library': 'في المكتبة',
      'completed': 'مكتملة',
      'reviews': 'المراجعات',
      'explored': 'مستكشف',
      'library_breakdown': 'تصنيف المكتبة',
      'currently_reading': 'قيد القراءة حالياً',
      'want_to_read': 'أرغب في قراءته',
      'favorites': 'المفضلة',
      'reading_journey': 'رحلة القراءة',
      'books': 'الكتب',
      'avg_rating': 'متوسط التقييم',
      'days_member': 'يوم',
      'rank': 'الرتبة',
      'streak': 'سلسلة القراءة',
      'language': 'اللغة',
      'select_language': 'اختر اللغة',
      'profile_updated': 'تم تحديث الحساب!',
      'books_year': 'كتاب/سنة',
    }
  };

  static String translate(String key, String lang) {
    return _values[lang]?.containsKey(key) == true
        ? _values[lang]![key]!
        : key;
  }
}

extension LocalizationExtension on BuildContext {
  String t(String key) {
    // Provider will automatically listen to LocaleProvider changes
    final localeProvider = Provider.of<LocaleProvider>(this, listen: true);
    final lang = localeProvider.locale.languageCode;
    return AppTranslations.translate(key, lang);
  }

  bool get isRtl {
    final localeProvider = Provider.of<LocaleProvider>(this, listen: true);
    return localeProvider.locale.languageCode == 'ar';
  }
}
