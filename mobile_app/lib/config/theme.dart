import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Brand Colors - "The Inspired Curator"
  static const Color primary = Color(0xFF6F26F6);
  static const Color onPrimary = Color(0xFFFDF7FF);
  static const Color primaryContainer = Color(0xFFCAB6FF);
  static const Color onPrimaryContainer = Color(0xFF4800B1);

  static const Color secondary = Color(0xFFA14200);
  static const Color onSecondary = Color(0xFFFFF7F5);
  static const Color secondaryContainer = Color(0xFFFFDBCB);
  static const Color onSecondaryContainer = Color(0xFF8B3800);

  static const Color tertiary = Color(0xFF7B5913);
  static const Color onTertiary = Color(0xFFFFF8F1);
  static const Color tertiaryContainer = Color(0xFFFED07F);
  static const Color onTertiaryContainer = Color(0xFF634500);

  static const Color error = Color(0xFFAC3149);
  static const Color onError = Color(0xFFFFF7F7);
  static const Color errorContainer = Color(0xFFF76A80);
  static const Color onErrorContainer = Color(0xFF68001F);

  static const Color background = Color(0xFFFFF9F0);
  static const Color onBackground = Color(0xFF353229);
  static const Color surface = Color(0xFFFFF9F0);
  static const Color onSurface = Color(0xFF353229);
  static const Color onSurfaceVariant = Color(0xFF635F54);

  static const Color outline = Color(0xFF7F7A6F);
  static const Color outlineVariant = Color(0xFFB7B1A4);

  // Surface Tiers
  static const Color surfaceContainerLowest = Color(0xFFFFFFFF);
  static const Color surfaceContainerLow = Color(0xFFF9F3E9);
  static const Color surfaceContainer = Color(0xFFF4EDE2);
  static const Color surfaceContainerHigh = Color(0xFFEEE7DB);
  static const Color surfaceContainerHighest = Color(0xFFE9E2D4);

  static ThemeData get lightTheme {
    final colorScheme = const ColorScheme(
      brightness: Brightness.light,
      primary: primary,
      onPrimary: onPrimary,
      primaryContainer: primaryContainer,
      onPrimaryContainer: onPrimaryContainer,
      secondary: secondary,
      onSecondary: onSecondary,
      secondaryContainer: secondaryContainer,
      onSecondaryContainer: onSecondaryContainer,
      tertiary: tertiary,
      onTertiary: onTertiary,
      tertiaryContainer: tertiaryContainer,
      onTertiaryContainer: onTertiaryContainer,
      error: error,
      onError: onError,
      errorContainer: errorContainer,
      onErrorContainer: onErrorContainer,
      background: background,
      onBackground: onBackground,
      surface: surface,
      onSurface: onSurface,
      onSurfaceVariant: onSurfaceVariant,
      outline: outline,
      outlineVariant: outlineVariant,
      surfaceContainerLowest: surfaceContainerLowest,
      surfaceContainerLow: surfaceContainerLow,
      surfaceContainer: surfaceContainer,
      surfaceContainerHigh: surfaceContainerHigh,
      surfaceContainerHighest: surfaceContainerHighest,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: background,
      
      // Typography
      textTheme: _buildTextTheme(colorScheme),

      // App Bar - Glassmorphism fallback
      appBarTheme: AppBarTheme(
        backgroundColor: surface.withValues(alpha: 0.8),
        scrolledUnderElevation: 0,
        centerTitle: true,
        iconTheme: const IconThemeData(color: primary),
        titleTextStyle: GoogleFonts.beVietnamPro(
          color: primary,
          fontSize: 24,
          fontWeight: FontWeight.w900,
          letterSpacing: -0.5,
        ),
      ),

      // Cards - xl/lg rounding, no lines
      cardTheme: CardTheme(
        color: surfaceContainerLowest,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(32), // lg
        ),
      ),

      // Buttons - xl rounding
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: onPrimary,
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(48)), // xl
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          textStyle: GoogleFonts.plusJakartaSans(
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ),
      
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: surfaceContainerLowest,
          foregroundColor: primary,
          elevation: 2,
          shadowColor: onSurface.withValues(alpha: 0.06),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(48)),
        ),
      ),

      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primary,
          side: const BorderSide(color: outlineVariant, width: 1.5),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(48)),
        ),
      ),

      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: primary,
          textStyle: GoogleFonts.plusJakartaSans(fontWeight: FontWeight.bold),
        ),
      ),

      // Inputs - md rounding, ghost border on focus
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceContainerHigh,
        contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(24), // md
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(24),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(24),
          borderSide: const BorderSide(color: primary, width: 2), // Ghost border
        ),
        labelStyle: const TextStyle(color: onSurfaceVariant),
      ),

      // Bottom Nav Bar
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: surfaceContainerLowest,
        elevation: 8,
        selectedItemColor: primary,
        unselectedItemColor: onSurfaceVariant,
        type: BottomNavigationBarType.fixed,
        selectedLabelStyle: GoogleFonts.plusJakartaSans(fontWeight: FontWeight.bold, fontSize: 12),
        unselectedLabelStyle: GoogleFonts.plusJakartaSans(fontWeight: FontWeight.w600, fontSize: 12),
      ),
      
      // Floating Action Button
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: primary,
        foregroundColor: onPrimary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      ),
    );
  }

  static TextTheme _buildTextTheme(ColorScheme cs) {
    // Headlines: Be Vietnam Pro
    // Body: Plus Jakarta Sans
    final displayFont = GoogleFonts.beVietnamPro();
    final bodyFont = GoogleFonts.plusJakartaSans();

    return TextTheme(
      displayLarge: displayFont.copyWith(fontSize: 57, fontWeight: FontWeight.w900, letterSpacing: -1.0, color: cs.onSurface),
      displayMedium: displayFont.copyWith(fontSize: 45, fontWeight: FontWeight.w800, letterSpacing: -0.5, color: cs.onSurface),
      displaySmall: displayFont.copyWith(fontSize: 36, fontWeight: FontWeight.w800, color: cs.onSurface),
      
      headlineLarge: displayFont.copyWith(fontSize: 32, fontWeight: FontWeight.w800, color: cs.onSurface),
      headlineMedium: displayFont.copyWith(fontSize: 28, fontWeight: FontWeight.w800, color: cs.onSurface),
      headlineSmall: displayFont.copyWith(fontSize: 24, fontWeight: FontWeight.bold, color: cs.onSurface),
      
      titleLarge: bodyFont.copyWith(fontSize: 22, fontWeight: FontWeight.bold, color: cs.onSurface),
      titleMedium: bodyFont.copyWith(fontSize: 16, fontWeight: FontWeight.bold, color: cs.onSurface),
      titleSmall: bodyFont.copyWith(fontSize: 14, fontWeight: FontWeight.bold, color: cs.onSurface),
      
      bodyLarge: bodyFont.copyWith(fontSize: 16, fontWeight: FontWeight.normal, height: 1.6, color: cs.onSurface), // Height 1.6 for Arabic
      bodyMedium: bodyFont.copyWith(fontSize: 14, fontWeight: FontWeight.normal, height: 1.6, color: cs.onSurface),
      bodySmall: bodyFont.copyWith(fontSize: 12, fontWeight: FontWeight.normal, height: 1.6, color: cs.onSurfaceVariant),
      
      labelLarge: bodyFont.copyWith(fontSize: 14, fontWeight: FontWeight.bold, color: cs.primary),
      labelMedium: bodyFont.copyWith(fontSize: 12, fontWeight: FontWeight.bold, color: cs.onSurfaceVariant),
      labelSmall: bodyFont.copyWith(fontSize: 11, fontWeight: FontWeight.bold, color: cs.onSurfaceVariant),
    );
  }
}
