import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../providers/auth_provider.dart';
import '../config/translations.dart';
import 'home_screen.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({Key? key}) : super(key: key);
  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final Set<String> _selected = {};
  bool _saving = false;

  static const _interests = [
    {'id': 'fiction', 'name': 'Fiction', 'emoji': '📖', 'color': 0xFF6366F1},
    {'id': 'science', 'name': 'Science', 'emoji': '🔬', 'color': 0xFF10B981},
    {'id': 'history', 'name': 'History', 'emoji': '🏛️', 'color': 0xFFF59E0B},
    {'id': 'philosophy', 'name': 'Philosophy', 'emoji': '🤔', 'color': 0xFF8B5CF6},
    {'id': 'psychology', 'name': 'Psychology', 'emoji': '🧠', 'color': 0xFFEC4899},
    {'id': 'technology', 'name': 'Technology', 'emoji': '💻', 'color': 0xFF3B82F6},
    {'id': 'business', 'name': 'Business', 'emoji': '💼', 'color': 0xFF14B8A6},
    {'id': 'self-help', 'name': 'Self Help', 'emoji': '🌱', 'color': 0xFF22C55E},
    {'id': 'poetry', 'name': 'Poetry', 'emoji': '✍️', 'color': 0xFFA855F7},
    {'id': 'religion', 'name': 'Religion', 'emoji': '🕌', 'color': 0xFF0EA5E9},
    {'id': 'mystery', 'name': 'Mystery', 'emoji': '🔍', 'color': 0xFFEF4444},
    {'id': 'romance', 'name': 'Romance', 'emoji': '💕', 'color': 0xFFF43F5E},
    {'id': 'biography', 'name': 'Biography', 'emoji': '👤', 'color': 0xFF64748B},
    {'id': 'art', 'name': 'Art', 'emoji': '🎨', 'color': 0xFFE11D48},
    {'id': 'fantasy', 'name': 'Fantasy', 'emoji': '🐉', 'color': 0xFF7C3AED},
    {'id': 'thriller', 'name': 'Thriller', 'emoji': '🔪', 'color': 0xFFDC2626},
    {'id': 'travel', 'name': 'Travel', 'emoji': '✈️', 'color': 0xFF06B6D4},
    {'id': 'cooking', 'name': 'Cooking', 'emoji': '🍳', 'color': 0xFFF97316},
  ];

  void _toggle(String id) {
    setState(() {
      if (_selected.contains(id)) {
        _selected.remove(id);
      } else {
        _selected.add(id);
      }
    });
  }

  Future<void> _finish() async {
    if (_selected.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(context.t('choose_at_least_one_interest')), behavior: SnackBarBehavior.floating),
      );
      return;
    }
    setState(() => _saving = true);
    
    // Call via Provider to update app state!
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    await authProvider.completeOnboarding(_selected.toList());
    
    if (mounted) {
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const HomeScreen()));
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 32, 24, 0),
              child: Column(
                children: [
                  Container(
                    width: 64, height: 64,
                    decoration: BoxDecoration(
                      color: cs.primary.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Icon(Icons.auto_stories, size: 32, color: cs.primary),
                  ),
                  const SizedBox(height: 20),
                  Text(context.t('welcome_to_elham'), style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w900)),
                  const SizedBox(height: 8),
                  Text(context.t('choose_interests_desc'), style: TextStyle(fontSize: 16, color: cs.onSurfaceVariant), textAlign: TextAlign.center),
                  const SizedBox(height: 6),
                  Text('${context.t('selected')} ${_selected.length} ${context.t('of')} ${_interests.length}', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: cs.primary)),
                  const SizedBox(height: 16),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: _selected.length / 3.0 > 1.0 ? 1.0 : _selected.length / 3.0,
                      backgroundColor: cs.surfaceContainerHighest,
                      minHeight: 6,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: GridView.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3, childAspectRatio: 0.95,
                    crossAxisSpacing: 10, mainAxisSpacing: 10,
                  ),
                  itemCount: _interests.length,
                  itemBuilder: (context, i) {
                    final item = _interests[i];
                    final id = item['id'] as String;
                    final isOn = _selected.contains(id);
                    final color = Color(item['color'] as int);
                    return GestureDetector(
                      onTap: () => _toggle(id),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        decoration: BoxDecoration(
                          color: isOn ? color.withValues(alpha: 0.15) : cs.surfaceContainerHighest.withValues(alpha: 0.5),
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: isOn ? color : Colors.transparent, width: 2.5),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(item['emoji'] as String, style: const TextStyle(fontSize: 32)),
                            const SizedBox(height: 8),
                            Text(context.t('interest_$id'), style: TextStyle(fontWeight: FontWeight.w800, fontSize: 13, color: isOn ? color : cs.onSurface)),
                            if (isOn) Padding(padding: const EdgeInsets.only(top: 4), child: Icon(Icons.check_circle, size: 18, color: color)),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
              child: Column(
                children: [
                  SizedBox(
                    width: double.infinity, height: 56,
                    child: FilledButton(
                      onPressed: _saving ? null : _finish,
                      style: FilledButton.styleFrom(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28))),
                      child: _saving
                          ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                          : Text('${context.t('start_your_journey')} (${_selected.length})', style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 16)),
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const HomeScreen())),
                    child: Text(context.t('skip_for_now'), style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
