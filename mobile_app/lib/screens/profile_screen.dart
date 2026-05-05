import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../services/user_service.dart';
import 'widgets/bottom_nav_bar.dart';
import 'widgets/stats_card.dart';
import 'login_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  Map<String, dynamic> _stats = {};

  @override
  void initState() {
    super.initState();
    _loadStats();
  }

  Future<void> _loadStats() async {
    final stats = await UserService.getStats();
    if (mounted) setState(() { _stats = stats; });
  }

  @override
  Widget build(BuildContext context) {
    final user = Provider.of<AuthProvider>(context).currentUser;

    return Scaffold(
      body: user == null
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _loadStats,
              child: CustomScrollView(
                slivers: [
                  // ─── Hero Banner ───
                  SliverToBoxAdapter(child: _buildHeroBanner(context, user)),
                  // ─── Stats Cards ───
                  SliverToBoxAdapter(child: _buildStatsGrid(context)),
                  // ─── Settings ───
                  SliverToBoxAdapter(child: _buildSettings(context, user)),
                  // ─── Library Breakdown ───
                  SliverToBoxAdapter(child: _buildLibraryBreakdown(context)),
                  // ─── Reading Journey ───
                  SliverToBoxAdapter(child: _buildReadingJourney(context)),
                  const SliverToBoxAdapter(child: SizedBox(height: 100)),
                ],
              ),
            ),
      bottomNavigationBar: const AppBottomNavBar(currentIndex: 3),
    );
  }

  Widget _buildHeroBanner(BuildContext context, dynamic user) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [Color(0xFF30007C), Color(0xFF5300C8), Color(0xFF6F26F6)],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 48),
          child: Column(
            children: [
              // Top actions
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  IconButton(
                    icon: const Icon(Icons.logout, color: Colors.white70, size: 20),
                    onPressed: () async {
                      await Provider.of<AuthProvider>(context, listen: false).logout();
                      if (context.mounted) Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const LoginScreen()));
                    },
                  ),
                ],
              ),
              // Avatar + Info
              CircleAvatar(
                radius: 48,
                backgroundColor: Colors.white24,
                child: Text(
                  user.name.isNotEmpty ? user.name[0].toUpperCase() : 'U',
                  style: const TextStyle(fontSize: 36, fontWeight: FontWeight.w900, color: Colors.white60),
                ),
              ),
              const SizedBox(height: 12),
              Text(user.name, style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w900, color: Colors.white)),
              const SizedBox(height: 4),
              Text(user.email, style: const TextStyle(color: Colors.white60)),
              const SizedBox(height: 12),
              // Badges
              Wrap(
                spacing: 8, runSpacing: 8, alignment: WrapAlignment.center,
                children: [
                  _badge('${_stats['days_member'] ?? 0} days', Icons.verified, Colors.white24, Colors.white),
                  _badge(_stats['rank'] ?? user.rank, Icons.workspace_premium, const Color(0xFFFED07F), const Color(0xFF4B3400)),
                  if ((_stats['streak'] ?? 0) > 1) _badge('${_stats['streak']} Day Streak', Icons.local_fire_department, Colors.orange.withValues(alpha: 0.2), Colors.orange[200]!),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _badge(String text, IconData icon, Color bg, Color fg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: fg),
          const SizedBox(width: 4),
          Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: fg)),
        ],
      ),
    );
  }

  Widget _buildStatsGrid(BuildContext context) {
    return Transform.translate(
      offset: const Offset(0, -24),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          mainAxisSpacing: 12, crossAxisSpacing: 12,
          childAspectRatio: 1.0,
          children: [
            StatsCard(icon: Icons.library_books, iconColor: Theme.of(context).colorScheme.primary, value: '${_stats['total_books'] ?? 0}', label: 'In Library', badge: (_stats['total_books'] ?? 0) >= 10 ? '📚 Collector' : null),
            StatsCard(icon: Icons.task_alt, iconColor: Colors.green, value: '${_stats['books_finished'] ?? 0}', label: 'Completed'),
            StatsCard(icon: Icons.rate_review, iconColor: Theme.of(context).colorScheme.secondary, value: '${_stats['total_reviews'] ?? 0}', label: 'Reviews', badge: (_stats['total_reviews'] ?? 0) >= 5 ? '✍️ Critic' : null),
            StatsCard(icon: Icons.visibility, iconColor: Colors.blue, value: '${_stats['total_views'] ?? 0}', label: 'Explored', badge: (_stats['total_views'] ?? 0) >= 20 ? '🧭 Explorer' : null),
          ],
        ),
      ),
    );
  }

  Widget _buildSettings(BuildContext context, dynamic user) {
    final cs = Theme.of(context).colorScheme;
    final nameCtrl = TextEditingController(text: user.name);
    final bioCtrl = TextEditingController(text: user.bio ?? '');
    final goalCtrl = TextEditingController(text: '${user.readingGoal}');

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.15)),
          boxShadow: [BoxShadow(color: cs.shadow.withValues(alpha: 0.05), blurRadius: 12, offset: const Offset(0, 4))],
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Icon(Icons.settings, color: cs.primary, size: 20),
                const SizedBox(width: 8),
                Text('Account Settings', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
              ]),
              const SizedBox(height: 16),
              _settingsField('Display Name', nameCtrl, Icons.badge),
              const SizedBox(height: 12),
              _settingsField('Bio', bioCtrl, Icons.history_edu, maxLines: 2),
              const SizedBox(height: 12),
              _settingsField('Reading Goal', goalCtrl, Icons.flag, keyboardType: TextInputType.number, suffix: 'books/year'),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: () async {
                    await UserService.updateProfile(name: nameCtrl.text, bio: bioCtrl.text, readingGoal: int.tryParse(goalCtrl.text) ?? 0);
                    if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile updated!'), behavior: SnackBarBehavior.floating));
                  },
                  icon: const Icon(Icons.save, size: 18),
                  label: const Text('Save Changes', style: TextStyle(fontWeight: FontWeight.bold)),
                  style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 14), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _settingsField(String label, TextEditingController ctrl, IconData icon, {int maxLines = 1, TextInputType? keyboardType, String? suffix}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [
          Icon(icon, size: 14, color: Theme.of(context).colorScheme.onSurfaceVariant),
          const SizedBox(width: 6),
          Text(label, style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Theme.of(context).colorScheme.onSurfaceVariant, letterSpacing: 1)),
        ]),
        const SizedBox(height: 6),
        TextField(
          controller: ctrl,
          maxLines: maxLines,
          keyboardType: keyboardType,
          decoration: InputDecoration(
            suffixText: suffix,
            contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
            filled: true,
            fillColor: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.3),
          ),
        ),
      ],
    );
  }

  Widget _buildLibraryBreakdown(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final items = [
      {'label': 'Currently Reading', 'value': _stats['books_reading'] ?? 0, 'color': Colors.blue, 'icon': Icons.auto_stories},
      {'label': 'Completed', 'value': _stats['books_finished'] ?? 0, 'color': Colors.green, 'icon': Icons.check_circle},
      {'label': 'Want to Read', 'value': _stats['books_later'] ?? 0, 'color': Colors.amber, 'icon': Icons.bookmark},
      {'label': 'Favorites', 'value': _stats['books_favorite'] ?? 0, 'color': Colors.red, 'icon': Icons.favorite},
    ];

    final maxVal = items.map((i) => i['value'] as int).fold(1, (a, b) => a > b ? a : b);

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.15)),
          boxShadow: [BoxShadow(color: cs.shadow.withValues(alpha: 0.05), blurRadius: 12, offset: const Offset(0, 4))],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(Icons.pie_chart, color: cs.secondary, size: 20),
              const SizedBox(width: 8),
              Text('Library Breakdown', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
            ]),
            const SizedBox(height: 16),
            ...items.map((item) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Row(
                children: [
                  Icon(item['icon'] as IconData, color: item['color'] as Color, size: 20),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(item['label'] as String, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                            Text('${item['value']}', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: item['color'] as Color)),
                          ],
                        ),
                        const SizedBox(height: 4),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: maxVal > 0 ? (item['value'] as int) / maxVal : 0,
                            backgroundColor: cs.surfaceContainerHighest,
                            color: item['color'] as Color,
                            minHeight: 6,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildReadingJourney(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.15)),
          boxShadow: [BoxShadow(color: cs.shadow.withValues(alpha: 0.05), blurRadius: 12, offset: const Offset(0, 4))],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(Icons.insights, color: Colors.green, size: 20),
              const SizedBox(width: 8),
              Text('Reading Journey', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
            ]),
            const SizedBox(height: 16),
            Row(
              children: [
                _milestone(context, Icons.menu_book, cs.primary, '${_stats['total_books'] ?? 0}', 'Books'),
                const SizedBox(width: 12),
                _milestone(context, Icons.star, cs.secondary, '${_stats['avg_rating'] ?? '—'}', 'Avg Rating'),
                const SizedBox(width: 12),
                _milestone(context, Icons.explore, Colors.blue, '${_stats['total_views'] ?? 0}', 'Explored'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _milestone(BuildContext context, IconData icon, Color color, String value, String label) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(colors: [color.withValues(alpha: 0.05), color.withValues(alpha: 0.1)]),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.1)),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 24),
            const SizedBox(height: 8),
            Text(value, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900)),
            Text(label, style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}
