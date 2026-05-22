import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../config/app_config.dart';
import '../providers/auth_provider.dart';
import '../providers/locale_provider.dart';
import '../config/translations.dart';
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
  bool _isUploading = false;

  @override
  void initState() {
    super.initState();
    _loadStats();
  }

  Future<void> _loadStats() async {
    final stats = await UserService.getStats();
    if (mounted) setState(() { _stats = stats; });
  }

  Future<void> _pickAndUploadImage(ImageSource source) async {
    final picker = ImagePicker();
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    try {
      final pickedFile = await picker.pickImage(
        source: source,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 85,
      );
      if (pickedFile == null) return;

      setState(() {
        _isUploading = true;
      });

      final success = await authProvider.updateProfilePicture(pickedFile);

      if (mounted) {
        setState(() {
          _isUploading = false;
        });
        if (success) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(context.t('profile_picture_updated')),
              behavior: SnackBarBehavior.floating,
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(context.t('error_uploading')),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${context.t('error_uploading')}: $e'),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  Future<void> _confirmAndDeleteImage() async {
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final confirm = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(context.t('delete_profile_picture')),
          content: Text(context.t('confirm_delete_picture')),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(context.t('cancel')),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, true),
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: Text(context.t('delete')),
            ),
          ],
        );
      },
    );

    if (confirm != true) return;

    try {
      setState(() {
        _isUploading = true;
      });

      final success = await authProvider.deleteProfilePicture();

      if (mounted) {
        setState(() {
          _isUploading = false;
        });
        if (success) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(context.t('profile_picture_deleted')),
              behavior: SnackBarBehavior.floating,
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(context.t('error_deleting')),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${context.t('error_deleting')}: $e'),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  void _showImageSourceBottomSheet() {
    final cs = Theme.of(context).colorScheme;
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final user = authProvider.currentUser;

    showModalBottomSheet(
      context: context,
      backgroundColor: cs.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  context.t('select_image_source'),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: cs.onSurface,
                  ),
                ),
              ),
              ListTile(
                leading: Icon(Icons.photo_library, color: cs.primary),
                title: Text(context.t('gallery')),
                onTap: () {
                  Navigator.pop(context);
                  _pickAndUploadImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: Icon(Icons.camera_alt, color: cs.primary),
                title: Text(context.t('camera')),
                onTap: () {
                  Navigator.pop(context);
                  _pickAndUploadImage(ImageSource.camera);
                },
              ),
              if (user != null && user.profilePicture != null)
                ListTile(
                  leading: const Icon(Icons.delete, color: Colors.red),
                  title: Text(
                    context.t('delete_profile_picture'),
                    style: const TextStyle(color: Colors.red),
                  ),
                  onTap: () {
                    Navigator.pop(context);
                    _confirmAndDeleteImage();
                  },
                ),
            ],
          ),
        );
      },
    );
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
              Stack(
                alignment: Alignment.center,
                children: [
                  GestureDetector(
                    onTap: _isUploading ? null : _showImageSourceBottomSheet,
                    child: Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 3),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.15),
                            blurRadius: 10,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                      child: CircleAvatar(
                        radius: 48,
                        backgroundColor: Colors.white24,
                        backgroundImage: user.profilePicture != null
                            ? CachedNetworkImageProvider(
                                '${AppConfig.serverBaseUrl}${user.profilePicture}',
                              )
                            : null,
                        child: user.profilePicture == null
                            ? Text(
                                user.name.isNotEmpty ? user.name[0].toUpperCase() : 'U',
                                style: const TextStyle(
                                  fontSize: 36,
                                  fontWeight: FontWeight.w900,
                                  color: Colors.white,
                                ),
                              )
                            : null,
                      ),
                    ),
                  ),
                  if (_isUploading)
                    Positioned.fill(
                      child: Container(
                        decoration: const BoxDecoration(
                          color: Colors.black45,
                          shape: BoxShape.circle,
                        ),
                        child: const Center(
                          child: CircularProgressIndicator(
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  if (!_isUploading)
                    Positioned(
                      bottom: 0,
                      right: 0,
                      child: GestureDetector(
                        onTap: _showImageSourceBottomSheet,
                        child: Container(
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(
                            color: Theme.of(context).colorScheme.primary,
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 2),
                          ),
                          child: const Icon(
                            Icons.camera_alt,
                            size: 14,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                ],
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
                  _badge('${_stats['days_member'] ?? 0} ${context.t('days_member')}', Icons.verified, Colors.white24, Colors.white),
                  _badge(_stats['rank'] ?? user.rank, Icons.workspace_premium, const Color(0xFFFED07F), const Color(0xFF4B3400)),
                  if ((_stats['streak'] ?? 0) > 1) _badge('${_stats['streak']} ${context.t('streak')}', Icons.local_fire_department, Colors.orange.withValues(alpha: 0.2), Colors.orange[200]!),
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
            StatsCard(icon: Icons.library_books, iconColor: Theme.of(context).colorScheme.primary, value: '${_stats['total_books'] ?? 0}', label: context.t('in_library'), badge: (_stats['total_books'] ?? 0) >= 10 ? '📚 Collector' : null),
            StatsCard(icon: Icons.task_alt, iconColor: Colors.green, value: '${_stats['books_finished'] ?? 0}', label: context.t('completed')),
            StatsCard(icon: Icons.rate_review, iconColor: Theme.of(context).colorScheme.secondary, value: '${_stats['total_reviews'] ?? 0}', label: context.t('reviews'), badge: (_stats['total_reviews'] ?? 0) >= 5 ? '✍️ Critic' : null),
            StatsCard(icon: Icons.visibility, iconColor: Colors.blue, value: '${_stats['total_views'] ?? 0}', label: context.t('explored'), badge: (_stats['total_views'] ?? 0) >= 20 ? '🧭 Explorer' : null),
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

    final localeProvider = Provider.of<LocaleProvider>(context);
    final currentLang = localeProvider.locale.languageCode;

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
                Text(context.t('settings'), style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
              ]),
              const SizedBox(height: 16),
              _settingsField(context.t('display_name'), nameCtrl, Icons.badge),
              const SizedBox(height: 12),
              _settingsField(context.t('bio'), bioCtrl, Icons.history_edu, maxLines: 2),
              const SizedBox(height: 12),
              _settingsField(context.t('reading_goal'), goalCtrl, Icons.flag, keyboardType: TextInputType.number, suffix: context.t('books_year')),
              const SizedBox(height: 16),
              
              // ─── Premium Language Selector ───
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Icon(Icons.language, size: 14, color: cs.onSurfaceVariant),
                      const SizedBox(width: 6),
                      Text(context.t('language'), style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: cs.onSurfaceVariant, letterSpacing: 1)),
                    ],
                  ),
                  Row(
                    children: [
                      ChoiceChip(
                        label: const Text('العربية', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                        selected: currentLang == 'ar',
                        selectedColor: cs.primary.withValues(alpha: 0.2),
                        labelStyle: TextStyle(color: currentLang == 'ar' ? cs.primary : cs.onSurfaceVariant),
                        onSelected: (selected) {
                          if (selected) {
                            localeProvider.setLocale(const Locale('ar'));
                          }
                        },
                      ),
                      const SizedBox(width: 8),
                      ChoiceChip(
                        label: const Text('English', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                        selected: currentLang == 'en',
                        selectedColor: cs.primary.withValues(alpha: 0.2),
                        labelStyle: TextStyle(color: currentLang == 'en' ? cs.primary : cs.onSurfaceVariant),
                        onSelected: (selected) {
                          if (selected) {
                            localeProvider.setLocale(const Locale('en'));
                          }
                        },
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 20),

              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: () async {
                    await UserService.updateProfile(name: nameCtrl.text, bio: bioCtrl.text, readingGoal: int.tryParse(goalCtrl.text) ?? 0);
                    if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(context.t('profile_updated')), behavior: SnackBarBehavior.floating));
                  },
                  icon: const Icon(Icons.save, size: 18),
                  label: Text(context.t('save_changes'), style: const TextStyle(fontWeight: FontWeight.bold)),
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
      {'label': context.t('currently_reading'), 'value': _stats['books_reading'] ?? 0, 'color': Colors.blue, 'icon': Icons.auto_stories},
      {'label': context.t('completed'), 'value': _stats['books_finished'] ?? 0, 'color': Colors.green, 'icon': Icons.check_circle},
      {'label': context.t('want_to_read'), 'value': _stats['books_later'] ?? 0, 'color': Colors.amber, 'icon': Icons.bookmark},
      {'label': context.t('favorites'), 'value': _stats['books_favorite'] ?? 0, 'color': Colors.red, 'icon': Icons.favorite},
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
              Text(context.t('library_breakdown'), style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
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
              Text(context.t('reading_journey'), style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: cs.onSurface)),
            ]),
            const SizedBox(height: 16),
            Row(
              children: [
                _milestone(context, Icons.menu_book, cs.primary, '${_stats['total_books'] ?? 0}', context.t('books')),
                const SizedBox(width: 12),
                _milestone(context, Icons.star, cs.secondary, '${_stats['avg_rating'] ?? '—'}', context.t('avg_rating')),
                const SizedBox(width: 12),
                _milestone(context, Icons.explore, Colors.blue, '${_stats['total_views'] ?? 0}', context.t('explored')),
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
