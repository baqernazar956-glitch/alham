import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../providers/books_provider.dart';
import '../providers/auth_provider.dart';
import '../models/book.dart';
import '../services/user_service.dart';
import 'widgets/bottom_nav_bar.dart';
import 'book_detail_screen.dart';


class BooksScreen extends StatefulWidget {
  const BooksScreen({Key? key}) : super(key: key);

  @override
  State<BooksScreen> createState() => _BooksScreenState();
}

class _BooksScreenState extends State<BooksScreen> {
  Map<String, dynamic> _stats = {'books_read': 0, 'streak_days': 0};
  bool _isLoading = false;
  String _selectedStatus = 'all';

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    final bp = Provider.of<BooksProvider>(context, listen: false);
    await bp.fetchUserLibrary();
    final stats = await UserService.getStats();
    if (mounted) {
      setState(() {
        _stats = stats;
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bp = Provider.of<BooksProvider>(context);
    final user = Provider.of<AuthProvider>(context).currentUser;

    if (user == null) {
      return const Scaffold(body: Center(child: Text("Please login first.")));
    }

    final readingBooks = bp.userLibrary.where((b) => b.status == 'reading').toList();
    final wishlistBooks = bp.userLibrary.where((b) => b.status == 'later').toList();
    final readBooks = bp.userLibrary.where((b) => b.status == 'finished').toList();
    final favoriteBooks = bp.userLibrary.where((b) => b.status == 'favorite').toList();

    List<Book> displayBooks = [];
    String sectionTitle = '';

    switch (_selectedStatus) {
      case 'all':
        displayBooks = bp.userLibrary;
        sectionTitle = 'All Books';
        break;
      case 'favorite':
        displayBooks = favoriteBooks;
        sectionTitle = 'Favorites';
        break;
      case 'reading':
        displayBooks = readingBooks;
        sectionTitle = 'Reading Now';
        break;
      case 'later':
        displayBooks = wishlistBooks;
        sectionTitle = 'Plan to Read';
        break;
      case 'finished':
        displayBooks = readBooks;
        sectionTitle = 'Completed Books';
        break;
    }

    return Scaffold(
      body: RefreshIndicator(
        onRefresh: _loadData,
        child: CustomScrollView(
          slivers: [
            // Header
            SliverAppBar(
              floating: true,
              pinned: true,
              title: const Text('Elham', style: TextStyle(fontWeight: FontWeight.w900, fontSize: 24)),
              actions: [
                Container(
                  margin: const EdgeInsets.only(right: 16),
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: cs.surfaceContainerHighest,
                    shape: BoxShape.circle,
                    border: Border.all(color: cs.surfaceContainerLow, width: 2),
                  ),
                  child: Icon(Icons.person, color: cs.onSurfaceVariant, size: 20),
                )
              ],
            ),

            if (_isLoading && bp.userLibrary.isEmpty)
              const SliverFillRemaining(child: Center(child: CircularProgressIndicator())),

            if (!(_isLoading && bp.userLibrary.isEmpty))
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 32),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('My Library', style: Theme.of(context).textTheme.displaySmall?.copyWith(color: cs.primary, fontFamily: 'Be Vietnam Pro', fontWeight: FontWeight.w900)),
                      const SizedBox(height: 8),
                      Text('Welcome back, continue your reading journey.', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: cs.onSurfaceVariant, fontWeight: FontWeight.normal)),
                    ],
                  ),
                ),
              ),

            if (!(_isLoading && bp.userLibrary.isEmpty))
              SliverToBoxAdapter(child: _buildStatsBento(context, _stats, bp.userLibrary.length)),

            if (!(_isLoading && bp.userLibrary.isEmpty))
              SliverToBoxAdapter(child: _buildCategorySelector(context)),

            if (!(_isLoading && bp.userLibrary.isEmpty) && displayBooks.isNotEmpty)
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  child: Text(sectionTitle, style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 24, fontWeight: FontWeight.bold, color: cs.onSurface)),
                ),
              ),

            if (!(_isLoading && bp.userLibrary.isEmpty) && displayBooks.isNotEmpty)
              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                sliver: SliverGrid(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 16,
                    crossAxisSpacing: 16,
                    childAspectRatio: 0.65,
                  ),
                  delegate: SliverChildBuilderDelegate(
                    (context, index) {
                      final book = displayBooks[index];
                      return Stack(
                        children: [
                          GestureDetector(
                            onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book))),
                            child: Container(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(20),
                                boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, 4))],
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(
                                    child: ClipRRect(
                                      borderRadius: BorderRadius.circular(20),
                                      child: CachedNetworkImage(
                                        imageUrl: book.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w400' : book.coverUrl)}' : 'https://placehold.co/200x300',
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(height: 12),
                                  Text(book.title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14), maxLines: 1, overflow: TextOverflow.ellipsis),
                                  Text(book.author, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12), maxLines: 1, overflow: TextOverflow.ellipsis),
                                  if (_selectedStatus == 'reading') ...[
                                    const SizedBox(height: 8),
                                    LinearProgressIndicator(
                                      value: book.readingProgress / 100,
                                      backgroundColor: cs.surfaceContainerHighest,
                                      borderRadius: BorderRadius.circular(4),
                                      minHeight: 4,
                                    ),
                                  ]
                                ],
                              ),
                            ),
                          ),
                          Positioned(
                            top: 8,
                            right: 8,
                            child: GestureDetector(
                              onTap: () async {
                                final confirm = await showDialog<bool>(
                                  context: context,
                                  builder: (context) => AlertDialog(
                                    title: const Text('Delete Book'),
                                    content: const Text('Are you sure you want to delete this book from your library?'),
                                    actions: [
                                      TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
                                      TextButton(
                                        onPressed: () => Navigator.pop(context, true), 
                                        child: const Text('Delete', style: TextStyle(color: Colors.red)),
                                      ),
                                    ],
                                  ),
                                );
                                
                                if (confirm == true) {
                                  final success = await bp.removeFromLibrary(book.uniqueId);
                                  if (success && mounted) {
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(content: Text('Book deleted successfully'))
                                    );
                                  }
                                }
                              },
                              child: Container(
                                padding: const EdgeInsets.all(6),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.5),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(Icons.delete_outline, color: Colors.white, size: 18),
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                    childCount: displayBooks.length,
                  ),
                ),
              ),

            if (!(_isLoading && bp.userLibrary.isEmpty) && displayBooks.isEmpty)
              SliverFillRemaining(
                hasScrollBody: false,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.library_books_outlined, size: 64, color: cs.onSurfaceVariant.withValues(alpha: 0.2)),
                      const SizedBox(height: 16),
                      Text('No books in this section yet.', style: TextStyle(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
      bottomNavigationBar: const AppBottomNavBar(currentIndex: 2),
    );
  }

  Widget _buildCategorySelector(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final categories = [
      {'key': 'all', 'icon': Icons.grid_view_rounded, 'label': 'All', 'color': Colors.deepPurple},
      {'key': 'reading', 'icon': Icons.menu_book, 'label': 'Reading', 'color': Colors.blue},
      {'key': 'favorite', 'icon': Icons.favorite, 'label': 'Favorites', 'color': Colors.red},
      {'key': 'finished', 'icon': Icons.check_circle, 'label': 'Finished', 'color': Colors.green},
    ];

    return Container(
      height: 100,
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: categories.length,
        separatorBuilder: (_, __) => const SizedBox(width: 12),
        itemBuilder: (context, i) {
          final cat = categories[i];
          final isActive = _selectedStatus == cat['key'];
          final color = cat['color'] as Color;

          return GestureDetector(
            onTap: () => setState(() => _selectedStatus = cat['key'] as String),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 85,
              decoration: BoxDecoration(
                color: isActive ? color.withValues(alpha: 0.1) : cs.surfaceContainerLow,
                borderRadius: BorderRadius.circular(24),
                border: isActive ? Border.all(color: color.withValues(alpha: 0.5), width: 2) : null,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(cat['icon'] as IconData, color: isActive ? color : cs.onSurfaceVariant, size: 28),
                  const SizedBox(height: 8),
                  Text(
                    cat['label'] as String,
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                      color: isActive ? color : cs.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildStatsBento(BuildContext context, Map<String, dynamic> stats, int libraryCount) {
    final cs = Theme.of(context).colorScheme;
    
    final int finishedCount = stats['books_finished'] ?? 0;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.green.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: Colors.green.withValues(alpha: 0.2)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.check_circle_outline, color: Colors.green, size: 28),
                      const SizedBox(height: 16),
                      Text('$finishedCount', style: const TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 28, fontWeight: FontWeight.bold)),
                      Text('${stats['total_pages_read'] ?? 0} Pages', style: TextStyle(color: Colors.green.withValues(alpha: 0.7), fontSize: 10, fontWeight: FontWeight.bold)),
                      const Text('Completed', style: TextStyle(color: Colors.green, fontSize: 12, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: cs.primaryContainer,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.library_books, color: cs.primary, size: 28),
                      const SizedBox(height: 16),
                      Text('$libraryCount', style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 28, fontWeight: FontWeight.bold, color: cs.onPrimaryContainer)),
                      Text('In Library', style: TextStyle(color: cs.primary, fontSize: 12, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildReadingNow(BuildContext context, Book book) {
    final cs = Theme.of(context).colorScheme;
    final progress = book.readingProgress;
    
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 32, 20, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Reading Now', style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 24, fontWeight: FontWeight.bold, color: cs.onSurface)),
          const SizedBox(height: 16),
          GestureDetector(
            onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book))),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: cs.surfaceContainerLowest,
                borderRadius: BorderRadius.circular(24),
                boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.03), blurRadius: 20, offset: const Offset(0, 4))],
              ),
              child: Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: CachedNetworkImage(
                      imageUrl: book.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w200' : book.coverUrl)}' : 'https://placehold.co/100x150',
                      width: 80, height: 112, fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(book.title, style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 18, fontWeight: FontWeight.bold, color: cs.onSurface), maxLines: 1, overflow: TextOverflow.ellipsis),
                        const SizedBox(height: 4),
                        Text(book.author, style: TextStyle(fontSize: 13, color: cs.tertiary), maxLines: 1, overflow: TextOverflow.ellipsis),
                        const SizedBox(height: 12),
                        // Progress bar
                        Container(
                          width: double.infinity,
                          height: 8,
                          decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(4)),
                          child: FractionallySizedBox(
                            alignment: Alignment.centerRight, // RTL
                            widthFactor: (progress / 100).clamp(0.0, 1.0),
                            child: Container(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(4),
                                gradient: LinearGradient(colors: [cs.tertiaryContainer, cs.secondary]),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text('$progress% Completed', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
                            Text('${book.pageCount > 0 ? (book.pageCount * (1 - progress/100)).toInt() : "--"} pages left', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
                          ],
                        )
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCollections(BuildContext context, List<Book> readBooks) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(top: 32, bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('My Collections', style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 24, fontWeight: FontWeight.bold, color: cs.onSurface)),
                TextButton(onPressed: () {}, child: Text('Show All', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: cs.primary))),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 150,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 20),
              children: [
                _buildCollectionCard(context, 'Historical Novels', 'history_edu', cs.primary, '8 books'),
                const SizedBox(width: 16),
                _buildCollectionCard(context, 'Science Fiction', 'rocket_launch', cs.secondary, '15 books'),
                const SizedBox(width: 16),
                // Add new collection
                Container(
                  width: 140,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: cs.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.3), width: 1.5, style: BorderStyle.solid),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: 48, height: 48,
                        decoration: BoxDecoration(color: cs.surfaceContainerHighest, shape: BoxShape.circle),
                        child: Icon(Icons.add, color: cs.onSurfaceVariant),
                      ),
                      const SizedBox(height: 12),
                      Text('New Collection', style: TextStyle(fontWeight: FontWeight.bold, color: cs.onSurfaceVariant, fontSize: 13)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCollectionCard(BuildContext context, String title, String iconId, Color brandColor, String subtitle) {
    final cs = Theme.of(context).colorScheme;
    IconData icon = Icons.history_edu;
    if (iconId == 'rocket_launch') icon = Icons.rocket_launch;

    return Container(
      width: 140,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 56, height: 56,
            decoration: BoxDecoration(color: brandColor.withValues(alpha: 0.1), shape: BoxShape.circle),
            child: Icon(icon, color: brandColor, size: 28),
          ),
          const SizedBox(height: 12),
          Text(title, style: TextStyle(fontWeight: FontWeight.bold, color: cs.onSurface, fontSize: 14)),
          const SizedBox(height: 4),
          Text(subtitle, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 11)),
        ],
      ),
    );
  }

  Widget _buildWishlist(BuildContext context, List<Book> wishlist) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 32, 20, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Wishlist', style: TextStyle(fontFamily: 'Be Vietnam Pro', fontSize: 24, fontWeight: FontWeight.bold, color: cs.onSurface)),
          const SizedBox(height: 16),
          ...wishlist.map((book) => Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: GestureDetector(
              onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book))),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: cs.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: CachedNetworkImage(
                        imageUrl: book.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w100' : book.coverUrl)}' : 'https://placehold.co/60x60',
                        width: 60, height: 60, fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(book.title, style: TextStyle(fontWeight: FontWeight.bold, color: cs.onSurface, fontSize: 14), maxLines: 1, overflow: TextOverflow.ellipsis),
                          const SizedBox(height: 4),
                          Text(book.author, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
                        ],
                      ),
                    ),
                    Container(
                      width: 40, height: 40,
                      decoration: BoxDecoration(color: cs.surfaceContainerHighest, shape: BoxShape.circle),
                      child: Icon(Icons.favorite, color: cs.primary, size: 20),
                    )
                  ],
                ),
              ),
            ),
          )),
        ],
      ),
    );
  }
}
