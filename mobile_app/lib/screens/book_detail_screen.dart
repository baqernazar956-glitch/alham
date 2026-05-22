import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:share_plus/share_plus.dart';
import 'package:url_launcher/url_launcher.dart';
import '../models/book.dart';
import '../models/review.dart';
import '../services/books_service.dart';
import '../services/user_service.dart';
import '../services/ai_service.dart';
// All recommendations now handled by server's unified pipeline
import 'package:provider/provider.dart';
import '../providers/books_provider.dart';
import 'widgets/star_rating.dart';
import 'widgets/review_card.dart';
import 'assistant_screen.dart';
import '../config/translations.dart';

class BookDetailScreen extends StatefulWidget {
  final Book book;
  const BookDetailScreen({Key? key, required this.book}) : super(key: key);

  @override
  State<BookDetailScreen> createState() => _BookDetailScreenState();
}

class _BookDetailScreenState extends State<BookDetailScreen> {
  Book? _fullBook;
  List<Review> _reviews = [];
  List<Book> _similarBooks = [];
  String? _currentStatus;
  String _noteText = '';
  bool _loadingReviews = true;
  late TextEditingController _noteController;

  @override
  void initState() {
    super.initState();
    _noteController = TextEditingController();
    _loadData();
  }

  @override
  void dispose() {
    _noteController.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    final gid = widget.book.uniqueId;
    if (gid.isEmpty) return;

    // Load full details, reviews, status, note in parallel
    final futures = await Future.wait([
      BooksService.getBookDetail(gid),
      BooksService.getBookReviews(gid),
      UserService.getBookStatus(gid),
      UserService.getNote(gid),
    ]);

    if (!mounted) return;
    setState(() {
      _fullBook = (futures[0] as Book?) ?? widget.book;
      _reviews = futures[1] as List<Review>;
      final statusData = futures[2] as Map<String, dynamic>?;
      _currentStatus = statusData?['status'];
      _noteText = futures[3] as String;
      _noteController.text = _noteText;
      _loadingReviews = false;
    });

    // Load similar books
    final similar = await BooksService.getRecommendByBook(widget.book.title, limit: 10);
    if (mounted) setState(() { _similarBooks = similar; });

    // Log view with categories for behavior tracking
    UserService.logBookView(gid);
    BooksService.logEvent('view', gid, metadata: {
      'title': widget.book.title,
      'categories': widget.book.categories,
      'author': widget.book.author,
    });

    // Log view event to server — triggers unified pipeline online learning
    BooksService.logEvent('view', widget.book.gid ?? '');
  }

  Book get book => _fullBook ?? widget.book;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          // ─── Hero App Bar ───
          SliverAppBar(
            expandedHeight: 420,
            pinned: true,
            stretch: true,
            backgroundColor: cs.primary,
            leading: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white.withValues(alpha: 0.2), width: 1),
                ),
                child: IconButton(
                  icon: Icon(context.isRtl ? Icons.arrow_forward : Icons.arrow_back, color: Colors.white, size: 20),
                  onPressed: () => Navigator.pop(context),
                  padding: EdgeInsets.zero,
                ),
              ),
            ),
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  if (book.coverUrl.isNotEmpty)
                    CachedNetworkImage(
                      imageUrl: 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w800' : book.coverUrl)}',
                      fit: BoxFit.cover,
                      color: Colors.black.withValues(alpha: 0.3),
                      colorBlendMode: BlendMode.darken,
                    ),
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [Colors.transparent, cs.primary.withValues(alpha: 0.95)],
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 24, left: 24, right: 24,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        // Cover
                        Hero(
                          tag: 'book-${book.uniqueId}',
                          child: Container(
                            width: 120, height: 180,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(24),
                              boxShadow: [BoxShadow(color: Colors.black38, blurRadius: 20, offset: Offset(0, 8))],
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(24),
                              child: book.coverUrl.isNotEmpty
                                  ? CachedNetworkImage(
                                      imageUrl: 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w600' : book.coverUrl)}', 
                                      fit: BoxFit.cover
                                    )
                                  : Container(color: cs.primaryContainer, child: Icon(Icons.book, size: 50, color: cs.onPrimaryContainer)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 16),
                        // Title & Author
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (book.categories.isNotEmpty)
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.white24,
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: Text(book.categories.first, style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold)),
                                ),
                              const SizedBox(height: 8),
                              Text(book.title, style: tt.headlineSmall?.copyWith(color: Colors.white, fontWeight: FontWeight.w900), maxLines: 3, overflow: TextOverflow.ellipsis),
                              const SizedBox(height: 4),
                              Text(book.author, style: tt.bodyLarge?.copyWith(color: Colors.white70, fontStyle: FontStyle.italic)),
                              const SizedBox(height: 12),
                              Row(
                                children: [
                                  const Icon(Icons.star_rounded, color: Colors.amber, size: 20),
                                  const SizedBox(width: 4),
                                  Text(
                                    book.averageRating > 0 ? book.averageRating.toStringAsFixed(1) : context.t('new_rating'),
                                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
                                  ),
                                  if (book.ratingsCount > 0) ...[
                                    const SizedBox(width: 8),
                                    Text(
                                      '(${book.ratingsCount} ${context.t('reviews')})',
                                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                                    ),
                                  ],
                                ],
                              ),
                              const SizedBox(height: 12),
                              Wrap(
                                spacing: 12,
                                runSpacing: 8,
                                children: [
                                  if (book.pageCount > 0)
                                    Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const Icon(Icons.menu_book, color: Colors.white70, size: 14),
                                        const SizedBox(width: 4),
                                        Text('${book.pageCount} ${context.t('pages')}', style: const TextStyle(color: Colors.white70, fontSize: 12)),
                                      ],
                                    ),
                                  if (book.publishedDate.isNotEmpty)
                                    Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const Icon(Icons.calendar_today, color: Colors.white70, size: 14),
                                        const SizedBox(width: 4),
                                        Text(book.publishedDate.length >= 4 ? book.publishedDate.substring(0, 4) : book.publishedDate, style: const TextStyle(color: Colors.white70, fontSize: 12)),
                                      ],
                                    ),
                                  if (book.language.isNotEmpty)
                                    Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const Icon(Icons.language, color: Colors.white70, size: 14),
                                        const SizedBox(width: 4),
                                        Text(book.language.toUpperCase(), style: const TextStyle(color: Colors.white70, fontSize: 12)),
                                      ],
                                    ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white.withValues(alpha: 0.2), width: 1),
                  ),
                  child: IconButton(
                    icon: const Icon(Icons.share, color: Colors.white, size: 20),
                    onPressed: _shareBook,
                    padding: EdgeInsets.zero,
                  ),
                ),
              ),
            ],
          ),

          // ─── Body ───
          SliverToBoxAdapter(child: _buildBody(context)),
        ],
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─── Library Actions ───
          _buildLibraryActions(context),
          const SizedBox(height: 24),

          // ─── Read Now Button (New) ───
          _buildReadButton(context),

          // ─── Synopsis ───
          Text(context.t('synopsis_title'), style: tt.labelSmall?.copyWith(color: cs.primary, fontWeight: FontWeight.w900, letterSpacing: 2)),
          const SizedBox(height: 8),
          Text(
            book.description.isNotEmpty ? book.description : context.t('no_synopsis'),
            style: tt.bodyLarge?.copyWith(height: 1.7, color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: 32),

          // ─── Book Details ───
          _buildAdditionalDetails(context),
          const SizedBox(height: 32),

          // ─── AI Features ───
          _buildAISection(context),
          const SizedBox(height: 32),

          // ─── Reviews ───
          _buildReviewsSection(context),
          const SizedBox(height: 32),

          // ─── Notes ───
          _buildNotesSection(context),
          const SizedBox(height: 32),

          // ─── Similar Books ───
          if (_similarBooks.isNotEmpty) _buildSimilarBooks(context),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildAdditionalDetails(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(context.t('book_details'), style: tt.labelSmall?.copyWith(color: cs.primary, fontWeight: FontWeight.w900, letterSpacing: 2)),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: cs.surfaceContainerLowest,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.5)),
          ),
          child: Column(
            children: [
              _detailRow(context.t('author_label'), book.author),
              if (book.publisher.isNotEmpty) ...[
                const Divider(height: 24),
                _detailRow(context.t('publisher_label'), book.publisher),
              ],
              if (book.publishedDate.isNotEmpty) ...[
                const Divider(height: 24),
                _detailRow(context.t('publication_date'), book.publishedDate),
              ],
              if (book.pageCount > 0) ...[
                const Divider(height: 24),
                _detailRow(context.t('pages'), '${book.pageCount} ${context.t('pages')}'),
              ],
              if (book.language.isNotEmpty) ...[
                const Divider(height: 24),
                _detailRow(context.t('language'), book.language.toUpperCase()),
              ],
              if (book.categories.isNotEmpty) ...[
                const Divider(height: 24),
                _detailRow(context.t('categories_label'), book.categories.join(', ')),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _detailRow(String title, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          flex: 2,
          child: Text(
            title,
            style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant, fontSize: 13, fontWeight: FontWeight.bold),
          ),
        ),
        Expanded(
          flex: 3,
          child: Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            textAlign: context.isRtl ? TextAlign.left : TextAlign.right,
          ),
        ),
      ],
    );
  }

  Widget _buildReadButton(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    
    String? readingUrl = book.previewLink;
    if (readingUrl == null || readingUrl.isEmpty) {
      readingUrl = book.infoLink;
    }
    
    String label = context.t('read_now');
    IconData icon = Icons.menu_book_rounded;

    // Fallback for Gutenberg if link is missing
    if ((readingUrl == null || readingUrl.isEmpty) && book.uniqueId.startsWith('gut_')) {
      final cleanId = book.uniqueId.replaceFirst('gut_', '');
      readingUrl = 'https://www.gutenberg.org/ebooks/$cleanId';
    }
    
    // Fallback for Archive.org if link is missing
    if ((readingUrl == null || readingUrl.isEmpty) && book.uniqueId.startsWith('arch_')) {
      final cleanId = book.uniqueId.replaceFirst('arch_', '');
      readingUrl = 'https://archive.org/details/$cleanId';
    }

    // Direct Google Books link fallback using gid (matching the web project behavior)
    if ((readingUrl == null || readingUrl.isEmpty) && book.gid != null && book.gid!.isNotEmpty && !book.gid!.startsWith('gut_') && !book.gid!.startsWith('arch_')) {
      readingUrl = 'https://books.google.com/books?id=${book.gid}';
    }

    // If still no URL, fallback to Google Books search for the book title
    if (readingUrl == null || readingUrl.isEmpty) {
      readingUrl = 'https://www.google.com/search?tbm=bks&q=${Uri.encodeComponent(book.title + ' ' + book.author)}';
      label = context.t('find_on_google_books');
      icon = Icons.search;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 32),
      child: Container(
        width: double.infinity,
        height: 60,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          gradient: LinearGradient(
            colors: [cs.primary, cs.secondary],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight,
          ),
          boxShadow: [
            BoxShadow(
              color: cs.primary.withValues(alpha: 0.3),
              blurRadius: 12,
              offset: const Offset(0, 6),
            )
          ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: () => _launchBookUrl(readingUrl!),
            borderRadius: BorderRadius.circular(20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, color: Colors.white, size: 24),
                const SizedBox(width: 12),
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMetaRow(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final items = <MapEntry<String, String>>[
      if (book.pageCount > 0) MapEntry(context.t('pages'), '${book.pageCount}'),
      if (book.publishedDate.isNotEmpty) MapEntry(context.t('year'), book.publishedDate.length >= 4 ? book.publishedDate.substring(0, 4) : book.publishedDate),
      if (book.language.isNotEmpty) MapEntry(context.t('language'), book.language.toUpperCase()),
      if (book.pageCount > 0) MapEntry(context.t('read_time'), '~${(book.pageCount / 25).round()}${context.t('hours_short')}'),
    ];

    return Row(
      children: items.map((e) => Expanded(
        child: Column(
          children: [
            Text(e.key, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w900, color: cs.onSurfaceVariant.withValues(alpha: 0.4), letterSpacing: 1.5)),
            const SizedBox(height: 4),
            Text(e.value, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w900)),
          ],
        ),
      )).toList(),
    );
  }

  Widget _buildLibraryActions(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final statuses = [
      {'key': 'favorite', 'icon': Icons.favorite, 'label': 'Favorite', 'color': Colors.red},
      {'key': 'reading', 'icon': Icons.auto_stories, 'label': 'Reading', 'color': Colors.blue},
      {'key': 'later', 'icon': Icons.bookmark, 'label': 'Later', 'color': Colors.amber},
      {'key': 'finished', 'icon': Icons.check_circle, 'label': 'Finished', 'color': Colors.green},
    ];

    return Row(
      children: statuses.map((s) {
        final isActive = _currentStatus == s['key'];
        final color = s['color'] as Color;
        return Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Material(
              color: isActive ? color.withValues(alpha: 0.15) : cs.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(24),
              child: InkWell(
                borderRadius: BorderRadius.circular(24),
                onTap: () => _toggleStatus(s['key'] as String),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Column(
                    children: [
                      Icon(s['icon'] as IconData, size: 22, color: isActive ? color : cs.onSurfaceVariant),
                      const SizedBox(height: 4),
                      Text(context.t('library_status_${s['key']}'), style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: isActive ? color : cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildAISection(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [
          Icon(Icons.psychology, color: cs.primary, size: 20),
          const SizedBox(width: 8),
          Text(context.t('interactive_features'), style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        ]),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(child: _aiButton(context, Icons.chat_bubble_outline, context.t('chat'), context.t('ask_about_book'), cs.primary, _openChat)),
            const SizedBox(width: 8),
            Expanded(child: _aiButton(context, Icons.summarize_outlined, context.t('summary'), context.t('ai_generated_summary'), cs.secondary, _generateSummary)),
          ],
        ),
      ],
    );
  }

  Widget _aiButton(BuildContext context, IconData icon, String title, String subtitle, Color color, VoidCallback onTap) {
    return Material(
      color: color.withValues(alpha: 0.1),
      borderRadius: BorderRadius.circular(24),
      child: InkWell(
        borderRadius: BorderRadius.circular(24),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(title, style: TextStyle(fontWeight: FontWeight.bold, color: color)),
                  Text(subtitle, style: TextStyle(fontSize: 11, color: color.withValues(alpha: 0.7))),
                ]),
              ),
              Icon(icon, color: color),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildReviewsSection(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(context.t('community_reviews'), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w900, letterSpacing: 2, color: cs.tertiary)),
            if (_reviews.isNotEmpty) Text('${_reviews.length} ${context.t('reviews')}', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
          ],
        ),
        const SizedBox(height: 12),
        if (_loadingReviews)
          const Center(child: CircularProgressIndicator())
        else if (_reviews.isEmpty)
          _emptyState(context, Icons.forum_outlined, context.t('no_reviews_yet'), context.t('be_first_review'))
        else
          ...List.generate(
            _reviews.length > 3 ? 3 : _reviews.length,
            (i) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: ReviewCard(
                review: _reviews[i],
              ),
            ),
          ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: FilledButton.tonal(
            onPressed: _showWriteReviewSheet,
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(48)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [const Icon(Icons.edit_note, size: 20), const SizedBox(width: 8), Text(context.t('write_review'), style: const TextStyle(fontWeight: FontWeight.bold))],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildNotesSection(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(context.t('personal_notes'), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w900, letterSpacing: 2, color: cs.secondary)),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
            color: cs.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(24),
          ),
          child: TextField(
            controller: _noteController,
            maxLines: 4,
            decoration: InputDecoration(
              hintText: context.t('write_notes_placeholder'),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.all(16),
              suffixIcon: IconButton(
                icon: Icon(Icons.save_outlined, color: cs.primary),
                onPressed: () => _saveNote(),
              ),
            ),
            onChanged: (v) => _noteText = v,
          ),
        ),
      ],
    );
  }

  Widget _buildSimilarBooks(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(context.t('similar_books'), style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        SizedBox(
          height: 200,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: _similarBooks.length,
            separatorBuilder: (_, __) => const SizedBox(width: 12),
            itemBuilder: (context, i) {
              final b = _similarBooks[i];
              return GestureDetector(
                onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: b))),
                child: SizedBox(
                  width: 120,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: CachedNetworkImage(
                            imageUrl: b.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(b.coverUrl.contains('?') ? '${b.coverUrl}&fife=w400' : b.coverUrl)}' : 'https://placehold.co/200x300/f8fafc/ea580c?text=Book',
                            fit: BoxFit.cover, width: 120,
                          ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(b.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
                      Text(b.author, maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant)),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _emptyState(BuildContext context, IconData icon, String title, String subtitle) {
    return Container(
      padding: const EdgeInsets.all(32),
      width: double.infinity,
      child: Column(
        children: [
          Icon(icon, size: 48, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.3)),
          const SizedBox(height: 8),
          Text(title, style: TextStyle(fontWeight: FontWeight.bold, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.5))),
          Text(subtitle, style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.3))),
        ],
      ),
    );
  }

  // ─── Actions ───
  void _toggleStatus(String status) async {
    final gid = book.uniqueId;
    if (_currentStatus == status) {
      await UserService.removeFromLibrary(gid);
      setState(() => _currentStatus = null);
    } else {
      await UserService.addToLibrary(gid, status, book: book);
      setState(() => _currentStatus = status);
      // Track library addition for behavior-based recommendations
      BooksService.logEvent('add_to_library', gid, metadata: {
        'title': book.title,
        'categories': book.categories,
        'author': book.author,
        'status': status,
      });
    }
    
    // Refresh recommendations silently in the background so HomeScreen reflects this change
    if (mounted) {
      try {
        Provider.of<BooksProvider>(context, listen: false).fetchPersonalizedRecommendations();
      } catch (_) {}
    }
  }

  Future<void> _launchBookUrl(String url) async {
    final uri = Uri.parse(url);
    try {
      if (await canLaunchUrl(uri)) {
        await launchUrl(uri, mode: LaunchMode.externalApplication);
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(context.t('could_not_open_preview')), behavior: SnackBarBehavior.floating),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e'), behavior: SnackBarBehavior.floating),
        );
      }
    }
  }

  void _shareBook() {
    final template = context.t('share_book_text');
    final shareText = template.replaceAll('{title}', book.title).replaceAll('{author}', book.author);
    Share.share(shareText);
  }

  void _saveNote() async {
    final ok = await UserService.saveNote(book.uniqueId, _noteController.text);
    if (mounted) {
      final msg = ok ? context.t('note_saved') : context.t('failed_to_save_note');
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg), behavior: SnackBarBehavior.floating));
    }
  }

  void _openChat() {
    Navigator.push(context, MaterialPageRoute(builder: (_) => AssistantScreen(book: book)));
  }

  void _generateSummary() async {
    showDialog(context: context, barrierDismissible: false, builder: (_) => const Center(child: CircularProgressIndicator()));
    final result = await AiService.getSummary(
      book.uniqueId,
      bookTitle: book.title,
      bookAuthor: book.author,
      description: book.description,
    );
    if (mounted) Navigator.pop(context);
    if (mounted) {
      showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
        builder: (_) => DraggableScrollableSheet(
          expand: false, initialChildSize: 0.6, maxChildSize: 0.9,
          builder: (_, controller) => ListView(
            controller: controller,
            padding: const EdgeInsets.all(24),
            children: [
              Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2)))),
              const SizedBox(height: 16),
              Text(context.t('ai_summary'), style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
              Text(result['summary'] ?? result['error'] ?? context.t('no_summary_available'), style: Theme.of(context).textTheme.bodyLarge?.copyWith(height: 1.7)),
            ],
          ),
        ),
      );
    }
  }

  void _showWriteReviewSheet() {
    int selectedRating = 0;
    final textController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheetState) => Padding(
          padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom, left: 24, right: 24, top: 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2)))),
              const SizedBox(height: 16),
              Text(context.t('your_review'), style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 12, letterSpacing: 2)),
              const SizedBox(height: 16),
              StarRating(rating: selectedRating, size: 40, onRatingChanged: (r) => setSheetState(() => selectedRating = r)),
              const SizedBox(height: 16),
              TextField(
                controller: textController,
                maxLines: 4,
                decoration: InputDecoration(
                  hintText: context.t('share_thoughts_placeholder'),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(20)),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: selectedRating == 0 ? null : () async {
                    Navigator.pop(ctx);
                    final ok = await BooksService.submitReview(book.uniqueId, selectedRating, textController.text);
                    if (ok) {
                      final reviews = await BooksService.getBookReviews(book.uniqueId);
                      if (mounted) setState(() => _reviews = reviews);
                      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(context.t('review_submitted')), behavior: SnackBarBehavior.floating));
                      
                      // Refresh recommendations silently in the background so HomeScreen reflects this change
                      if (mounted) {
                        try {
                          Provider.of<BooksProvider>(context, listen: false).fetchPersonalizedRecommendations();
                        } catch (_) {}
                      }
                    }
                  },
                  style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  child: Text(context.t('submit_review'), style: const TextStyle(fontWeight: FontWeight.bold)),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}
