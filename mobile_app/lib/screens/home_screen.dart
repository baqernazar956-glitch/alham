import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../providers/books_provider.dart';
import '../models/book.dart';
import '../services/books_service.dart';
import '../services/user_service.dart';
import 'book_detail_screen.dart';
import 'assistant_screen.dart';
import 'widgets/bottom_nav_bar.dart';
import 'widgets/book_card.dart';
import '../models/user.dart';
import 'public_library_screen.dart';
import '../config/translations.dart';
import '../config/app_config.dart';
import '../providers/auth_provider.dart';
import 'profile_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver, RouteAware {
  List<Book> _moodBooks = [];
  bool _loadingMood = false;
  bool _showAllRecommendations = false;
  bool _initialLoadDone = false;
  int _trendingLimit = 6;
  int _communityPulseLimit = 6;
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadAllData();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _searchController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Refresh recommendations when app resumes from background
    if (state == AppLifecycleState.resumed && _initialLoadDone) {
      final bp = Provider.of<BooksProvider>(context, listen: false);
      bp.fetchPersonalizedRecommendations();
    }
  }

  void _loadAllData() {
    final bp = Provider.of<BooksProvider>(context, listen: false);
    bp.loadHomeData();
    _initialLoadDone = true;
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final bp = Provider.of<BooksProvider>(context);
    final currentUser = Provider.of<AuthProvider>(context).currentUser;

    return Scaffold(
      body: RefreshIndicator(
        onRefresh: () async {
          await bp.fetchTrending();
          await bp.fetchTopRated();
          await bp.fetchPersonalizedRecommendations();
          await bp.fetchCuratedCollections();
          await bp.fetchBecauseYouSearched();
        },
        child: CustomScrollView(
          slivers: [
            // ─── App Bar ───
            SliverAppBar(
              floating: true, snap: true, pinned: true,
              backgroundColor: Theme.of(context).scaffoldBackgroundColor,
              elevation: 0,
              centerTitle: false,
              title: Row(
                children: [
                  GestureDetector(
                    onTap: () {
                      Navigator.pushReplacement(
                        context,
                        MaterialPageRoute(builder: (_) => const ProfileScreen()),
                      );
                    },
                    child: CircleAvatar(
                      radius: 22,
                      backgroundColor: cs.primary.withValues(alpha: 0.1),
                      backgroundImage: (currentUser?.profilePicture != null && currentUser!.profilePicture!.isNotEmpty)
                          ? CachedNetworkImageProvider(
                              '${AppConfig.serverBaseUrl}${currentUser.profilePicture}',
                            )
                          : null,
                      child: (currentUser?.profilePicture == null || currentUser!.profilePicture!.isEmpty)
                          ? Icon(Icons.person_outline, color: cs.primary, size: 24)
                          : null,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Elham', 
                      style: TextStyle(
                        fontWeight: FontWeight.w900, 
                        fontStyle: FontStyle.italic, 
                        color: cs.primary, 
                        fontSize: 26,
                        letterSpacing: -0.5
                      )
                    ),
                  ),
                ],
              ),
              actions: [
                IconButton(
                  icon: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: cs.primary.withValues(alpha: 0.1),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(Icons.psychology_outlined, color: cs.primary, size: 22),
                  ),
                  onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const AssistantScreen())),
                ),
                const SizedBox(width: 8),
              ],
              bottom: PreferredSize(
                preferredSize: const Size.fromHeight(70),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
                  child: Container(
                    decoration: BoxDecoration(
                      color: cs.surface,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: cs.primary.withValues(alpha: 0.05),
                          blurRadius: 15,
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: TextField(
                      controller: _searchController,
                      onSubmitted: (val) {
                        if (val.trim().isNotEmpty) {
                          UserService.saveSearchQuery(val.trim());
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => PublicLibraryScreen(initialSearch: val.trim()),
                            ),
                          );
                          _searchController.clear();
                        }
                      },
                      decoration: InputDecoration(
                        hintText: context.t('search_hint'),
                        hintStyle: TextStyle(color: cs.onSurface.withValues(alpha: 0.4), fontSize: 15),
                        prefixIcon: Icon(Icons.search_rounded, color: cs.primary, size: 22),
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                      ),
                    ),
                  ),
                ),
              ),
            ),

            // ─── Hero Quote ───
            SliverToBoxAdapter(key: const ValueKey('hero-quote'), child: _buildHeroQuote(context)),

            // ─── Featured Book (Editor's Pick) ───
            if (bp.trendingBooks.isNotEmpty)
              SliverToBoxAdapter(key: const ValueKey('featured-book'), child: _buildFeaturedBook(context, bp.trendingBooks.first)),

            // ─── Recommended For You (AI) ───
            if (bp.personalizedSections.isNotEmpty) ...[
              for (int i = 0; i < bp.personalizedSections.length; i++) ...[
                if (!bp.personalizedSections[i]['title'].toString().toLowerCase().contains('search')) ...[
                  SliverToBoxAdapter(
                    key: ValueKey('rec-header-$i'),
                    child: _sectionHeader(
                      context, 
                      '', 
                      context.t(bp.personalizedSections[i]['title'] ?? 'Recommended for You'), 
                      cs.secondary,
                      icon: null,
                      trailing: (bp.personalizedSections[i]['books'] as List).length > 6 ? TextButton(
                        onPressed: () {
                          setState(() {
                            _showAllRecommendations = !_showAllRecommendations;
                          });
                        },
                        child: Text(_showAllRecommendations ? context.t('show_less') : context.t('show_all')),
                      ) : null,
                    ),
                  ),
                  SliverToBoxAdapter(
                    key: ValueKey('rec-list-$i'),
                    child: _buildHorizontalBookList(context, [bp.personalizedSections[i]], showAll: _showAllRecommendations)
                  ),
                ]
              ]
            ],

            // ─── Community Pulse (Most Viewed) ───
            if (bp.trendingBooks.length > 1) ...[
              SliverToBoxAdapter(
                key: const ValueKey('pulse-header'),
                child: _sectionHeader(
                  context, 
                  '', 
                  context.t('community_pulse'), 
                  cs.onSurface,
                  trailing: bp.trendingBooks.length > 6 ? TextButton(
                    onPressed: () {
                      setState(() {
                        if (_communityPulseLimit >= bp.trendingBooks.length || _communityPulseLimit >= 15) {
                          _communityPulseLimit = 6;
                        } else {
                          _communityPulseLimit = 15;
                        }
                      });
                    },
                    child: Text((_communityPulseLimit >= bp.trendingBooks.length || _communityPulseLimit >= 15) ? context.t('show_less') : context.t('show_all')),
                  ) : null,
                ),
              ),
              SliverToBoxAdapter(
                key: const ValueKey('pulse-list'),
                child: _buildCommunityPulse(context, bp.trendingBooks.skip(1).take(_communityPulseLimit).toList())
              ),
            ],

            // ─── Trending Now (Top Rated) ───
            if (bp.topRatedBooks.isNotEmpty) ...[
              SliverToBoxAdapter(
                child: _sectionHeader(
                  context, 
                  '', 
                  context.t('trending_now'), 
                  cs.secondary,
                  icon: null,
                  trailing: bp.topRatedBooks.length > 6 ? TextButton(
                    onPressed: () {
                      setState(() {
                        if (_trendingLimit >= bp.topRatedBooks.length || _trendingLimit >= 100) {
                          _trendingLimit = 6;
                        } else {
                          _trendingLimit = 100;
                        }
                      });
                    },
                    child: Text((_trendingLimit >= bp.topRatedBooks.length || _trendingLimit >= 100) ? context.t('show_less') : context.t('show_all')),
                  ) : null,
                ),
              ),
              SliverToBoxAdapter(child: _buildTrendingNow(context, bp.topRatedBooks.take(_trendingLimit).toList())),
            ],


            // ─── Quote Block ───
            SliverToBoxAdapter(child: _buildQuoteBlock(context)),

            // ─── Because You Searched (Custom UI) ───
            if (bp.searchHistoryBooks.isNotEmpty || bp.personalizedSections.any((s) => s['title'].toString().contains('search'))) ...[
              const SliverToBoxAdapter(child: SizedBox(height: 32)),
              SliverToBoxAdapter(
                child: Builder(
                  builder: (context) {
                    // Try to find it in personalized sections first (from server)
                    final serverSectionIdx = bp.personalizedSections.indexWhere((s) => s['title'].toString().contains('search'));
                    
                    String query = bp.lastSearchTerm;
                    List<Book> books = bp.searchHistoryBooks;

                    if (serverSectionIdx != -1) {
                      final section = bp.personalizedSections[serverSectionIdx];
                      final fullTitle = section['title'].toString();
                      // Extract query from title like "🔍 Because you searched for 'Python'"
                      final match = RegExp(r'[«"“](.+)[»"”]').firstMatch(fullTitle);
                      if (match != null) {
                        query = match.group(1)!;
                      } else {
                        query = fullTitle.split('for').last.replaceAll('"', '').trim();
                      }
                      books = (section['books'] as List).map((b) => Book.fromJson(b)).toList();
                    }

                    return _buildBecauseYouSearchedUI(context, query, books);
                  }
                ),
              ),
            ],

            // ─── Curated Collections (Book Series) ───
            if (bp.curatedCollections.isNotEmpty) ...[
              SliverToBoxAdapter(child: _sectionHeader(context, '', context.t('curated_collections'), cs.onSurface, icon: null)),
              SliverToBoxAdapter(child: _buildCuratedCollections(context, bp.curatedCollections)),
            ],

            // ─── Mood Recommendations ───
            SliverToBoxAdapter(child: _buildMoodSelector(context)),
            if (_moodBooks.isNotEmpty)
              SliverToBoxAdapter(child: _buildMoodResults(context)),

            const SliverToBoxAdapter(child: SizedBox(height: 100)),
          ],
        ),
      ),
      bottomNavigationBar: const AppBottomNavBar(currentIndex: 0),
    );
  }

  // ════════════════════════════════════════════════════════════
  // WIDGETS
  // ════════════════════════════════════════════════════════════

  Widget _buildHeroQuote(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 8, 24, 32),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(width: 48, height: 1, color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.15)),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: Icon(Icons.auto_stories, size: 28, color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.3)),
              ),
              Container(width: 48, height: 1, color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.15)),
            ],
          ),
          const SizedBox(height: 20),
          Text(
            context.t('hero_quote'),
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.displaySmall?.copyWith(fontWeight: FontWeight.w900, height: 1.2, letterSpacing: -1),
          ),
        ],
      ),
    );
  }

  Widget _buildFeaturedBook(BuildContext context, Book book) {
    final cs = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book))),
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 16),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: cs.primary,
          borderRadius: BorderRadius.circular(32),
          boxShadow: [BoxShadow(color: cs.primary.withValues(alpha: 0.3), blurRadius: 24, offset: const Offset(0, 12))],
        ),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: CachedNetworkImage(imageUrl: book.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w400' : book.coverUrl)}' : 'https://placehold.co/200x300', width: 100, height: 150, fit: BoxFit.cover),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(color: Colors.white24, borderRadius: BorderRadius.circular(20)),
                    child: Text(context.t('featured_read'), style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(height: 8),
                  Text(book.title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 20), maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Text(book.author, style: const TextStyle(color: Colors.white70, fontSize: 13)),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(20)),
                    child: Text(context.t('start_reading'), style: TextStyle(color: cs.primary, fontWeight: FontWeight.bold, fontSize: 12)),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sectionHeader(BuildContext context, String subtitle, String title, Color color, {String? icon, Widget? trailing}) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 32, 20, 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Subtitle removed for cleaner UI
                Row(
                  children: [
                    // Icon removed for cleaner UI
                    Expanded(child: Text(title, style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w900))),
                  ],
                ),
              ],
            ),
          ),
          if (trailing != null) trailing,
        ],
      ),
    );
  }

  Widget _buildHorizontalBookList(BuildContext context, List<dynamic> sections, {bool showAll = false}) {
    final allBooks = <Book>[];
    for (final section in sections) {
      if (section is Map && section['books'] != null) {
        for (final b in section['books']) {
          allBooks.add(Book.fromJson(b is Map<String, dynamic> ? b : {}));
        }
      }
    }
    if (allBooks.isEmpty) return const SizedBox.shrink();
    
    final limit = showAll ? allBooks.length : (allBooks.length > 6 ? 6 : allBooks.length);
    
    return SizedBox(
      height: 350,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: limit,
        separatorBuilder: (_, __) => const SizedBox(width: 14),
        itemBuilder: (context, i) {
          final book = allBooks[i];
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Algorithm tag badge removed for cleaner UI
              Expanded(child: BookCard(book: book, width: 140, height: 190)),
            ],
          );
        },
      ),
    );
  }



  Widget _buildBecauseYouSearchedUI(BuildContext context, String query, List<Book> books) {
    if (books.isEmpty) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFFF9F0E1), // Light beige background
        borderRadius: BorderRadius.circular(32),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                flex: 5,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      context.t('because_you_searched').replaceAll('{query}', query),
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFF6B4E2B),
                        height: 1.1,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      context.t('because_you_searched_desc'),
                      style: const TextStyle(
                        fontSize: 12,
                        color: Color(0xFF8B7355),
                        height: 1.4,
                      ),
                    ),
                    const SizedBox(height: 24),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => PublicLibraryScreen(initialSearch: query),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF6B4E2B),
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      child: Text(context.t('discover_more'), style: const TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                flex: 6,
                child: Column(
                  children: books.take(3).map((book) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: GestureDetector(
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(builder: (_) => BookDetailScreen(book: book)),
                        ),
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.05),
                                blurRadius: 10,
                                offset: const Offset(0, 4),
                              )
                            ],
                          ),
                          child: Row(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(10),
                                child: CachedNetworkImage(
                                  imageUrl: book.coverUrl.isNotEmpty 
                                      ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl)}&w=150' 
                                      : 'https://placehold.co/150x225',
                                  width: 60,
                                  height: 90,
                                  fit: BoxFit.cover,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      book.title,
                                      style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 13, color: Color(0xFF2D2D2D)),
                                      maxLines: 2,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      book.author,
                                      style: const TextStyle(color: Colors.grey, fontSize: 11),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                    const SizedBox(height: 10),
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFF3E5F5),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Text(
                                        '${(85 + (book.title.length % 15))}% ${context.t('match')}',
                                        style: const TextStyle(color: Color(0xFF8E24AA), fontSize: 8, fontWeight: FontWeight.bold),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCommunityPulse(BuildContext context, List<Book> books) {
    return SizedBox(
      height: 350,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: books.length,
        separatorBuilder: (_, __) => const SizedBox(width: 14),
        itemBuilder: (context, i) => BookCard(book: books[i], width: 140, height: 200),
      ),
    );
  }

  // ─── NEW: Trending Now (Top Rated cards with rank badge) ───
  Widget _buildTrendingNow(BuildContext context, List<Book> books) {
    final cs = Theme.of(context).colorScheme;
    return SizedBox(
      height: 350,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: books.length,
        separatorBuilder: (_, __) => const SizedBox(width: 14),
        itemBuilder: (context, i) {
          final book = books[i];
          return GestureDetector(
            onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book))),
            child: SizedBox(
              width: 200,
              child: Card(
                elevation: 2,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Stack(
                      children: [
                        ClipRRect(
                          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
                          child: CachedNetworkImage(imageUrl: book.coverUrl.isNotEmpty ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl.contains('?') ? '${book.coverUrl}&fife=w400' : book.coverUrl)}' : 'https://placehold.co/400x300', width: 200, height: 160, fit: BoxFit.cover),
                        ),
                        Positioned(
                          top: 0, left: 0,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                            decoration: BoxDecoration(color: cs.primary, borderRadius: const BorderRadius.only(topLeft: Radius.circular(20), bottomRight: Radius.circular(16))),
                            child: Text('#${i + 1}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 16)),
                          ),
                        ),
                        Positioned(
                          top: 8, right: 8,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(color: cs.secondary.withValues(alpha: 0.9), borderRadius: BorderRadius.circular(12)),
                            child: Text(context.t('top_rated'), style: const TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.w900, letterSpacing: 1)),
                          ),
                        ),
                      ],
                    ),
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(book.title, style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 14), maxLines: 2, overflow: TextOverflow.ellipsis),
                          const SizedBox(height: 4),
                          Text(book.author, style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Icon(Icons.star, size: 14, color: cs.tertiary),
                              const SizedBox(width: 4),
                              Text(book.averageRating > 0 ? book.averageRating.toStringAsFixed(1) : context.t('new_rating'), style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: cs.onSurface)),
                              if (book.ratingsCount > 0) ...[
                                const SizedBox(width: 4),
                                Text('(${book.ratingsCount})', style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
                              ],
                            ],
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ─── NEW: Curated Collections (Book Series) ───
  Widget _buildCuratedCollections(BuildContext context, List<Map<String, dynamic>> collections) {
    return SizedBox(
      height: 300,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        itemCount: collections.length,
        separatorBuilder: (_, __) => const SizedBox(width: 16),
        itemBuilder: (context, i) {
          final col = collections[i];
          final color = Color(col['color'] as int);
          final covers = (col['covers'] as List<dynamic>?) ?? [];
          final books = (col['books'] as List<dynamic>?) ?? [];
          final title = col['title'] as String? ?? '';
          final count = col['count'] as int? ?? 20;

          return GestureDetector(
            onTap: () {
              if (books.isNotEmpty) {
                final book = books.first as Book;
                Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: book)));
              }
            },
            child: Container(
              width: 220,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(28),
                boxShadow: [BoxShadow(color: color.withValues(alpha: 0.2), blurRadius: 20, offset: const Offset(0, 8))],
              ),
              child: Column(
                children: [
                  // Covers Stack
                  Container(
                    height: 170,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.15),
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
                    ),
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        if (covers.length >= 3)
                          Positioned(left: 24, child: Transform.rotate(angle: -0.2, child: ClipRRect(borderRadius: BorderRadius.circular(8), child: CachedNetworkImage(imageUrl: 'https://wsrv.nl/?url=${Uri.encodeComponent((covers[2] as String).contains('?') ? '${covers[2]}&fife=w200' : covers[2] as String)}', width: 60, height: 90, fit: BoxFit.cover)))),
                        if (covers.length >= 2)
                          Positioned(right: 24, child: Transform.rotate(angle: 0.15, child: ClipRRect(borderRadius: BorderRadius.circular(8), child: CachedNetworkImage(imageUrl: 'https://wsrv.nl/?url=${Uri.encodeComponent((covers[1] as String).contains('?') ? '${covers[1]}&fife=w200' : covers[1] as String)}', width: 65, height: 95, fit: BoxFit.cover)))),
                        if (covers.isNotEmpty)
                          ClipRRect(borderRadius: BorderRadius.circular(10), child: CachedNetworkImage(imageUrl: 'https://wsrv.nl/?url=${Uri.encodeComponent((covers[0] as String).contains('?') ? '${covers[0]}&fife=w200' : covers[0] as String)}', width: 75, height: 110, fit: BoxFit.cover)),
                      ],
                    ),
                  ),
                  // Info
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Container(width: 20, height: 3, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
                              const SizedBox(width: 8),
                              Text('$count+ ${context.t('volumes')}', style: TextStyle(fontSize: 9, fontWeight: FontWeight.w900, color: Colors.grey[500], letterSpacing: 1.5)),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Text(title, style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 16), maxLines: 1, overflow: TextOverflow.ellipsis),
                          const Spacer(),
                          Text(context.t('explore'), style: TextStyle(fontSize: 10, fontWeight: FontWeight.w900, color: Theme.of(context).colorScheme.primary, letterSpacing: 1.5)),
                        ],
                      ),
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

  Widget _buildQuoteBlock(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.all(20),
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: cs.surfaceContainerLowest, borderRadius: BorderRadius.circular(24)),
      child: Column(
        children: [
          Icon(Icons.format_quote, size: 36, color: cs.onSurface.withValues(alpha: 0.1)),
          const SizedBox(height: 12),
          Text(context.t('quote_manifesto'), textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontStyle: FontStyle.italic, fontWeight: FontWeight.bold, height: 1.5)),
          const SizedBox(height: 12),
          Container(width: 32, height: 3, decoration: BoxDecoration(color: cs.primary, borderRadius: BorderRadius.circular(2))),
          const SizedBox(height: 8),
          Text(context.t('quote_author'), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w900, letterSpacing: 1.5, color: cs.onSurfaceVariant)),
        ],
      ),
    );
  }

  Widget _buildMoodSelector(BuildContext context) {
    final moods = [
      {'key': 'happy', 'emoji': '😊', 'label': 'Happy'},
      {'key': 'adventurous', 'emoji': '🗺️', 'label': 'Adventure'},
      {'key': 'romantic', 'emoji': '💕', 'label': 'Romantic'},
      {'key': 'intellectual', 'emoji': '🧠', 'label': 'Intellectual'},
      {'key': 'mysterious', 'emoji': '🔮', 'label': 'Mysterious'},
      {'key': 'relaxed', 'emoji': '☕', 'label': 'Relaxed'},
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(context.t('mood_selector_title'), style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8, runSpacing: 8,
            children: moods.map((m) => ActionChip(
              avatar: Text(m['emoji'] as String, style: const TextStyle(fontSize: 16)),
              label: Text(context.t('mood_${m['key']}'), style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
              onPressed: () => _loadMoodBooks(m['key'] as String),
              side: BorderSide.none,
              backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            )).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildMoodResults(BuildContext context) {
    if (_loadingMood) return const Padding(padding: EdgeInsets.all(24), child: Center(child: CircularProgressIndicator()));
    return Padding(
      padding: const EdgeInsets.only(top: 16),
      child: SizedBox(
        height: 300,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 20),
          itemCount: _moodBooks.length,
          separatorBuilder: (_, __) => const SizedBox(width: 14),
          itemBuilder: (context, i) => BookCard(book: _moodBooks[i], width: 140, height: 200),
        ),
      ),
    );
  }

  void _loadMoodBooks(String mood) async {
    setState(() { _loadingMood = true; _moodBooks = []; });
    final books = await BooksService.getMoodRecommendations(mood, limit: 12);
    if (mounted) setState(() { _moodBooks = books; _loadingMood = false; });
  }
}
