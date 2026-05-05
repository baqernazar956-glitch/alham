import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'widgets/bottom_nav_bar.dart';
import 'widgets/book_card.dart';
import '../providers/books_provider.dart';
import '../services/books_service.dart';
import '../services/user_service.dart';
import '../models/book.dart';

class PublicLibraryScreen extends StatefulWidget {
  final String? initialSearch;
  const PublicLibraryScreen({Key? key, this.initialSearch}) : super(key: key);

  @override
  State<PublicLibraryScreen> createState() => _PublicLibraryScreenState();
}

class _PublicLibraryScreenState extends State<PublicLibraryScreen> {
  List<dynamic> _categories = [];
  List<Book> _books = [];
  bool _isLoading = false;
  bool _isLoadingMore = false;
  bool _hasError = false;
  int _currentPage = 1;
  bool _hasMore = true;
  String _selectedCategory = '';
  final TextEditingController _searchController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    
    if (widget.initialSearch != null && widget.initialSearch!.isNotEmpty) {
      _searchController.text = widget.initialSearch!;
      _loadCategories().then((_) {
        _searchBooks(widget.initialSearch!);
      });
    } else {
      _loadCategories();
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (_scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 200) {
      if (!_isLoading && !_isLoadingMore && _hasMore) {
        _loadMore();
      }
    }
  }

  Future<void> _loadCategories() async {
    setState(() {
      _isLoading = true;
      _hasError = false;
    });
    final cats = await BooksService.getCategories();
    if (mounted) {
      setState(() {
        _categories = cats;
        if (cats.isNotEmpty) {
          _selectedCategory = cats[0]['id'];
          _loadBooksByCategory(_selectedCategory);
        } else {
          _isLoading = false;
        }
      });
    }
  }

  Future<void> _loadBooksByCategory(String categoryId, {bool isLoadMore = false}) async {
    if (isLoadMore) {
      setState(() => _isLoadingMore = true);
    } else {
      setState(() {
        _isLoading = true;
        _hasError = false;
        _currentPage = 1;
        _books = [];
        _hasMore = true;
      });
    }

    try {
      final books = await BooksService.getBooksByCategory(categoryId, page: _currentPage);
      
      if (mounted) {
        setState(() {
          if (isLoadMore) {
            _books.addAll(books);
            _isLoadingMore = false;
          } else {
            _books = books;
            _isLoading = false;
          }
          _hasError = false;
          if (books.length < 10) {
            _hasMore = false;
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _isLoadingMore = false;
          _hasError = _books.isEmpty;
        });
      }
    }
  }

  Future<void> _loadMore() async {
    _currentPage++;
    if (_searchController.text.isNotEmpty) {
      _searchBooks(_searchController.text, isLoadMore: true);
    } else {
      _loadBooksByCategory(_selectedCategory, isLoadMore: true);
    }
  }

  Future<void> _searchBooks(String query, {bool isLoadMore = false}) async {
    if (query.trim().isEmpty) {
      if (_categories.isNotEmpty) {
        _loadBooksByCategory(_selectedCategory);
      }
      return;
    }

    if (isLoadMore) {
      setState(() => _isLoadingMore = true);
    } else {
      setState(() {
        _isLoading = true;
        _hasError = false;
        _selectedCategory = ''; 
        _currentPage = 1;
        _books = [];
        _hasMore = true;
      });
    }

    // Search is automatically logged by backend via BooksService.search call below
    
    try {
      final books = await BooksService.search(query.trim(), page: _currentPage);
      if (mounted) {
        setState(() {
          if (isLoadMore) {
            _books.addAll(books);
            _isLoadingMore = false;
          } else {
            _books = books;
            _isLoading = false;
          }
          _hasError = false;
          if (books.length < 10) {
            _hasMore = false;
          }
        });

        // Update recommendations in background
        if (!isLoadMore) {
          try {
            final bp = Provider.of<BooksProvider>(context, listen: false);
            bp.setLastSearchTerm(query.trim()); // Immediate update
            bp.fetchPersonalizedRecommendations();
            bp.fetchBecauseYouSearched();
          } catch (_) {}
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _isLoadingMore = false;
          _hasError = _books.isEmpty;
        });
      }
    }
  }

  Future<void> _onRefresh() async {
    if (_searchController.text.isNotEmpty) {
      await _searchBooks(_searchController.text);
    } else if (_selectedCategory.isNotEmpty) {
      await _loadBooksByCategory(_selectedCategory);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Public Library', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: const [],
      ),
      body: Column(
        children: [
          // Search Bar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: 'Search across Google, Gutenberg, Archive, OpenLib, IT...',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          if (_categories.isNotEmpty) {
                            setState(() => _selectedCategory = _categories[0]['id']);
                            _loadBooksByCategory(_selectedCategory);
                          }
                        },
                      )
                    : null,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: BorderSide.none,
                ),
                filled: true,
                fillColor: theme.colorScheme.surfaceContainerHighest,
              ),
              onChanged: (val) => setState(() {}),
              onSubmitted: (val) => _searchBooks(val),
            ),
          ),
          
          // Categories list
          if (_categories.isNotEmpty && _searchController.text.isEmpty)
            SizedBox(
              height: 50,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 12.0),
                itemCount: _categories.length,
                itemBuilder: (context, index) {
                  final cat = _categories[index];
                  final isSelected = _selectedCategory == cat['id'];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4.0),
                    child: ChoiceChip(
                      label: Text(
                        cat['name'],
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                        ),
                      ),
                      selected: isSelected,
                      onSelected: (selected) {
                        if (selected) {
                          setState(() {
                            _selectedCategory = cat['id'];
                          });
                          _loadBooksByCategory(cat['id']);
                        }
                      },
                    ),
                  );
                },
              ),
            ),
            
          const SizedBox(height: 8),

          // Books Grid
          Expanded(
            child: _isLoading && _books.isEmpty
                ? _buildLoadingState()
                : _hasError && _books.isEmpty
                    ? _buildErrorState()
                    : _books.isEmpty && !_isLoading
                        ? _buildEmptyState()
                        : RefreshIndicator(
                            onRefresh: _onRefresh,
                            child: GridView.builder(
                              controller: _scrollController,
                              padding: const EdgeInsets.all(16),
                              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount: 2,
                                childAspectRatio: 0.45,
                                crossAxisSpacing: 16,
                                mainAxisSpacing: 16,
                              ),
                              itemCount: _books.length + (_isLoadingMore ? 2 : 0),
                              itemBuilder: (context, index) {
                                if (index >= _books.length) {
                                  return const Center(child: CircularProgressIndicator());
                                }
                                return BookCard(book: _books[index], width: double.infinity);
                              },
                            ),
                          ),
          ),
        ],
      ),
      bottomNavigationBar: const AppBottomNavBar(currentIndex: 1),
    );
  }

  Widget _buildLoadingState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 20),
          Text(
            'Fetching from 5 libraries...',
            style: TextStyle(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Google • Gutenberg • Archive • OpenLib • IT',
            style: TextStyle(
              fontSize: 11,
              color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.6),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.cloud_off_rounded,
              size: 64,
              color: Theme.of(context).colorScheme.error.withValues(alpha: 0.5),
            ),
            const SizedBox(height: 16),
            Text(
              'Connection Error',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Theme.of(context).colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Could not load books from the libraries.\nPlease check your connection and try again.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _onRefresh,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.auto_stories_outlined,
            size: 64,
            color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.3),
          ),
          const SizedBox(height: 16),
          Text(
            'No books found',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onSurface,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Try a different search or category',
            style: TextStyle(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}
