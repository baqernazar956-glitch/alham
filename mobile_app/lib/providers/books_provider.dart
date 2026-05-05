import 'package:flutter/material.dart';
import '../models/book.dart';
import '../services/books_service.dart';
import '../services/user_service.dart';
// All recommendation logic runs on the server's unified pipeline

class BooksProvider with ChangeNotifier {
  List<Book> _trendingBooks = [];
  List<Book> _topRatedBooks = [];
  List<dynamic> _personalizedSections = [];
  List<Map<String, dynamic>> _curatedCollections = [];
  List<Book> _searchHistoryBooks = [];
  String _lastSearchTerm = '';
  bool _isLoading = false;

  List<Book> _userLibrary = [];

  List<Book> get trendingBooks => _trendingBooks;
  List<Book> get topRatedBooks => _topRatedBooks;
  List<dynamic> get personalizedSections => _personalizedSections;
  List<Map<String, dynamic>> get curatedCollections => _curatedCollections;
  List<Book> get searchHistoryBooks => _searchHistoryBooks;
  String get lastSearchTerm => _lastSearchTerm;
  List<Book> get userLibrary => _userLibrary;
  bool get isLoading => _isLoading;
  
  void setLastSearchTerm(String term) {
    if (term.isNotEmpty) {
      _lastSearchTerm = term;
      notifyListeners();
    }
  }

  Future<void> fetchTrending() async {
    _setLoading(true);
    _trendingBooks = await BooksService.getTrending(limit: 100);
    _setLoading(false);
  }

  Future<void> fetchTopRated() async {
    print('DEBUG: fetchTopRated called');
    _topRatedBooks = await BooksService.getTopRated(limit: 100);
    print('DEBUG: topRatedBooks length is ${_topRatedBooks.length}');
    notifyListeners();
  }

  Future<void> fetchPersonalizedRecommendations() async {
    _setLoading(true);
    _personalizedSections = await BooksService.getPersonalizedRecommendations();
    _setLoading(false);
  }

  Future<void> fetchCuratedCollections() async {
    _curatedCollections = await BooksService.getCuratedCollections();
    notifyListeners();
  }

  Future<void> fetchBecauseYouSearched() async {
    try {
      // Fetch search history from server (syncs with recommendations)
      final searchHistory = await UserService.getServerSearchHistory();
      if (searchHistory.isNotEmpty) {
        _lastSearchTerm = searchHistory.first;
      }
      
      // Fallback for new users
      if (_lastSearchTerm.isEmpty) {
        _lastSearchTerm = 'Self Growth';
      }
      
      _searchHistoryBooks = await BooksService.search(_lastSearchTerm, logSearch: false);
    } catch (_) {
      // Last resort fallback
      _lastSearchTerm = 'Self Growth';
      try {
        _searchHistoryBooks = await BooksService.search(_lastSearchTerm, logSearch: false);
      } catch (_) {}
    }
    notifyListeners();
  }

  Future<void> fetchUserLibrary() async {
    _setLoading(true);
    try {
      final libraryData = await UserService.getLibrary();
      _userLibrary = libraryData.map((e) => Book.fromJson(e)).toList();
    } catch (e) {
      _userLibrary = [];
    }
    _setLoading(false);
  }

  Future<bool> removeFromLibrary(String gid) async {
    final success = await UserService.removeFromLibrary(gid);
    if (success) {
      _userLibrary.removeWhere((b) => b.uniqueId == gid);
      notifyListeners();
    }
    return success;
  }

  Future<void> loadHomeData() async {
    _isLoading = true;
    notifyListeners();
    
    await Future.wait([
      BooksService.getTrending(limit: 100).then((v) => _trendingBooks = v),
      BooksService.getTopRated(limit: 100).then((v) => _topRatedBooks = v),
      BooksService.getPersonalizedRecommendations().then((v) => _personalizedSections = v),
      BooksService.getCuratedCollections().then((v) => _curatedCollections = v),
    ]);
    
    try {
      final searchHistory = await UserService.getServerSearchHistory();
      if (searchHistory.isNotEmpty) _lastSearchTerm = searchHistory.first;
      if (_lastSearchTerm.isEmpty) _lastSearchTerm = 'Self Growth';
      _searchHistoryBooks = await BooksService.search(_lastSearchTerm, logSearch: false);
    } catch (_) {}

    _isLoading = false;
    notifyListeners();
  }

  void _setLoading(bool value) {
    _isLoading = value;
    notifyListeners();
  }
}
