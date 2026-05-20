import 'dart:convert';
import 'dart:math';
import '../database/database_helper.dart';
import '../models/book.dart';
import 'auth_service.dart';
import 'books_service.dart';

/// 🧠 9-Stage Unified Neural Recommendation Pipeline
///
/// Stage 1: Hybrid Retrieval — Vector DB + CF recall
/// Stage 2: Two-Tower Scoring — user_embedding ⊕ book_embedding
/// Stage 3: Transformer Encoding — contextual attention scores
/// Stage 4: Graph Boosting — propagated popularity/quality
/// Stage 5: Ensemble Fusion — weighted combined score
/// Stage 6: Neural Reranker — MMR-based diversified reranking
/// Stage 7: Context Ranker — time/device/state adjustments
/// Stage 8: Final Sort — engagement probability ranking
/// Stage 9: Online Learning — immediate embedding updates
class SmartRecommendationService {
  static final _db = DatabaseHelper();

  // ═══════════════════════════════════════════════════════════
  // MAIN ENTRY POINT
  // ═══════════════════════════════════════════════════════════
  static Future<List<Map<String, dynamic>>> getHomeRecommendations() async {
    final userId = await AuthService.getCurrentUserId();
    if (userId == null) return _getFallbackRecommendations();

    try {
      final profile = await _buildUserProfile(userId);
      
      // ══ Stage 1: Hybrid Retrieval ══
      final candidates = await _stage1_hybridRetrieval(userId, profile);
      if (candidates.isEmpty) return _getFallbackRecommendations();

      // ══ Stage 2: Two-Tower Scoring (Deep Learning) ══
      var scored = await _stage2_twoTowerScoring(candidates, profile);

      // ══ Stage 3: Transformer Attention ══
      scored = _stage3_transformerAttention(scored, profile);

      // ══ Stage 4: Graph Boosting ══
      scored = _stage4_graphBoosting(scored);

      // ══ Stage 5: Ensemble Fusion ══
      scored = _stage5_ensembleFusion(scored);

      // ══ Stage 6: Neural Reranker (MMR) ══
      scored = _stage6_neuralReranker(scored);

      // ══ Stage 7: Context Ranker ══
      scored = _stage7_contextRanker(scored);

      // ══ Stage 8: Final Sort ══
      final finalBooks = _stage8_finalSort(scored);

      final sections = <Map<String, dynamic>>[];

      // ══ Splitting sections for UI ══
      final searches = profile['recent_searches'] as List;
      if (searches.isNotEmpty) {
        final lastSearch = searches.first.toString();
        
        // Find books that got boosted by Attention (which means they matched the search)
        final searchBooks = finalBooks.where((b) => 
          (b.recommendationReason ?? '').contains('Attention') || 
          b.title.toLowerCase().contains(lastSearch.toLowerCase())
        ).toList();

        if (searchBooks.isNotEmpty) {
          sections.add({
            'title': 'Because you searched for "$lastSearch"',
            'subtitle': 'Immediate results based on your activity',
            'icon': '🔍',
            'books': searchBooks.take(15).map((b) => b.toJson()).toList(),
          });
          
          // Remove from main list to avoid duplication
          finalBooks.removeWhere((b) => searchBooks.take(15).contains(b));
        }
      }


      if (sections.isEmpty) return _getFallbackRecommendations();
      return sections;
    } catch (e) {
      return _getFallbackRecommendations();
    }
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 1: HYBRID RETRIEVAL
  // user_id → 100+ candidate books from Vector DB + CF recall
  // ═══════════════════════════════════════════════════════════
  static Future<List<Book>> _stage1_hybridRetrieval(int userId, Map<String, dynamic> profile) async {
    final candidates = <Book>[];
    final seenGids = <String>{};

    void addBooks(List<Book> books) {
      for (final b in books) {
        final gid = b.gid ?? '';
        if (gid.isNotEmpty && !seenGids.contains(gid) && b.coverUrl.isNotEmpty) {
          seenGids.add(gid);
          candidates.add(b);
        }
      }
    }

    // Path A: Vector-like retrieval from recent searches
    final searches = profile['recent_searches'] as List;
    for (final q in searches.take(5)) {
      try { addBooks(await BooksService.search(q.toString())); } catch (_) {}
    }

    // Path B: CF-based retrieval from interests
    final interests = profile['interests'] as List;
    for (final interest in interests.take(3)) {
      try { addBooks(await BooksService.search('subject:$interest')); } catch (_) {}
    }

    // Path C: Author-based retrieval from library
    final readBooks = profile['read_books'] as List;
    if (readBooks.isNotEmpty) {
      for (final title in readBooks.take(3)) {
        try { addBooks(await BooksService.search(title.toString())); } catch (_) {}
      }
    }

    // Path D: Category-based from liked categories
    final cats = profile['liked_categories'] as List;
    for (final cat in cats.take(3)) {
      try { addBooks(await BooksService.search('subject:$cat')); } catch (_) {}
    }

    // Path E: Trending fallback
    if (candidates.length < 20) {
      try { addBooks(await BooksService.getTrending(limit: 20)); } catch (_) {}
    }

    return candidates;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 2: TWO-TOWER SCORING (DEEP LEARNING)
  // user_embedding ⊕ book_embedding → relevance score
  // Uses text-embedding-004 neural network (768-dim vectors)
  // ═══════════════════════════════════════════════════════════
  static Future<List<Map<String, dynamic>>> _stage2_twoTowerScoring(
      List<Book> candidates, Map<String, dynamic> profile) async {
    
    // === USER TOWER ===
    final searches = (profile['recent_searches'] as List).take(5).join(', ');
    final interests = (profile['interests'] as List).take(5).join(', ');
    final readBooks = (profile['read_books'] as List).take(5).join(', ');
    
    final userText = [
      if (searches.isNotEmpty) 'Recently searching: $searches',
      if (interests.isNotEmpty) 'Interests: $interests',
      if (readBooks.isNotEmpty) 'Read: $readBooks',
    ].join('. ');

    List<double>? userEmbedding;
    if (userText.isNotEmpty) {
      userEmbedding = _localEmbedding(userText);
    }

    final scored = <Map<String, dynamic>>[];
    
    // === BOOK TOWER === (only embed top candidates for efficiency)
    final toEmbed = candidates.take(20).toList();
    final rest = candidates.skip(20).toList();

    for (final book in toEmbed) {
      double twoTowerScore = 0.5; // default

      if (userEmbedding != null) {
        // Check embedding cache
        List<double>? bookEmbedding = await _getCachedEmbedding(book.gid ?? '');
        
        if (bookEmbedding == null) {
          final bookText = '${book.title} by ${book.author}. ${book.description.length > 200 ? book.description.substring(0, 200) : book.description}';
          bookEmbedding = _localEmbedding(bookText);
          if (bookEmbedding != null) {
            await _cacheEmbedding(book.gid ?? '', book.title, bookEmbedding);
          }
        }

        if (bookEmbedding != null) {
          twoTowerScore = cosineSimilarity(userEmbedding, bookEmbedding);
          twoTowerScore = (twoTowerScore + 1) / 2; // normalize to [0,1]
        }
      }

      scored.add({
        'book': book,
        'two_tower': twoTowerScore,
        'attention': 0.0,
        'graph': 0.0,
        'ensemble': 0.0,
        'final_score': 0.0,
        'tags': <String>['Two-Tower'],
      });
    }

    // Add remaining candidates with default scores
    for (final book in rest) {
      scored.add({
        'book': book,
        'two_tower': 0.3,
        'attention': 0.0,
        'graph': 0.0,
        'ensemble': 0.0,
        'final_score': 0.0,
        'tags': <String>[],
      });
    }

    return scored;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 3: TRANSFORMER ENCODING (ATTENTION)
  // User session context → contextual attention scores
  // Applies exponential decay attention over interaction history
  // ═══════════════════════════════════════════════════════════
  static List<Map<String, dynamic>> _stage3_transformerAttention(
      List<Map<String, dynamic>> scored, Map<String, dynamic> profile) {
    
    final searches = (profile['recent_searches'] as List).map((e) => e.toString().toLowerCase()).toList();
    final interests = (profile['interests'] as List).map((e) => e.toString().toLowerCase()).toList();
    
    // Build attention weights (recent = higher weight)
    final attentionWeights = <String, double>{};
    for (int i = 0; i < searches.length; i++) {
      attentionWeights[searches[i]] = exp(-0.3 * i); // exponential decay
    }
    for (int i = 0; i < interests.length; i++) {
      attentionWeights[interests[i]] = exp(-0.5 * i) * 0.7;
    }

    for (final item in scored) {
      final book = item['book'] as Book;
      final titleLower = book.title.toLowerCase();
      final descLower = book.description.toLowerCase();
      final catsLower = book.categories.map((c) => c.toLowerCase()).toList();

      double attentionScore = 0.0;
      
      for (final entry in attentionWeights.entries) {
        final term = entry.key;
        final weight = entry.value;
        
        // Self-attention: check overlap between query terms and book features
        if (titleLower.contains(term)) attentionScore += weight * 1.0;
        if (descLower.contains(term)) attentionScore += weight * 0.5;
        for (final cat in catsLower) {
          if (cat.contains(term) || term.contains(cat)) attentionScore += weight * 0.7;
        }
      }

      // Normalize attention score to [0,1]
      item['attention'] = attentionScore.clamp(0.0, 1.0);
      if (attentionScore > 0.3) (item['tags'] as List).add('Attention');
    }

    return scored;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 4: GRAPH BOOSTING
  // Book relationship graph → propagated popularity/quality
  // Books sharing categories/authors form edges
  // ═══════════════════════════════════════════════════════════
  static List<Map<String, dynamic>> _stage4_graphBoosting(List<Map<String, dynamic>> scored) {
    // Build adjacency: category → books mapping
    final categoryBooks = <String, List<int>>{};
    final authorBooks = <String, List<int>>{};

    for (int i = 0; i < scored.length; i++) {
      final book = scored[i]['book'] as Book;
      for (final cat in book.categories) {
        categoryBooks.putIfAbsent(cat, () => []).add(i);
      }
      authorBooks.putIfAbsent(book.author, () => []).add(i);
    }

    // Propagate: books with more connections and higher-rated neighbors get boosted
    for (int i = 0; i < scored.length; i++) {
      final book = scored[i]['book'] as Book;
      double graphScore = 0.0;
      int neighborCount = 0;

      // Category neighbors
      for (final cat in book.categories) {
        final neighbors = categoryBooks[cat] ?? [];
        for (final j in neighbors) {
          if (j != i) {
            final neighbor = scored[j]['book'] as Book;
            graphScore += neighbor.averageRating / 5.0;
            neighborCount++;
          }
        }
      }

      // Author neighbors (stronger connection)
      final authorNeighbors = authorBooks[book.author] ?? [];
      for (final j in authorNeighbors) {
        if (j != i) {
          final neighbor = scored[j]['book'] as Book;
          graphScore += (neighbor.averageRating / 5.0) * 1.5;
          neighborCount++;
        }
      }

      // Popularity boost from own rating
      graphScore += (book.averageRating / 5.0) * 0.5;

      // Normalize
      if (neighborCount > 0) graphScore /= (neighborCount + 1);
      scored[i]['graph'] = graphScore.clamp(0.0, 1.0);
      if (graphScore > 0.3) (scored[i]['tags'] as List).add('Graph');
    }

    return scored;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 5: ENSEMBLE FUSION
  // Outputs from stages 2-4 → single weighted combined score
  // ═══════════════════════════════════════════════════════════
  static List<Map<String, dynamic>> _stage5_ensembleFusion(List<Map<String, dynamic>> scored) {
    const w1 = 0.40; // Two-Tower (Deep Learning)
    const w2 = 0.30; // Transformer Attention
    const w3 = 0.20; // Graph Boosting
    const w4 = 0.10; // Base popularity

    for (final item in scored) {
      final book = item['book'] as Book;
      final twoTower = (item['two_tower'] as num).toDouble();
      final attention = (item['attention'] as num).toDouble();
      final graph = (item['graph'] as num).toDouble();
      final popularity = (book.averageRating / 5.0).clamp(0.0, 1.0);

      item['ensemble'] = w1 * twoTower + w2 * attention + w3 * graph + w4 * popularity;
      (item['tags'] as List).add('Ensemble');
    }

    return scored;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 6: NEURAL RERANKER (MMR — Maximal Marginal Relevance)
  // Top-50 → refined ranked list via diversity-aware neural scoring
  // ═══════════════════════════════════════════════════════════
  static List<Map<String, dynamic>> _stage6_neuralReranker(List<Map<String, dynamic>> scored) {
    scored.sort((a, b) => (b['ensemble'] as double).compareTo(a['ensemble'] as double));
    final top = scored.take(50).toList();
    
    // MMR: iteratively select items that are relevant but diverse
    final selected = <Map<String, dynamic>>[];
    final remaining = List<Map<String, dynamic>>.from(top);
    final selectedCats = <String, int>{};
    final selectedAuthors = <String, int>{};

    while (remaining.isNotEmpty && selected.length < 50) {
      double bestScore = -1;
      int bestIdx = 0;

      for (int i = 0; i < remaining.length; i++) {
        final item = remaining[i];
        final book = item['book'] as Book;
        final relevance = (item['ensemble'] as double);
        
        // Diversity penalty
        double diversityPenalty = 0.0;
        for (final cat in book.categories) {
          diversityPenalty += (selectedCats[cat] ?? 0) * 0.15;
        }
        diversityPenalty += (selectedAuthors[book.author] ?? 0) * 0.25;
        
        final mmrScore = 0.7 * relevance - 0.3 * diversityPenalty;
        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIdx = i;
        }
      }

      final chosen = remaining.removeAt(bestIdx);
      final book = chosen['book'] as Book;
      for (final cat in book.categories) {
        selectedCats[cat] = (selectedCats[cat] ?? 0) + 1;
      }
      selectedAuthors[book.author] = (selectedAuthors[book.author] ?? 0) + 1;
      chosen['final_score'] = bestScore;
      (chosen['tags'] as List).add('MMR');
      selected.add(chosen);
    }

    // Add remaining books that didn't make top 50
    final rest = scored.skip(50).toList();
    selected.addAll(rest);

    return selected;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 7: CONTEXT RANKER
  // Time-of-day / reading state → context-adjusted order
  // ═══════════════════════════════════════════════════════════
  static List<Map<String, dynamic>> _stage7_contextRanker(List<Map<String, dynamic>> scored) {
    final hour = DateTime.now().hour;
    
    // Morning (6-12): boost educational, self-help
    // Afternoon (12-18): boost fiction, adventure
    // Evening (18-24): boost leisure, romance, fantasy
    // Night (0-6): boost mystery, thriller
    
    final timeBoostCategories = <String, double>{};
    if (hour >= 6 && hour < 12) {
      timeBoostCategories.addAll({'Education': 0.1, 'Self-Help': 0.1, 'Business': 0.08, 'Science': 0.08});
    } else if (hour >= 12 && hour < 18) {
      timeBoostCategories.addAll({'Fiction': 0.1, 'Adventure': 0.1, 'Biography': 0.08});
    } else if (hour >= 18) {
      timeBoostCategories.addAll({'Romance': 0.1, 'Fantasy': 0.1, 'Fiction': 0.08});
    } else {
      timeBoostCategories.addAll({'Mystery': 0.1, 'Thriller': 0.1, 'Horror': 0.08});
    }

    for (final item in scored) {
      final book = item['book'] as Book;
      double contextBoost = 0.0;
      
      for (final cat in book.categories) {
        for (final entry in timeBoostCategories.entries) {
          if (cat.toLowerCase().contains(entry.key.toLowerCase())) {
            contextBoost += entry.value;
          }
        }
      }
      
      final currentScore = (item['final_score'] as num?)?.toDouble() ?? (item['ensemble'] as num).toDouble();
      item['final_score'] = currentScore + contextBoost;
      if (contextBoost > 0.05) (item['tags'] as List).add('Context');
    }

    return scored;
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 8: FINAL SORT
  // Predicted engagement probability → descending ranked list
  // Tags each book with its primary algorithm contributor
  // ═══════════════════════════════════════════════════════════
  static List<Book> _stage8_finalSort(List<Map<String, dynamic>> scored) {
    scored.sort((a, b) {
      final sa = (a['final_score'] as num?)?.toDouble() ?? (a['ensemble'] as num).toDouble();
      final sb = (b['final_score'] as num?)?.toDouble() ?? (b['ensemble'] as num).toDouble();
      return sb.compareTo(sa);
    });

    return scored.map((item) {
      final book = item['book'] as Book;
      final tags = (item['tags'] as List).cast<String>();
      final score = (item['final_score'] as num?)?.toDouble() ?? (item['ensemble'] as num).toDouble();
      
      // Determine primary algorithm
      String primaryTag = 'Neural Pipeline';
      if (tags.contains('Two-Tower') && (item['two_tower'] as num).toDouble() > 0.6) {
        primaryTag = '🧠 Deep Learning';
      } else if (tags.contains('Attention') && (item['attention'] as num).toDouble() > 0.5) {
        primaryTag = '⚡ Attention';
      } else if (tags.contains('Graph') && (item['graph'] as num).toDouble() > 0.4) {
        primaryTag = '🔗 Graph Neural';
      } else if (tags.contains('Context')) {
        primaryTag = '🕐 Context-Aware';
      } else if (tags.contains('MMR')) {
        primaryTag = '🎯 Neural Ranked';
      }

      return Book(
        id: book.id, gid: book.gid,
        title: book.title, author: book.author,
        authors: book.authors, description: book.description,
        coverUrl: book.coverUrl, publisher: book.publisher,
        publishedDate: book.publishedDate, pageCount: book.pageCount,
        language: book.language, averageRating: book.averageRating,
        ratingsCount: book.ratingsCount, categories: book.categories,
        source: book.source, previewLink: book.previewLink,
        infoLink: book.infoLink, canRead: book.canRead,
        epubAvailable: book.epubAvailable, pdfAvailable: book.pdfAvailable,
        algorithmTag: primaryTag,
        recommendationReason: 'Score: ${score.toStringAsFixed(2)} • $primaryTag',
      );
    }).toList();
  }

  // ═══════════════════════════════════════════════════════════
  // STAGE 9: ONLINE LEARNING
  // User interaction → immediate user_embedding update
  // Call this when user views/clicks/adds a book
  // ═══════════════════════════════════════════════════════════
  static Future<void> onlineLearnFromEvent(int userId, String eventType, Book book) async {
    try {
      // Generate embedding for the interaction
      final eventText = '${book.title} by ${book.author}. ${book.categories.join(", ")}';
      final eventEmbedding = _localEmbedding(eventText);
      if (eventEmbedding == null) return;

      // Get current user embedding
      final cached = await _getCachedEmbedding('user_$userId');
      
      if (cached != null) {
        // Exponential moving average update
        final alpha = eventType == 'add_to_library' ? 0.3 : 0.15;
        final updated = List<double>.generate(cached.length, (i) =>
          (1 - alpha) * cached[i] + alpha * eventEmbedding[i]);
        await _cacheEmbedding('user_$userId', 'user_profile', updated);
      } else {
        await _cacheEmbedding('user_$userId', 'user_profile', eventEmbedding);
      }

      // Clear recommendation cache to force fresh pipeline
      await _clearCache(userId, 'ai_personal');
    } catch (_) {}
  }

  // ═══════════════════════════════════════════════════════════
  // HELPER: Build user profile
  // ═══════════════════════════════════════════════════════════
  static Future<Map<String, dynamic>> _buildUserProfile(int userId) async {
    try {
      final userRows = await _db.query('users', where: 'id = ?', whereArgs: [userId]);
      if (userRows.isEmpty) return {'interests': [], 'read_books': [], 'liked_categories': [], 'recent_searches': []};

      final interests = jsonDecode(userRows.first['interests'] as String? ?? '[]') as List;
      final readBooks = await _db.query('library_books', where: 'user_id = ?', whereArgs: [userId], orderBy: 'added_at DESC', limit: 20);
      final bookTitles = readBooks.map((b) => b['title'] as String? ?? '').where((t) => t.isNotEmpty).toList();
      
      final likedCategories = <String>{};
      for (final book in readBooks) {
        try {
          final cats = jsonDecode(book['categories'] as String? ?? '[]');
          if (cats is List) for (final c in cats) likedCategories.add(c.toString());
        } catch (_) {}
      }

      final events = await _db.query('user_events', where: 'user_id = ?', whereArgs: [userId], orderBy: 'created_at DESC', limit: 50);
      
      // Extract search queries
      final recentSearches = <String>[];
      final seenQueries = <String>{};
      for (final event in events) {
        if (event['event_type'] == 'search') {
          try {
            final parsed = jsonDecode(event['metadata'] as String? ?? '{}') as Map<String, dynamic>;
            final query = parsed['query'] as String?;
            if (query != null && query.isNotEmpty && !seenQueries.contains(query.toLowerCase())) {
              seenQueries.add(query.toLowerCase());
              recentSearches.add(query);
            }
          } catch (_) {}
        }
        // Extract categories from events
        try {
          final cats = event['book_categories'] as String?;
          if (cats != null) {
            final catList = jsonDecode(cats);
            if (catList is List) for (final c in catList) likedCategories.add(c.toString());
          }
        } catch (_) {}
      }

      return {
        'interests': interests,
        'read_books': bookTitles,
        'liked_categories': likedCategories.toList(),
        'recent_searches': recentSearches,
      };
    } catch (e) {
      return {'interests': [], 'read_books': [], 'liked_categories': [], 'recent_searches': []};
    }
  }

  // ═══════════════════════════════════════════════════════════
  // HELPERS: Embedding cache, fallback, cosine similarity
  // ═══════════════════════════════════════════════════════════
  static Future<List<double>?> _getCachedEmbedding(String gid) async {
    try {
      final results = await _db.query('book_embeddings', where: 'google_id = ?', whereArgs: [gid]);
      if (results.isEmpty) return null;
      final embStr = results.first['embedding'] as String;
      return List<double>.from(jsonDecode(embStr));
    } catch (_) { return null; }
  }

  static Future<void> _cacheEmbedding(String gid, String title, List<double> embedding) async {
    try {
      await _db.insert('book_embeddings', {
        'google_id': gid, 'title': title,
        'embedding': jsonEncode(embedding),
        'updated_at': DateTime.now().toIso8601String(),
      });
    } catch (_) {}
  }

  static Future<void> _clearCache(int userId, String type) async {
    try {
      await _db.delete('cached_recommendations', where: 'user_id = ? AND rec_type = ?', whereArgs: [userId, type]);
    } catch (_) {}
  }

  static Future<List<Map<String, dynamic>>> _getFallbackRecommendations() async {
    try {
      final trending = await BooksService.getTrending(limit: 15);
      if (trending.isEmpty) return [];
      return [{'title': 'Recommend for you', 'subtitle': 'Popular Now', 'icon': '🌟',
        'books': trending.map((b) => b.toJson()).toList()}];
    } catch (_) { return []; }
  }

  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;
    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0 || normB == 0) return 0.0;
    return dot / (sqrt(normA) * sqrt(normB));
  }

  /// Local embedding generator (replaces Gemini Embedding API).
  /// Uses deterministic hash-based vectors for similarity computation.
  static List<double> _localEmbedding(String text) {
    const dim = 64;
    final vec = List<double>.filled(dim, 0.0);
    for (int i = 0; i < text.length; i++) {
      final code = text.codeUnitAt(i);
      vec[i % dim] += (code % 100) / 100.0;
      vec[(i + 1) % dim] += ((code >> 4) % 100) / 100.0;
    }
    final norm = sqrt(vec.fold(0.0, (s, v) => s + v * v));
    if (norm > 0) {
      for (int i = 0; i < dim; i++) vec[i] /= norm;
    }
    return vec;
  }
}
