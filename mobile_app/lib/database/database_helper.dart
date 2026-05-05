import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi_web/sqflite_ffi_web.dart';
import 'package:path/path.dart';

/// Local SQLite database for standalone operation.
class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  DatabaseHelper._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    if (kIsWeb) {
      // Initialize web factory for SQLite on the web
      databaseFactory = databaseFactoryFfiWeb;
      return await openDatabase('elham.db', version: 1, onCreate: _onCreate);
    } else {
      // Mobile & Desktop initialization
      final dbPath = await getDatabasesPath();
      final path = join(dbPath, 'elham.db');
      return await openDatabase(path, version: 1, onCreate: _onCreate);
    }
  }

  Future<void> _onCreate(Database db, int version) async {
    // ─── Book Embeddings Cache ───
    await db.execute('''
      CREATE TABLE book_embeddings (
        google_id TEXT PRIMARY KEY,
        title TEXT,
        embedding TEXT NOT NULL,
        updated_at TEXT DEFAULT (datetime('now'))
      )
    ''');

    // ─── Cached Recommendations ───
    await db.execute('''
      CREATE TABLE cached_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        rec_type TEXT NOT NULL,
        books_json TEXT NOT NULL,
        generated_at TEXT DEFAULT (datetime('now')),
        expires_at TEXT NOT NULL
      )
    ''');
  }

  // ─── Helper Methods ───

  Future<int> insert(String table, Map<String, dynamic> data) async {
    final db = await database;
    return await db.insert(table, data, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<Map<String, dynamic>>> query(
    String table, {
    String? where,
    List<dynamic>? whereArgs,
    String? orderBy,
    int? limit,
  }) async {
    final db = await database;
    return await db.query(table,
        where: where,
        whereArgs: whereArgs,
        orderBy: orderBy,
        limit: limit);
  }

  Future<int> update(
    String table,
    Map<String, dynamic> data, {
    String? where,
    List<dynamic>? whereArgs,
  }) async {
    final db = await database;
    return await db.update(table, data, where: where, whereArgs: whereArgs);
  }

  Future<int> delete(
    String table, {
    String? where,
    List<dynamic>? whereArgs,
  }) async {
    final db = await database;
    return await db.delete(table, where: where, whereArgs: whereArgs);
  }

  Future<void> close() async {
    final db = await database;
    await db.close();
    _database = null;
  }
}
