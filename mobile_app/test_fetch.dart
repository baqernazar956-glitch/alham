import 'lib/services/books_service.dart';

void main() async {
  print('Fetching...');
  final books = await BooksService.getTopRated(limit: 100);
  print('Books fetched: ${books.length}');
  for (var b in books) {
    print('- ${b.title} (${b.averageRating})');
  }
}
