import 'dart:convert';

class Book {
  final String gid;
  final String title;
  final String author;
  final double averageRating;
  final List<String> categories;
  
  Book({
    required this.gid,
    required this.title,
    required this.author,
    required this.averageRating,
    required this.categories,
  });
}

Book parseServerBook(Map<String, dynamic> item) {
  String cover = item['cover_url'] ?? item['cover'] ?? '';
  if (cover.isNotEmpty && cover.startsWith('http://')) {
    cover = cover.replaceFirst('http://', 'https://');
  }

  return Book(
    gid: item['id']?.toString() ?? '',
    title: item['title'] ?? 'بدون عنوان',
    author: item['author'] ?? 'غير معروف',
    averageRating: (item['rating'] ?? 0.0).toDouble(),
    categories: List<String>.from(item['categories'] ?? []),
  );
}

void main() {
  String jsonStr = '{"books":[{"author":"John Clark Ridpath","categories":["Literary Criticism / General"],"cover":"https://books.google.com/books/content?id=zUaIZwEACAAJ&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE71tKvnosN2hck9RD03kStM_d8b9lQuart5QvWFYR7BbYNNNlMWlBQ3ZzF3IpfKKw01SgZ0ZTMctV4eNG4XvJpjUTza3KKL1HbBpTzcSzMmCiePRnQR2RFDwk_Vjbswyyr7PXhen&source=gbs_api","cover_url":"https://books.google.com/books/content?id=zUaIZwEACAAJ&printsec=frontcover&img=1&zoom=1&imgtk=AFLRE71tKvnosN2hck9RD03kStM_d8b9lQuart5QvWFYR7BbYNNNlMWlBQ3ZzF3IpfKKw01SgZ0ZTMctV4eNG4XvJpjUTza3KKL1HbBpTzcSzMmCiePRnQR2RFDwk_Vjbswyyr7PXhen&source=gbs_api","desc":"This Elibron Classics title is a reprint of the original edition published by the Globe Publishing Company in New York, 1900.","id":"zUaIZwEACAAJ","isbn":"0543722422","language":"en","pageCount":480,"publishedDate":"1999","rating":5.0,"reason":"Community: 5.0 (1)","source":"Community","title":"The Ridpath Library of Universal Literature"}],"success":true}';
  var data = jsonDecode(jsonStr);
  var items = data['books'] as List<dynamic>;
  for (var item in items) {
    var b = parseServerBook(item);
    print('Parsed: \${b.title} - \${b.averageRating}');
  }
}
