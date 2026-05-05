import 'package:flutter/material.dart';
import '../../models/book.dart';
import 'book_card.dart';

class HorizontalBookList extends StatelessWidget {
  final List<Book> books;
  final double cardWidth;
  final double cardHeight;

  const HorizontalBookList({
    Key? key,
    required this.books,
    this.cardWidth = 140,
    this.cardHeight = 210,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (books.isEmpty) {
      return const SizedBox(
        height: 250,
        child: Center(child: Text('No books found.')),
      );
    }

    return SizedBox(
      height: cardHeight + 80, // Space for title/author
      child: ListView.builder(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        physics: const BouncingScrollPhysics(),
        itemCount: books.length,
        itemBuilder: (context, index) {
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: BookCard(
              book: books[index],
              width: cardWidth,
              height: cardHeight,
            ),
          );
        },
      ),
    );
  }
}
