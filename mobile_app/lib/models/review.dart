class Review {
  final int id;
  final int userId;
  final String? userName;
  final String? googleId;
  final int? bookId;
  final int rating;
  final String reviewText;
  final int likesCount;
  final int dislikesCount;
  final DateTime createdAt;

  Review({
    required this.id,
    required this.userId,
    this.userName,
    this.googleId,
    this.bookId,
    required this.rating,
    this.reviewText = '',
    this.likesCount = 0,
    this.dislikesCount = 0,
    required this.createdAt,
  });

  factory Review.fromJson(Map<String, dynamic> json) {
    return Review(
      id: json['id'] ?? 0,
      userId: json['user_id'] ?? 0,
      userName: json['user_name'],
      googleId: json['google_id'],
      bookId: json['book_id'],
      rating: json['rating'] ?? 0,
      reviewText: json['review_text'] ?? '',
      likesCount: json['likes_count'] ?? 0,
      dislikesCount: json['dislikes_count'] ?? 0,
      createdAt: json['created_at'] != null 
          ? DateTime.parse(json['created_at']) 
          : DateTime.now(),
    );
  }
}
