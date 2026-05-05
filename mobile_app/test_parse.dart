void main() {
  Map<String, dynamic> item = {'rating': null};
  double rating = (item['rating'] ?? 0.0).toDouble();
  print(rating);
}
