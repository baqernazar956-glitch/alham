class User {
  final int id;
  final String name;
  final String email;
  final bool onboardingCompleted;
  final String? profilePicture;
  final String? bio;
  final int readingGoal;
  final String rank;
  final int currentStreak;
  final List<String> interests;

  User({
    required this.id,
    required this.name,
    required this.email,
    required this.onboardingCompleted,
    this.profilePicture,
    this.bio,
    this.readingGoal = 0,
    this.rank = "Novice Reader",
    this.currentStreak = 0,
    this.interests = const [],
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] ?? 0,
      name: json['name'] ?? '',
      email: json['email'] ?? '',
      onboardingCompleted: json['onboarding_completed'] ?? false,
      profilePicture: json['profile_picture'],
      bio: json['bio'],
      readingGoal: json['reading_goal'] ?? 0,
      rank: json['rank'] ?? "Novice Reader",
      currentStreak: json['current_streak'] ?? 0,
      interests: json['interests'] != null ? List<String>.from(json['interests']) : [],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'onboarding_completed': onboardingCompleted,
      'profile_picture': profilePicture,
      'bio': bio,
      'reading_goal': readingGoal,
      'rank': rank,
      'current_streak': currentStreak,
      'interests': interests,
    };
  }
}
