import 'package:flutter/material.dart';
import '../../models/review.dart';
import '../../config/translations.dart';
import 'star_rating.dart';

class ReviewCard extends StatelessWidget {
  final Review review;

  const ReviewCard({
    Key? key,
    required this.review,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            children: [
              CircleAvatar(
                radius: 20,
                backgroundColor: cs.tertiaryContainer,
                child: Text(
                  (review.userName ?? 'R')[0].toUpperCase(),
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: cs.onTertiaryContainer,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      review.userName ?? '${context.t('reader_no')} #${review.userId}',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    Text(
                      _timeAgo(review.createdAt, context),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: cs.onSurfaceVariant.withValues(alpha: 0.6),
                          ),
                    ),
                  ],
                ),
              ),
              StarRating(rating: review.rating, size: 16),
            ],
          ),

          // Review text
          if (review.reviewText.isNotEmpty) ...[
            const SizedBox(height: 12),
            Text(
              '"${review.reviewText}"',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontStyle: FontStyle.italic,
                    height: 1.5,
                  ),
            ),
          ],
        ],
      ),
    );
  }

  String _timeAgo(DateTime date, BuildContext context) {
    final diff = DateTime.now().difference(date);
    if (diff.inDays > 365) {
      final years = (diff.inDays / 365).floor();
      return context.t('y_ago').replaceAll('{count}', years.toString());
    }
    if (diff.inDays > 30) {
      final months = (diff.inDays / 30).floor();
      return context.t('mo_ago').replaceAll('{count}', months.toString());
    }
    if (diff.inDays > 0) {
      return context.t('d_ago').replaceAll('{count}', diff.inDays.toString());
    }
    if (diff.inHours > 0) {
      return context.t('h_ago').replaceAll('{count}', diff.inHours.toString());
    }
    return context.t('just_now');
  }
}


