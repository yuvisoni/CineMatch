"""Hybrid Recommendation Engine combining Content-Based and Collaborative Filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender


class HybridRecommender:
    """Combines Content-Based and Collaborative Filtering using configurable weighted scoring."""

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        cf_model: CollaborativeFilteringRecommender,
        content_weight: float = 0.5,
    ):
        """Initialize hybrid recommender.

        Args:
            content_model: Fitted ContentBasedRecommender instance.
            cf_model: Fitted CollaborativeFilteringRecommender instance.
            content_weight: Weight for content similarity score (between 0.0 and 1.0).
                            Collaborative weight is (1.0 - content_weight).
        """
        self.content_model = content_model
        self.cf_model = cf_model
        self.set_weights(content_weight)

    def set_weights(self, content_weight: float) -> None:
        """Update scoring weight parameters."""
        self.content_weight = float(np.clip(content_weight, 0.0, 1.0))
        self.cf_weight = 1.0 - self.content_weight

    def recommend(
        self,
        user_ratings: Dict[int, float],
        user_id: Optional[int] = None,
        top_n: int = 10,
        candidate_pool_size: int = 60,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations for a user.

        Args:
            user_ratings: Dictionary mapping movieId to user rating (e.g. {1: 5.0, 32: 4.5}).
            user_id: Optional existing userId in MovieLens dataset.
            top_n: Number of recommendations to return (default 10).
            candidate_pool_size: Number of initial content candidates to evaluate.
            exclude_ids: Set of movieIds to exclude.

        Returns:
            List of top_n recommended movie dictionaries with genuine component scores.
        """
        excluded = set(user_ratings.keys())
        if exclude_ids:
            excluded.update(exclude_ids)

        if not user_ratings:
            return []

        # Step 1: Content candidate generation using user preference profile
        content_candidates = self.content_model.recommend_from_profile(
            movie_ratings=user_ratings,
            top_n=candidate_pool_size,
            exclude_ids=excluded,
        )

        if not content_candidates:
            return []

        # Determine if collaborative filtering can be applied
        is_known = user_id is not None and self.cf_model.is_known_user(user_id)

        recommendations: List[Dict[str, Any]] = []

        for cand in content_candidates:
            mid = cand["movieId"]
            content_score = float(cand.get("similarity_score", 0.0))

            if is_known and user_id is not None:
                # User has rating history in collaborative model
                predicted_cf_rating = self.cf_model.predict_rating(user_id, mid)
                # Normalize CF rating [0.5, 5.0] to [0.0, 1.0]
                norm_cf_score = float(np.clip((predicted_cf_rating - 0.5) / 4.5, 0.0, 1.0))

                hybrid_score = (
                    self.content_weight * content_score
                    + self.cf_weight * norm_cf_score
                )
                est_stars = round(0.5 + 4.5 * hybrid_score, 1)

                reason = (
                    f"Hybrid match ({int(self.content_weight * 100)}% content / "
                    f"{int(self.cf_weight * 100)}% collaborative): "
                    f"Content similarity {content_score:.2f}, predicted rating {predicted_cf_rating:.1f}/5.0"
                )
            else:
                # Cold-start user fallback: pure content similarity with profile guidance
                predicted_cf_rating = float(cand.get("vote_average", self.cf_model.global_mean_rating))
                norm_cf_score = float(np.clip((predicted_cf_rating - 0.5) / 4.5, 0.0, 1.0))

                # If user is new, content similarity drives recommendation,
                # with baseline quality damping
                hybrid_score = 0.85 * content_score + 0.15 * norm_cf_score
                est_stars = round(0.5 + 4.5 * hybrid_score, 1)

                reason = (
                    f"Cold-Start Content match: Strong thematic & genre alignment ({int(content_score * 100)}%) "
                    f"with your selected favorites."
                )

            item = cand.copy()
            item["content_score"] = round(content_score, 4)
            item["cf_score"] = round(norm_cf_score, 4)
            item["predicted_rating"] = round(predicted_cf_rating, 2)
            item["hybrid_score"] = round(hybrid_score, 4)
            item["estimated_stars"] = est_stars
            item["reason"] = reason
            recommendations.append(item)

        # Rank candidates by hybrid score descending
        recommendations.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return recommendations[:top_n]
