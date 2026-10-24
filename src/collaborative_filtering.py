"""Collaborative Filtering Recommender using Matrix Factorization (Surprise SVD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


class CollaborativeFilteringRecommender:
    """Collaborative Filtering model utilizing SVD matrix factorization."""

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 25,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        self.algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        self.user_seen_movies: Dict[int, Set[int]] = {}
        self.all_movie_ids: List[int] = []
        self.known_users: Set[int] = set()
        self.is_fitted: bool = False
        self.global_mean_rating: float = 3.5

    def fit(self, ratings_df: pd.DataFrame) -> CollaborativeFilteringRecommender:
        """Train SVD model on the provided ratings DataFrame.

        Args:
            ratings_df: DataFrame containing ['userId', 'movieId', 'rating'].
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader=reader
        )

        trainset = data.build_full_trainset()
        self.algo.fit(trainset)

        self.global_mean_rating = float(trainset.global_mean)
        self.known_users = set(ratings_df["userId"].unique())
        self.all_movie_ids = list(ratings_df["movieId"].unique())

        # Build map of already-rated movies per user for fast filtering
        self.user_seen_movies = (
            ratings_df.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        self.is_fitted = True
        return self

    def is_known_user(self, user_id: int) -> bool:
        """Check if user has prior rating history in the training data."""
        return user_id in self.known_users

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie.

        Returns:
            Predicted rating clamped to valid scale [0.5, 5.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call fit() or load() first.")

        pred = self.algo.predict(uid=user_id, iid=movie_id)
        return float(np_clip(pred.est, 0.5, 5.0))

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_movie_ids: Optional[List[int]] = None,
        movies_df: Optional[pd.DataFrame] = None,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate personalized collaborative recommendations for a user.

        Args:
            user_id: ID of the user.
            top_n: Number of recommendations to return.
            candidate_movie_ids: Optional pool of candidate movieIds to rank.
            movies_df: Optional preprocessed movies DataFrame for metadata lookup.
            exclude_ids: Optional additional movie IDs to exclude.

        Returns:
            List of recommendations ranked by predicted rating.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")

        already_rated = self.user_seen_movies.get(user_id, set()).copy()
        if exclude_ids:
            already_rated.update(exclude_ids)

        # Determine items to score
        if candidate_movie_ids is not None:
            pool = [m for m in candidate_movie_ids if m not in already_rated]
        else:
            pool = [m for m in self.all_movie_ids if m not in already_rated]

        if not pool:
            return []

        # Predict ratings for all items in the candidate pool
        scored: List[Tuple[int, float]] = []
        for mid in pool:
            pred_score = self.predict_rating(user_id, mid)
            scored.append((mid, pred_score))

        # Sort descending by predicted rating
        scored.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored[:top_n]

        # Build movie lookup dictionary if DataFrame is provided
        movie_lookup = {}
        if movies_df is not None:
            movie_lookup = movies_df.set_index("movieId").to_dict(orient="index")

        results: List[Dict[str, Any]] = []
        for mid, score in top_scored:
            info = movie_lookup.get(mid, {})
            results.append({
                "movieId": mid,
                "title": info.get("title", f"Movie #{mid}"),
                "clean_title": info.get("clean_title", f"Movie #{mid}"),
                "year": info.get("year", None),
                "genres_str": info.get("genres_str", "Unknown"),
                "vote_average": float(info.get("vote_average", 0.0)),
                "vote_count": int(info.get("vote_count", 0)),
                "predicted_rating": round(score, 2),
                "similarity_score": round((score - 0.5) / 4.5, 4),  # Normalized to [0, 1]
                "reason": f"Collaborative filtering: Predicted user rating of {score:.1f} / 5.0",
                "imdbId": info.get("imdbId", ""),
                "tmdbId": info.get("tmdbId", ""),
            })

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Save trained model and lookup tables."""
        data = {
            "algo": self.algo,
            "user_seen_movies": self.user_seen_movies,
            "all_movie_ids": self.all_movie_ids,
            "known_users": self.known_users,
            "global_mean_rating": self.global_mean_rating,
            "params": {
                "n_factors": self.n_factors,
                "n_epochs": self.n_epochs,
                "lr_all": self.lr_all,
                "reg_all": self.reg_all,
                "random_state": self.random_state,
            },
        }
        joblib.dump(data, filepath, compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CollaborativeFilteringRecommender:
        """Load trained model from disk."""
        data = joblib.load(filepath)
        params = data.get("params", {})
        instance = cls(**params)
        instance.algo = data["algo"]
        instance.user_seen_movies = data["user_seen_movies"]
        instance.all_movie_ids = data["all_movie_ids"]
        instance.known_users = data["known_users"]
        instance.global_mean_rating = data.get("global_mean_rating", 3.5)
        instance.is_fitted = True
        return instance


def np_clip(val: float, low: float, high: float) -> float:
    """Helper to clip numeric prediction within bounds."""
    if val < low:
        return low
    if val > high:
        return high
    return val
