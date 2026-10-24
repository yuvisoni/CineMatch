"""Unified service layer providing business logic, caching, and model access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from src.data_loader import get_default_data_dir, load_raw_data, validate_dataset_presence
from src.preprocessing import prepare_movie_metadata
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridRecommender


class MovieService:
    """Service facade providing movie data discovery, caching, and multi-model recommendations."""

    def __init__(self, data_dir: Optional[Path | str] = None, models_dir: Optional[Path | str] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir else get_default_data_dir()
        self.models_dir = Path(models_dir) if models_dir else (self.project_root / "models")

        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        self.content_model: Optional[ContentBasedRecommender] = None
        self.cf_model: Optional[CollaborativeFilteringRecommender] = None
        self.hybrid_model: Optional[HybridRecommender] = None
        self.eval_results: Dict[str, Any] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Load data and model artifacts."""
        # 1. Load movies
        processed_path = self.data_dir / "movies_processed.csv"
        if processed_path.exists():
            self.movies_df = pd.read_csv(processed_path)
            # Reconstruct list for genres
            if "genres" in self.movies_df.columns and isinstance(self.movies_df["genres"].iloc[0], str):
                import ast
                try:
                    self.movies_df["genres"] = self.movies_df["genres"].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
                    )
                except Exception:
                    pass
        else:
            is_valid, _ = validate_dataset_presence(self.data_dir)
            if not is_valid:
                raise FileNotFoundError(f"MovieLens dataset not found in {self.data_dir}")
            raw = load_raw_data(self.data_dir)
            self.movies_df = prepare_movie_metadata(
                raw["movies"], raw["tags"], raw["ratings"], raw["links"]
            )
            self.movies_df.to_csv(processed_path, index=False)

        # 2. Load ratings
        ratings_path = self.data_dir / "ratings.csv"
        if ratings_path.exists():
            self.ratings_df = pd.read_csv(ratings_path)

        # 3. Load or train content model
        cb_path = self.models_dir / "content_model.joblib"
        if cb_path.exists():
            self.content_model = ContentBasedRecommender.load(cb_path)
        else:
            self.content_model = ContentBasedRecommender(max_features=10000, ngram_range=(1, 2))
            self.content_model.fit(self.movies_df)

        # 4. Load or train collaborative filtering model
        cf_path = self.models_dir / "cf_model.joblib"
        if cf_path.exists():
            self.cf_model = CollaborativeFilteringRecommender.load(cf_path)
        elif not self.ratings_df.empty:
            self.cf_model = CollaborativeFilteringRecommender()
            self.cf_model.fit(self.ratings_df)

        # 5. Initialize hybrid engine
        if self.content_model and self.cf_model:
            self.hybrid_model = HybridRecommender(
                content_model=self.content_model,
                cf_model=self.cf_model,
                content_weight=0.5,
            )

        # 6. Load evaluation results
        eval_path = self.models_dir / "evaluation_results.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    self.eval_results = json.load(f)
            except Exception:
                self.eval_results = {}

    def get_all_genres(self) -> List[str]:
        """Return unique sorted list of all movie genres."""
        genre_set = set()
        for g_list in self.movies_df["genres"]:
            if isinstance(g_list, list):
                genre_set.update(g_list)
        return sorted([g for g in genre_set if g and g != "(no genres listed)"])

    def get_all_users(self) -> List[int]:
        """Return sorted list of user IDs in the dataset."""
        if self.ratings_df.empty:
            return []
        return sorted(self.ratings_df["userId"].unique().tolist())

    def get_user_ratings(self, user_id: int) -> Dict[int, float]:
        """Return dictionary of {movieId: rating} for a given user."""
        if self.ratings_df.empty:
            return {}
        user_data = self.ratings_df[self.ratings_df["userId"] == user_id]
        return dict(zip(user_data["movieId"], user_data["rating"]))

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Fetch complete metadata record for a movie."""
        matches = self.movies_df[self.movies_df["movieId"] == movie_id]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return row.to_dict()

    def get_popular_movies(self, limit: int = 12, genre: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return most-rated movies, optionally filtered by genre."""
        df = self.movies_df
        if genre and genre != "All":
            df = df[df["genres"].apply(lambda g: genre in g if isinstance(g, list) else False)]
        top = df.sort_values(by="vote_count", ascending=False).head(limit)
        return top.to_dict(orient="records")

    def get_highest_rated_movies(self, limit: int = 12, genre: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return highest-rated movies using Bayesian weighted rating safeguard."""
        df = self.movies_df
        if genre and genre != "All":
            df = df[df["genres"].apply(lambda g: genre in g if isinstance(g, list) else False)]
        top = df.sort_values(by="weighted_rating", ascending=False).head(limit)
        return top.to_dict(orient="records")

    def search_movies(
        self,
        query: str = "",
        genre: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        min_rating: float = 0.0,
        limit: int = 24,
    ) -> List[Dict[str, Any]]:
        """Search and filter movies by title query, genre, release years, and minimum rating."""
        df = self.movies_df

        if query and query.strip():
            q = query.strip().lower()
            df = df[
                df["clean_title"].str.lower().str.contains(q, regex=False)
                | df["title"].str.lower().str.contains(q, regex=False)
            ]

        if genre and genre != "All":
            df = df[df["genres"].apply(lambda g: genre in g if isinstance(g, list) else False)]

        if year_range:
            min_y, max_y = year_range
            df = df[(df["year"].isna()) | ((df["year"] >= min_y) & (df["year"] <= max_y))]

        if min_rating > 0.0:
            df = df[df["vote_average"] >= min_rating]

        # Prioritize by popularity/ratings count
        results = df.sort_values(by=["vote_count", "vote_average"], ascending=[False, False]).head(limit)
        return results.to_dict(orient="records")

    def recommend_similar(self, movie_identifier: Union[str, int], top_n: int = 10) -> List[Dict[str, Any]]:
        """Content-based similar movie recommendation."""
        if not self.content_model:
            return []
        return self.content_model.recommend_similar_movies(movie_identifier, top_n=top_n)

    def recommend_collaborative(
        self, user_id: int, top_n: int = 10, exclude_ids: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Collaborative filtering personalized recommendation."""
        if not self.cf_model:
            return []
        return self.cf_model.recommend_for_user(
            user_id=user_id,
            top_n=top_n,
            movies_df=self.movies_df,
            exclude_ids=exclude_ids,
        )

    def recommend_hybrid(
        self,
        user_ratings: Dict[int, float],
        user_id: Optional[int] = None,
        content_weight: float = 0.5,
        top_n: int = 10,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid recommendation blending content similarity and collaborative ratings."""
        if not self.hybrid_model:
            return []
        self.hybrid_model.set_weights(content_weight)
        return self.hybrid_model.recommend(
            user_ratings=user_ratings,
            user_id=user_id,
            top_n=top_n,
            exclude_ids=exclude_ids,
        )
