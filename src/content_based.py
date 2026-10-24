"""Content-Based Recommendation Engine using TF-IDF and Cosine Similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """Recommender utilizing TF-IDF text features and on-demand cosine similarity.

    Memory-efficient design: stores sparse TF-IDF feature matrix rather than a
    precomputed dense N x N pairwise similarity matrix.
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
        )
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.movie_id_to_idx: Dict[int, int] = {}
        self.title_to_id: Dict[str, int] = {}
        self.clean_title_to_id: Dict[str, int] = {}

    def fit(self, movies_df: pd.DataFrame) -> ContentBasedRecommender:
        """Fit TF-IDF vectorizer on movie content soup and index mappings.

        Args:
            movies_df: Preprocessed DataFrame containing 'movieId', 'title',
                       'clean_title', 'genres_str', and 'content_soup'.
        """
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Ensure content_soup exists and has no NaN
        if "content_soup" not in self.movies_df.columns:
            raise ValueError("movies_df must contain 'content_soup' column.")

        soups = self.movies_df["content_soup"].fillna("").astype(str)
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)

        # Build lookup tables
        self.movie_id_to_idx = {
            int(row["movieId"]): idx for idx, row in self.movies_df.iterrows()
        }
        self.title_to_id = {
            str(row["title"]).strip().lower(): int(row["movieId"])
            for _, row in self.movies_df.iterrows()
        }
        self.clean_title_to_id = {
            str(row["clean_title"]).strip().lower(): int(row["movieId"])
            for _, row in self.movies_df.iterrows()
        }
        return self

    def find_movie_id(self, query: Union[str, int]) -> Optional[int]:
        """Find matching movieId for an ID or search string."""
        if isinstance(query, (int, np.integer)):
            return int(query) if int(query) in self.movie_id_to_idx else None

        q_str = str(query).strip().lower()

        # Check direct clean_title or title match
        if q_str in self.clean_title_to_id:
            return self.clean_title_to_id[q_str]
        if q_str in self.title_to_id:
            return self.title_to_id[q_str]

        # Partial matching
        for title, m_id in self.clean_title_to_id.items():
            if q_str == title or q_str in title:
                return m_id
        for title, m_id in self.title_to_id.items():
            if q_str in title:
                return m_id

        return None

    def recommend_similar_movies(
        self,
        movie_identifier: Union[str, int],
        top_n: int = 10,
        exclude_ids: Optional[set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend top_n movies most similar to the given movie.

        Args:
            movie_identifier: Movie title string or movieId integer.
            top_n: Number of recommendations to return (default 10).
            exclude_ids: Optional set of movie IDs to exclude (e.g., user seen).

        Returns:
            List of dictionaries containing movie details and similarity scores.
        """
        if self.tfidf_matrix is None or self.movies_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        m_id = self.find_movie_id(movie_identifier)
        if m_id is None:
            raise ValueError(f"Movie '{movie_identifier}' not found in dataset.")

        idx = self.movie_id_to_idx[m_id]
        query_vector = self.tfidf_matrix[idx]  # Shape: (1, num_features)

        # On-demand cosine similarity (since TF-IDF vectors are L2-normalized)
        sim_scores = linear_kernel(query_vector, self.tfidf_matrix).flatten()

        # Sort indices by descending similarity
        ranked_indices = np.argsort(sim_scores)[::-1]

        target_movie_row = self.movies_df.iloc[idx]
        target_genres = set(target_movie_row.get("genres", []))

        results: List[Dict[str, Any]] = []
        excluded = {m_id}
        if exclude_ids:
            excluded.update(exclude_ids)

        for rank_idx in ranked_indices:
            candidate_row = self.movies_df.iloc[rank_idx]
            cand_id = int(candidate_row["movieId"])

            if cand_id in excluded:
                continue

            score = float(sim_scores[rank_idx])
            # If similarity is virtually 0, stop adding
            if score <= 0.0001 and len(results) >= top_n:
                break

            cand_genres = set(candidate_row.get("genres", []))
            common_genres = list(target_genres.intersection(cand_genres))
            genre_reason = (
                f"Shares genres: {', '.join(common_genres)}"
                if common_genres
                else "Content & theme similarity"
            )

            year_val = candidate_row.get("year", None) if "year" in candidate_row else None
            results.append({
                "movieId": cand_id,
                "title": candidate_row.get("title", f"Movie #{cand_id}"),
                "clean_title": candidate_row.get("clean_title", f"Movie #{cand_id}"),
                "year": year_val if (year_val is not None and pd.notna(year_val)) else None,
                "genres_str": candidate_row.get("genres_str", "Unknown"),
                "vote_average": float(candidate_row.get("vote_average", 0.0)),
                "vote_count": int(candidate_row.get("vote_count", 0)),
                "similarity_score": round(score, 4),
                "reason": genre_reason,
                "imdbId": candidate_row.get("imdbId", ""),
                "tmdbId": candidate_row.get("tmdbId", ""),
            })

            if len(results) >= top_n:
                break

        return results

    def recommend_from_profile(
        self,
        movie_ratings: Dict[int, float],
        top_n: int = 10,
        exclude_ids: Optional[set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate content recommendations from a weighted user preference profile.

        Args:
            movie_ratings: Dict mapping movieId to rating value (e.g. 1.0 to 5.0).
            top_n: Number of recommendations to return.
            exclude_ids: Set of movie IDs to exclude.

        Returns:
            List of recommendations matching the aggregated preference profile.
        """
        if self.tfidf_matrix is None or self.movies_df is None:
            raise RuntimeError("Model has not been fitted yet.")

        valid_preferences = [
            (self.movie_id_to_idx[mid], rating)
            for mid, rating in movie_ratings.items()
            if mid in self.movie_id_to_idx
        ]

        if not valid_preferences:
            return []

        # Center ratings around neutral 3.0 so negative preferences downweight features
        profile_vec = np.zeros((1, self.tfidf_matrix.shape[1]), dtype=np.float32)
        total_weight = 0.0

        for idx, rating in valid_preferences:
            weight = max(0.5, float(rating))
            row_vec = self.tfidf_matrix[idx].toarray()
            profile_vec += weight * row_vec
            total_weight += weight

        if total_weight > 0:
            profile_vec /= total_weight
            norm = np.linalg.norm(profile_vec)
            if norm > 0:
                profile_vec /= norm

        sim_scores = linear_kernel(profile_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(sim_scores)[::-1]

        results: List[Dict[str, Any]] = []
        excluded = set(movie_ratings.keys())
        if exclude_ids:
            excluded.update(exclude_ids)

        for rank_idx in ranked_indices:
            row = self.movies_df.iloc[rank_idx]
            m_id = int(row["movieId"])
            if m_id in excluded:
                continue

            score = float(sim_scores[rank_idx])
            year_val = row.get("year", None) if "year" in row else None
            results.append({
                "movieId": m_id,
                "title": row.get("title", f"Movie #{m_id}"),
                "clean_title": row.get("clean_title", f"Movie #{m_id}"),
                "year": year_val if (year_val is not None and pd.notna(year_val)) else None,
                "genres_str": row.get("genres_str", "Unknown"),
                "vote_average": float(row.get("vote_average", 0.0)),
                "vote_count": int(row.get("vote_count", 0)),
                "similarity_score": round(score, 4),
                "reason": "Matches your taste profile across genres and themes",
                "imdbId": row.get("imdbId", ""),
                "tmdbId": row.get("tmdbId", ""),
            })

            if len(results) >= top_n:
                break

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist model state and indices to disk."""
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "movies_df": self.movies_df,
            "movie_id_to_idx": self.movie_id_to_idx,
            "title_to_id": self.title_to_id,
            "clean_title_to_id": self.clean_title_to_id,
        }
        joblib.dump(data, filepath, compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ContentBasedRecommender:
        """Load persisted model state from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.tfidf_matrix = data["tfidf_matrix"]
        instance.movies_df = data["movies_df"]
        instance.movie_id_to_idx = data["movie_id_to_idx"]
        instance.title_to_id = data["title_to_id"]
        instance.clean_title_to_id = data["clean_title_to_id"]
        return instance
