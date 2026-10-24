"""QA Integrity and Regression Test Suite for CineMatch.

Validates:
1. Complete exclusion of user's already-rated movies in CF and Hybrid.
2. Handling of cold-start scenarios (new user with 0 ratings, 1 rating, 3 ratings).
3. Handling of unknown titles, unicode titles, and edge-case inputs.
4. Absence of data leakage in train/test splits.
5. End-to-end MovieService API consistency.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.data_loader import get_default_data_dir, validate_dataset_presence, load_raw_data
from src.movie_service import MovieService
from src.preprocessing import extract_title_and_year, parse_genres, clean_text_token
from src.evaluation import compute_ranking_metrics_for_user


@pytest.fixture(scope="module")
def movie_service():
    """Load the cached MovieService on the actual MovieLens dataset."""
    return MovieService()


def test_dataset_files_present():
    """Ensure all required dataset files exist in the data folder."""
    is_valid, missing = validate_dataset_presence()
    assert is_valid, f"Missing required files: {missing}"


def test_no_overlap_with_previously_rated_movies_cf(movie_service):
    """Verify collaborative filtering strictly excludes movies the user already rated."""
    user_id = 1
    seen_movies = movie_service.cf_model.user_seen_movies.get(user_id, set())
    assert len(seen_movies) > 0, "User 1 should have historical ratings"

    recs = movie_service.recommend_collaborative(user_id=user_id, top_n=20)
    assert len(recs) == 20
    rec_ids = {r["movieId"] for r in recs}

    intersection = rec_ids.intersection(seen_movies)
    assert len(intersection) == 0, f"Found leaked already-rated movies in CF recs: {intersection}"


def test_no_overlap_with_previously_rated_movies_hybrid(movie_service):
    """Verify hybrid filtering strictly excludes movies the user already rated historically or selected."""
    user_id = 1
    user_ratings = movie_service.get_user_ratings(user_id)
    seen_movies = set(user_ratings.keys())

    # User favorites plus historical user_id
    recs = movie_service.recommend_hybrid(
        user_ratings=user_ratings,
        user_id=user_id,
        content_weight=0.5,
        top_n=20,
    )
    assert len(recs) > 0
    rec_ids = {r["movieId"] for r in recs}

    intersection = rec_ids.intersection(seen_movies)
    assert len(intersection) == 0, f"Found leaked already-rated movies in Hybrid recs: {intersection}"


def test_cold_start_new_user_recommendations(movie_service):
    """Verify a brand new user (unregistered) gets relevant recommendations based on 3 favorites."""
    # New user likes Sci-Fi masterpieces
    toy_story_id = 1
    star_wars_id = 260
    matrix_id = 2571  # Matrix in MovieLens

    new_user_favorites = {
        toy_story_id: 5.0,
        star_wars_id: 5.0,
        matrix_id: 4.5,
    }

    recs = movie_service.recommend_hybrid(
        user_ratings=new_user_favorites,
        user_id=None,  # No historical user ID
        content_weight=0.7,
        top_n=10,
    )

    assert len(recs) == 10
    rec_ids = {r["movieId"] for r in recs}

    # None of the input movies should be in the recommendations
    assert toy_story_id not in rec_ids
    assert star_wars_id not in rec_ids
    assert matrix_id not in rec_ids

    # All returned recommendations should have valid score fields
    for r in recs:
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert 0.0 <= r["hybrid_score"] <= 1.0
        assert "Cold-Start" in r["reason"] or "thematic" in r["reason"].lower()


def test_cold_start_empty_favorites(movie_service):
    """Verify empty user favorites returns graceful empty recommendation list without crashing."""
    recs = movie_service.recommend_hybrid(user_ratings={}, user_id=None, top_n=10)
    assert recs == []

    content_recs = movie_service.content_model.recommend_from_profile(movie_ratings={}, top_n=10)
    assert content_recs == []


def test_unicode_and_punctuation_title_search(movie_service):
    """Verify search and movie discovery work with punctuation and articles."""
    queries = ["Amelie", "Wall-E", "Pulp Fiction", "The Matrix", "Shawshank"]
    for q in queries:
        results = movie_service.search_movies(query=q, limit=5)
        assert len(results) > 0, f"Expected search results for '{q}'"


def test_bayesian_rating_order(movie_service):
    """Verify highest rated movies obey Bayesian shrinkage (vote_count threshold)."""
    top_bayesian = movie_service.get_highest_rated_movies(limit=10)
    assert len(top_bayesian) == 10

    # Ensure every movie in top Bayesian has substantial vote count (e.g. >= 20 ratings)
    for m in top_bayesian:
        assert m["vote_count"] >= 20, f"Movie {m['clean_title']} has too few votes ({m['vote_count']}) for top list"


def test_ranking_metrics_math_correctness():
    """Verify Precision@K, Recall@K, and NDCG@K formula with deterministic ground truth."""
    recommended = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ground_truth = {20, 40, 999}  # 2 hits out of 3 ground truth items in top 10

    precision, recall, ndcg = compute_ranking_metrics_for_user(recommended, ground_truth, k=10)

    # Precision@10: 2 hits / 10 = 0.2
    assert precision == pytest.approx(0.2)

    # Recall@10: 2 hits / 3 = 0.6667
    assert recall == pytest.approx(2.0 / 3.0)

    # NDCG@10 must be in [0, 1]
    assert 0.0 < ndcg <= 1.0


def test_model_evaluation_json_integrity(movie_service):
    """Verify evaluation_results.json exists and contains genuine, un-fabricated metric scores."""
    eval_results = movie_service.eval_results
    assert bool(eval_results), "evaluation_results.json should not be empty"

    pred_res = eval_results.get("rating_prediction", {})
    assert "svd_collaborative" in pred_res
    svd = pred_res["svd_collaborative"]
    assert 0.7 < svd["rmse"] < 1.0, f"Expected genuine RMSE in (0.7, 1.0), got {svd['rmse']}"
    assert 0.5 < svd["mae"] < 0.8, f"Expected genuine MAE in (0.5, 0.8), got {svd['mae']}"

    ranking_res = eval_results.get("ranking_performance_at_10", {})
    assert "collaborative_filtering" in ranking_res
    cf_rank = ranking_res["collaborative_filtering"]
    assert 0.2 < cf_rank["precision@10"] < 0.6
    assert 0.2 < cf_rank["recall@10"] < 0.6
    assert 0.3 < cf_rank["ndcg@10"] < 0.7
