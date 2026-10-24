"""Unit tests for Collaborative Filtering and Hybrid Recommender modules."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridRecommender


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Fixture with synthetic rating interactions."""
    return pd.DataFrame([
        # User 1 likes movies 1, 2, 3
        {"userId": 1, "movieId": 1, "rating": 5.0},
        {"userId": 1, "movieId": 2, "rating": 4.5},
        {"userId": 1, "movieId": 3, "rating": 4.0},
        # User 2 likes movies 1, 2
        {"userId": 2, "movieId": 1, "rating": 5.0},
        {"userId": 2, "movieId": 2, "rating": 4.0},
        # User 3 likes movie 4, 5
        {"userId": 3, "movieId": 4, "rating": 5.0},
        {"userId": 3, "movieId": 5, "rating": 4.5},
        # User 4 likes movie 3, 4
        {"userId": 4, "movieId": 3, "rating": 4.0},
        {"userId": 4, "movieId": 4, "rating": 4.5},
    ])


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"movieId": 1, "title": "Toy Story", "clean_title": "Toy Story", "genres": ["Animation"], "genres_str": "Animation", "content_soup": "animation toys buzz woody", "vote_average": 4.5, "vote_count": 2},
        {"movieId": 2, "title": "A Bug's Life", "clean_title": "A Bug's Life", "genres": ["Animation"], "genres_str": "Animation", "content_soup": "animation ants bugs", "vote_average": 4.0, "vote_count": 2},
        {"movieId": 3, "title": "Toy Story 2", "clean_title": "Toy Story 2", "genres": ["Animation"], "genres_str": "Animation", "content_soup": "animation toys buzz woody sequel", "vote_average": 4.0, "vote_count": 2},
        {"movieId": 4, "title": "The Matrix", "clean_title": "The Matrix", "genres": ["Action", "Sci-Fi"], "genres_str": "Action, Sci-Fi", "content_soup": "action sci-fi keanu neo cyber", "vote_average": 4.8, "vote_count": 2},
        {"movieId": 5, "title": "The Matrix Reloaded", "clean_title": "The Matrix Reloaded", "genres": ["Action", "Sci-Fi"], "genres_str": "Action, Sci-Fi", "content_soup": "action sci-fi keanu neo sequel", "vote_average": 4.5, "vote_count": 1},
    ])


def test_cf_excludes_already_rated_movies(sample_ratings_df, sample_movies_df):
    cf = CollaborativeFilteringRecommender(n_factors=10, n_epochs=10, random_state=42)
    cf.fit(sample_ratings_df)

    # User 1 has already rated 1, 2, 3
    recs = cf.recommend_for_user(user_id=1, top_n=2, movies_df=sample_movies_df)

    rec_ids = [r["movieId"] for r in recs]
    assert 1 not in rec_ids
    assert 2 not in rec_ids
    assert 3 not in rec_ids
    assert all(mid in (4, 5) for mid in rec_ids)


def test_cf_cold_start_unseen_user_detection(sample_ratings_df):
    cf = CollaborativeFilteringRecommender(n_factors=10, n_epochs=10, random_state=42)
    cf.fit(sample_ratings_df)

    assert cf.is_known_user(1) is True
    assert cf.is_known_user(9999) is False  # New unknown user


def test_cf_model_save_and_load(sample_ratings_df):
    cf = CollaborativeFilteringRecommender(n_factors=10, n_epochs=10, random_state=42)
    cf.fit(sample_ratings_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "cf_model.joblib"
        cf.save(save_path)
        assert save_path.exists()

        loaded_cf = CollaborativeFilteringRecommender.load(save_path)
        pred = loaded_cf.predict_rating(user_id=1, movie_id=4)
        assert 0.5 <= pred <= 5.0


def test_hybrid_recommender_formula_and_weights(sample_movies_df, sample_ratings_df):
    cb = ContentBasedRecommender()
    cb.fit(sample_movies_df)

    cf = CollaborativeFilteringRecommender(n_factors=10, n_epochs=10, random_state=42)
    cf.fit(sample_ratings_df)

    hybrid = HybridRecommender(content_model=cb, cf_model=cf, content_weight=0.7)
    assert hybrid.content_weight == 0.7
    assert hybrid.cf_weight == pytest.approx(0.3)

    # Test recommendations for known user 2
    user_ratings = {1: 5.0, 2: 4.0}
    recs = hybrid.recommend(user_ratings=user_ratings, user_id=2, top_n=2)

    assert len(recs) > 0
    for r in recs:
        # User already rated 1 and 2, so they must be excluded
        assert r["movieId"] not in (1, 2)
        # Genuine score fields must be present
        assert "content_score" in r
        assert "cf_score" in r
        assert "hybrid_score" in r
        assert 0.0 <= r["hybrid_score"] <= 1.0


def test_hybrid_cold_start_fallback_for_new_user(sample_movies_df, sample_ratings_df):
    cb = ContentBasedRecommender()
    cb.fit(sample_movies_df)

    cf = CollaborativeFilteringRecommender(n_factors=10, n_epochs=10, random_state=42)
    cf.fit(sample_ratings_df)

    hybrid = HybridRecommender(content_model=cb, cf_model=cf)

    # New visitor without a registered user_id
    new_user_favorites = {4: 5.0}  # Likes The Matrix
    recs = hybrid.recommend(user_ratings=new_user_favorites, user_id=None, top_n=2)

    assert len(recs) > 0
    rec_ids = [r["movieId"] for r in recs]
    assert 4 not in rec_ids
    # Matrix Reloaded (id 5) should be top recommended
    assert recs[0]["movieId"] == 5
    assert "Cold-Start" in recs[0]["reason"]
