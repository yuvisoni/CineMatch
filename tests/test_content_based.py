"""Unit tests for Content-Based Filtering recommender."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.content_based import ContentBasedRecommender


@pytest.fixture
def mock_movies_df() -> pd.DataFrame:
    """Fixture with a small controlled set of movies and features."""
    return pd.DataFrame([
        {
            "movieId": 1,
            "title": "Toy Story (1995)",
            "clean_title": "Toy Story",
            "year": 1995,
            "genres": ["Animation", "Children", "Comedy"],
            "genres_str": "Animation, Children, Comedy",
            "content_soup": "animation animation children children comedy comedy pixar toys woody buzz",
            "vote_average": 4.0,
            "vote_count": 100,
        },
        {
            "movieId": 2,
            "title": "A Bug's Life (1998)",
            "clean_title": "A Bug's Life",
            "year": 1998,
            "genres": ["Animation", "Children", "Comedy"],
            "genres_str": "Animation, Children, Comedy",
            "content_soup": "animation animation children children comedy comedy pixar insects ants",
            "vote_average": 3.8,
            "vote_count": 80,
        },
        {
            "movieId": 3,
            "title": "Toy Story 2 (1999)",
            "clean_title": "Toy Story 2",
            "year": 1999,
            "genres": ["Animation", "Children", "Comedy"],
            "genres_str": "Animation, Children, Comedy",
            "content_soup": "animation animation children children comedy comedy pixar toys buzz woody sequel",
            "vote_average": 3.9,
            "vote_count": 90,
        },
        {
            "movieId": 4,
            "title": "The Matrix (1999)",
            "clean_title": "The Matrix",
            "year": 1999,
            "genres": ["Action", "Sci-Fi"],
            "genres_str": "Action, Sci-Fi",
            "content_soup": "action action sci-fi sci-fi cyber simulation virtual reality keanu neo",
            "vote_average": 4.5,
            "vote_count": 150,
        },
        {
            "movieId": 5,
            "title": "Matrix Reloaded, The (2003)",
            "clean_title": "The Matrix Reloaded",
            "year": 2003,
            "genres": ["Action", "Sci-Fi"],
            "genres_str": "Action, Sci-Fi",
            "content_soup": "action action sci-fi sci-fi cyber simulation virtual reality sequel neo",
            "vote_average": 3.6,
            "vote_count": 70,
        },
    ])


def test_fit_and_recommend_excludes_input_movie(mock_movies_df):
    recommender = ContentBasedRecommender()
    recommender.fit(mock_movies_df)

    recs = recommender.recommend_similar_movies("Toy Story", top_n=3)

    assert len(recs) > 0
    # The input movie (movieId 1) MUST NOT appear in the recommendations
    rec_ids = [r["movieId"] for r in recs]
    assert 1 not in rec_ids

    # Top similar movies should be Toy Story 2 or A Bug's Life (Animation/Pixar), not The Matrix
    top_rec = recs[0]
    assert top_rec["movieId"] in (2, 3)
    assert top_rec["similarity_score"] > 0.0


def test_recommend_unknown_movie_raises_value_error(mock_movies_df):
    recommender = ContentBasedRecommender()
    recommender.fit(mock_movies_df)

    with pytest.raises(ValueError, match="not found"):
        recommender.recommend_similar_movies("Nonexistent Alien Movie 9999")


def test_recommend_top_n_constraint(mock_movies_df):
    recommender = ContentBasedRecommender()
    recommender.fit(mock_movies_df)

    recs = recommender.recommend_similar_movies(4, top_n=1)
    assert len(recs) == 1
    assert recs[0]["movieId"] == 5  # Matrix Reloaded


def test_recommend_from_profile_cold_start(mock_movies_df):
    recommender = ContentBasedRecommender()
    recommender.fit(mock_movies_df)

    # User profile likes The Matrix
    user_ratings = {4: 5.0}
    recs = recommender.recommend_from_profile(user_ratings, top_n=2)

    assert len(recs) > 0
    rec_ids = [r["movieId"] for r in recs]
    assert 4 not in rec_ids
    assert 5 in rec_ids  # The Matrix Reloaded matches profile


def test_model_persistence(mock_movies_df):
    recommender = ContentBasedRecommender()
    recommender.fit(mock_movies_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "test_cb.joblib"
        recommender.save(model_file)
        assert model_file.exists()

        loaded_recommender = ContentBasedRecommender.load(model_file)
        recs = loaded_recommender.recommend_similar_movies("Toy Story", top_n=2)
        assert len(recs) == 2
        assert recs[0]["movieId"] in (2, 3)
