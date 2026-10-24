"""
Shared pytest fixtures for CineMatch test suite.
All fixtures use small, self-contained DataFrames — not the real dataset.
"""
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Small movie fixture (10 movies, diverse genres)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_movies_df():
    """10-movie fixture covering various genres."""
    data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'GoodFellas (1990)',
            'The Matrix (1999)',
            'Inception (2010)',
            'The Dark Knight (2008)',
            'Pulp Fiction (1994)',
            'Forrest Gump (1994)',
            'Interstellar (2014)',
            'The Godfather (1972)',
        ],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy',
            'Crime|Drama',
            'Action|Sci-Fi|Thriller',
            'Action|Crime|Drama|Mystery|Sci-Fi|Thriller',
            'Action|Crime|Drama|IMAX',
            'Crime|Drama',
            'Comedy|Drama|Romance',
            'Adventure|Drama|Sci-Fi',
            'Crime|Drama',
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_tags_df():
    """Small tags fixture."""
    data = {
        'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5],
        'movieId': [1, 4, 4, 5, 5, 6, 7, 10, 9],
        'tag': [
            'animation', 'mind-bending', 'sci-fi', 'dreams',
            'christopher nolan', 'batman', 'tarantino', 'mafia',
            'space',
        ],
        'timestamp': [1000000] * 9,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ratings_df():
    """~50 ratings from 5 users across 10 movies."""
    np.random.seed(42)
    rows = []
    user_ids = [1, 2, 3, 4, 5]
    movie_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    timestamp_base = 1000000

    for uid in user_ids:
        # Each user rates 8-10 movies
        n_ratings = np.random.randint(8, 11)
        rated_movies = np.random.choice(movie_ids, size=n_ratings, replace=False)
        for mid in rated_movies:
            rating = round(np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), 1)
            rows.append({
                'userId': uid,
                'movieId': mid,
                'rating': rating,
                'timestamp': timestamp_base + uid * 1000 + mid,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_links_df():
    """Small links fixture."""
    data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'imdbId': [114709, 113497, 96256, 133093, 1375666, 468569, 110912, 109830, 816692, 68646],
        'tmdbId': [862, 8844, 769, 603, 27205, 155, 680, 13, 157336, 238],
    }
    return pd.DataFrame(data)
