"""Unit tests for CineMatch data preprocessing and feature extraction."""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    extract_title_and_year,
    parse_genres,
    clean_text_token,
    aggregate_tags,
    compute_movie_statistics,
    prepare_movie_metadata,
)


def test_extract_title_and_year_standard():
    title, year = extract_title_and_year("Toy Story (1995)")
    assert title == "Toy Story"
    assert year == 1995


def test_extract_title_and_year_with_article():
    title, year = extract_title_and_year("Shawshank Redemption, The (1994)")
    assert title == "The Shawshank Redemption"
    assert year == 1994

    title2, year2 = extract_title_and_year("Great Gatsby, A (1926)")
    assert title2 == "A Great Gatsby"
    assert year2 == 1926


def test_extract_title_and_year_missing_or_invalid():
    title, year = extract_title_and_year("Unknown Title Without Year")
    assert title == "Unknown Title Without Year"
    assert year is None

    title_none, year_none = extract_title_and_year("")
    assert title_none == "Unknown Movie"
    assert year_none is None


def test_parse_genres():
    assert parse_genres("Action|Adventure|Sci-Fi") == ["Action", "Adventure", "Sci-Fi"]
    assert parse_genres("(no genres listed)") == []
    assert parse_genres("") == []
    assert parse_genres(None) == []


def test_clean_text_token():
    assert clean_text_token("Sci-Fi & Action!") == "sci-fi action"
    assert clean_text_token("   Spaces   ") == "spaces"
    assert clean_text_token(None) == ""


def test_aggregate_tags():
    tags_data = pd.DataFrame({
        "userId": [1, 1, 2, 3],
        "movieId": [10, 10, 10, 20],
        "tag": ["mind-bending", "sci-fi", "mind-bending", "classic"],
        "timestamp": [100, 101, 102, 103],
    })
    agg = aggregate_tags(tags_data)
    assert len(agg) == 2
    row10 = agg[agg["movieId"] == 10].iloc[0]
    assert "mind-bending" in row10["tags_text"]
    assert "sci-fi" in row10["tags_text"]


def test_compute_movie_statistics_bayesian():
    movies = pd.DataFrame({"movieId": [1, 2, 3]})
    # Ratings where global mean is 3.0:
    # Movie 1 has five 5.0 ratings
    # Movie 2 has one 5.0 rating
    # Other movies have low ratings pulling down the global average
    ratings = pd.DataFrame({
        "userId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "movieId": [1, 1, 1, 1, 1, 2, 3, 3, 3, 3],
        "rating": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
    })
    stats = compute_movie_statistics(ratings, movies)
    # Movie 1 has 5 ratings of 5.0, Movie 2 has only 1 rating of 5.0.
    # Bayesian weighted rating shrinks Movie 2 closer to global mean (~3.1).
    m1_score = stats[stats["movieId"] == 1]["weighted_rating"].iloc[0]
    m2_score = stats[stats["movieId"] == 2]["weighted_rating"].iloc[0]
    assert m1_score > m2_score


def test_prepare_movie_metadata_integration():
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story (1995)", "Matrix, The (1999)"],
        "genres": ["Animation|Children|Comedy", "Action|Sci-Fi"],
    })
    tags = pd.DataFrame({
        "userId": [1],
        "movieId": [1],
        "tag": ["pixar fun"],
        "timestamp": [1000],
    })
    ratings = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [1, 2],
        "rating": [4.5, 5.0],
        "timestamp": [100, 200],
    })
    meta = prepare_movie_metadata(movies, tags, ratings)
    assert "clean_title" in meta.columns
    assert meta.loc[meta["movieId"] == 2, "clean_title"].iloc[0] == "The Matrix"
    assert "content_soup" in meta.columns
    assert len(meta) == 2
