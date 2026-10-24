"""Data preprocessing and feature engineering for CineMatch."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_title_and_year(raw_title: str) -> Tuple[str, Optional[int]]:
    """Extract clean title and release year from MovieLens title format.

    Examples:
        'Toy Story (1995)' -> ('Toy Story', 1995)
        'American President, The (1995)' -> ('The American President', 1995)
        'Blade Runner (a.k.a. Dangerous Days) (1982)' -> ('Blade Runner', 1982)
        'Unknown Movie' -> ('Unknown Movie', None)
    """
    if not isinstance(raw_title, str) or not raw_title.strip():
        return "Unknown Movie", None

    raw_title = raw_title.strip()

    # Match year in parentheses at the end of the title
    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    year = int(year_match.group(1)) if year_match else None

    # Remove the year from the title
    title_no_year = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip()

    # Handle alternate titles if any: e.g. "Title (a.k.a. Alt)" -> keep primary or clean
    title_no_aka = re.sub(r"\s*\(a\.k\.a\..*?\)\s*", " ", title_no_year).strip()

    # Handle trailing articles: 'Matrix, The' -> 'The Matrix', 'Lover, A' -> 'A Lover'
    article_match = re.search(r"^(.*?),\s*(The|A|An)\s*$", title_no_aka, re.IGNORECASE)
    if article_match:
        main_part = article_match.group(1).strip()
        article = article_match.group(2).strip().capitalize()
        cleaned_title = f"{article} {main_part}"
    else:
        cleaned_title = title_no_aka

    return cleaned_title, year


def parse_genres(genres_str: str) -> List[str]:
    """Parse pipe-separated genres string into a list of normalized genres."""
    if not isinstance(genres_str, str) or not genres_str.strip():
        return []
    if genres_str.strip() == "(no genres listed)":
        return []
    return [g.strip() for g in genres_str.split("|") if g.strip()]


def clean_text_token(text: str) -> str:
    """Normalize text by lowering, removing non-alphanumeric chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tags by movieId into a cleaned string of unique tags."""
    if tags_df is None or tags_df.empty:
        return pd.DataFrame(columns=["movieId", "tag_list", "tags_text"])

    valid_tags = tags_df.dropna(subset=["tag", "movieId"]).copy()
    valid_tags["clean_tag"] = valid_tags["tag"].astype(str).apply(clean_text_token)
    valid_tags = valid_tags[valid_tags["clean_tag"] != ""]

    def combine_tags(series: pd.Series) -> str:
        # Preserve uniqueness while preserving order
        seen = set()
        unique_tokens = []
        for tag in series:
            words = tag.split()
            for w in words:
                if w not in seen and len(w) > 1:
                    seen.add(w)
                    unique_tokens.append(w)
        return " ".join(unique_tokens)

    aggregated = (
        valid_tags.groupby("movieId")["clean_tag"]
        .apply(combine_tags)
        .reset_index(name="tags_text")
    )
    return aggregated


def compute_movie_statistics(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rating count, average rating, and Bayesian weighted score for each movie."""
    if ratings_df is None or ratings_df.empty:
        df = movies_df[["movieId"]].copy()
        df["vote_count"] = 0
        df["vote_average"] = 0.0
        df["weighted_rating"] = 0.0
        return df

    stats = ratings_df.groupby("movieId")["rating"].agg(
        vote_count="count",
        vote_average="mean"
    ).reset_index()

    # Calculate Bayesian weighted rating: (v / (v + m)) * R + (m / (v + m)) * C
    C = ratings_df["rating"].mean()
    m = stats["vote_count"].quantile(0.85)  # 85th percentile threshold safeguard

    # Safeguard if m is 0 or low
    m = max(5.0, float(m))

    def weighted_score(row: pd.Series) -> float:
        v = row["vote_count"]
        R = row["vote_average"]
        return float((v / (v + m)) * R + (m / (v + m)) * C)

    stats["weighted_rating"] = stats.apply(weighted_score, axis=1)
    return stats


def prepare_movie_metadata(
    movies_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
    ratings_df: Optional[pd.DataFrame] = None,
    links_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Enrich raw movies dataframe with cleaned titles, parsed genres, tags, statistics, and links.

    Returns:
        DataFrame with columns:
        ['movieId', 'title', 'clean_title', 'year', 'genres', 'genres_str',
         'tags_text', 'content_soup', 'vote_count', 'vote_average', 'weighted_rating',
         'imdbId', 'tmdbId']
    """
    df = movies_df.copy()

    # Clean titles and extract years
    title_year = df["title"].apply(extract_title_and_year)
    df["clean_title"] = [t[0] for t in title_year]
    df["year"] = [t[1] for t in title_year]

    # Parse genres
    df["genres"] = df["genres"].apply(parse_genres)
    df["genres_str"] = df["genres"].apply(lambda g: ", ".join(g) if g else "Unknown")

    # Aggregate and merge tags
    if tags_df is not None and not tags_df.empty:
        agg_tags = aggregate_tags(tags_df)
        df = df.merge(agg_tags, on="movieId", how="left")
        df["tags_text"] = df["tags_text"].fillna("")
    else:
        df["tags_text"] = ""

    # Rating statistics
    if ratings_df is not None and not ratings_df.empty:
        stats = compute_movie_statistics(ratings_df, df)
        df = df.merge(stats, on="movieId", how="left")
        df["vote_count"] = df["vote_count"].fillna(0).astype(int)
        df["vote_average"] = df["vote_average"].fillna(0.0).round(2)
        df["weighted_rating"] = df["weighted_rating"].fillna(0.0).round(2)
    else:
        df["vote_count"] = 0
        df["vote_average"] = 0.0
        df["weighted_rating"] = 0.0

    # Links (imdb, tmdb)
    if links_df is not None and not links_df.empty:
        df = df.merge(links_df, on="movieId", how="left")
        df["imdbId"] = df["imdbId"].fillna("").astype(str).apply(
            lambda x: f"tt{int(float(x)):07d}" if x and x != "nan" and x.replace('.', '', 1).isdigit() else ""
        )
        df["tmdbId"] = df["tmdbId"].fillna("").astype(str).apply(
            lambda x: str(int(float(x))) if x and x != "nan" and x.replace('.', '', 1).isdigit() else ""
        )
    else:
        df["imdbId"] = ""
        df["tmdbId"] = ""

    # Build content soup:
    # 1. Genres repeated twice to boost genre importance
    # 2. Aggregated tags
    # 3. Clean title tokens
    def build_soup(row: pd.Series) -> str:
        genre_tokens = " ".join([clean_text_token(g) for g in row["genres"]])
        # Weight genres by repeating them
        weighted_genres = f"{genre_tokens} {genre_tokens}"
        tags = row["tags_text"]
        title_tokens = clean_text_token(row["clean_title"])
        parts = [weighted_genres, tags, title_tokens]
        return " ".join([p for p in parts if p]).strip()

    df["content_soup"] = df.apply(build_soup, axis=1)

    return df
