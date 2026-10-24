"""Prepare, validate, and preprocess the MovieLens dataset for CineMatch."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, validate_dataset_presence, get_default_data_dir
from src.preprocessing import prepare_movie_metadata


def main() -> None:
    data_dir = get_default_data_dir()
    print("=" * 60)
    print(" CineMatch: Data Preparation & Validation Pipeline")
    print("=" * 60)
    print(f"Dataset directory: {data_dir}")

    is_valid, missing = validate_dataset_presence(data_dir)
    if not is_valid:
        print(f"[ERROR] Missing dataset files: {missing}")
        print("Please place 'movies.csv', 'ratings.csv', 'tags.csv', and 'links.csv' inside data/")
        sys.exit(1)

    print("[1/3] Loading raw MovieLens CSV files...")
    raw_data = load_raw_data(data_dir)
    movies_df = raw_data["movies"]
    ratings_df = raw_data["ratings"]
    tags_df = raw_data["tags"]
    links_df = raw_data["links"]

    print(f"      - Movies count:  {len(movies_df):,}")
    print(f"      - Ratings count: {len(ratings_df):,}")
    print(f"      - Tags count:    {len(tags_df):,}")
    print(f"      - Links count:   {len(links_df):,}")
    print(f"      - Unique users:  {ratings_df['userId'].nunique():,}")

    n_users = ratings_df["userId"].nunique()
    n_movies = ratings_df["movieId"].nunique()
    total_possible = n_users * n_movies
    actual_ratings = len(ratings_df)
    sparsity = (1.0 - (actual_ratings / total_possible)) * 100.0
    print(f"      - Interaction Matrix Sparsity: {sparsity:.2f}%")

    print("\n[2/3] Preprocessing and enriching movie metadata...")
    processed_movies = prepare_movie_metadata(
        movies_df=movies_df,
        tags_df=tags_df,
        ratings_df=ratings_df,
        links_df=links_df
    )

    out_path = data_dir / "movies_processed.csv"
    processed_movies.to_csv(out_path, index=False)
    print(f"      - Preprocessed metadata saved to: {out_path}")

    print("\n[3/3] Validation check on sample processed records:")
    sample = processed_movies[["movieId", "clean_title", "year", "genres_str", "vote_count", "vote_average", "weighted_rating"]].head(5)
    print(sample.to_string(index=False))

    print("\n[SUCCESS] Data preparation completed successfully.")


if __name__ == "__main__":
    main()
