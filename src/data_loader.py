"""Data loading utilities for CineMatch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required MovieLens dataset files are not found."""
    pass


def get_default_data_dir() -> Path:
    """Return the default data directory path relative to project root."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data"


def validate_dataset_presence(data_dir: Optional[Path | str] = None) -> Tuple[bool, list[str]]:
    """Check if all required MovieLens CSV files exist.

    Args:
        data_dir: Directory where datasets should be located.

    Returns:
        Tuple of (all_present_bool, missing_files_list)
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    else:
        data_dir = Path(data_dir)

    required_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    missing = [f for f in required_files if not (data_dir / f).is_file()]
    return len(missing) == 0, missing


def load_raw_data(data_dir: Optional[Path | str] = None) -> Dict[str, pd.DataFrame]:
    """Load the four MovieLens CSV files into pandas DataFrames.

    Args:
        data_dir: Directory containing movies.csv, ratings.csv, tags.csv, links.csv.

    Returns:
        Dictionary with keys 'movies', 'ratings', 'tags', 'links'.

    Raises:
        DatasetNotFoundError: If any of the required CSV files are missing.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    else:
        data_dir = Path(data_dir)

    is_valid, missing = validate_dataset_presence(data_dir)
    if not is_valid:
        error_msg = (
            f"Required MovieLens dataset files are missing in '{data_dir}': {missing}.\n"
            "Please ensure you have placed the MovieLens Latest Small dataset CSVs\n"
            "('movies.csv', 'ratings.csv', 'tags.csv', 'links.csv') inside the 'data/' folder.\n"
            "You can download them from: https://grouplens.org/datasets/movielens/ (ml-latest-small.zip)."
        )
        raise DatasetNotFoundError(error_msg)

    movies = pd.read_csv(data_dir / "movies.csv", encoding="utf-8")
    ratings = pd.read_csv(data_dir / "ratings.csv", encoding="utf-8")
    tags = pd.read_csv(data_dir / "tags.csv", encoding="utf-8")
    links = pd.read_csv(data_dir / "links.csv", encoding="utf-8")

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
    }
