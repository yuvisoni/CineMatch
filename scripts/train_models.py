"""Train Content-Based and Collaborative Filtering models and persist artifacts."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, get_default_data_dir
from src.preprocessing import prepare_movie_metadata
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender


def main() -> None:
    print("=" * 60)
    print(" CineMatch: Model Training Pipeline")
    print("=" * 60)

    data_dir = get_default_data_dir()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    processed_movies_path = data_dir / "movies_processed.csv"
    if processed_movies_path.exists():
        print("[1/3] Loading preprocessed movie metadata...")
        movies_df = pd.read_csv(processed_movies_path)
        # Parse genres back to list
        if "genres" in movies_df.columns and isinstance(movies_df["genres"].iloc[0], str):
            import ast
            try:
                movies_df["genres"] = movies_df["genres"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
                )
            except Exception:
                pass
    else:
        print("[1/3] Preprocessed metadata not found. Processing raw data...")
        raw = load_raw_data(data_dir)
        movies_df = prepare_movie_metadata(
            raw["movies"], raw["tags"], raw["ratings"], raw["links"]
        )
        movies_df.to_csv(processed_movies_path, index=False)

    raw_ratings = pd.read_csv(data_dir / "ratings.csv")

    # 2. Train Content-Based Recommender
    print("\n[2/3] Fitting Content-Based TF-IDF Recommender...")
    start_cb = time.time()
    cb_recommender = ContentBasedRecommender(max_features=10000, ngram_range=(1, 2))
    cb_recommender.fit(movies_df)
    cb_path = models_dir / "content_model.joblib"
    cb_recommender.save(cb_path)
    print(f"      - TF-IDF Matrix shape: {cb_recommender.tfidf_matrix.shape}")
    print(f"      - Saved Content model to: {cb_path} ({time.time() - start_cb:.2f}s)")

    # 3. Train Collaborative Filtering (SVD)
    print("\n[3/3] Training Matrix Factorization (SVD) Recommender...")
    start_cf = time.time()
    cf_recommender = CollaborativeFilteringRecommender(
        n_factors=50,
        n_epochs=25,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )
    cf_recommender.fit(raw_ratings)
    cf_path = models_dir / "cf_model.joblib"
    cf_recommender.save(cf_path)
    print(f"      - Trained on {len(raw_ratings):,} ratings from {len(cf_recommender.known_users):,} users.")
    print(f"      - Global mean rating: {cf_recommender.global_mean_rating:.3f}")
    print(f"      - Saved Collaborative model to: {cf_path} ({time.time() - start_cf:.2f}s)")

    print("\n[SUCCESS] All CineMatch recommendation models trained and saved successfully.")


if __name__ == "__main__":
    main()
