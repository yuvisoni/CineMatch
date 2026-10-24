"""Run rigorous model evaluation on an 80/20 train/test split and save metrics."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_default_data_dir
from src.content_based import ContentBasedRecommender
from src.evaluation import (
    compute_prediction_metrics,
    compute_baseline_prediction_metrics,
    evaluate_ranking_models,
    save_evaluation_results,
)


def main() -> None:
    print("=" * 65)
    print(" CineMatch: Model Evaluation & Benchmarking Pipeline")
    print("=" * 65)

    data_dir = get_default_data_dir()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies_processed.csv"

    print("[1/4] Loading ratings and preprocessed movies...")
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    # Convert genres string representation back if needed
    if "genres" in movies_df.columns and isinstance(movies_df["genres"].iloc[0], str):
        import ast
        try:
            movies_df["genres"] = movies_df["genres"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
            )
        except Exception:
            pass

    # 2. Strict 80/20 Train/Test Split (ensuring no data leakage)
    print("\n[2/4] Splitting ratings into 80% Train / 20% Test (random_state=42)...")
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=0.20, random_state=42, stratify=None
    )
    print(f"      - Train ratings count: {len(train_ratings):,}")
    print(f"      - Test ratings count:  {len(test_ratings):,}")

    # Train SVD exclusively on the training partition
    print("      - Training SVD strictly on training set partition...")
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(
        train_ratings[["userId", "movieId", "rating"]], reader=reader
    )
    trainset = train_data.build_full_trainset()
    eval_svd = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)
    eval_svd.fit(trainset)

    # Load or fit content recommender
    cb_model_path = models_dir / "content_model.joblib"
    if cb_model_path.exists():
        cb_recommender = ContentBasedRecommender.load(cb_model_path)
    else:
        cb_recommender = ContentBasedRecommender(max_features=10000, ngram_range=(1, 2))
        cb_recommender.fit(movies_df)

    # 3. Rating Prediction Evaluation
    print("\n[3/4] Evaluating Rating Prediction Accuracy (RMSE & MAE)...")
    baseline_metrics = compute_baseline_prediction_metrics(train_ratings, test_ratings)
    svd_metrics = compute_prediction_metrics(eval_svd, test_ratings)

    print(f"      - Global Mean Baseline -> RMSE: {baseline_metrics['rmse']:.4f}, MAE: {baseline_metrics['mae']:.4f}")
    print(f"      - SVD (Collaborative)  -> RMSE: {svd_metrics['rmse']:.4f}, MAE: {svd_metrics['mae']:.4f}")
    rmse_improvement = ((baseline_metrics['rmse'] - svd_metrics['rmse']) / baseline_metrics['rmse']) * 100
    print(f"      - SVD Error Reduction:   {rmse_improvement:.2f}% improvement over baseline")

    # 4. Recommendation Ranking Evaluation (Precision@10, Recall@10, NDCG@10)
    print("\n[4/4] Evaluating Recommendation Ranking Performance (@10)...")
    print("      (Ground-truth relevant items: user test ratings >= 4.0)")
    ranking_results = evaluate_ranking_models(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        cf_algo=eval_svd,
        content_recommender=cb_recommender,
        k=10,
        relevance_threshold=4.0,
        sample_users=100,
        random_state=42,
    )

    print("\n" + "=" * 65)
    print(f"{'Algorithm / Model':<26} | {'Precision@10':<12} | {'Recall@10':<10} | {'NDCG@10':<10}")
    print("-" * 65)
    for model_name, metrics in ranking_results.items():
        name_display = model_name.replace("_", " ").title()
        p = metrics["precision@10"]
        r = metrics["recall@10"]
        n = metrics["ndcg@10"]
        print(f"{name_display:<26} | {p:<12.4f} | {r:<10.4f} | {n:<10.4f}")
    print("=" * 65)

    # Save results to json
    full_results = {
        "dataset": {
            "name": "MovieLens Latest Small",
            "total_ratings": len(ratings_df),
            "train_ratings": len(train_ratings),
            "test_ratings": len(test_ratings),
            "users": int(ratings_df["userId"].nunique()),
            "movies": int(ratings_df["movieId"].nunique()),
        },
        "rating_prediction": {
            "global_mean_baseline": baseline_metrics,
            "svd_collaborative": svd_metrics,
            "rmse_improvement_pct": round(rmse_improvement, 2),
        },
        "ranking_performance_at_10": ranking_results,
        "methodology": {
            "split": "80/20 train/test without cross-leakage",
            "relevance_threshold": 4.0,
            "top_k": 10,
            "evaluation_users": 100,
        },
    }

    results_file = models_dir / "evaluation_results.json"
    save_evaluation_results(full_results, results_file)
    print(f"\n[SAVED] Comprehensive evaluation metrics saved to: {results_file}")


if __name__ == "__main__":
    main()
