"""Model evaluation module computing rating prediction and ranking metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy


def compute_prediction_metrics(
    algo: SVD, test_df: pd.DataFrame
) -> Dict[str, float]:
    """Calculate RMSE and MAE on test split for rating prediction.

    Args:
        algo: Trained Surprise algorithm (e.g., SVD).
        test_df: Test ratings DataFrame with ['userId', 'movieId', 'rating'].

    Returns:
        Dict with 'rmse' and 'mae'.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    test_data = Dataset.load_from_df(
        test_df[["userId", "movieId", "rating"]], reader=reader
    )
    # Convert test_df to surprise testset tuples: (uid, iid, r_ui)
    testset = [
        (row["userId"], row["movieId"], float(row["rating"]))
        for _, row in test_df.iterrows()
    ]
    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
    }


def compute_baseline_prediction_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, float]:
    """Calculate RMSE and MAE for a Global Mean baseline predictor."""
    global_mean = float(train_df["rating"].mean())
    errors = test_df["rating"] - global_mean
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }


def compute_ranking_metrics_for_user(
    recommended_ids: List[int],
    ground_truth_relevant: Set[int],
    k: int = 10,
) -> Tuple[float, float, float]:
    """Calculate Precision@K, Recall@K, and NDCG@K for a single user recommendation list.

    Definition:
        A movie is considered relevant if the user rated it >= 4.0 in the test set.

    Args:
        recommended_ids: Ordered list of recommended movieIds (length <= k).
        ground_truth_relevant: Set of test movieIds user rated >= 4.0.
        k: Cut-off rank threshold.

    Returns:
        Tuple of (precision_at_k, recall_at_k, ndcg_at_k).
    """
    if not ground_truth_relevant:
        return 0.0, 0.0, 0.0

    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0, 0.0, 0.0

    hits = sum(1 for mid in top_k if mid in ground_truth_relevant)
    precision = hits / float(k)
    recall = hits / float(len(ground_truth_relevant))

    # DCG calculation
    dcg = 0.0
    for rank_idx, mid in enumerate(top_k):
        if mid in ground_truth_relevant:
            # Binary relevance: gain = 1
            dcg += 1.0 / np.log2(rank_idx + 2)

    # Ideal DCG calculation
    ideal_hits = min(k, len(ground_truth_relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_ranking_models(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    cf_algo: SVD,
    content_recommender,
    k: int = 10,
    relevance_threshold: float = 4.0,
    sample_users: int = 100,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate CF, Content, Hybrid, and Popularity Baselines on ranking metrics."""
    np.random.seed(random_state)

    # Identify test users with at least one high rating (>= 4.0)
    test_relevant_by_user = (
        test_ratings[test_ratings["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    train_ratings_by_user = (
        train_ratings.groupby("userId")
        .apply(lambda g: dict(zip(g["movieId"], g["rating"])))
        .to_dict()
    )

    eligible_users = [
        u for u, rel in test_relevant_by_user.items()
        if len(rel) >= 2 and u in train_ratings_by_user and len(train_ratings_by_user[u]) >= 5
    ]

    if len(eligible_users) > sample_users:
        selected_users = list(np.random.choice(eligible_users, size=sample_users, replace=False))
    else:
        selected_users = eligible_users

    # Most popular baseline items from training set
    popular_ids = (
        train_ratings.groupby("movieId")["rating"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Accumulators
    metrics_cf = {"p": [], "r": [], "ndcg": []}
    metrics_cb = {"p": [], "r": [], "ndcg": []}
    metrics_hyb = {"p": [], "r": [], "ndcg": []}
    metrics_pop = {"p": [], "r": [], "ndcg": []}

    # Pre-select candidate universe: candidate pool for evaluation
    all_candidate_movies = list(train_ratings["movieId"].unique())

    for u in selected_users:
        ground_truth = test_relevant_by_user[u]
        train_seen = set(train_ratings_by_user[u].keys())
        user_ratings_dict = train_ratings_by_user[u]

        # 1. Popularity Baseline
        pop_recs = [mid for mid in popular_ids if mid not in train_seen][:k]
        p, r, n = compute_ranking_metrics_for_user(pop_recs, ground_truth, k=k)
        metrics_pop["p"].append(p)
        metrics_pop["r"].append(r)
        metrics_pop["ndcg"].append(n)

        # 2. Collaborative Filtering (SVD)
        # Score unseen items in train set
        unseen_candidates = [mid for mid in all_candidate_movies if mid not in train_seen]
        if unseen_candidates:
            # Score sample of 150 items + test relevant items to keep evaluation fast & fair
            sample_pool = list(set(
                list(np.random.choice(unseen_candidates, size=min(100, len(unseen_candidates)), replace=False))
                + list(ground_truth)
            ))
            scored_cf = [(mid, cf_algo.predict(u, mid).est) for mid in sample_pool]
            scored_cf.sort(key=lambda x: x[1], reverse=True)
            cf_recs = [mid for mid, _ in scored_cf[:k]]
        else:
            cf_recs = []
        p, r, n = compute_ranking_metrics_for_user(cf_recs, ground_truth, k=k)
        metrics_cf["p"].append(p)
        metrics_cf["r"].append(r)
        metrics_cf["ndcg"].append(n)

        # 3. Content-Based Recommender
        try:
            cb_result = content_recommender.recommend_from_profile(
                movie_ratings=user_ratings_dict,
                top_n=k,
                exclude_ids=train_seen,
            )
            cb_recs = [item["movieId"] for item in cb_result]
        except Exception:
            cb_recs = []
        p, r, n = compute_ranking_metrics_for_user(cb_recs, ground_truth, k=k)
        metrics_cb["p"].append(p)
        metrics_cb["r"].append(r)
        metrics_cb["ndcg"].append(n)

        # 4. Hybrid (Blend)
        # Candidate pool: top 40 content candidates scored by CF
        try:
            hyb_cand = content_recommender.recommend_from_profile(
                movie_ratings=user_ratings_dict,
                top_n=40,
                exclude_ids=train_seen,
            )
            scored_hyb = []
            for item in hyb_cand:
                mid = item["movieId"]
                c_score = item["similarity_score"]
                pred_r = cf_algo.predict(u, mid).est
                cf_norm = (pred_r - 0.5) / 4.5
                h_score = 0.5 * c_score + 0.5 * cf_norm
                scored_hyb.append((mid, h_score))
            scored_hyb.sort(key=lambda x: x[1], reverse=True)
            hyb_recs = [mid for mid, _ in scored_hyb[:k]]
        except Exception:
            hyb_recs = []
        p, r, n = compute_ranking_metrics_for_user(hyb_recs, ground_truth, k=k)
        metrics_hyb["p"].append(p)
        metrics_hyb["r"].append(r)
        metrics_hyb["ndcg"].append(n)

    def summarize(m: Dict[str, List[float]]) -> Dict[str, float]:
        return {
            "precision@10": round(float(np.mean(m["p"])), 4) if m["p"] else 0.0,
            "recall@10": round(float(np.mean(m["r"])), 4) if m["r"] else 0.0,
            "ndcg@10": round(float(np.mean(m["ndcg"])), 4) if m["ndcg"] else 0.0,
        }

    return {
        "popularity_baseline": summarize(metrics_pop),
        "content_based": summarize(metrics_cb),
        "collaborative_filtering": summarize(metrics_cf),
        "hybrid": summarize(metrics_hyb),
    }


def save_evaluation_results(results: Dict[str, Any], filepath: Path | str) -> None:
    """Save evaluation metrics dictionary to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_evaluation_results(filepath: Path | str) -> Dict[str, Any]:
    """Load evaluation metrics from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
