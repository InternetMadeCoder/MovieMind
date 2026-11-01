"""
Comprehensive Evaluation Module for MovieMind.

Evaluates RL agents (Tabular Q-Learning, DQN) and Traditional Baselines
(Random, Popularity, Content-Based) on a consistent set of evaluation users
and candidate pools.

Clearly separates:
1. Held-Out Offline Recommendation Metrics (computed strictly on unseen test_df):
   - Precision@5, Precision@10
   - Recall@5, Recall@10
   - NDCG@5, NDCG@10
2. Diversity & Coverage Metrics:
   - Intra-List Diversity (ILD): mean pairwise cosine genre distance within top-K list
   - Genre Diversity: unique genres covered per user list (out of 18) & normalized Shannon entropy
   - Catalog Coverage: fraction of candidate pool items recommended across all evaluation users
3. Simulated Environment RL Metrics:
   - Average Cumulative Total Reward
   - Average Cumulative Preference Reward
   - Average Cumulative Diversity Bonus
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.data_preprocessing import PreprocessedData
from src.environment import MovieRecEnv


def compute_precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of top-k recommended items that are in the held-out relevant test set."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return float(hits) / float(k)


def compute_recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of held-out relevant test items recovered in the top-k recommendations."""
    if len(relevant) == 0 or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return float(hits) / float(len(relevant))


def compute_ndcg_at_k(
    recommended: List[int],
    relevant: Set[int],
    test_ratings: Dict[int, float],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K) on held-out test interactions.
    Uses binary relevance (1 if item in relevant set else 0) with graded tie-bonus from test rating.
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    top_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in relevant:
            # Standard binary relevance gain = 1.0
            dcg += 1.0 / np.log2(idx + 2.0)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2.0) for idx in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_intra_list_diversity(
    recommended: List[int], movie_genre_norm: np.ndarray
) -> float:
    """
    Compute Intra-List Diversity (ILD) as average pairwise cosine dissimilarity
    between genre vectors of recommended movies in a list.
    """
    n = len(recommended)
    if n <= 1:
        return 0.0
    vecs = movie_genre_norm[recommended]  # (n, num_genres)
    sim_matrix = vecs @ vecs.T
    # Extract upper triangle off-diagonal similarities
    triu_indices = np.triu_indices(n, k=1)
    pairwise_dissim = 1.0 - np.clip(sim_matrix[triu_indices], 0.0, 1.0)
    return float(np.mean(pairwise_dissim))


def compute_genre_diversity(
    recommended: List[int], movie_genre_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Genre Diversity metrics for a single recommendation list:
    1. Genre Coverage Ratio: fraction of all 18 genres present in the recommended list.
    2. Normalized Shannon Genre Entropy in [0, 1].
    """
    if len(recommended) == 0:
        return 0.0, 0.0
    num_genres = movie_genre_matrix.shape[1]
    genre_counts = movie_genre_matrix[recommended].sum(axis=0)
    unique_genres_covered = float(np.sum(genre_counts > 0)) / float(num_genres)

    total_tags = float(np.sum(genre_counts))
    if total_tags <= 0:
        return unique_genres_covered, 0.0

    probs = genre_counts[genre_counts > 0] / total_tags
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = float(np.log2(num_genres))
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return unique_genres_covered, norm_entropy


def select_evaluation_users(
    data: PreprocessedData,
    num_eval_users: int = 300,
    min_test_relevant: int = 3,
    seed: int = 42,
) -> List[int]:
    """
    Select a consistent cohort of evaluation users who have at least `min_test_relevant`
    positively rated (rating >= 4) movies in the held-out test split.
    """
    eligible_users = [
        u
        for u in range(data.num_users)
        if len(data.test_relevant_movies[u]) >= min_test_relevant
    ]
    if len(eligible_users) <= num_eval_users:
        return eligible_users
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible_users, size=num_eval_users, replace=False)
    return sorted(int(u) for u in chosen)


def generate_user_recommendations(
    env: MovieRecEnv,
    policy_fn: Callable[[np.ndarray, np.ndarray, int, int], int],
    user_idx: int,
) -> Dict[str, Any]:
    """
    Roll out one full recommendation episode (K=10 items) for `user_idx` using `policy_fn`.
    Returns recommended movie indices and per-step reward breakdown.
    """
    obs, info = env.reset(options={"user_idx": user_idx})
    s_disc = info["discrete_state"]
    done = False

    rec_movies: List[int] = []
    pref_rewards: List[float] = []
    div_bonuses: List[float] = []
    total_rewards: List[float] = []

    while not done:
        mask = env.get_action_mask()
        action = int(policy_fn(obs, mask, s_disc, user_idx))
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        s_disc = step_info["discrete_state"]

        rec_movies.append(int(step_info["movie_idx"]))
        pref_rewards.append(float(step_info["preference_reward"]))
        div_bonuses.append(float(step_info["diversity_bonus"]))
        total_rewards.append(float(reward))

    return {
        "user_idx": user_idx,
        "recommended_movies": rec_movies,
        "cumulative_reward": float(np.sum(total_rewards)),
        "cumulative_pref_reward": float(np.sum(pref_rewards)),
        "cumulative_div_bonus": float(np.sum(div_bonuses)),
    }


def evaluate_model(
    model_name: str,
    policy_fn: Callable[[np.ndarray, np.ndarray, int, int], int],
    data: PreprocessedData,
    eval_users: List[int],
    diversity_lambda: float = 0.25,
    episode_length: int = 10,
    use_candidate_pool: bool = True,
) -> Tuple[Dict[str, float], Dict[int, List[int]]]:
    """
    Evaluate a recommendation policy across all `eval_users` on both held-out test metrics
    and simulated environment rewards.
    """
    env = MovieRecEnv(
        data=data,
        episode_length=episode_length,
        diversity_lambda=diversity_lambda,
        use_candidate_pool=use_candidate_pool,
        seed=42,
    )

    p5_list, p10_list = [], []
    r5_list, r10_list = [], []
    ndcg5_list, ndcg10_list = [], []
    ild_list = []
    genre_cov_list, genre_ent_list = [], []
    cum_reward_list, cum_pref_list, cum_div_list = [], [], []

    all_recommended_items: Set[int] = set()
    user_recommendations: Dict[int, List[int]] = {}

    for u in eval_users:
        rollout = generate_user_recommendations(env, policy_fn, u)
        recs = rollout["recommended_movies"]
        user_recommendations[u] = recs
        all_recommended_items.update(recs)

        # Held-out relevant set (test movies with rating >= 4.0; fallback to all test movies if empty)
        relevant = data.test_relevant_movies[u]
        if len(relevant) == 0:
            relevant = set(data.test_all_movies[u].keys())
        test_ratings = data.test_all_movies[u]

        p5_list.append(compute_precision_at_k(recs, relevant, k=5))
        p10_list.append(compute_precision_at_k(recs, relevant, k=10))
        r5_list.append(compute_recall_at_k(recs, relevant, k=5))
        r10_list.append(compute_recall_at_k(recs, relevant, k=10))
        ndcg5_list.append(compute_ndcg_at_k(recs, relevant, test_ratings, k=5))
        ndcg10_list.append(compute_ndcg_at_k(recs, relevant, test_ratings, k=10))

        ild_list.append(compute_intra_list_diversity(recs, data.movie_genre_norm))
        g_cov, g_ent = compute_genre_diversity(recs, data.movie_genre_matrix)
        genre_cov_list.append(g_cov)
        genre_ent_list.append(g_ent)

        cum_reward_list.append(rollout["cumulative_reward"])
        cum_pref_list.append(rollout["cumulative_pref_reward"])
        cum_div_list.append(rollout["cumulative_div_bonus"])

    denom_catalog = (
        len(data.candidate_pool_indices) if use_candidate_pool else data.num_movies
    )
    catalog_coverage = float(len(all_recommended_items)) / float(max(1, denom_catalog))

    metrics = {
        "Model": model_name,
        "Precision@5": float(np.mean(p5_list)),
        "Precision@10": float(np.mean(p10_list)),
        "Recall@5": float(np.mean(r5_list)),
        "Recall@10": float(np.mean(r10_list)),
        "NDCG@5": float(np.mean(ndcg5_list)),
        "NDCG@10": float(np.mean(ndcg10_list)),
        "Intra-List Diversity": float(np.mean(ild_list)),
        "Genre Diversity": float(np.mean(genre_cov_list)),
        "Genre Entropy": float(np.mean(genre_ent_list)),
        "Catalog Coverage": float(catalog_coverage),
        "Avg Cumulative Reward": float(np.mean(cum_reward_list)),
        "Avg Preference Reward": float(np.mean(cum_pref_list)),
        "Avg Diversity Bonus": float(np.mean(cum_div_list)),
    }
    return metrics, user_recommendations
