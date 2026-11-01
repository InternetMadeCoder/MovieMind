"""
Data Loading, Cleaning, and Preprocessing for MovieMind (MovieLens 100K).

Strict anti-leakage guarantee:
- Performs an 80:20 per-user split into train_df and test_df.
- All user preference profiles, item quality statistics, collaborative SVD factors,
  and candidate pools are computed EXCLUSIVELY from train_df.
- test_df is held out strictly for final offline recommendation evaluation.
"""

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

ALL_GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

# The 18 meaningful named genres in MovieLens 100K
GENRE_NAMES = ALL_GENRE_COLUMNS[1:]


@dataclass
class PreprocessedData:
    """Container holding all cleaned datasets, encodings, and training-only features."""

    ratings_df: pd.DataFrame
    movies_df: pd.DataFrame
    users_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    genre_names: List[str]
    num_users: int
    num_movies: int
    num_genres: int
    # Movie matrices (num_movies, num_genres)
    movie_genre_matrix: np.ndarray
    movie_genre_norm: np.ndarray
    movie_primary_genre: np.ndarray
    # Training-derived item statistics (num_movies,)
    item_train_count: np.ndarray
    item_train_mean_rating: np.ndarray
    item_bayesian_rating: np.ndarray
    item_popularity_score: np.ndarray
    # Training-derived user profiles (num_users, num_genres) and stats (num_users, 3)
    user_genre_prefs: np.ndarray
    user_train_stats: np.ndarray
    # Training-derived user-movie preference score matrix in [0, 1] (num_users, num_movies)
    train_preference_matrix: np.ndarray
    # Training-derived latent embeddings for state/action features
    user_svd_factors: np.ndarray
    item_svd_factors: np.ndarray
    # Lookup sets per user
    train_seen_movies: Dict[int, Set[int]]
    test_relevant_movies: Dict[int, Set[int]]
    test_all_movies: Dict[int, Dict[int, float]]
    # Top candidate pool indices (computed strictly from training popularity/quality)
    candidate_pool_indices: np.ndarray


def download_movielens_100k(data_dir: str = "data") -> str:
    """Download and extract the official MovieLens 100K dataset from GroupLens if needed."""
    os.makedirs(data_dir, exist_ok=True)
    ml_dir = os.path.join(data_dir, "ml-100k")
    u_data_path = os.path.join(ml_dir, "u.data")

    if os.path.exists(u_data_path):
        return ml_dir

    zip_path = os.path.join(data_dir, "ml-100k.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens 100K from {MOVIELENS_100K_URL}...")
        urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)

    print("Extracting MovieLens 100K dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return ml_dir


def load_raw_movielens(ml_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ratings (u.data), movie metadata/genres (u.item), and user demographics (u.user)."""
    # 1. Load ratings
    ratings_path = os.path.join(ml_dir, "u.data")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    # 2. Load movies and genres
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + ALL_GENRE_COLUMNS
    items_path = os.path.join(ml_dir, "u.item")
    movies_df = pd.read_csv(
        items_path,
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python",
    )

    # 3. Load users
    users_path = os.path.join(ml_dir, "u.user")
    users_df = pd.read_csv(
        users_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
        engine="python",
    )

    return ratings_df, movies_df, users_df


def clean_and_encode_data(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean metadata, extract release years, and encode contiguous 0-based user_idx and movie_idx."""
    movies_df = movies_df.copy()
    ratings_df = ratings_df.copy()
    users_df = users_df.copy()

    # Sort by original IDs for deterministic 0-based encoding
    movies_df = movies_df.sort_values("item_id").reset_index(drop=True)
    users_df = users_df.sort_values("user_id").reset_index(drop=True)

    movies_df["movie_idx"] = np.arange(len(movies_df), dtype=np.int64)
    users_df["user_idx"] = np.arange(len(users_df), dtype=np.int64)

    item_to_idx = dict(zip(movies_df["item_id"], movies_df["movie_idx"]))
    user_to_idx = dict(zip(users_df["user_id"], users_df["user_idx"]))

    ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_idx).astype(np.int64)
    ratings_df["movie_idx"] = ratings_df["item_id"].map(item_to_idx).astype(np.int64)

    # Clean titles and extract release year
    movies_df["title"] = movies_df["title"].fillna("Unknown Title").str.strip()
    extracted_year = movies_df["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    parsed_date_year = pd.to_datetime(movies_df["release_date"], errors="coerce").dt.year
    movies_df["year"] = (
        extracted_year.fillna(parsed_date_year).fillna(1995).astype(int)
    )

    # If a movie has 0 across all 18 named genres (e.g. 'unknown' only), assign Drama as fallback
    genre_mat = movies_df[GENRE_NAMES].values.astype(np.float32)
    zero_genre_mask = genre_mat.sum(axis=1) == 0
    if np.any(zero_genre_mask):
        drama_col = GENRE_NAMES.index("Drama")
        movies_df.loc[zero_genre_mask, "Drama"] = 1
        genre_mat = movies_df[GENRE_NAMES].values.astype(np.float32)

    # Build human-readable genre list and string per movie
    genre_lists = []
    genre_strs = []
    primary_genres = []
    for row in genre_mat:
        active = [GENRE_NAMES[i] for i, val in enumerate(row) if val > 0]
        genre_lists.append(active)
        genre_strs.append(" | ".join(active))
        primary_genres.append(int(np.argmax(row)))

    movies_df["genres_list"] = genre_lists
    movies_df["genres_str"] = genre_strs
    movies_df["primary_genre_idx"] = primary_genres
    movies_df["primary_genre"] = [GENRE_NAMES[idx] for idx in primary_genres]

    return ratings_df, movies_df, users_df


def split_train_test_per_user(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform an 80:20 per-user train/test split.
    Shuffles each user's ratings reproducibly with `seed` so every user has 80% in train and 20% in test.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, group in ratings_df.groupby("user_idx", sort=True):
        n_interactions = len(group)
        shuffled_indices = rng.permutation(group.index.values)
        n_train = max(1, int(round(n_interactions * train_ratio)))
        if n_train >= n_interactions:
            n_train = n_interactions - 1

        train_idx = shuffled_indices[:n_train]
        test_idx = shuffled_indices[n_train:]
        train_parts.append(ratings_df.loc[train_idx])
        test_parts.append(ratings_df.loc[test_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_preprocessed_dataset(
    data_dir: str = "data",
    train_ratio: float = 0.80,
    candidate_pool_size: int = 300,
    svd_rank: int = 16,
    seed: int = 42,
) -> PreprocessedData:
    """
    Full pipeline: download, load, clean, encode, split 80:20 per user, and construct
    user preference profiles & environment simulation parameters strictly from train_df.
    """
    ml_dir = download_movielens_100k(data_dir=data_dir)
    raw_ratings, raw_movies, raw_users = load_raw_movielens(ml_dir)
    ratings_df, movies_df, users_df = clean_and_encode_data(raw_ratings, raw_movies, raw_users)
    train_df, test_df = split_train_test_per_user(ratings_df, train_ratio=train_ratio, seed=seed)

    num_users = len(users_df)
    num_movies = len(movies_df)
    num_genres = len(GENRE_NAMES)

    # 1. Movie genre matrices
    movie_genre_matrix = movies_df[GENRE_NAMES].values.astype(np.float32)
    genre_row_norms = np.linalg.norm(movie_genre_matrix, axis=1, keepdims=True)
    genre_row_norms = np.where(genre_row_norms == 0, 1.0, genre_row_norms)
    movie_genre_norm = (movie_genre_matrix / genre_row_norms).astype(np.float32)
    movie_primary_genre = movies_df["primary_genre_idx"].values.astype(np.int64)

    # 2. Item statistics computed STRICTLY from train_df
    global_train_mean = float(train_df["rating"].mean())
    item_counts = np.zeros(num_movies, dtype=np.float32)
    item_rating_sums = np.zeros(num_movies, dtype=np.float32)

    grouped_items = train_df.groupby("movie_idx")["rating"].agg(["count", "sum"])
    item_counts[grouped_items.index.values] = grouped_items["count"].values.astype(np.float32)
    item_rating_sums[grouped_items.index.values] = grouped_items["sum"].values.astype(np.float32)

    item_train_mean = np.where(
        item_counts > 0,
        item_rating_sums / np.maximum(item_counts, 1.0),
        global_train_mean,
    ).astype(np.float32)

    # Bayesian damped rating (prior weight m = 10 ratings)
    prior_m = 10.0
    item_bayesian_rating = (
        (item_rating_sums + prior_m * global_train_mean) / (item_counts + prior_m)
    ).astype(np.float32)

    # Normalized popularity score in [0, 1] using log(1 + count)
    log_counts = np.log1p(item_counts)
    max_log_count = float(np.max(log_counts)) if np.max(log_counts) > 0 else 1.0
    item_popularity_score = (log_counts / max_log_count).astype(np.float32)

    # Attach training stats to movies_df for transparent inspection & UI
    movies_df["train_rating_count"] = item_counts.astype(int)
    movies_df["train_mean_rating"] = np.round(item_train_mean, 2)
    movies_df["bayesian_rating"] = np.round(item_bayesian_rating, 2)
    movies_df["popularity_score"] = np.round(item_popularity_score, 4)

    # 3. User preference profiles & historical stats computed STRICTLY from train_df
    user_genre_prefs = np.zeros((num_users, num_genres), dtype=np.float32)
    user_train_stats = np.zeros((num_users, 3), dtype=np.float32)

    train_seen_movies: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
    test_relevant_movies: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
    test_all_movies: Dict[int, Dict[int, float]] = {u: {} for u in range(num_users)}

    # Populate lookup dictionaries
    for u_idx, m_idx, rating in zip(
        train_df["user_idx"].values,
        train_df["movie_idx"].values,
        train_df["rating"].values,
    ):
        train_seen_movies[int(u_idx)].add(int(m_idx))

    for u_idx, m_idx, rating in zip(
        test_df["user_idx"].values,
        test_df["movie_idx"].values,
        test_df["rating"].values,
    ):
        u_i, m_i, r_f = int(u_idx), int(m_idx), float(rating)
        test_all_movies[u_i][m_i] = r_f
        if r_f >= 4.0:
            test_relevant_movies[u_i].add(m_i)

    # Verify strict disjointness between train and test per user
    for u in range(num_users):
        overlap = train_seen_movies[u].intersection(test_all_movies[u].keys())
        assert len(overlap) == 0, f"Data leakage detected for user {u}: {overlap}"

    # Compute user genre preference profiles from training interactions
    max_user_log_count = np.log1p(float(train_df.groupby("user_idx").size().max()))
    for u_idx, group in train_df.groupby("user_idx"):
        u = int(u_idx)
        m_indices = group["movie_idx"].values
        ratings = group["rating"].values.astype(np.float32)

        # Positive preference weight: higher ratings receive stronger positive weight
        # Map ratings {1, 2, 3, 4, 5} -> {0.1, 0.3, 0.6, 1.0, 1.4}
        weights = np.clip((ratings - 1.0) / 4.0 + 0.15, 0.1, 1.5)
        genres_watched = movie_genre_matrix[m_indices]  # (n_watched, num_genres)

        weighted_genre_sum = (genres_watched * weights[:, None]).sum(axis=0)
        genre_exposure_count = genres_watched.sum(axis=0) + 1.5  # smoothed denominator
        # Combine total affinity volume and mean rating per genre
        volume_pref = weighted_genre_sum / (np.max(weighted_genre_sum) + 1e-6)
        rate_pref = weighted_genre_sum / genre_exposure_count
        rate_pref = rate_pref / (np.max(rate_pref) + 1e-6)

        combined_pref = 0.65 * volume_pref + 0.35 * rate_pref
        if np.max(combined_pref) > 0:
            combined_pref = combined_pref / np.max(combined_pref)
        else:
            combined_pref = np.ones(num_genres, dtype=np.float32) / num_genres

        user_genre_prefs[u] = combined_pref.astype(np.float32)

        # User historical summary features: [norm_mean_rating, norm_rating_std, norm_log_activity]
        u_mean = (float(np.mean(ratings)) - 1.0) / 4.0
        u_std = float(np.std(ratings)) / 2.0
        u_act = float(np.log1p(len(ratings))) / max(max_user_log_count, 1.0)
        user_train_stats[u] = np.array([u_mean, u_std, u_act], dtype=np.float32)

    # 4. Training-only SVD Collaborative Filtering Factors (for realistic simulated user reward)
    # Construct centered/weighted implicit+explicit training matrix R_train
    interaction_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    for u_idx, m_idx, rating in zip(
        train_df["user_idx"].values,
        train_df["movie_idx"].values,
        train_df["rating"].values,
    ):
        # Implicit engagement + explicit rating satisfaction signal derived ONLY from train_df
        interaction_matrix[int(u_idx), int(m_idx)] = 0.35 + 0.65 * ((float(rating) - 1.0) / 4.0)

    # Low-rank SVD on training interaction matrix
    u_mat, s_vals, vt_mat = np.linalg.svd(interaction_matrix, full_matrices=False)
    k = min(svd_rank, len(s_vals))
    sqrt_s = np.sqrt(s_vals[:k]).astype(np.float32)
    user_svd_factors = (u_mat[:, :k] * sqrt_s[None, :]).astype(np.float32)
    item_svd_factors = (vt_mat[:k, :].T * sqrt_s[None, :]).astype(np.float32)

    svd_recon = user_svd_factors @ item_svd_factors.T  # (num_users, num_movies)
    # Normalize svd_recon per user into [0, 1]
    svd_min = np.percentile(svd_recon, 5, axis=1, keepdims=True)
    svd_max = np.percentile(svd_recon, 99, axis=1, keepdims=True)
    svd_norm = np.clip((svd_recon - svd_min) / (svd_max - svd_min + 1e-6), 0.0, 1.0)

    # Content cosine affinity between user_genre_prefs and movie_genre_norm
    user_pref_norms = np.linalg.norm(user_genre_prefs, axis=1, keepdims=True)
    user_pref_norms = np.where(user_pref_norms == 0, 1.0, user_pref_norms)
    user_genre_unit = user_genre_prefs / user_pref_norms
    content_sim = np.clip(user_genre_unit @ movie_genre_norm.T, 0.0, 1.0)

    # Item Bayesian quality normalized to [0, 1]
    bayesian_norm = np.clip((item_bayesian_rating - 1.0) / 4.0, 0.0, 1.0)

    # Combine training-derived signals into a calibrated preference score in [0, 1]
    # 45% training collaborative SVD + 35% training genre affinity + 20% training popularity/quality
    train_preference_matrix = (
        0.45 * svd_norm
        + 0.35 * content_sim
        + 0.12 * item_popularity_score[None, :]
        + 0.08 * bayesian_norm[None, :]
    ).astype(np.float32)
    train_preference_matrix = np.clip(train_preference_matrix, 0.0, 1.0)

    # If a movie was actually rated in train_df, record its true normalized rating in train_preference_matrix
    for u_idx, m_idx, rating in zip(
        train_df["user_idx"].values,
        train_df["movie_idx"].values,
        train_df["rating"].values,
    ):
        train_preference_matrix[int(u_idx), int(m_idx)] = float((rating - 1.0) / 4.0)

    # 5. Candidate pool selection (computed strictly from training popularity & quality)
    candidate_score = 0.70 * item_popularity_score + 0.30 * bayesian_norm
    pool_size = min(candidate_pool_size, num_movies)
    candidate_pool_indices = np.argsort(-candidate_score)[:pool_size].astype(np.int64)

    return PreprocessedData(
        ratings_df=ratings_df,
        movies_df=movies_df,
        users_df=users_df,
        train_df=train_df,
        test_df=test_df,
        genre_names=list(GENRE_NAMES),
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        movie_genre_matrix=movie_genre_matrix,
        movie_genre_norm=movie_genre_norm,
        movie_primary_genre=movie_primary_genre,
        item_train_count=item_counts,
        item_train_mean_rating=item_train_mean,
        item_bayesian_rating=item_bayesian_rating,
        item_popularity_score=item_popularity_score,
        user_genre_prefs=user_genre_prefs,
        user_train_stats=user_train_stats,
        train_preference_matrix=train_preference_matrix,
        user_svd_factors=user_svd_factors,
        item_svd_factors=item_svd_factors,
        train_seen_movies=train_seen_movies,
        test_relevant_movies=test_relevant_movies,
        test_all_movies=test_all_movies,
        candidate_pool_indices=candidate_pool_indices,
    )
