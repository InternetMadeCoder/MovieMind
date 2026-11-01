"""
Traditional Recommendation Baselines for MovieMind:
1. Random Recommender
2. Popularity-Based Recommender (derived strictly from training interaction counts & ratings)
3. Content-Based Filtering Recommender (Cosine similarity on movie genre vectors & user genre profiles)
"""

from typing import Optional
import numpy as np

from src.data_preprocessing import PreprocessedData


class RandomRecommender:
    """Recommends uniformly at random from the eligible unseen candidate pool."""

    def __init__(self, data: PreprocessedData, seed: int = 42):
        self.num_movies = data.num_movies
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        user_idx: Optional[int] = None,
    ) -> int:
        eligible = np.flatnonzero(action_mask)
        if len(eligible) == 0:
            return int(self.rng.integers(0, self.num_movies))
        return int(self.rng.choice(eligible))


class PopularityRecommender:
    """
    Recommends the most popular / highest Bayesian-rated unseen movies
    computed strictly from the training split.
    """

    def __init__(self, data: PreprocessedData):
        self.num_movies = data.num_movies
        bayesian_norm = np.clip((data.item_bayesian_rating - 1.0) / 4.0, 0.0, 1.0)
        self.scores = (
            0.75 * data.item_popularity_score + 0.25 * bayesian_norm
        ).astype(np.float32)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        user_idx: Optional[int] = None,
    ) -> int:
        eligible = np.flatnonzero(action_mask)
        if len(eligible) == 0:
            return 0
        masked_scores = np.full(self.num_movies, -1e9, dtype=np.float32)
        masked_scores[eligible] = self.scores[eligible]
        return int(np.argmax(masked_scores))


class ContentBasedRecommender:
    """
    Content-based filtering using cosine similarity between the user's training
    genre preference profile and movie genre vectors.
    """

    def __init__(self, data: PreprocessedData):
        self.data = data
        self.num_movies = data.num_movies
        self.num_genres = data.num_genres
        self.movie_genre_norm = data.movie_genre_norm.astype(np.float32)
        self.bayesian_norm = np.clip(
            (data.item_bayesian_rating - 1.0) / 4.0, 0.0, 1.0
        ).astype(np.float32)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        user_idx: Optional[int] = None,
    ) -> int:
        eligible = np.flatnonzero(action_mask)
        if len(eligible) == 0:
            return 0

        if user_idx is not None:
            user_prefs = self.data.user_genre_prefs[int(user_idx)]
        else:
            user_prefs = obs[: self.num_genres]

        u_norm = user_prefs / (np.linalg.norm(user_prefs) + 1e-6)
        cosine_sim = self.movie_genre_norm @ u_norm
        # Pure genre cosine similarity with a minor quality tie-breaker
        scores = 0.88 * cosine_sim + 0.12 * self.bayesian_norm

        masked_scores = np.full(self.num_movies, -1e9, dtype=np.float32)
        masked_scores[eligible] = scores[eligible]
        return int(np.argmax(masked_scores))
