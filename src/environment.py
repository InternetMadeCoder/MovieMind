"""
Custom Gymnasium-Compatible Reinforcement Learning Environment for MovieMind.

Markov Decision Process (MDP) Formulation:
- State s_t (57-dim continuous vector):
    1. User genre preference profile p_{u,t} (18 dims in [0, 1])
    2. User historical & session interaction features h_{u,t} (20 dims):
       - [norm_mean_rating, norm_rating_std, norm_log_activity, session_avg_pref_reward] (4 dims)
       - Training-derived historical collaborative interaction embedding u_svd (16 dims)
    3. Session genre exposure vector e_t (18 dims): fraction of recommendations so far in each genre
    4. Normalized episode step progress t / T_max (1 dim)
- Action a_t:
    Discrete movie index selected from the eligible candidate pool (excluding movies
    already rated by the user in train_df and movies already recommended in the episode).
- Reward r_t:
    Total Reward = Preference Reward + lambda_div * Diversity Bonus
- Episode Termination:
    Terminates after `episode_length` (default K=10) recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.data_preprocessing import PreprocessedData


class MovieRecEnv(gym.Env):
    """
    Gymnasium-compatible adaptive movie recommendation environment.
    Supports both continuous state vectors (for DQN) and discretized state indices (for Tabular Q-Learning).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: PreprocessedData,
        episode_length: int = 10,
        diversity_lambda: float = 0.25,
        use_candidate_pool: bool = True,
        user_pool: Optional[List[int]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data = data
        self.episode_length = int(episode_length)
        self.diversity_lambda = float(diversity_lambda)
        self.use_candidate_pool = bool(use_candidate_pool)
        self.user_pool = (
            list(user_pool) if user_pool is not None else list(range(data.num_users))
        )
        self.rng = np.random.default_rng(seed)

        self.num_users = data.num_users
        self.num_movies = data.num_movies
        self.num_genres = data.num_genres
        self.svd_dim = data.user_svd_factors.shape[1]

        # Pre-normalize user SVD factors to [-1, 1] for state stability
        max_svd = float(np.max(np.abs(data.user_svd_factors))) + 1e-6
        self.norm_user_svd = (data.user_svd_factors / max_svd).astype(np.float32)

        # State vector dimension:
        # 18 (user genre prefs) + 4 (user stats & session reward) + 16 (historical SVD features)
        # + 18 (session genre exposure) + 1 (step progress) = 57
        self.state_dim = self.num_genres * 2 + 5 + self.svd_dim
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_movies)

        # Discretized state space size for Tabular Q-Learning:
        # 18 (top preferred genre) * 19 (most exposed session genre or 18='none') * 2 (saturation flag) = 684 states
        self.num_discrete_states = self.num_genres * (self.num_genres + 1) * 2

        # Episode state variables
        self.current_user: int = 0
        self.current_step: int = 0
        self.current_user_prefs: np.ndarray = np.zeros(self.num_genres, dtype=np.float32)
        self.session_genre_counts: np.ndarray = np.zeros(self.num_genres, dtype=np.float32)
        self.recommended_movies: List[int] = []
        self.session_pref_rewards: List[float] = []
        self.session_div_bonuses: List[float] = []
        self._eligible_mask: np.ndarray = np.ones(self.num_movies, dtype=bool)

    def _init_eligible_mask(self, user_idx: int) -> None:
        """Initialize candidate mask excluding training-seen movies for `user_idx`."""
        if self.use_candidate_pool:
            mask = np.zeros(self.num_movies, dtype=bool)
            mask[self.data.candidate_pool_indices] = True
        else:
            mask = np.ones(self.num_movies, dtype=bool)

        # Mask out movies already seen by the user in train_df
        seen = self.data.train_seen_movies.get(user_idx, set())
        if seen:
            seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
            mask[seen_arr] = False

        # Fallback if a very active user has seen almost the entire top pool
        if np.sum(mask) < self.episode_length * 2:
            mask = np.ones(self.num_movies, dtype=bool)
            if seen:
                seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
                mask[seen_arr] = False

        self._eligible_mask = mask

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask of shape (num_movies,) indicating currently eligible movies."""
        return self._eligible_mask.copy()

    def get_eligible_actions(self) -> np.ndarray:
        """Return 1D array of currently eligible movie indices."""
        return np.flatnonzero(self._eligible_mask)

    def _get_obs(self) -> np.ndarray:
        """Construct the 57-dimensional Markov state representation."""
        user_stats = self.data.user_train_stats[self.current_user]  # (3,)
        avg_session_pref = (
            float(np.mean(self.session_pref_rewards))
            if self.session_pref_rewards
            else float(user_stats[0])
        )
        interaction_stats = np.array(
            [user_stats[0], user_stats[1], user_stats[2], avg_session_pref],
            dtype=np.float32,
        )
        user_svd = self.norm_user_svd[self.current_user]  # (16,)

        if self.current_step > 0:
            genre_exposure = np.clip(
                self.session_genre_counts / float(self.current_step), 0.0, 1.0
            )
        else:
            genre_exposure = np.zeros(self.num_genres, dtype=np.float32)

        step_progress = np.array(
            [min(1.0, float(self.current_step) / max(1.0, float(self.episode_length)))],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                self.current_user_prefs,
                interaction_stats,
                user_svd,
                genre_exposure,
                step_progress,
            ],
            axis=0,
        ).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def discretize_state(self, obs: Optional[np.ndarray] = None) -> int:
        """
        Convert continuous Markov state into a discrete state index for Tabular Q-Learning.
        Captures:
          1. User's dominant preferred genre (0..17)
          2. Session's most heavily exposed genre so far (0..17, or 18 if step == 0)
          3. Saturation indicator: 1 if dominant preferred genre already exceeds 35% of session recommendations
        """
        if obs is None:
            obs = self._get_obs()

        user_prefs = obs[: self.num_genres]
        exposure_start = self.num_genres + 4 + self.svd_dim
        genre_exposure = obs[exposure_start : exposure_start + self.num_genres]

        top_user_genre = int(np.argmax(user_prefs))
        if np.max(genre_exposure) <= 1e-5:
            most_exposed_genre = self.num_genres  # 18 = no exposure yet
            is_saturated = 0
        else:
            most_exposed_genre = int(np.argmax(genre_exposure))
            is_saturated = 1 if genre_exposure[top_user_genre] >= 0.35 else 0

        state_idx = (
            top_user_genre * ((self.num_genres + 1) * 2)
            + most_exposed_genre * 2
            + is_saturated
        )
        return int(state_idx)

    def compute_reward_components(self, movie_idx: int) -> Tuple[float, float, float]:
        """
        Compute (preference_reward, diversity_bonus, total_reward) for recommending `movie_idx`
        in the current state without mutating state.
        """
        # 1. Simulated Preference Reward derived strictly from training profiles
        base_train_pref = float(self.data.train_preference_matrix[self.current_user, movie_idx])
        movie_g_norm = self.data.movie_genre_norm[movie_idx]
        user_p_norm = self.current_user_prefs / (np.linalg.norm(self.current_user_prefs) + 1e-6)
        dynamic_genre_match = float(np.clip(np.dot(user_p_norm, movie_g_norm), 0.0, 1.0))

        pref_reward = 0.75 * base_train_pref + 0.25 * dynamic_genre_match

        # 2. Genre Diversity Bonus
        movie_genres = self.data.movie_genre_matrix[movie_idx]
        active_genres = np.flatnonzero(movie_genres > 0)

        if self.current_step == 0 or len(self.recommended_movies) == 0:
            session_novelty = 1.0
            intra_list_dissim = 1.0
        else:
            genre_exposure = self.session_genre_counts / float(self.current_step)
            avg_exposure_of_movie_genres = float(np.mean(genre_exposure[active_genres]))
            session_novelty = float(np.clip(1.0 - 1.8 * avg_exposure_of_movie_genres, 0.0, 1.0))

            prev_norms = self.data.movie_genre_norm[self.recommended_movies]
            sims = prev_norms @ movie_g_norm
            intra_list_dissim = float(np.clip(1.0 - np.mean(sims), 0.0, 1.0))

        raw_diversity = 0.55 * session_novelty + 0.45 * intra_list_dissim
        # Gate diversity bonus mildly by user genre relevance so irrelevant items do not dominate
        relevance_gate = 0.35 + 0.65 * dynamic_genre_match
        diversity_bonus = float(np.clip(raw_diversity * relevance_gate, 0.0, 1.0))

        total_reward = float(pref_reward + self.diversity_lambda * diversity_bonus)
        return pref_reward, diversity_bonus, total_reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for a new user episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "user_idx" in options:
            self.current_user = int(options["user_idx"])
        else:
            self.current_user = int(self.rng.choice(self.user_pool))

        self.current_step = 0
        self.current_user_prefs = self.data.user_genre_prefs[self.current_user].copy()
        self.session_genre_counts = np.zeros(self.num_genres, dtype=np.float32)
        self.recommended_movies = []
        self.session_pref_rewards = []
        self.session_div_bonuses = []
        self._init_eligible_mask(self.current_user)

        obs = self._get_obs()
        info = {
            "user_idx": self.current_user,
            "discrete_state": self.discretize_state(obs),
            "eligible_count": int(np.sum(self._eligible_mask)),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute movie recommendation action `action` and return (next_obs, reward, terminated, truncated, info)."""
        movie_idx = int(action)

        # Enforce candidate masking if an invalid/already-seen movie was passed
        already_used_penalty = 0.0
        if not self._eligible_mask[movie_idx]:
            eligible = self.get_eligible_actions()
            if len(eligible) > 0:
                movie_idx = int(self.rng.choice(eligible))
            already_used_penalty = -0.15

        pref_reward, div_bonus, total_reward = self.compute_reward_components(movie_idx)
        total_reward += already_used_penalty

        # Update session state and adaptive user preferences based on simulated feedback
        movie_genres = self.data.movie_genre_matrix[movie_idx]
        self.session_genre_counts += movie_genres
        self.recommended_movies.append(movie_idx)
        self.session_pref_rewards.append(pref_reward)
        self.session_div_bonuses.append(div_bonus)
        self._eligible_mask[movie_idx] = False
        self.current_step += 1

        # Adaptive preference update: positive satisfaction gently reinforces movie's genres
        feedback_delta = (pref_reward - 0.55) * 0.08
        self.current_user_prefs = np.clip(
            self.current_user_prefs + feedback_delta * movie_genres,
            0.05,
            1.0,
        )
        if np.max(self.current_user_prefs) > 0:
            self.current_user_prefs = self.current_user_prefs / np.max(self.current_user_prefs)

        terminated = bool(
            self.current_step >= self.episode_length
            or np.sum(self._eligible_mask) == 0
        )
        truncated = False

        next_obs = self._get_obs()
        info = {
            "user_idx": self.current_user,
            "movie_idx": movie_idx,
            "preference_reward": pref_reward,
            "diversity_bonus": div_bonus,
            "total_reward": total_reward,
            "discrete_state": self.discretize_state(next_obs),
            "step": self.current_step,
        }
        return next_obs, float(total_reward), terminated, truncated, info
