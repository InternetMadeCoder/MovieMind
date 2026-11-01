"""
Reinforcement Learning Agents for MovieMind:
1. Tabular Q-Learning Agent (Discretized state representation, Q-table, Bellman updates, epsilon-greedy)
2. Deep Q-Network (DQN) Agent (PyTorch neural network, Experience Replay Buffer, Target Network, Huber loss)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_preprocessing import PreprocessedData
from src.environment import MovieRecEnv


class TabularQLearningAgent:
    """
    Tabular Q-Learning agent operating on the discretized Markov state space.
    Maintains an explicit state-action Q-table Q(s_disc, a) and state-genre Q-table
    updated via Tabular Bellman equations with epsilon-greedy exploration.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        data: PreprocessedData,
        lr: float = 0.20,
        gamma: float = 0.85,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.num_genres = data.num_genres
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

        self.movie_genre_matrix = data.movie_genre_matrix.astype(np.float32)
        self.movie_genre_norm = data.movie_genre_norm.astype(np.float32)
        genre_row_sums = np.maximum(1.0, self.movie_genre_matrix.sum(axis=1, keepdims=True))
        self.movie_genre_l1 = (self.movie_genre_matrix / genre_row_sums).astype(np.float32)
        self.movie_primary_genre = data.movie_primary_genre.astype(np.int64)

        # Training-derived item quality prior (strictly from train_df)
        bayesian_norm = np.clip((data.item_bayesian_rating - 1.0) / 4.0, 0.0, 1.0)
        self.item_prior = (
            0.70 * data.item_popularity_score + 0.30 * bayesian_norm
        ).astype(np.float32)

        # Calibrate Q-table initialization to the Bellman fixed-point structure:
        # Q*(s, a) = E[r(s, a)] + (gamma / (1 - gamma)) * r_max
        r_max_est = 0.82
        continuation_val = (self.gamma / (1.0 - self.gamma)) * r_max_est
        base_step_r = 0.62 * self.item_prior + 0.18
        self.q_table = np.tile(
            (base_step_r + continuation_val)[None, :],
            (self.num_states, 1),
        ).astype(np.float32)
        genre_base_r = (self.movie_genre_l1.T @ base_step_r) / np.maximum(
            1.0, self.movie_genre_l1.sum(axis=0)
        )
        self.genre_q_table = np.tile(
            (genre_base_r + continuation_val)[None, :],
            (self.num_states, 1),
        ).astype(np.float32)
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        # Encode structural prior into genre_q_table based on discretized state semantics
        for s_idx in range(self.num_states):
            top_u_genre = s_idx // ((self.num_genres + 1) * 2)
            rem = s_idx % ((self.num_genres + 1) * 2)
            exposed_genre = rem // 2
            is_saturated = rem % 2
            self.genre_q_table[s_idx, top_u_genre] += 0.15 * (
                1.0 - 0.55 * is_saturated
            )
            if exposed_genre < self.num_genres:
                self.genre_q_table[s_idx, exposed_genre] -= (
                    0.12 * (0.5 + 0.5 * is_saturated)
                )

    def _tabular_q(self, discrete_state: int) -> np.ndarray:
        """Pure tabular Q-estimate from state-action and state-genre Q-tables."""
        direct_q = self.q_table[discrete_state]
        genre_q = self.movie_genre_l1 @ self.genre_q_table[discrete_state]
        return 0.72 * direct_q + 0.28 * genre_q

    def get_state_q_values(
        self, discrete_state: int, obs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Q(s_disc, :) across all candidate movies."""
        combined_q = self._tabular_q(discrete_state)

        if obs is not None:
            # Discretized top-3 user genre preference alignment & session saturation penalty
            user_prefs = obs[: self.num_genres]
            exposure_start = self.num_genres + 4 + (len(obs) - 2 * self.num_genres - 5)
            genre_exposure = obs[exposure_start : exposure_start + self.num_genres]

            top3 = np.argsort(-user_prefs)[:3]
            coarse_pref = np.zeros(self.num_genres, dtype=np.float32)
            coarse_pref[top3] = user_prefs[top3]

            pref_align = self.movie_genre_norm @ coarse_pref
            sat_penalty = self.movie_genre_norm @ genre_exposure
            combined_q = combined_q + (0.26 * pref_align - 0.16 * sat_penalty)

        return combined_q.astype(np.float32)

    def select_action(
        self,
        discrete_state: int,
        action_mask: np.ndarray,
        obs: Optional[np.ndarray] = None,
        greedy: bool = False,
    ) -> int:
        """Select an eligible movie action using epsilon-greedy exploration."""
        eligible_actions = np.flatnonzero(action_mask)
        if len(eligible_actions) == 0:
            return int(self.rng.integers(0, self.num_actions))

        if (not greedy) and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(eligible_actions))

        q_vals = self.get_state_q_values(discrete_state, obs=obs)
        masked_q = np.full(self.num_actions, -1e9, dtype=np.float32)
        masked_q[eligible_actions] = q_vals[eligible_actions]
        return int(np.argmax(masked_q))

    def update(
        self,
        discrete_state: int,
        action: int,
        reward: float,
        next_discrete_state: int,
        done: bool,
        next_action_mask: np.ndarray,
        next_obs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Perform Tabular Bellman Q-Learning update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        """
        if done:
            max_next_q = 0.0
        else:
            next_q_vals = self._tabular_q(next_discrete_state)
            eligible_next = np.flatnonzero(next_action_mask)
            if len(eligible_next) > 0:
                max_next_q = float(np.max(next_q_vals[eligible_next]))
            else:
                max_next_q = float(np.max(next_q_vals))

        # Finite-horizon return scaling when near end of episode
        td_target = float(reward) + self.gamma * (
            max_next_q if not done else float(np.mean(self.q_table[discrete_state]))
        )

        # Update direct state-action Q-table entry
        current_q = float(self.q_table[discrete_state, action])
        td_error = td_target - current_q
        self.q_table[discrete_state, action] += self.lr * td_error
        self.visit_counts[discrete_state, action] += 1.0

        # Update state-genre Q-table across active genres of the chosen movie
        active_genres = np.flatnonzero(self.movie_genre_matrix[action] > 0)
        for g_idx in active_genres:
            g_err = td_target - float(self.genre_q_table[discrete_state, g_idx])
            self.genre_q_table[discrete_state, g_idx] += (self.lr * 0.25) * g_err

        return abs(td_error)

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    """Fixed-capacity Experience Replay Buffer for Deep Q-Network training."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        num_actions: int,
        seed: int = 42,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, self.num_actions), dtype=bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        """Store a transition tuple in the replay buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.next_masks[self.ptr] = next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch of transitions."""
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_masks[indices],
        )

    def __len__(self) -> int:
        return self.size


class DQNNetwork(nn.Module):
    """
    Dueling Feature-Interactive Deep Q-Network for Movie Recommendation.

    Architecture:
    - Takes the 57-dimensional continuous Markov state s_t.
    - Encodes state through a 2-layer MLP with LayerNorm and ReLU activations.
    - Combines:
      1. State Value Stream V(s_t) -> (batch, 1)
      2. Neural Feature Projection Stream w(s_t) @ M_features^T -> (batch, num_movies)
      3. State-Conditioned Context Gating over (Collaborative SVD Match, Genre Affinity,
         Session Genre Novelty, and Training Item Quality/Popularity) -> (batch, num_movies)
      4. Residual Movie Advantage Stream A_res(s_t) -> (batch, num_movies)
    """

    def __init__(
        self,
        state_dim: int,
        num_movies: int,
        data: PreprocessedData,
        hidden_dim: int = 128,
        diversity_lambda: float = 0.25,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_movies = int(num_movies)
        self.num_genres = int(data.num_genres)
        self.svd_dim = int(data.user_svd_factors.shape[1])
        self.diversity_lambda = float(diversity_lambda)

        # Register training-derived movie feature buffers (no data leakage from test set)
        movie_genre_norm = torch.tensor(data.movie_genre_norm, dtype=torch.float32)
        max_user_svd = float(np.max(np.abs(data.user_svd_factors))) + 1e-6
        scaled_item_svd = torch.tensor(
            data.item_svd_factors * max_user_svd, dtype=torch.float32
        )
        bayesian_norm = torch.tensor(
            np.clip((data.item_bayesian_rating - 1.0) / 4.0, 0.0, 1.0),
            dtype=torch.float32,
        )
        pop_score = torch.tensor(data.item_popularity_score, dtype=torch.float32)

        self.register_buffer("movie_genre_norm", movie_genre_norm)  # (M, 18)
        self.register_buffer("scaled_item_svd", scaled_item_svd)  # (M, 16)
        self.register_buffer("bayesian_norm", bayesian_norm)  # (M,)
        self.register_buffer("pop_score", pop_score)  # (M,)

        # Concatenated movie feature matrix: 18 (genres) + 16 (SVD) + 2 (pop, quality) = 36 dims
        movie_features = torch.cat(
            [
                movie_genre_norm,
                scaled_item_svd,
                pop_score.unsqueeze(1),
                bayesian_norm.unsqueeze(1),
            ],
            dim=1,
        )
        self.movie_feat_dim = movie_features.shape[1]
        self.register_buffer("movie_features", movie_features)

        # Shared state representation network
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # 1. Dueling state-value head V(s)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # 2. Learned state-to-movie-feature projection head
        self.feature_proj_head = nn.Linear(hidden_dim // 2, self.movie_feat_dim)

        # 3. Context-dependent gating head over 5 interpretable interaction signals:
        # [svd_collab_match, genre_pref_match, session_genre_novelty, popularity, bayesian_quality]
        self.gate_head = nn.Linear(hidden_dim // 2, 5)

        # 4. Per-movie residual advantage head
        self.residual_head = nn.Linear(hidden_dim // 2, self.num_movies)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.005)
        with torch.no_grad():
            # Initialize novelty gate proportional to diversity_lambda
            nov_init = max(-2.5, float(np.log(np.exp(max(0.02, self.diversity_lambda * 1.4)) - 1.0)))
            self.gate_head.bias.copy_(
                torch.tensor([0.55, 0.50, nov_init, 0.18, 0.12], dtype=torch.float32)
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a) for all movies given batch of states `state` of shape (B, state_dim).
        Returns tensor of shape (B, num_movies).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.state_encoder(state)  # (B, hidden_dim // 2)

        # Extract structured state components
        g_dim = self.num_genres
        s_dim = self.svd_dim
        user_prefs = state[:, :g_dim]  # (B, 18)
        user_svd = state[:, g_dim + 4 : g_dim + 4 + s_dim]  # (B, 16)
        genre_exposure = state[:, g_dim + 4 + s_dim : 2 * g_dim + 4 + s_dim]  # (B, 18)

        # Compute explicit interaction signals across all movies in parallel
        user_pref_unit = user_prefs / (torch.norm(user_prefs, dim=1, keepdim=True) + 1e-6)
        genre_match = torch.clamp(user_pref_unit @ self.movie_genre_norm.T, 0.0, 1.0)  # (B, M)

        raw_svd = user_svd @ self.scaled_item_svd.T  # (B, M)
        svd_match = torch.sigmoid(2.5 * (raw_svd - 0.35))  # (B, M)

        # Genre novelty vs saturation penalty in current episode
        exposure_overlap = genre_exposure @ self.movie_genre_norm.T  # (B, M)
        novelty_weights = torch.clamp(1.0 - 2.2 * genre_exposure, 0.0, 1.0)  # (B, 18)
        genre_novelty = (novelty_weights @ self.movie_genre_norm.T) * (
            0.35 + 0.65 * genre_match
        ) - 0.85 * exposure_overlap  # (B, M)

        pop_mat = self.pop_score.unsqueeze(0).expand(state.size(0), -1)  # (B, M)
        qual_mat = self.bayesian_norm.unsqueeze(0).expand(state.size(0), -1)  # (B, M)

        # State-conditioned softplus gates
        gates = torch.nn.functional.softplus(self.gate_head(h))  # (B, 5)
        gated_interaction = (
            gates[:, 0:1] * svd_match
            + gates[:, 1:2] * genre_match
            + gates[:, 2:3] * genre_novelty
            + gates[:, 3:4] * pop_mat
            + gates[:, 4:5] * qual_mat
        )  # (B, M)

        # Neural feature projection + residual advantage
        feat_weights = self.feature_proj_head(h)  # (B, movie_feat_dim)
        feat_scores = 0.08 * (feat_weights @ self.movie_features.T)  # (B, M)
        res_scores = 0.05 * self.residual_head(h)  # (B, M)

        advantages = gated_interaction + feat_scores + res_scores
        state_value = self.value_head(h)  # (B, 1)

        # Dueling Q combination
        q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent with Experience Replay, Target Network, Huber Loss,
    and Action-Masked Epsilon-Greedy Exploration.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        data: PreprocessedData,
        hidden_dim: int = 128,
        diversity_lambda: float = 0.25,
        lr: float = 1e-3,
        gamma: float = 0.85,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 15000,
        batch_size: int = 64,
        target_update_freq: int = 80,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.diversity_lambda = float(diversity_lambda)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DQNNetwork(
            state_dim=self.state_dim,
            num_movies=self.num_actions,
            data=data,
            hidden_dim=hidden_dim,
            diversity_lambda=self.diversity_lambda,
        ).to(self.device)
        self.target_net = DQNNetwork(
            state_dim=self.state_dim,
            num_movies=self.num_actions,
            data=data,
            hidden_dim=hidden_dim,
            diversity_lambda=self.diversity_lambda,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            seed=seed,
        )
        self.train_steps = 0

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q(s, :) numpy array for a single state vector."""
        self.policy_net.eval()
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(s_tensor).squeeze(0).cpu().numpy()
        return q_vals

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        """Select an eligible movie action using epsilon-greedy exploration."""
        eligible_actions = np.flatnonzero(action_mask)
        if len(eligible_actions) == 0:
            return int(self.rng.integers(0, self.num_actions))

        if (not greedy) and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(eligible_actions))

        q_vals = self.get_q_values(state)
        masked_q = np.full(self.num_actions, -1e9, dtype=np.float32)
        masked_q[eligible_actions] = q_vals[eligible_actions]
        return int(np.argmax(masked_q))

    def train_step(self) -> Optional[float]:
        """Sample a mini-batch from replay buffer and perform a Double-DQN Bellman gradient update."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)

        # Current Q(s, a)
        q_pred_all = self.policy_net(states_t)
        q_pred = q_pred_all.gather(1, actions_t).squeeze(1)

        # Double-DQN target: select best eligible next action using policy_net, evaluate with target_net
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)
            next_q_policy = next_q_policy.masked_fill(~next_masks_t, -1e9)
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states_t)
            max_next_q = next_q_target.gather(1, best_next_actions).squeeze(1)
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * max_next_q

        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_q_learning_agent(
    env: MovieRecEnv,
    agent: TabularQLearningAgent,
    num_episodes: int = 800,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train the Tabular Q-Learning agent across `num_episodes` episodes and log metrics."""
    rng = np.random.default_rng(seed)
    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "preference_rewards": [],
        "diversity_bonuses": [],
        "epsilons": [],
        "td_errors": [],
    }

    for ep in range(num_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        s_disc = info["discrete_state"]
        done = False

        ep_total_reward = 0.0
        ep_pref_reward = 0.0
        ep_div_bonus = 0.0
        ep_td_errs: List[float] = []

        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(s_disc, mask, obs=obs, greedy=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            next_s_disc = step_info["discrete_state"]
            next_mask = env.get_action_mask()

            td_err = agent.update(
                discrete_state=s_disc,
                action=action,
                reward=reward,
                next_discrete_state=next_s_disc,
                done=done,
                next_action_mask=next_mask,
                next_obs=next_obs,
            )

            ep_total_reward += reward
            ep_pref_reward += step_info["preference_reward"]
            ep_div_bonus += step_info["diversity_bonus"]
            ep_td_errs.append(td_err)

            obs = next_obs
            s_disc = next_s_disc

        history["episode_rewards"].append(ep_total_reward)
        history["preference_rewards"].append(ep_pref_reward)
        history["diversity_bonuses"].append(ep_div_bonus)
        history["epsilons"].append(agent.epsilon)
        history["td_errors"].append(float(np.mean(ep_td_errs)) if ep_td_errs else 0.0)

        agent.decay_epsilon()

    return history


def train_dqn_agent(
    env: MovieRecEnv,
    agent: DQNAgent,
    num_episodes: int = 800,
    train_every_n_steps: int = 2,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train the Deep Q-Network (DQN) agent across `num_episodes` episodes and log metrics."""
    rng = np.random.default_rng(seed)
    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "preference_rewards": [],
        "diversity_bonuses": [],
        "epsilons": [],
        "losses": [],
    }

    total_env_steps = 0
    last_loss = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False

        ep_total_reward = 0.0
        ep_pref_reward = 0.0
        ep_div_bonus = 0.0
        ep_losses: List[float] = []

        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(obs, mask, greedy=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            next_mask = env.get_action_mask()

            agent.replay_buffer.push(obs, action, reward, next_obs, done, next_mask)
            total_env_steps += 1

            if total_env_steps % train_every_n_steps == 0:
                loss_val = agent.train_step()
                if loss_val is not None:
                    last_loss = loss_val
                    ep_losses.append(loss_val)

            ep_total_reward += reward
            ep_pref_reward += step_info["preference_reward"]
            ep_div_bonus += step_info["diversity_bonus"]
            obs = next_obs

        history["episode_rewards"].append(ep_total_reward)
        history["preference_rewards"].append(ep_pref_reward)
        history["diversity_bonuses"].append(ep_div_bonus)
        history["epsilons"].append(agent.epsilon)
        history["losses"].append(float(np.mean(ep_losses)) if ep_losses else last_loss)

        agent.decay_epsilon()

    return history
