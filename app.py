"""
MovieMind — Interactive Streamlit Web Demonstration
A modern dark movie-streaming interface powered by Reinforcement Learning (DQN & Tabular Q-Learning)
and traditional recommendation baselines on the MovieLens 100K dataset.
"""

import os
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.agents import (
    DQNAgent,
    TabularQLearningAgent,
    train_dqn_agent,
    train_q_learning_agent,
)
from src.baselines import (
    ContentBasedRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from src.data_preprocessing import PreprocessedData, build_preprocessed_dataset
from src.environment import MovieRecEnv
from src.evaluation import (
    compute_genre_diversity,
    compute_intra_list_diversity,
)


st.set_page_config(
    page_title="MovieMind — RL Adaptive Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Cinema Streaming Theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1e293b;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0f172a 100%);
        border: 1px solid #4c1d95;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.0rem;
        color: #cbd5e1;
        margin: 0;
    }
    .movie-card {
        background: linear-gradient(180deg, #161f33 0%, #0f172a 100%);
        border: 1px solid #26334d;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }
    .movie-card:hover {
        border-color: #8b5cf6;
    }
    .rank-badge {
        display: inline-block;
        background: #7c3aed;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 9px;
        border-radius: 6px;
        margin-right: 8px;
    }
    .movie-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .genre-pill {
        display: inline-block;
        background: #1e293b;
        color: #a78bfa;
        border: 1px solid #334155;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        margin-right: 5px;
        margin-top: 6px;
    }
    .metric-pill {
        display: inline-block;
        background: #0f172a;
        color: #94a3b8;
        border: 1px solid #1e293b;
        font-size: 0.78rem;
        padding: 3px 9px;
        border-radius: 6px;
        margin-right: 8px;
        margin-top: 8px;
    }
    .kpi-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MovieLens 100K & Trained RL Models...")
def load_system_resources() -> Tuple[PreprocessedData, Dict[str, object]]:
    """Load preprocessed MovieLens 100K dataset and trained RL agents."""
    data = build_preprocessed_dataset(
        data_dir="data", train_ratio=0.80, candidate_pool_size=300, seed=42
    )
    env = MovieRecEnv(data=data, episode_length=10, diversity_lambda=0.25, seed=42)

    # 1. Tabular Q-Learning Agent
    q_agent = TabularQLearningAgent(
        num_states=env.num_discrete_states,
        num_actions=data.num_movies,
        data=data,
        lr=0.15,
        gamma=0.85,
        seed=42,
    )
    q_path = "models/q_learning_agent.npz"
    if os.path.exists(q_path):
        npz = np.load(q_path)
        q_agent.q_table = npz["q_table"]
        q_agent.genre_q_table = npz["genre_q_table"]
        q_agent.visit_counts = npz["visit_counts"]
    else:
        train_q_learning_agent(env, q_agent, num_episodes=600, seed=42)

    # 2. DQN (No Diversity, lambda=0.0)
    dqn_no_div = DQNAgent(
        state_dim=env.state_dim,
        num_actions=data.num_movies,
        data=data,
        hidden_dim=128,
        diversity_lambda=0.0,
        seed=42,
    )
    dqn_no_div_path = "models/dqn_no_diversity.pt"
    if os.path.exists(dqn_no_div_path):
        dqn_no_div.policy_net.load_state_dict(
            torch.load(dqn_no_div_path, map_location="cpu", weights_only=True)
        )
        dqn_no_div.target_net.load_state_dict(dqn_no_div.policy_net.state_dict())
    else:
        env0 = MovieRecEnv(data=data, episode_length=10, diversity_lambda=0.0, seed=42)
        train_dqn_agent(env0, dqn_no_div, num_episodes=600, seed=42)

    # 3. DQN (Diversity-Aware, lambda=0.25)
    dqn_div = DQNAgent(
        state_dim=env.state_dim,
        num_actions=data.num_movies,
        data=data,
        hidden_dim=128,
        diversity_lambda=0.25,
        seed=42,
    )
    dqn_div_path = "models/dqn_diversity.pt"
    if os.path.exists(dqn_div_path):
        dqn_div.policy_net.load_state_dict(
            torch.load(dqn_div_path, map_location="cpu", weights_only=True)
        )
        dqn_div.target_net.load_state_dict(dqn_div.policy_net.state_dict())
    else:
        train_dqn_agent(env, dqn_div, num_episodes=600, seed=42)

    models = {
        "DQN (Diversity-Aware)": dqn_div,
        "DQN (No Diversity, λ=0.0)": dqn_no_div,
        "Tabular Q-Learning": q_agent,
        "Content-Based Filtering": ContentBasedRecommender(data),
        "Popularity-Based": PopularityRecommender(data),
        "Random Baseline": RandomRecommender(data, seed=42),
    }
    return data, models


def init_session_state(data: PreprocessedData) -> None:
    """Initialize interactive session state for user preferences and feedback history."""
    if "active_user_idx" not in st.session_state:
        st.session_state.active_user_idx = 0
    if "custom_prefs" not in st.session_state:
        st.session_state.custom_prefs = data.user_genre_prefs[0].copy()
    if "feedback_history" not in st.session_state:
        # List of dicts: {movie_idx, title, genres, action_type, rating, delta}
        st.session_state.feedback_history = []
    if "session_seen_movies" not in st.session_state:
        st.session_state.session_seen_movies = set()
    if "session_genre_exposure" not in st.session_state:
        st.session_state.session_genre_exposure = np.zeros(data.num_genres, dtype=np.float32)


def apply_user_feedback(
    data: PreprocessedData,
    movie_idx: int,
    feedback_type: str,
    star_rating: float = 5.0,
) -> None:
    """
    Update the user's live genre preference profile and session genre exposure
    in response to explicit Like / Dislike / Star Rating feedback.
    """
    movie_row = data.movies_df.loc[movie_idx]
    m_genres = data.movie_genre_matrix[movie_idx]

    if feedback_type == "like":
        delta = 0.18
        label = "👍 Liked (+0.18 genre boost)"
    elif feedback_type == "dislike":
        delta = -0.22
        label = "👎 Disliked (-0.22 genre penalty)"
    else:
        delta = (float(star_rating) - 3.0) * 0.10
        label = f"⭐ Rated {int(star_rating)}/5 ({delta:+.2f} genre shift)"

    prefs = st.session_state.custom_prefs.copy()
    prefs = np.clip(prefs + delta * m_genres, 0.02, 1.0)
    if np.max(prefs) > 0:
        prefs = prefs / np.max(prefs)
    st.session_state.custom_prefs = prefs.astype(np.float32)

    st.session_state.session_seen_movies.add(int(movie_idx))
    st.session_state.session_genre_exposure += m_genres

    st.session_state.feedback_history.insert(
        0,
        {
            "movie_idx": int(movie_idx),
            "Title": movie_row["title"],
            "Genres": movie_row["genres_str"],
            "Feedback": label,
        },
    )


def generate_adaptive_recommendations(
    data: PreprocessedData,
    models: Dict[str, object],
    algorithm_name: str,
    user_idx: int,
    live_user_prefs: np.ndarray,
    session_seen: Set[int],
    session_exposure_counts: np.ndarray,
    diversity_lambda: float,
    top_k: int = 10,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """
    Run a sequential episode in MovieRecEnv using the selected algorithm, live user
    preference profile, and interactive feedback history.
    """
    env = MovieRecEnv(
        data=data,
        episode_length=top_k,
        diversity_lambda=diversity_lambda,
        use_candidate_pool=True,
        seed=42,
    )
    obs, info = env.reset(options={"user_idx": user_idx})

    # Inject live interactive user preferences and feedback history into the environment
    env.current_user_prefs = live_user_prefs.copy()
    for m_seen in session_seen:
        env._eligible_mask[int(m_seen)] = False

    if np.sum(session_exposure_counts) > 0:
        env.session_genre_counts = session_exposure_counts.copy() * 0.5

    obs = env._get_obs()
    s_disc = env.discretize_state(obs)
    model = models[algorithm_name]

    rec_items: List[Dict[str, object]] = []
    rec_indices: List[int] = []

    for step_i in range(top_k):
        mask = env.get_action_mask()
        eligible = np.flatnonzero(mask)
        if len(eligible) == 0:
            break

        if algorithm_name in ("DQN (Diversity-Aware)", "DQN (No Diversity, λ=0.0)"):
            dqn_ag: DQNAgent = model  # type: ignore
            q_vals = dqn_ag.get_q_values(obs)
            # Apply slider adjustment only when using Diversity-Aware DQN and slider differs from trained default (0.25)
            if algorithm_name == "DQN (Diversity-Aware)" and abs(diversity_lambda - dqn_ag.diversity_lambda) > 1e-4:
                if step_i > 0 or np.sum(env.session_genre_counts) > 0:
                    exp_vec = env.session_genre_counts / max(1.0, float(step_i + 1))
                    novelty_adj = 0.35 * ((1.0 - 1.8 * exp_vec) @ data.movie_genre_norm.T) * (
                        diversity_lambda - dqn_ag.diversity_lambda
                    )
                    q_vals = q_vals + novelty_adj
            masked_q = np.where(mask, q_vals, -1e9)
            action = int(np.argmax(masked_q))
            action_score = float(q_vals[action])
        elif algorithm_name == "Tabular Q-Learning":
            q_ag: TabularQLearningAgent = model  # type: ignore
            q_vals = q_ag.get_state_q_values(s_disc, obs=obs)
            masked_q = np.where(mask, q_vals, -1e9)
            action = int(np.argmax(masked_q))
            action_score = float(q_vals[action])
        else:
            action = int(model.select_action(obs, mask, user_idx=user_idx))  # type: ignore
            action_score = float(data.train_preference_matrix[user_idx, action])

        next_obs, total_r, terminated, truncated, step_info = env.step(action)
        m_idx = int(step_info["movie_idx"])
        m_row = data.movies_df.loc[m_idx]

        rec_indices.append(m_idx)
        rec_items.append(
            {
                "rank": step_i + 1,
                "movie_idx": m_idx,
                "title": m_row["title"],
                "year": int(m_row["year"]),
                "genres_list": m_row["genres_list"],
                "genres_str": m_row["genres_str"],
                "bayesian_rating": float(m_row["bayesian_rating"]),
                "train_votes": int(m_row["train_rating_count"]),
                "pref_reward": float(step_info["preference_reward"]),
                "div_bonus": float(step_info["diversity_bonus"]),
                "total_reward": float(total_r),
                "action_score": action_score,
            }
        )

        obs = next_obs
        s_disc = step_info["discrete_state"]
        if terminated or truncated:
            break

    ild = compute_intra_list_diversity(rec_indices, data.movie_genre_norm)
    g_cov, g_ent = compute_genre_diversity(rec_indices, data.movie_genre_matrix)
    summary_stats = {
        "ild": ild,
        "genre_coverage": g_cov,
        "genre_entropy": g_ent,
        "unique_genres": int(round(g_cov * data.num_genres)),
        "avg_pref": float(np.mean([x["pref_reward"] for x in rec_items])) if rec_items else 0.0,
        "cum_reward": float(np.sum([x["total_reward"] for x in rec_items])) if rec_items else 0.0,
    }
    return rec_items, summary_stats


def main() -> None:
    data, models = load_system_resources()
    init_session_state(data)

    # Hero Header
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎬 MovieMind — Adaptive RL Movie Streaming Studio</div>
            <p class="hero-sub">
                Reinforcement Learning recommendation engine balancing personalized user satisfaction
                with sequential genre diversity on the MovieLens 100K catalog.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Recommendation Engine")
        algorithm_name = st.selectbox(
            "Select Algorithm",
            options=[
                "DQN (Diversity-Aware)",
                "DQN (No Diversity, λ=0.0)",
                "Tabular Q-Learning",
                "Content-Based Filtering",
                "Popularity-Based",
                "Random Baseline",
            ],
            index=0,
            help="Compare Deep Q-Network, Tabular Q-Learning, and traditional recommender baselines.",
        )

        diversity_lambda = st.slider(
            "Diversity Weight (λ)",
            min_value=0.0,
            max_value=0.80,
            value=0.25 if "No Diversity" not in algorithm_name else 0.0,
            step=0.05,
            help="Controls Total Reward = Preference Reward + λ × Diversity Bonus.",
        )

        top_k = st.slider(
            "Recommendation List Size (K)",
            min_value=5,
            max_value=15,
            value=10,
            step=1,
        )

        st.markdown("---")
        st.markdown("### 👤 User Profile & Genres")

        user_options = [
            f"User #{int(data.users_df.loc[u, 'user_id'])} ({data.users_df.loc[u, 'Age' if 'Age' in data.users_df else 'age']}y, {data.users_df.loc[u, 'occupation']})"
            for u in range(min(150, data.num_users))
        ]
        selected_user_label = st.selectbox(
            "Base MovieLens User Profile",
            options=user_options,
            index=st.session_state.active_user_idx,
        )
        new_user_idx = user_options.index(selected_user_label)

        if new_user_idx != st.session_state.active_user_idx:
            st.session_state.active_user_idx = new_user_idx
            st.session_state.custom_prefs = data.user_genre_prefs[new_user_idx].copy()
            st.session_state.feedback_history = []
            st.session_state.session_seen_movies = set()
            st.session_state.session_genre_exposure = np.zeros(data.num_genres, dtype=np.float32)

        # Allow explicit genre boosting/selection
        current_top_genres = [
            data.genre_names[i]
            for i in np.argsort(-st.session_state.custom_prefs)[:3]
        ]
        selected_fav_genres = st.multiselect(
            "Boost Favorite Genres",
            options=data.genre_names,
            default=current_top_genres,
            help="Select genres to emphasize in your live preference vector.",
        )

        if st.button("Apply Genre Boost", use_container_width=True):
            base_p = data.user_genre_prefs[st.session_state.active_user_idx].copy() * 0.45
            for g_name in selected_fav_genres:
                g_i = data.genre_names.index(g_name)
                base_p[g_i] = 1.0
            if np.max(base_p) > 0:
                base_p = base_p / np.max(base_p)
            st.session_state.custom_prefs = base_p.astype(np.float32)

        if st.button("🔄 Reset Session Feedback", use_container_width=True):
            st.session_state.custom_prefs = data.user_genre_prefs[
                st.session_state.active_user_idx
            ].copy()
            st.session_state.feedback_history = []
            st.session_state.session_seen_movies = set()
            st.session_state.session_genre_exposure = np.zeros(data.num_genres, dtype=np.float32)
            st.rerun()

    # Compute adaptive recommendations
    rec_items, summary_stats = generate_adaptive_recommendations(
        data=data,
        models=models,
        algorithm_name=algorithm_name,
        user_idx=st.session_state.active_user_idx,
        live_user_prefs=st.session_state.custom_prefs,
        session_seen=st.session_state.session_seen_movies,
        session_exposure_counts=st.session_state.session_genre_exposure,
        diversity_lambda=diversity_lambda,
        top_k=top_k,
    )

    # Top-level Navigation Tabs
    tab_recs, tab_explore, tab_analytics = st.tabs(
        [
            "🍿 Personalized & Adaptive Feed",
            "🧭 Genre Explorer & Catalog",
            "📊 RL Evaluation & Benchmarks",
        ]
    )

    # =========================================================================
    # TAB 1: Personalized & Adaptive Feed
    # =========================================================================
    with tab_recs:
        # KPI Metrics Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">Active Policy</div>'
                f'<div class="kpi-value" style="font-size:1.05rem;color:#a78bfa;">{algorithm_name}</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">Cumulative Episode Reward</div>'
                f'<div class="kpi-value">{summary_stats["cum_reward"]:.2f}</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">Intra-List Diversity (ILD)</div>'
                f'<div class="kpi-value">{summary_stats["ild"]:.3f}</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">Genres Covered in Feed</div>'
                f'<div class="kpi-value">{summary_stats["unique_genres"]} / {data.num_genres}</div></div>',
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-label">Session Feedback Events</div>'
                f'<div class="kpi-value">{len(st.session_state.feedback_history)}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_feed, col_profile = st.columns([1.85, 1.0], gap="large")

        with col_feed:
            st.subheader(f"Top-{len(rec_items)} Adaptive Movie Recommendations")
            st.caption(
                "Interact with any movie below using 👍 Like, 👎 Dislike, or ⭐ Star Ratings. "
                "The RL state immediately updates and re-ranks your unseen candidates."
            )

            for item in rec_items:
                m_idx = int(item["movie_idx"])
                genre_badges_html = "".join(
                    f'<span class="genre-pill">{g}</span>' for g in item["genres_list"]
                )
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div>
                            <span class="rank-badge">#{item['rank']}</span>
                            <span class="movie-title">{item['title']}</span>
                        </div>
                        <div style="margin-top:4px;">{genre_badges_html}</div>
                        <div>
                            <span class="metric-pill">⭐ Bayesian Rating: <b>{item['bayesian_rating']:.2f}</b> ({item['train_votes']} votes)</span>
                            <span class="metric-pill">🎯 Preference Match: <b>{item['pref_reward']*100:.0f}%</b></span>
                            <span class="metric-pill">🌈 Diversity Bonus: <b>{item['div_bonus']*100:.0f}%</b></span>
                            <span class="metric-pill">⚡ Step Reward: <b>{item['total_reward']:.2f}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                b_col1, b_col2, b_col3 = st.columns([1.0, 1.0, 2.2])
                with b_col1:
                    if st.button("👍 Like", key=f"like_{m_idx}", use_container_width=True):
                        apply_user_feedback(data, m_idx, "like")
                        st.rerun()
                with b_col2:
                    if st.button("👎 Dislike", key=f"dislike_{m_idx}", use_container_width=True):
                        apply_user_feedback(data, m_idx, "dislike")
                        st.rerun()
                with b_col3:
                    rating_choice = st.select_slider(
                        "Rate Movie",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        key=f"slider_{m_idx}",
                        label_visibility="collapsed",
                    )
                    if st.button(
                        f"Submit ⭐ {rating_choice}/5 Rating",
                        key=f"rate_btn_{m_idx}",
                        use_container_width=True,
                    ):
                        apply_user_feedback(data, m_idx, "rate", float(rating_choice))
                        st.rerun()

        with col_profile:
            st.subheader("Live Markov State & Genre Profile")
            # Plot User Preference Profile vs. Recommended Feed Genre Distribution
            rec_indices = [int(x["movie_idx"]) for x in rec_items]
            feed_genre_counts = (
                data.movie_genre_matrix[rec_indices].sum(axis=0)
                if rec_indices
                else np.zeros(data.num_genres)
            )
            feed_genre_norm = feed_genre_counts / max(1.0, float(np.max(feed_genre_counts)))

            fig_prof, ax_prof = plt.subplots(figsize=(6.2, 6.5))
            fig_prof.patch.set_facecolor("#111827")
            ax_prof.set_facecolor("#111827")

            y_pos = np.arange(data.num_genres)
            ax_prof.barh(
                y_pos - 0.18,
                st.session_state.custom_prefs,
                height=0.36,
                label="User Genre Affinity p(u)",
                color="#8b5cf6",
            )
            ax_prof.barh(
                y_pos + 0.18,
                feed_genre_norm,
                height=0.36,
                label="Recommended Feed Exposure",
                color="#10b981",
            )
            ax_prof.set_yticks(y_pos)
            ax_prof.set_yticklabels(data.genre_names, color="#e2e8f0", fontsize=8.5)
            ax_prof.set_xlabel("Normalized Weight in [0, 1]", color="#cbd5e1", fontsize=9)
            ax_prof.set_title(
                "User Preference Profile vs. Feed Genre Exposure",
                color="#f8fafc",
                fontweight="bold",
                fontsize=10.5,
            )
            ax_prof.tick_params(colors="#cbd5e1")
            for spine in ax_prof.spines.values():
                spine.set_color("#334155")
            ax_prof.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)
            ax_prof.grid(axis="x", linestyle="--", alpha=0.25, color="#64748b")
            plt.tight_layout()
            st.pyplot(fig_prof)
            plt.close(fig_prof)

            st.markdown("#### 📜 Live Session Feedback Log")
            if st.session_state.feedback_history:
                fb_df = pd.DataFrame(st.session_state.feedback_history)[
                    ["Title", "Genres", "Feedback"]
                ]
                st.dataframe(fb_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No feedback recorded in this session yet. Click 👍 Like, 👎 Dislike, or submit a ⭐ Star Rating on any movie card to watch the RL state adapt!"
                )

            # Show user's historical top rated training movies
            st.markdown("#### 🎞️ User's Historical Training Favorites")
            u_train = (
                data.train_df[data.train_df["user_idx"] == st.session_state.active_user_idx]
                .sort_values("rating", ascending=False)
                .head(6)
            )
            u_hist_rows = []
            for _, r_row in u_train.iterrows():
                m_info = data.movies_df.loc[int(r_row["movie_idx"])]
                u_hist_rows.append(
                    {
                        "Movie": m_info["title"],
                        "Genres": m_info["genres_str"],
                        "Rating": f"{int(r_row['rating'])} ⭐",
                    }
                )
            st.dataframe(pd.DataFrame(u_hist_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Genre Explorer & Catalog Discovery
    # =========================================================================
    with tab_explore:
        st.subheader("🧭 Explore MovieLens 100K by Genre")
        col_g1, col_g2, col_g3 = st.columns([1.5, 1.5, 1.0])
        with col_g1:
            explore_genre = st.selectbox("Filter by Genre", options=["All Genres"] + data.genre_names)
        with col_g2:
            sort_by = st.selectbox(
                "Sort Catalog By",
                options=["Bayesian Rating", "Popularity (Training Votes)", "Release Year (Newest)"],
            )
        with col_g3:
            min_votes = st.number_input("Minimum Training Votes", min_value=0, max_value=200, value=20, step=10)

        cat_df = data.movies_df.copy()
        cat_df = cat_df[cat_df["train_rating_count"] >= min_votes]
        if explore_genre != "All Genres":
            cat_df = cat_df[cat_df[explore_genre] == 1]

        if sort_by == "Bayesian Rating":
            cat_df = cat_df.sort_values("bayesian_rating", ascending=False)
        elif sort_by == "Popularity (Training Votes)":
            cat_df = cat_df.sort_values("train_rating_count", ascending=False)
        else:
            cat_df = cat_df.sort_values("year", ascending=False)

        display_cat = cat_df[
            ["title", "year", "genres_str", "bayesian_rating", "train_rating_count"]
        ].head(30).rename(
            columns={
                "title": "Movie Title",
                "year": "Year",
                "genres_str": "Genres",
                "bayesian_rating": "Bayesian Rating",
                "train_rating_count": "Training Votes",
            }
        )
        st.dataframe(display_cat, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: RL Evaluation & Benchmarks
    # =========================================================================
    with tab_analytics:
        st.subheader("📊 Held-Out Evaluation Metrics & Experimental Studies")
        st.markdown(
            """
            All models are evaluated on a consistent cohort of **250 held-out MovieLens users** using strict
            **80:20 per-user train/test splitting**. Training interactions are used exclusively for preference profile
            construction and RL environment simulation, while held-out test ratings are reserved strictly for offline ranking evaluation.
            """
        )

        if os.path.exists("results/model_comparison.csv"):
            comp_df = pd.read_csv("results/model_comparison.csv")
            st.markdown("#### 1. Main Algorithm Comparison Table (Held-Out Test Set)")
            st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)

        if os.path.exists("results/lambda_ablation.csv"):
            lam_df = pd.read_csv("results/lambda_ablation.csv")
            st.markdown("#### 2. Diversity Weight (λ) Sensitivity & Ablation Table")
            st.dataframe(
                lam_df[
                    [
                        "Lambda",
                        "Precision@5",
                        "Precision@10",
                        "Recall@10",
                        "NDCG@10",
                        "Intra-List Diversity",
                        "Genre Diversity",
                        "Catalog Coverage",
                        "Avg Preference Reward",
                        "Avg Cumulative Reward",
                    ]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 3. Experimental Visualizations")
        fig_files = [
            ("figures/03_model_comparison.png", "Algorithm & Held-Out Ranking Comparison"),
            ("figures/02_training_convergence.png", "RL Training Reward & Loss Convergence"),
            ("figures/04_diversity_weight_ablation.png", "Diversity Weight (λ) Sensitivity Analysis"),
            ("figures/05_genre_distribution.png", "Genre Exposure Distribution Across Models"),
            ("figures/06_epsilon_exploration.png", "Exploration vs. Exploitation (Epsilon Decay)"),
            ("figures/01_eda_overview.png", "MovieLens 100K Exploratory Data Analysis"),
        ]
        for f_path, caption in fig_files:
            if os.path.exists(f_path):
                st.markdown(f"**{caption}**")
                st.image(f_path, use_container_width=True)


if __name__ == "__main__":
    main()
