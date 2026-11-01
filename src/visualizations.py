"""
Visualization Suite for MovieMind.
Generates publication-quality Matplotlib and Seaborn plots for Exploratory Data Analysis (EDA)
and all 7 Reinforcement Learning experimental studies.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preprocessing import PreprocessedData


# Set consistent modern plot styling
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 10


PALETTE = {
    "Random": "#8c969e",
    "Popularity": "#f59e0b",
    "Content-Based": "#10b981",
    "Tabular Q-Learning": "#3b82f6",
    "DQN (No Diversity, λ=0.0)": "#ec4899",
    "DQN (Diversity-Aware, λ=0.25)": "#8b5cf6",
    "DQN (Diversity-Aware)": "#8b5cf6",
}


def rolling_mean(values: List[float], window: int = 30) -> np.ndarray:
    """Compute smooth rolling mean with edge handling."""
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().values


def rolling_std(values: List[float], window: int = 30) -> np.ndarray:
    """Compute smooth rolling standard deviation."""
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).std().fillna(0.0).values


def plot_eda_overview(
    data: PreprocessedData, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a 4-panel Exploratory Data Analysis figure:
    1. Rating distribution across 100,000 interactions
    2. Movie catalog distribution across the 18 genres
    3. User rating activity distribution (interactions per user)
    4. Average rating vs. log-popularity scatter across movies
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "MovieLens 100K — Exploratory Data Analysis (EDA)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Rating Distribution
    ax1 = axes[0, 0]
    rating_counts = data.ratings_df["rating"].value_counts().sort_index()
    bars = ax1.bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"],
        edgecolor="black",
        alpha=0.88,
    )
    ax1.set_title("1. Distribution of Explicit Movie Ratings (1–5 Stars)", fontweight="bold")
    ax1.set_xlabel("Star Rating")
    ax1.set_ylabel("Number of Ratings")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 600,
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 2: Genre Frequency in Movie Catalog
    ax2 = axes[0, 1]
    genre_totals = data.movie_genre_matrix.sum(axis=0)
    order = np.argsort(genre_totals)
    sorted_genres = [data.genre_names[i] for i in order]
    sorted_counts = genre_totals[order]
    colors = plt.cm.viridis(np.linspace(0.2, 0.88, len(sorted_genres)))
    ax2.barh(sorted_genres, sorted_counts, color=colors, edgecolor="black", alpha=0.88)
    ax2.set_title("2. Number of Movies per Genre in Catalog", fontweight="bold")
    ax2.set_xlabel("Number of Movies")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel 3: User Activity Distribution
    ax3 = axes[1, 0]
    user_counts = data.ratings_df.groupby("user_idx").size()
    ax3.hist(user_counts, bins=35, color="#6366f1", edgecolor="black", alpha=0.82)
    ax3.axvline(
        user_counts.mean(),
        color="#ef4444",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {user_counts.mean():.1f} ratings/user",
    )
    ax3.axvline(
        user_counts.median(),
        color="#10b981",
        linestyle=":",
        linewidth=2,
        label=f"Median = {user_counts.median():.1f} ratings/user",
    )
    ax3.set_title("3. User Interaction Activity Distribution", fontweight="bold")
    ax3.set_xlabel("Ratings per User")
    ax3.set_ylabel("Number of Users")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 4: Movie Popularity vs Bayesian Rating (Training Set)
    ax4 = axes[1, 1]
    sc = ax4.scatter(
        data.item_train_count,
        data.item_bayesian_rating,
        c=data.item_popularity_score,
        cmap="plasma",
        alpha=0.65,
        edgecolors="none",
        s=28,
    )
    plt.colorbar(sc, ax=ax4, label="Normalized Popularity Score")
    ax4.set_title("4. Movie Training Rating Count vs. Bayesian Rating", fontweight="bold")
    ax4.set_xlabel("Training Rating Count")
    ax4.set_ylabel("Bayesian Smoothed Rating (Stars)")
    ax4.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_training_convergence(
    q_history: Dict[str, List[float]],
    dqn_history: Dict[str, List[float]],
    window: int = 35,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot RL training convergence:
    1. Episode Total Reward Curves (Tabular Q-Learning vs. DQN)
    2. Preference Reward vs. Diversity Bonus Evolution (DQN)
    3. Epsilon Decay Schedule (Exploration vs. Exploitation)
    4. Training Loss (DQN Huber Loss) & Tabular TD Error Curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Reinforcement Learning Training Dynamics — Tabular Q-Learning vs. Deep Q-Network (DQN)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    episodes_q = np.arange(1, len(q_history["episode_rewards"]) + 1)
    episodes_dqn = np.arange(1, len(dqn_history["episode_rewards"]) + 1)

    # 1. Episode Reward Convergence
    ax1 = axes[0, 0]
    q_rm = rolling_mean(q_history["episode_rewards"], window)
    q_rs = rolling_std(q_history["episode_rewards"], window)
    dqn_rm = rolling_mean(dqn_history["episode_rewards"], window)
    dqn_rs = rolling_std(dqn_history["episode_rewards"], window)

    ax1.plot(episodes_q, q_rm, color="#3b82f6", linewidth=2.2, label="Tabular Q-Learning (Rolling Mean)")
    ax1.fill_between(episodes_q, q_rm - 0.5 * q_rs, q_rm + 0.5 * q_rs, color="#3b82f6", alpha=0.18)
    ax1.plot(episodes_dqn, dqn_rm, color="#8b5cf6", linewidth=2.4, label="Deep Q-Network (DQN) (Rolling Mean)")
    ax1.fill_between(episodes_dqn, dqn_rm - 0.5 * dqn_rs, dqn_rm + 0.5 * dqn_rs, color="#8b5cf6", alpha=0.20)
    ax1.set_title("1. Episode Cumulative Reward Convergence", fontweight="bold")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cumulative Episode Reward")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # 2. Preference Reward & Diversity Bonus Decomposition
    ax2 = axes[0, 1]
    ax2.plot(
        episodes_dqn,
        rolling_mean(dqn_history["preference_rewards"], window),
        color="#10b981",
        linewidth=2.2,
        label="DQN Preference Reward",
    )
    ax2.plot(
        episodes_q,
        rolling_mean(q_history["preference_rewards"], window),
        color="#10b981",
        linestyle="--",
        linewidth=1.8,
        label="Q-Learning Preference Reward",
    )
    ax2.plot(
        episodes_dqn,
        rolling_mean(dqn_history["diversity_bonuses"], window),
        color="#f59e0b",
        linewidth=2.2,
        label="DQN Diversity Bonus (Unweighted)",
    )
    ax2.plot(
        episodes_q,
        rolling_mean(q_history["diversity_bonuses"], window),
        color="#f59e0b",
        linestyle="--",
        linewidth=1.8,
        label="Q-Learning Diversity Bonus (Unweighted)",
    )
    ax2.set_title("2. Preference Satisfaction vs. Genre Diversity Bonus", fontweight="bold")
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Cumulative Component Score")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # 3. Epsilon Decay Schedule
    ax3 = axes[1, 0]
    ax3.plot(episodes_dqn, dqn_history["epsilons"], color="#ef4444", linewidth=2.4, label="Epsilon (ε) Schedule")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        episodes_dqn,
        dqn_rm,
        color="#8b5cf6",
        linestyle=":",
        linewidth=2.0,
        label="DQN Reward",
    )
    ax3.set_title("3. Exploration-Exploitation Trade-off (Epsilon Decay)", fontweight="bold")
    ax3.set_xlabel("Training Episode")
    ax3.set_ylabel("Exploration Rate (ε)", color="#ef4444")
    ax3_twin.set_ylabel("DQN Rolling Reward", color="#8b5cf6")
    ax3.grid(True, linestyle="--", alpha=0.35)

    # 4. Training Loss & TD Error Curves
    ax4 = axes[1, 1]
    loss_rm = rolling_mean(dqn_history["losses"], window)
    td_rm = rolling_mean(q_history["td_errors"], window)
    ax4.plot(episodes_dqn, loss_rm, color="#8b5cf6", linewidth=2.2, label="DQN Smooth L1 (Huber) Loss")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(episodes_q, td_rm, color="#3b82f6", linestyle="--", linewidth=2.0, label="Tabular Mean |TD Error|")
    ax4.set_title("4. DQN Bellman Loss & Tabular Q-Learning TD Error", fontweight="bold")
    ax4.set_xlabel("Training Episode")
    ax4.set_ylabel("DQN Huber Loss", color="#8b5cf6")
    ax4_twin.set_ylabel("Tabular |TD Error|", color="#3b82f6")
    ax4.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_ranking_and_algorithm_comparison(
    eval_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a 4-panel comparison across all models:
    1. Precision@5, Recall@5, NDCG@5 on Held-Out Test Split
    2. Precision@10, Recall@10, NDCG@10 on Held-Out Test Split
    3. Intra-List Diversity (ILD), Genre Diversity, and Catalog Coverage
    4. Average Cumulative Reward (Decomposed into Preference + λ*Diversity)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Comprehensive Model Evaluation — Held-Out Ranking Accuracy, Diversity & Cumulative Reward",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    models = eval_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    # Panel 1: Top-5 Held-Out Ranking Metrics
    ax1 = axes[0, 0]
    ax1.bar(x - width, eval_df["Precision@5"], width, label="Precision@5", color="#3b82f6", edgecolor="black")
    ax1.bar(x, eval_df["Recall@5"], width, label="Recall@5", color="#10b981", edgecolor="black")
    ax1.bar(x + width, eval_df["NDCG@5"], width, label="NDCG@5", color="#8b5cf6", edgecolor="black")
    ax1.set_title("1. Held-Out Top-5 Ranking Metrics (Precision@5, Recall@5, NDCG@5)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=18, ha="right")
    ax1.set_ylabel("Metric Score")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 2: Top-10 Held-Out Ranking Metrics
    ax2 = axes[0, 1]
    ax2.bar(x - width, eval_df["Precision@10"], width, label="Precision@10", color="#2563eb", edgecolor="black")
    ax2.bar(x, eval_df["Recall@10"], width, label="Recall@10", color="#059669", edgecolor="black")
    ax2.bar(x + width, eval_df["NDCG@10"], width, label="NDCG@10", color="#7c3aed", edgecolor="black")
    ax2.set_title("2. Held-Out Top-10 Ranking Metrics (Precision@10, Recall@10, NDCG@10)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=18, ha="right")
    ax2.set_ylabel("Metric Score")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 3: Diversity & Catalog Coverage Comparison
    ax3 = axes[1, 0]
    ax3.bar(
        x - width,
        eval_df["Intra-List Diversity"],
        width,
        label="Intra-List Diversity (ILD)",
        color="#f59e0b",
        edgecolor="black",
    )
    ax3.bar(
        x,
        eval_df["Genre Diversity"],
        width,
        label="Genre Coverage Ratio",
        color="#ec4899",
        edgecolor="black",
    )
    ax3.bar(
        x + width,
        eval_df["Catalog Coverage"],
        width,
        label="Catalog Coverage",
        color="#06b6d4",
        edgecolor="black",
    )
    ax3.set_title("3. Recommendation Diversity & Catalog Coverage", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=18, ha="right")
    ax3.set_ylabel("Score in [0, 1]")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 4: Simulated Environment Cumulative Reward
    ax4 = axes[1, 1]
    pref_vals = eval_df["Avg Preference Reward"].values
    tot_vals = eval_df["Avg Cumulative Reward"].values
    div_contrib = np.maximum(0.0, tot_vals - pref_vals)
    ax4.bar(x, pref_vals, width=0.48, label="Preference Satisfaction Component", color="#10b981", edgecolor="black")
    ax4.bar(
        x,
        div_contrib,
        width=0.48,
        bottom=pref_vals,
        label="Weighted Diversity Bonus (λ · R_div)",
        color="#8b5cf6",
        edgecolor="black",
    )
    for i, tot in enumerate(tot_vals):
        ax4.text(i, tot + 0.12, f"{tot:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax4.set_title("4. Average Cumulative Episode Reward Decomposition", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=18, ha="right")
    ax4.set_ylabel("Cumulative Episode Reward (K=10)")
    ax4.set_ylim(0, max(tot_vals) * 1.15)
    ax4.legend()
    ax4.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_diversity_weight_ablation(
    lambda_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the impact of varying the diversity weight lambda:
    1. Preference Reward vs. Intra-List Diversity & Genre Diversity across lambda
    2. Held-out NDCG@10 & Precision@10 vs. Catalog Coverage across lambda
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Sensitivity & Ablation Study — Impact of Diversity Weight (λ) on DQN Recommendations",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    lambdas = lambda_df["Lambda"].values

    # Panel 1: Trade-off between Preference Satisfaction and Diversity
    ax1 = axes[0]
    ax1.plot(
        lambdas,
        lambda_df["Avg Preference Reward"] / 10.0,
        marker="o",
        linewidth=2.4,
        color="#10b981",
        label="Per-Step Preference Reward",
    )
    ax1.plot(
        lambdas,
        lambda_df["Intra-List Diversity"],
        marker="s",
        linewidth=2.4,
        color="#f59e0b",
        label="Intra-List Diversity (ILD)",
    )
    ax1.plot(
        lambdas,
        lambda_df["Genre Diversity"],
        marker="^",
        linewidth=2.4,
        color="#8b5cf6",
        label="Genre Coverage Ratio",
    )
    ax1.axvline(0.25, color="#ef4444", linestyle="--", alpha=0.7, label="Balanced Operating Point (λ=0.25)")
    ax1.set_title("1. Preference Alignment vs. Intra-List & Genre Diversity", fontweight="bold")
    ax1.set_xlabel("Diversity Weight (λ)")
    ax1.set_ylabel("Normalized Score in [0, 1]")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: Held-out Ranking Quality & Catalog Coverage across Lambda
    ax2 = axes[1]
    ax2.plot(
        lambdas,
        lambda_df["NDCG@10"],
        marker="o",
        linewidth=2.4,
        color="#3b82f6",
        label="Held-Out NDCG@10",
    )
    ax2.plot(
        lambdas,
        lambda_df["Precision@10"],
        marker="d",
        linewidth=2.4,
        color="#ec4899",
        label="Held-Out Precision@10",
    )
    ax2.plot(
        lambdas,
        lambda_df["Catalog Coverage"],
        marker="s",
        linewidth=2.4,
        color="#06b6d4",
        label="Catalog Coverage",
    )
    ax2.axvline(0.25, color="#ef4444", linestyle="--", alpha=0.7, label="Balanced Operating Point (λ=0.25)")
    ax2.set_title("2. Held-Out Accuracy (NDCG@10, Precision@10) & Catalog Coverage", fontweight="bold")
    ax2.set_xlabel("Diversity Weight (λ)")
    ax2.set_ylabel("Metric Value")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_genre_distribution_comparison(
    data: PreprocessedData,
    model_recs: Dict[str, Dict[int, List[int]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare the distribution of recommended genres across models to illustrate how
    diversity-aware RL prevents single-genre collapse and improves long-tail genre exposure.
    """
    genre_names = data.genre_names
    records = []

    for model_name, user_rec_dict in model_recs.items():
        genre_counts = np.zeros(data.num_genres, dtype=np.float64)
        for recs in user_rec_dict.values():
            genre_counts += data.movie_genre_matrix[recs].sum(axis=0)
        total = max(1.0, float(np.sum(genre_counts)))
        genre_shares = (genre_counts / total) * 100.0
        for g_idx, g_name in enumerate(genre_names):
            records.append(
                {
                    "Model": model_name,
                    "Genre": g_name,
                    "Share (%)": genre_shares[g_idx],
                }
            )

    df_genre = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(
        data=df_genre,
        x="Genre",
        y="Share (%)",
        hue="Model",
        ax=ax,
        edgecolor="black",
    )
    ax.set_title(
        "Recommended Genre Exposure Distribution Across Algorithms (% of Total Genre Tags)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Movie Genre")
    ax.set_ylabel("Share of Recommended Genre Tags (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Algorithm", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_epsilon_exploration_study(
    decay_histories: Dict[str, Dict[str, List[float]]],
    window: int = 35,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the Exploration vs. Exploitation study across different epsilon decay schedules.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.suptitle(
        "Exploration vs. Exploitation Analysis — Impact of Epsilon (ε) Decay Schedules",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    colors = ["#ef4444", "#8b5cf6", "#10b981", "#f59e0b"]
    for idx, (label, hist) in enumerate(decay_histories.items()):
        c = colors[idx % len(colors)]
        eps = np.arange(1, len(hist["epsilons"]) + 1)
        axes[0].plot(eps, hist["epsilons"], label=label, color=c, linewidth=2.2)
        axes[1].plot(
            eps,
            rolling_mean(hist["episode_rewards"], window),
            label=label,
            color=c,
            linewidth=2.2,
        )

    axes[0].set_title("1. Exploration Rate (ε) Decay Trajectories", fontweight="bold")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Epsilon (ε)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("2. Resulting Episode Reward Convergence", fontweight="bold")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Rolling Mean Episode Reward")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
