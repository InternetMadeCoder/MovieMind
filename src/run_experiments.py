"""
Master Experiment & Model Persistence Runner for MovieMind.
Runs all 7 experimental studies, saves trained RL model checkpoints to `models/`,
exports evaluation tables to `results/`, and saves figures to `figures/`.
"""

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
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
    evaluate_model,
    generate_user_recommendations,
    select_evaluation_users,
)
from src.visualizations import (
    plot_diversity_weight_ablation,
    plot_eda_overview,
    plot_epsilon_exploration_study,
    plot_genre_distribution_comparison,
    plot_ranking_and_algorithm_comparison,
    plot_training_convergence,
)


def run_all_experiments(
    data_dir: str = "data",
    models_dir: str = "models",
    results_dir: str = "results",
    figures_dir: str = "figures",
    num_episodes: int = 900,
    num_eval_users: int = 250,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the full experimental suite and persist all artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    t_start = time.time()
    print("1. Loading and preprocessing MovieLens 100K dataset (80:20 per-user split)...")
    data: PreprocessedData = build_preprocessed_dataset(
        data_dir=data_dir,
        train_ratio=0.80,
        candidate_pool_size=300,
        svd_rank=16,
        seed=seed,
    )
    eval_users = select_evaluation_users(
        data, num_eval_users=num_eval_users, min_test_relevant=3, seed=seed
    )
    print(
        f"   Users: {data.num_users} | Movies: {data.num_movies} | "
        f"Train ratings: {len(data.train_df):,} | Test ratings: {len(data.test_df):,} | "
        f"Eval cohort: {len(eval_users)} users"
    )

    # Save EDA Figure
    plot_eda_overview(data, save_path=os.path.join(figures_dir, "01_eda_overview.png"))

    # 2. Train Tabular Q-Learning Agent (lambda = 0.25)
    print("2. Training Tabular Q-Learning Agent (λ = 0.25)...")
    env_main = MovieRecEnv(
        data=data, episode_length=10, diversity_lambda=0.25, seed=seed
    )
    q_agent = TabularQLearningAgent(
        num_states=env_main.num_discrete_states,
        num_actions=data.num_movies,
        data=data,
        lr=0.15,
        gamma=0.85,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )
    q_history = train_q_learning_agent(
        env_main, q_agent, num_episodes=num_episodes, seed=seed
    )

    # Save Tabular Q-Learning tables
    np.savez_compressed(
        os.path.join(models_dir, "q_learning_agent.npz"),
        q_table=q_agent.q_table,
        genre_q_table=q_agent.genre_q_table,
        visit_counts=q_agent.visit_counts,
    )

    # 3. Train Deep Q-Network (DQN) WITHOUT Diversity Bonus (lambda = 0.0)
    print("3. Training Baseline DQN without Diversity Bonus (λ = 0.0)...")
    env_no_div = MovieRecEnv(
        data=data, episode_length=10, diversity_lambda=0.0, seed=seed
    )
    dqn_no_div = DQNAgent(
        state_dim=env_no_div.state_dim,
        num_actions=data.num_movies,
        data=data,
        hidden_dim=128,
        diversity_lambda=0.0,
        lr=1.2e-3,
        gamma=0.85,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=80,
        seed=seed,
    )
    dqn_no_div_history = train_dqn_agent(
        env_no_div, dqn_no_div, num_episodes=num_episodes, train_every_n_steps=2, seed=seed
    )
    torch.save(
        dqn_no_div.policy_net.state_dict(),
        os.path.join(models_dir, "dqn_no_diversity.pt"),
    )

    # 4. Train Diversity-Aware Deep Q-Network (DQN) (lambda = 0.25)
    print("4. Training Diversity-Aware DQN (λ = 0.25)...")
    dqn_div = DQNAgent(
        state_dim=env_main.state_dim,
        num_actions=data.num_movies,
        data=data,
        hidden_dim=128,
        diversity_lambda=0.25,
        lr=1.2e-3,
        gamma=0.85,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=80,
        seed=seed,
    )
    dqn_div_history = train_dqn_agent(
        env_main, dqn_div, num_episodes=num_episodes, train_every_n_steps=2, seed=seed
    )
    torch.save(
        dqn_div.policy_net.state_dict(),
        os.path.join(models_dir, "dqn_diversity.pt"),
    )

    # Save Training Convergence Figure
    plot_training_convergence(
        q_history,
        dqn_div_history,
        window=35,
        save_path=os.path.join(figures_dir, "02_training_convergence.png"),
    )

    # 5. Evaluate Baselines & RL Models on Held-Out Test Split
    print("5. Evaluating Traditional Baselines and RL Models on Held-Out Test Set...")
    rand_rec = RandomRecommender(data, seed=seed)
    pop_rec = PopularityRecommender(data)
    cb_rec = ContentBasedRecommender(data)

    policy_map = [
        ("Random", lambda obs, mask, s_disc, u: rand_rec.select_action(obs, mask, u)),
        ("Popularity", lambda obs, mask, s_disc, u: pop_rec.select_action(obs, mask, u)),
        ("Content-Based", lambda obs, mask, s_disc, u: cb_rec.select_action(obs, mask, u)),
        (
            "Tabular Q-Learning",
            lambda obs, mask, s_disc, u: q_agent.select_action(
                s_disc, mask, obs=obs, greedy=True
            ),
        ),
        (
            "DQN (No Diversity, λ=0.0)",
            lambda obs, mask, s_disc, u: dqn_no_div.select_action(obs, mask, greedy=True),
        ),
        (
            "DQN (Diversity-Aware, λ=0.25)",
            lambda obs, mask, s_disc, u: dqn_div.select_action(obs, mask, greedy=True),
        ),
    ]

    eval_rows = []
    model_user_recs: Dict[str, Dict[int, List[int]]] = {}
    for model_name, policy_fn in policy_map:
        metrics, recs_dict = evaluate_model(
            model_name=model_name,
            policy_fn=policy_fn,
            data=data,
            eval_users=eval_users,
            diversity_lambda=0.25,
            episode_length=10,
        )
        eval_rows.append(metrics)
        model_user_recs[model_name] = recs_dict

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    plot_ranking_and_algorithm_comparison(
        eval_df, save_path=os.path.join(figures_dir, "03_model_comparison.png")
    )

    # Genre distribution comparison across representative models
    selected_genre_models = {
        "Popularity": model_user_recs["Popularity"],
        "Content-Based": model_user_recs["Content-Based"],
        "DQN (No Diversity, λ=0.0)": model_user_recs["DQN (No Diversity, λ=0.0)"],
        "DQN (Diversity-Aware, λ=0.25)": model_user_recs["DQN (Diversity-Aware, λ=0.25)"],
    }
    plot_genre_distribution_comparison(
        data,
        selected_genre_models,
        save_path=os.path.join(figures_dir, "05_genre_distribution.png"),
    )

    # 6. Diversity Weight Ablation Study (lambda in {0.0, 0.10, 0.25, 0.50, 0.80})
    print("6. Running Diversity Weight (λ) Sensitivity Study...")
    lambda_values = [0.0, 0.10, 0.25, 0.50, 0.80]
    lambda_rows = []
    for lam in lambda_values:
        if lam == 0.0:
            agent_lam = dqn_no_div
        elif lam == 0.25:
            agent_lam = dqn_div
        else:
            env_lam = MovieRecEnv(
                data=data, episode_length=10, diversity_lambda=lam, seed=seed
            )
            agent_lam = DQNAgent(
                state_dim=env_lam.state_dim,
                num_actions=data.num_movies,
                data=data,
                hidden_dim=128,
                diversity_lambda=lam,
                lr=1.2e-3,
                gamma=0.85,
                epsilon_decay=0.994,
                seed=seed,
            )
            train_dqn_agent(
                env_lam, agent_lam, num_episodes=650, train_every_n_steps=2, seed=seed
            )

        m_lam, _ = evaluate_model(
            model_name=f"DQN (λ={lam:.2f})",
            policy_fn=lambda obs, mask, s_disc, u, ag=agent_lam: ag.select_action(
                obs, mask, greedy=True
            ),
            data=data,
            eval_users=eval_users,
            diversity_lambda=lam,
            episode_length=10,
        )
        m_lam["Lambda"] = lam
        lambda_rows.append(m_lam)

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(os.path.join(results_dir, "lambda_ablation.csv"), index=False)
    plot_diversity_weight_ablation(
        lambda_df, save_path=os.path.join(figures_dir, "04_diversity_weight_ablation.png")
    )

    # 7. Exploration vs. Exploitation (Epsilon Decay) Study
    print("7. Running Exploration vs. Exploitation (Epsilon Decay) Study...")
    decay_configs = [
        ("Fast Decay (γ_ε = 0.988)", 0.988),
        ("Balanced Decay (γ_ε = 0.995)", 0.995),
        ("Slow Decay (γ_ε = 0.998)", 0.998),
    ]
    decay_histories: Dict[str, Dict[str, List[float]]] = {}
    epsilon_eval_rows = []
    for label, decay_rate in decay_configs:
        if decay_rate == 0.995:
            hist_eps = dqn_div_history
            ag_eps = dqn_div
        else:
            env_eps = MovieRecEnv(
                data=data, episode_length=10, diversity_lambda=0.25, seed=seed
            )
            ag_eps = DQNAgent(
                state_dim=env_eps.state_dim,
                num_actions=data.num_movies,
                data=data,
                hidden_dim=128,
                diversity_lambda=0.25,
                lr=1.2e-3,
                gamma=0.85,
                epsilon_decay=decay_rate,
                seed=seed,
            )
            hist_eps = train_dqn_agent(
                env_eps, ag_eps, num_episodes=num_episodes, train_every_n_steps=2, seed=seed
            )
        decay_histories[label] = hist_eps
        m_eps, _ = evaluate_model(
            model_name=label,
            policy_fn=lambda obs, mask, s_disc, u, ag=ag_eps: ag.select_action(
                obs, mask, greedy=True
            ),
            data=data,
            eval_users=eval_users,
            diversity_lambda=0.25,
        )
        m_eps["Epsilon Decay"] = decay_rate
        m_eps["Final Rolling Reward"] = float(np.mean(hist_eps["episode_rewards"][-50:]))
        epsilon_eval_rows.append(m_eps)

    epsilon_df = pd.DataFrame(epsilon_eval_rows)
    epsilon_df.to_csv(os.path.join(results_dir, "epsilon_ablation.csv"), index=False)
    plot_epsilon_exploration_study(
        decay_histories,
        window=35,
        save_path=os.path.join(figures_dir, "06_epsilon_exploration.png"),
    )

    # 8. Sample Recommendations for Different User Archetypes
    print("8. Generating Qualitative Recommendations for Sample Users...")
    sample_users = eval_users[:3]
    sample_records = []
    for u in sample_users:
        top_genres_idx = np.argsort(-data.user_genre_prefs[u])[:3]
        top_genres_str = ", ".join(data.genre_names[g] for g in top_genres_idx)
        rollout = generate_user_recommendations(
            env_main,
            lambda obs, mask, s_disc, uid: dqn_div.select_action(obs, mask, greedy=True),
            u,
        )
        for rank_i, m_idx in enumerate(rollout["recommended_movies"], start=1):
            row = data.movies_df.loc[m_idx]
            in_test_rel = m_idx in data.test_relevant_movies[u]
            sample_records.append(
                {
                    "User ID": int(data.users_df.loc[u, "user_id"]),
                    "User Top Genres (Train)": top_genres_str,
                    "Rank": rank_i,
                    "Movie Title": row["title"],
                    "Movie Genres": row["genres_str"],
                    "Train Bayesian Rating": float(row["bayesian_rating"]),
                    "Held-Out Test Hit": "Yes" if in_test_rel else "No",
                }
            )

    sample_recs_df = pd.DataFrame(sample_records)
    sample_recs_df.to_csv(
        os.path.join(results_dir, "sample_user_recommendations.csv"), index=False
    )

    # Save training histories for fast loading
    with open(os.path.join(results_dir, "training_histories.json"), "w") as f:
        json.dump(
            {
                "q_learning": q_history,
                "dqn_no_diversity": dqn_no_div_history,
                "dqn_diversity": dqn_div_history,
            },
            f,
        )

    print(f"All experiments completed successfully in {time.time() - t_start:.1f}s!")
    return {
        "data": data,
        "eval_df": eval_df,
        "lambda_df": lambda_df,
        "epsilon_df": epsilon_df,
        "sample_recs_df": sample_recs_df,
        "q_history": q_history,
        "dqn_no_div_history": dqn_no_div_history,
        "dqn_div_history": dqn_div_history,
    }


if __name__ == "__main__":
    res = run_all_experiments()
    print("\n=== MAIN EVALUATION TABLE ===")
    print(res["eval_df"].to_string(index=False))
    print("\n=== LAMBDA ABLATION TABLE ===")
    print(
        res["lambda_df"][
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
        ].to_string(index=False)
    )
