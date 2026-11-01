# MovieMind — Reinforcement Learning-Based Adaptive Movie Recommendation System

**MovieMind** is an end-to-end Reinforcement Learning (RL) recommendation system built on the official **MovieLens 100K** dataset. It formulates sequential movie recommendation as a **Markov Decision Process (MDP)** that learns user preferences through simulated feedback while explicitly balancing **recommendation relevance** with **sequential genre diversity**.

---

## 1. Project Folder Structure

```text
moviemind/
├── MovieMind.ipynb                  # Self-contained, executed Google Colab notebook (14 sections)
├── app.py                           # Interactive dark-themed Streamlit movie streaming studio
├── requirements.txt                 # Python package dependencies
├── README.md                        # Comprehensive project documentation & experimental report
├── src/                             # Shared modular Python package
│   ├── __init__.py
│   ├── data_preprocessing.py        # MovieLens 100K downloader, cleaner, and anti-leakage 80:20 splitter
│   ├── environment.py               # Custom Gymnasium environment (MovieRecEnv) & Markov state logic
│   ├── agents.py                    # Tabular Q-Learning & Deep Q-Network (DQN) implementations
│   ├── baselines.py                 # Random, Popularity-Based, and Content-Based Filtering baselines
│   ├── evaluation.py                # Held-out Precision@K, Recall@K, NDCG@K, ILD, Coverage & Reward suite
│   ├── visualizations.py            # Matplotlib/Seaborn visualization generators
│   └── run_experiments.py           # Reproducible CLI runner for all 7 experimental studies
├── data/
│   └── ml-100k/                     # Official GroupLens MovieLens 100K dataset files
├── models/
│   ├── dqn_diversity.pt             # Trained Diversity-Aware DQN weights (λ = 0.25)
│   ├── dqn_no_diversity.pt          # Trained Baseline DQN weights (λ = 0.0)
│   └── q_learning_agent.npz         # Trained Tabular Q-Learning tables
├── results/
│   ├── model_comparison.csv         # Held-out test evaluation metrics across all 6 models
│   ├── lambda_ablation.csv          # Sensitivity study across diversity weights λ ∈ {0.0, 0.1, 0.25, 0.5, 0.8}
│   ├── epsilon_ablation.csv         # Exploration vs. exploitation (ε-decay) study results
│   └── sample_user_recommendations.csv
└── figures/
    ├── 01_eda_overview.png
    ├── 02_training_convergence.png
    ├── 03_model_comparison.png
    ├── 04_diversity_weight_ablation.png
    ├── 05_genre_distribution.png
    └── 06_epsilon_exploration.png
```

---

## 2. Setup and Execution Instructions

### A. Local Environment Setup
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install required dependencies
pip install -r requirements.txt
```

### B. Running the Google Colab Notebook (`MovieMind.ipynb`)
- **On Google Colab**: Upload `MovieMind.ipynb` directly to Google Colab and select **Runtime → Run all**. The notebook is **100% self-contained** — it automatically downloads the MovieLens 100K dataset from GroupLens and executes all 14 sections without requiring external files.
- **Locally via Jupyter**:
  ```bash
  jupyter notebook MovieMind.ipynb
  ```

### C. Running the Interactive Streamlit Web Application (`app.py`)
```bash
streamlit run app.py
```
This launches the dark-themed **MovieMind Streaming Studio** in your browser (`http://localhost:8501`), featuring:
- Real-time algorithm switching (`DQN (Diversity-Aware)`, `DQN (No Diversity)`, `Tabular Q-Learning`, `Content-Based Filtering`, `Popularity-Based`, `Random Baseline`).
- Live **Diversity Weight ($\lambda$)** slider and **User Genre Profile** customization.
- Interactive **👍 Like**, **👎 Dislike**, and **⭐ 1–5 Star Rating** buttons on every movie card that immediately update the RL Markov state vector and re-rank eligible unseen candidates.
- **Genre Explorer** and **RL Analytics & Benchmarks** tabs.

### D. Re-running the Full Experimental Suite via CLI
```bash
python -m src.run_experiments
```

---

## 3. Dataset Attribution & Zero-Leakage Preprocessing

### Dataset Attribution
We use the **MovieLens 100K** dataset collected by the **GroupLens Research Project** at the University of Minnesota:
- **Source**: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)
- **Scale**: 100,000 explicit ratings (1–5 stars) from **943 users** across **1,682 movies** and **18 named genres**.
- **Reference**: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.

### Strict Anti-Leakage Protocol
1. **Per-User 80:20 Split**: Each user's interactions are reproducibly partitioned (`seed=42`) into **80% training (`train_df`, 80,000 ratings)** and **20% held-out testing (`test_df`, 20,000 ratings)**.
2. **Disjoint Verification**: We explicitly assert $\mathcal{I}_{\text{train}}(u) \cap \mathcal{I}_{\text{test}}(u) = \emptyset$ for every user $u$.
3. **Training-Only Feature Construction**: All user genre preference vectors $\mathbf{p}_u$, Bayesian item quality scores, log-popularity statistics, and rank-16 SVD collaborative factors are computed **exclusively from `train_df`**. Held-out `test_df` interactions are never accessed during training or environment reward generation.

---

## 4. Reinforcement Learning Environment & Markov State Formulation

We implement `MovieRecEnv(gym.Env)` following the standard Gymnasium API (`reset`, `step`, `observation_space`, `action_space`).

### Markov State Representation ($s_t \in [-1, 1]^{57}$)
At step $t$ of an episode for user $u$, the continuous state vector $s_t$ comprises:
1. **Dynamic User Genre Preferences ($\mathbf{p}_{u,t} \in [0, 1]^{18}$)**: Normalized user affinity across the 18 genres, updated online as the user reacts to recommendations during the episode.
2. **Historical & Session Interaction Features ($\mathbf{h}_{u,t} \in [-1, 1]^{20}$)**: Normalized training mean rating, rating standard deviation, log interaction activity, running session average preference satisfaction, and the user's 16-dimensional training SVD embedding.
3. **Session Genre Exposure Vector ($\mathbf{e}_t \in [0, 1]^{18}$)**: Fraction of movies recommended so far in the current episode belonging to each genre.
4. **Episode Step Progress ($t / K \in [0, 1]^1$)**: Normalized step counter for horizon $K=10$.

**Why $s_t$ is Markovian**: The transition distribution $P(s_{t+1} \mid s_t, a_t)$ and immediate reward $R(s_t, a_t)$ depend solely on the current preference vector $\mathbf{p}_{u,t}$, historical context $\mathbf{h}_{u,t}$, and current session genre exposure $\mathbf{e}_t$ stored directly inside $s_t$.

### Action Space & Candidate Masking
- **Action ($a_t$)**: Selecting a movie index from the eligible unseen candidate pool $\mathcal{C}_t(u) = \mathcal{C}_{\text{pool}} \setminus (\mathcal{I}_{\text{train}}(u) \cup \mathcal{I}_{\text{rec}, t}(u))$.
- Already-watched training movies and movies already recommended in the current session are strictly masked out via `env.get_action_mask()`.

### Diversity-Aware Reward Function
$$\text{Total Reward } R_t(s_t, a_t) = R_{\text{pref}}(u, a_t) + \lambda \cdot R_{\text{div}}(\mathbf{e}_t, a_t)$$
- **Preference Reward ($R_{\text{pref}} \in [0, 1]$)**: Combines training-derived user-item preference affinity with dynamic session genre alignment $\cos(\mathbf{p}_{u,t}, \mathbf{g}(a_t))$.
- **Diversity Bonus ($R_{\text{div}} \in [0, 1]$)**: Combines session genre novelty $1 - 1.8 \cdot \text{mean}_{g \in a_t}(e_{t,g})$ and intra-list cosine dissimilarity $1 - \frac{1}{t}\sum_{\tau < t}\cos(\mathbf{g}(a_t), \mathbf{g}(a_\tau))$, softly gated by user genre relevance ($0.35 + 0.65 \cos(\mathbf{p}_{u,t}, \mathbf{g}(a_t))$) so the agent is rewarded for exploring **diverse genres the user enjoys** rather than recommending irrelevant noise.

---

## 5. RL Algorithms Implemented

### A. Tabular Q-Learning (`TabularQLearningAgent`)
- **Discretized State Space**: Maps $s_t$ into $684$ discrete states encoding $(\text{dominant\_user\_genre} \in [0..17], \text{most\_exposed\_session\_genre} \in [0..18], \text{is\_saturated} \in \{0, 1\})$.
- **Q-Tables & Bellman Update**: Maintains an explicit state-action table $Q(s_{\text{disc}}, a)$ and a state-genre table $Q_{\text{genre}}(s_{\text{disc}}, g)$ updated via:
  $$Q(s_{\text{disc}}, a) \leftarrow Q(s_{\text{disc}}, a) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}_{\text{eligible}}(s'_{t+1})} Q(s'_{\text{disc}}, a') - Q(s_{\text{disc}}, a) \right]$$

### B. Deep Q-Network (`DQNAgent` & `DQNNetwork`)
- **Neural Architecture**: A PyTorch Dueling Feature-Interactive Q-Network that encodes the 57-dimensional continuous state $s_t$ through a 2-layer LayerNorm-ReLU MLP and combines a state-value stream $V(s_t)$ with state-conditioned softplus gates over movie features (collaborative SVD match, genre affinity, session genre novelty penalty, popularity, and Bayesian rating) plus a per-movie residual advantage stream $A(s_t, a)$.
- **Stabilization Mechanisms**: Experience Replay Buffer (`capacity=15,000`, `batch_size=64`), periodic Target Network synchronization (`target_update_freq=80`), Double-DQN target evaluation, Smooth L1 (Huber) loss, and gradient norm clipping.

---

## 6. Experimental Results

All models were evaluated on a consistent cohort of **250 held-out users** (`episode_length = 10`, `candidate_pool_size = 300`, `seed = 42`).

### Table 1: Main Model Comparison (Held-Out Test Metrics & Simulated Rewards)

| Model | Precision@5 | Precision@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Intra-List Diversity | Genre Diversity | Catalog Coverage | Avg Preference Reward | Avg Cumulative Reward |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.0432 | 0.0424 | 0.0151 | 0.0282 | 0.0451 | 0.0468 | 0.7635 | 0.5424 | 1.0000 | 4.2567 | 5.3312 |
| **Popularity** | 0.1752 | 0.1524 | 0.0737 | 0.1277 | 0.2024 | 0.1955 | 0.6837 | 0.5698 | 0.1433 | 5.4479 | 6.3928 |
| **Content-Based** | 0.0968 | 0.0832 | 0.0398 | 0.0654 | 0.1033 | 0.0990 | 0.3765 | 0.4384 | 0.4667 | 5.9588 | 6.5053 |
| **Tabular Q-Learning** | 0.1144 | 0.1056 | 0.0517 | 0.0877 | 0.1214 | 0.1232 | 0.5090 | 0.4320 | 0.8067 | 5.6744 | 6.4270 |
| **DQN (No Diversity, $\lambda=0.0$)** | 0.3280 | 0.2760 | 0.1513 | 0.2479 | 0.3545 | 0.3456 | 0.5423 | 0.4933 | 0.7367 | **7.1544** | 7.9011 |
| **DQN (Diversity-Aware, $\lambda=0.25$)** | **0.3392** | **0.2812** | **0.1613** | **0.2496** | **0.3697** | **0.3560** | **0.6240** | **0.5622** | 0.7033 | 7.0936 | **8.0589** |

### Table 2: Diversity Weight ($\lambda$) Sensitivity & Ablation Study (DQN)

| Diversity Weight ($\lambda$) | Precision@5 | Precision@10 | Recall@10 | NDCG@10 | Intra-List Diversity | Genre Diversity | Catalog Coverage | Avg Preference Reward | Avg Cumulative Reward |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (Zero Diversity)** | 0.3280 | 0.2760 | 0.2479 | 0.3456 | 0.5423 | 0.4933 | 0.7367 | **7.1544** | 7.1544 |
| **0.10** | **0.3448** | 0.2696 | 0.2389 | 0.3484 | 0.5862 | 0.5196 | 0.7100 | 7.1110 | 7.4460 |
| **0.25 (Balanced)** | 0.3392 | **0.2812** | **0.2496** | **0.3560** | 0.6240 | 0.5622 | 0.7033 | 7.0936 | 8.0589 |
| **0.50** | 0.3312 | 0.2688 | 0.2352 | 0.3368 | 0.6879 | 0.6144 | 0.7267 | 6.9553 | 9.2620 |
| **0.80 (High Diversity)** | 0.3136 | 0.2564 | 0.2286 | 0.3292 | **0.7315** | **0.6562** | 0.6800 | 6.8243 | **10.8052** |

### Key Takeaways
- **DQN vs. Tabular Q-Learning**: DQN improves held-out `NDCG@10` from `0.1232` to `0.3560` (+189%) and `Precision@5` from `0.1144` to `0.3392` (+197%) by preserving the continuous 57-dimensional state representation and generalizing across movie feature embeddings.
- **Overcoming Filter Bubbles**: Greedy Content-Based Filtering collapses into single-genre recommendations (`Intra-List Diversity = 0.3765`). Diversity-Aware DQN ($\lambda=0.25$) increases Intra-List Diversity to `0.6240` (+65.7% over Content-Based and +15.1% over $\lambda=0.0$ DQN) while simultaneously achieving the highest held-out `Precision@10` (`0.2812`) and `NDCG@10` (`0.3560`).

---

## 7. Limitations

1. **Offline-to-Online Simulation Gap**: Because MovieLens 100K is a fixed historical dataset, interactive rewards during RL training are generated by a preference model calibrated strictly on the 80% training split. Real-world deployment requires counterfactual off-policy evaluation (e.g., IPS / Doubly Robust estimators) or live A/B testing.
2. **Action Space Scalability**: Scoring all candidate items via a Q-network scales linearly with catalog size $O(|\mathcal{I}|)$. For industrial catalogs with millions of titles, a two-stage architecture (approximate nearest-neighbor candidate generation followed by DQN top-$K$ re-ranking) is recommended.
3. **Cold-Start Onboarding**: Users without historical ratings require explicit initial preference elicitation, which is supported interactively in the Streamlit web application.
