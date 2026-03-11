# Machine Learning Training Pipeline

This document outlines the end-to-end machine learning training pipeline for the SMB AdTech Platform, emphasizing privacy-first data handling and the transition from academic research (GNN, PPO) to industrial production.

## 1. Synthetic Data Generation (Privacy-First)

A core pillar of our platform is the strict adherence to privacy regulations (CCPA/GDPR/ATT). To facilitate robust model training without compromising user-level privacy, we utilize a specialized **Synthetic Data Generator**.

### Core Principles:
- **Zero PII (Personally Identifiable Information)**: The generator creates high-fidelity interaction logs (clicks, impressions, bids) based on statistical distributions derived from historical data, ensuring no real user identity is ever part of the training set.
- **Differential Privacy**: Statistical noise is injected into the generative process to prevent any potential re-identification of original data patterns.
- **Contextual Realism**: The data simulates realistic U.S. SMB advertising scenarios, including time-of-day effects, seasonal surges, and diverse device distributions across all 50 states.

## 2. The Training Workflow

The pipeline is designed for continuous iteration and high-frequency model updates.

### Step A: Data Preprocessing & Cleaning
- Normalizing multi-currency bid values to USD.
- Handling missing fields via Bayesian imputation (as defined in our BiddingService).
- Aggregating interaction logs into graph structures for the GNN encoder.

### Step B: Feature Engineering
- **Categorical Encoding**: High-cardinality features (Ad IDs, Publisher IDs) are hashed using the DeepFM hasher to maintain a constant-size input vector.
- **Temporal Features**: Engineering features like "Time to Budget Exhaustion" and "Historical CTR Decay" to provide the PPO agent with critical state information.

### Step C: Model Training
- **DeepFM**: Training the deep and wide components simultaneously to capture complex user-ad interactions.
- **GNN (GAT)**: Performing attention-based message passing over the heterogeneous ad-context graph.
- **PPO RL Agent**: Utilizing a simulated auction environment for the Reinforcement Learning agent to learn optimal bid adjustment policies.

### Step D: Offline Evaluation
Before any model is deployed (moved to `ml/artifacts/`), it must pass a rigorous offline evaluation check:
- **Log Loss & AUC**: For the DeepFM CTR prediction.
- **Reward Convergence**: For the PPO Bidding Agent.
- **Backtesting**: Running the new model against historical synthetic logs to ensure it outperforms the current GBM baseline.

## 3. Research & Visualization (Jupyter Notebooks)

To bridge the gap between technical implementation and strategic oversight, we maintain a suite of **Experimental Notebooks** (`notebooks/` directory).

- **`01_model_training_experiment.ipynb`**: Visualizes the loss curves and provides a playground for hyperparameter tuning.
- **`02_bidding_simulation_analysis.ipynb`**: Analyzes the PPO agent's behavior under different market stress scenarios (e.g., sudden high-competition events).
- **`03_graph_embedding_visualization.ipynb`**: Uses t-SNE to project GNN-learned ad/user embeddings into a 2D space, demonstrating the model's ability to cluster similar high-intent segments without PII.

---
*Note: This pipeline ensures that our technological innovations in Graph Neural Networks and Reinforcement Learning are deployed in a stable, compliant, and measurable manner.*
