# Machine Learning Training Pipeline

This document outlines the end-to-end machine learning training pipeline for the SMB AdTech Platform, emphasizing privacy-first data handling and the transition from academic research (GNN, PPO) to industrial production.

## 1. Synthetic Data Generation (Privacy-First)

A core pillar of our platform is the strict adherence to privacy regulations (CCPA/GDPR/ATT). To facilitate robust model training without compromising user-level privacy, we utilize a specialized **Synthetic Data Generator**.

### Core Principles:
- **Zero PII (Personally Identifiable Information)**: The generator (`ml/data/synthetic_generator.py`) creates high-fidelity interaction logs based on statistical distributions. 
- **Irreversible Anonymization**: All generated identifiers are passed through a **SHA-256 hashing** process and truncated, ensuring that synthetic IDs cannot be mapped back to any real-world entities.
- **Contextual Realism**: The data simulates realistic U.S. SMB advertising scenarios, including time-of-day effects and device-specific modifiers (e.g., higher CTR for mobile devices).

## 2. The Training Workflow (`scripts/train.py`)

The pipeline is managed via a centralized training script that supports multiple model architectures and persistent training sessions.

### Execution Command
```bash
python scripts/train.py --model [deepfm|ppo|gbm] [--resume]
```

### Key Features:
- **Model Selection**: Supports switching between DeepFM for CTR, PPO for bidding, and GBM for baselines.
- **Checkpointing & Persistence**: Using the `--resume` flag, the script automatically detects existing model artifacts in `ml/artifacts/` and continues training from the last saved state, preventing redundant computation.
- **Graceful Error Handling**: The script includes isolation for each training phase, ensuring that data generation failures do not corrupt existing model states.

### Step A: Data Preprocessing & Cleaning
- Normalizing multi-currency bid values to USD.
- Handling missing fields via Bayesian imputation (as defined in our BiddingService).

### Step B: Feature Engineering
- **Categorical Encoding**: High-cardinality features are hashed using the DeepFM hasher to maintain a constant-size input vector.
- **Temporal Features**: Engineering features like "Time of Day" and "Day of Week" into cyclical representations for neural network compatibility.

### Step C: Model Training
- **DeepFM**: Optimizing Log Loss and AUC for high-precision CTR prediction.
- **PPO RL Agent**: Training the bid adjustment policy within a simulated auction environment to maximize long-term campaign ROI.

### Step D: Offline Evaluation
Before any model is finalized, it undergoes:
- **AUC/LogLoss Metrics**: Validating predictive accuracy.
- **Reward Curve Analysis**: Ensuring PPO policy convergence.
- **Artifact Export**: Models are serialized and saved to `ml/artifacts/` for live inference by the `BiddingService`.

## 3. Research & Visualization (Jupyter Notebooks)

To bridge the gap between technical implementation and strategic oversight, we maintain a suite of **Experimental Notebooks** in the `notebooks/` directory.

- **`training_demo.ipynb`**: A comprehensive walkthrough of the synthetic data generation process, model training logs, and loss curve visualizations. It serves as the primary tool for researchers to validate hypothesis testing before scaling to full training runs.

---
*Note: This pipeline ensures that our technological innovations in Graph Neural Networks and Reinforcement Learning are deployed in a stable, compliant, and measurable manner.*
