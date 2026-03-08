# Machine Learning Models Overview

The SMB AdTech Platform uses a multi-layered ML stack to optimize advertising performance while maintaining privacy.

## 1. GBM Baseline (`ml/models/bidding_model.py`)

- **Core Algorithm**: Gradient Boosting Classifier (scikit-learn) + Calibrated probability estimation
- **Inputs**: Flattened feature vectors — campaign params (budget, bid, age), device type, inventory source, time-of-day, historical CTR/CVR
- **Outputs**: Win probability score `(float, [0, 1])` + recommended bid price
- **Scenario**: Fast, reliable baseline for win-probability estimation. Deployed in early campaign stages when data is sparse; supports warm-start incremental learning.
- **Relationship**: Provides benchmark metrics for DeepFM and PPO agent; also serves as fallback when deep models are unavailable.

---

## 2. DeepFM CTR Prediction (`ml/models/deep_ctr_model.py`)

- **Core Algorithm**: Deep Factorization Machine — FM layer (second-order feature interactions) + Deep MLP layers, SHA-256 privacy-preserving feature hashing
- **Inputs**: High-dimensional sparse features (ad ID, publisher category, device, geo) encoded via hash trick — no user-level PII required
- **Outputs**: Predicted click-through rate `pCTR (float, [0, 1])`
- **Scenario**: CTR prediction for display, native, and in-app ad formats across open-internet inventory. Handles cold-start via feature hashing.
- **Relationship**: pCTR feeds into the PPO bidding agent's state vector; outperforms GBM baseline on high-cardinality categorical features.

---

## 3. Graph Attention Network (`ml/models/gnn_ad_model.py`)

- **Core Algorithm**: Multi-head Graph Attention Network (GAT) with heterogeneous node encoding — ad slots, content categories, temporal context nodes; edge-weighted attention on co-occurrence and behavioral similarity
- **Inputs**: Anonymous interaction graph — nodes are contextual entities (no user IDs), edges represent co-occurrence frequency or semantic similarity
- **Outputs**: Node embeddings for ad slots and content categories; ad feedback score prediction
- **Scenario**: Models anonymous user interaction patterns without device-level identifiers. Privacy-compliant with Apple ATT and post-cookie environments.
- **Relationship**: GAT embeddings augment DeepFM input features; also provides context signals to the PPO agent's observation space.

---

## 4. PPO Reinforcement Learning Bidding Agent (`ml/models/rl_bidding_agent.py`)

- **Core Algorithm**: Proximal Policy Optimization (PPO) with Actor-Critic architecture, Generalized Advantage Estimation (GAE), clipped surrogate objective
- **Inputs**: State vector — campaign budget utilization, recent win rate, market floor price distribution, time features, pCTR from DeepFM, GAT context embeddings
- **Outputs**: Bid adjustment factor `(continuous action, [0.5×, 2.0×] of base bid)`; value estimate for critic
- **Scenario**: Dynamic bid optimization across campaign lifecycle. Maximizes ROI under budget constraints via reinforcement learning from win/loss feedback signals.
- **Relationship**: Integrates all upstream model outputs (GBM win probability, DeepFM pCTR, GAT embeddings) into a unified bidding decision. Top of the ML stack.

---

## Model Integration Architecture

```
Raw Bid Request
      │
      ├─► GBM Baseline ──────────────────► win_prob
      │
      ├─► DeepFM CTR Model ──────────────► pCTR
      │
      ├─► Graph Attention Network ────────► context_embedding
      │
      └─► PPO Bidding Agent ◄─────────────┘
              │  (state = win_prob + pCTR + context_embedding + budget_state)
              │
              └─► final_bid_price ──► Ad Exchange
```

All models operate without user-level persistent identifiers, ensuring compliance with ATT, GDPR, and emerging U.S. state privacy regulations.
