# Machine Learning Models Overview

The SMB AdTech Platform uses a multi-layered ML stack to optimize advertising performance while maintaining privacy. All models are integrated via `api/services/bidding_service.py` with lazy loading and multi-tier graceful degradation.

## Integration Architecture (Actual Call Chain)

```
Bid Request (campaign + inventory context)
        │
        ▼
BiddingService._lazy_load()   ← loads models on first request, non-blocking
        │
        ├─ 1. predict_ctr()
        │       ├── DeepFM          → pCTR  (priority 1)
        │       ├── sklearn GBM     → pCTR  (fallback 2, if PyTorch unavailable)
        │       └── Bayesian smoothing heuristic  (fallback 3)
        │
        ├─ 2. GAT ad_match_score   = normalize(pCTR × 10)
        │       └── (full GAT inference in next iteration)
        │
        ├─ 3. get_bid_adjustment()
        │       ├── PPO Agent.act(state, deterministic=True)  → δ ∈ [-1, 1]
        │       └── 0.0 if PPO unavailable
        │
        └─ 4. final_bid = base_bid × (1 + δ)
               ecpm     = final_bid × pCTR × 1000
               model_version = "deepfm+ppo" | "heuristic"
```

**Response time target:** < 50ms end-to-end

---

## 1. GBM Baseline (`ml/models/bidding_model.py`)

- **Core Algorithm**: Gradient Boosting Classifier (scikit-learn) + calibrated probability estimation
- **Inputs**: Feature vectors — campaign params (budget, bid, age), device type, inventory source, time-of-day, historical CTR/CVR
- **Outputs**: Win probability score `(float, [0, 1])` + recommended bid price
- **Scenario**: Fast baseline for win-probability estimation; used in early campaign stages when data is sparse. Supports warm-start incremental learning.
- **Relationship**: Primary fallback when PyTorch models are unavailable; also provides benchmark metrics for DeepFM.

---

## 2. DeepFM CTR Prediction (`ml/models/deep_ctr_model.py`)

- **Core Algorithm**: Deep Factorization Machine — FM layer (second-order feature interactions) + Deep MLP; SHA-256 privacy-preserving feature hashing
- **Inputs**: High-dimensional sparse features (ad format, device, geo, bidding strategy, hour, day-of-week, campaign_id hash) — no user-level PII
- **Outputs**: Predicted click-through rate `pCTR (float, [0, 1])`
- **Scenario**: CTR prediction for display, native, and in-app ad formats. Handles cold-start via feature hashing. Outperforms GBM on high-cardinality categorical features.
- **Integration**: `BiddingService.predict_ctr()` calls `DeepFMModel.predict(X)` with hashed feature matrix

---

## 3. Graph Attention Network (`ml/models/gnn_ad_model.py`)

- **Core Algorithm**: Multi-head GAT with heterogeneous node encoding — ad slots, content categories, temporal context; edge-weighted attention on co-occurrence and behavioral similarity
- **Inputs**: Anonymous interaction graph — nodes are contextual entities (no user IDs), edges represent co-occurrence frequency or semantic similarity
- **Outputs**: Node embeddings for ad slots and content categories; ad feedback score `[0, 1]`
- **Scenario**: Models anonymous user interaction patterns without device-level identifiers. Privacy-compliant with Apple ATT and post-cookie environments.
- **Integration**: Currently initialized with random weights for demo; ad_match_score normalized from pCTR as interim until full training pipeline is set up.

---

## 4. PPO Reinforcement Learning Bidding Agent (`ml/models/rl_bidding_agent.py`)

- **Core Algorithm**: Proximal Policy Optimization — Actor-Critic with GAE (Generalized Advantage Estimation), PPO-CLIP objective, entropy bonus, orthogonal weight initialization
- **Inputs**: State vector — campaign spend_ratio, pCTR (from DeepFM), market conditions, time features
- **Outputs**: Bid adjustment factor `δ (continuous, [-1, 1])` via `act(state, deterministic=True)`
- **Scenario**: Dynamic bid optimization across campaign lifecycle. Maximizes ROI under budget constraints via reinforcement learning from win/loss signals.
- **Integration**: `BiddingService.get_bid_adjustment()` → `PPOBiddingAgent.act()` → `final_bid = base_bid × (1 + δ)`

---

## Deployment Notes

- All PyTorch models are **lazy-loaded** on first request (no startup blocking)
- Graceful degradation: `deepfm+ppo` → `gbm` → `heuristic` (Bayesian smoothing)
- `model_version` field in `BidDecision` response indicates which tier was used
- No persistent user identifiers flow into any model layer
