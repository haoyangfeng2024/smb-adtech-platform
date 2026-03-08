# API Endpoints Skeleton

This document outlines the core API structure for the Ad Delivery and Measurement services.

## Authentication
All requests must include a Bearer Token in the Authorization header.
`Authorization: Bearer <YOUR_API_KEY>`

---

## 1. Campaign Management (`/v1/campaigns`)

### Create Campaign
- **POST** `/v1/campaigns`
- **Request Body**: `CampaignObject`
- **Response**: `201 Created`

### List Campaigns
- **GET** `/v1/campaigns`
- **Response**: `Array<CampaignObject>`

---

## 2. AI Assistant Interface (`/v1/ai`)

### Generate Creative
- **POST** `/v1/ai/generate-creative`
- **Params**: `business_description`, `target_audience`
- **Response**: `Array<AdCopySuggestion>`

---

## 3. Real-time Bidding Control (`/v1/bidding`)

### Submit a Bid Request
- **POST** `/v1/bidding/bid`
- **Request Body**:
```json
{
  "request_id": "req_abc123",
  "floor_price": 0.5,
  "inventory_type": "display",
  "device_type": "mobile",
  "os": "ios",
  "ad_format": "banner",
  "geo": {"countries": ["US"]}
}
```
- **Response**: `200 OK`
- **Response Body**:
```json
{
  "campaign_id": "camp_789",
  "win_prob": 0.425,
  "predicted_ctr": 0.084,
  "context_embedding": [0.12, -0.05, 0.33, 0.07, 0.21, -0.18, 0.09, 0.44],
  "ad_match_score": 0.509,
  "base_bid": 2.5,
  "bid_adjustment": -0.003,
  "final_bid": 2.50,
  "ecpm": 210.6,
  "model_version": "deepfm+gat+ppo",
  "decision_ms": 12.5,
  "fallbacks": {
    "gbm": true,
    "deepfm": false,
    "gat": false,
    "ppo": false
  }
}
```

**Response field reference:**

| Field | Type | Description |
|-------|------|-------------|
| `win_prob` | float [0,1] | GBM predicted auction win probability |
| `predicted_ctr` | float [0,1] | DeepFM predicted click-through rate |
| `context_embedding` | float[8] or null | GAT ad-slot embedding vector (truncated to 8 dims) |
| `ad_match_score` | float [0,1] | GAT ad-content match score |
| `base_bid` | float | Campaign configured bid amount |
| `bid_adjustment` | float [-1,1] | PPO Agent output δ |
| `final_bid` | float ≥ 0.01 | `base_bid × (1 + δ) × match_weight`, min 0.01 |
| `ecpm` | float | `final_bid × pCTR × 1000` |
| `model_version` | string | Active model backends (see below) |
| `decision_ms` | float | End-to-end ML inference latency (ms) |
| `fallbacks` | object | Per-layer fallback flags (`true` = used fallback) |

**`model_version` values:**

| Value | Active layers |
|-------|--------------|
| `"deepfm+gat+ppo"` | Full 4-layer ML stack |
| `"deepfm+gat"` | PPO unavailable, δ=0 |
| `"gbm"` | PyTorch unavailable, GBM baseline only |
| `"heuristic"` | All models unavailable, Bayesian smoothing |

**`fallbacks` keys:** `gbm` / `deepfm` / `gat` / `ppo` — each is `true` when that layer fell back to its degraded mode.

**Degradation tiers (automatic, transparent to caller):**
1. **DeepFM + GAT + PPO** — primary path (PyTorch + pretrained weights available)
2. **sklearn GBM** — fallback if PyTorch import fails
3. **Bayesian heuristic** — final fallback: `win_prob = (clicks + 0.5) / (impressions + 100)`

**`bid_adjustment`**: PPO Agent output δ ∈ [-1, 1]; `final_bid = base_bid × (1 + δ) × match_weight`, minimum `0.01`

### Update Bid Strategy
- **PATCH** `/v1/bidding/strategy`
- **Params**: `campaign_id`, `bid_cap`, `target_roas`

---

## 4. Privacy-Safe Analytics (`/v1/analytics`)

### Get Attribution Report
- **GET** `/v1/analytics/attribution`
- **Query**: `start_date`, `end_date`, `granularity`
- **Note**: Data is returned in aggregate form to comply with privacy standards.
