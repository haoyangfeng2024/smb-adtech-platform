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
  "base_bid": 1.0,
  "adjustment": 0.25,
  "final_bid": 1.25,
  "predicted_ctr": 0.015,
  "ad_match_score": 0.87,
  "ecpm": 18.75,
  "model_version": "deepfm+ppo",
  "decision_ms": 12.5
}
```

**`model_version` values:**
| Value | Meaning |
|-------|---------|
| `"deepfm+ppo"` | Full ML stack: DeepFM pCTR + PPO bid adjustment |
| `"heuristic"` | PyTorch unavailable; using Bayesian smoothing fallback |

**Degradation tiers (automatic, transparent to caller):**
1. **DeepFM + PPO** — primary path (PyTorch available)
2. **sklearn GBM** — fallback if PyTorch import fails
3. **Bayesian heuristic** — final fallback (`(clicks+0.1)/(impressions+100)`)

**`adjustment`**: PPO Agent output δ ∈ [-1, 1]; `final_bid = base_bid × (1 + δ)`, minimum `0.01`

### Update Bid Strategy
- **PATCH** `/v1/bidding/strategy`
- **Params**: `campaign_id`, `bid_cap`, `target_roas`

---

## 4. Privacy-Safe Analytics (`/v1/analytics`)

### Get Attribution Report
- **GET** `/v1/analytics/attribution`
- **Query**: `start_date`, `end_date`, `granularity`
- **Note**: Data is returned in aggregate form to comply with privacy standards.
