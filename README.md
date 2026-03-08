# SMB AdTech Platform

> AI-powered, privacy-compliant digital advertising infrastructure for U.S. small and medium-sized businesses.

![Architecture](docs/screenshots/architecture.png)

## Overview

The SMB AdTech Platform democratizes access to enterprise-grade programmatic advertising for the 36.2 million small and medium-sized businesses (SMBs) in the United States. Most SMBs are locked into a handful of "walled garden" platforms (Meta, Google, TikTok) with no access to the broader open internet and in-app inventory ecosystem.

This platform provides:
- **Cross-publisher ad delivery** — reach audiences across open internet, mobile apps, and CTV
- **Privacy-first measurement** — probabilistic attribution without user-level tracking
- **AI-assisted campaign management** — LLM-powered assistant for non-expert advertisers
- **Automated ML bidding** — gradient boosting + reinforcement learning for real-time optimization

## Architecture

Five integrated components form the platform:

| Component | Tech Stack | Role |
|-----------|-----------|------|
| Self-Serve Frontend | React + TypeScript | Campaign creation & management UI |
| AI Marketing Assistant | FastAPI + LLM | Real-time campaign guidance |
| Ad Delivery API | FastAPI + Redis | Bid request routing & ad serving |
| ML Bidding Engine | Python + GBM/RL | Real-time bid optimization |
| Privacy Measurement | Python + Synthetic Data | Attribution without tracking |

## API Demo

![API Demo](docs/screenshots/api-demo.png)

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Redis (or use docker-compose)

### Installation

```bash
git clone https://github.com/kouji175/smb-adtech-platform.git
cd smb-adtech-platform
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Create a campaign
curl -X POST http://localhost:8000/api/v1/campaigns/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Summer Sale 2026",
    "advertiser_id": "adv_001",
    "budget": {"total": 5000.0, "daily": 200.0, "currency": "USD"},
    "bid_amount": 1.5,
    "start_date": "2026-06-01T00:00:00Z"
  }'

# Submit a bid request
curl -X POST http://localhost:8000/api/v1/bidding/bid \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_abc123",
    "floor_price": 0.5,
    "inventory_type": "display",
    "device_type": "mobile"
  }'
```

## Project Structure

```
smb-adtech-platform/
├── api/                    # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── models/             # Pydantic data models
│   ├── routers/            # API route handlers
│   └── services/           # Business logic layer
├── ml/
│   └── models/
│       ├── deep_ctr_model.py   # Deep Click-Through Rate prediction (PyTorch)
│       ├── gnn_ad_model.py     # Graph Neural Network for ad-user graph (PyTorch)
│       └── rl_bidding_agent.py # Reinforcement Learning bidding agent (PyTorch)
├── measurement/
│   └── attribution/
│       └── probabilistic.py   # Privacy-preserving attribution
├── frontend/               # React + TypeScript UI (WIP)
├── assistant/              # LLM assistant integration (WIP)
├── docs/                   # Architecture docs & API reference
└── docker-compose.yml      # Full stack orchestration
```

## Technical Highlights

### ML Engine (PyTorch)
The platform utilizes a multi-model ML stack for high-precision ad targeting and bidding optimization:

- **Deep CTR Prediction (`deep_ctr_model.py`)**
  - **Architecture**: DeepFM (Deep Factorization Machines)
  - **Usage**: Predicts click-through rates by modeling low-order feature interactions (FM) and high-order interactions (Deep).
  - **Key Method**: `model.predict(user_features, ad_features)` returns click probability [0, 1].

- **GNN Ad Graph (`gnn_ad_model.py`)**
  - **Architecture**: GAT (Graph Attention Network)
  - **Usage**: Models relationships between ads, users, and contexts as a heterogeneous graph to discover hidden audience segments.
  - **Key Method**: `model.get_node_embeddings(graph_data)` for vector-based similarity matching.

- **RL Bidding Agent (`rl_bidding_agent.py`)**
  - **Architecture**: PPO (Proximal Policy Optimization)
  - **Usage**: A reinforcement learning agent that manages budget pacing and dynamic bidding to maximize ROI within campaign constraints.
  - **Key Method**: `agent.select_action(state)` returns the optimal bid adjustment for the current auction.

### Privacy Measurement (`measurement/attribution/probabilistic.py`)
- Probabilistic attribution using Shapley value decomposition
- Synthetic data generation for model training without PII
- Compatible with post-ATT (Apple App Tracking Transparency) environments
- No user-level identifiers required

### Ad Delivery API (`api/routers/bidding.py`)
- OpenRTB-compatible bid request/response format
- Budget pacing with token bucket algorithm
- Fraud detection hooks
- Win notification handling

## Roadmap

- [ ] Frontend React dashboard
- [ ] LLM-powered assistant integration
- [ ] Multi-SSP supply integration
- [ ] Real-time reporting dashboard
- [ ] e-commerce conversion tracking

## License

MIT
