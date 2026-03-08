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
│       └── bidding_model.py    # Gradient Boosting + RL bidding
├── measurement/
│   └── attribution/
│       └── probabilistic.py   # Privacy-preserving attribution
├── frontend/               # React + TypeScript UI (WIP)
├── assistant/              # LLM assistant integration (WIP)
├── docs/                   # Architecture docs & API reference
└── docker-compose.yml      # Full stack orchestration
```

## Technical Highlights

### ML Bidding Engine (`ml/models/bidding_model.py`)
- Gradient Boosting Classifier for win probability estimation
- Feature engineering: CTR history, floor price ratio, device type, time-of-day
- Real-time inference < 10ms latency target
- Continuous learning from win/loss feedback

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
