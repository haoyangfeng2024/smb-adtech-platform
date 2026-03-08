# SMB AdTech Platform

## Overview
The SMB AdTech Platform is an AI-driven advertising infrastructure designed specifically for Small and Medium-sized Businesses (SMBs) in the United States. Our mission is to provide high-performance, privacy-compliant advertising tools that were previously only accessible to large enterprises. 

By leveraging cutting-edge machine learning and a privacy-first architecture, we help SMBs navigate the complex digital advertising landscape while ensuring strict adherence to privacy regulations and maximizing ROI.

## Core Features
- **AI-Powered Ad Creation**: Simplifies the campaign setup process for non-experts.
- **Privacy-First Measurement**: Advanced attribution that respects user privacy.
- **Real-time ML Bidding**: Optimizes ad spend across multiple platforms.
- **SMB-Centric Dashboard**: Intuitive interface for managing complex ad operations.

## Architecture
The platform is composed of five primary components:
1. **Self-Service Frontend**: A React-based dashboard for business owners.
2. **AI Marketing Assistant**: LLM-driven interface for campaign strategy and creative generation.
3. **Unified Ad Delivery API**: High-throughput gateway for cross-platform ad execution.
4. **ML Bidding & Optimization Engine**: Real-time models for dynamic bid adjustment.
5. **Privacy Measurement Engine**: Secure analytics and attribution tracking.

## Quick Start
### Prerequisites
- Node.js v20+
- Docker & Docker Compose
- API Keys for supported Ad Networks (Google, Meta, etc.)

### Installation
```bash
git clone https://github.com/your-repo/smb-adtech-platform.git
cd smb-adtech-platform
cp .env.example .env
npm install
```

### Running the Platform
```bash
docker-compose up -d
npm run dev
```

## API Demo

### Start the server
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### Create a Campaign
```bash
curl -X POST http://localhost:8000/api/v1/campaigns/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Summer Sale 2026",
    "advertiser_id": "adv_001",
    "status": "active",
    "budget": {"total": 5000.00, "daily": 200.00},
    "bid_amount": 1.50,
    "bidding_strategy": "cpc",
    "start_date": "2026-03-08T00:00:00Z",
    "targeting": {
      "geo": {"countries": ["US"]},
      "device": {"devices": ["mobile", "desktop"]}
    }
  }'
```

### Submit a Bid Request (RTB)
```bash
curl -X POST http://localhost:8000/api/v1/bidding/bid \
  -H "Content-Type: application/json" \
  -d '{
    "imp_id": "imp_abc123",
    "site_id": "site_techblog",
    "floor_price": 0.5,
    "device_type": "mobile",
    "geo": {"countries": ["US"]}
  }'
# Response: bid_price, ad_markup, predicted_ctr, decision_ms
```

### Health / Metrics
```bash
curl http://localhost:8000/health   # {"status":"ok"}
curl http://localhost:8000/metrics  # Prometheus metrics
```

## Mission
This project aims to democratize high-end advertising technology for the American SMB sector. By focusing on technological innovation and privacy-first design, we ensure that digital small businesses have the tools to compete effectively while leading in privacy standards and data ethics.
