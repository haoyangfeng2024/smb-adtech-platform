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

### Update Bid Strategy
- **PATCH** `/v1/bidding/strategy`
- **Params**: `campaign_id`, `bid_cap`, `target_roas`

---

## 4. Privacy-Safe Analytics (`/v1/analytics`)

### Get Attribution Report
- **GET** `/v1/analytics/attribution`
- **Query**: `start_date`, `end_date`, `granularity`
- **Note**: Data is returned in aggregate form to comply with privacy standards.
