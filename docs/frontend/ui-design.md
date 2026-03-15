# UI Interaction Design Specification

This document outlines the user interface and interaction design for the SMB AdTech Platform, focusing on accessibility for non-expert advertisers and the visual representation of complex AI/ML logic.

## 1. Design Philosophy

- **Simplicity Over Complexity**: SMB owners are not data scientists. The UI must hide the underlying complexity (GNN/PPO) while surfacing actionable insights.
- **Trust via Transparency**: Since the platform manages advertising budgets, the UI must clearly communicate the state of "Privacy-Safe" processing and AI decision-making.
- **Action-Oriented Flow**: Every screen should lead the user toward a specific business goal (e.g., Increase ROI, Launch Campaign).

## 2. Core Modules & Interaction Flows

### A. Campaign Creation Workbench (`CampaignManager`)
- **Step 1: The "Seed" Input**: User enters a simple business description and target goal (e.g., "Sell more artisanal coffee in Austin").
- **Step 2: AI Creative Generation**: Integrating with the `AI Assistant`, the UI displays generated ad copies and suggested audience segments derived from GNN embeddings.
- **Step 3: Quick Launch**: A "One-Click" deployment toggle that interfaces with the `Ad Delivery API`.

### B. ROI & Performance Dashboard (`AnalyticsDashboard`)
- **Visual KPIs**: Large, clear cards for "Total Spend," "Clicks," and "Current ROI."
- **Conversion Funnel**: A visual funnel showing the transition from Impressions -> Clicks -> Conversions.
- **Predictive Trends**: Using the ML Bidding Engine data, the dashboard shows a "Projected 7-Day Performance" graph (using Recharts).

### C. Privacy & Security Indicator
- A persistent "Shield" icon in the navigation bar that, when clicked, shows a summary of the anonymization process (e.g., "Active: SHA-256 ID Anonymization," "No PII stored").

## 3. Data Visualization & Reporting (Sprint 3 Phase 2)

To empower SMBs with actionable data, the dashboard integrates advanced Recharts-based visualizations.

### 📊 ROI Trend Analysis (Line Chart)
- **Data Source**: Aggregated historical performance from `/api/v1/analytics/attribution`.
- **Visualization**: A 7-day rolling line chart comparing "Actual ROI" vs. "ML-Projected ROI".
- **Key Metric**: Surfacing the efficiency of the PPO Bidding Agent in maximizing return on ad spend.

### 🌪️ Conversion Funnel (Funnel Chart)
- **Visual Path**: Ad Impressions → Clicks → Conversions (Leads/Sales).
- **Insight**: Highlights where in the customer journey users are dropping off, allowing SMBs to optimize their creative assets or landing pages.

### ⏱️ Budget Utilization (Gauge/Donut Chart)
- **Real-time Monitoring**: A circular progress bar showing "Spent Amount" relative to the "Daily Budget Cap".
- **AI Pacing**: A status label indicating if the **PPO Agent** is currently "Pacing Aggressively" (high opportunity) or "Pacing Conservatively" (low liquidity).

### 📈 PPO Bidding Dynamics (Live Curve)
- **ML Transparency & Accountability**: A specialized chart showing real-time bid adjustments (δ ∈ [-1, 1]) executed by the PPO RL Agent. 
- **Strategic Insight**: This visualization serves as a primary tool for "Explainable AI" (XAI). It demonstrates to the SMB user exactly how the AI is dynamically adjusting strategy based on auction pressure and budget pacing, providing a level of transparency and accountability that exceeds industry standard "black-box" bidding algorithms.

## 4. Technical Specification (Frontend MVP)

- **Framework**: React 18 (Vite-based).
- **Styling**: Tailwind CSS for responsive and modern aesthetics.
- **Visualization Library**: **Recharts** for performance metrics.
- **API Integration**: Axios for communication with the FastAPI backend endpoints defined in `docs/api/endpoints.md`.
