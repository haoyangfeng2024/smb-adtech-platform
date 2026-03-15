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

## 3. Technical Specification (Frontend MVP)

- **Framework**: React 18 (Vite-based).
- **Styling**: Tailwind CSS for responsive and modern aesthetics.
- **Iconography**: Heroicons for standard UI actions.
- **State Management**: React Context or Zustand for campaign configuration flow.
- **API Integration**: Axios for communication with the FastAPI backend endpoints defined in `docs/api/endpoints.md`.

---
*Note: This design ensures that the high-tech backend (DeepFM, PPO) is translated into a competitive advantage for U.S. small businesses through a professional and intuitive interface.*
