# Frontend Architecture & Technical Specification

This document provides the technical roadmap for the SMB AdTech Platform's frontend, ensuring a robust, scalable, and professional interface for small business owners.

## 1. Core Technology Stack

- **Framework**: React 18 with TypeScript (Vite as the build tool).
- **Styling**: Tailwind CSS for utility-first responsive design.
- **Routing**: React Router 6 for seamless navigation between the Dashboard and Campaign Workbench.
- **State Management**: React Context API (for theme/auth) and local state for form management.
- **Data Visualization**: Recharts for rendering performance metrics (ROI, Spend, CTR).
- **API Client**: Axios with interceptors for handling authentication headers.

## 2. Component Structure

The frontend is organized into a modular hierarchy to facilitate maintenance and testing:

```text
frontend/src/
├── components/          # Reusable UI elements (Buttons, Cards, Modals)
│   ├── Layout/          # Sidebar, Navbar, and Footer
│   └── Shared/          # Reusable form inputs and KPI widgets
├── pages/               # Top-level route components
│   ├── Dashboard/       # Performance overview with charts
│   └── Campaign/        # Multi-step campaign creation form
├── services/            # API abstraction layer
│   └── api.ts           # Axios instance and endpoints
├── hooks/               # Custom React hooks (e.g., useAuth, useCampaigns)
├── assets/              # Images, icons, and global styles
└── types/               # TypeScript interfaces (Campaign, BidRequest, User)
```

## 3. Frontend-Backend Interaction

The frontend communicates with the FastAPI backend defined in the `api/` directory.

### Key Data Flows:
1. **Fetching Metrics**: The Dashboard pulls aggregated data from `/api/v1/analytics/attribution` to render ROI graphs.
2. **Campaign Deployment**: The Campaign form sends a `POST` request to `/api/v1/campaigns/`.
3. **AI Assistance**: The creative generation field calls `/api/v1/ai/generate-creative` to fetch suggestions from the LLM based on user input.

## 4. Accessibility & Responsive Design

- **Mobile-First**: The UI is optimized for business owners on the go, utilizing Tailwind's responsive breakpoints.
- **Compliance**: Adherence to basic WCAG guidelines to ensure accessibility for all users.
- **Visual Feedback**: Skeleton loaders and optimistic UI updates are used to maintain a high-performance feel during ML inference wait times.
