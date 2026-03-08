# System Architecture Overview

This document provides a detailed overview of the five core components of the SMB AdTech Platform.

## 1. Self-Service Frontend (The Dashboard)
- **Tech Stack**: React, TailwindCSS, Vite.
- **Function**: Provides an intuitive interface for business owners to manage campaigns, view analytics, and configure account settings. It focuses on accessibility and data visualization.

## 2. AI Marketing Assistant
- **Tech Stack**: LLM (OpenAI/Gemini), LangChain, Vector Database.
- **Function**: Acts as a consultant for the user. It can generate ad copy, suggest audience segments, and interpret complex performance reports into plain English.

## 3. Unified Ad Delivery API
- **Tech Stack**: Go (Golang), gRPC, Redis.
- **Function**: A high-performance gateway that interfaces with various ad exchanges (Google Ads, Meta, TikTok). It handles request normalization, rate limiting, and real-time execution.

## 4. ML Bidding & Optimization Engine
- **Tech Stack**: Python, PyTorch/TensorFlow, Apache Kafka.
- **Function**: Processes real-time data streams to calculate the optimal bid for each ad impression. It utilizes historical performance data and real-time signals to maximize conversion probability.

## 5. Privacy Measurement Engine
- **Tech Stack**: Rust, WebAssembly, Confidential Computing (TEE).
- **Function**: Handles attribution and analytics without compromising individual user identity. It implements differential privacy and secure aggregation to provide insights while maintaining 100% compliance with CCPA/GDPR and future U.S. privacy laws.

## Data Flow
1. User interacts with the **Frontend** / **AI Assistant**.
2. Campaign configuration is sent to the **Ad Delivery API**.
3. **ML Bidding Engine** continuously optimizes bids based on live performance data.
4. **Privacy Measurement Engine** aggregates results and feeds them back to the **Frontend** for reporting.
