import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Campaign Types
export interface Campaign {
  id: string
  name: string
  status: 'draft' | 'active' | 'paused' | 'completed' | 'archived'
  budget: {
    total: number
    daily?: number
    currency: string
    pacing: string
  }
  bidding_strategy: string
  bid_amount: number
  impressions: number
  clicks: number
  conversions: number
  spend: number
  ctr: number
  created_at: string
  updated_at: string
}

export interface CampaignCreate {
  name: string
  budget: {
    total: number
    daily?: number
    currency: string
  }
  bidding_strategy: string
  bid_amount: number
  ad_format: string
  start_date: string
  end_date?: string
  targeting?: {
    geo?: { countries?: string[] }
    audience?: { age_min?: number; age_max?: number }
    device?: { devices?: string[] }
  }
}

// API Methods
export const campaignApi = {
  list: (params?: { page?: number; page_size?: number; status?: string }) =>
    api.get<{ items: Campaign[]; total: number; page: number }>('/campaigns', { params }),
  
  get: (id: string) =>
    api.get<Campaign>(`/campaigns/${id}`),
  
  create: (data: CampaignCreate) =>
    api.post<Campaign>('/campaigns', data),
  
  update: (id: string, data: Partial<CampaignCreate>) =>
    api.patch<Campaign>(`/campaigns/${id}`, data),
  
  delete: (id: string) =>
    api.delete(`/campaigns/${id}`),
}

// Bidding API
export interface BidRequest {
  id: string
  imp_id: string
  site_id: string
  device_type?: string
  os?: string
  geo?: { countries?: string[] }
  floor_price: number
  ad_format: string
}

export interface BidResponse {
  request_id: string
  campaign_id?: string
  bid_price?: number
  decision_ms: number
  predicted_ctr?: number
  reason: string
  // ML Pipeline fields (Sprint 2)
  model_version?: string
  fallbacks?: {
    gbm?: boolean
    deepfm?: boolean
    gat?: boolean
    ppo?: boolean
  }
  context_embedding?: number[]
  ad_match_score?: number
}

export const biddingApi = {
  submitBid: (request: BidRequest) =>
    api.post<BidResponse>('/bidding/bid', request),
}

export default api
