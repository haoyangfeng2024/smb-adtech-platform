"""
广告活动 Pydantic 数据模型
定义所有 API 请求/响应结构，同时作为数据契约层
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class BiddingStrategy(str, Enum):
    CPC = "cpc"           # Cost per click
    CPM = "cpm"           # Cost per mille (千次展示成本)
    CPA = "cpa"           # Cost per acquisition
    TARGET_ROAS = "target_roas"  # 目标广告支出回报率
    SMART = "smart"       # ML 自动竞价


class AdFormat(str, Enum):
    BANNER = "banner"
    VIDEO = "video"
    NATIVE = "native"
    INTERSTITIAL = "interstitial"


# ─────────────────────────────────────────────
# Targeting 子模型
# ─────────────────────────────────────────────

class GeoTargeting(BaseModel):
    countries: list[str] = Field(default_factory=list, description="ISO 3166-1 alpha-2")
    regions: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    radius_km: Optional[float] = Field(None, ge=0, description="地理围栏半径（km）")
    lat: Optional[float] = None
    lon: Optional[float] = None


class AudienceTargeting(BaseModel):
    age_min: Optional[int] = Field(None, ge=13, le=100)
    age_max: Optional[int] = Field(None, ge=13, le=100)
    genders: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    custom_segments: list[str] = Field(default_factory=list, description="自定义受众包 ID")

    @model_validator(mode="after")
    def validate_age_range(self) -> AudienceTargeting:
        if self.age_min and self.age_max and self.age_min > self.age_max:
            raise ValueError("age_min 不能大于 age_max")
        return self


class DeviceTargeting(BaseModel):
    devices: list[str] = Field(default_factory=list)   # mobile/desktop/tablet
    os: list[str] = Field(default_factory=list)         # ios/android/windows
    browsers: list[str] = Field(default_factory=list)


class Targeting(BaseModel):
    geo: GeoTargeting = Field(default_factory=GeoTargeting)
    audience: AudienceTargeting = Field(default_factory=AudienceTargeting)
    device: DeviceTargeting = Field(default_factory=DeviceTargeting)
    keywords: list[str] = Field(default_factory=list)
    excluded_placements: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Budget 子模型
# ─────────────────────────────────────────────

class Budget(BaseModel):
    total: Decimal = Field(..., gt=0, description="总预算（USD）")
    daily: Optional[Decimal] = Field(None, gt=0, description="每日预算上限")
    currency: str = Field(default="USD", max_length=3)
    pacing: str = Field(default="standard", pattern="^(standard|accelerated)$")

    @field_validator("total", "daily", mode="before")
    @classmethod
    def round_currency(cls, v):
        if v is not None:
            return round(Decimal(str(v)), 2)
        return v


# ─────────────────────────────────────────────
# Campaign 核心模型
# ─────────────────────────────────────────────

class CampaignBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    advertiser_id: str = Field(..., description="广告主 ID")
    status: CampaignStatus = CampaignStatus.DRAFT
    bidding_strategy: BiddingStrategy = BiddingStrategy.CPC
    bid_amount: Decimal = Field(..., gt=0, description="竞价金额（USD）")
    budget: Budget
    targeting: Targeting = Field(default_factory=Targeting)
    ad_format: AdFormat = AdFormat.BANNER
    start_date: datetime
    end_date: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dates(self) -> CampaignBase:
        if self.end_date and self.end_date <= self.start_date:
            raise ValueError("end_date 必须晚于 start_date")
        return self


class CampaignCreate(CampaignBase):
    """创建广告活动请求体"""
    pass


class CampaignUpdate(BaseModel):
    """PATCH 语义更新，所有字段可选"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[CampaignStatus] = None
    bid_amount: Optional[Decimal] = Field(None, gt=0)
    budget: Optional[Budget] = None
    targeting: Optional[Targeting] = None
    end_date: Optional[datetime] = None
    tags: Optional[list[str]] = None


class CampaignResponse(CampaignBase):
    """API 响应，包含服务端生成字段 + 实时统计"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # 实时指标
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: Decimal = Decimal("0.00")
    ctr: float = 0.0     # Click-through rate
    cpa: Optional[float] = None  # Cost per acquisition

    model_config = {"from_attributes": True}


class CampaignListResponse(BaseModel):
    items: list[CampaignResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


# ─────────────────────────────────────────────
# Bid Request / Response (OpenRTB 2.x 兼容)
# ─────────────────────────────────────────────

class BidRequest(BaseModel):
    """OpenRTB 2.x 兼容竞价请求"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    imp_id: str = Field(..., description="广告展示位 ID")
    site_id: str = Field(..., description="发布商站点 ID")
    user_id: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    geo: Optional[GeoTargeting] = None
    device_type: Optional[str] = None
    os: Optional[str] = None
    ad_format: AdFormat = AdFormat.BANNER
    floor_price: Decimal = Field(default=Decimal("0.01"), gt=0, description="底价 USD CPM")
    timeout_ms: int = Field(default=100, ge=10, le=300)
    context: dict = Field(default_factory=dict, description="额外上下文特征")


class BidResponse(BaseModel):
    """竞价响应"""
    request_id: str
    campaign_id: Optional[str] = None
    bid_price: Optional[Decimal] = None   # None = 不参与竞价
    ad_markup: Optional[str] = None
    win_notice_url: Optional[str] = None
    click_url: Optional[str] = None
    decision_ms: float = Field(..., description="决策耗时毫秒")
    reason: str = Field(default="win")   # win/no_budget/no_targeting/below_floor
    predicted_ctr: Optional[float] = None
    predicted_cvr: Optional[float] = None
