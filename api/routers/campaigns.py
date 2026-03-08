"""
广告活动 CRUD 路由
提供完整的 Create / List / Get / Update 接口
"""
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from api.models.campaign import (
    CampaignCreate,
    CampaignListResponse,
    CampaignResponse,
    CampaignStatus,
    CampaignUpdate,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/campaigns", tags=["campaigns"])

# ─────────────────────────────────────────────
# 内存存储（开发阶段占位，生产替换为 DB）
# ─────────────────────────────────────────────
_campaigns: dict[str, dict] = {}


def _not_found(campaign_id: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Campaign {campaign_id!r} not found",
    )


# ─────────────────────────────────────────────
# POST /campaigns — 创建广告活动
# ─────────────────────────────────────────────
@router.post(
    "/",
    response_model=CampaignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建广告活动",
)
async def create_campaign(payload: CampaignCreate) -> CampaignResponse:
    """
    创建一个新的广告活动。

    - 自动生成 UUID 作为活动 ID
    - 初始状态为 draft（草稿）
    - 返回包含实时指标的完整响应
    """
    campaign_id = str(uuid.uuid4())
    now = datetime.utcnow()

    data = payload.model_dump()
    data.update(
        id=campaign_id,
        created_at=now,
        updated_at=now,
        impressions=0,
        clicks=0,
        conversions=0,
        spend=Decimal("0.00"),
        ctr=0.0,
        cpa=None,
    )
    _campaigns[campaign_id] = data

    logger.info("campaign.created", campaign_id=campaign_id, name=payload.name)
    return CampaignResponse(**data)


# ─────────────────────────────────────────────
# GET /campaigns — 分页列表
# ─────────────────────────────────────────────
@router.get(
    "/",
    response_model=CampaignListResponse,
    summary="获取广告活动列表",
)
async def list_campaigns(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    status: Optional[CampaignStatus] = Query(None, description="按状态过滤"),
    advertiser_id: Optional[str] = Query(None, description="按广告主过滤"),
) -> CampaignListResponse:
    """
    分页获取广告活动列表，支持状态和广告主过滤。
    """
    items = list(_campaigns.values())

    # 过滤
    if status:
        items = [c for c in items if c["status"] == status.value]
    if advertiser_id:
        items = [c for c in items if c["advertiser_id"] == advertiser_id]

    total = len(items)

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return CampaignListResponse(
        items=[CampaignResponse(**c) for c in page_items],
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
    )


# ─────────────────────────────────────────────
# GET /campaigns/{id} — 获取单个活动
# ─────────────────────────────────────────────
@router.get(
    "/{campaign_id}",
    response_model=CampaignResponse,
    summary="获取广告活动详情",
)
async def get_campaign(campaign_id: str) -> CampaignResponse:
    """根据 ID 获取单个广告活动详情。"""
    if campaign_id not in _campaigns:
        _not_found(campaign_id)
    return CampaignResponse(**_campaigns[campaign_id])


# ─────────────────────────────────────────────
# PATCH /campaigns/{id} — 更新活动（部分更新）
# ─────────────────────────────────────────────
@router.patch(
    "/{campaign_id}",
    response_model=CampaignResponse,
    summary="更新广告活动（部分更新）",
)
async def update_campaign(campaign_id: str, payload: CampaignUpdate) -> CampaignResponse:
    """
    部分更新广告活动（PATCH 语义）。

    - 只更新提供的字段
    - 自动更新 updated_at 时间戳
    - 状态机校验：completed/archived 活动不可修改
    """
    if campaign_id not in _campaigns:
        _not_found(campaign_id)

    existing = _campaigns[campaign_id]

    # 状态机保护
    current_status = existing.get("status")
    if current_status in (CampaignStatus.COMPLETED.value, CampaignStatus.ARCHIVED.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Campaign in status {current_status!r} cannot be modified",
        )

    # 仅更新非 None 字段
    updates = payload.model_dump(exclude_none=True)
    for key, value in updates.items():
        existing[key] = value
    existing["updated_at"] = datetime.utcnow()

    logger.info("campaign.updated", campaign_id=campaign_id, fields=list(updates.keys()))
    return CampaignResponse(**existing)


# ─────────────────────────────────────────────
# DELETE /campaigns/{id} — 软删除（归档）
# ─────────────────────────────────────────────
@router.delete(
    "/{campaign_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="归档广告活动（软删除）",
)
async def archive_campaign(campaign_id: str):
    """将广告活动状态设置为 archived（软删除，不物理删除数据）。"""
    if campaign_id not in _campaigns:
        _not_found(campaign_id)

    _campaigns[campaign_id]["status"] = CampaignStatus.ARCHIVED.value
    _campaigns[campaign_id]["updated_at"] = datetime.utcnow()
    logger.info("campaign.archived", campaign_id=campaign_id)
