"""
竞价（Bidding）路由
实现实时竞价（RTB）端点：bid request → bid response
目标延迟 < 100ms（含 ML 推理）
"""
import time
import uuid
from decimal import Decimal
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.models.campaign import BidRequest, BidResponse, CampaignStatus

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/bidding", tags=["bidding"])


# ─────────────────────────────────────────────
# 竞价引擎（轻量内联版，生产应抽成 BiddingService）
# ─────────────────────────────────────────────

class BiddingEngine:
    """
    实时竞价决策引擎

    工作流程：
    1. 候选活动检索（索引 + 定向过滤）
    2. ML 模型预测 pCTR / pCVR
    3. eCPM 计算 = bid_amount × pCTR × 1000
    4. 底价校验
    5. 返回出价最高的活动
    """

    def __init__(self):
        # 生产环境：注入 CampaignRepository + MLBiddingModel
        self._ml_model = None   # 懒加载，避免启动时间过长

    def _load_ml_model(self):
        """懒加载 ML 模型（避免冷启动超时）"""
        if self._ml_model is None:
            try:
                from ml.models.bidding_model import BiddingModel
                self._ml_model = BiddingModel.load("ml/artifacts/bidding_model.pkl")
            except Exception:
                logger.warning("ml_model.load_failed, using fallback heuristic")
        return self._ml_model

    def _match_targeting(self, campaign: dict, request: BidRequest) -> bool:
        """简化定向匹配：生产环境使用倒排索引 + bloom filter"""
        targeting = campaign.get("targeting", {})

        # 地理定向
        geo_cfg = targeting.get("geo", {})
        if geo_cfg.get("countries") and request.geo:
            req_countries = request.geo.countries or []
            if not any(c in geo_cfg["countries"] for c in req_countries):
                return False

        # 设备定向
        device_cfg = targeting.get("device", {})
        if device_cfg.get("devices") and request.device_type:
            if request.device_type not in device_cfg["devices"]:
                return False

        return True

    def _predict_ctr(self, campaign: dict, request: BidRequest) -> float:
        """
        预测点击率（pCTR）
        优先使用 ML 模型，降级到启发式规则
        """
        model = self._load_ml_model()
        if model:
            try:
                features = model.build_features(campaign, request.model_dump())
                return float(model.predict_ctr(features))
            except Exception as e:
                logger.warning("ml_predict.failed", error=str(e))

        # 降级：基于历史 CTR 的启发式估算
        impressions = campaign.get("impressions", 1)
        clicks = campaign.get("clicks", 0)
        historical_ctr = clicks / max(impressions, 1)
        # 贝叶斯平滑：引入先验（行业平均 CTR ~0.1%）
        prior_ctr = 0.001
        prior_weight = 100
        smoothed_ctr = (clicks + prior_ctr * prior_weight) / (impressions + prior_weight)
        return max(smoothed_ctr, 0.0001)

    def select_winner(
        self, candidates: list[dict], request: BidRequest
    ) -> tuple[Optional[dict], Optional[float], Optional[float]]:
        """
        从候选活动中选出最优出价者

        Returns:
            (winner_campaign, predicted_ctr, ecpm)
        """
        best_campaign = None
        best_ecpm = 0.0
        best_ctr = 0.0

        for campaign in candidates:
            # 1. 状态校验
            if campaign.get("status") != CampaignStatus.ACTIVE.value:
                continue

            # 2. 预算校验
            spend = float(campaign.get("spend", 0))
            budget_total = float(campaign.get("budget", {}).get("total", 0))
            if spend >= budget_total:
                continue

            # 3. 定向匹配
            if not self._match_targeting(campaign, request):
                continue

            # 4. pCTR 预测 + eCPM 计算
            pCTR = self._predict_ctr(campaign, request)
            bid_amount = float(campaign.get("bid_amount", 0))
            # eCPM = bid × pCTR × 1000 (CPC 转 CPM)
            ecpm = bid_amount * pCTR * 1000

            if ecpm > best_ecpm:
                best_ecpm = ecpm
                best_campaign = campaign
                best_ctr = pCTR

        return best_campaign, best_ctr, best_ecpm if best_campaign else None


# 单例引擎
_engine = BiddingEngine()


# ─────────────────────────────────────────────
# POST /bidding/bid — 主竞价端点
# ─────────────────────────────────────────────
@router.post(
    "/bid",
    response_model=BidResponse,
    summary="提交竞价请求",
    description="OpenRTB 2.x 兼容竞价端点，目标 P99 延迟 < 100ms",
)
async def submit_bid(request: BidRequest, background_tasks: BackgroundTasks) -> BidResponse:
    """
    实时竞价主入口

    处理流程：
    1. 解析竞价请求
    2. 检索候选广告活动
    3. ML 模型打分 + 选优
    4. 底价校验
    5. 返回出价（或 no-bid）
    6. 异步记录曝光日志
    """
    t0 = time.perf_counter()

    # 从内存存储拿候选（生产环境：Redis 活动索引 + 数据库）
    from api.routers.campaigns import _campaigns
    candidates = list(_campaigns.values())

    winner, predicted_ctr, ecpm = _engine.select_winner(candidates, request)

    decision_ms = (time.perf_counter() - t0) * 1000

    # No bid
    if winner is None or ecpm is None:
        logger.info(
            "bid.no_bid",
            request_id=request.id,
            site_id=request.site_id,
            decision_ms=round(decision_ms, 2),
        )
        return BidResponse(
            request_id=request.id,
            reason="no_targeting",
            decision_ms=round(decision_ms, 2),
        )

    # 底价校验（floor price 单位：CPM USD）
    floor_cpm = float(request.floor_price)
    if ecpm < floor_cpm:
        return BidResponse(
            request_id=request.id,
            reason="below_floor",
            decision_ms=round(decision_ms, 2),
        )

    # 二价出价（Vickrey auction：出价最高但支付第二高价）
    # 简化：bid_price = floor_price + 0.01（生产环境接入竞价服务器）
    bid_price = Decimal(str(round(floor_cpm + 0.01, 4)))

    # 异步记录竞价日志（不阻塞响应）
    background_tasks.add_task(
        _log_bid_event,
        request_id=request.id,
        campaign_id=winner["id"],
        bid_price=float(bid_price),
        predicted_ctr=predicted_ctr,
        ecpm=ecpm,
    )

    logger.info(
        "bid.placed",
        request_id=request.id,
        campaign_id=winner["id"],
        bid_price=float(bid_price),
        ecpm=round(ecpm, 4),
        decision_ms=round(decision_ms, 2),
    )

    return BidResponse(
        request_id=request.id,
        campaign_id=winner["id"],
        bid_price=bid_price,
        ad_markup=f"<div data-ad-id='{winner['id']}'><!-- AD MARKUP --></div>",
        win_notice_url=f"https://api.smb-adtech.com/bidding/win/{request.id}",
        click_url=f"https://api.smb-adtech.com/click/{winner['id']}",
        decision_ms=round(decision_ms, 2),
        reason="win",
        predicted_ctr=predicted_ctr,
        predicted_cvr=None,  # TODO: CVR 模型
    )


# ─────────────────────────────────────────────
# POST /bidding/win/{request_id} — 曝光赢得通知
# ─────────────────────────────────────────────
@router.post(
    "/win/{request_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="曝光赢得通知（Win Notice）",
)
async def win_notice(
    request_id: str,
    campaign_id: str,
    clearing_price: float,
    background_tasks: BackgroundTasks,
):
    """
    SSP 回调通知广告获胜。
    触发：更新 campaign spend、记录曝光、触发归因流水线。
    """
    background_tasks.add_task(
        _process_win,
        request_id=request_id,
        campaign_id=campaign_id,
        clearing_price=clearing_price,
    )
    logger.info("bid.won", request_id=request_id, campaign_id=campaign_id, price=clearing_price)


# ─────────────────────────────────────────────
# 异步后台任务
# ─────────────────────────────────────────────

async def _log_bid_event(**kwargs):
    """异步写入竞价事件到 Kafka / ClickHouse（TODO: 接入 MQ）"""
    logger.debug("bid.event_logged", **kwargs)


async def _process_win(request_id: str, campaign_id: str, clearing_price: float):
    """处理获胜：扣预算、更新展示计数"""
    from api.routers.campaigns import _campaigns
    if campaign_id in _campaigns:
        _campaigns[campaign_id]["impressions"] += 1
        current_spend = float(_campaigns[campaign_id].get("spend", 0))
        _campaigns[campaign_id]["spend"] = round(current_spend + clearing_price / 1000, 6)
    logger.info("win.processed", request_id=request_id, campaign_id=campaign_id)
