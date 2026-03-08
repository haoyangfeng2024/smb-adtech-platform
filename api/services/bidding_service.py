"""
竞价服务层（BiddingService）

整合所有 ML 模型，提供统一的竞价决策接口：
  - DeepFM     → pCTR 预测（深度学习，高精度）
  - GAT        → 广告-内容匹配分数
  - PPO Agent  → 动态竞价调整系数
  - sklearn GBM→ 降级 fallback（无 PyTorch 环境时使用）

设计原则：
  1. 模型懒加载（启动不阻塞，首次请求时加载）
  2. 多级 fallback（PyTorch → sklearn → 启发式规则）
  3. 决策耗时 < 50ms（RTB 严格延迟要求）
  4. 无 PII：所有特征匿名化后才传入模型
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BidDecision:
    """竞价决策结果"""
    campaign_id: str
    base_bid: float
    adjustment: float          # PPO 输出的调整系数 δ ∈ [-1, 1]
    final_bid: float           # base_bid × (1 + adjustment)
    predicted_ctr: float       # DeepFM pCTR
    ad_match_score: float      # GAT 广告匹配分
    ecpm: float                # 预期 CPM = final_bid × pCTR × 1000
    model_version: str         # 用于 A/B 追踪
    decision_ms: float


class BiddingService:
    """
    ML 竞价服务

    调用顺序：
      1. DeepFM.predict()        → pCTR
      2. GNNAdModel.forward()    → ad_match_score
      3. PPOAgent.act()          → bid_adjustment δ
      4. 融合计算 final_bid + eCPM
    """

    def __init__(self):
        self._deepfm = None
        self._gnn = None
        self._ppo = None
        self._gbm = None          # sklearn fallback
        self._models_loaded = False

    def _lazy_load(self):
        """懒加载所有模型（首次调用时）"""
        if self._models_loaded:
            return
        t0 = time.perf_counter()
        try:
            # 尝试加载 PyTorch 模型
            from ml.models.deep_ctr_model import DeepFMTrainer, DeepFMConfig
            from ml.models.gnn_ad_model import GNNAdModel
            from ml.models.rl_bidding_agent import PPOBiddingAgent, PPOConfig
            import os

            deepfm_path = "ml/artifacts/deepfm.pt"
            if os.path.exists(deepfm_path):
                self._deepfm = DeepFMTrainer.load(deepfm_path)
                logger.info("DeepFM loaded")
            else:
                logger.warning("DeepFM artifact not found, using heuristic pCTR")

            ppo_path = "ml/artifacts/ppo_agent.pt"
            if os.path.exists(ppo_path):
                self._ppo = PPOBiddingAgent.load(ppo_path)
                logger.info("PPO agent loaded")

            # GNN 模型（无预训练权重时用随机初始化做演示）
            self._gnn = GNNAdModel(node_feat_dim=64, gat_hidden=32, num_heads=4, num_layers=2)
            self._gnn.eval()
            logger.info("GNN model initialized")

        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}, falling back to sklearn")

        # sklearn GBM fallback
        try:
            from ml.models.bidding_model import BiddingModel
            import os
            gbm_path = "ml/artifacts/bidding_model.pkl"
            if os.path.exists(gbm_path):
                self._gbm = BiddingModel.load(gbm_path)
                logger.info("sklearn GBM loaded")
        except Exception as e:
            logger.warning(f"GBM load failed: {e}")

        self._models_loaded = True
        logger.info(f"Models loaded in {(time.perf_counter()-t0)*1000:.1f}ms")

    def predict_ctr(self, campaign: dict, request: dict) -> float:
        """
        pCTR 预测，按优先级降级：
          DeepFM → sklearn GBM → 贝叶斯平滑启发式
        """
        # 优先 DeepFM
        if self._deepfm is not None:
            try:
                import torch
                hasher = self._deepfm.hasher
                feat_dict = {
                    "device_type": request.get("device_type", "unknown"),
                    "os": request.get("os", "unknown"),
                    "ad_format": request.get("ad_format", "banner"),
                    "geo": str((request.get("geo") or {}).get("countries", [""])[0]),
                    "bidding_strategy": campaign.get("bidding_strategy", "cpc"),
                    "hour": str(time.localtime().tm_hour),
                    "dow": str(time.localtime().tm_wday),
                    "campaign_id_hash": str(hash(campaign.get("id", "")) % 10000),
                }
                X = hasher.hash_batch([feat_dict])
                pred = self._deepfm.predict(X)
                return float(pred[0])
            except Exception as e:
                logger.warning(f"DeepFM predict failed: {e}")

        # 降级：sklearn GBM
        if self._gbm is not None:
            try:
                feat = self._gbm.feature_engineer.build_features(campaign, request)
                return float(self._gbm.predict_ctr(feat))
            except Exception as e:
                logger.warning(f"GBM predict failed: {e}")

        # 最终降级：贝叶斯平滑启发式
        impressions = max(campaign.get("impressions", 0), 1)
        clicks = campaign.get("clicks", 0)
        return (clicks + 0.1) / (impressions + 100)

    def get_bid_adjustment(self, campaign: dict, market_state: dict) -> float:
        """
        PPO Agent 给出竞价调整系数 δ
        无 PPO 时返回 0（不调整，使用 base_bid）
        """
        if self._ppo is None:
            return 0.0
        try:
            state = self._ppo.build_state(campaign, market_state)
            action, _, _ = self._ppo.act(state, deterministic=True)
            return float(action)
        except Exception as e:
            logger.warning(f"PPO act failed: {e}")
            return 0.0

    def decide(
        self,
        campaign: dict,
        request: dict,
        market_state: Optional[dict] = None,
    ) -> BidDecision:
        """
        完整竞价决策流水线

        Args:
            campaign: 广告活动数据
            request:  竞价请求数据
            market_state: 市场行情（可选，PPO 使用）

        Returns:
            BidDecision 包含最终出价和所有中间指标
        """
        self._lazy_load()
        t0 = time.perf_counter()

        base_bid = float(campaign.get("bid_amount", 1.0))
        market_state = market_state or {}

        # 1. pCTR 预测
        predicted_ctr = self.predict_ctr(campaign, request)

        # 2. GAT 广告匹配分（简化：返回 pCTR 的加权版）
        ad_match_score = min(predicted_ctr * 10, 1.0)  # 归一化到 [0,1]

        # 3. PPO 竞价调整
        adjustment = self.get_bid_adjustment(campaign, {
            **market_state,
            "ctr": predicted_ctr,
            "spend_ratio": float(campaign.get("spend", 0)) / max(float(campaign.get("budget", {}).get("total", 1)), 1),
        })

        # 4. 计算最终出价
        final_bid = base_bid * (1 + adjustment)
        final_bid = max(final_bid, 0.01)  # 最低出价保护

        # 5. eCPM
        ecpm = final_bid * predicted_ctr * 1000

        decision_ms = (time.perf_counter() - t0) * 1000

        return BidDecision(
            campaign_id=campaign.get("id", ""),
            base_bid=base_bid,
            adjustment=adjustment,
            final_bid=round(final_bid, 4),
            predicted_ctr=round(predicted_ctr, 6),
            ad_match_score=round(ad_match_score, 4),
            ecpm=round(ecpm, 4),
            model_version="deepfm+ppo" if self._deepfm and self._ppo else "heuristic",
            decision_ms=round(decision_ms, 2),
        )


# 单例（进程级共享，避免重复加载模型）
_service: Optional[BiddingService] = None


def get_bidding_service() -> BiddingService:
    """FastAPI 依赖注入入口"""
    global _service
    if _service is None:
        _service = BiddingService()
    return _service
