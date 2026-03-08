"""
竞价服务层（BiddingService）

四层 ML 决策流水线：
  Step 1: GBM win_prob    — sklearn 梯度提升，预测竞价获胜概率（快速 baseline）
  Step 2: DeepFM pCTR     — PyTorch DeepFM，深度学习点击率预测
  Step 3: GAT context     — Graph Attention Network，广告-内容匹配嵌入
  Step 4: PPO final_bid   — 强化学习 Agent，动态调整最终出价系数

设计原则：
  - Graceful Degradation：每层模型都有 try/except + fallback，任何单点失败不影响 API
  - 懒加载：首次请求时加载模型，不阻塞服务启动
  - 无 PII：所有特征匿名化（SHA-256 哈希）后才传入模型
  - 决策耗时目标 < 50ms（RTB 严格延迟要求）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 决策结果数据结构
# ─────────────────────────────────────────────

@dataclass
class BidDecision:
    """四层 ML 流水线的完整决策结果"""
    campaign_id: str

    # Step 1: GBM
    win_prob: float = 0.0           # GBM 预测的竞价获胜概率

    # Step 2: DeepFM
    predicted_ctr: float = 0.0     # DeepFM pCTR

    # Step 3: GAT
    context_embedding: Optional[list] = None  # GAT 输出的上下文嵌入向量
    ad_match_score: float = 0.0    # 广告-内容匹配分（embedding 相似度）

    # Step 4: PPO
    base_bid: float = 0.0
    bid_adjustment: float = 0.0    # PPO 输出的调整系数 δ ∈ [-1, 1]
    final_bid: float = 0.0         # base_bid × (1 + δ)

    # 综合指标
    ecpm: float = 0.0              # 预期 CPM = final_bid × pCTR × 1000
    model_version: str = "heuristic"
    decision_ms: float = 0.0

    # 各层 fallback 标志（用于监控）
    fallbacks: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# 竞价服务主体
# ─────────────────────────────────────────────

class BiddingService:
    """
    四层 ML 竞价服务

    调用链：
      GBM.predict_proba() → win_prob
          ↓
      DeepFM.predict()    → pCTR
          ↓
      GNNAdModel.forward() → context_embedding + match_score
          ↓
      PPOAgent.act()      → bid_adjustment δ
          ↓
      final_bid = base_bid × (1 + δ) × win_prob_weight
    """

    def __init__(self):
        # 模型实例（懒加载，None 表示尚未加载或加载失败）
        self._gbm = None          # sklearn GBM baseline
        self._deepfm = None       # PyTorch DeepFM
        self._gnn = None          # PyTorch GAT
        self._ppo = None          # PyTorch PPO Agent
        self._gnn_graph = None    # 预构建的广告生态图（静态缓存）
        self._loaded = False

    # ──────────────────────────────────────────
    # 懒加载
    # ──────────────────────────────────────────

    def _load_models(self):
        """首次调用时加载所有模型，失败不抛出（fallback 机制）"""
        if self._loaded:
            return
        t0 = time.perf_counter()

        # ── GBM（sklearn，无 GPU 依赖，优先加载）──
        self._load_gbm()

        # ── PyTorch 模型（可选，环境不支持时跳过）──
        try:
            import torch  # noqa: F401 — 检测 PyTorch 是否可用
            self._load_deepfm()
            self._load_gnn()
            self._load_ppo()
        except ImportError:
            logger.warning("PyTorch not available, all deep models disabled")

        self._loaded = True
        logger.info(f"BiddingService models loaded in {(time.perf_counter()-t0)*1000:.1f}ms | "
                    f"GBM={'✓' if self._gbm else '✗'} "
                    f"DeepFM={'✓' if self._deepfm else '✗'} "
                    f"GAT={'✓' if self._gnn else '✗'} "
                    f"PPO={'✓' if self._ppo else '✗'}")

    def _load_gbm(self):
        try:
            import os
            from ml.models.bidding_model import BiddingModel
            path = "ml/artifacts/bidding_model.pkl"
            if os.path.exists(path):
                self._gbm = BiddingModel.load(path)
                logger.info("GBM loaded from artifact")
            else:
                # 无预训练权重时用未训练实例（返回 fallback 值）
                self._gbm = BiddingModel(model_type="logistic")
                logger.warning("GBM artifact not found, using unfit instance (will fallback)")
        except Exception as e:
            logger.warning(f"GBM load failed: {e}")

    def _load_deepfm(self):
        try:
            import os
            from ml.models.deep_ctr_model import DeepFMTrainer, DeepFMConfig
            path = "ml/artifacts/deepfm.pt"
            if os.path.exists(path):
                self._deepfm = DeepFMTrainer.load(path)
                logger.info("DeepFM loaded from artifact")
            else:
                # 用默认配置初始化（推理会给出随机但合法的概率值）
                config = DeepFMConfig(num_fields=8, vocab_size=10_000, embed_dim=16,
                                      hidden_dims=[128, 64])
                self._deepfm = DeepFMTrainer(config)
                logger.warning("DeepFM artifact not found, using randomly initialized model")
        except Exception as e:
            logger.warning(f"DeepFM load failed: {e}")

    def _load_gnn(self):
        try:
            import os
            from ml.models.gnn_ad_model import GNNAdModel, build_synthetic_graph
            self._gnn = GNNAdModel(node_feat_dim=64, gat_hidden=32, num_heads=4, num_layers=2)
            self._gnn.eval()
            # 预构建静态图（生产环境从数据库/Redis 加载真实图结构）
            self._gnn_graph = build_synthetic_graph()
            logger.info("GAT model initialized (synthetic graph)")
        except Exception as e:
            logger.warning(f"GNN load failed: {e}")

    def _load_ppo(self):
        try:
            import os
            from ml.models.rl_bidding_agent import PPOBiddingAgent, PPOConfig
            path = "ml/artifacts/ppo_agent.pt"
            if os.path.exists(path):
                self._ppo = PPOBiddingAgent.load(path)
                logger.info("PPO agent loaded from artifact")
            else:
                config = PPOConfig(state_dim=20, action_dim=1, hidden_dim=128)
                self._ppo = PPOBiddingAgent(config)
                logger.warning("PPO artifact not found, using untrained agent (δ≈0)")
        except Exception as e:
            logger.warning(f"PPO load failed: {e}")

    # ──────────────────────────────────────────
    # Step 1: GBM win_prob
    # ──────────────────────────────────────────

    def _predict_win_prob(self, campaign: dict, request: dict) -> tuple[float, bool]:
        """
        GBM 预测竞价获胜概率

        Returns:
            (win_prob, used_fallback)
        """
        if self._gbm is not None and self._gbm._model is not None:
            try:
                feat = self._gbm.feature_engineer.build_features(campaign, request)
                # win_prob ≈ pCTR（竞价获胜与 CTR 强相关）
                win_prob = float(self._gbm.predict_ctr(feat))
                return min(max(win_prob, 0.0), 1.0), False
            except Exception as e:
                logger.warning(f"GBM predict failed: {e}")

        # Fallback：基于历史 CTR 的贝叶斯平滑估计
        impressions = max(campaign.get("impressions", 0), 1)
        clicks = campaign.get("clicks", 0)
        win_prob = (clicks + 0.5) / (impressions + 100)
        return min(win_prob * 10, 0.5), True  # 粗略映射到获胜概率

    # ──────────────────────────────────────────
    # Step 2: DeepFM pCTR
    # ──────────────────────────────────────────

    def _predict_ctr_deepfm(self, campaign: dict, request: dict) -> tuple[float, bool]:
        """
        DeepFM 深度学习 pCTR 预测

        Returns:
            (pCTR, used_fallback)
        """
        if self._deepfm is not None:
            try:
                import torch
                hasher = self._deepfm.hasher
                feat_dict = {
                    "device_type": str(request.get("device_type", "unknown")),
                    "os": str(request.get("os", "unknown")),
                    "ad_format": str(request.get("ad_format", "banner")),
                    "geo_country": str(
                        ((request.get("geo") or {}).get("countries") or [""])[0]
                    ),
                    "bidding_strategy": str(campaign.get("bidding_strategy", "cpc")),
                    "hour": str(time.localtime().tm_hour),
                    "dow": str(time.localtime().tm_wday),
                    "budget_tier": str(
                        int(float(campaign.get("budget", {}).get("total", 0)) // 100)
                    ),
                }
                X = hasher.hash_batch([feat_dict])
                preds = self._deepfm.predict(X)
                return float(preds[0]), False
            except Exception as e:
                logger.warning(f"DeepFM predict failed: {e}")

        # Fallback：基于 GBM win_prob 的简单映射
        impressions = max(campaign.get("impressions", 0), 1)
        clicks = campaign.get("clicks", 0)
        return (clicks + 0.1) / (impressions + 100), True

    # ──────────────────────────────────────────
    # Step 3: GAT context embedding
    # ──────────────────────────────────────────

    def _get_context_embedding(
        self, campaign: dict, request: dict
    ) -> tuple[Optional[list], float, bool]:
        """
        GAT 广告-内容匹配嵌入

        Returns:
            (embedding_list, match_score, used_fallback)
        """
        if self._gnn is not None and self._gnn_graph is not None:
            try:
                import torch
                # 用 campaign/request 特征映射到图节点索引（简化：哈希到节点范围）
                ad_slot_idx = hash(request.get("imp_id", "default")) % 50
                category_idx = 50 + (hash(request.get("site_id", "default")) % 20)
                query_pairs = torch.tensor([[ad_slot_idx, category_idx]], dtype=torch.long)

                self._gnn.eval()
                with torch.no_grad():
                    score = self._gnn(self._gnn_graph, query_pairs)
                    match_score = float(score[0])

                    # 提取节点 embedding 用于下游（可选）
                    embeddings = self._gnn.get_node_embeddings(self._gnn_graph)
                    ad_embed = embeddings[ad_slot_idx].tolist()

                return ad_embed, match_score, False
            except Exception as e:
                logger.warning(f"GAT forward failed: {e}")

        # Fallback：返回 None embedding，match_score = 0.5（中性）
        return None, 0.5, True

    # ──────────────────────────────────────────
    # Step 4: PPO bid adjustment
    # ──────────────────────────────────────────

    def _get_ppo_adjustment(
        self,
        campaign: dict,
        request: dict,
        win_prob: float,
        predicted_ctr: float,
    ) -> tuple[float, bool]:
        """
        PPO Agent 竞价调整系数

        Returns:
            (adjustment δ, used_fallback)
        """
        if self._ppo is not None:
            try:
                market_state = {
                    "avg_market_cpm": float(request.get("floor_price", 1.0)) * 1000,
                    "win_rate": win_prob,
                    "floor_price": float(request.get("floor_price", 0.01)),
                    "competition_level": 1.0 - win_prob,
                    "time_pressure": 0.0,
                    "supply_index": 1.0,
                }
                campaign_metrics = {
                    "spend_ratio": float(campaign.get("spend", 0)) / max(
                        float(campaign.get("budget", {}).get("total", 1)), 1
                    ),
                    "ctr": predicted_ctr,
                    "cvr": 0.02,
                    "budget_utilization": float(campaign.get("spend", 0)) / max(
                        float(campaign.get("budget", {}).get("total", 1)), 1
                    ),
                    "impressions": campaign.get("impressions", 0),
                    "clicks": campaign.get("clicks", 0),
                }
                state = self._ppo.build_state(campaign_metrics, market_state)
                adjustment, _, _ = self._ppo.act(state, deterministic=True)
                return float(adjustment), False
            except Exception as e:
                logger.warning(f"PPO act failed: {e}")

        # Fallback：根据 win_prob 简单调整（高获胜率时适当降价）
        if win_prob > 0.7:
            return -0.1, True   # 获胜率高，降低出价节省预算
        elif win_prob < 0.3:
            return 0.15, True   # 获胜率低，适当提高出价
        return 0.0, True

    # ──────────────────────────────────────────
    # 主决策接口
    # ──────────────────────────────────────────

    def decide(self, campaign: dict, request: dict) -> BidDecision:
        """
        完整四层 ML 竞价决策

        Args:
            campaign: 广告活动数据（含 bid_amount, budget, impressions, clicks 等）
            request:  竞价请求数据（含 floor_price, device_type, geo 等）

        Returns:
            BidDecision — 包含所有中间结果和最终出价
        """
        self._load_models()
        t0 = time.perf_counter()

        fallbacks = {}
        base_bid = float(campaign.get("bid_amount", 1.0))
        campaign_id = str(campaign.get("id", "unknown"))

        # ── Step 1: GBM win_prob ──
        win_prob, fb1 = self._predict_win_prob(campaign, request)
        fallbacks["gbm"] = fb1

        # ── Step 2: DeepFM pCTR ──
        predicted_ctr, fb2 = self._predict_ctr_deepfm(campaign, request)
        fallbacks["deepfm"] = fb2

        # ── Step 3: GAT context embedding ──
        context_embedding, ad_match_score, fb3 = self._get_context_embedding(campaign, request)
        fallbacks["gat"] = fb3

        # ── Step 4: PPO bid adjustment ──
        adjustment, fb4 = self._get_ppo_adjustment(campaign, request, win_prob, predicted_ctr)
        fallbacks["ppo"] = fb4

        # ── 融合计算最终出价 ──
        # 将 GAT 匹配分融入出价（高匹配度的广告位值得多出价）
        match_weight = 0.8 + 0.4 * ad_match_score  # [0.8, 1.2]
        final_bid = base_bid * (1 + adjustment) * match_weight
        final_bid = max(round(final_bid, 4), 0.01)  # 最低出价保护

        # eCPM = 出价 × pCTR × 1000
        ecpm = round(final_bid * predicted_ctr * 1000, 4)

        # 模型版本标签（用于 A/B 测试追踪）
        active_models = []
        if not fb2: active_models.append("deepfm")
        if not fb3: active_models.append("gat")
        if not fb4: active_models.append("ppo")
        if not fb1: active_models.append("gbm")
        model_version = "+".join(active_models) if active_models else "heuristic"

        decision_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "bid.decision",
            extra=dict(
                campaign_id=campaign_id,
                win_prob=win_prob,
                predicted_ctr=predicted_ctr,
                ad_match_score=ad_match_score,
                adjustment=adjustment,
                final_bid=final_bid,
                ecpm=ecpm,
                decision_ms=decision_ms,
                fallbacks=fallbacks,
            )
        )

        return BidDecision(
            campaign_id=campaign_id,
            win_prob=round(win_prob, 6),
            predicted_ctr=round(predicted_ctr, 6),
            context_embedding=context_embedding[:8] if context_embedding else None,  # 截断避免响应过大
            ad_match_score=round(ad_match_score, 4),
            base_bid=base_bid,
            bid_adjustment=round(adjustment, 4),
            final_bid=final_bid,
            ecpm=ecpm,
            model_version=model_version,
            decision_ms=decision_ms,
            fallbacks=fallbacks,
        )


# ─────────────────────────────────────────────
# 单例 + FastAPI 依赖注入
# ─────────────────────────────────────────────

_service: Optional[BiddingService] = None


def get_bidding_service() -> BiddingService:
    """
    FastAPI 依赖注入入口

    用法：
        @router.post("/bid")
        async def bid(request: BidRequest, svc: BiddingService = Depends(get_bidding_service)):
            decision = svc.decide(campaign, request.model_dump())
    """
    global _service
    if _service is None:
        _service = BiddingService()
    return _service
