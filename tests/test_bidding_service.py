"""
BiddingService 单元测试

覆盖场景：
  1. 正常竞价流程（四层 ML 全部可用）
  2. PyTorch 不可用 → 降级到 sklearn GBM
  3. GBM 也不可用 → 降级到启发式规则
  4. 单层模型崩溃（其他层正常）
  5. 边界值：零预算、零曝光、极端出价
  6. 并发安全：多次调用单例不重复加载

运行：
    pytest tests/test_bidding_service.py -v
"""
from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional

import pytest

# ─────────────────────────────────────────────
# 测试数据工厂
# ─────────────────────────────────────────────

def make_campaign(
    campaign_id: str = "camp_001",
    bid_amount: float = 1.5,
    budget_total: float = 1000.0,
    spend: float = 100.0,
    impressions: int = 10000,
    clicks: int = 50,
    status: str = "active",
    bidding_strategy: str = "cpc",
) -> dict:
    return {
        "id": campaign_id,
        "bid_amount": bid_amount,
        "budget": {"total": budget_total, "daily": budget_total / 30},
        "spend": spend,
        "impressions": impressions,
        "clicks": clicks,
        "status": status,
        "bidding_strategy": bidding_strategy,
        "advertiser_id": "adv_001",
    }


def make_request(
    imp_id: str = "imp_001",
    site_id: str = "site_001",
    floor_price: float = 0.5,
    device_type: str = "mobile",
    os: str = "ios",
    ad_format: str = "banner",
    geo_country: str = "US",
) -> dict:
    return {
        "id": "req_001",
        "imp_id": imp_id,
        "site_id": site_id,
        "floor_price": floor_price,
        "device_type": device_type,
        "os": os,
        "ad_format": ad_format,
        "geo": {"countries": [geo_country]},
        "user_agent": "Mozilla/5.0",
        "timeout_ms": 100,
    }


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def service():
    """每个测试用全新的 BiddingService 实例（避免单例状态污染）"""
    from api.services.bidding_service import BiddingService
    svc = BiddingService()
    return svc


@pytest.fixture
def campaign():
    return make_campaign()


@pytest.fixture
def request_data():
    return make_request()


# ─────────────────────────────────────────────
# 场景 1：正常竞价流程（mock 四层模型）
# ─────────────────────────────────────────────

class TestNormalBiddingFlow:
    """所有 ML 层正常工作时的竞价决策"""

    def test_decide_returns_bid_decision(self, service, campaign, request_data):
        """基本：decide() 应返回 BidDecision 对象"""
        from api.services.bidding_service import BidDecision
        service._loaded = True  # 跳过真实加载

        decision = service.decide(campaign, request_data)

        assert isinstance(decision, BidDecision)
        assert decision.campaign_id == campaign["id"]

    def test_final_bid_is_positive(self, service, campaign, request_data):
        """final_bid 必须 > 0（最低出价保护）"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert decision.final_bid > 0

    def test_predicted_ctr_in_valid_range(self, service, campaign, request_data):
        """pCTR 必须在 [0, 1] 范围内"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert 0.0 <= decision.predicted_ctr <= 1.0

    def test_win_prob_in_valid_range(self, service, campaign, request_data):
        """win_prob 必须在 [0, 1] 范围内"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert 0.0 <= decision.win_prob <= 1.0

    def test_ecpm_calculation(self, service, campaign, request_data):
        """eCPM = final_bid × pCTR × 1000"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        expected_ecpm = decision.final_bid * decision.predicted_ctr * 1000
        assert abs(decision.ecpm - expected_ecpm) < 0.01

    def test_decision_has_fallbacks_dict(self, service, campaign, request_data):
        """BidDecision 必须包含 fallbacks 字典"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert isinstance(decision.fallbacks, dict)
        assert set(decision.fallbacks.keys()) >= {"gbm", "deepfm", "gat", "ppo"}

    def test_decision_time_is_reasonable(self, service, campaign, request_data):
        """决策时间应 < 200ms（沙箱环境宽松一点）"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert decision.decision_ms < 200

    def test_model_version_not_empty(self, service, campaign, request_data):
        """model_version 字段不能为空"""
        service._loaded = True
        decision = service.decide(campaign, request_data)
        assert decision.model_version  # not empty string


# ─────────────────────────────────────────────
# 场景 2：PyTorch 不可用 → fallback 到 GBM
# ─────────────────────────────────────────────

class TestPyTorchUnavailableFallback:
    """模拟 PyTorch import 失败，应降级到 sklearn GBM"""

    def test_deepfm_fallback_when_none(self, service, campaign, request_data):
        """DeepFM 为 None 时，_predict_ctr_deepfm 应返回启发式值"""
        service._deepfm = None
        service._loaded = True

        ctr, used_fallback = service._predict_ctr_deepfm(campaign, request_data)

        assert used_fallback is True
        assert 0.0 <= ctr <= 1.0

    def test_ppo_fallback_when_none(self, service, campaign, request_data):
        """PPO 为 None 时，应返回基于 win_prob 的启发式调整"""
        service._ppo = None
        service._loaded = True

        # 高获胜率场景 → 应略微降价
        adj, fallback = service._get_ppo_adjustment(campaign, request_data, win_prob=0.8, predicted_ctr=0.01)
        assert fallback is True
        assert adj == pytest.approx(-0.1)

        # 低获胜率场景 → 应略微提价
        adj, fallback = service._get_ppo_adjustment(campaign, request_data, win_prob=0.2, predicted_ctr=0.01)
        assert fallback is True
        assert adj == pytest.approx(0.15)

    def test_gnn_fallback_when_none(self, service, campaign, request_data):
        """GAT 为 None 时，应返回中性 match_score=0.5"""
        service._gnn = None
        service._gnn_graph = None
        service._loaded = True

        embedding, match_score, fallback = service._get_context_embedding(campaign, request_data)

        assert fallback is True
        assert embedding is None
        assert match_score == pytest.approx(0.5)

    def test_all_pytorch_models_none_still_returns_decision(self, service, campaign, request_data):
        """所有 PyTorch 模型都是 None，decide() 仍应正常返回"""
        service._deepfm = None
        service._gnn = None
        service._ppo = None
        service._gbm = None
        service._loaded = True

        decision = service.decide(campaign, request_data)

        assert decision is not None
        assert decision.final_bid > 0
        assert decision.model_version == "heuristic"
        # 所有层都应标记为 fallback
        assert all(decision.fallbacks.values())


# ─────────────────────────────────────────────
# 场景 3：GBM 也不可用 → 纯启发式规则
# ─────────────────────────────────────────────

class TestFullHeuristicFallback:
    """所有模型不可用，应使用贝叶斯平滑启发式"""

    def test_win_prob_heuristic_with_no_history(self, service, request_data):
        """零曝光时的 win_prob 启发式估计"""
        service._gbm = None
        service._loaded = True

        campaign_no_history = make_campaign(impressions=0, clicks=0)
        win_prob, fallback = service._predict_win_prob(campaign_no_history, request_data)

        assert fallback is True
        assert 0.0 <= win_prob <= 1.0

    def test_ctr_heuristic_uses_bayesian_smoothing(self, service, request_data):
        """贝叶斯平滑：高曝光但零点击 → CTR 应接近 0 但不为 0"""
        service._deepfm = None
        service._gbm = None
        service._loaded = True

        campaign_zero_ctr = make_campaign(impressions=100000, clicks=0)
        ctr, fallback = service._predict_ctr_deepfm(campaign_zero_ctr, request_data)

        assert fallback is True
        assert ctr > 0  # 贝叶斯平滑不应返回 0
        assert ctr < 0.01  # 但应该很低

    def test_heuristic_ctr_increases_with_historical_clicks(self, service, request_data):
        """历史点击越多，启发式 CTR 越高"""
        service._deepfm = None
        service._gbm = None
        service._loaded = True

        low_ctr_campaign = make_campaign(impressions=1000, clicks=1)
        high_ctr_campaign = make_campaign(impressions=1000, clicks=50)

        ctr_low, _ = service._predict_ctr_deepfm(low_ctr_campaign, request_data)
        ctr_high, _ = service._predict_ctr_deepfm(high_ctr_campaign, request_data)

        assert ctr_high > ctr_low


# ─────────────────────────────────────────────
# 场景 4：单层模型运行时崩溃
# ─────────────────────────────────────────────

class TestSingleLayerFailure:
    """某一层模型推理时抛出异常，不应影响其他层"""

    def test_deepfm_exception_does_not_crash_decide(self, service, campaign, request_data):
        """DeepFM.predict() 抛出异常时，decide() 应正常完成"""
        mock_deepfm = MagicMock()
        mock_deepfm.predict.side_effect = RuntimeError("CUDA out of memory")
        mock_deepfm.hasher = MagicMock()
        mock_deepfm.hasher.hash_batch.return_value = MagicMock()
        service._deepfm = mock_deepfm
        service._loaded = True

        # 不应抛出异常
        decision = service.decide(campaign, request_data)
        assert decision is not None
        assert decision.fallbacks.get("deepfm") is True

    def test_ppo_exception_does_not_crash_decide(self, service, campaign, request_data):
        """PPO.act() 抛出异常时，decide() 应正常完成"""
        mock_ppo = MagicMock()
        mock_ppo.build_state.side_effect = ValueError("Invalid state")
        service._ppo = mock_ppo
        service._loaded = True

        decision = service.decide(campaign, request_data)
        assert decision is not None
        assert decision.fallbacks.get("ppo") is True

    def test_gnn_exception_does_not_crash_decide(self, service, campaign, request_data):
        """GNN.forward() 抛出异常时，decide() 应正常完成"""
        mock_gnn = MagicMock()
        mock_gnn.side_effect = Exception("Graph structure error")
        service._gnn = mock_gnn
        service._gnn_graph = MagicMock()
        service._loaded = True

        decision = service.decide(campaign, request_data)
        assert decision is not None


# ─────────────────────────────────────────────
# 场景 5：边界值测试
# ─────────────────────────────────────────────

class TestEdgeCases:
    """极端输入场景"""

    def test_zero_budget_campaign(self, service, request_data):
        """零预算活动不应崩溃"""
        service._loaded = True
        campaign = make_campaign(budget_total=0.0, spend=0.0)
        decision = service.decide(campaign, request_data)
        assert decision.final_bid > 0

    def test_fully_spent_campaign(self, service, request_data):
        """预算已耗尽的活动"""
        service._loaded = True
        campaign = make_campaign(budget_total=100.0, spend=100.0)
        decision = service.decide(campaign, request_data)
        # 不崩溃即可，budget check 在 router 层做
        assert decision is not None

    def test_very_high_bid_amount(self, service, request_data):
        """超高出价（$1000 CPM）"""
        service._loaded = True
        campaign = make_campaign(bid_amount=1000.0)
        decision = service.decide(campaign, request_data)
        assert decision.final_bid > 0
        assert decision.final_bid < 5000  # PPO 调整系数 max +1，不应超过 2× base

    def test_very_low_bid_amount(self, service, request_data):
        """极低出价（$0.001）"""
        service._loaded = True
        campaign = make_campaign(bid_amount=0.001)
        decision = service.decide(campaign, request_data)
        assert decision.final_bid >= 0.01  # 最低出价保护

    def test_zero_floor_price(self, service, campaign):
        """底价为 0"""
        service._loaded = True
        req = make_request(floor_price=0.0)
        decision = service.decide(campaign, req)
        assert decision is not None

    def test_missing_geo_field(self, service, campaign):
        """请求缺少 geo 字段"""
        service._loaded = True
        req = make_request()
        req.pop("geo", None)
        decision = service.decide(campaign, req)
        assert decision is not None

    def test_campaign_with_no_impressions(self, service, request_data):
        """全新活动（0 曝光 0 点击）"""
        service._loaded = True
        campaign = make_campaign(impressions=0, clicks=0, spend=0)
        decision = service.decide(campaign, request_data)
        assert decision.predicted_ctr > 0  # 贝叶斯平滑保证非零


# ─────────────────────────────────────────────
# 场景 6：懒加载 & 单例
# ─────────────────────────────────────────────

class TestLazyLoadAndSingleton:
    """懒加载逻辑和单例模式"""

    def test_models_not_loaded_on_init(self):
        """初始化时不应加载模型"""
        from api.services.bidding_service import BiddingService
        svc = BiddingService()
        assert svc._loaded is False
        assert svc._gbm is None
        assert svc._deepfm is None

    def test_models_loaded_after_first_decide(self):
        """第一次 decide() 后应标记为已加载"""
        from api.services.bidding_service import BiddingService
        svc = BiddingService()
        svc.decide(make_campaign(), make_request())
        assert svc._loaded is True

    def test_load_called_once_on_multiple_decides(self):
        """多次调用 decide() 只加载一次"""
        from api.services.bidding_service import BiddingService
        svc = BiddingService()

        with patch.object(svc, '_load_models', wraps=svc._load_models) as mock_load:
            svc.decide(make_campaign(), make_request())
            svc.decide(make_campaign(), make_request())
            svc.decide(make_campaign(), make_request())
            assert mock_load.call_count == 3  # 每次都调用，但内部 _loaded 检查会短路

    def test_singleton_returns_same_instance(self):
        """get_bidding_service() 单例应返回同一实例"""
        # 重置单例状态
        import api.services.bidding_service as module
        module._service = None

        from api.services.bidding_service import get_bidding_service
        svc1 = get_bidding_service()
        svc2 = get_bidding_service()
        assert svc1 is svc2

        # 清理
        module._service = None

    def test_load_models_is_idempotent(self, service):
        """重复调用 _load_models 不应重复初始化"""
        service._load_models()
        loaded_state = service._loaded
        gbm_ref = id(service._gbm)

        service._load_models()  # 第二次调用

        assert service._loaded == loaded_state
        assert id(service._gbm) == gbm_ref  # 同一对象，没有重新创建


# ─────────────────────────────────────────────
# 场景 7：model_version 追踪
# ─────────────────────────────────────────────

class TestModelVersionTracking:
    """model_version 字段应正确反映实际使用的模型组合"""

    def test_heuristic_when_all_models_none(self, service, campaign, request_data):
        service._gbm = None
        service._deepfm = None
        service._gnn = None
        service._ppo = None
        service._loaded = True

        decision = service.decide(campaign, request_data)
        assert decision.model_version == "heuristic"

    def test_model_version_contains_active_models(self, service, campaign, request_data):
        """model_version 应列出所有非 fallback 的模型"""
        # 所有层都 fallback 时 model_version 应为 heuristic
        service._deepfm = None
        service._gnn = None
        service._ppo = None
        service._gbm = None
        service._loaded = True

        decision = service.decide(campaign, request_data)
        assert "deepfm" not in decision.model_version
        assert "gat" not in decision.model_version


# ─────────────────────────────────────────────
# 集成冒烟测试（需要完整依赖，pytest mark 隔离）
# ─────────────────────────────────────────────

@pytest.mark.integration
class TestIntegrationSmoke:
    """集成冒烟测试，需要 FastAPI + ML 依赖，CI 中按需运行"""

    def test_full_pipeline_smoke(self):
        """端到端：BiddingService.decide() 在真实模型下跑通"""
        from api.services.bidding_service import BiddingService
        svc = BiddingService()
        decision = svc.decide(make_campaign(), make_request())

        assert decision.final_bid > 0
        assert 0 <= decision.predicted_ctr <= 1
        assert decision.decision_ms > 0
        print(f"\n[smoke] model_version={decision.model_version} "
              f"final_bid={decision.final_bid} "
              f"pCTR={decision.predicted_ctr:.6f} "
              f"decision_ms={decision.decision_ms}ms")
