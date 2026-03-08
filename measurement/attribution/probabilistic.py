"""
概率归因框架（Probabilistic Attribution）

解决问题：
  多触点营销归因——用户在转化前经过多个广告触点，
  如何公平地将转化功劳分配给每个触点？

支持的归因模型：
  1. Last-Touch      — 最后触点 100% 归因（默认/行业基准）
  2. First-Touch     — 首次触点 100% 归因
  3. Linear          — 均等分配所有触点
  4. Time-Decay      — 时间衰减（近期触点权重更大）
  5. Position-Based  — U型归因（首尾各 40%，中间均分 20%）
  6. Markov Chain    — 马尔可夫链（概率数据驱动，移除效应）
  7. Shapley Value   — 博弈论公平归因（计算密集，适合离线批处理）

参考资料：
  - "Scalable and Accurate Deep Learning for Recommendation Systems"
  - Google Analytics 4 数据驱动归因白皮书
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, permutations
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class Touchpoint:
    """单个广告触点"""
    channel: str              # 渠道：sem/display/social/email/organic
    campaign_id: str
    timestamp_ms: int         # 触点时间戳（毫秒）
    ad_format: str = "banner"
    cost: float = 0.0         # 该触点的广告成本
    interaction_type: str = "impression"  # impression / click / view

    def __post_init__(self):
        if self.interaction_type not in ("impression", "click", "view"):
            raise ValueError(f"Invalid interaction_type: {self.interaction_type}")


@dataclass
class ConversionPath:
    """用户转化路径"""
    user_id: str
    touchpoints: list[Touchpoint]
    converted: bool = True
    conversion_value: float = 0.0     # 转化价值（例如订单金额）
    conversion_ts_ms: Optional[int] = None

    def __post_init__(self):
        # 按时间排序触点
        self.touchpoints = sorted(self.touchpoints, key=lambda t: t.timestamp_ms)

    @property
    def channels(self) -> list[str]:
        return [t.channel for t in self.touchpoints]

    @property
    def channel_set(self) -> frozenset[str]:
        return frozenset(self.channels)


@dataclass
class AttributionResult:
    """归因结果"""
    model: str
    credits: dict[str, float] = field(default_factory=dict)   # channel → 归因比例
    campaign_credits: dict[str, float] = field(default_factory=dict)  # campaign_id → 归因比例
    conversion_value: float = 0.0
    meta: dict = field(default_factory=dict)


class AttributionModel(str, Enum):
    LAST_TOUCH = "last_touch"
    FIRST_TOUCH = "first_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    MARKOV = "markov"
    SHAPLEY = "shapley"


# ─────────────────────────────────────────────
# 归因引擎
# ─────────────────────────────────────────────

class ProbabilisticAttributionEngine:
    """
    概率归因引擎

    使用方式：
        engine = ProbabilisticAttributionEngine()

        # 单路径归因
        path = ConversionPath(user_id="u1", touchpoints=[...])
        result = engine.attribute(path, model=AttributionModel.MARKOV)

        # 批量归因（Shapley / Markov 需要批量数据）
        results = engine.attribute_batch(paths, model=AttributionModel.SHAPLEY)
    """

    def __init__(self, half_life_hours: float = 24.0):
        """
        Args:
            half_life_hours: 时间衰减归因的半衰期（小时）
        """
        self.half_life_hours = half_life_hours
        self._markov_transitions: Optional[dict] = None   # 缓存马尔可夫转移矩阵

    # ──────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────

    def attribute(
        self, path: ConversionPath, model: AttributionModel = AttributionModel.LINEAR
    ) -> AttributionResult:
        """单路径归因（不需要历史数据的规则型模型）"""
        model_fn: dict[AttributionModel, Callable] = {
            AttributionModel.LAST_TOUCH: self._last_touch,
            AttributionModel.FIRST_TOUCH: self._first_touch,
            AttributionModel.LINEAR: self._linear,
            AttributionModel.TIME_DECAY: self._time_decay,
            AttributionModel.POSITION_BASED: self._position_based,
        }
        if model not in model_fn:
            raise ValueError(f"{model} requires batch data. Use attribute_batch() instead.")

        weights = model_fn[model](path)
        return self._build_result(model.value, path, weights)

    def attribute_batch(
        self, paths: list[ConversionPath], model: AttributionModel = AttributionModel.MARKOV
    ) -> list[AttributionResult]:
        """
        批量归因（数据驱动模型：Markov / Shapley）

        Args:
            paths: 所有用户转化路径（含未转化路径，用于计算移除效应）
            model: 归因模型

        Returns:
            每条路径的归因结果列表
        """
        if model == AttributionModel.MARKOV:
            return self._markov_batch(paths)
        elif model == AttributionModel.SHAPLEY:
            return self._shapley_batch(paths)
        else:
            # 规则型模型也支持批量
            return [self.attribute(p, model) for p in paths]

    # ──────────────────────────────────────────
    # 规则型归因模型
    # ──────────────────────────────────────────

    def _last_touch(self, path: ConversionPath) -> dict[str, float]:
        """最后触点归因 — 全部功劳给最后一个触点"""
        if not path.touchpoints:
            return {}
        last = path.touchpoints[-1]
        return {last.channel: 1.0}

    def _first_touch(self, path: ConversionPath) -> dict[str, float]:
        """首次触点归因 — 全部功劳给第一个触点"""
        if not path.touchpoints:
            return {}
        first = path.touchpoints[0]
        return {first.channel: 1.0}

    def _linear(self, path: ConversionPath) -> dict[str, float]:
        """线性均等归因 — 每个触点平分功劳"""
        if not path.touchpoints:
            return {}
        n = len(path.touchpoints)
        weights: dict[str, float] = defaultdict(float)
        for tp in path.touchpoints:
            weights[tp.channel] += 1.0 / n
        return dict(weights)

    def _time_decay(self, path: ConversionPath) -> dict[str, float]:
        """
        时间衰减归因

        权重公式：w_i = 2^(-Δt_i / half_life)
        Δt_i = 转化时间 - 触点时间
        """
        if not path.touchpoints:
            return {}

        conversion_ts = path.conversion_ts_ms or path.touchpoints[-1].timestamp_ms
        half_life_ms = self.half_life_hours * 3600 * 1000

        raw_weights = []
        for tp in path.touchpoints:
            delta_ms = max(conversion_ts - tp.timestamp_ms, 0)
            # 指数衰减：越近权重越大
            w = math.pow(2, -delta_ms / half_life_ms)
            raw_weights.append(w)

        total = sum(raw_weights)
        weights: dict[str, float] = defaultdict(float)
        for tp, w in zip(path.touchpoints, raw_weights):
            weights[tp.channel] += w / total

        return dict(weights)

    def _position_based(self, path: ConversionPath) -> dict[str, float]:
        """
        U型/位置归因

        分配规则：
        - 首触点 40%
        - 末触点 40%
        - 中间触点均分 20%
        """
        if not path.touchpoints:
            return {}

        n = len(path.touchpoints)
        weights: dict[str, float] = defaultdict(float)

        if n == 1:
            weights[path.touchpoints[0].channel] = 1.0
        elif n == 2:
            weights[path.touchpoints[0].channel] += 0.5
            weights[path.touchpoints[1].channel] += 0.5
        else:
            weights[path.touchpoints[0].channel] += 0.4
            weights[path.touchpoints[-1].channel] += 0.4
            middle_weight = 0.2 / (n - 2)
            for tp in path.touchpoints[1:-1]:
                weights[tp.channel] += middle_weight

        return dict(weights)

    # ──────────────────────────────────────────
    # 马尔可夫链归因（数据驱动）
    # ──────────────────────────────────────────

    def _build_transition_matrix(
        self, paths: list[ConversionPath]
    ) -> dict[str, dict[str, float]]:
        """
        构建渠道间转移概率矩阵

        状态空间：所有渠道 + START + CONVERSION + NULL（未转化）

        转移概率 P(channel_j | channel_i) = 从 i 到 j 的路径数 / 从 i 出发的总路径数
        """
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for path in paths:
            channels = ["START"] + path.channels
            terminal = "CONVERSION" if path.converted else "NULL"
            channels.append(terminal)

            for i in range(len(channels) - 1):
                counts[channels[i]][channels[i + 1]] += 1

        # 归一化为概率
        transitions: dict[str, dict[str, float]] = {}
        for state, next_states in counts.items():
            total = sum(next_states.values())
            transitions[state] = {s: c / total for s, c in next_states.items()}

        return transitions

    def _removal_effect(
        self,
        transitions: dict[str, dict[str, float]],
        channel: str,
        all_channels: list[str],
        max_steps: int = 50,
    ) -> float:
        """
        计算移除某渠道后的转化率（用于马尔可夫移除效应计算）

        模拟移除 channel 后，重新计算从 START 到 CONVERSION 的到达概率
        """
        # 深拷贝转移矩阵并移除目标渠道
        modified: dict[str, dict[str, float]] = {}
        for state, nexts in transitions.items():
            if state == channel:
                continue  # 移除该状态
            new_nexts = {}
            total = 0.0
            for next_state, prob in nexts.items():
                if next_state != channel:
                    new_nexts[next_state] = prob
                    total += prob
            # 重新归一化（被移除的概率分散到 NULL）
            if total > 0:
                modified[state] = {s: p / total for s, p in new_nexts.items()}
            else:
                modified[state] = {"NULL": 1.0}

        # 蒙特卡洛模拟转化率
        n_sim = 10000
        conversions = 0
        for _ in range(n_sim):
            state = "START"
            for _ in range(max_steps):
                if state in ("CONVERSION", "NULL"):
                    break
                nexts = modified.get(state, {"NULL": 1.0})
                state = np.random.choice(list(nexts.keys()), p=list(nexts.values()))
            if state == "CONVERSION":
                conversions += 1

        return conversions / n_sim

    def _markov_batch(self, paths: list[ConversionPath]) -> list[AttributionResult]:
        """
        马尔可夫链批量归因

        算法：
        1. 构建转移概率矩阵
        2. 计算整体转化率 R_all
        3. 对每个渠道计算移除后转化率 R_{-i}
        4. 移除效应 = R_all - R_{-i}
        5. 归一化为归因比例
        """
        logger.info(f"Running Markov attribution on {len(paths)} paths")

        transitions = self._build_transition_matrix(paths)
        all_channels = list(set(
            ch for p in paths for ch in p.channels
        ))

        # 计算基准转化率
        baseline_cvr = self._removal_effect(transitions, "__none__", all_channels)

        # 计算每个渠道的移除效应
        removal_effects: dict[str, float] = {}
        for channel in all_channels:
            removed_cvr = self._removal_effect(transitions, channel, all_channels)
            removal_effects[channel] = max(baseline_cvr - removed_cvr, 0)

        # 归一化
        total_effect = sum(removal_effects.values()) or 1.0
        channel_credits = {ch: eff / total_effect for ch, eff in removal_effects.items()}

        logger.info(f"Markov credits: {channel_credits}")

        # 为每条路径生成归因结果
        results = []
        for path in paths:
            if not path.converted:
                results.append(AttributionResult(model="markov", credits={}))
                continue

            # 按路径实际触点过滤
            path_channels = set(path.channels)
            path_credits = {ch: v for ch, v in channel_credits.items() if ch in path_channels}
            path_total = sum(path_credits.values()) or 1.0
            normalized = {ch: v / path_total for ch, v in path_credits.items()}
            results.append(self._build_result("markov", path, normalized))

        return results

    # ──────────────────────────────────────────
    # Shapley Value 归因（博弈论）
    # ──────────────────────────────────────────

    def _shapley_batch(self, paths: list[ConversionPath]) -> list[AttributionResult]:
        """
        Shapley Value 归因

        原理：将每个渠道的贡献视为博弈论中联盟游戏的边际贡献期望值

        对于渠道 i：
            φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S∪{i}) - v(S)]

        v(S) = 包含渠道集合 S 的路径的转化率

        ⚠️ 计算复杂度 O(2^n)，n>10 时需近似（采样 Shapley）
        """
        # 统计各渠道组合的转化率
        coalition_cvr: dict[frozenset, float] = {}
        total_paths = len(paths)
        converted = [p for p in paths if p.converted]

        all_channels = sorted(set(ch for p in paths for ch in p.channels))
        n = len(all_channels)

        logger.info(f"Computing Shapley values for {n} channels, {total_paths} paths")

        if n > 12:
            logger.warning(f"n={n} channels is large, Shapley will be approximate (sampled)")
            return self._approximate_shapley(paths, all_channels)

        # 枚举所有子集
        for size in range(n + 1):
            for subset in combinations(all_channels, size):
                s = frozenset(subset)
                # 包含子集所有渠道的路径中，转化率 = 转化数/总数
                subset_paths = [p for p in paths if s.issubset(p.channel_set)]
                if not subset_paths:
                    coalition_cvr[s] = 0.0
                else:
                    coalition_cvr[s] = sum(1 for p in subset_paths if p.converted) / len(subset_paths)

        # 计算每个渠道的 Shapley 值
        shapley: dict[str, float] = {}
        grand_coalition = frozenset(all_channels)

        for i, channel in enumerate(all_channels):
            phi = 0.0
            others = [c for c in all_channels if c != channel]

            for size in range(n):
                for subset in combinations(others, size):
                    s = frozenset(subset)
                    s_with_i = s | {channel}
                    weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    marginal = coalition_cvr.get(s_with_i, 0) - coalition_cvr.get(s, 0)
                    phi += weight * marginal

            shapley[channel] = max(phi, 0)  # 修正负值

        # 归一化
        total = sum(shapley.values()) or 1.0
        channel_credits = {ch: v / total for ch, v in shapley.items()}

        results = []
        for path in paths:
            if not path.converted:
                results.append(AttributionResult(model="shapley", credits={}))
                continue
            path_channels = set(path.channels)
            path_credits = {ch: v for ch, v in channel_credits.items() if ch in path_channels}
            path_total = sum(path_credits.values()) or 1.0
            normalized = {ch: v / path_total for ch, v in path_credits.items()}
            results.append(self._build_result("shapley", path, normalized))

        return results

    def _approximate_shapley(
        self, paths: list[ConversionPath], all_channels: list[str], n_samples: int = 1000
    ) -> list[AttributionResult]:
        """采样近似 Shapley（用于渠道数 > 12 的场景）"""
        n = len(all_channels)
        shapley: dict[str, float] = defaultdict(float)

        # 计算联盟价值的快速版本（直接查路径匹配）
        def coalition_value(subset: frozenset) -> float:
            matching = [p for p in paths if subset.issubset(p.channel_set)]
            if not matching:
                return 0.0
            return sum(1 for p in matching if p.converted) / len(matching)

        # 蒙特卡洛采样随机排列
        for _ in range(n_samples):
            perm = list(np.random.permutation(all_channels))
            current_set: frozenset = frozenset()
            v_current = 0.0
            for channel in perm:
                new_set = current_set | {channel}
                v_new = coalition_value(new_set)
                shapley[channel] += v_new - v_current
                current_set = new_set
                v_current = v_new

        # 归一化（shapley[ch] 是 n_samples 次边际贡献之和，先平均再归一化）
        total = sum(max(v, 0) for v in shapley.values()) or 1.0
        channel_credits = {ch: max(v, 0) / total for ch, v in shapley.items()}

        return [
            self._build_result("shapley_approx", p, {
                ch: v for ch, v in channel_credits.items() if ch in set(p.channels)
            }) for p in paths
        ]

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def _build_result(
        self, model_name: str, path: ConversionPath, channel_weights: dict[str, float]
    ) -> AttributionResult:
        """将渠道权重映射到具体活动 ID"""
        # 渠道 → campaign 的映射（同一渠道可能有多个活动，按时间取最后一个）
        channel_to_campaign: dict[str, str] = {}
        for tp in path.touchpoints:
            channel_to_campaign[tp.channel] = tp.campaign_id

        campaign_credits: dict[str, float] = {}
        for channel, weight in channel_weights.items():
            cid = channel_to_campaign.get(channel, "unknown")
            campaign_credits[cid] = campaign_credits.get(cid, 0) + weight

        return AttributionResult(
            model=model_name,
            credits=channel_weights,
            campaign_credits=campaign_credits,
            conversion_value=path.conversion_value,
            meta={"user_id": path.user_id, "n_touchpoints": len(path.touchpoints)},
        )


# ─────────────────────────────────────────────
# 快速演示
# ─────────────────────────────────────────────

def demo():
    """演示各归因模型效果"""
    import random
    rng = random.Random(42)

    channels = ["sem", "display", "social", "email", "organic"]

    def make_path(converted=True, length=3):
        tps = [
            Touchpoint(
                channel=rng.choice(channels),
                campaign_id=f"camp_{rng.randint(1,5)}",
                timestamp_ms=1_700_000_000_000 + i * 3_600_000,
                cost=rng.uniform(0.01, 2.0),
                interaction_type=rng.choice(["impression", "click"]),
            )
            for i in range(length)
        ]
        return ConversionPath(
            user_id=f"user_{rng.randint(1,100)}",
            touchpoints=tps,
            converted=converted,
            conversion_value=rng.uniform(10, 500) if converted else 0,
            conversion_ts_ms=tps[-1].timestamp_ms + 1_800_000,
        )

    paths = [make_path(converted=True, length=rng.randint(1,5)) for _ in range(200)]
    paths += [make_path(converted=False, length=rng.randint(1,3)) for _ in range(800)]

    engine = ProbabilisticAttributionEngine(half_life_hours=12.0)

    # 规则型模型（单路径）
    for model in [AttributionModel.LAST_TOUCH, AttributionModel.LINEAR,
                  AttributionModel.TIME_DECAY, AttributionModel.POSITION_BASED]:
        result = engine.attribute(paths[0], model=model)
        print(f"{model.value:20s}: {result.credits}")

    # 数据驱动模型（批量）
    markov_results = engine.attribute_batch(paths, model=AttributionModel.MARKOV)
    converted_results = [r for r, p in zip(markov_results, paths) if p.converted]
    if converted_results:
        print(f"\nMarkov (sample): {converted_results[0].credits}")

    print("\n✅ Attribution demo complete")


if __name__ == "__main__":
    demo()
