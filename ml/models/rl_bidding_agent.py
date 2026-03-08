"""
PPO 强化学习竞价 Agent

算法原理：
  PPO (Proximal Policy Optimization) 是当前最流行的策略梯度算法之一。
  核心思想：限制策略更新幅度，避免单步更新过大导致训练不稳定。

  目标函数：
    L_CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
    其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (重要性采样比)

  完整损失：
    L(θ) = -L_CLIP + c₁ L_VF - c₂ H[π_θ]
    - L_VF：值函数误差（Critic 损失）
    - H[π_θ]：策略熵奖励（鼓励探索）

竞价场景建模：
  State:  活动实时指标 + 市场状态 + 时间特征
  Action: 竞价调整系数 δ ∈ [-1, 1]（连续动作空间）
          实际出价 = base_bid × (1 + δ)
  Reward: 基于预算效率和 ROI 的复合奖励

参考论文：
  "Proximal Policy Optimization Algorithms" - Schulman et al., 2017
  "Budget Constrained Bidding by Model-free Reinforcement Learning" - Wu et al., 2018
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

@dataclass
class PPOConfig:
    # 网络结构
    state_dim: int = 20          # 状态空间维度
    action_dim: int = 1          # 动作维度（竞价调整系数）
    hidden_dim: int = 256        # 隐层维度

    # PPO 超参数
    clip_epsilon: float = 0.2    # PPO clip 范围 ε
    gamma: float = 0.99          # 折扣因子
    gae_lambda: float = 0.95     # GAE λ（广义优势估计）
    entropy_coef: float = 0.01   # 熵奖励系数 c₂（鼓励探索）
    value_loss_coef: float = 0.5 # 值函数损失系数 c₁
    max_grad_norm: float = 0.5   # 梯度裁剪

    # 训练参数
    lr: float = 3e-4
    num_epochs: int = 10         # 每次 rollout 后的 PPO 更新轮数
    batch_size: int = 64
    rollout_steps: int = 2048    # 每次收集的步数

    # 动作空间
    action_low: float = -1.0     # 最低调整系数（-100% 出价）
    action_high: float = 1.0     # 最高调整系数（+100% 出价）
    log_std_init: float = -0.5   # 初始策略标准差（log 空间）
    log_std_min: float = -4.0    # 标准差下限（防止策略过于确定性）
    log_std_max: float = 0.5     # 标准差上限（防止探索过多）


# ─────────────────────────────────────────────
# Actor-Critic 网络
# ─────────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    策略网络（Actor）

    输出高斯分布的均值 μ 和标准差 σ：
      π(a|s) = N(μ(s), σ²(s))

    连续动作空间设计：
    - 输出 tanh(μ) 保证均值在 [-1, 1]
    - σ 通过 log_std 参数化，限制在合理范围
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config

        # 共享特征提取网络
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # 均值输出层
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # 将均值限制在 [-1, 1]
        )

        # 状态无关的对数标准差（可学习参数）
        # 训练初期较大（探索），随训练收敛逐渐减小
        self.log_std = nn.Parameter(
            torch.full((action_dim,), config.log_std_init)
        )

        self._init_weights()

    def _init_weights(self):
        """正交初始化，对 RL 训练稳定性有显著帮助"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        # 输出层用更小的增益（输出范围小）
        nn.init.orthogonal_(self.mu_head[-2].weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [B, state_dim]
        Returns:
            mu: [B, action_dim] 动作均值
            std: [B, action_dim] 动作标准差
        """
        features = self.backbone(state)
        mu = self.mu_head(features)

        # 裁剪 log_std，防止策略退化
        log_std = self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        std = log_std.exp().expand_as(mu)

        return mu, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """获取动作分布"""
        mu, std = self(state)
        return Normal(mu, std)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            state: [B, state_dim]
            deterministic: True=使用均值（推理期间），False=采样（训练期间）

        Returns:
            action: [B, action_dim] 裁剪后的动作
            log_prob: [B] 动作对数概率（用于重要性采样）
        """
        dist = self.get_distribution(state)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # 重参数化采样（梯度可传播）

        # 裁剪到动作范围
        action = action.clamp(self.config.action_low, self.config.action_high)

        # 计算 log prob（裁剪后需要修正）
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob


class CriticNetwork(nn.Module):
    """
    价值网络（Critic）

    估计状态价值函数 V(s)，用于计算优势函数 A(s,a) = Q(s,a) - V(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # 正交初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            value: [B, 1] 状态价值估计
        """
        return self.net(state)


# ─────────────────────────────────────────────
# 经验缓冲区（Rollout Buffer）
# ─────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    收集 trajectory 数据用于 PPO 更新

    存储 T 步的 (s, a, r, done, log_prob, value) 数据
    """
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "states": torch.stack(self.states),
            "actions": torch.stack(self.actions),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.stack(self.values).squeeze(-1),
        }


# ─────────────────────────────────────────────
# PPO Agent 主体
# ─────────────────────────────────────────────

class PPOBiddingAgent:
    """
    PPO 强化学习竞价 Agent

    状态空间（20维）：
      活动指标 (6)：spend_ratio, ctr, cvr, budget_utilization, impressions_log, clicks_log
      市场状态 (6)：avg_market_cpm, win_rate, floor_price, competition_level, time_pressure, supply_index
      时间特征 (4)：hour_sin, hour_cos, dow_sin, dow_cos
      历史动作 (4)：上4步的竞价调整系数（趋势感知）

    动作空间（1维，连续）：
      δ ∈ [-1, 1]，实际出价 = base_bid × (1 + δ)

    奖励函数：
      r = α × ROI_bonus - β × overspend_penalty + γ × win_bonus
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cpu")

        # Actor + Critic 网络
        self.actor = ActorNetwork(config.state_dim, config.action_dim, config.hidden_dim, config)
        self.critic = CriticNetwork(config.state_dim, config.hidden_dim)

        # 共用优化器（联合训练）
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.lr,
            eps=1e-5,
        )

        # 学习率线性退火
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

        self.buffer = RolloutBuffer()
        self.training_step = 0

        # 历史动作队列（状态特征的一部分）
        self._action_history = deque([0.0] * 4, maxlen=4)

    # ──────────────────────────────────────────
    # 交互接口
    # ──────────────────────────────────────────

    def build_state(self, campaign_metrics: dict, market_state: dict) -> torch.Tensor:
        """
        从活动指标和市场状态构建状态向量

        Args:
            campaign_metrics: 活动实时数据
            market_state: 市场行情数据

        Returns:
            state: [state_dim] 归一化状态向量
        """
        import math, time
        ts = time.time()
        hour = (ts % 86400) / 3600
        dow = (ts // 86400) % 7

        features = [
            # 活动指标（归一化到 [0, 1] 或 [-1, 1]）
            float(campaign_metrics.get("spend_ratio", 0)),          # 已消耗/总预算
            float(campaign_metrics.get("ctr", 0)) * 100,            # CTR × 100
            float(campaign_metrics.get("cvr", 0)) * 100,            # CVR × 100
            float(campaign_metrics.get("budget_utilization", 0)),   # 预算利用率
            math.log1p(float(campaign_metrics.get("impressions", 0))) / 15,
            math.log1p(float(campaign_metrics.get("clicks", 0))) / 10,

            # 市场状态
            float(market_state.get("avg_market_cpm", 1)) / 10,
            float(market_state.get("win_rate", 0.5)),
            float(market_state.get("floor_price", 0.01)) / 5,
            float(market_state.get("competition_level", 0.5)),
            float(market_state.get("time_pressure", 0)),            # 投放时间紧迫度
            float(market_state.get("supply_index", 1)) / 2,

            # 时间特征（周期编码）
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),

            # 历史动作（趋势感知）
            *list(self._action_history),
        ]

        assert len(features) == self.config.state_dim, \
            f"State dim mismatch: {len(features)} vs {self.config.state_dim}"

        return torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[float, float, float]:
        """
        根据当前状态选择竞价调整系数

        Returns:
            action: 竞价调整系数 δ ∈ [-1, 1]
            log_prob: 动作对数概率
            value: 状态价值估计
        """
        self.actor.eval()
        self.critic.eval()

        state_batch = state.unsqueeze(0)  # [1, state_dim]
        action, log_prob = self.actor.get_action(state_batch, deterministic)
        value = self.critic(state_batch)

        action_val = action.squeeze().item()
        self._action_history.append(action_val)

        return action_val, log_prob.squeeze().item(), value.squeeze().item()

    def store_transition(
        self,
        state: torch.Tensor,
        action: float,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """将一步交互数据存入缓冲区"""
        self.buffer.add(
            state,
            torch.tensor([action], dtype=torch.float32),
            reward,
            done,
            torch.tensor([log_prob], dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32),
        )

    # ──────────────────────────────────────────
    # GAE 优势估计
    # ──────────────────────────────────────────

    def compute_gae(
        self,
        rewards: torch.Tensor,   # [T]
        values: torch.Tensor,    # [T]
        dones: torch.Tensor,     # [T]
        last_value: float,       # V(s_{T+1})
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        广义优势估计（GAE）

        A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD 误差)

        γ：折扣因子（远期奖励折现）
        λ：平衡偏差-方差 (λ=0 → TD, λ=1 → Monte Carlo)

        Returns:
            advantages: [T] 优势函数
            returns: [T] 折扣回报（用于 Critic 训练目标）
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0

        # 反向遍历计算 GAE
        next_value = last_value
        for t in reversed(range(T)):
            # TD 误差
            next_non_terminal = 1.0 - dones[t].item()
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]

            # GAE 递推
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = values[t].item()

        # 优势归一化（减少梯度方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values  # V(s) + A(s,a) ≈ Q(s,a)

        return advantages, returns

    # ──────────────────────────────────────────
    # PPO 更新
    # ──────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> dict[str, float]:
        """
        PPO 更新步骤

        1. 计算 GAE 优势
        2. 多轮 minibatch 更新（复用 rollout 数据，提高数据效率）
        3. CLIP 目标函数限制策略偏移
        4. 更新 Actor + Critic

        Returns:
            训练指标字典
        """
        if len(self.buffer) == 0:
            return {}

        data = self.buffer.to_tensors()
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"].squeeze(-1)
        rewards = data["rewards"]
        dones = data["dones"]
        values = data["values"]

        # 计算 GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # 多轮 PPO 更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        self.actor.train()
        self.critic.train()

        T = states.shape[0]
        for _ in range(self.config.num_epochs):
            # 随机打乱 minibatch
            perm = torch.randperm(T)

            for start in range(0, T, self.config.batch_size):
                idx = perm[start:start + self.config.batch_size]
                if len(idx) < 4:  # 跳过过小的 batch
                    continue

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # ── 计算新策略的 log prob ──
                dist = self.actor.get_distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)  # [B]
                entropy = dist.entropy().sum(dim=-1).mean()               # 策略熵

                # ── 重要性采样比 r_t(θ) ──
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = log_ratio.exp()  # π_new / π_old

                # 监控 KL 散度（用于早停）
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                # ── PPO CLIP 策略损失 ──
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── 价值函数损失（Huber Loss，对 outlier 更鲁棒）──
                new_values = self.critic(batch_states).squeeze(-1)
                value_loss = F.huber_loss(new_values, batch_returns, delta=1.0)

                # ── 总损失 ──
                loss = (policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_frac += clip_frac
                num_updates += 1

        self.buffer.clear()
        self.scheduler.step()
        self.training_step += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "clip_fraction": total_clip_frac / n,
            "training_step": self.training_step,
        }

    # ──────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
        }, path)
        print(f"PPO Agent saved to {path} (step={self.training_step})")

    @classmethod
    def load(cls, path: str) -> PPOBiddingAgent:
        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(checkpoint["config"])
        agent.actor.load_state_dict(checkpoint["actor_state"])
        agent.critic.load_state_dict(checkpoint["critic_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.training_step = checkpoint["training_step"]
        return agent


# ─────────────────────────────────────────────
# 模拟竞价环境（用于训练验证）
# ─────────────────────────────────────────────

class BiddingEnvironment:
    """
    简化竞价环境，用于 PPO Agent 训练

    奖励设计：
      win_reward = 竞价成功时的 CTR × 转化价值
      spend_penalty = 超出日预算时的惩罚
      roi_bonus = 当 ROI > 目标时的奖励
    """

    def __init__(self, config: PPOConfig, base_bid: float = 1.0, daily_budget: float = 100.0):
        self.config = config
        self.base_bid = base_bid
        self.daily_budget = daily_budget
        # 在环境中维护历史动作队列，避免每步实例化 Agent
        self._action_history: deque = deque([0.0] * 4, maxlen=4)
        self.reset()

    def reset(self) -> torch.Tensor:
        self.spend = 0.0
        self.clicks = 0
        self.impressions = 0
        self.step_count = 0
        self._action_history = deque([0.0] * 4, maxlen=4)
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        """直接构建状态向量，不创建 Agent 实例（避免每步实例化神经网络）"""
        import math, random, time
        ts = time.time()
        hour = (ts % 86400) / 3600
        dow = (ts // 86400) % 7
        features = [
            self.spend / max(self.daily_budget, 1),
            (self.clicks / max(self.impressions, 1)) * 100,
            random.uniform(0.01, 0.05) * 100,
            self.spend / max(self.daily_budget, 1),
            math.log1p(self.impressions) / 15,
            math.log1p(self.clicks) / 10,
            random.uniform(0.5, 3.0) / 10,
            random.uniform(0.1, 0.9),
            random.uniform(0.01, 1.0) / 5,
            random.uniform(0, 1),
            self.step_count / 100,
            random.uniform(0.5, 2.0) / 2,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            *list(self._action_history),
        ]
        return torch.tensor(features, dtype=torch.float32)

    def step(self, action: float) -> tuple[torch.Tensor, float, bool]:
        """
        执行竞价动作，返回 (next_state, reward, done)
        """
        import random
        self._action_history.append(action)  # 记录动作历史
        actual_bid = self.base_bid * (1 + action)
        market_price = random.uniform(0.1, 2.0)

        # 竞价结果
        win = actual_bid > market_price
        reward = 0.0

        if win:
            cost = market_price  # 二价竞价，支付市场价
            self.spend += cost / 1000  # CPM → 每次展示成本
            self.impressions += 1

            # 点击概率（简化模型）
            base_ctr = 0.02
            if random.random() < base_ctr:
                self.clicks += 1
                reward += 1.0  # 点击奖励

            reward += 0.1  # 展示奖励

        # 超预算惩罚
        if self.spend > self.daily_budget:
            reward -= 5.0

        # ROI 奖励（花费效率高时正向激励）
        if self.spend > 0:
            roi = (self.clicks * 5) / self.spend  # 假设每次点击价值 $5
            if roi > 2.0:
                reward += 0.5

        self.step_count += 1
        done = self.step_count >= 200 or self.spend >= self.daily_budget

        return self._get_state(), reward, done


# ─────────────────────────────────────────────
# 演示训练
# ─────────────────────────────────────────────

def demo():
    config = PPOConfig(
        state_dim=20,
        action_dim=1,
        hidden_dim=128,
        rollout_steps=256,
        num_epochs=4,
        batch_size=64,
    )

    agent = PPOBiddingAgent(config)
    env = BiddingEnvironment(config)

    print("Training PPO Bidding Agent...")
    total_rewards = []

    for episode in range(5):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # 采样动作
            action, log_prob, value = agent.act(state)

            # 环境交互
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # 存储
            agent.store_transition(state, action, reward, done, log_prob, value)
            state = next_state

            # 到达 rollout 步数时更新
            if len(agent.buffer) >= config.rollout_steps:
                _, _, last_val = agent.act(state)
                metrics = agent.update(last_value=last_val)

        # episode 结束时也更新
        if len(agent.buffer) > 0:
            metrics = agent.update()

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} | Reward={episode_reward:.2f} | "
              f"Spend=${env.spend:.2f} | Clicks={env.clicks} | "
              f"PolicyLoss={metrics.get('policy_loss', 0):.4f}")

    print(f"\n✅ PPO demo done | Avg reward: {sum(total_rewards)/len(total_rewards):.2f}")
    return agent


if __name__ == "__main__":
    demo()
