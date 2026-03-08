"""
DeepFM CTR 预测模型

算法原理：
  DeepFM = FM（因子分解机）+ Deep（深层神经网络）
  - FM 层：捕获二阶特征交互（低阶）
  - Deep 层：通过 MLP 捕获高阶非线性特征交互
  - 共享 Embedding：FM 和 Deep 共用同一套特征 Embedding，参数高效

隐私保护设计：
  - 无用户 ID，仅使用匿名化特征哈希
  - 特征哈希使用单向 SHA-256，不可逆推原始值
  - 符合 GDPR / CCPA 隐私要求

参考论文：
  "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
  Guo et al., IJCAI 2017
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

@dataclass
class DeepFMConfig:
    num_fields: int = 10           # 特征域数量（每个域对应一个 embedding）
    vocab_size: int = 100_000      # 哈希特征词表大小（哈希空间）
    embed_dim: int = 16            # Embedding 维度
    hidden_dims: list = None       # Deep 层隐层维度
    dropout: float = 0.1           # Dropout 防过拟合
    learning_rate: float = 1e-3
    batch_size: int = 1024
    num_epochs: int = 10
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [400, 400, 400]


# ─────────────────────────────────────────────
# 隐私保护特征哈希
# ─────────────────────────────────────────────

class PrivacyPreservingHasher:
    """
    匿名特征哈希器

    将原始特征值映射到固定大小的哈希空间。
    使用 SHA-256 确保单向性（不可逆推原始 ID）。
    不存储任何用户 PII（Personally Identifiable Information）。
    """

    def __init__(self, vocab_size: int = 100_000, salt: str = "smb-adtech-2024"):
        self.vocab_size = vocab_size
        self.salt = salt  # 加盐防止彩虹表攻击

    def hash_feature(self, field: str, value: str) -> int:
        """
        对单个特征值做隐私保护哈希

        Args:
            field: 特征域名（如 "device_type", "geo_country"）
            value: 特征值（如 "mobile", "US"）

        Returns:
            哈希后的整数索引 [0, vocab_size)
        """
        raw = f"{self.salt}:{field}:{value}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return int(digest, 16) % self.vocab_size

    def hash_batch(self, features: list[dict[str, str]]) -> torch.Tensor:
        """
        批量哈希特征字典列表

        Returns:
            shape [batch_size, num_fields] 的整数张量
        """
        fields = sorted(features[0].keys())
        result = []
        for feat_dict in features:
            row = [self.hash_feature(f, str(feat_dict.get(f, ""))) for f in fields]
            result.append(row)
        return torch.tensor(result, dtype=torch.long)


# ─────────────────────────────────────────────
# FM 层（二阶特征交互）
# ─────────────────────────────────────────────

class FMLayer(nn.Module):
    """
    因子分解机（FM）二阶交互层

    公式：
        y_FM = Σ_{i<j} <v_i, v_j> x_i x_j
             = 0.5 * (||Σ v_i||² - Σ||v_i||²)

    复杂度从 O(n²) 降至 O(nk)，k 为 embedding 维度
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, num_fields, embed_dim]
        Returns:
            fm_out: [batch, 1]
        """
        # sum_square: (Σ v_i)²
        sum_of_embed = embeddings.sum(dim=1)          # [B, embed_dim]
        sum_of_square = sum_of_embed.pow(2)           # [B, embed_dim]

        # square_sum: Σ(v_i²)
        square_of_embed = embeddings.pow(2)           # [B, fields, embed_dim]
        square_of_sum = square_of_embed.sum(dim=1)    # [B, embed_dim]

        # FM 交互输出
        fm_interaction = 0.5 * (sum_of_square - square_of_sum)  # [B, embed_dim]
        return fm_interaction.sum(dim=1, keepdim=True)           # [B, 1]


# ─────────────────────────────────────────────
# Deep 层（高阶特征交互）
# ─────────────────────────────────────────────

class DeepLayer(nn.Module):
    """
    深层 MLP，捕获高阶特征交互

    输入：Flatten 后的 embedding 拼接向量
    输出：scalar 分数
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),   # BN 加速收敛，缓解梯度消失
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields * embed_dim]
        Returns:
            [batch, 1]
        """
        return self.net(x)


# ─────────────────────────────────────────────
# DeepFM 主模型
# ─────────────────────────────────────────────

class DeepFM(nn.Module):
    """
    DeepFM CTR 预测模型

    架构：
        Input (feature indices)
            ↓
        Shared Embedding Table
         ↙              ↘
      FM Layer        Deep Layer
      (2nd order)    (high-order)
         ↘              ↙
          Add + Sigmoid
            ↓
          pCTR [0,1]
    """

    def __init__(self, config: DeepFMConfig):
        super().__init__()
        self.config = config

        # 共享 Embedding 表（FM 和 Deep 复用）
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )
        # Xavier 初始化，避免梯度爆炸
        nn.init.xavier_uniform_(self.embedding.weight)

        # 一阶线性项（FM 偏置项）
        self.linear = nn.Embedding(config.vocab_size, 1, padding_idx=0)

        # FM 二阶交互层
        self.fm = FMLayer()

        # Deep MLP 层
        deep_input_dim = config.num_fields * config.embed_dim
        self.deep = DeepLayer(deep_input_dim, config.hidden_dims, config.dropout)

        # 全局偏置
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields] 特征哈希索引

        Returns:
            pCTR: [batch] 点击概率
        """
        # Embedding 查表: [B, fields, embed_dim]
        embed = self.embedding(x)

        # 一阶项: [B, 1]
        linear_out = self.linear(x).sum(dim=1)  # [B, 1]

        # FM 二阶交互: [B, 1]
        fm_out = self.fm(embed)

        # Deep 高阶: [B, 1]
        deep_input = embed.flatten(start_dim=1)  # [B, fields * embed_dim]
        deep_out = self.deep(deep_input)

        # 融合输出
        logit = linear_out + fm_out + deep_out + self.bias  # [B, 1]
        return torch.sigmoid(logit).squeeze(1)               # [B]


# ─────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────

class DeepFMTrainer:
    """
    DeepFM 训练 / 评估 / 预测封装

    使用 Binary Cross-Entropy Loss（BCELoss），因为 CTR 预测是二分类问题。
    """

    def __init__(self, config: DeepFMConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = DeepFM(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,  # L2 正则
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5
        )
        self.criterion = nn.BCELoss()
        self.hasher = PrivacyPreservingHasher(config.vocab_size)

    def train(
        self,
        X_train: torch.Tensor,  # [N, num_fields] 特征哈希索引
        y_train: torch.Tensor,  # [N] 0/1 标签
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> dict[str, list[float]]:
        """
        训练模型

        Returns:
            包含 train_loss 和 val_auc 历史的字典
        """
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history = {"train_loss": [], "val_auc": []}

        for epoch in range(self.config.num_epochs):
            # ── 训练阶段 ──
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.float().to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            # ── 验证阶段 ──
            val_auc = 0.0
            if X_val is not None and y_val is not None:
                val_auc = self.evaluate(X_val, y_val)
                self.scheduler.step(1 - val_auc)  # AUC 越高越好
                history["val_auc"].append(val_auc)

            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                  f"Loss={avg_loss:.4f} Val-AUC={val_auc:.4f}")

        return history

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """计算验证集 AUC"""
        from sklearn.metrics import roc_auc_score
        preds = self.predict(X)
        return roc_auc_score(y.numpy(), preds.numpy())

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        批量推理，返回 pCTR

        Args:
            X: [N, num_fields] 特征哈希索引

        Returns:
            pCTR: [N] 点击概率
        """
        self.model.eval()
        results = []
        loader = DataLoader(TensorDataset(X), batch_size=4096, shuffle=False)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                results.append(preds.cpu())
        return torch.cat(results)

    def save(self, path: str):
        torch.save({"model_state": self.model.state_dict(), "config": self.config}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> DeepFMTrainer:
        checkpoint = torch.load(path, map_location="cpu")
        trainer = cls(checkpoint["config"])
        trainer.model.load_state_dict(checkpoint["model_state"])
        return trainer


# ─────────────────────────────────────────────
# 快速演示
# ─────────────────────────────────────────────

def demo():
    """生成合成数据，训练并验证 DeepFM 模型"""
    import torch

    config = DeepFMConfig(
        num_fields=8,
        vocab_size=10_000,
        embed_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.1,
        num_epochs=3,
        batch_size=512,
    )

    # 合成数据：随机特征哈希索引 + 二值标签
    N = 5000
    X = torch.randint(0, config.vocab_size, (N, config.num_fields))
    y = torch.bernoulli(torch.full((N,), 0.05))  # ~5% 正样本（典型 CTR）

    trainer = DeepFMTrainer(config)
    history = trainer.train(X[:4000], y[:4000], X[4000:], y[4000:])

    preds = trainer.predict(X[4000:])
    print(f"✅ DeepFM demo done | pCTR range: [{preds.min():.4f}, {preds.max():.4f}]")
    return trainer


if __name__ == "__main__":
    demo()
