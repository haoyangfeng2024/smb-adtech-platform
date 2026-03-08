"""
Graph Attention Network (GAT) 广告反馈预测模型

算法原理：
  将广告生态系统建模为异构图：
  - 节点类型：广告位（Ad Slot）/ 内容类别（Category）/ 时间上下文（Time Context）
  - 边类型：共现关系（Co-occurrence）/ 行为相似度（Behavioral Similarity）

  GAT 在图卷积基础上引入注意力机制：
    h_i^{(l+1)} = σ(Σ_{j∈N(i)} α_{ij} W h_j^{(l)})
    α_{ij} = softmax(LeakyReLU(a^T [W h_i || W h_j]))

  多头注意力（Multi-Head Attention）：
    h_i = ||_{k=1}^{K} σ(Σ α_{ij}^k W^k h_j)

隐私保护设计：
  - 节点特征使用匿名化上下文特征（无用户 ID / PII）
  - 边权重基于内容共现和行为模式，不存储个人轨迹
  - 满足 k-匿名性要求

参考论文：
  "Graph Attention Networks" - Veličković et al., ICLR 2018
  "Heterogeneous Graph Neural Network" - Wang et al., KDD 2019
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 图数据结构
# ─────────────────────────────────────────────

class NodeType:
    AD_SLOT = 0      # 广告位节点：位置、尺寸、页面类型
    CATEGORY = 1     # 内容类别节点：新闻/体育/科技等
    TIME_CTX = 2     # 时间上下文节点：时段/星期/季节


@dataclass
class AdGraph:
    """
    广告生态系统图

    Attributes:
        node_features:  [N, feat_dim] 所有节点的特征向量（匿名化）
        node_types:     [N] 节点类型（AD_SLOT / CATEGORY / TIME_CTX）
        edge_index:     [2, E] 边的端点索引（COO 格式，源 → 目标）
        edge_weights:   [E] 边权重（共现频率 / 行为相似度）
        edge_types:     [E] 边类型（0=共现, 1=行为相似）
        num_nodes:      总节点数
    """
    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_index: torch.Tensor
    edge_weights: Optional[torch.Tensor] = None
    edge_types: Optional[torch.Tensor] = None

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


# ─────────────────────────────────────────────
# 图注意力层（单头）
# ─────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    单头图注意力层 (GAT Layer)

    核心计算：
      1. 线性变换：h' = W·h （提升表达能力）
      2. 注意力系数：e_ij = LeakyReLU(a^T [h'_i || h'_j])
      3. 归一化：α_ij = softmax_j(e_ij)  （只对邻居归一化）
      4. 聚合：h_i^new = σ(Σ_{j∈N(i)} α_ij · h'_j)

    Edge-weighted attention：
      将边权重融入注意力分数，强化高共现节点的影响
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
        use_edge_weights: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_weights = use_edge_weights

        # 节点特征线性变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # 注意力向量 a = [a_i || a_j]，维度 2*out_dim
        self.attention = nn.Linear(2 * out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)

        # Glorot 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(
        self,
        h: torch.Tensor,            # [N, in_dim] 节点特征
        edge_index: torch.Tensor,   # [2, E] 边索引
        edge_weights: Optional[torch.Tensor] = None,  # [E] 边权重
    ) -> torch.Tensor:
        """
        Returns:
            h_new: [N, out_dim] 更新后的节点特征
        """
        N = h.shape[0]

        # 步骤 1：线性变换
        h_prime = self.W(h)  # [N, out_dim]
        h_prime = self.dropout(h_prime)

        src, dst = edge_index[0], edge_index[1]  # 各 [E]

        # 步骤 2：计算注意力系数
        # 拼接源节点和目标节点特征
        h_concat = torch.cat([h_prime[src], h_prime[dst]], dim=1)  # [E, 2*out_dim]
        e = self.leaky_relu(self.attention(h_concat)).squeeze(-1)   # [E]

        # 融入边权重（log scale 防止极值）
        if self.use_edge_weights and edge_weights is not None:
            e = e + torch.log1p(edge_weights.clamp(min=0))

        # 步骤 3：按目标节点做 softmax（邻域归一化）
        alpha = self._softmax_by_node(e, dst, N)   # [E]
        alpha = self.dropout(alpha)

        # 步骤 4：加权聚合
        # scatter_add: 将邻居贡献累加到目标节点
        weighted = alpha.unsqueeze(-1) * h_prime[src]   # [E, out_dim]
        h_new = torch.zeros(N, self.out_dim, device=h.device)
        h_new.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)

        return F.elu(h_new)  # ELU 激活

    @staticmethod
    def _softmax_by_node(e: torch.Tensor, dst: torch.Tensor, N: int) -> torch.Tensor:
        """
        对每个目标节点的所有入边做 softmax（邻域内归一化）

        实现 sparse softmax（不能用标准 softmax，因为不同节点的邻居数不同）
        """
        # 数值稳定：减去每个节点的最大值
        e_max = torch.zeros(N, device=e.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        e_shifted = e - e_max[dst]

        exp_e = torch.exp(e_shifted)
        sum_exp = torch.zeros(N, device=e.device)
        sum_exp.scatter_add_(0, dst, exp_e)

        return exp_e / (sum_exp[dst] + 1e-8)


# ─────────────────────────────────────────────
# 多头图注意力层
# ─────────────────────────────────────────────

class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层

    K 个独立注意力头并行计算，再拼接（最后一层用平均代替拼接）：
      h_i = ||_{k=1}^K σ(Σ α_{ij}^k W^k h_j)    （中间层）
      h_i = σ(1/K Σ_{k=1}^K Σ α_{ij}^k W^k h_j)  （最后层）

    多头的优势：
    - 允许模型同时关注不同子空间的特征（类比 Transformer）
    - 稳定训练，降低方差
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,   # True=拼接（中间层）, False=平均（最后层）
    ):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim, dropout)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        head_outputs = [head(h, edge_index, edge_weights) for head in self.heads]

        if self.concat:
            return torch.cat(head_outputs, dim=-1)  # [N, num_heads * out_dim]
        else:
            return torch.stack(head_outputs, dim=0).mean(dim=0)  # [N, out_dim]


# ─────────────────────────────────────────────
# 异构节点编码器
# ─────────────────────────────────────────────

class HeterogeneousNodeEncoder(nn.Module):
    """
    异构节点类型编码器

    不同类型的节点（广告位/类别/时间）有不同的特征空间，
    先用各自的 MLP 投影到统一隐层空间，再送入 GAT。
    """

    def __init__(self, feat_dims: dict[int, int], hidden_dim: int):
        """
        Args:
            feat_dims: {node_type: input_feature_dim}
            hidden_dim: 统一隐层维度
        """
        super().__init__()
        self.encoders = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(v, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            for k, v in feat_dims.items()
        })

    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """将不同类型节点特征统一到 hidden_dim"""
        out = torch.zeros(node_features.shape[0],
                          next(iter(self.encoders.values()))[0].out_features,
                          device=node_features.device)
        for node_type_str, encoder in self.encoders.items():
            mask = node_types == int(node_type_str)
            if mask.any():
                out[mask] = encoder(node_features[mask])
        return out


# ─────────────────────────────────────────────
# GAT 广告预测模型主体
# ─────────────────────────────────────────────

class GNNAdModel(nn.Module):
    """
    Graph Attention Network 广告反馈预测模型

    架构：
      输入节点特征（匿名化）
           ↓
      异构节点编码器（统一维度）
           ↓
      GAT Layer 1（多头注意力，concat）
           ↓
      GAT Layer 2（多头注意力，concat）
           ↓
      GAT Layer 3（多头注意力，average）
           ↓
      节点 Embedding
           ↓
      广告位-类别对 Readout（目标节点特征）
           ↓
      MLP 预测头 → CTR / 参与度分数
    """

    def __init__(
        self,
        node_feat_dim: int = 64,   # 统一后的节点特征维度
        gat_hidden: int = 32,      # GAT 隐层每头维度
        num_heads: int = 4,        # 注意力头数
        num_layers: int = 3,       # GAT 层数
        dropout: float = 0.1,
        output_dim: int = 1,       # 输出维度（1=CTR分数）
    ):
        super().__init__()
        self.num_layers = num_layers

        # 节点编码器（统一到 node_feat_dim）
        feat_dims = {
            NodeType.AD_SLOT: node_feat_dim,
            NodeType.CATEGORY: node_feat_dim,
            NodeType.TIME_CTX: node_feat_dim,
        }
        self.node_encoder = HeterogeneousNodeEncoder(feat_dims, node_feat_dim)

        # GAT 层堆叠
        gat_layers = []
        in_dim = node_feat_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = gat_hidden if not is_last else gat_hidden
            gat_layers.append(MultiHeadGATLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=not is_last,  # 最后一层用平均
            ))
            in_dim = out_dim * num_heads if not is_last else out_dim
        self.gat_layers = nn.ModuleList(gat_layers)

        # 残差连接投影（维度对齐）
        self.residual_proj = nn.Linear(node_feat_dim, in_dim)

        # 预测头：拼接广告位和类别节点特征 → CTR
        pred_input_dim = gat_hidden * 2  # 广告位 + 类别节点特征拼接
        self.predictor = nn.Sequential(
            nn.Linear(pred_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: AdGraph,
        query_pairs: torch.Tensor,   # [Q, 2] 查询节点对 (ad_slot_idx, category_idx)
    ) -> torch.Tensor:
        """
        Args:
            graph: AdGraph 图结构
            query_pairs: [Q, 2] 要预测的（广告位, 类别）节点对索引

        Returns:
            scores: [Q] CTR 预测分数（sigmoid 后为概率）
        """
        h = graph.node_features
        edge_index = graph.edge_index
        edge_weights = graph.edge_weights

        # 异构节点编码
        h = self.node_encoder(h, graph.node_types)
        h_residual = self.residual_proj(h)

        # GAT 前向传播（多层）
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, edge_index, edge_weights)
            h_new = self.dropout(h_new)

            # 残差连接（仅维度匹配时）
            if h_new.shape == h_residual.shape:
                h_new = h_new + h_residual

            h = h_new

        # 节点对 Readout：提取查询节点的 embedding
        ad_slot_idx = query_pairs[:, 0]   # [Q]
        category_idx = query_pairs[:, 1]  # [Q]

        h_ad_slot = h[ad_slot_idx]    # [Q, gat_hidden]
        h_category = h[category_idx]  # [Q, gat_hidden]

        # 拼接两种节点表示
        pair_repr = torch.cat([h_ad_slot, h_category], dim=-1)  # [Q, gat_hidden*2]

        # 预测
        logits = self.predictor(pair_repr).squeeze(-1)  # [Q]
        return torch.sigmoid(logits)

    def get_node_embeddings(self, graph: AdGraph) -> torch.Tensor:
        """提取所有节点 Embedding（用于下游任务或可视化）"""
        self.eval()
        with torch.no_grad():
            h = self.node_encoder(graph.node_features, graph.node_types)
            for gat_layer in self.gat_layers:
                h = gat_layer(h, graph.edge_index, graph.edge_weights)
        return h


# ─────────────────────────────────────────────
# 训练工具
# ─────────────────────────────────────────────

class GNNTrainer:
    """GAT 模型训练封装"""

    def __init__(self, model: GNNAdModel, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCELoss()

    def train_epoch(
        self,
        graph: AdGraph,
        query_pairs: torch.Tensor,   # [Q, 2]
        labels: torch.Tensor,        # [Q] 0/1
        batch_size: int = 512,
    ) -> float:
        """训练一个 epoch，返回平均 loss"""
        self.model.train()
        N = query_pairs.shape[0]
        perm = torch.randperm(N)
        total_loss = 0.0
        steps = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch_pairs = query_pairs[idx]
            batch_labels = labels[idx].float()

            self.optimizer.zero_grad()
            preds = self.model(graph, batch_pairs)
            loss = self.criterion(preds, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)


# ─────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────

def build_synthetic_graph(num_ad_slots=50, num_categories=20, num_time_ctx=24) -> AdGraph:
    """构建合成广告生态图用于测试"""
    N = num_ad_slots + num_categories + num_time_ctx
    feat_dim = 64

    # 节点特征（随机初始化，模拟匿名化上下文特征）
    node_features = torch.randn(N, feat_dim)

    # 节点类型
    node_types = torch.cat([
        torch.zeros(num_ad_slots, dtype=torch.long),
        torch.ones(num_categories, dtype=torch.long),
        torch.full((num_time_ctx,), 2, dtype=torch.long),
    ])

    # 构建边：广告位-类别 共现边 + 类别-时间上下文 边
    edges_src, edges_dst = [], []

    # 广告位 ↔ 类别（随机稀疏连接）
    for i in range(num_ad_slots):
        connected = torch.randperm(num_categories)[:3]
        for j in connected:
            edges_src.extend([i, num_ad_slots + j.item()])
            edges_dst.extend([num_ad_slots + j.item(), i])

    # 类别 ↔ 时间上下文
    for i in range(num_categories):
        t = torch.randperm(num_time_ctx)[:4]
        for j in t:
            edges_src.extend([num_ad_slots + i, num_ad_slots + num_categories + j.item()])
            edges_dst.extend([num_ad_slots + num_categories + j.item(), num_ad_slots + i])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weights = torch.rand(edge_index.shape[1])

    return AdGraph(node_features, node_types, edge_index, edge_weights)


def demo():
    graph = build_synthetic_graph()
    model = GNNAdModel(node_feat_dim=64, gat_hidden=32, num_heads=4, num_layers=2)

    # 生成查询对（广告位, 类别）
    Q = 200
    ad_slot_idx = torch.randint(0, 50, (Q,))
    category_idx = torch.randint(50, 70, (Q,))
    query_pairs = torch.stack([ad_slot_idx, category_idx], dim=1)
    labels = torch.bernoulli(torch.full((Q,), 0.3))

    trainer = GNNTrainer(model)
    for epoch in range(3):
        loss = trainer.train_epoch(graph, query_pairs, labels, batch_size=64)
        print(f"Epoch {epoch+1} Loss={loss:.4f}")

    # 推理
    model.eval()
    with torch.no_grad():
        scores = model(graph, query_pairs[:10])
    print(f"✅ GNN demo done | Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # 节点 Embedding
    embeddings = model.get_node_embeddings(graph)
    print(f"✅ Node embeddings shape: {embeddings.shape}")
    return model


if __name__ == "__main__":
    demo()
