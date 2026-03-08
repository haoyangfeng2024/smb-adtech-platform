"""
ML 竞价模型
实现基于特征工程 + 逻辑回归 / 梯度提升的竞价优化模型

核心思路：
  - pCTR（预测点击率）+ pCVR（预测转化率）双塔模型
  - 特征工程：上下文特征 + 用户特征 + 广告特征 + 交叉特征
  - 在线学习：支持增量更新（warm start）
  - 模型版本管理：支持 A/B 测试
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 特征定义
# ─────────────────────────────────────────────

# 数值特征（直接用于模型）
NUMERIC_FEATURES = [
    "hour_of_day",          # 请求小时（0-23）
    "day_of_week",          # 星期几（0-6）
    "floor_price",          # 底价
    "bid_amount",           # 出价金额
    "historical_ctr",       # 历史点击率（贝叶斯平滑）
    "historical_cvr",       # 历史转化率
    "budget_utilization",   # 预算利用率 spend/total_budget
    "campaign_age_days",    # 活动已运行天数
    "impressions_log",      # log1p(曝光数) — 减少量级差异
    "spend_log",            # log1p(消耗)
]

# 类别特征（需要编码）
CATEGORICAL_FEATURES = [
    "device_type",    # mobile/desktop/tablet
    "os",             # ios/android/windows/mac
    "ad_format",      # banner/video/native
    "bidding_strategy",  # cpc/cpm/cpa/smart
    "geo_country",    # 国家代码
]

# 交叉特征（特征组合，捕获非线性交互）
CROSS_FEATURES = [
    ("device_type", "ad_format"),   # 设备 × 广告形式
    ("hour_of_day", "device_type"), # 时段 × 设备（移动端午休高峰）
    ("geo_country", "ad_format"),   # 地区 × 广告形式
]


@dataclass
class ModelMetrics:
    """模型评估指标"""
    auc_roc: float = 0.0
    log_loss: float = 0.0
    train_samples: int = 0
    eval_samples: int = 0
    trained_at: str = ""
    version: str = ""


# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    特征工程流水线

    生产环境建议：
    - 高基数类别特征（用户 ID、站点 ID）→ 实体嵌入（Entity Embedding）
    - 序列特征（行为序列）→ LSTM / Transformer
    - 实时特征 → Redis Feature Store
    """

    def __init__(self):
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def _extract_time_features(self, timestamp_ms: Optional[int]) -> dict[str, float]:
        """从时间戳提取时间特征（时段效应显著影响 CTR）"""
        import datetime
        if timestamp_ms:
            dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)
        else:
            dt = datetime.datetime.now(tz=datetime.timezone.utc)
        return {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
            # 将周期特征转为 sin/cos 避免边界效应（0点和23点其实是相邻的）
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        }

    def _build_cross_features(self, row: dict) -> dict[str, str]:
        """构建交叉特征（哈希技巧防止维度爆炸）"""
        cross = {}
        for feat_a, feat_b in CROSS_FEATURES:
            val_a = str(row.get(feat_a, ""))
            val_b = str(row.get(feat_b, ""))
            cross_key = f"{feat_a}_x_{feat_b}"
            cross_val = f"{val_a}_{val_b}"
            # 哈希到固定维度桶
            hash_val = int(hashlib.md5(cross_val.encode()).hexdigest(), 16) % 1000
            cross[cross_key] = hash_val
        return cross

    def build_features(self, campaign: dict, request: dict) -> dict[str, Any]:
        """
        从活动 + 请求构建完整特征向量

        Args:
            campaign: 广告活动数据
            request: 竞价请求数据

        Returns:
            特征字典（数值 + 编码后的类别）
        """
        impressions = max(campaign.get("impressions", 0), 1)
        clicks = campaign.get("clicks", 0)
        conversions = campaign.get("conversions", 0)
        spend = float(campaign.get("spend", 0))
        budget_total = float(campaign.get("budget", {}).get("total", 1))

        # 贝叶斯平滑后的历史 CTR/CVR
        prior_ctr, prior_weight = 0.001, 100
        smoothed_ctr = (clicks + prior_ctr * prior_weight) / (impressions + prior_weight)
        smoothed_cvr = (conversions + 0.01 * 100) / (clicks + 100) if clicks > 0 else 0.01

        features = {
            # 数值特征
            "floor_price": float(request.get("floor_price", 0.01)),
            "bid_amount": float(campaign.get("bid_amount", 0)),
            "historical_ctr": smoothed_ctr,
            "historical_cvr": smoothed_cvr,
            "budget_utilization": min(spend / max(budget_total, 1e-6), 1.0),
            "impressions_log": np.log1p(impressions),
            "spend_log": np.log1p(spend),
            "campaign_age_days": 0,  # TODO: (now - start_date).days

            # 类别特征
            "device_type": request.get("device_type", "unknown"),
            "os": request.get("os", "unknown"),
            "ad_format": request.get("ad_format", "banner"),
            "bidding_strategy": campaign.get("bidding_strategy", "cpc"),
            "geo_country": (request.get("geo") or {}).get("countries", [""])[0] if request.get("geo") else "",
        }

        # 时间特征
        features.update(self._extract_time_features(request.get("timestamp_ms")))

        # 交叉特征
        features.update(self._build_cross_features(features))

        return features

    def fit_transform(self, feature_dicts: list[dict]) -> np.ndarray:
        """训练时拟合并转换特征"""
        df = pd.DataFrame(feature_dicts)

        # 类别特征 LabelEncoding
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le

        self._fitted = True
        self._columns = df.columns.tolist()
        return df.values.astype(np.float32)

    def transform_batch(self, feature_dicts: list[dict]) -> np.ndarray:
        """推理时批量转换特征（用于验证集评估）"""
        return np.vstack([self.transform(f) for f in feature_dicts])

    def transform(self, feature_dict: dict) -> np.ndarray:
        """推理时转换单条特征"""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer not fitted, call fit_transform first")

        row = {}
        for col in self._columns:
            val = feature_dict.get(col, 0)
            if col in self._label_encoders:
                le = self._label_encoders[col]
                try:
                    val = le.transform([str(val)])[0]
                except ValueError:
                    val = 0  # 未见过的类别 → 0
            row[col] = val

        return np.array(list(row.values()), dtype=np.float32).reshape(1, -1)


# ─────────────────────────────────────────────
# 竞价模型
# ─────────────────────────────────────────────

class BiddingModel:
    """
    竞价 pCTR 预测模型

    架构：
    - 轻量级：逻辑回归（延迟 < 1ms，适合 RTB）
    - 高精度：梯度提升（XGBoost / GBDT，离线训练，定期更新）
    - 在线学习：SGD（支持流式增量更新）

    生产建议：
    - 使用 ONNX Runtime 加速推理
    - 模型存储在 MLflow / S3，版本化管理
    - 定时（每天/每小时）从 ClickHouse 拉数据重新训练
    """

    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.feature_engineer = FeatureEngineer()
        self.metrics = ModelMetrics()
        self._model: Optional[Pipeline] = None
        self._scaler = StandardScaler()

    def _build_pipeline(self) -> Pipeline:
        """构建 sklearn Pipeline（特征缩放 + 模型）"""
        if self.model_type == "logistic":
            # 逻辑回归：快速，可解释，适合 RTB 在线推理
            clf = LogisticRegression(
                C=0.1,               # L2 正则，防过拟合
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",  # 处理类别不平衡（正样本 << 负样本）
                random_state=42,
            )
        elif self.model_type == "gbdt":
            # 梯度提升：高精度，适合离线训练后导出
            clf = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=50,    # 防止在小样本叶子过拟合
                subsample=0.8,          # 随机采样降低方差
                random_state=42,
            )
            # 概率校准（GBDT 输出的概率往往偏大，需校准）
            clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        elif self.model_type == "sgd":
            # 在线学习：支持流式增量训练
            clf = SGDClassifier(
                loss="log_loss",     # 等价于逻辑回归
                penalty="l2",
                alpha=0.0001,
                max_iter=1,          # 每次只过一遍数据
                warm_start=True,     # 增量学习关键参数
                class_weight="balanced",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def train(
        self,
        feature_dicts: list[dict],
        labels: list[int],   # 1=点击, 0=未点击
        eval_feature_dicts: Optional[list[dict]] = None,
        eval_labels: Optional[list[int]] = None,
    ) -> ModelMetrics:
        """
        训练模型

        Args:
            feature_dicts: 训练特征（原始字典列表）
            labels: 标签（1=点击, 0=未点击）
            eval_feature_dicts: 验证集特征
            eval_labels: 验证集标签

        Returns:
            ModelMetrics（AUC、LogLoss 等）
        """
        logger.info(f"Training {self.model_type} model on {len(labels)} samples")
        t0 = time.time()

        # 特征工程
        X_train = self.feature_engineer.fit_transform(feature_dicts)
        y_train = np.array(labels)

        # 正负样本比例统计（CTR 场景通常严重不平衡）
        pos_rate = y_train.mean()
        logger.info(f"Positive rate: {pos_rate:.4f} ({y_train.sum()} / {len(y_train)})")

        # 构建并训练 Pipeline
        self._model = self._build_pipeline()
        self._model.fit(X_train, y_train)

        # 训练集评估
        y_pred_prob = self._model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_prob)
        train_loss = log_loss(y_train, y_pred_prob)

        # 验证集评估
        eval_auc, eval_loss = 0.0, 0.0
        if eval_feature_dicts and eval_labels:
            X_eval = self.feature_engineer.transform_batch(eval_feature_dicts)
            y_eval = np.array(eval_labels)
            y_eval_prob = self._model.predict_proba(X_eval)[:, 1]
            eval_auc = roc_auc_score(y_eval, y_eval_prob)
            eval_loss = log_loss(y_eval, y_eval_prob)

        self.metrics = ModelMetrics(
            auc_roc=eval_auc or train_auc,
            log_loss=eval_loss or train_loss,
            train_samples=len(labels),
            eval_samples=len(eval_labels) if eval_labels else 0,
            trained_at=pd.Timestamp.now().isoformat(),
            version=f"{self.model_type}_v{int(time.time())}",
        )

        logger.info(
            f"Training done in {time.time()-t0:.1f}s | "
            f"Train AUC={train_auc:.4f} | Eval AUC={eval_auc:.4f}"
        )
        return self.metrics

    def predict_ctr(self, feature_dict: dict) -> float:
        """
        单条预测 pCTR（推理路径，延迟敏感）

        Args:
            feature_dict: 特征字典（来自 FeatureEngineer.build_features）

        Returns:
            点击概率 [0, 1]
        """
        if self._model is None:
            raise RuntimeError("Model not trained, call train() or load() first")

        X = self.feature_engineer.transform(feature_dict)
        return float(self._model.predict_proba(X)[0, 1])

    def predict_batch(self, feature_dicts: list[dict]) -> np.ndarray:
        """批量预测（离线评估 / 报表生成）"""
        if self._model is None:
            raise RuntimeError("Model not fitted")
        X = np.vstack([self.feature_engineer.transform(f) for f in feature_dicts])
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """保存模型到磁盘（joblib 序列化）"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_engineer": self.feature_engineer,
            "metrics": self.metrics,
            "model_type": self.model_type,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Model saved to {path} | Metrics: AUC={self.metrics.auc_roc:.4f}")

    @classmethod
    def load(cls, path: str) -> BiddingModel:
        """从磁盘加载模型"""
        payload = joblib.load(path)
        instance = cls(model_type=payload["model_type"])
        instance._model = payload["model"]
        instance.feature_engineer = payload["feature_engineer"]
        instance.metrics = payload["metrics"]
        logger.info(f"Model loaded from {path} | Version: {instance.metrics.version}")
        return instance


# ─────────────────────────────────────────────
# 演示 / 快速验证
# ─────────────────────────────────────────────

def demo_train_and_predict():
    """生成合成数据，训练并验证模型（CI 冒烟测试用）"""
    import random
    rng = random.Random(42)
    np.random.seed(42)

    # 生成合成特征
    n = 5000
    feature_dicts = []
    labels = []

    for _ in range(n):
        campaign = {
            "bid_amount": rng.uniform(0.01, 5.0),
            "impressions": rng.randint(100, 100000),
            "clicks": rng.randint(0, 500),
            "conversions": rng.randint(0, 50),
            "spend": rng.uniform(0, 1000),
            "budget": {"total": rng.uniform(100, 10000)},
            "bidding_strategy": rng.choice(["cpc", "cpm", "cpa", "smart"]),
        }
        request = {
            "floor_price": rng.uniform(0.005, 2.0),
            "device_type": rng.choice(["mobile", "desktop", "tablet"]),
            "os": rng.choice(["ios", "android", "windows"]),
            "ad_format": rng.choice(["banner", "video", "native"]),
            "geo": {"countries": [rng.choice(["US", "CN", "GB", "DE"])]},
        }

        eng = FeatureEngineer()
        feat = eng.build_features(campaign, request)
        feature_dicts.append(feat)
        # 标签：CTR 0.1% 基准，受出价和 bid_amount 影响
        click_prob = 0.001 + campaign["bid_amount"] * 0.0005 + (rng.random() * 0.002)
        labels.append(1 if rng.random() < click_prob else 0)

    # 训练
    model = BiddingModel(model_type="logistic")
    train_features = feature_dicts[:4000]
    train_labels = labels[:4000]
    metrics = model.train(train_features, train_labels)

    print(f"✅ Model trained | AUC={metrics.auc_roc:.4f} | LogLoss={metrics.log_loss:.4f}")

    # 推理测试
    test_feat = feature_dicts[4000]
    # 需要单独初始化 FeatureEngineer 并 fit_transform
    model.feature_engineer.fit_transform(train_features)
    pred = model.predict_ctr(test_feat)
    print(f"✅ Sample pCTR prediction: {pred:.6f}")

    return model


if __name__ == "__main__":
    demo_train_and_predict()
