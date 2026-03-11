"""
合成数据生成器 (Synthetic Data Generator)

用于生成模拟的广告请求数据，用于模型训练和测试。

设计原则：
1. **隐私合规 (Privacy-First)**：
   - 完全不包含任何真实用户 PII（姓名、邮箱、真实用户 ID）
   - 使用随机哈希或 UUID 替代用户标识
   - 符合 GDPR/CCPA 隐私保护要求

2. **数据分布**：
   - 模拟真实广告生态的数据分布（CTR ~ 1-5%, CVR ~ 1-10%）
   - 包含多种设备类型、地理位置、时间段

3. **用途**：
   - 供给 `scripts/train.py` 进行模型训练
   - 用于离线评估模型效果
"""
import csv
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Optional


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

class Config:
    NUM_SAMPLES = 100_000      # 总样本数
    CTR_BASE = 0.02             # 基础 CTR (~2%)
    CVR_BASE = 0.05            # 基础 CVR (~5%)
    
    DEVICES = ["mobile", "desktop", "tablet"]
    OS = ["ios", "android", "windows", "mac", "linux"]
    AD_FORMATS = ["banner", "video", "native", "interstitial"]
    COUNTRIES = ["US", "CA", "GB", "DE", "FR", "JP", "AU"]
    CAMPAIGNS = [f"camp_{i:04d}" for i in range(1, 51)]  # 50 个活动


# ─────────────────────────────────────────────
# 匿名化工具
# ─────────────────────────────────────────────

def generate_anonymous_id(seed: str) -> str:
    """
    生成不可逆的匿名 ID
    使用 SHA-256 哈希，无法反推原始种子
    """
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def generate_impression_id(index: int) -> str:
    """生成展示 ID"""
    return f"imp_{index:010d}"


def generate_click_id(impression_id: str, timestamp: int) -> str:
    """生成点击 ID（基于展示 ID 和时间戳哈希）"""
    seed = f"{impression_id}_{timestamp}"
    return f"clk_{generate_anonymous_id(seed)}"


# ─────────────────────────────────────────────
# 数据生成逻辑
# ─────────────────────────────────────────────

class SyntheticGenerator:
    """合成数据生成器"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.rng = random.Random(42)  # 固定种子，保证可复现性
    
    def generate_impression(self, index: int) -> dict:
        """
        生成单条展示数据
        
        Returns:
            dict: 包含展示、点击、转化标记的记录
        """
        # 随机特征
        device = self.rng.choice(self.config.DEVICES)
        os_name = self.rng.choice(self.config.OS)
        ad_format = self.rng.choice(self.config.AD_FORMATS)
        country = self.rng.choice(self.config.COUNTRIES)
        campaign_id = self.rng.choice(self.config.CAMPAIGNS)
        
        # 时间特征（模拟 7 天数据）
        days_ago = self.rng.randint(0, 6)
        hour = self.rng.randint(0, 23)
        timestamp = int((datetime.now() - timedelta(days=days_ago, hours=23-hour)).timestamp() * 1000)
        
        # 匿名用户 ID（不可逆哈希）
        user_seed = f"user_{self.rng.randint(1, 1000000)}"
        anonymous_user_id = generate_anonymous_id(user_seed)
        
        # 匿名设备 ID
        device_seed = f"device_{self.rng.randint(1, 500000)}"
        anonymous_device_id = generate_anonymous_id(device_seed)
        
        # CTR 模拟（设备/格式/地区差异）
        ctr = self.config.CTR_BASE
        if device == "mobile": ctr *= 1.2
        if ad_format == "video": ctr *= 1.5
        if country == "US": ctr *= 1.1
        
        # 决定是否点击
        clicked = self.rng.random() < ctr
        
        # CVR 模拟（点击后是否转化）
        cvr = self.config.CVR_BASE
        if country == "US": cvr *= 1.2  # 美国用户转化率高
        converted = clicked and (self.rng.random() < cvr)
        
        return {
            "impression_id": generate_impression_id(index),
            "timestamp_ms": timestamp,
            "anonymous_user_id": anonymous_user_id,
            "anonymous_device_id": anonymous_device_id,
            "campaign_id": campaign_id,
            "ad_format": ad_format,
            "device_type": device,
            "os": os_name,
            "geo_country": country,
            "hour_of_day": hour,
            "is_clicked": int(clicked),
            "is_converted": int(converted),
            # 原始 bid request 特征（用于模型输入）
            "floor_price": round(self.rng.uniform(0.1, 5.0), 2),
            "site_category": self.rng.choice(["news", "sports", "tech", "entertainment", "shopping"]),
        }
    
    def generate_dataset(self, num_samples: Optional[int] = None) -> list[dict]:
        """生成完整数据集"""
        n = num_samples or self.config.NUM_SAMPLES
        print(f"生成 {n} 条合成数据...")
        
        data = []
        for i in range(n):
            record = self.generate_impression(i)
            data.append(record)
            
            if (i + 1) % 10000 == 0:
                print(f"  进度: {i+1}/{n}")
        
        return data
    
    def export_csv(self, filepath: str, data: list[dict]):
        """导出为 CSV"""
        if not data:
            return
        
        keys = data[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"已导出 CSV: {filepath}")
    
    def export_json(self, filepath: str, data: list[dict]):
        """导出为 JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"已导出 JSON: {filepath}")


# ─────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────

def demo():
    """生成少量数据用于演示"""
    gen = SyntheticGenerator()
    # 快速生成 1000 条
    data = gen.generate_dataset(1000)
    
    # 统计
    clicks = sum(r["is_clicked"] for r in data)
    conversions = sum(r["is_converted"] for r in data)
    print(f"\n统计:")
    print(f"  总展示: {len(data)}")
    print(f"  点击数: {clicks} ({clicks/len(data)*100:.2f}%)")
    print(f"  转化数: {conversions} ({conversions/len(data)*100:.2f}%)")
    print(f"\n示例数据 (第一条):")
    print(json.dumps(data[0], indent=2))


if __name__ == "__main__":
    demo()
