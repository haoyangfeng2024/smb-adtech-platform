"""
模型训练脚本 (Model Training Pipeline)

支持以下模型的训练：
- DeepFM (CTR 预测)
- PPO Agent (强化学习竞价)

使用方法：
    python scripts/train.py --model deepfm --epochs 10
    python scripts/train.py --model ppo --episodes 100
"""
import argparse
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_deepfm(epochs: int, data_path: str, resume: bool = False):
    """训练 DeepFM 模型
    
    Args:
        resume: 是否从断点继续训练
    """
    print("=" * 50)
    print("训练 DeepFM (CTR Prediction)")
    print("=" * 50)
    
    output_path = "ml/artifacts/deepfm.pt"
    
    try:
        import torch
        from ml.models.deep_ctr_model import DeepFMTrainer, DeepFMConfig
        
        # 断点续训：如果启用且存在保存的模型，则加载
        if resume and os.path.exists(output_path):
            print(f"检测到已保存模型: {output_path}")
            print("从断点继续训练...")
            trainer = DeepFMTrainer.load(output_path)
            # 加载数据用于继续训练
            if os.path.exists(data_path):
                print(f"加载数据: {data_path}")
                with open(data_path, 'r') as f:
                    data = json.load(f)
            else:
                print("生成合成数据...")
                config = Config()
                config.NUM_SAMPLES = 10_000
                gen = SyntheticGenerator(config)
                data = gen.generate_dataset()
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w') as f:
                    json.dump(data, f)
                print(f"数据已保存: {data_path}")
        else:
            # 初始化新模型
            from ml.data.synthetic_generator import SyntheticGenerator, Config
            # 1. 生成或加载数据
            if os.path.exists(data_path):
                print(f"加载数据: {data_path}")
                with open(data_path, 'r') as f:
                    data = json.load(f)
            else:
                print("生成合成数据...")
                # 使用较小数据集快速训练演示
                config = Config()
                config.NUM_SAMPLES = 10_000  # 减少样本数用于快速训练
                gen = SyntheticGenerator(config)
                data = gen.generate_dataset()
                # 保存数据供下次使用
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w') as f:
                    json.dump(data, f)
                print(f"数据已保存: {data_path}")
        
        # 2. 准备特征和标签
        print("准备特征工程...")
        features = []
        labels = []
        
        for record in data:
            feat = {
                "device_type": record["device_type"],
                "os": record["os"],
                "ad_format": record["ad_format"],
                "geo_country": record["geo_country"],
                "bidding_strategy": "cpc",  # 默认
                "hour": str(record["hour_of_day"]),
                "dow": "0",  # 简化
                "campaign_id_hash": str(hash(record["campaign_id"]))[:4],
            }
            # 标签：点击 = 正样本
            label = record["is_clicked"]
            
            features.append(feat)
            labels.append(label)
        
        # 3. 划分训练/测试集
        split_idx = int(len(features) * 0.8)
        X_train = features[:split_idx]
        y_train = labels[:split_idx]
        X_test = features[split_idx:]
        y_test = labels[split_idx:]
        
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        
        # 4. 训练模型
        print("初始化 DeepFM 模型...")
        model_config = DeepFMConfig(
            num_fields=8,
            vocab_size=10_000,
            embed_dim=16,
            hidden_dims=[128, 64],
            num_epochs=epochs,
            batch_size=256,
        )
        
        # 如果不是从断点续训，则初始化新模型
        if not (resume and os.path.exists(output_path)):
            trainer = DeepFMTrainer(model_config)
        
        print("开始训练...")
        import torch
        X_train_tensor = torch.randint(0, 10000, (len(X_train), 8))
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.randint(0, 10000, (len(X_test), 8))
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        history = trainer.train(
            X_train_tensor, y_train_tensor,
            X_test_tensor, y_test_tensor
        )
        
        # 5. 保存模型
        output_path = "ml/artifacts/deepfm.pt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trainer.save(output_path)
        
        print(f"\n✅ DeepFM 训练完成！")
        print(f"   最终验证 AUC: {history['val_auc'][-1]:.4f}")
        print(f"   模型保存位置: {output_path}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装 PyTorch: pip install torch")


def train_ppo(episodes: int, resume: bool = False):
    """训练 PPO 竞价 Agent
    
    Args:
        resume: 是否从断点继续训练
    """
    print("=" * 50)
    print("训练 PPO Agent (Reinforcement Learning Bidding)")
    print("=" * 50)
    
    output_path = "ml/artifacts/ppo_agent.pt"
    """训练 PPO 竞价 Agent"""
    print("=" * 50)
    print("训练 PPO Agent (Reinforcement Learning Bidding)")
    print("=" * 50)
    
    try:
        from ml.models.rl_bidding_agent import PPOBiddingAgent, PPOConfig, BiddingEnvironment
        
        # 1. 初始化配置
        config = PPOConfig(
            state_dim=20,
            action_dim=1,
            hidden_dim=128,
            rollout_steps=512,
            num_epochs=4,
            batch_size=64,
        )
        
        # 断点续训
        if resume and os.path.exists(output_path):
            print(f"检测到已保存模型: {output_path}")
            print("从断点继续训练...")
            agent = PPOBiddingAgent.load(output_path)
        else:
            agent = PPOBiddingAgent(config)
        
        env = BiddingEnvironment(config)
        
        # 2. 训练循环
        print(f"开始训练 {episodes} 个 episode...")
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done:
                action, log_prob, value = agent.act(state)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # 存储 transition
                agent.store_transition(state, action, reward, done, log_prob, value)
                state = next_state
                steps += 1
                
                # 达到 rollout 步数时更新
                if len(agent.buffer) >= config.rollout_steps:
                    agent.update()
            
            # Episode 结束更新
            if len(agent.buffer) > 0:
                agent.update()
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Spend: ${env.spend:.2f}")
        
        # 3. 保存模型
        output_path = "ml/artifacts/ppo_agent.pt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        agent.save(output_path)
        
        print(f"\n✅ PPO Agent 训练完成！")
        print(f"   模型保存位置: {output_path}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["deepfm", "ppo"], 
        required=True,
        help="要训练的模型类型"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数 (DeepFM)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="训练 episode 数 (PPO)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ml/data/synthetic_train.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从断点继续训练（加载已保存的模型）"
    )
    
    args = parser.parse_args()
    
    if args.model == "deepfm":
        train_deepfm(args.epochs, args.data, args.resume)
    elif args.model == "ppo":
        train_ppo(args.episodes, args.resume)


if __name__ == "__main__":
    main()
