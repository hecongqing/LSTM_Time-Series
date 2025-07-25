#!/usr/bin/env python3
"""
Transformer模型使用示例
演示如何训练和使用Transformer进行时间序列预测
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).resolve().parent))

def train_transformer_model():
    """训练Transformer模型示例"""
    print("=" * 60)
    print("训练Transformer模型")
    print("=" * 60)
    
    # 导入训练函数
    from transformer_train import train_model, parse_args
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.data_path = "data/raw/london_merged.csv"  # 请替换为实际数据路径
            self.epochs = 30
            self.batch_size = 64
            self.lr = 3e-4
            self.window_size = 24
            self.pred_horizon = 6
            self.embed_size = 256
            self.num_heads = 8
            self.num_blocks = 2
            self.dropout = 0.1
            self.device = "cpu"
            self.save_dir = "transformer_outputs"
            self.use_categorical = False  # 先使用简化版本
            self.loss_type = "mse"
            self.time_shuffle = False
    
    args = Args()
    
    # 检查数据文件是否存在
    if not Path(args.data_path).exists():
        print(f"错误: 数据文件 {args.data_path} 不存在")
        print("请确保数据文件路径正确")
        return False
    
    try:
        print(f"开始训练Transformer模型...")
        print(f"数据路径: {args.data_path}")
        print(f"输出目录: {args.save_dir}")
        print(f"使用分类特征: {args.use_categorical}")
        print(f"窗口大小: {args.window_size}, 预测步长: {args.pred_horizon}")
        print(f"模型参数: embed_size={args.embed_size}, num_heads={args.num_heads}, num_blocks={args.num_blocks}")
        print()
        
        train_model(args)
        print("\n✅ 模型训练完成!")
        print(f"模型文件保存在: {args.save_dir}/")
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return False


def run_transformer_inference():
    """运行Transformer推理示例"""
    print("=" * 60)
    print("Transformer模型推理")
    print("=" * 60)
    
    import torch
    import joblib
    import numpy as np
    from transformer_dataset import TransformerBikeDataset, create_features_transformer
    from transformer_model import SimplifiedTransformerBikePredictor, TransformerBikePredictor
    import pandas as pd
    
    # 模型路径
    model_path = "transformer_outputs/transformer_model_best.pt"
    scaler_path = "transformer_outputs/transformer_feature_scaler.pkl"
    config_path = "transformer_outputs/transformer_model_config.pkl"
    data_path = "data/raw/london_merged.csv"
    
    # 检查文件是否存在
    required_files = [model_path, scaler_path, config_path, data_path]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            print("请先运行训练脚本生成模型文件")
            return False
    
    try:
        # 加载配置和预处理器
        config = joblib.load(config_path)
        scaler = joblib.load(scaler_path)
        
        print(f"✅ 加载模型配置: {config['model_type']} 模型")
        print(f"   - 嵌入维度: {config['embed_size']}")
        print(f"   - 注意力头数: {config['num_heads']}")
        print(f"   - Transformer块数: {config['num_blocks']}")
        
        # 加载模型
        if config['model_type'] == 'simplified':
            model = SimplifiedTransformerBikePredictor(
                input_dim=config['num_numeric_features'],
                embed_size=config['embed_size'],
                num_heads=config['num_heads'],
                num_blocks=config['num_blocks'],
                output_dim=config['output_dim'],
                dropout=config['dropout'],
            )
        else:
            model = TransformerBikePredictor(
                numeric_features=config['num_numeric_features'],
                categorical_features_dims=config['categorical_dims'],
                static_features_dims=config['static_dims'],
                embed_size=config['embed_size'],
                num_heads=config['num_heads'],
                num_blocks=config['num_blocks'],
                output_dim=config['output_dim'],
                dropout=config['dropout'],
            )
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("✅ 模型加载成功")
        
        # 加载数据
        df = pd.read_csv(data_path)
        df = create_features_transformer(df)
        
        # 准备特征
        numeric_cols = [
            "cnt", "t1", "t2", "hum", "wind_speed",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "dayofweek_sin", "dayofweek_cos",
        ]
        
        numeric_data = df[numeric_cols].values
        numeric_scaled = scaler.transform(numeric_data)
        
        # 创建数据集
        window_size = 24
        pred_horizon = 6
        ds = TransformerBikeDataset(
            numeric_scaled, None, None, window_size, pred_horizon
        )
        
        print(f"✅ 数据集准备完成: {len(ds)} 个样本")
        
        # 进行推理
        test_idx = len(ds) - 100  # 使用倒数第100个样本
        x_numeric, x_categorical, x_static, y_true = ds[test_idx]
        
        with torch.no_grad():
            x_batch = x_numeric.unsqueeze(0)
            
            if config['model_type'] == 'simplified':
                y_pred = model(x_batch).numpy().flatten()
            else:
                y_pred = model(x_batch, None, None).numpy().flatten()
        
        y_true = y_true.numpy()[:pred_horizon]
        
        # 计算评估指标
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        
        print("\n" + "=" * 40)
        print("预测结果")
        print("=" * 40)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print()
        print("逐小时预测对比:")
        print("小时\t真实值\t预测值\t绝对误差")
        print("-" * 40)
        for i in range(pred_horizon):
            error = abs(y_true[i] - y_pred[i])
            print(f"{i+1:2d}\t{y_true[i]:.4f}\t{y_pred[i]:.4f}\t{error:.4f}")
        
        print("\n✅ 推理完成!")
        return True
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🤖 Transformer时间序列预测示例")
    print("本示例演示如何使用Transformer进行伦敦自行车需求预测")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 训练Transformer模型")
        print("2. 运行模型推理")
        print("3. 启动Streamlit应用")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            success = train_transformer_model()
            if success:
                print("\n可以选择选项2进行推理测试，或选择选项3启动Web应用")
        
        elif choice == "2":
            run_transformer_inference()
            
        elif choice == "3":
            print("\n启动Streamlit应用...")
            print("运行以下命令:")
            print("cd src && streamlit run transformer_app.py")
            break
            
        elif choice == "4":
            print("再见!")
            break
            
        else:
            print("❌ 无效选择，请重新输入")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()