#!/usr/bin/env python3
"""
测试 EGNN 设置是否正确
"""

import yaml
import torch
from CIRT.models import model
from CIRT import dataset

def test_config():
    """测试配置文件加载"""
    print("=" * 60)
    print("1. 测试配置文件加载...")
    
    with open('CIRT/configs/EGNN.yaml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    print(f"✓ 模型名称: {model_args['model_name']}")
    print(f"✓ Input size: {model_args['input_size']}")
    print(f"✓ Output size: {model_args['output_size']}")
    print(f"✓ Hidden size: {model_args['hidden_sizes']}")
    print(f"✓ Batch size: {data_args['batch_size']}")
    print(f"✓ Kernel size: {data_args['kernel_size']}")
    return model_args, data_args

def test_dataset(data_args):
    """测试数据集"""
    print("\n" + "=" * 60)
    print("2. 测试数据集...")
    
    try:
        # 创建一个小的测试数据集
        test_years = data_args['val_years'][:1]  # 只用一年来测试
        
        test_dataset = dataset.S2SGraphDataset(
            data_dir=data_args['data_dir'],
            years=test_years,
            n_step=data_args['n_step'],
            lead_time=data_args['lead_time'],
            kernel_size=data_args['kernel_size'],
            single_vars=data_args['single_vars'],
            pred_single_vars=data_args['pred_single_vars'],
            pred_pressure_vars=data_args['pred_pressure_vars'],
            is_normalized=data_args['is_normalized']
        )
        
        print(f"✓ 数据集大小: {len(test_dataset)}")
        print(f"✓ 节点数: {test_dataset.num_nodes}")
        print(f"✓ 边数: {test_dataset.edge_index.shape[1]}")
        
        # 测试加载一个样本
        sample = test_dataset[0]
        print(f"\n样本信息:")
        print(f"  x shape: {sample.x.shape}")
        print(f"  y shape: {sample.y.shape}")
        print(f"  edge_index shape: {sample.edge_index.shape}")
        print(f"  edge_feat shape: {sample.edge_feat.shape}")
        print(f"  radial shape: {sample.radial.shape}")
        print(f"  coord shape: {sample.coord.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model(model_args, data_args):
    """测试模型"""
    print("\n" + "=" * 60)
    print("3. 测试模型...")
    
    try:
        # 创建模型
        baseline = model.S2SBenchmarkModel(
            model_args=model_args,
            data_args=data_args
        )
        
        print(f"✓ 模型创建成功")
        print(f"✓ 模型类型: {type(baseline.model)}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in baseline.model.parameters())
        trainable_params = sum(p.numel() for p in baseline.model.parameters() if p.requires_grad)
        
        print(f"✓ 总参数数: {total_params:,}")
        print(f"✓ 可训练参数数: {trainable_params:,}")
        
        # 测试forward pass
        print("\n测试 forward pass...")
        batch_size = 2
        num_nodes = 121 * 240
        input_features = 52  # input_size - 11
        
        # 模拟输入
        h = torch.randn(batch_size * num_nodes, input_features)
        u = torch.randn(batch_size * num_nodes, 11)
        v = torch.randn(batch_size * num_nodes, 11)
        
        # 创建简单的边索引
        edges = torch.randint(0, batch_size * num_nodes, (2, 1000))
        edge_attr = torch.randn(1000, 1)
        radial = torch.randn(1000, 2)
        
        with torch.no_grad():
            output = baseline(h=h, u=u, v=v, radial=radial, edges=edges, edge_attr=edge_attr)
        
        print(f"✓ Forward pass 成功")
        print(f"✓ 输出 shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 EGNN 设置测试")
    print("=" * 60)
    
    # 1. 测试配置
    try:
        model_args, data_args = test_config()
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return
    
    # 2. 测试数据集
    dataset_ok = test_dataset(data_args)
    
    # 3. 测试模型
    model_ok = test_model(model_args, data_args)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  配置加载: ✓")
    print(f"  数据集: {'✓' if dataset_ok else '✗'}")
    print(f"  模型: {'✓' if model_ok else '✗'}")
    
    if dataset_ok and model_ok:
        print("\n✅ 所有测试通过！可以开始训练。")
        print("\n开始训练:")
        print("  bash run.sh EGNN")
        print("  或")
        print("  python3 -u train.py --config_filepath CIRT/configs/EGNN.yaml")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()

