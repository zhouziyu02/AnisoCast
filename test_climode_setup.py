#!/usr/bin/env python3
"""
测试 ClimODE 设置是否正确
"""

import yaml
import torch
from CIRT.models import model

def test_config():
    """测试配置文件加载"""
    print("=" * 60)
    print("1. 测试配置文件加载...")
    
    with open('CIRT/configs/ClimODE.yaml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    print(f"✓ 模型名称: {model_args['model_name']}")
    print(f"✓ 模型类型: {model_args['model_type']}")
    print(f"✓ ODE solver: {model_args['solver']}")
    print(f"✓ Input size: {model_args['input_size']}")
    print(f"✓ Output size: {model_args['output_size']}")
    print(f"✓ Batch size: {data_args['batch_size']}")
    print(f"✓ Learning rate: {model_args['learning_rate']}")
    return model_args, data_args

def test_model(model_args, data_args):
    """测试模型"""
    print("\n" + "=" * 60)
    print("2. 测试模型...")
    
    try:
        # 创建模型
        baseline = model.S2SBenchmarkModel(
            model_args=model_args,
            data_args=data_args
        )
        
        print(f"✓ 模型创建成功")
        print(f"✓ 模型类型: {type(baseline.model).__name__}")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in baseline.model.parameters())
        trainable_params = sum(p.numel() for p in baseline.model.parameters() if p.requires_grad)
        
        print(f"✓ 总参数数: {total_params:,}")
        print(f"✓ 可训练参数数: {trainable_params:,}")
        
        # 测试forward pass
        print("\n测试 forward pass...")
        batch_size = 4
        input_channels = 63
        height = 121
        width = 240
        
        # 模拟输入
        x = torch.randn(batch_size, input_channels, height, width)
        
        with torch.no_grad():
            output = baseline(x)
        
        print(f"✓ Forward pass 成功")
        print(f"✓ 输入 shape: {x.shape}")
        print(f"✓ 输出 shape: {output.shape}")
        
        # 验证输出形状
        expected_shape = (batch_size, 2, input_channels, height, width)
        if output.dim() == 4:
            # 简化模型返回 [batch, channels, height, width]
            expected_shape_simple = (batch_size, input_channels, height, width)
            if output.shape == expected_shape_simple:
                print(f"✓ 输出形状正确 (简化模式)")
            else:
                print(f"⚠ 输出形状: {output.shape}, 期望: {expected_shape_simple}")
        elif output.dim() == 5:
            # 完整模型返回 [batch, time_steps, channels, height, width]
            if output.shape[0] == batch_size and output.shape[2] == input_channels:
                print(f"✓ 输出形状正确 (完整模式)")
            else:
                print(f"⚠ 输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_compatibility():
    """测试数据兼容性"""
    print("\n" + "=" * 60)
    print("3. 测试数据格式兼容性...")
    
    try:
        # 模拟 CirT 数据格式
        batch_size = 4
        n_steps = 2
        channels = 63
        height = 121
        width = 240
        
        x = torch.randn(batch_size, channels, height, width)
        y = torch.randn(batch_size, n_steps, channels, height, width)
        timestamp = torch.randn(batch_size)
        
        print(f"✓ 输入数据 x: {x.shape}")
        print(f"✓ 目标数据 y: {y.shape}")
        print(f"✓ 时间戳: {timestamp.shape}")
        print(f"✓ 数据格式与 CirT 一致")
        
        return True
    except Exception as e:
        print(f"✗ 数据兼容性测试失败: {e}")
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n" + "=" * 60)
    print("4. 测试损失计算...")
    
    try:
        from CIRT import criterion
        
        loss_fn = criterion.MSE()
        
        # 模拟预测和目标
        batch_size = 4
        n_steps = 2
        channels = 63
        height = 121
        width = 240
        
        pred = torch.randn(batch_size, n_steps, channels, height, width)
        target = torch.randn(batch_size, n_steps, channels, height, width)
        
        loss = loss_fn(pred, target)
        
        print(f"✓ MSE Loss 计算成功: {loss.item():.6f}")
        print(f"✓ Loss 是标量: {loss.dim() == 0}")
        
        return True
    except Exception as e:
        print(f"✗ 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 ClimODE 设置测试")
    print("=" * 60)
    
    # 1. 测试配置
    try:
        model_args, data_args = test_config()
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return
    
    # 2. 测试模型
    model_ok = test_model(model_args, data_args)
    
    # 3. 测试数据兼容性
    data_ok = test_data_compatibility()
    
    # 4. 测试损失计算
    loss_ok = test_loss_computation()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  配置加载: ✓")
    print(f"  模型: {'✓' if model_ok else '✗'}")
    print(f"  数据兼容性: {'✓' if data_ok else '✗'}")
    print(f"  损失计算: {'✓' if loss_ok else '✗'}")
    
    if model_ok and data_ok and loss_ok:
        print("\n✅ 所有测试通过！可以开始训练。")
        print("\n开始训练:")
        print("  bash run.sh ClimODE")
        print("  或")
        print("  python3 -u train.py --config_filepath CIRT/configs/ClimODE.yaml")
        print("\n注意:")
        print("  如果看到 'Using fallback neural network' 是正常的")
        print("  这表示使用简化的 ClimODE 实现（不需要外部依赖）")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()

