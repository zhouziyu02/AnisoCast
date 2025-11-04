#!/usr/bin/env python3
"""
快速测试 ClimODE loss 修复
验证模型初始化和前向传播是否正常
"""

import torch
import sys
sys.path.append('/Users/bytedance/Desktop/ziyu_cli')

from CIRT.models.climode import create_climode_model

def test_climode_initialization():
    print("=" * 60)
    print("测试 ClimODE 模型初始化和前向传播")
    print("=" * 60)
    
    # 模拟配置
    model_args = {
        'hidden_size': 256,
        'drop_rate': 0.1,
        'solver': 'euler',
        'use_att': True,
        'use_err': True,
        'use_pos': False
    }
    
    input_size = 63
    output_size = 63
    num_steps = 2
    
    # 创建模型
    print("\n1. 创建模型...")
    model = create_climode_model(model_args, input_size, output_size, num_steps)
    print(f"   ✓ 模型类型: {type(model).__name__}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ 总参数数: {total_params:,}")
    print(f"   ✓ 可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print("\n2. 测试前向传播...")
    batch_size = 4
    x = torch.randn(batch_size, input_size, 121, 240)
    print(f"   输入形状: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   输出形状: {output.shape}")
    
    # 检查形状
    expected_shape = (batch_size, num_steps, output_size, 121, 240)
    if output.shape == expected_shape:
        print(f"   ✓ 输出形状正确: {output.shape}")
    else:
        print(f"   ✗ 输出形状错误！预期 {expected_shape}, 实际 {output.shape}")
        return False
    
    # 检查输出范围
    print("\n3. 检查输出数值范围...")
    output_min = output.min().item()
    output_max = output.max().item()
    output_mean = output.mean().item()
    output_std = output.std().item()
    
    print(f"   输出范围: [{output_min:.4f}, {output_max:.4f}]")
    print(f"   输出均值: {output_mean:.4f}")
    print(f"   输出标准差: {output_std:.4f}")
    
    # 检查权重初始化
    print("\n4. 检查权重初始化...")
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break
    
    if first_conv is not None:
        weight_mean = first_conv.weight.data.mean().item()
        weight_std = first_conv.weight.data.std().item()
        print(f"   第一层卷积权重均值: {weight_mean:.6f}")
        print(f"   第一层卷积权重标准差: {weight_std:.6f}")
        
        if abs(weight_mean) < 0.1 and 0.01 < weight_std < 0.5:
            print(f"   ✓ 权重初始化合理")
        else:
            print(f"   ⚠ 权重初始化可能不理想")
    
    # 模拟损失计算
    print("\n5. 模拟损失计算...")
    # 模拟归一化的目标数据
    y = torch.randn(batch_size, num_steps, output_size, 121, 240) * 2  # 模拟归一化后的数据范围
    
    mse_loss = torch.nn.functional.mse_loss(output, y)
    print(f"   MSE Loss: {mse_loss.item():.6f}")
    
    if mse_loss.item() < 100:
        print(f"   ✓ Loss 在合理范围内 (< 100)")
    else:
        print(f"   ✗ Loss 过大 (> 100)，可能存在问题")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！ClimODE 模型初始化和前向传播正常")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_climode_initialization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

