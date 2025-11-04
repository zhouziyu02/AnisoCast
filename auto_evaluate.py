"""
自动评估脚本
支持CirT、ClimODE、ClimaX、ViT、EGNN等模型的自动评估
"""

import os
import sys
import argparse
import yaml
import glob
from pathlib import Path
import subprocess

def find_latest_checkpoint(model_name):
    """
    查找最新的checkpoint文件
    """
    checkpoint_dir = "lightning_logs"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 未找到checkpoint目录: {checkpoint_dir}")
        return None
    
    # 查找最新的version目录
    version_dirs = glob.glob(os.path.join(checkpoint_dir, "version_*"))
    if not version_dirs:
        print(f"❌ 未找到version目录")
        return None
    
    # 按版本号排序，获取最新的
    latest_version = sorted(version_dirs, key=lambda x: int(x.split('_')[-1]))[-1]
    print(f"📁 最新版本目录: {latest_version}")
    
    # 查找checkpoint文件
    checkpoint_patterns = [
        "best.ckpt",
        "last.ckpt",
        "epoch=*.ckpt"
    ]
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(os.path.join(latest_version, "checkpoints", pattern))
        if checkpoints:
            # 按修改时间排序，获取最新的
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"✅ 找到checkpoint: {latest_checkpoint}")
            return latest_checkpoint
    
    print(f"❌ 在 {latest_version}/checkpoints/ 中未找到checkpoint文件")
    return None

def evaluate_cirt(config_file, checkpoint_path):
    """
    评估CirT模型
    """
    print("🎯 开始CirT模型评估...")
    
    eval_script = "inference/others/evaluate_cirt.py"
    if not os.path.exists(eval_script):
        print(f"❌ 评估脚本不存在: {eval_script}")
        return False
    
    cmd = [
        "python3", eval_script,
        "--config_filepath", config_file,
        "--checkpoint_path", checkpoint_path
    ]
    
    print(f"🚀 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ CirT评估完成！")
        print("📁 结果保存在: ./results/CirT/")
        print("📊 包含指标: RMSE, MAE, Bias, R², ACC, MS-SSIM, SpectralDiv, SpectralRes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ CirT评估失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def evaluate_climode(config_file, checkpoint_path):
    """
    评估ClimODE模型
    """
    print("🎯 开始ClimODE模型评估...")
    
    # ClimODE使用与CirT相同的评估脚本
    return evaluate_cirt(config_file, checkpoint_path)

def evaluate_climax(config_file, checkpoint_path):
    """
    评估ClimaX模型
    """
    print("🎯 开始ClimaX模型评估...")
    
    eval_script = "inference/others/evaluate_climax.py"
    if not os.path.exists(eval_script):
        print(f"❌ 评估脚本不存在: {eval_script}")
        return False
    
    cmd = [
        "python3", eval_script,
        "--config_filepath", config_file,
        "--checkpoint_path", checkpoint_path
    ]
    
    print(f"🚀 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ ClimaX评估完成！")
        print("📁 结果保存在: ./results/ClimaX/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ClimaX评估失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def evaluate_vit(config_file, checkpoint_path):
    """
    评估ViT模型
    """
    print("🎯 开始ViT模型评估...")
    
    # ViT使用与CirT相同的评估脚本
    return evaluate_cirt(config_file, checkpoint_path)

def evaluate_egnn(config_file, checkpoint_path):
    """
    评估EGNN模型
    """
    print("🎯 开始EGNN模型评估...")
    
    # EGNN使用与CirT相同的评估脚本
    return evaluate_cirt(config_file, checkpoint_path)

def evaluate_fno(config_file, checkpoint_path):
    """
    评估FNO模型（使用与CirT相同的评估脚本）
    """
    print("🎯 开始FNO模型评估...")
    return evaluate_fno(config_file, checkpoint_path)

def auto_evaluate(model_type, config_file, checkpoint_path=None):
    """
    自动评估指定模型
    """
    print(f"🚀 开始自动评估 {model_type} 模型...")
    
    # 如果没有提供checkpoint路径，自动查找
    if not checkpoint_path:
        print("🔍 自动查找最新checkpoint...")
        checkpoint_path = find_latest_checkpoint(model_type)
        if not checkpoint_path:
            print("❌ 未找到checkpoint文件")
            return False
    
    # 验证checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return False
    
    print(f"✅ 使用checkpoint: {checkpoint_path}")
    
    # 根据模型类型选择评估函数
    evaluators = {
        'CirT': evaluate_cirt,
        'ClimODE': evaluate_climode,
        'ClimaX': evaluate_climax,
        'ViT': evaluate_vit,
        'EGNN': evaluate_egnn,
        'FNO': evaluate_fno,
    }
    
    if model_type not in evaluators:
        print(f"❌ 不支持的模型类型: {model_type}")
        print(f"支持的模型: {', '.join(evaluators.keys())}")
        return False
    
    # 运行评估
    evaluator = evaluators[model_type]
    success = evaluator(config_file, checkpoint_path)
    
    if success:
        print(f"🎉 {model_type} 模型评估完成！")
    else:
        print(f"❌ {model_type} 模型评估失败")
    
    return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动评估模型')
    parser.add_argument('--model_type', required=True, 
                       choices=['CirT', 'ClimODE', 'ClimaX', 'ViT', 'EGNN', 'FNO'],
                       help='模型类型')
    parser.add_argument('--config_file', required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint_path', 
                       help='checkpoint文件路径（可选，不提供则自动查找）')
    
    args = parser.parse_args()
    
    # 验证配置文件是否存在
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        sys.exit(1)
    
    # 运行自动评估
    success = auto_evaluate(args.model_type, args.config_file, args.checkpoint_path)
    
    if success:
        print("\n✅ 自动评估完成！")
        sys.exit(0)
    else:
        print("\n❌ 自动评估失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
