"""
测试自动评估功能
"""

import os
import sys
import subprocess

def test_auto_evaluate_script():
    """测试自动评估脚本"""
    print("🧪 测试自动评估脚本...")
    
    try:
        # 测试帮助信息
        result = subprocess.run([
            "python3", "auto_evaluate.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 自动评估脚本帮助信息正常")
            print("📋 支持的模型类型:")
            if "CirT" in result.stdout:
                print("  - CirT")
            if "ClimODE" in result.stdout:
                print("  - ClimODE")
            if "ClimaX" in result.stdout:
                print("  - ClimaX")
            if "ViT" in result.stdout:
                print("  - ViT")
            if "EGNN" in result.stdout:
                print("  - EGNN")
            if "ours" in result.stdout:
                print("  - ours")
        else:
            print(f"❌ 自动评估脚本测试失败: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_config_files():
    """测试配置文件是否存在"""
    print("\n🧪 测试配置文件...")
    
    config_files = [
        "CIRT/configs/CirT.yaml",
        "CIRT/configs/ClimODE.yaml",
        "CIRT/configs/ClimaX.yaml",
        "CIRT/configs/ViT.yaml",
        "CIRT/configs/EGNN.yaml",
        "CIRT/configs/ours.yaml"
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - 不存在")
            all_exist = False
    
    return all_exist

def test_evaluation_scripts():
    """测试评估脚本是否存在"""
    print("\n🧪 测试评估脚本...")
    
    eval_scripts = [
        "inference/others/evaluate_cirt.py",
        "inference/others/evaluate_climax.py",
        "inference/others/evaluate_ours.py"
    ]
    
    all_exist = True
    for script in eval_scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - 不存在")
            all_exist = False
    
    return all_exist

def test_run_script():
    """测试run.sh脚本"""
    print("\n🧪 测试run.sh脚本...")
    
    try:
        # 测试帮助信息
        result = subprocess.run([
            "bash", "run.sh", "invalid_model"
        ], capture_output=True, text=True)
        
        if "Error: Invalid model type" in result.stderr:
            print("✅ run.sh脚本参数验证正常")
            return True
        else:
            print("⚠️  run.sh脚本参数验证可能有问题")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 自动评估功能测试")
    print("=" * 50)
    
    # 运行所有测试
    script_test = test_auto_evaluate_script()
    config_test = test_config_files()
    eval_test = test_evaluation_scripts()
    run_test = test_run_script()
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"自动评估脚本: {'✅ 通过' if script_test else '❌ 失败'}")
    print(f"配置文件: {'✅ 通过' if config_test else '❌ 失败'}")
    print(f"评估脚本: {'✅ 通过' if eval_test else '❌ 失败'}")
    print(f"run.sh脚本: {'✅ 通过' if run_test else '❌ 失败'}")
    
    all_passed = script_test and config_test and eval_test and run_test
    
    if all_passed:
        print("\n🎉 所有测试通过！自动评估功能已就绪。")
        print("\n🚀 使用方法:")
        print("  bash ./run.sh CirT      # 训练CirT并自动评估")
        print("  bash ./run.sh ClimODE   # 训练ClimODE并自动评估")
        print("  bash ./run.sh ClimaX    # 训练ClimaX并自动评估")
    else:
        print("\n❌ 部分测试失败，请检查相关文件。")
    
    return all_passed

if __name__ == "__main__":
    main()
