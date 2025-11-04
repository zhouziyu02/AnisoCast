 
import argparse
import yaml
import torch
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ['WANDB_MODE'] = 'disabled'

pl.seed_everything(42)
from CIRT.models import model
from CIRT import dataset, config, criterion
from datetime import datetime

test_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def reverse_normalize(predict, data_args):
    """
    反归一化预测结果
    """
    if not data_args.get('is_normalized', True):
        print("📊 数据未归一化，跳过反归一化步骤")
        return predict
    
    print("📊 开始反归一化...")
    
    # 构建climatology文件路径
    normalization_file = [
        Path(data_args['data_dir']) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
        Path(data_args['data_dir']) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
    ]
    
    pred_single_vars = data_args['pred_single_vars']
    pred_pressure_vars = data_args['pred_pressure_vars']
    
    try:
        # 加载pressure level的均值和标准差
        mean_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['mean'].sel(
            param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:,
                                                np.newaxis, np.newaxis])
        sigma_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['sigma'].sel(
            param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:,
                                                 np.newaxis, np.newaxis])
        
        # 加载single level的均值和标准差
        mean_single_level_merge_pred = torch.tensor(
            xr.open_dataset(normalization_file[1], engine='zarr')['mean'].sel(param=pred_single_vars).values[:, np.newaxis,
            np.newaxis])
        sigma_single_level_merge_pred = torch.tensor(
            xr.open_dataset(normalization_file[1], engine='zarr')['sigma'].sel(param=pred_single_vars).values[:, np.newaxis,
            np.newaxis])
        
        # 拼接均值和标准差
        mean = torch.cat((mean_pressure_level_pred, mean_single_level_merge_pred), dim=0)
        sigma = torch.cat((sigma_pressure_level_pred, sigma_single_level_merge_pred), dim=0)
        
        # 反归一化: predict = (predict * sigma) + mean
        predict = predict * sigma + mean
        print("✅ 反归一化完成")
        
    except Exception as e:
        print(f"⚠️ 反归一化失败: {e}")
        print("📊 使用原始预测值")
    
    return predict

def calculate_full_metrics(all_pred, all_y, model_args, data_args, save_dir):
    """
    计算完整的评估指标
    """
    print("📊 开始计算完整评估指标...")
    
    # 转换为PyTorch张量
    if isinstance(all_pred, np.ndarray):
        all_pred = torch.from_numpy(all_pred)
    if isinstance(all_y, np.ndarray):
        all_y = torch.from_numpy(all_y)
    
    device = torch.device('cpu')
    all_pred = all_pred.to(device)
    all_y = all_y.to(device)

    # 构建特征名称
    feature_names = []
    for param in data_args['pred_pressure_vars']:
        for level in config.PRESSURE_LEVELS:
            feature_names.append(f"{param}-{level}")
    feature_names += data_args['pred_single_vars']

    # 初始化评估指标
    RMSE = criterion.RMSE(lat_adjusted = False)
    Bias = criterion.Bias()
    MAE = criterion.MAE()
    R2 = criterion.R2()
    ACC = criterion.ACC()
    MS_SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)

    # 创建结果DataFrame
    columns = ['steps', 'pred_vars', 'RMSE', 'MAE', 'Bias', 'R2', 'ACC', 'MS_SSIM', 'SpecDiv', 'SpecRes']
    df = pd.DataFrame(columns=columns)
    
    S, steps, V, H, W = all_pred.shape
    print(f"📊 数据形状: {all_pred.shape}")
    print(f"📊 预测步数: {steps}, 变量数: {V}")

    # 计算每个时间步和每个变量的指标
    for step_idx in range(steps):
        pred_step_idx = all_pred[:, step_idx]
        y_step_idx = all_y[:, step_idx]

        print(f"📊 处理预测步 {step_idx + 1}/{steps}")
        
        for i, feature_name in enumerate(tqdm(feature_names, desc=f"Step {step_idx + 1}")):
            # 判断数据源
            if '-' in feature_name:
                source = "pressure_level"
            else:
                source = "single_level"

            y_hat = pred_step_idx[:, i, :, :]
            y = y_step_idx[:, i, :, :]

            # 计算各种指标
            try:
                rmse_val = RMSE(y_hat, y)
                mae_val = MAE(y_hat, y)
                bias_val = Bias(y_hat, y)
                r2_val = R2(y_hat, y)
                acc_val = ACC(y_hat, y, feature_name, source)
                ms_ssim_val = MS_SSIM(y_hat, y)
                spec_div_val = SpecDiv(y_hat, y)
                spec_res_val = SpecRes(y_hat, y)

                row_data = {
                    'steps': step_idx,
                    'pred_vars': feature_name,
                    'RMSE': f"{rmse_val:.4f}",
                    'MAE': f"{mae_val:.4f}",
                    'Bias': f"{bias_val:.4f}",
                    'R2': f"{r2_val:.4f}",
                    'ACC': f"{acc_val:.4f}",
                    'MS_SSIM': f"{ms_ssim_val:.4f}",
                    'SpecDiv': f"{spec_div_val:.4f}",
                    'SpecRes': f"{spec_res_val:.4f}",
                }
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                
            except Exception as e:
                print(f"⚠️ 计算指标失败 {feature_name}: {e}")
                # 添加默认值
                row_data = {
                    'steps': step_idx,
                    'pred_vars': feature_name,
                    'RMSE': "0.0000",
                    'MAE': "0.0000",
                    'Bias': "0.0000",
                    'R2': "0.0000",
                    'ACC': "0.0000",
                    'MS_SSIM': "0.0000",
                    'SpecDiv': "0.0000",
                    'SpecRes': "0.0000",
                }
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # 清理内存
        del pred_step_idx, y_step_idx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 保存结果到CSV
    csv_path = f"{save_dir}/{model_args['model_name']}_full_metrics_{test_time}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 完整指标结果已保存到: {csv_path}")

    # 计算并打印汇总统计
    print("\n📊 汇总统计:")
    for step in range(steps):
        step_data = df[df['steps'] == step]
        avg_rmse = step_data['RMSE'].astype(float).mean()
        avg_mae = step_data['MAE'].astype(float).mean()
        avg_r2 = step_data['R2'].astype(float).mean()
        avg_acc = step_data['ACC'].astype(float).mean()
        avg_ms_ssim = step_data['MS_SSIM'].astype(float).mean()
        print(f"   步 {step + 1}: 平均RMSE={avg_rmse:.4f}, 平均MAE={avg_mae:.4f}, 平均R²={avg_r2:.4f}, 平均ACC={avg_acc:.4f}, 平均MS-SSIM={avg_ms_ssim:.4f}")

    return df

def main(args):
    """
    主函数：加载模型，生成预测，计算指标
    """
    print("🚀 开始EIMP(EGNN)模型完整评估...")
    
    # 加载配置文件
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
    
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    print(f"📊 模型名称: {model_args['model_name']}")
    print(f"📊 测试年份: {data_args['test_years']}")
    print(f"📊 数据目录: {data_args['data_dir']}")

    # 创建结果保存目录
    save_dir = Path(f"./results/{model_args['model_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果保存目录: {save_dir}")

    # 加载训练好的模型
    print("📦 加载训练好的模型...")
    checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    try:
        model_checkpoint = model.S2SBenchmarkModel.load_from_checkpoint(
            str(checkpoint_path), 
            model_args=model_args, 
            data_args=data_args
        )
        print(f"✅ 模型加载成功: {checkpoint_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 设置数据加载器
    model_checkpoint.setup()
    print(f"✅ 数据集设置完成")
    print(f"   测试样本数: {len(model_checkpoint.test_dataset)}")

    # 初始化训练器
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # 生成预测
    print("🔮 开始生成预测...")
    try:
        predictions = trainer.predict(model_checkpoint, dataloaders=model_checkpoint.test_dataloader())
        print("✅ 预测生成完成")
    except Exception as e:
        print(f"❌ 预测生成失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 收集所有预测结果
    all_pred_list = []
    all_y_list = []
    all_timestamp_list = []
    
    for pred, y, timestamp in predictions:
        all_pred_list.append(pred)
        all_y_list.append(y)
        all_timestamp_list.append(timestamp)
    
    all_pred = torch.cat(all_pred_list, dim=0)
    all_y = torch.cat(all_y_list, dim=0)
    
    print(f"📊 预测形状: {all_pred.shape}")
    print(f"📊 真实值形状: {all_y.shape}")

    # 反归一化
    all_pred = reverse_normalize(all_pred, data_args)
    all_y = reverse_normalize(all_y, data_args)

    # 保存原始预测结果
    pred_save_path = f"{save_dir}/{model_args['model_name']}_pred_{test_time}.npy"
    y_save_path = f"{save_dir}/{model_args['model_name']}_y_{test_time}.npy"
    
    np.save(pred_save_path, all_pred.numpy() if isinstance(all_pred, torch.Tensor) else all_pred)
    np.save(y_save_path, all_y.numpy() if isinstance(all_y, torch.Tensor) else all_y)
    print(f"✅ 预测结果已保存:")
    print(f"   预测: {pred_save_path}")
    print(f"   真实值: {y_save_path}")

    # 计算完整评估指标
    if args.calculate_metrics:
        metrics_df = calculate_full_metrics(all_pred, all_y, model_args, data_args, save_dir)
    
    print("🎉 完整评估完成!")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description='EIMP模型完整评估脚本')
    parser.add_argument('--config_filepath', 
                       default='CIRT/configs/CirT.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint_path', 
                       default='logs/CirT/lightning_logs/version_19/checkpoints/epoch=6-step=385.ckpt',
                       help='模型检查点路径')
    parser.add_argument('--save_predictions', 
                       action='store_true',
                       default=True,
                       help='是否保存预测结果')
    parser.add_argument('--calculate_metrics', 
                       action='store_true', 
                       default=True,
                       help='是否计算评估指标')
    
    args = parser.parse_args()
    main(args)