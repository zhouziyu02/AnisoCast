import argparse
import yaml
import torch
import xarray as xr
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ['WANDB_MODE'] = 'disabled'

# Set random seed for reproducibility
pl.seed_everything(42)

# Import CirT modules
from CIRT.models import model
from CIRT.utils import *
from CIRT import criterion, config

test_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def reverse_normalize(predict, data_args):
    """
    Reverse normalization for predictions and ground truth
    """
    normalization_file = [
        Path('/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S') / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
        Path('/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S') / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
    ]
    # /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT_old/data/S2S/climatology_1.5/climatology_pressure_level_1.5_new.zarr
    pred_single_vars = data_args['pred_single_vars']  # 修正变量名
    pred_pressure_vars = data_args['pred_pressure_vars']  # 修正变量名

    mean_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['mean'].sel(
        param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:,
                                            np.newaxis, np.newaxis])
    mean_single_level_merge_pred = torch.tensor(
        xr.open_dataset(normalization_file[1], engine='zarr')['mean'].sel(param=pred_single_vars).values[:, np.newaxis,
        np.newaxis])
    sigma_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['sigma'].sel(
        param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:,
                                             np.newaxis, np.newaxis])
    sigma_single_level_merge_pred = torch.tensor(
        xr.open_dataset(normalization_file[1], engine='zarr')['sigma'].sel(param=pred_single_vars).values[:, np.newaxis,
        np.newaxis])
    mean = torch.cat((mean_pressure_level_pred, mean_single_level_merge_pred), dim=0)
    sigma = torch.cat((sigma_pressure_level_pred, sigma_single_level_merge_pred), dim=0)
    predict = predict * sigma + mean
    return predict


def calculate_metrics(all_pred, all_y, model_args, data_args, save_dir):
    """
    Calculate comprehensive metrics for model predictions
    """
    # Convert to PyTorch tensors if needed
    if isinstance(all_pred, np.ndarray):
        all_pred = torch.from_numpy(all_pred)
    if isinstance(all_y, np.ndarray):
        all_y = torch.from_numpy(all_y)
    
    device = torch.device('cpu')
    all_pred = all_pred.to(device)
    all_y = all_y.to(device)

    # Define features relationships
    feature_names = []
    for param in data_args['pred_pressure_vars']:
        for level in config.PRESSURE_LEVELS:
            feature_names.append(f"{param}-{level}")
    feature_names += data_args['pred_single_vars']

    # Initialize metric calculators
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    MS_SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)

    # Create results DataFrame
    columns = ['steps', 'pred_vars', 'RMSE', 'Bias', 'ACC', 'MS_SSIM', 'SpecDiv', 'SpecRes']
    df = pd.DataFrame(columns=columns)
    S, steps, V, H, W = all_pred.shape

    print(f"Calculating metrics for {S} samples, {steps} steps, {V} variables...")

    for step_idx in range(steps):
        pred_step_idx = all_pred[:, step_idx]
        y_step_idx = all_y[:, step_idx]

        print(f"\nProcessing forecasting step: {step_idx + 1}/{steps}")
        for i, feature_name in enumerate(tqdm(feature_names, desc=f"Step {step_idx + 1}")):
            if '-' in feature_name:
                source = "pressure_level"  # 修正为ACC类期望的键名
            else:
                source = "single_level"    # 修正为ACC类期望的键名

            y_hat = pred_step_idx[:, i, :, :]
            y = y_step_idx[:, i, :, :]

            # Calculate metrics
            rmse_val = RMSE(y_hat, y)
            bias_val = Bias(y_hat, y)
            acc_val = ACC(y_hat, y, feature_name, source)
            ms_ssim_val = MS_SSIM(y_hat, y)
            spec_div_val = SpecDiv(y_hat, y)
            spec_res_val = SpecRes(y_hat, y)

            row_data = {
                'steps': step_idx,
                'pred_vars': feature_name,
                'RMSE': f"{rmse_val:.4f}",
                'Bias': f"{bias_val:.4f}",
                'ACC': f"{acc_val:.4f}",
                'MS_SSIM': f"{ms_ssim_val:.4f}",
                'SpecDiv': f"{spec_div_val:.4f}",
                'SpecRes': f"{spec_res_val:.4f}",
            }
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # Free memory after processing each step
        del pred_step_idx, y_step_idx
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return df


def load_model_and_predict(model_args, data_args, checkpoint_path):
    """
    Load model from checkpoint and generate predictions
    """
    print("Loading model from checkpoint...")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Load checkpoint to get saved hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters from checkpoint if available
    # Priority: checkpoint hyperparameters > provided config file
    if 'hyper_parameters' in checkpoint:
        saved_hyperparams = checkpoint['hyper_parameters']
        # Use saved model_args and data_args if available, otherwise use provided ones
        if 'model_args' in saved_hyperparams:
            saved_model_name = saved_hyperparams['model_args'].get('model_name', 'unknown')
            print(f"⚠️  Found saved model_args in checkpoint with model_name: {saved_model_name}")
            print(f"⚠️  Overriding config file model_name: {model_args.get('model_name', 'unknown')}")
            model_args = saved_hyperparams['model_args']
        if 'data_args' in saved_hyperparams:
            print(f"⚠️  Using data_args from checkpoint (overriding config file)")
            # Merge data_args: use saved ones, but allow config file to override test_years if needed
            saved_data_args = saved_hyperparams['data_args']
            # Keep test_years from config if explicitly provided, otherwise use checkpoint
            if 'test_years' in data_args:
                saved_data_args['test_years'] = data_args['test_years']
            data_args = saved_data_args
    
    print(f"Model name: {model_args['model_name']}")
    print(f"Test years: {data_args['test_years']}")

    # Load model from checkpoint
    model_checkpoint = model.S2SBenchmarkModel.load_from_checkpoint(
        str(checkpoint_path), 
        model_args=model_args, 
        data_args=data_args,
        strict=False  # Allow partial loading if there are minor mismatches
    )

    # Set up the dataloaders
    model_checkpoint.setup()

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("Generating predictions...")
    # Generate predictions
    predictions = trainer.predict(model_checkpoint, dataloaders=model_checkpoint.test_dataloader())

    # Concatenate all predictions
    all_pred_list = []
    all_y_list = []
    all_timestamp_list = []
    
    for pred, y, timestamp in predictions:
        all_pred_list.append(pred)
        all_y_list.append(y)
        all_timestamp_list.append(timestamp)
    
    all_pred = torch.cat(all_pred_list, dim=0)
    all_y = torch.cat(all_y_list, dim=0)
    
    print(f'Prediction shape: {all_pred.shape}')
    print(f'Ground truth shape: {all_y.shape}')

    # Reverse normalization
    print("Reversing normalization...")
    all_pred = reverse_normalize(all_pred, data_args)
    all_y = reverse_normalize(all_y, data_args)
    
    assert all_pred.shape == all_y.shape
    print(f'Final shapes - Pred: {all_pred.shape}, GT: {all_y.shape}')

    return all_pred, all_y


def main(args):
    """
    Main function to evaluate model and generate metrics CSV
    """
    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Try to infer from config file
        if args.config_filepath:
            with open(args.config_filepath, 'r') as config_file:
                hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
                model_name = hyperparams.get('model_args', {}).get('model_name', 'ours')
            checkpoint_path = f"./checkpoints/{model_name}/best.ckpt"
        else:
            raise ValueError("Either --checkpoint_path or --config_filepath must be provided")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # 首先从checkpoint的hyper_parameters中读取model_name（最准确，因为这是训练时实际使用的）
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    final_model_name = None
    
    if 'hyper_parameters' in checkpoint and 'model_args' in checkpoint['hyper_parameters']:
        final_model_name = checkpoint['hyper_parameters']['model_args'].get('model_name')
        if final_model_name:
            print(f"📌 从checkpoint的hyper_parameters读取model_name: {final_model_name}")
    
    # 优先从checkpoint目录读取配置（解决并行训练时的参数读取混乱问题）
    checkpoint_dir = Path(checkpoint_path).parent.parent  # checkpoint在version_X/checkpoints/下
    config_in_checkpoint = checkpoint_dir / 'config.yaml'
    
    if config_in_checkpoint.exists():
        print(f"📋 从checkpoint目录读取配置: {config_in_checkpoint}")
        with open(config_in_checkpoint, 'r') as config_file:
            hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    elif args.config_filepath and os.path.exists(args.config_filepath):
        print(f"⚠️  checkpoint目录中未找到config.yaml，使用提供的配置文件: {args.config_filepath}")
        with open(args.config_filepath, 'r') as config_file:
            hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"无法找到配置文件。请确保checkpoint目录包含config.yaml，或提供--config_filepath")
    
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    # 确定最终的model_name：优先使用checkpoint的hyper_parameters中的，否则使用配置文件中的
    if final_model_name:
        config_model_name = model_args.get('model_name', 'unknown')
        if final_model_name != config_model_name:
            print(f"✅ 使用checkpoint中保存的model_name: {final_model_name} (覆盖配置文件中的: {config_model_name})")
        else:
            print(f"✅ 使用checkpoint中保存的model_name: {final_model_name} (与配置文件一致)")
        model_args['model_name'] = final_model_name
    else:
        final_model_name = model_args.get('model_name', 'ours')
        print(f"⚠️  checkpoint中未找到model_name，使用配置文件中的: {final_model_name}")
        model_args['model_name'] = final_model_name

    # Load model and generate predictions
    all_pred, all_y = load_model_and_predict(model_args, data_args, checkpoint_path)
    
    # 再次确认model_name（load_model_and_predict可能会修改model_args，但我们已经确定了final_model_name）
    # 确保使用训练时保存的model_name，而不是load_model_and_predict内部可能修改的值
    model_args['model_name'] = final_model_name
    
    # Create save directory using the final model_name
    save_dir = Path(f"./results/{final_model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果将保存到: {save_dir.absolute()}")

    # Calculate metrics
    print("\nCalculating comprehensive metrics...")
    metrics_df = calculate_metrics(all_pred, all_y, model_args, data_args, save_dir)

    # Save results
    # 如果提供了tag，在文件名中包含tag
    if args.tag:
        csv_filename = f"{model_args['model_name']}_metrics_{test_time}_{args.tag}.csv"
    else:
        csv_filename = f"{model_args['model_name']}_metrics_{test_time}.csv"
    csv_path = save_dir / csv_filename
    metrics_df.to_csv(csv_path, index=False)
    
    # 使用绝对路径并明确输出
    csv_abs_path = os.path.abspath(csv_path)
    print(f"\n{'='*60}")
    print(f"📄 Results saved to: {csv_abs_path}")
    print(f"📊 Total metrics calculated: {len(metrics_df)}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for step in metrics_df['steps'].unique():
        step_data = metrics_df[metrics_df['steps'] == step]
        print(f"\nStep {step + 1}:")
        print(f"  Average RMSE: {step_data['RMSE'].astype(float).mean():.4f}")
        print(f"  Average ACC: {step_data['ACC'].astype(float).mean():.4f}")
        print(f"  Average MS_SSIM: {step_data['MS_SSIM'].astype(float).mean():.4f}")

    # Clean up memory
    del all_pred, all_y
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model and generate metrics CSV')
    parser.add_argument('--config_filepath', 
                       default='CIRT/configs/ours.yaml',
                       help='Path to configuration YAML file (model_args/data_args will be overridden by checkpoint if available)')
    parser.add_argument('--checkpoint_path', 
                       default=None,
                       help='Path to model checkpoint (default: ./checkpoints/ours/best.ckpt)')
    parser.add_argument('--output_dir',
                       default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--tag',
                       default=None,
                       help='Custom tag to append to CSV filename (e.g., --tag exp1)')
    
    args = parser.parse_args()
    
    try:
        result_path = main(args)
        result_abs_path = os.path.abspath(result_path)
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📄 Metrics CSV file: {result_abs_path}")
        print(f"📁 Directory: {os.path.dirname(result_abs_path)}")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise



