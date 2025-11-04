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
    Calculate comprehensive metrics for CirT model predictions
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
    Load CirT model from checkpoint and generate predictions
    """
    print("Loading CirT model from checkpoint...")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Model name: {model_args['model_name']}")
    print(f"Test years: {data_args['test_years']}")

    # Load model from checkpoint
    model_checkpoint = model.S2SBenchmarkModel.load_from_checkpoint(
        str(checkpoint_path), 
        model_args=model_args, 
        data_args=data_args
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

    # Align spatial dimensions if they differ (e.g., ViT 120 vs GT 121)
    if all_pred.dim() == 5 and all_y.dim() == 5:
        ph, pw = all_pred.shape[-2], all_pred.shape[-1]
        yh, yw = all_y.shape[-2], all_y.shape[-1]
        if (ph != yh) or (pw != yw):
            all_y = all_y[..., :ph, :pw]
            print(f'Aligned GT to prediction spatial size: {all_y.shape[-2:]}')

    # Reverse normalization
    print("Reversing normalization...")
    all_pred = reverse_normalize(all_pred, data_args)
    all_y = reverse_normalize(all_y, data_args)
    
    assert all_pred.shape == all_y.shape
    print(f'Final shapes - Pred: {all_pred.shape}, GT: {all_y.shape}')

    return all_pred, all_y


def main(args):
    """
    Main function to evaluate CirT model and generate metrics CSV
    """
    # Load configuration
    with open(args.config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    # Create save directory
    save_dir = Path(f"./results/{model_args['model_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default checkpoint path
        checkpoint_path = f"./checkpoints/{model_args['model_name']}/best.ckpt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load model and generate predictions
    all_pred, all_y = load_model_and_predict(model_args, data_args, checkpoint_path)

    # Calculate metrics
    print("\nCalculating comprehensive metrics...")
    metrics_df = calculate_metrics(all_pred, all_y, model_args, data_args, save_dir)

    # Save results
    csv_filename = f"{model_args['model_name']}_metrics_{test_time}.csv"
    csv_path = save_dir / csv_filename
    metrics_df.to_csv(csv_path, index=False)

    # Save predictions and ground truth as .npy in the same directory
    pred_npy_path = save_dir / f"{model_args['model_name']}_pred_{test_time}.npy"
    gt_npy_path = save_dir / f"{model_args['model_name']}_gt_{test_time}.npy"
    np.save(pred_npy_path, all_pred.detach().cpu().numpy())
    np.save(gt_npy_path, all_y.detach().cpu().numpy())
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Predictions saved to: {pred_npy_path}")
    print(f"Ground truth saved to: {gt_npy_path}")
    print(f"Total metrics calculated: {len(metrics_df)}")
    
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
    parser = argparse.ArgumentParser(description='Evaluate ClimaX model and generate metrics CSV')
    parser.add_argument('--config_filepath', 
                       default='CIRT/configs/ClimaX.yaml',
                       help='Path to ClimaX configuration YAML file')
    parser.add_argument('--checkpoint_path', 
                       default='/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/lightning_logs/version_86/checkpoints/epoch=13-step=770.ckpt',
                       help='Path to ClimaX model checkpoint (default: ./checkpoints/CirT/best.ckpt)')
    parser.add_argument('--output_dir',
                       default='./results',
                       help='Output directory for results (default: ./results)')
    
    args = parser.parse_args()
    
    try:
        result_path = main(args)
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Metrics CSV saved to: {result_path}")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise



