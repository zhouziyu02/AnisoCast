"""Evaluate a compatible SOON checkpoint on all samples in the test split."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Support script execution as well as `python -m inference.evaluate_soon`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
import lightning.pytorch as pl
from tqdm import tqdm

from CIRT import criterion, config
from CIRT.models.model import S2SBenchmarkModel


def resolve_hyperparameters(checkpoint, supplied=None):
    """Keep saved architecture and target definitions; allow test-data relocation."""
    supplied = supplied or {}
    saved = checkpoint.get('hyper_parameters', {})
    model_args = dict(saved.get('model_args', supplied.get('model_args', {})))
    data_args = dict(saved.get('data_args', supplied.get('data_args', {})))
    for key in ('data_dir', 'test_years', 'batch_size'):
        if key in supplied.get('data_args', {}):
            data_args[key] = supplied['data_args'][key]
    if 'num_workers' in supplied.get('model_args', {}):
        model_args['num_workers'] = supplied['model_args']['num_workers']
    if not model_args or not data_args:
        raise ValueError('Provide model_args/data_args in the checkpoint or configuration.')
    root = Path(data_args.get('data_dir', config.DATA_DIR)).expanduser()
    if not root.is_absolute():
        root = config.ABS_PATH / root
    data_args['data_dir'] = str(root.resolve())
    return model_args, data_args


def reverse_normalize(predict, data_args):
    """Apply the ordered scalar mean/sigma statistics used by S2SDataset."""
    root = Path(data_args['data_dir']) / 'climatology_1.5'
    means, sigmas = [], []
    for filename, params in (
        ('climatology_pressure_level_1.5_new.zarr',
         [f'{var}-{level}' for var in data_args['pred_pressure_vars'] for level in config.PRESSURE_LEVELS]),
        ('climatology_single_level_1.5_new.zarr', data_args['pred_single_vars']),
    ):
        if not params:
            continue
        with xr.open_dataset(root / filename, engine='zarr') as ds:
            means.append(ds['mean'].sel(param=params).values[:, None, None])
            sigmas.append(ds['sigma'].sel(param=params).values[:, None, None])
    mean = torch.as_tensor(np.concatenate(means), dtype=predict.dtype, device=predict.device)
    sigma = torch.as_tensor(np.concatenate(sigmas), dtype=predict.dtype, device=predict.device)
    return predict * sigma + mean


def calculate_metrics(all_pred, all_y, model_args, data_args, save_dir=None):
    all_pred = torch.as_tensor(all_pred).detach().cpu()
    all_y = torch.as_tensor(all_y).detach().cpu()
    features = [f'{var}-{level}' for var in data_args['pred_pressure_vars']
                for level in config.PRESSURE_LEVELS] + list(data_args['pred_single_vars'])
    if all_pred.shape != all_y.shape or all_pred.ndim != 5:
        raise ValueError('Metrics require matching [sample, window, variable, latitude, longitude] tensors.')
    if all_pred.shape[1] != 2 or all_pred.shape[2] != len(features):
        raise ValueError('Prediction dimensions differ from the configured two-window target variables.')
    metrics = {
        'RMSE': criterion.RMSE(), 'Bias': criterion.Bias(),
        'ACC': criterion.ACC(data_dir=data_args['data_dir']),
        'MS_SSIM': criterion.MS_SSIM(),
        'SpecDiv': criterion.SpectralDiv(percentile=0.9, is_train=False),
        'SpecRes': criterion.SpectralRes(percentile=0.9, is_train=False),
    }
    pressure_count = len(data_args['pred_pressure_vars']) * len(config.PRESSURE_LEVELS)
    rows = []
    with torch.no_grad():
        for step in range(2):
            for index, feature in enumerate(tqdm(features, desc=f'Target window {step}')):
                pred, target = all_pred[:, step, index], all_y[:, step, index]
                source = 'pressure_level' if index < pressure_count else 'single_level'
                row = {'steps': step, 'pred_vars': feature,
                       'target_start_day': data_args['lead_time'] + step * 14,
                       'target_end_day': data_args['lead_time'] + step * 14 + 13}
                for name, metric in metrics.items():
                    value = metric(pred, target, feature, source) if name == 'ACC' else metric(pred, target)
                    row[name] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def load_model_and_predict(model_args, data_args, checkpoint_path):
    checkpoint_model = S2SBenchmarkModel.load_from_checkpoint(
        str(checkpoint_path), model_args=model_args, data_args=data_args,
        map_location='cpu', strict=True,
    )
    checkpoint_model.setup('predict')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1, enable_progress_bar=True,
                         enable_model_summary=False, logger=False)
    predictions = trainer.predict(checkpoint_model, dataloaders=checkpoint_model.test_dataloader())
    if not predictions:
        raise ValueError('No predictions were produced for the requested test split.')
    all_pred = torch.cat([pred.detach().cpu() for pred, _, _ in predictions], dim=0)
    all_y = torch.cat([target.detach().cpu() for _, target, _ in predictions], dim=0)
    if all_pred.shape[0] != len(checkpoint_model.test_dataset):
        raise ValueError('Prediction count does not cover every sample in the test split.')
    return reverse_normalize(all_pred, data_args), reverse_normalize(all_y, data_args)


def main(args):
    pl.seed_everything(42)
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    supplied = None
    if args.config_filepath:
        with open(args.config_filepath) as stream:
            supplied = yaml.safe_load(stream)
    model_args, data_args = resolve_hyperparameters(checkpoint, supplied)
    print(f"Evaluating {model_args['model_name']} on years {data_args['test_years']}")
    print(f"Data directory: {data_args['data_dir']}")
    all_pred, all_y = load_model_and_predict(model_args, data_args, checkpoint_path)
    metrics = calculate_metrics(all_pred, all_y, model_args, data_args)
    save_dir = Path(args.output_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = args.tag.replace('/', '_').replace('\\', '_') if args.tag else ''
    suffix = f'_{safe_tag}' if safe_tag else ''
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_path = save_dir / f"{model_args['model_name']}_metrics_{stamp}{suffix}.csv"
    metrics.to_csv(csv_path, index=False)
    print(f'Metrics for {all_pred.shape[0]} test samples saved to: {csv_path.resolve()}')
    return csv_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_path', required=True, help='Path to a compatible trained checkpoint')
    parser.add_argument('--config_filepath', help='Optional YAML overriding data_dir, test_years, batch_size and num_workers')
    parser.add_argument('--output_dir', default='./results/soon', help='Directory for metrics CSV output')
    parser.add_argument('--tag', default=None, help='Optional metrics filename suffix')
    main(parser.parse_args())
