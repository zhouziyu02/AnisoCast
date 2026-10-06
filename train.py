"""Train AnisoCast from an explicit YAML configuration."""
import argparse
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from CIRT.models.model import S2SBenchmarkModel


def main(args):
    pl.seed_everything(42, workers=True)
    with Path(args.config_filepath).open() as handle:
        hyperparams = yaml.safe_load(handle)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    # Resolve data paths once so checkpoints work when the caller changes directory.
    data_root = Path(data_args['data_dir']).expanduser()
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent / data_root
    data_args['data_dir'] = str(data_root.resolve())
    run_dir = Path(args.run_dir or Path(args.config_filepath).resolve().parent).resolve()
    checkpoint_dir = run_dir / 'checkpoints'
    if checkpoint_dir.exists() and any(checkpoint_dir.glob('*.ckpt')):
        raise FileExistsError(f'Checkpoints already exist in {checkpoint_dir}; choose another --run-dir')
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = args.accelerator or os.environ.get('ACCELERATOR') or ('gpu' if torch.cuda.is_available() else 'cpu')
    devices = args.devices if args.devices is not None else int(os.environ.get('NP', 1))
    precision = args.precision or ('16-mixed' if accelerator == 'gpu' else '32-true')
    strategy = args.strategy or ('ddp_find_unused_parameters_true' if devices > 1 else 'auto')
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename='best', monitor='val_loss',
                                 mode='min', save_top_k=1, save_last=False)
    loggers = [CSVLogger(save_dir=str(run_dir), name='metrics', version='')]
    if args.use_tensorboard:
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name='tensorboard', version=''))
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, strategy=strategy,
                         precision=precision, max_epochs=int(model_args['epochs']),
                         callbacks=[checkpoint], logger=loggers,
                         gradient_clip_val=float(model_args.get('grad_clip_norm', 0.0)),
                         gradient_clip_algorithm='norm')
    if trainer.is_global_zero:
        with (run_dir / 'config.yaml').open('w') as handle:
            yaml.safe_dump(hyperparams, handle, sort_keys=False)
    trainer.fit(S2SBenchmarkModel(model_args=model_args, data_args=data_args))
    if trainer.is_global_zero:
        if not checkpoint.best_model_path:
            raise RuntimeError('Training completed without a validation checkpoint')
        (run_dir / 'run_summary.json').write_text(json.dumps({
            'checkpoint': checkpoint.best_model_path,
            'best_val_loss': float(checkpoint.best_model_score),
            'seed': 42, 'torch_version': torch.__version__, 'lightning_version': pl.__version__,
            'accelerator': accelerator, 'devices': devices, 'precision': precision,
        }, indent=2) + '\n')
        print(f'Best checkpoint: {checkpoint.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_filepath', required=True)
    parser.add_argument('--run-dir', default=None, help='Run directory; defaults to configuration parent directory')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--precision', default=None)
    main(parser.parse_args())
