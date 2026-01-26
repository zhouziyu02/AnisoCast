import os
# Keep user/external CUDA_VISIBLE_DEVICES, do not override here
# Set environment variables to avoid tqdm progress bar duplicate output
os.environ['TQDM_DISABLE'] = '0'  # Keep tqdm enabled
import sys
# Set stdout to line buffering mode to avoid duplicate output when progress bar refreshes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass
import argparse
from pathlib import Path
import yaml
import time
from datetime import datetime
# from lightning.pytorch.loggers import WandbLogger  # Disable WandB
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import GradientAccumulationScheduler
pl.seed_everything(42)

from CIRT.models import model
from CIRT.callbacks import TrainingSpeedCallback
import shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


os.environ['WANDB_MODE'] = 'disabled'


def main(args):
    """
    Training script given .yaml config
    Example usage:
        1) `python train.py --config_filepath CIRT/configs/fno_s2s.yaml --use_tensorboard`
    """
    
    # Retrieve hyperparameters
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
        
    # Initialize model
    baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)

    baseline.setup()
    
    # Initialize training
    log_dir = Path('logs') / model_args['model_name']
    
    # ModelCheckpoint configuration
    # save_top_k=1: Only save 1 checkpoint with minimum val_loss (best model)
    # save_last=False: Do not save last epoch checkpoint, only keep best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        mode='min',
        filename='{epoch}-{step}',
        save_top_k=1,
        save_last=False  # Only save best model, do not save last.ckpt
    )
    
    # Create a custom callback to save config copy when checkpoint is saved
    class ConfigSaverCallback(pl.Callback):
        def __init__(self, config_path, hyperparams):
            super().__init__()
            self.config_path = config_path
            self.hyperparams = hyperparams
            self.saved_to_checkpoint_dir = False
        
        def on_train_start(self, trainer, pl_module):
            # Save config to checkpoint directory at training start
            if trainer.global_rank == 0 and not self.saved_to_checkpoint_dir:
                # Lightning will create version directory at training start, get path via logger
                if trainer.logger and hasattr(trainer.logger, 'log_dir'):
                    version_dir = Path(trainer.logger.log_dir)
                    config_copy_path = version_dir / 'config.yaml'
                    with open(config_copy_path, 'w') as f:
                        yaml.dump(self.hyperparams, f, default_flow_style=False, sort_keys=False)
                    print(f"Config saved to checkpoint directory: {config_copy_path}")
                    self.saved_to_checkpoint_dir = True
                else:
                    # Fallback: Wait for Lightning to create version directory
                    import time
                    time.sleep(2)
                    version_dirs = sorted(
                        [d for d in Path('lightning_logs').glob('version_*') if d.is_dir()],
                        key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0
                    )
                    if version_dirs:
                        latest_version = version_dirs[-1]
                        config_copy_path = latest_version / 'config.yaml'
                        with open(config_copy_path, 'w') as f:
                            yaml.dump(self.hyperparams, f, default_flow_style=False, sort_keys=False)
                        print(f"Config saved to checkpoint directory: {config_copy_path}")
                        self.saved_to_checkpoint_dir = True
    
    config_saver = ConfigSaverCallback(args.config_filepath, hyperparams)
    
    # In DDP mode, only print on rank 0 to avoid duplicate output
    is_rank_zero = True
    if 'RANK' in os.environ:
        rank = int(os.environ.get('RANK', '0'))
        is_rank_zero = (rank == 0)
    
    # Decide whether to use TensorBoard based on arguments
    tb_logger = None
    if args.use_tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir,
            log_graph=False,
            default_hp_metric=False,
        )
        if is_rank_zero:
            print("TensorBoard logging enabled")
    else:
        if is_rank_zero:
            print("Using maximum performance mode (TensorBoard disabled)")
    
    # Create training speed monitoring callback
    speed_callback = TrainingSpeedCallback(log_every_n_steps=10)  # Print every 10 steps to reduce output frequency
    
    # Assemble accelerator/device configuration (can be overridden via command line or environment variables)
    default_accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    accelerator = getattr(args, 'accelerator', None) or os.environ.get('ACCELERATOR', default_accelerator)

    # Device priority: CLI --devices > environment NP > 1 (CPU or single GPU)
    cli_devices = getattr(args, 'devices', None)
    env_np = os.environ.get('NP')
    devices = int(cli_devices) if cli_devices is not None else (int(env_np) if env_np else (1 if accelerator != 'gpu' else 1))

    # Strategy: DDP for multi-GPU, otherwise auto
    strategy = getattr(args, 'strategy', None) or ('ddp_find_unused_parameters_true' if accelerator == 'gpu' and devices and int(devices) > 1 else 'auto')

    # Precision: GPU default 16-mixed, CPU fixed 32
    precision = getattr(args, 'precision', None) or ('16-mixed' if accelerator == 'gpu' else '32-true')

    # In DDP mode, only show progress bar on rank 0 to avoid multi-process output confusion
    # os is already imported at the top, no need to re-import
    if 'RANK' in os.environ:
        rank = int(os.environ.get('RANK', '0'))
        enable_progress_bar = (rank == 0)  # Only show progress bar on rank 0
    else:
        enable_progress_bar = True  # Non-DDP mode, show progress bar

    if is_rank_zero:
        print(f"Trainer config -> accelerator={accelerator}, devices={devices}, strategy={strategy}, precision={precision}")

    # Use custom progress bar callback to avoid duplicate output
    # In DDP mode, TQDMProgressBar will automatically handle showing only on rank 0
    progress_bar_callback = None
    if enable_progress_bar:
        try:
            from lightning.pytorch.callbacks import TQDMProgressBar
            # Use TQDMProgressBar, it automatically handles output in DDP mode
            progress_bar_callback = TQDMProgressBar(refresh_rate=1)
        except ImportError:
            # If TQDMProgressBar is not available, use default progress bar
            pass
    
    callbacks_list = [checkpoint_callback, speed_callback, config_saver]
    if progress_bar_callback:
        callbacks_list.append(progress_bar_callback)
    
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=model_args['epochs'],
        logger=tb_logger,
        callbacks=callbacks_list,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=True,
        precision=precision,
    )


    # Start training
    training_start_time = time.time()
    
    # Print current time (only on rank 0 to avoid duplicate output)
    if is_rank_zero:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_time}")
        print("Starting training...")
    
    # Model summary will be automatically printed at training start (because enable_model_summary=True)
    # No need to manually call print_summary(), Lightning handles it automatically
    # In DDP mode, model summary will automatically only print on rank 0
    
    trainer.fit(baseline)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Print training end time
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training completed at: {end_time}")
    print("Total training time: {:.4f}s".format(total_training_time))
    
    # Note: Do not test directly in training script because:
    # 1. Testing in DDP mode may cause DataLoader worker to exit unexpectedly
    # 2. train_ours.sh will automatically call evaluate_soon.py for evaluation after training
    # 3. evaluate_soon.py uses single device for evaluation, more suitable for testing scenarios
    # trainer.test(baseline, ckpt_path="best")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', required=True, help='Provide the filepath string to the model config...')
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    # Allow overriding device/accelerator/strategy/precision from command line
    parser.add_argument('--devices', type=int, default=None, help='Number of devices to use (e.g., 1)')
    parser.add_argument('--accelerator', type=str, default=None, help="'gpu' or 'cpu'")
    parser.add_argument('--strategy', type=str, default=None, help="Lightning strategy, e.g., 'ddp' or 'auto'")
    parser.add_argument('--precision', type=str, default=None, help="'16-mixed', '32-true', etc.")
    args = parser.parse_args()
    main(args)