import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
from pathlib import Path
import yaml
import time
from datetime import datetime
# from lightning.pytorch.loggers import WandbLogger  # 禁用WandB
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import GradientAccumulationScheduler
pl.seed_everything(42)

from CIRT.models import model
from CIRT.callbacks import TrainingSpeedCallback

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
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')
    
    # 根据参数决定是否使用TensorBoard
    tb_logger = None
    if args.use_tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir,
            log_graph=False,
            default_hp_metric=False,
        )
        print("📊 TensorBoard日志已启用")
    else:
        print("🚀 使用最高性能模式（禁用TensorBoard）")
    
    # 创建训练速度监控回调
    speed_callback = TrainingSpeedCallback(log_every_n_steps=10)  # 每50个step打印一次，减少输出频率
    
    # 组装加速器/设备配置（可通过命令行或环境变量覆盖）
    default_accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    accelerator = getattr(args, 'accelerator', None) or os.environ.get('ACCELERATOR', default_accelerator)

    # devices 优先级：CLI --devices > 环境 NP > 1（CPU或单卡）
    cli_devices = getattr(args, 'devices', None)
    env_np = os.environ.get('NP')
    devices = int(cli_devices) if cli_devices is not None else (int(env_np) if env_np else (1 if accelerator != 'gpu' else 1))

    # strategy：多卡DDP，否则auto
    strategy = getattr(args, 'strategy', None) or ('ddp_find_unused_parameters_true' if accelerator == 'gpu' and devices and int(devices) > 1 else 'auto')

    # precision：GPU默认16-mixed，CPU固定32
    precision = getattr(args, 'precision', None) or ('16-mixed' if accelerator == 'gpu' else '32-true')

    print(f"Trainer config -> accelerator={accelerator}, devices={devices}, strategy={strategy}, precision={precision}")

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=model_args['epochs'],
        logger=tb_logger,
        callbacks=[checkpoint_callback, speed_callback],
        enable_progress_bar=True,
        precision=precision,
     )

    # 开始训练
    training_start_time = time.time()
    
    # 打印当前UTC+8时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 当前时间 (UTC+8): {current_time}")
    print("🚀 开始训练...")
    
    trainer.fit(baseline)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 打印训练结束时间
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ 训练完成时间 (UTC+8): {end_time}")
    print("Total training time: {:.4f}s".format(total_training_time))
    
    trainer.test(baseline, ckpt_path="best")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', required=True, help='Provide the filepath string to the model config...')
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    # 允许从命令行覆盖设备/加速器/策略/精度
    parser.add_argument('--devices', type=int, default=None, help='Number of devices to use (e.g., 1)')
    parser.add_argument('--accelerator', type=str, default=None, help="'gpu' or 'cpu'")
    parser.add_argument('--strategy', type=str, default=None, help="Lightning strategy, e.g., 'ddp' or 'auto'")
    parser.add_argument('--precision', type=str, default=None, help="'16-mixed', '32-true', etc.")
    args = parser.parse_args()
    main(args)