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
    
    # 创建训练器 - 8卡训练配置
    trainer = pl.Trainer(
        devices=8,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',  # 启用未使用参数检测，解决DDP错误
        max_epochs=model_args['epochs'],
        logger=tb_logger,  # 根据参数决定是否使用TensorBoard
        callbacks=[checkpoint_callback, speed_callback],
        enable_progress_bar=True,  # 禁用默认进度条，避免多进程重复显示
        precision='16-mixed',  # 使用混合精度训练提高性能
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
    args = parser.parse_args()
    main(args)