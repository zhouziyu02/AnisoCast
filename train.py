import os
# 保持用户/外部传入的 CUDA_VISIBLE_DEVICES，不在此强行覆盖
# 设置环境变量，避免tqdm进度条重复输出
os.environ['TQDM_DISABLE'] = '0'  # 保持启用tqdm
import sys
# 设置stdout为行缓冲模式，避免进度条刷新时的重复输出
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
# from lightning.pytorch.loggers import WandbLogger  # 禁用WandB
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
    
    # ModelCheckpoint配置
    # save_top_k=1: 只保存val_loss最小的1个checkpoint（最佳模型）
    # save_last=False: 不保存最后一个epoch的checkpoint，只保留最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        mode='min',
        filename='{epoch}-{step}',
        save_top_k=1,
        save_last=False  # 只保存最佳模型，不保存last.ckpt
    )
    
    # 创建一个自定义回调，在checkpoint保存时也保存配置副本
    class ConfigSaverCallback(pl.Callback):
        def __init__(self, config_path, hyperparams):
            super().__init__()
            self.config_path = config_path
            self.hyperparams = hyperparams
            self.saved_to_checkpoint_dir = False
        
        def on_train_start(self, trainer, pl_module):
            # 在训练开始时，将配置保存到checkpoint目录
            if trainer.global_rank == 0 and not self.saved_to_checkpoint_dir:
                # Lightning会在训练开始时创建version目录，我们通过logger获取路径
                if trainer.logger and hasattr(trainer.logger, 'log_dir'):
                    version_dir = Path(trainer.logger.log_dir)
                    config_copy_path = version_dir / 'config.yaml'
                    with open(config_copy_path, 'w') as f:
                        yaml.dump(self.hyperparams, f, default_flow_style=False, sort_keys=False)
                    print(f"💾 配置已保存到checkpoint目录: {config_copy_path}")
                    self.saved_to_checkpoint_dir = True
                else:
                    # Fallback: 等待Lightning创建version目录
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
                        print(f"💾 配置已保存到checkpoint目录: {config_copy_path}")
                        self.saved_to_checkpoint_dir = True
    
    config_saver = ConfigSaverCallback(args.config_filepath, hyperparams)
    
    # 在DDP模式下，只在rank 0打印信息，避免重复输出
    is_rank_zero = True
    if 'RANK' in os.environ:
        rank = int(os.environ.get('RANK', '0'))
        is_rank_zero = (rank == 0)
    
    # 根据参数决定是否使用TensorBoard
    tb_logger = None
    if args.use_tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir,
            log_graph=False,
            default_hp_metric=False,
        )
        if is_rank_zero:
            print("📊 TensorBoard日志已启用")
    else:
        if is_rank_zero:
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

    # 在DDP模式下，只在rank 0显示进度条，避免多进程输出混乱
    # os已经在文件开头导入，不需要重新导入
    if 'RANK' in os.environ:
        rank = int(os.environ.get('RANK', '0'))
        enable_progress_bar = (rank == 0)  # 只在rank 0显示进度条
    else:
        enable_progress_bar = True  # 非DDP模式，显示进度条

    if is_rank_zero:
        print(f"Trainer config -> accelerator={accelerator}, devices={devices}, strategy={strategy}, precision={precision}")

    # 使用自定义进度条回调，避免重复输出
    # 只在单卡或rank 0显示；多卡时直接关闭进度条，避免tee/DPP重复
    progress_bar_callback = None
    multi_gpu = accelerator == 'gpu' and devices and int(devices) > 1
    if multi_gpu:
        enable_progress_bar = False
    if enable_progress_bar:
        try:
            from lightning.pytorch.callbacks import TQDMProgressBar
            progress_bar_callback = TQDMProgressBar(refresh_rate=1)
        except ImportError:
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
        enable_model_summary=is_rank_zero,
        precision=precision,
    )


    # 开始训练
    training_start_time = time.time()
    
    # 打印当前UTC+8时间（只在rank 0打印，避免重复输出）
    if is_rank_zero:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🕐 当前时间 (UTC+8): {current_time}")
        print("🚀 开始训练...")
    
    # 模型摘要会在训练开始时自动打印（因为enable_model_summary=True）
    # 不需要手动调用print_summary()，Lightning会自动处理
    # 在DDP模式下，模型摘要会自动只在rank 0打印
    
    trainer.fit(baseline)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 打印训练结束时间
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ 训练完成时间 (UTC+8): {end_time}")
    print("Total training time: {:.4f}s".format(total_training_time))
    
    # 注意：不在训练脚本中直接测试，因为：
    # 1. DDP模式下测试可能导致DataLoader worker意外退出
    # 2. train_ours.sh 会在训练完成后自动调用 auto_evaluate.py 进行评估
    # 3. evaluate_ours.py 使用单设备进行评估，更适合测试场景
    # trainer.test(baseline, ckpt_path="best")

    

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