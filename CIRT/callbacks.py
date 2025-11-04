import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from datetime import datetime, timedelta, timezone


class TrainingSpeedCallback(Callback):
    """
    自定义回调函数，用于监控训练速度
    每隔一定数量的steps打印一次训练信息，避免每个batch都打印
    """
    
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0
        
    def on_train_start(self, trainer, pl_module):
        """训练开始时记录开始时间"""
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0
        
        # 只在主进程打印配置信息
        if trainer.global_rank == 0:
            print(f"\n🚀 开始训练 {pl_module.model_args.get('model_name', 'Unknown')} 模型...")
            print(f"📊 训练配置:")
            print(f"   - 设备: {trainer.device_ids}")
            print(f"   - 最大epochs: {trainer.max_epochs}")
            print(f"   - 每 {self.log_every_n_steps} 个step打印一次信息")
            print(f"   - 学习率: {pl_module.model_args.get('learning_rate', 'N/A')}")
            print(f"   - 批次大小: {pl_module.data_args.get('batch_size', 'N/A')}")
            print("-" * 80)
            
            # 打印开始时间（UTC+8）
            tz = timezone(timedelta(hours=8))
            start_time_str = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            print(f"🕒 开始时间(UTC+8): {start_time_str}")
            
            # 打印YAML配置参数信息
            self._print_yaml_config(pl_module)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练batch结束后检查是否需要打印信息"""
        global_step = trainer.global_step
        
        # 每隔log_every_n_steps个step打印一次
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            current_time = time.time()
            
            # 计算耗时和速度
            elapsed_time = current_time - self.last_log_time
            steps_processed = global_step - self.last_log_step
            
            if elapsed_time > 0 and steps_processed > 0:
                steps_per_second = steps_processed / elapsed_time
                seconds_per_step = elapsed_time / steps_processed
                
                # 获取当前的loss
                if isinstance(outputs, dict):
                    train_loss = outputs.get('loss', 'N/A')
                elif torch.is_tensor(outputs):
                    train_loss = outputs.item()
                else:
                    train_loss = 'N/A'
                
                # 计算剩余时间估计
                total_steps = trainer.max_epochs * len(trainer.train_dataloader)
                remaining_steps = total_steps - global_step
                estimated_remaining_time = remaining_steps / steps_per_second
                
                # 格式化时间显示
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining_time)
                
                # 只在主进程打印训练日志
                if trainer.global_rank == 0:
                    print(f"\nStep {global_step:4d} | Loss: {train_loss:.4f} | "
                          f"Speed: {steps_per_second:.2f} steps/s ({seconds_per_step:.2f}s/step) | "
                          f"Elapsed: {elapsed_str} | ETA: {remaining_str}")
            
            self.last_log_time = current_time
            self.last_log_step = global_step
            
    def on_train_epoch_end(self, trainer, pl_module):
        """训练epoch结束后打印epoch信息"""
        if trainer.global_rank == 0:
            current_time = time.time()
            
            # 计算epoch速度
            if self.start_time is not None:
                total_elapsed = current_time - self.start_time
                avg_speed = trainer.global_step / total_elapsed if total_elapsed > 0 else 0
                
                print(f"\n📈 Epoch {trainer.current_epoch + 1} 完成 | "
                      f"总steps: {trainer.global_step} | "
                      f"平均速度: {avg_speed:.2f} steps/s | "
                      f"总耗时: {self._format_time(total_elapsed)}")
                print("-" * 80)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束后打印验证信息"""
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
            
        # 获取验证loss
        val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
            
        # 获取训练loss
        train_loss = trainer.callback_metrics.get('train_loss', 'N/A')
        if torch.is_tensor(train_loss):
            train_loss = train_loss.item()
        
        train_steps = len(trainer.train_dataloader)
        
        # 安全地格式化loss值
        def format_loss(loss):
            if isinstance(loss, (int, float)):
                return f"{loss:.7f}"
            else:
                return str(loss)
        
        # 按照指定格式打印
        print(f"📊 Epoch {trainer.current_epoch + 1}, Steps: {train_steps} | "
              f"Train Loss: {format_loss(train_loss)} | "
              f"Val Loss: {format_loss(val_loss)}")
        
        # 打印当前学习率
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f"📚 Current learning rate: {current_lr:.2e}")
        
        print("-" * 80)
        
    def on_train_end(self, trainer, pl_module):
        """训练结束时打印总结信息"""
        if trainer.global_rank == 0:
            current_time = time.time()
            
            if self.start_time is not None:
                total_elapsed = current_time - self.start_time
                avg_speed = trainer.global_step / total_elapsed if total_elapsed > 0 else 0
                
                print(f"\n🎉 训练完成!")
                print(f"📊 训练总结:")
                print(f"   - 总epochs: {trainer.current_epoch + 1}")
                print(f"   - 总steps: {trainer.global_step}")
                print(f"   - 总耗时: {self._format_time(total_elapsed)}")
                print(f"   - 平均速度: {avg_speed:.2f} steps/s")
                print(f"   - 平均每epoch耗时: {total_elapsed/(trainer.current_epoch + 1)/60:.1f} 分钟")
                print("=" * 80)
    
    def _print_yaml_config(self, pl_module):
        """打印YAML配置参数信息"""
        print("\n📋 YAML配置参数:")
        print("=" * 60)
        
        # 模型参数
        print("🔧 模型参数 (model_args):")
        model_args = pl_module.model_args
        print(f"   ├─ 模型名称: {model_args.get('model_name', 'N/A')}")
        print(f"   ├─ 输入维度: {model_args.get('input_size', 'N/A')}")
        print(f"   ├─ 输出维度: {model_args.get('output_size', 'N/A')}")
        print(f"   ├─ 学习率: {model_args.get('learning_rate', 'N/A')}")
        print(f"   ├─ 训练轮数: {model_args.get('epochs', 'N/A')}")
        print(f"   ├─ 数据加载器数量: {model_args.get('num_workers', 'N/A')}")
        print(f"   ├─ 余弦退火T_max: {model_args.get('t_max', 'N/A')}")
        print(f"   ├─ 预测长度: {model_args.get('pred_len', 'N/A')}")
        print(f"   └─ 仅标题行: {model_args.get('only_headline', 'N/A')}")
        
        # 数据参数
        print("\n📊 数据参数 (data_args):")
        data_args = pl_module.data_args
        print(f"   ├─ 批次大小: {data_args.get('batch_size', 'N/A')}")
        print(f"   ├─ 训练年份: {len(data_args.get('train_years', []))} 年 ({data_args.get('train_years', [])[0] if data_args.get('train_years') else 'N/A'}-{data_args.get('train_years', [])[-1] if data_args.get('train_years') else 'N/A'})")
        print(f"   ├─ 验证年份: {data_args.get('val_years', 'N/A')}")
        print(f"   ├─ 测试年份: {data_args.get('test_years', 'N/A')}")
        print(f"   ├─ 数据目录: {data_args.get('data_dir', 'N/A')}")
        print(f"   ├─ 时间步数: {data_args.get('n_step', 'N/A')}")
        print(f"   ├─ 提前时间: {data_args.get('lead_time', 'N/A')}")
        print(f"   ├─ 输入单层变量: {len(data_args.get('single_vars', []))} 个")
        print(f"   ├─ 预测单层变量: {len(data_args.get('pred_single_vars', []))} 个")
        print(f"   ├─ 预测压力层变量: {len(data_args.get('pred_pressure_vars', []))} 个")
        print(f"   └─ 数据归一化: {data_args.get('is_normalized', 'N/A')}")
        
        print("=" * 60)
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"