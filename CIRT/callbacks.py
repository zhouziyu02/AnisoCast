import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from datetime import datetime, timedelta, timezone


class TrainingSpeedCallback(Callback):
    """
    Custom callback function for monitoring training speed
    Print training information every N steps to avoid printing every batch
    """
    
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0
        
    def on_train_start(self, trainer, pl_module):
        """Record start time when training begins"""
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0
        
        # Only print configuration info on main process
        if trainer.global_rank == 0:
            print(f"\nStarting training {pl_module.model_args.get('model_name', 'Unknown')} model...")
            print(f"Training configuration:")
            print(f"   - Devices: {trainer.device_ids}")
            print(f"   - Max epochs: {trainer.max_epochs}")
            print(f"   - Print info every {self.log_every_n_steps} steps")
            print(f"   - Learning rate: {pl_module.model_args.get('learning_rate', 'N/A')}")
            print(f"   - Batch size: {pl_module.data_args.get('batch_size', 'N/A')}")
            print("-" * 80)
            
            # Print start time
            tz = timezone(timedelta(hours=8))
            start_time_str = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Start time: {start_time_str}")
            
            # Print YAML configuration parameters
            self._print_yaml_config(pl_module)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check if need to print info after each training batch"""
        global_step = trainer.global_step
        
        # Print every log_every_n_steps
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            current_time = time.time()
            
            # Calculate elapsed time and speed
            elapsed_time = current_time - self.last_log_time
            steps_processed = global_step - self.last_log_step
            
            if elapsed_time > 0 and steps_processed > 0:
                steps_per_second = steps_processed / elapsed_time
                seconds_per_step = elapsed_time / steps_processed
                
                # Get current loss
                if isinstance(outputs, dict):
                    train_loss = outputs.get('loss', 'N/A')
                elif torch.is_tensor(outputs):
                    train_loss = outputs.item()
                else:
                    train_loss = 'N/A'
                
                # Calculate estimated remaining time
                total_steps = trainer.max_epochs * len(trainer.train_dataloader)
                remaining_steps = total_steps - global_step
                estimated_remaining_time = remaining_steps / steps_per_second
                
                # Format time display
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_remaining_time)
                
                # Only print training log on main process
                if trainer.global_rank == 0:
                    print(f"\nStep {global_step:4d} | Loss: {train_loss:.4f} | "
                          f"Speed: {steps_per_second:.2f} steps/s ({seconds_per_step:.2f}s/step) | "
                          f"Elapsed: {elapsed_str} | ETA: {remaining_str}")
            
            self.last_log_time = current_time
            self.last_log_step = global_step
            
    def on_train_epoch_end(self, trainer, pl_module):
        """Print epoch info after training epoch ends"""
        if trainer.global_rank == 0:
            current_time = time.time()
            
            # Calculate epoch speed
            if self.start_time is not None:
                total_elapsed = current_time - self.start_time
                avg_speed = trainer.global_step / total_elapsed if total_elapsed > 0 else 0
                
                print(f"\nEpoch {trainer.current_epoch + 1} completed | "
                      f"Total steps: {trainer.global_step} | "
                      f"Average speed: {avg_speed:.2f} steps/s | "
                      f"Total elapsed: {self._format_time(total_elapsed)}")
                print("-" * 80)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation info after validation epoch ends"""
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
            
        # Get validation loss
        val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
            
        # Get training loss
        train_loss = trainer.callback_metrics.get('train_loss', 'N/A')
        if torch.is_tensor(train_loss):
            train_loss = train_loss.item()
        
        train_steps = len(trainer.train_dataloader)
        
        # Safely format loss values
        def format_loss(loss):
            if isinstance(loss, (int, float)):
                return f"{loss:.7f}"
            else:
                return str(loss)
        
        # Print in specified format
        print(f"Epoch {trainer.current_epoch + 1}, Steps: {train_steps} | "
              f"Train Loss: {format_loss(train_loss)} | "
              f"Val Loss: {format_loss(val_loss)}")
        
        # Print current learning rate
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
        
        print("-" * 80)
        
    def on_train_end(self, trainer, pl_module):
        """Print summary info when training ends"""
        if trainer.global_rank == 0:
            current_time = time.time()
            
            if self.start_time is not None:
                total_elapsed = current_time - self.start_time
                avg_speed = trainer.global_step / total_elapsed if total_elapsed > 0 else 0
                
                print(f"\nTraining completed!")
                print(f"Training summary:")
                print(f"   - Total epochs: {trainer.current_epoch + 1}")
                print(f"   - Total steps: {trainer.global_step}")
                print(f"   - Total elapsed: {self._format_time(total_elapsed)}")
                print(f"   - Average speed: {avg_speed:.2f} steps/s")
                print(f"   - Average time per epoch: {total_elapsed/(trainer.current_epoch + 1)/60:.1f} minutes")
                print("=" * 80)
    
    def _print_yaml_config(self, pl_module):
        """Print YAML configuration parameters"""
        print("\nYAML Configuration Parameters:")
        print("=" * 60)
        
        # Model parameters
        print("Model Parameters (model_args):")
        model_args = pl_module.model_args
        print(f"   ├─ Model name: {model_args.get('model_name', 'N/A')}")
        print(f"   ├─ Input size: {model_args.get('input_size', 'N/A')}")
        print(f"   ├─ Output size: {model_args.get('output_size', 'N/A')}")
        print(f"   ├─ Learning rate: {model_args.get('learning_rate', 'N/A')}")
        print(f"   ├─ Epochs: {model_args.get('epochs', 'N/A')}")
        print(f"   ├─ Number of workers: {model_args.get('num_workers', 'N/A')}")
        print(f"   ├─ Cosine annealing T_max: {model_args.get('t_max', 'N/A')}")
        print(f"   ├─ Prediction length: {model_args.get('pred_len', 'N/A')}")
        print(f"   └─ Only headline: {model_args.get('only_headline', 'N/A')}")
        
        # Data parameters
        print("\nData Parameters (data_args):")
        data_args = pl_module.data_args
        print(f"   ├─ Batch size: {data_args.get('batch_size', 'N/A')}")
        print(f"   ├─ Training years: {len(data_args.get('train_years', []))} years ({data_args.get('train_years', [])[0] if data_args.get('train_years') else 'N/A'}-{data_args.get('train_years', [])[-1] if data_args.get('train_years') else 'N/A'})")
        print(f"   ├─ Validation years: {data_args.get('val_years', 'N/A')}")
        print(f"   ├─ Test years: {data_args.get('test_years', 'N/A')}")
        print(f"   ├─ Data directory: {data_args.get('data_dir', 'N/A')}")
        print(f"   ├─ Number of steps: {data_args.get('n_step', 'N/A')}")
        print(f"   ├─ Lead time: {data_args.get('lead_time', 'N/A')}")
        print(f"   ├─ Input single-level variables: {len(data_args.get('single_vars', []))}")
        print(f"   ├─ Prediction single-level variables: {len(data_args.get('pred_single_vars', []))}")
        print(f"   ├─ Prediction pressure-level variables: {len(data_args.get('pred_pressure_vars', []))}")
        print(f"   └─ Data normalization: {data_args.get('is_normalized', 'N/A')}")
        
        print("=" * 60)
    
    def _format_time(self, seconds):
        """Format time display"""
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