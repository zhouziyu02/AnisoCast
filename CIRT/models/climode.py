import os
import sys
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Add ClimODE to path
climode_path = Path(__file__).parent.parent.parent / "ClimODE"
sys.path.append(str(climode_path))

try:
    from model_function import Climate_encoder_free_uncertain
    from utils import (
        add_constant_info,
        nll,
    )
    CLIMODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ClimODE modules not available: {e}")
    print("Using simplified ClimODE implementation")
    CLIMODE_AVAILABLE = False
    
    # Create dummy classes for fallback
    class Climate_encoder_free_uncertain:
        def __init__(self, *args, **kwargs):
            pass
    
    class SimplifiedClimODE(nn.Module):
        """简化的ClimODE模型实现 - 使用卷积网络处理空间数据，与CirT保持一致的设计"""
        def __init__(self, input_size, output_size, hidden_size=128, num_steps=2, drop_rate=0.1):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_size = hidden_size
            self.num_steps = num_steps  # Number of time steps to predict (e.g., 2 for 2 weeks)
            
            # 使用卷积网络处理空间数据，添加 BatchNorm 和 Dropout 提高稳定性
            self.encoder = nn.Sequential(
                nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU()
            )
            
            # Decoder outputs num_steps * output_size channels
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(hidden_size, output_size * num_steps, kernel_size=3, padding=1)
            )
            
            # 初始化权重 - 与CirT保持一致
            self._initialize_weights()
        
        def _initialize_weights(self):
            """初始化模型权重，与CirT保持一致"""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # x: [batch, channels, height, width]
            batch_size, _, height, width = x.shape
            
            # 编码
            encoded = self.encoder(x)
            # 解码
            output = self.decoder(encoded)
            # output: [batch, output_size * num_steps, height, width]
            
            # Reshape to [batch, num_steps, output_size, height, width]
            output = output.view(batch_size, self.num_steps, self.output_size, height, width)
            
            return output
        
        def parameters(self):
            """返回模型参数"""
            return super().parameters()

# Import CirT dataset and config
from CIRT import dataset, config


class ClimODELightningModule(pl.LightningModule):
    def __init__(self, model_args, data_args):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        
        # Calculate input and output sizes based on CirT data format
        # CirT uses pressure level and single level variables
        pressure_vars = len(data_args.get('pred_pressure_vars', [])) * len(config.PRESSURE_LEVELS)
        single_vars = len(data_args.get('pred_single_vars', []))
        self.output_size = pressure_vars + single_vars
        
        # Initialize ClimODE model
        if CLIMODE_AVAILABLE:
            self.model = Climate_encoder_free_uncertain(
                num_channels=self.output_size,
                const_channels=2,
                out_types=self.output_size,
                method=model_args['solver'],
                use_att=model_args['use_att'],
                use_err=model_args['use_err'],
                use_pos=model_args['use_pos']
            )
        else:
            # Fallback to a simple neural network if ClimODE is not available
            print("Using fallback neural network instead of ClimODE")
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(self.output_size, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, self.output_size, 3, padding=1)
            )
        
        # Training parameters
        self.lr = model_args['lr']
        self.weight_decay = model_args['weight_decay']
        self.var_coeff = 0.001
        
        # Data attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.const_channels_info = None
        self.lat_map = None
        self.lon_map = None
        
    def setup(self, stage=None):
        """Setup data using CirT dataset format"""
        if stage == 'fit' or stage is None:
            # Use CirT dataset format
            self.train_dataset = dataset.S2SDataset(
                data_dir=self.data_args['data_dir'],
                years=self.data_args['train_years'], 
                n_step=self.data_args['n_step'],
                lead_time=self.data_args['lead_time'],
                single_vars=self.data_args['single_vars'],
                pred_single_vars=self.data_args['pred_single_vars'],
                pred_pressure_vars=self.data_args['pred_pressure_vars'],
            )
            
            self.val_dataset = dataset.S2SDataset(
                data_dir=self.data_args['data_dir'],
                years=self.data_args['val_years'], 
                n_step=self.data_args['n_step'],
                lead_time=self.data_args['lead_time'],
                single_vars=self.data_args['single_vars'],
                pred_single_vars=self.data_args['pred_single_vars'],
                pred_pressure_vars=self.data_args['pred_pressure_vars'],
            )
            
            self.test_dataset = dataset.S2SDataset(
                data_dir=self.data_args['data_dir'],
                years=self.data_args['test_years'], 
                n_step=self.data_args['n_step'],
                lead_time=self.data_args['lead_time'],
                single_vars=self.data_args['single_vars'],
                pred_single_vars=self.data_args['pred_single_vars'],
                pred_pressure_vars=self.data_args['pred_pressure_vars'],
            )
            
            # Load constant information (if available)
            if 'const_info_path' in self.data_args:
                self.const_channels_info, self.lat_map, self.lon_map = add_constant_info([self.data_args['const_info_path']])
            else:
                # Create dummy constant info if not available
                self.const_channels_info = torch.zeros(1, 2, 32, 64)
                self.lat_map = torch.zeros(32, 64)
                self.lon_map = torch.zeros(32, 64)
            
            print(f"Training dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
            print(f"Test dataset size: {len(self.test_dataset)}")
        
    def training_step(self, batch, batch_idx):
        """Training step using CirT data format"""
        timestamp, x, y = batch  # x: [batch, input_size, height, width], y: [batch, step, output_size, height, width]
        
        # For ClimODE, we need to adapt the data format
        batch_size, input_channels, H, W = x.shape
        
        if CLIMODE_AVAILABLE:
            # Create dummy velocity data for now (this would need proper velocity fitting)
            # In a real implementation, you would need to fit velocity fields
            past_sample = torch.zeros(batch_size, 2*self.output_size, H, W).to(self.device)
            
            # Update model parameters
            self.model.update_param([
                past_sample,
                self.const_channels_info.to(self.device),
                self.lat_map.to(self.device),
                self.lon_map.to(self.device)
            ])
            
            # Create time steps (simplified for now)
            t = torch.linspace(0, 1, 2).to(self.device)
            
            # Forward pass
            mean, std, _ = self.model(t, x)
            
            # Compute loss (simplified - using MSE for now)
            # Use first time step of target
            target = y[:, 0] if y.dim() > 4 else y
            loss = torch.nn.functional.mse_loss(mean, target)
        else:
            # Fallback: simple forward pass
            pred = self.model(x)
            target = y[:, 0] if y.dim() > 4 else y
            loss = torch.nn.functional.mse_loss(pred, target)
        
        # Add L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step using CirT data format"""
        timestamp, x, y = batch
        
        batch_size, input_channels, H, W = x.shape
        
        if CLIMODE_AVAILABLE:
            # Create dummy velocity data
            past_sample = torch.zeros(batch_size, 2*self.output_size, H, W).to(self.device)
            
            # Update model parameters
            self.model.update_param([
                past_sample,
                self.const_channels_info.to(self.device),
                self.lat_map.to(self.device),
                self.lon_map.to(self.device)
            ])
            
            # Create time steps
            t = torch.linspace(0, 1, 2).to(self.device)
            
            # Forward pass
            mean, std, _ = self.model(t, x)
            
            # Compute loss
            target = y[:, 0] if y.dim() > 4 else y
            loss = torch.nn.functional.mse_loss(mean, target)
        else:
            # Fallback: simple forward pass
            pred = self.model(x)
            target = y[:, 0] if y.dim() > 4 else y
            loss = torch.nn.functional.mse_loss(pred, target)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.model_args['niters']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        
    def train_dataloader(self):
        """Training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_args['batch_size'],
            shuffle=True,
            num_workers=self.model_args.get('num_workers', 4),
            pin_memory=True
        )
        
    def val_dataloader(self):
        """Validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_args['batch_size'],
            shuffle=False,
            num_workers=self.model_args.get('num_workers', 4),
            pin_memory=True
        )
        
    def test_dataloader(self):
        """Test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_args['batch_size'],
            shuffle=False,
            num_workers=self.model_args.get('num_workers', 4),
            pin_memory=True
        )


def create_climode_model(model_args, input_size, output_size, num_steps=2):
    """
    创建ClimODE模型的工厂函数
    
    Args:
        model_args: 模型参数配置
        input_size: 输入通道数
        output_size: 输出通道数
        num_steps: 预测时间步数（默认2，表示两周）
    """
    if CLIMODE_AVAILABLE:
        # 使用真实的ClimODE模型
        # ClimODE expects num_channels, const_channels, out_types
        model = Climate_encoder_free_uncertain(
            num_channels=output_size,  # 输出通道数
            const_channels=2,  # 常量通道数（如地形、纬度等）
            out_types=output_size,  # 输出类型数
            method=model_args.get('solver', 'euler'),
            use_att=model_args.get('use_att', True),
            use_err=model_args.get('use_err', True),
            use_pos=model_args.get('use_pos', False)
        )
    else:
        # 使用简化的ClimODE模型
        print("Warning: Using simplified ClimODE model")
        model = SimplifiedClimODE(
            input_size=input_size,
            output_size=output_size,
            hidden_size=model_args.get('hidden_size', 128),
            num_steps=num_steps,
            drop_rate=model_args.get('drop_rate', 0.1)  # 与CirT保持一致
        )
    
    return model