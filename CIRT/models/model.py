import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch

# torch.autograd.set_detect_anomaly(True)
# from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from CIRT.models import CirT, egnn
from CIRT import dataset, config, utils, criterion

class S2SBenchmarkModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SBenchmarkModel, self).__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size'] 
        
                
        if 'CirT' == self.model_args['model_name']:
            self.model = CirT.Model(input_size=input_size)

        if 'ours' == self.model_args['model_name']:
            from .ours import Model as OursModel
            self.model = OursModel(
                input_size=self.model_args['input_size'] ,
                output_size=self.model_args['output_size'],
                pred_len=self.model_args.get('pred_len', 2)
            )

        if 'egnn' in self.model_args['model_name'].lower():
            from .egnn import EGNN
            
            # 使用与EIMP完全一致的参数
            self.model = EGNN(
                in_node_nf=input_size - 11,  # 63 - 11 = 52 (移除速度分量)
                in_edge_nf=1,  # 边特征维度
                hidden_nf=self.model_args['hidden_sizes'],  # 128
                output_dim=output_size - 22  # 63 - 22 = 41 (移除速度输出)
            )
        
        if 'ClimODE' == self.model_args['model_name']:
            from .climode import create_climode_model
            # ClimODE预测两周（num_steps=2）
            num_steps = 2  # 与数据格式一致：[batch, 2, 63, 121, 240]
            self.model = create_climode_model(self.model_args, input_size, output_size, num_steps=num_steps)
        
        if 'ClimaX' == self.model_args['model_name']:
            from .climax import ClimaX
            self.model = ClimaX(
                default_vars=self.model_args.get('default_vars', [str(i) for i in range(input_size)]),
                img_size=self.model_args.get('img_size', [121, 240]),
                patch_size=self.model_args.get('patch_size', 4),
                embed_dim=self.model_args.get('embed_dim', 128),
                depth=self.model_args.get('depth', 8),
                decoder_depth=self.model_args.get('decoder_depth', 1),
                num_heads=self.model_args.get('num_heads', 16),
                mlp_ratio=self.model_args.get('mlp_ratio', 4.0),
                drop_path=self.model_args.get('drop_path', 0.1),
                drop_rate=self.model_args.get('drop_rate', 0.1),
                parallel_patch_embed=self.model_args.get('parallel_patch_embed', False),
            )
        
        if 'ViT' == self.model_args['model_name']:
            from .vit import ViT
            self.model = ViT(
                img_size=self.model_args.get('img_size', [124, 240]),
                input_size=input_size,
                patch_size=self.model_args.get('patch_size', 4),
                embed_dim=self.model_args.get('embed_dim', 128),
                depth=self.model_args.get('depth', 8),
                decoder_depth=self.model_args.get('decoder_depth', 1),
                num_heads=self.model_args.get('num_heads', 16),
                mlp_ratio=self.model_args.get('mlp_ratio', 4.0),
                drop_path=self.model_args.get('drop_path', 0.1),
                drop_rate=self.model_args.get('drop_rate', 0.1),
            )

        if 'FNO' == self.model_args['model_name']:
            # Use local FNO implementation
            from .fno import FNO2d
            num_steps = self.model_args.get('pred_len', 2)  # Default to 2 steps for S2S
            self.model = FNO2d(
                input_size=input_size,
                modes1=self.model_args.get('modes1', 4),
                modes2=self.model_args.get('modes2', 4),
                width=self.model_args.get('width', [64, 128, 256, 512, 1024]),
                initial_step=self.model_args.get('initial_step', 1),
                num_steps=num_steps,
            )
        
        if 'TelePiT' == self.model_args['model_name']:
            # Import TelePiT wrapper that loads external TelePiT implementation
            from .telepit import Model as TelePiTModel
            self.model = TelePiTModel(
                input_size=input_size,
                output_size=output_size,
                img_size=self.model_args.get('img_size', [121, 240]),
                embed_dim=self.model_args.get('embed_dim', 256),
                depth=self.model_args.get('depth', 6),
                num_heads=self.model_args.get('num_heads', 8),
                mlp_ratio=self.model_args.get('mlp_ratio', 4.0),
                wavelet_levels=self.model_args.get('wavelet_levels', 3),
                drop_rate=self.model_args.get('drop_rate', 0.1),
                attn_drop_rate=self.model_args.get('attn_drop_rate', 0.1),
            )

        if 'Transformer' == self.model_args['model_name']:
            from .transformer import TransformerModel
            self.model = TransformerModel(
                input_size=input_size,
                output_size=output_size,
                img_size=self.model_args.get('img_size', [120, 240]),
                patch_size=self.model_args.get('patch_size', 4),
                embed_dim=self.model_args.get('embed_dim', 256),
                num_heads=self.model_args.get('num_heads', 8),
                num_encoder_layers=self.model_args.get('num_encoder_layers', 6),
                dim_feedforward=self.model_args.get('dim_feedforward', 1024),
                dropout=self.model_args.get('dropout', 0.1),
                pred_len=self.model_args.get('pred_len', 2),
            )
        
        if 'TianQuan' == self.model_args['model_name']:
            from .tianquan import TianQuanWrapper
            self.model = TianQuanWrapper(
                input_size=input_size,
                output_size=output_size,
                img_size=tuple(self.model_args.get('img_size', [121, 240])),
                patch_size=self.model_args.get('patch_size', 2),
                embed_dim=self.model_args.get('embed_dim', 384),
                decoder_depth=self.model_args.get('decoder_depth', 2),
                num_heads=self.model_args.get('num_heads', 12),
                mlp_ratio=self.model_args.get('mlp_ratio', 4.0),
                drop_path=self.model_args.get('drop_path', 0.1),
                drop_rate=self.model_args.get('drop_rate', 0.1),
                root_dir=self.model_args.get('root_dir', './logs/TianQuan'),
                default_vars=self.model_args.get('default_vars', None),
            )
        
        self.loss = self.init_loss_fn()
        self.val_loss = criterion.RMSE()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, x, u=None, v=None, radial=None, edges=None, edge_attr=None, timestamp=None, lead_times=None):
        if 'egnn' in self.model_args.get('model_name', '').lower():
            return self.model(x, u, v, radial, edges, edge_attr, timestamp)
        elif 'ClimaX' == self.model_args.get('model_name', ''):
            # ClimaX expects lead_times parameter
            if lead_times is None:
                lead_times = torch.zeros(x.size(0), device=x.device)
            return self.model(x, lead_times=lead_times)
        else:
            return self.model(x)



    def _egnn_training_step(self, batch, batch_idx):
        """Training step for EGNN model - 与EIMP实现一致"""
        coord, x_data, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y, batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        # 与EIMP完全一致的数据处理
        u = torch.cat((x_data[:, 30:40], x_data[:, 61:62]), dim=1)
        v = torch.cat((x_data[:, 40:50], x_data[:, 62:63]), dim=1)
        x = torch.cat((x_data[:, :30], x_data[:, 50:61]), dim=1)

        # Clamp to prevent overflow in sqrt, then compute speed
        u_clamped = torch.clamp(u, min=-1e6, max=1e6)
        v_clamped = torch.clamp(v, min=-1e6, max=1e6)
        speed = torch.sqrt(u_clamped**2 + v_clamped**2 + 1e-8)
        x = torch.cat([x, speed], dim=1)

        row, col = edge_index

        preds = self(x=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        
        # Check for NaN/Inf in predictions
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print(f"⚠️ Warning: NaN/Inf detected in predictions at step {batch_idx}")
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
        
        loss = self.loss(preds, y)
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Warning: NaN/Inf loss detected at step {batch_idx}, skipping update")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _egnn_validation_step(self, batch, batch_idx):
        """Validation step for EGNN model - 与EIMP实现一致"""
        coord, x_data, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y, batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        # 与EIMP完全一致的数据处理
        u = torch.cat((x_data[:, 30:40], x_data[:, 61:62]), dim=1)
        v = torch.cat((x_data[:, 40:50], x_data[:, 62:63]), dim=1)
        x = torch.cat((x_data[:, :30], x_data[:, 50:61]), dim=1)

        # Clamp to prevent overflow in sqrt, then compute speed
        u_clamped = torch.clamp(u, min=-1e6, max=1e6)
        v_clamped = torch.clamp(v, min=-1e6, max=1e6)
        speed = torch.sqrt(u_clamped**2 + v_clamped**2 + 1e-8)
        x = torch.cat([x, speed], dim=1)

        row, col = edge_index

        preds = self(x=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        
        loss = self.loss(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        if 'egnn' in self.model_args.get('model_name', '').lower():
            return self._egnn_training_step(batch, batch_idx)
        else:
            timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
            n_steps = y.size(1)
            loss = 0
            
            # Pass lead_times for ClimaX
            if 'ClimaX' == self.model_args.get('model_name', ''):
                lead_times = torch.zeros(x.size(0), device=x.device)
                preds = self(x, lead_times=lead_times)
                # Adjust y to match ClimaX output shape (ClimaX output may be smaller due to patching)
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif 'ClimODE' == self.model_args.get('model_name', ''):
                # ClimODE outputs [batch, step, channels, height, width]
                preds = self(x)
                # No need to adjust shape, ClimODE directly outputs the correct shape
            elif 'TianQuan' == self.model_args.get('model_name', ''):
                # TianQuan outputs [batch, time, channels, height, width]
                preds = self(x)
                # Align target spatial dims to preds if necessary
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    y = y[:, 0]
            else:
                # Ensure ViT/Transformer inputs match configured img_size (e.g., crop to 120x240)
                if self.model_args.get('model_name', '') in ('ViT', 'Transformer'):
                    target_h, target_w = self.model_args.get('img_size', [120, 240])
                    x = x[:, :, :target_h, :target_w]
                preds = self(x)
                # Align target spatial dims to preds if necessary (e.g., ViT 120 vs target 121)
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    # Single-step models like FNO output 4D; compare against first step of targets
                    y = y[:, 0]
            
            loss=self.loss(preds,y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        if 'egnn' in self.model_args.get('model_name', '').lower():
            return self._egnn_validation_step(batch, batch_idx)
        else:
            timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
            n_steps = y.size(1)
            loss = 0
            
            # Pass lead_times for ClimaX
            if 'ClimaX' == self.model_args.get('model_name', ''):
                lead_times = torch.zeros(x.size(0), device=x.device)
                preds = self(x, lead_times=lead_times)
                # Adjust y to match ClimaX output shape (ClimaX output may be smaller due to patching)
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif 'ClimODE' == self.model_args.get('model_name', ''):
                # ClimODE outputs [batch, step, channels, height, width]
                preds = self(x)
                # No need to adjust shape, ClimODE directly outputs the correct shape
            elif 'TianQuan' == self.model_args.get('model_name', ''):
                # TianQuan outputs [batch, time, channels, height, width]
                preds = self(x)
                # Align target spatial dims to preds if necessary
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    y = y[:, 0]
            else:
                # Ensure ViT/Transformer inputs match configured img_size (e.g., crop to 120x240)
                if self.model_args.get('model_name', '') in ('ViT', 'Transformer'):
                    target_h, target_w = self.model_args.get('img_size', [120, 240])
                    x = x[:, :, :target_h, :target_w]
                preds = self(x)
                # Align target spatial dims to preds if necessary (e.g., ViT 120 vs target 121)
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    y = y[:, 0]
            
            loss=self.loss(preds,y)
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def test_step(self, batch, batch_idx):
        timestamp, x, y = batch
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        
        # Pass lead_times for ClimaX
        if 'ClimaX' == self.model_args.get('model_name', ''):
            lead_times = torch.zeros(x.size(0), device=x.device)
            preds = self(x, lead_times=lead_times)
            # Adjust y to match ClimaX output shape (ClimaX output may be smaller due to patching)
            y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
        elif 'ClimODE' == self.model_args.get('model_name', ''):
            # ClimODE outputs [batch, step, channels, height, width]
            preds = self(x)
            # No need to adjust shape, ClimODE directly outputs the correct shape
        elif 'TianQuan' == self.model_args.get('model_name', ''):
            # TianQuan outputs [batch, time, channels, height, width]
            preds = self(x)
            # Align target spatial dims to preds if necessary
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        else:
            # Ensure ViT/Transformer inputs match configured img_size (e.g., crop to 120x240)
            if self.model_args.get('model_name', '') in ('ViT', 'Transformer'):
                target_h, target_w = self.model_args.get('img_size', [120, 240])
                x = x[:, :, :target_h, :target_w]
            preds = self(x)
            # Align target spatial dims to preds if necessary (e.g., ViT 120 vs target 121)
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        
        loss=self.val_loss(preds,y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.model_args['learning_rate'],
            weight_decay=self.model_args.get('weight_decay', 1e-5),
            eps=1e-8  # Add epsilon for numerical stability
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping to stabilize training across all models."""
        max_norm = self.model_args.get('grad_clip_norm', 0.0)
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def setup(self, stage=None):
        if 'egnn' in self.model_args.get('model_name', '').lower():
            # 使用图数据集
            self.train_dataset = dataset.S2SGraphDataset(data_dir=self.data_args['data_dir'],
                                                        years=self.data_args['train_years'], 
                                                       n_step=self.data_args['n_step'],
                                                       lead_time=self.data_args['lead_time'],
                                                       kernel_size=self.data_args.get('kernel_size', 4),
                                                        single_vars=self.data_args['single_vars'],
                                                        pred_single_vars=self.data_args['pred_single_vars'],
                                                        pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                      )
            self.val_dataset = dataset.S2SGraphDataset(data_dir=self.data_args['data_dir'],
                                                      years=self.data_args['val_years'], 
                                                     n_step=self.data_args['n_step'],
                                                     lead_time=self.data_args['lead_time'],
                                                     kernel_size=self.data_args.get('kernel_size', 4),
                                                     single_vars=self.data_args['single_vars'],
                                                     pred_single_vars=self.data_args['pred_single_vars'],
                                                     pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    )
            
            self.test_dataset = dataset.S2SGraphDataset(data_dir=self.data_args['data_dir'],
                                                      years=self.data_args['test_years'], 
                                                     n_step=self.data_args['n_step'],
                                                     lead_time=self.data_args['lead_time'],
                                                     kernel_size=self.data_args.get('kernel_size', 4),
                                                     single_vars=self.data_args['single_vars'],
                                                     pred_single_vars=self.data_args['pred_single_vars'],
                                                     pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    )
        else:
            # 使用标准数据集
            self.train_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                        years=self.data_args['train_years'], 
                                                       n_step=self.data_args['n_step'],
                                                       lead_time=self.data_args['lead_time'],
                                                        single_vars=self.data_args['single_vars'],
                                                        pred_single_vars=self.data_args['pred_single_vars'],
                                                        pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                      )
            self.val_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                      years=self.data_args['val_years'], 
                                                     n_step=self.data_args['n_step'],
                                                     lead_time=self.data_args['lead_time'],
                                                     single_vars=self.data_args['single_vars'],
                                                     pred_single_vars=self.data_args['pred_single_vars'],
                                                     pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    )
            
            self.test_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                      years=self.data_args['test_years'], 
                                                     n_step=self.data_args['n_step'],
                                                     lead_time=self.data_args['lead_time'],
                                                     single_vars=self.data_args['single_vars'],
                                                     pred_single_vars=self.data_args['pred_single_vars'],
                                                     pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def predict_step(self, batch, batch_idx):
        """预测步骤，用于评估"""
        # Handle EGNN graph data format
        if 'egnn' in self.model_args.get('model_name', '').lower():
            return self._egnn_predict_step(batch, batch_idx)
        
        # Standard format: (timestamp, x, y)
        timestamp, x, y = batch
        
        # Pass lead_times for ClimaX
        if 'ClimaX' == self.model_args.get('model_name', ''):
            lead_times = torch.zeros(x.size(0), device=x.device)
            preds = self(x, lead_times=lead_times)
            # Adjust y to match ClimaX output shape (ClimaX output may be smaller due to patching)
            y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
        elif 'ClimODE' == self.model_args.get('model_name', ''):
            # ClimODE outputs [batch, step, channels, height, width]
            preds = self(x)
            # No need to adjust shape, ClimODE directly outputs the correct shape
        elif 'TianQuan' == self.model_args.get('model_name', ''):
            # TianQuan outputs [batch, time, channels, height, width]
            preds = self(x)
            # Align target spatial dims to preds if necessary
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        else:
            # Ensure ViT/Transformer inputs match configured img_size (e.g., crop to 120x240)
            if self.model_args.get('model_name', '') in ('ViT', 'Transformer'):
                target_h, target_w = self.model_args.get('img_size', [120, 240])
                x = x[:, :, :target_h, :target_w]
            preds = self(x)
            # Align target spatial dims to preds if necessary (e.g., ViT 120 vs target 121)
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        
        return preds, y, timestamp

    
    def _egnn_predict_step(self, batch, batch_idx):
        """预测步骤 for EGNN model - 与EIMP实现一致"""
        coord, x_data, y, edge_index, edge_feat, radial, mask = batch.coord, batch.x, batch.y, batch.edge_index, batch.edge_feat, batch.radial, batch.mask
        
        # 与EIMP完全一致的数据处理
        u = torch.cat((x_data[:, 30:40], x_data[:, 61:62]), dim=1)
        v = torch.cat((x_data[:, 40:50], x_data[:, 62:63]), dim=1)
        x = torch.cat((x_data[:, :30], x_data[:, 50:61]), dim=1)

        # Clamp to prevent overflow in sqrt, then compute speed
        u_clamped = torch.clamp(u, min=-1e6, max=1e6)
        v_clamped = torch.clamp(v, min=-1e6, max=1e6)
        speed = torch.sqrt(u_clamped**2 + v_clamped**2 + 1e-8)
        x = torch.cat([x, speed], dim=1)

        row, col = edge_index

        preds = self(x=x, u=u, v=v, radial=radial, edges=edge_index, edge_attr=edge_feat)
        
        # Convert from graph format [num_nodes, pred_len, output_dim] to grid format [batch, pred_len, output_dim, height, width]
        # Graph format: y is [num_nodes, pred_len, output_dim] = [121*240, 2, 63]
        # Grid format: [batch, pred_len, output_dim, height, width] = [batch, 2, 63, 121, 240]
        num_nodes = 121 * 240
        batch_size = x_data.shape[0] // num_nodes if x_data.shape[0] >= num_nodes else 1
        
        # Reshape predictions: [num_nodes, pred_len, output_dim] -> [batch, height, width, pred_len, output_dim] -> [batch, pred_len, output_dim, height, width]
        preds = preds.reshape(batch_size, 121, 240, preds.shape[1], preds.shape[2])
        preds = preds.permute(0, 3, 4, 1, 2)  # [batch, pred_len, output_dim, height, width]
        
        # Reshape y similarly: [num_nodes, pred_len, output_dim] -> [batch, pred_len, output_dim, height, width]
        y = y.reshape(batch_size, 121, 240, y.shape[1], y.shape[2])
        y = y.permute(0, 3, 4, 1, 2)  # [batch, pred_len, output_dim, height, width]
        
        # Create dummy timestamp for compatibility (EGNN uses graph format without explicit timestamp)
        timestamp = torch.zeros(batch_size, dtype=torch.long, device=y.device)
        
        return preds, y, timestamp