import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size'] 
        
        if 'soon' == self.model_args['model_name']:
            from .soon import Model as SOONModel
            # Read model architecture parameters from model_args, use defaults if not provided
            self.model = SOONModel(
                input_size=self.model_args['input_size'],
                img_size=self.model_args.get('img_size', [121, 240]),
                embed_dim=self.model_args.get('embed_dim', 256),
                depth=self.model_args.get('depth', 8),
                decoder_depth=self.model_args.get('decoder_depth', 2),
                num_heads=self.model_args.get('num_heads', 16),
                mlp_ratio=self.model_args.get('mlp_ratio', 4.0),
                drop_path=self.model_args.get('drop_path', 0.1),
                drop_rate=self.model_args.get('drop_rate', 0.1),
                patch_size=self.model_args.get('patch_size', 124),
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.model_args['model_name']}. Only 'soon' is supported.")
        
        self.loss = self.init_loss_fn()
        self.val_loss = criterion.RMSE()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, x, u=None, v=None, radial=None, edges=None, edge_attr=None, timestamp=None, lead_times=None):
            return self.model(x)

    def training_step(self, batch, batch_idx):
        timestamp, x, y = batch  # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        preds = self(x)
        
                # Align target spatial dims to preds if necessary
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    y = y[:, 0]
        
        loss = self.loss(preds, y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch
        preds = self(x)
        
                # Align target spatial dims to preds if necessary
                if preds.dim() == 5:
                    y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
                elif preds.dim() == 4 and y.dim() == 5:
                    y = y[:, 0]
            
        loss = self.loss(preds, y)
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def test_step(self, batch, batch_idx):
        timestamp, x, y = batch
        preds = self(x)
        
            # Align target spatial dims to preds if necessary
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        
        loss = self.val_loss(preds, y)
        return loss

    def configure_optimizers(self):
        # Ensure hyperparameters are correct numeric types (avoid YAML/script passing strings)
        lr = float(self.model_args['learning_rate'])
        weight_decay = float(self.model_args.get('weight_decay', 1e-5))
        t_max = int(self.model_args['t_max'])

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(
                    optimizer,
                    T_max=t_max,
                    eta_min=lr / 10.0,
                ),
                'interval': 'epoch',
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping to stabilize training."""
        max_norm = self.model_args.get('grad_clip_norm', 0.0)
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

    def setup(self, stage=None):
        # Use standard dataset
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
            batch_size=self.data_args['batch_size'], 
            shuffle=True, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
            batch_size=self.data_args['batch_size']
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
            batch_size=self.data_args['batch_size'], 
            drop_last=True
        )
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for evaluation"""
        timestamp, x, y = batch
        preds = self(x)
        
            # Align target spatial dims to preds if necessary
            if preds.dim() == 5:
                y = y[:, :, :, :preds.shape[3], :preds.shape[4]]
            elif preds.dim() == 4 and y.dim() == 5:
                y = y[:, 0]
        
        return preds, y, timestamp

    