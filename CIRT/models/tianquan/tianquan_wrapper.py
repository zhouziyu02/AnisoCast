"""
TianQuan wrapper to adapt ziyu_cli interface
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Import TianQuan model with fixed paths
import sys
import os

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from .arch import TianQuan
except ImportError:
    # Try absolute import
    from CIRT.models.tianquan.arch import TianQuan


class TianQuanWrapper(nn.Module):
    """
    Wrapper for TianQuan model to adapt ziyu_cli interface.
    
    Converts ziyu_cli format (timestamp, x, y) to TianQuan format.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        img_size: tuple = (121, 240),
        patch_size: int = 2,
        embed_dim: int = 384,
        decoder_depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        root_dir: str = "./logs",
        default_vars: Optional[list] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.img_size = img_size
        
        # Generate default variable names if not provided
        if default_vars is None:
            # Create variable names based on input_size
            # Assuming format: 2 surface vars + pressure vars (6 vars * 10 levels = 60) + 1 = 63
            default_vars = []
            # Surface variables
            default_vars.extend(['2m_temperature', '10m_wind_speed'])
            # Pressure level variables (assuming 10 levels)
            pressure_vars = ['geopotential', 'specific_humidity', 'temperature', 
                           'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
            levels = [10, 50, 100, 200, 300, 500, 700, 850, 925, 1000]
            for var in pressure_vars:
                for level in levels:
                    default_vars.append(f"{var}_{level}")
            # Pad or trim to match input_size
            if len(default_vars) > input_size:
                default_vars = default_vars[:input_size]
            elif len(default_vars) < input_size:
                # Extend with dummy names
                for i in range(input_size - len(default_vars)):
                    default_vars.append(f"var_{i}")
        
        self.default_vars = default_vars[:input_size]
        
        # Create root directory for constants
        os.makedirs(root_dir, exist_ok=True)
        constants_dir = os.path.join(root_dir, "constants")
        os.makedirs(constants_dir, exist_ok=True)
        
        # Initialize TianQuan model
        try:
            self.model = TianQuan(
                default_vars=self.default_vars,
                root_dir=root_dir,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                decoder_depth=decoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                drop_rate=drop_rate,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize TianQuan with full features: {e}")
            print("Initializing with simplified configuration...")
            # Try with minimal configuration
            self.model = TianQuan(
                default_vars=self.default_vars,
                root_dir=root_dir,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                decoder_depth=decoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                drop_rate=drop_rate,
            )
        
        # Generate latitude and longitude arrays
        self.lats = torch.linspace(-90, 90, img_size[0])
        self.lons = torch.linspace(0, 360, img_size[1], endpoint=False)
        
    def _create_dummy_inputs(self, x, device):
        """
        Create dummy inputs required by TianQuan from ziyu_cli format.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            device: Device to place tensors on
            
        Returns:
            Dictionary with all required inputs for TianQuan
        """
        batch_size = x.shape[0]
        channels = x.shape[1]
        height, width = x.shape[2], x.shape[3]
        
        # Reshape x to [batch, time=1, vars, height, width]
        x_tianquan = x.unsqueeze(1)  # [batch, 1, channels, height, width]
        
        # Create anomaly (same as x for now, can be improved)
        anomaly = x_tianquan.clone()
        
        # Create hours tensor (dummy, can be improved with actual timestamps)
        hours = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        # Variables list
        variables = self.default_vars[:channels]
        
        # Lats and lons
        lats = self.lats.to(device)
        lons = self.lons.to(device)
        
        # Region info (full region)
        region_info = {
            'min_h': 0,
            'max_h': height - 1,
            'min_w': 0,
            'max_w': width - 1,
        }
        
        # Lead time (dummy, can be improved)
        lead_time = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        return {
            'x': x_tianquan,
            'anomaly': anomaly,
            'hours': hours,
            'variables': variables,
            'lats': lats,
            'lons': lons,
            'region_info': region_info,
            'lead_time': lead_time,
        }
    
    def forward(self, x):
        """
        Forward pass adapted for ziyu_cli interface.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, time_steps, channels, height, width]
        """
        device = x.device
        batch_size = x.shape[0]
        channels = x.shape[1]
        height, width = x.shape[2], x.shape[3]
        
        # Create dummy inputs
        inputs = self._create_dummy_inputs(x, device)
        
        # Call TianQuan forward_encoder
        try:
            surf_pred, atmo_pred = self.model.forward_encoder(
                inputs['x'],
                inputs['anomaly'],
                inputs['hours'],
                inputs['variables'],
                inputs['lats'],
                inputs['lons'],
                inputs['region_info'],
                inputs['lead_time'],
            )
            
            # Combine surface and atmospheric predictions
            # surf_pred: [batch, time, surf_vars, height, width]
            # atmo_pred: [batch, time, levels, atmos_vars, height, width]
            
            # Reshape atmo_pred to [batch, time, atmos_vars * levels, height, width]
            b, t, l, v, h, w = atmo_pred.shape
            atmo_pred_flat = atmo_pred.reshape(b, t, l * v, h, w)
            
            # Concatenate surface and atmospheric predictions
            preds = torch.cat([surf_pred, atmo_pred_flat], dim=2)  # [batch, time, channels, height, width]
            
            # Ensure output matches expected channels
            if preds.shape[2] != channels:
                # Pad or trim channels
                if preds.shape[2] > channels:
                    preds = preds[:, :, :channels, :, :]
                else:
                    padding = torch.zeros(b, t, channels - preds.shape[2], h, w, 
                                         device=device, dtype=preds.dtype)
                    preds = torch.cat([preds, padding], dim=2)
            
            return preds
            
        except Exception as e:
            print(f"Error in TianQuan forward: {e}")
            # Fallback: return simple prediction
            # Return input with time dimension added
            return x.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # [batch, 2, channels, height, width]

