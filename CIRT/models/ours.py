# import torch
# import torch.nn as nn
# class ConvBlock(nn.Module):
#     """Lightweight Conv-BN-Activation block."""

#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
#         super().__init__()
#         padding = kernel_size // 2
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)


# class Model(nn.Module):
#     """
#     Minimal convolutional baseline.

#     Expects inputs with shape [batch, input_size, height, width] and produces [32, 63, 121, 240], where 32 is batch size, 63 means there are 63 variables to predict,
#     121 is the height and 240 is the width of the grid.
#     predictions shaped [batch, pred_len, output_size, height, width]. [32, 2, 63, 121, 240], where 2 means there are two periods to predict, weeks 3-4 (from day 15 to day 28) and weeks 5-6 (from day 29 to day 42)
#     """

#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         pred_len: int = 2,
#         hidden_dim: int = 128,
#     ) -> None:
#         super().__init__()
#         self.pred_len = pred_len
#         self.output_size = output_size

#         self.encoder = nn.Sequential(
#             ConvBlock(input_size, hidden_dim, kernel_size=5),
#             ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
#         )

#         self.head = nn.Conv2d(hidden_dim, output_size * pred_len, kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feats = self.encoder(x)
#         logits = self.head(feats)
#         b, _, h, w = logits.shape
#         preds = logits.view(b, self.pred_len, self.output_size, h, w)
#         return preds
 
 

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Basic Components
# ==========================================

class LayerNorm2d(nn.Module):
    """Spatial LayerNorm for CNN-style feature maps [B, C, H, W]."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ==========================================
# 2. Stream A: Planetary Dynamics (Low-Frequency)
# ==========================================

class PlanetaryTransformerLayer(nn.Module):
    """
    Processes the 'Planetary Wave' stream.
    Operates in a LOW-RESOLUTION latent space (e.g., 16x30).
    This stream is isolated from high-frequency local noise.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Standard global self-attention to capture teleconnections
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_tokens, C]
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. Stream B: Synoptic Dynamics (High-Frequency)
# ==========================================

class SynopticConvLayer(nn.Module):
    """
    Processes the 'Synoptic Weather' stream.
    Operates in HIGH-RESOLUTION latent space using Large Kernel Convs.
    Focuses on local continuity and texture.
    """
    def __init__(self, dim: int, drop: float = 0.):
        super().__init__()
        # Large kernel (7x7) depthwise conv to capture local weather systems
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm2d(dim)
        self.mlp_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        res = x
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp_conv(x)
        x = self.drop(x)
        return res + x

# ==========================================
# 4. Interaction: Energy Cascade Injection
# ==========================================

class CascadeInjector(nn.Module):
    """
    Implements the Physical Inductive Bias: 'Large scales drive small scales'.
    Unlike TelePiT which adds a bias term, we use Cross-Attention.

    Query = Synoptic Stream (Local)
    Key/Value = Planetary Stream (Global)

    This allows the local weather to 'query' the global context relevant to its location.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_synoptic = nn.LayerNorm(dim)
        self.norm_planetary = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Gating parameter (learnable), initialized to 0 to preserve stability at start
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x_synoptic: torch.Tensor, x_planetary: torch.Tensor) -> torch.Tensor:
        """
        x_synoptic: [B, C, H, W] (High Res)
        x_planetary: [B, N_low, C] (Low Res Sequence)
        """
        B, C, H, W = x_synoptic.shape

        # Flatten spatial dimensions for attention query
        q = x_synoptic.flatten(2).transpose(1, 2) # [B, H*W, C]
        k = v = x_planetary                       # [B, N_low, C]

        q = self.norm_synoptic(q)
        k = self.norm_planetary(k)

        # Cross-Attention: Synoptic (Q) attends to Planetary (K,V)
        # This is computationally feasible because sequence length of K/V is very small.
        out, _ = self.cross_attn(query=q, key=k, value=v)

        # Reshape back to spatial grid
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Injection: Original Local Features + Gamma * Global Context
        return x_synoptic + self.gamma * out

# ==========================================
# 5. Full Model: Cascade-S2S
# ==========================================

class Model(nn.Module):
    """
    Cascade-S2S (Scale-Separated Transformer).

    Key Differentiation from TelePiT:
    1. Explicit Scale Separation (Dual Stream) vs. Implicit Bias.
    2. Cross-Attention Injection vs. Additive Attention Bias.
    3. Pure DL Architecture vs. Neural ODEs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pred_len: int = 2,
        hidden_dim: int = 128,
        depth: int = 4,
        patch_size: int = 4,
        # Fixed low-resolution grid for Planetary Stream (e.g., 16x30 captures global modes well)
        planetary_grid_size: tuple[int, int] = (16, 30) 
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_size = output_size
        self.patch_size = patch_size
        self.planetary_grid_size = planetary_grid_size

        # --- 1. Input Embedding (Entry to Latent Space) ---
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_size, hidden_dim, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(hidden_dim)
        )

        # --- 2. Positional Encodings ---
        # Use resizeable positional embedding to handle different resolutions if needed
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 32, 64)) 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 3. Dual-Stream Backbone ---
        self.layers = nn.ModuleList([])

        # Adapter to create Low-Freq Planetary Stream from High-Freq input
        self.to_planetary = nn.AdaptiveAvgPool2d(planetary_grid_size)

        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'planetary_layer': PlanetaryTransformerLayer(hidden_dim, num_heads=4),
                'synoptic_layer': SynopticConvLayer(hidden_dim),
                'injector': CascadeInjector(hidden_dim, num_heads=4)
            }))

        # --- 4. Prediction Head ---
        # Upsample back to patch resolution
        self.upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Final projection
        self.head = nn.Conv2d(hidden_dim, output_size * pred_len, kernel_size=1)

    def _forward_dual_stream(self, x_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_feat.shape

        # --- A. Initialize Streams ---
        # Synoptic Stream starts as the full feature map
        x_synoptic = x_feat

        # Planetary Stream is downsampled to isolate low-frequency signals
        # [B, C, 16, 30] -> Flatten -> [B, 480, C]
        x_planetary_map = self.to_planetary(x_feat)
        x_planetary = x_planetary_map.flatten(2).transpose(1, 2)

        # --- B. Cascade Evolution ---
        for layer in self.layers:
            # 1. Evolve Planetary Stream (Global Context) independently
            # This pure Transformer layer captures Teleconnections without local noise
            x_planetary = layer['planetary_layer'](x_planetary)

            # 2. Evolve Synoptic Stream (Local Texture) using CNN
            x_synoptic = layer['synoptic_layer'](x_synoptic)

            # 3. Cascade Injection:
            # The Planetary context is injected into the Synoptic stream via Cross-Attn
            x_synoptic = layer['injector'](x_synoptic, x_planetary)

        return x_synoptic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_size, H_orig, W_orig] (e.g., 121, 240)
        B, _, H_orig, W_orig = x.shape

        # --- Padding for Patch Embedding ---
        # Ensure divisibility by patch_size (e.g., 4)
        pad_h = (self.patch_size - (H_orig % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W_orig % self.patch_size)) % self.patch_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # --- Embedding ---
        x_feat = self.patch_embed(x_padded)

        # Add Positional Embedding
        pos_embed = F.interpolate(self.pos_embed, size=x_feat.shape[-2:], mode='bilinear', align_corners=False)
        x_feat = x_feat + pos_embed

        # --- Dual-Stream Processing ---
        x_refined = self._forward_dual_stream(x_feat)

        # --- Reconstruction ---
        x_out = self.upsample(x_refined)

        # Final projection
        logits = self.head(x_out)

        # Crop padding
        logits = logits[:, :, :H_orig, :W_orig]

        # Reshape to [B, Pred_Len, Output_Vars, H, W]
        preds = logits.view(B, self.pred_len, self.output_size, H_orig, W_orig)

        return preds

# if __name__ == "__main__":
#     # Quick sanity check
#     B, C, H, W = 2, 63, 121, 240
#     model = Model(input_size=C, output_size=C, hidden_dim=64, depth=2)
#     x = torch.randn(B, C, H, W)
#     y = model(x)
#     print(f"Input: {x.shape}, Output: {y.shape}")
#     # Expected: torch.Size([2, 2, 63, 121, 240])