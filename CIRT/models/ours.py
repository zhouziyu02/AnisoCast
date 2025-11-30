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
 
 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ==========================================
# # 1. Basic Components
# # ==========================================

# class LayerNorm2d(nn.Module):
#     """Spatial LayerNorm for CNN-style feature maps [B, C, H, W]."""
#     def __init__(self, num_channels: int, eps: float = 1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x

# class MLP(nn.Module):
#     def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.):
#         super().__init__()
#         hidden_dim = int(dim * mlp_ratio)
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, dim)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# # ==========================================
# # 2. Stream A: Planetary Dynamics (Low-Frequency)
# # ==========================================

# class PlanetaryTransformerLayer(nn.Module):
#     """
#     Processes the 'Planetary Wave' stream.
#     Operates in a LOW-RESOLUTION latent space (e.g., 16x30).
#     This stream is isolated from high-frequency local noise.
#     """
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         # Standard global self-attention to capture teleconnections
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MLP(dim, mlp_ratio, drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, N_tokens, C]
#         res = x
#         x_norm = self.norm1(x)
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
#         x = res + attn_out

#         x = x + self.mlp(self.norm2(x))
#         return x

# # ==========================================
# # 3. Stream B: Synoptic Dynamics (High-Frequency)
# # ==========================================

# class SynopticConvLayer(nn.Module):
#     """
#     Processes the 'Synoptic Weather' stream.
#     Operates in HIGH-RESOLUTION latent space using Large Kernel Convs.
#     Focuses on local continuity and texture.
#     """
#     def __init__(self, dim: int, drop: float = 0.):
#         super().__init__()
#         # Large kernel (7x7) depthwise conv to capture local weather systems
#         self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
#         self.norm = LayerNorm2d(dim)
#         self.mlp_conv = nn.Sequential(
#             nn.Conv2d(dim, dim * 4, 1),
#             nn.GELU(),
#             nn.Conv2d(dim * 4, dim, 1),
#         )
#         self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, H, W]
#         res = x
#         x = self.conv_dw(x)
#         x = self.norm(x)
#         x = self.mlp_conv(x)
#         x = self.drop(x)
#         return res + x

# # ==========================================
# # 4. Interaction: Energy Cascade Injection
# # ==========================================

# class CascadeInjector(nn.Module):
#     """
#     Implements the Physical Inductive Bias: 'Large scales drive small scales'.
#     Unlike TelePiT which adds a bias term, we use Cross-Attention.

#     Query = Synoptic Stream (Local)
#     Key/Value = Planetary Stream (Global)

#     This allows the local weather to 'query' the global context relevant to its location.
#     """
#     def __init__(self, dim: int, num_heads: int = 4):
#         super().__init__()
#         self.norm_synoptic = nn.LayerNorm(dim)
#         self.norm_planetary = nn.LayerNorm(dim)
#         self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

#         # Gating parameter (learnable), initialized to 0 to preserve stability at start
#         self.gamma = nn.Parameter(torch.zeros(1)) 

#     def forward(self, x_synoptic: torch.Tensor, x_planetary: torch.Tensor) -> torch.Tensor:
#         """
#         x_synoptic: [B, C, H, W] (High Res)
#         x_planetary: [B, N_low, C] (Low Res Sequence)
#         """
#         B, C, H, W = x_synoptic.shape

#         # Flatten spatial dimensions for attention query
#         q = x_synoptic.flatten(2).transpose(1, 2) # [B, H*W, C]
#         k = v = x_planetary                       # [B, N_low, C]

#         q = self.norm_synoptic(q)
#         k = self.norm_planetary(k)

#         # Cross-Attention: Synoptic (Q) attends to Planetary (K,V)
#         # This is computationally feasible because sequence length of K/V is very small.
#         out, _ = self.cross_attn(query=q, key=k, value=v)

#         # Reshape back to spatial grid
#         out = out.transpose(1, 2).reshape(B, C, H, W)

#         # Injection: Original Local Features + Gamma * Global Context
#         return x_synoptic + self.gamma * out

# # ==========================================
# # 5. Full Model: Cascade-S2S
# # ==========================================

# class Model(nn.Module):
#     """
#     Cascade-S2S (Scale-Separated Transformer).

#     Key Differentiation from TelePiT:
#     1. Explicit Scale Separation (Dual Stream) vs. Implicit Bias.
#     2. Cross-Attention Injection vs. Additive Attention Bias.
#     3. Pure DL Architecture vs. Neural ODEs.
#     """

#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         pred_len: int = 2,
#         hidden_dim: int = 128,
#         depth: int = 4,
#         patch_size: int = 4,
#         # Fixed low-resolution grid for Planetary Stream (e.g., 16x30 captures global modes well)
#         planetary_grid_size: tuple[int, int] = (16, 30) 
#     ) -> None:
#         super().__init__()
#         self.pred_len = pred_len
#         self.output_size = output_size
#         self.patch_size = patch_size
#         self.planetary_grid_size = planetary_grid_size

#         # --- 1. Input Embedding (Entry to Latent Space) ---
#         self.patch_embed = nn.Sequential(
#             nn.Conv2d(input_size, hidden_dim, kernel_size=patch_size, stride=patch_size),
#             LayerNorm2d(hidden_dim)
#         )

#         # --- 2. Positional Encodings ---
#         # Use resizeable positional embedding to handle different resolutions if needed
#         self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 32, 64)) 
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)

#         # --- 3. Dual-Stream Backbone ---
#         self.layers = nn.ModuleList([])

#         # Adapter to create Low-Freq Planetary Stream from High-Freq input
#         self.to_planetary = nn.AdaptiveAvgPool2d(planetary_grid_size)

#         for _ in range(depth):
#             self.layers.append(nn.ModuleDict({
#                 'planetary_layer': PlanetaryTransformerLayer(hidden_dim, num_heads=4),
#                 'synoptic_layer': SynopticConvLayer(hidden_dim),
#                 'injector': CascadeInjector(hidden_dim, num_heads=4)
#             }))

#         # --- 4. Prediction Head ---
#         # Upsample back to patch resolution
#         self.upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

#         # Final projection
#         self.head = nn.Conv2d(hidden_dim, output_size * pred_len, kernel_size=1)

#     def _forward_dual_stream(self, x_feat: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = x_feat.shape

#         # --- A. Initialize Streams ---
#         # Synoptic Stream starts as the full feature map
#         x_synoptic = x_feat

#         # Planetary Stream is downsampled to isolate low-frequency signals
#         # [B, C, 16, 30] -> Flatten -> [B, 480, C]
#         x_planetary_map = self.to_planetary(x_feat)
#         x_planetary = x_planetary_map.flatten(2).transpose(1, 2)

#         # --- B. Cascade Evolution ---
#         for layer in self.layers:
#             # 1. Evolve Planetary Stream (Global Context) independently
#             # This pure Transformer layer captures Teleconnections without local noise
#             x_planetary = layer['planetary_layer'](x_planetary)

#             # 2. Evolve Synoptic Stream (Local Texture) using CNN
#             x_synoptic = layer['synoptic_layer'](x_synoptic)

#             # 3. Cascade Injection:
#             # The Planetary context is injected into the Synoptic stream via Cross-Attn
#             x_synoptic = layer['injector'](x_synoptic, x_planetary)

#         return x_synoptic

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, input_size, H_orig, W_orig] (e.g., 121, 240)
#         B, _, H_orig, W_orig = x.shape

#         # --- Padding for Patch Embedding ---
#         # Ensure divisibility by patch_size (e.g., 4)
#         pad_h = (self.patch_size - (H_orig % self.patch_size)) % self.patch_size
#         pad_w = (self.patch_size - (W_orig % self.patch_size)) % self.patch_size
#         x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

#         # --- Embedding ---
#         x_feat = self.patch_embed(x_padded)

#         # Add Positional Embedding
#         pos_embed = F.interpolate(self.pos_embed, size=x_feat.shape[-2:], mode='bilinear', align_corners=False)
#         x_feat = x_feat + pos_embed

#         # --- Dual-Stream Processing ---
#         x_refined = self._forward_dual_stream(x_feat)

#         # --- Reconstruction ---
#         x_out = self.upsample(x_refined)

#         # Final projection
#         logits = self.head(x_out)

#         # Crop padding
#         logits = logits[:, :, :H_orig, :W_orig]

#         # Reshape to [B, Pred_Len, Output_Vars, H, W]
#         preds = logits.view(B, self.pred_len, self.output_size, H_orig, W_orig)

#         return preds

# # if __name__ == "__main__":
# #     # Quick sanity check
# #     B, C, H, W = 2, 63, 121, 240
# #     model = Model(input_size=C, output_size=C, hidden_dim=64, depth=2)
# #     x = torch.randn(B, C, H, W)
# #     y = model(x)
# #     print(f"Input: {x.shape}, Output: {y.shape}")
# #     # Expected: torch.Size([2, 2, 63, 121, 240])

 
 
  
 

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import trunc_normal_
import torch.fft

# ==========================================
# 1. Core Modules
# ==========================================

class SpectralGating(nn.Module):
    """
    Frequency Domain Branch (Inspired by CirT/FNO).
    Captures Global Periodic Patterns (Planetary Waves).

    Improvement over CirT: Instead of heavy Frequency-Attention, 
    we use efficient Frequency-Gating (Element-wise multiplication in Freq Domain).
    This saves compute to allow adding a Spatial Branch.
    """
    def __init__(self, dim, h=121):
        super().__init__()
        self.dim = dim
        self.h = h
        # Only process half frequencies due to conjugate symmetry of RFFT
        self.freq_dim = h // 2 + 1 

        # Complex weights: [Dim, Freq_Dim]
        # Treat channel mixing efficiently
        scale = (1 / (dim * dim))
        self.weights = nn.Parameter(
            scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
        )

    def forward(self, x):
        # x: [B, N, C] -> N is Latitude (121)
        B, N, C = x.shape

        # 1. FFT along the latitude dimension
        # x_ft: [B, N//2+1, C] (Complex)
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')

        # 2. Spectral Gating
        # We define weights as real (..., 2) to handle AMP safely, convert to complex here
        w = torch.view_as_complex(self.weights) # [C, Freq]

        # Element-wise multiplication (Broadcasting over Batch)
        # x_ft: [B, Freq, C] * w.T: [C, Freq] -> need alignment
        # Let's permute x_ft to [B, C, Freq]
        x_ft = x_ft.permute(0, 2, 1)

        # Spectral Filtering: [B, C, Freq] * [C, Freq] -> [B, C, Freq]
        out_ft = x_ft * w.unsqueeze(0)

        # 3. IFFT
        out_ft = out_ft.permute(0, 2, 1) # Back to [B, Freq, C]
        x = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')

        return x

class SpatialAttention(nn.Module):
    """
    Spatial Domain Branch (Standard MHSA).
    Captures Sparse Teleconnections and Non-periodic dependencies 
    (e.g., Tropics -> Poles interaction) which FFT struggles with.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Standard MHSA
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LocalMixer(nn.Module):
    """
    Local Refinement Branch.
    Uses 1D Conv over Latitude to capture local gradients (North-South exchange).
    Crucial for physical consistency (smoothness).
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class HoloBlock(nn.Module):
    """
    The Holographic Block: Combines Spectral, Spatial, and Local information.
    Novelty: Parallel disentangled processing.
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            seq_len=121
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # --- Dual-Domain Mixer ---
        # 1. Global Periodic (Spectral)
        self.spectral = SpectralGating(dim, h=seq_len)
        # 2. Global Sparse (Spatial)
        self.spatial = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # 3. Local Continuity
        self.local = LocalMixer(dim)

        # Learnable gating weights to balance the three
        self.w_spec = nn.Parameter(torch.ones(dim) * 0.5)
        self.w_spat = nn.Parameter(torch.ones(dim) * 0.5)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        shortcut = x
        x_norm = self.norm1(x)

        # Parallel Execution
        # Novelty: Explicitly modeling Wave (Spectral) vs Particle (Spatial) behavior
        feat_spectral = self.spectral(x_norm)
        feat_spatial = self.spatial(x_norm)
        feat_local = self.local(x_norm)

        # Gated Fusion
        # w_spec controls how much we trust the wave dynamics
        # w_spat controls how much we trust the teleconnection attention
        mixed = (feat_spectral * self.w_spec) + (feat_spatial * self.w_spat) + feat_local

        x = shortcut + self.drop_path(mixed)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """
    Standard Patch Embedding from CirT (Kept for compatibility and performance).
    Compresses Longitude (W) into Channels.
    """
    def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, V, H, W]
        x = self.proj(x) # [B, Dim, H, 1]
        x = x.flatten(2).transpose(1, 2) # [B, H, Dim]
        x = self.norm(x)
        return x

# ==========================================
# 2. Main Model Class
# ==========================================

class Model(nn.Module):
    """
    Holo-Former (Holographic Transformer for S2S).

    A replacement for CirT that introduces Spatial-Spectral Parallel Mixing.
    Uses CirT's high-performance hyperparams (Embed=768, Depth=8).
    """
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        # Patch size is conceptually W here due to embedding strategy
        patch_size=240, 
        # CirT defaults
        embed_dim=768,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.input_size = input_size
        self.patch_size = img_size[1] # 240

        # 1. Embedding (CirT Style)
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = 121 # Latitude points

        # Learnable Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Backbone (HoloBlocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            HoloBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                seq_len=self.num_patches
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 3. Prediction Head (CirT Style)
        # Matches output shape logic of CirT
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())

        # Output dim handles (2 weeks) * (Input Vars) * (Longitude Width)
        # Because we compressed Longitude into channel earlier, we expand it back here.
        output_dim = self.input_size * 2 * self.img_size[1]
        self.head.append(nn.Linear(embed_dim, output_dim))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def unpatchify(self, x):
        """
        Reconstructs spatial dimensions from the flattened prediction.
        x: (B, H, V * 2 * W) -> (B, 2, V, H, W)
        """
        B, H, D = x.shape
        W = self.img_size[1]
        V = self.input_size

        # CirT's output format is essentially flattened longitude
        # Reshape: [B, H, 2 * V * W] -> [B, H, 2, V, W]
        x = x.reshape(B, H, 2, V, W)

        # Permute to target: [B, 2, V, H, W]
        x = x.permute(0, 2, 3, 1, 4)
        return x

    def forward_encoder(self, x):
        # x: [B, V, H, W]
        x = self.token_embeds(x) # [B, 121, 768]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        # Input: [B, 63, 121, 240]
        feats = self.forward_encoder(x) # [B, 121, 768]
        preds = self.head(feats)        # [B, 121, 63*2*240]
        preds = self.unpatchify(preds)  # [B, 2, 63, 121, 240]
        return preds