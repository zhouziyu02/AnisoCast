# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.layers import DropPath, Mlp
# from timm.models.vision_transformer import trunc_normal_
# import torch.fft

# # ==========================================
# # 1. Core Modules
# # ==========================================

# class SpectralGating(nn.Module):
#     """
#     Frequency Domain Branch (Inspired by CirT/FNO).
#     Captures Global Periodic Patterns (Planetary Waves).

#     Improvement over CirT: Instead of heavy Frequency-Attention, 
#     we use efficient Frequency-Gating (Element-wise multiplication in Freq Domain).
#     This saves compute to allow adding a Spatial Branch.
#     """
#     def __init__(self, dim, h=121):
#         super().__init__()
#         self.dim = dim
#         self.h = h
#         # Only process half frequencies due to conjugate symmetry of RFFT
#         self.freq_dim = h // 2 + 1 

#         # Complex weights: [Dim, Freq_Dim]
#         # Treat channel mixing efficiently
#         scale = (1 / (dim * dim))
#         self.weights = nn.Parameter(
#             scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
#         )

#     def forward(self, x):
#         # x: [B, N, C] -> N is Latitude (121)
#         B, N, C = x.shape

#         # 1. FFT along the latitude dimension
#         # x_ft: [B, N//2+1, C] (Complex)
#         x_ft = torch.fft.rfft(x, dim=1, norm='ortho')

#         # 2. Spectral Gating
#         # We define weights as real (..., 2) to handle AMP safely, convert to complex here
#         w = torch.view_as_complex(self.weights) # [C, Freq]

#         # Element-wise multiplication (Broadcasting over Batch)
#         # x_ft: [B, Freq, C] * w.T: [C, Freq] -> need alignment
#         # Let's permute x_ft to [B, C, Freq]
#         x_ft = x_ft.permute(0, 2, 1)

#         # Spectral Filtering: [B, C, Freq] * [C, Freq] -> [B, C, Freq]
#         out_ft = x_ft * w.unsqueeze(0)

#         # 3. IFFT
#         out_ft = out_ft.permute(0, 2, 1) # Back to [B, Freq, C]
#         x = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')

#         return x

# class SpatialAttention(nn.Module):
#     """
#     Spatial Domain Branch (Standard MHSA).
#     Captures Sparse Teleconnections and Non-periodic dependencies 
#     (e.g., Tropics -> Poles interaction) which FFT struggles with.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         # Standard MHSA
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class LocalMixer(nn.Module):
#     """
#     Local Refinement Branch.
#     Uses 1D Conv over Latitude to capture local gradients (North-South exchange).
#     Crucial for physical consistency (smoothness).
#     """
#     def __init__(self, dim, kernel_size=5):
#         super().__init__()
#         self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

#     def forward(self, x):
#         # x: [B, N, C] -> [B, C, N]
#         x = x.transpose(1, 2)
#         x = self.conv(x)
#         x = x.transpose(1, 2)
#         return x

# class HoloBlock(nn.Module):
#     """
#     The Holographic Block: Combines Spectral, Spatial, and Local information.
#     Novelty: Parallel disentangled processing.
#     """
#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             drop=0.,
#             attn_drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             seq_len=121
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)

#         # --- Dual-Domain Mixer ---
#         # 1. Global Periodic (Spectral)
#         self.spectral = SpectralGating(dim, h=seq_len)
#         # 2. Global Sparse (Spatial)
#         self.spatial = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # 3. Local Continuity
#         self.local = LocalMixer(dim)

#         # Learnable gating weights to balance the three
#         self.w_spec = nn.Parameter(torch.ones(dim) * 0.5)
#         self.w_spat = nn.Parameter(torch.ones(dim) * 0.5)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         # x: [B, N, C]
#         shortcut = x
#         x_norm = self.norm1(x)

#         # Parallel Execution
#         # Novelty: Explicitly modeling Wave (Spectral) vs Particle (Spatial) behavior
#         feat_spectral = self.spectral(x_norm)
#         feat_spatial = self.spatial(x_norm)
#         feat_local = self.local(x_norm)

#         # Gated Fusion
#         # w_spec controls how much we trust the wave dynamics
#         # w_spat controls how much we trust the teleconnection attention
#         mixed = (feat_spectral * self.w_spec) + (feat_spatial * self.w_spat) + feat_local

#         x = shortcut + self.drop_path(mixed)

#         # FFN
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class PatchEmbed(nn.Module):
#     """
#     Standard Patch Embedding from CirT (Kept for compatibility and performance).
#     Compresses Longitude (W) into Channels.
#     """
#     def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         # x: [B, V, H, W]
#         x = self.proj(x) # [B, Dim, H, 1]
#         x = x.flatten(2).transpose(1, 2) # [B, H, Dim]
#         x = self.norm(x)
#         return x

# # ==========================================
# # 2. Main Model Class
# # ==========================================

# class Model(nn.Module):
#     """
#     Holo-Former (Holographic Transformer for S2S).

#     A replacement for CirT that introduces Spatial-Spectral Parallel Mixing.
#     Uses CirT's high-performance hyperparams (Embed=768, Depth=8).
#     """
#     def __init__(
#         self,
#         img_size=[121, 240],
#         input_size=63,
#         # Patch size is conceptually W here due to embedding strategy
#         patch_size=240, 
#         # CirT defaults
#         embed_dim=768,
#         depth=8,
#         decoder_depth=2,
#         num_heads=16,
#         mlp_ratio=4.0,
#         drop_path=0.1,
#         drop_rate=0.1
#     ):
#         super().__init__()

#         self.img_size = img_size
#         self.input_size = input_size
#         self.patch_size = img_size[1] # 240

#         # 1. Embedding (CirT Style)
#         self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
#         self.num_patches = 121 # Latitude points

#         # Learnable Positional Embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         trunc_normal_(self.pos_embed, std=0.02)
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # 2. Backbone (HoloBlocks)
#         dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
#         self.blocks = nn.ModuleList([
#             HoloBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 drop=drop_rate,
#                 drop_path=dpr[i],
#                 norm_layer=nn.LayerNorm,
#                 seq_len=self.num_patches
#             )
#             for i in range(depth)
#         ])

#         self.norm = nn.LayerNorm(embed_dim)

#         # 3. Prediction Head (CirT Style)
#         # Matches output shape logic of CirT
#         self.head = nn.ModuleList()
#         for _ in range(decoder_depth):
#             self.head.append(nn.Linear(embed_dim, embed_dim))
#             self.head.append(nn.GELU())

#         # Output dim handles (2 weeks) * (Input Vars) * (Longitude Width)
#         # Because we compressed Longitude into channel earlier, we expand it back here.
#         output_dim = self.input_size * 2 * self.img_size[1]
#         self.head.append(nn.Linear(embed_dim, output_dim))
#         self.head = nn.Sequential(*self.head)

#         self.initialize_weights()

#     def initialize_weights(self):
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def unpatchify(self, x):
#         """
#         Reconstructs spatial dimensions from the flattened prediction.
#         x: (B, H, V * 2 * W) -> (B, 2, V, H, W)
#         """
#         B, H, D = x.shape
#         W = self.img_size[1]
#         V = self.input_size

#         # CirT's output format is essentially flattened longitude
#         # Reshape: [B, H, 2 * V * W] -> [B, H, 2, V, W]
#         x = x.reshape(B, H, 2, V, W)

#         # Permute to target: [B, 2, V, H, W]
#         x = x.permute(0, 2, 3, 1, 4)
#         return x

#     def forward_encoder(self, x):
#         # x: [B, V, H, W]
#         x = self.token_embeds(x) # [B, 121, 768]
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.norm(x)
#         return x

#     def forward(self, x):
#         # Input: [B, 63, 121, 240]
#         feats = self.forward_encoder(x) # [B, 121, 768]
#         preds = self.head(feats)        # [B, 121, 63*2*240]
#         preds = self.unpatchify(preds)  # [B, 2, 63, 121, 240]
#         return preds







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.layers import DropPath, Mlp
# from timm.models.vision_transformer import trunc_normal_
# import torch.fft

# # ==========================================
# # 1. Core Modules
# # ==========================================

# class SpectralGating(nn.Module):
#     """
#     [View 1] Frequency Domain Branch (Inspired by CirT/FNO).
#     Captures Global Periodic Patterns (Planetary Waves).
#     """
#     def __init__(self, dim, h=121):
#         super().__init__()
#         self.dim = dim
#         self.h = h
#         # Only process half frequencies due to conjugate symmetry of RFFT
#         self.freq_dim = h // 2 + 1 

#         # Complex weights: [Dim, Freq_Dim]
#         # Treat channel mixing efficiently
#         scale = (1 / (dim * dim))
#         # Define weights as real (..., 2) to handle AMP safely
#         self.weights = nn.Parameter(
#             scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
#         )

#     def forward(self, x):
#         # x: [B, N, C] -> N is Latitude (121)
#         B, N, C = x.shape

#         # 1. FFT along the latitude dimension
#         # x_ft: [B, N//2+1, C] (Complex)
#         x_ft = torch.fft.rfft(x, dim=1, norm='ortho')

#         # 2. Spectral Gating
#         w = torch.view_as_complex(self.weights) # [C, Freq]

#         # Permute x_ft to [B, C, Freq] for broadcasting
#         x_ft = x_ft.permute(0, 2, 1)

#         # Spectral Filtering: [B, C, Freq] * [C, Freq] -> [B, C, Freq]
#         out_ft = x_ft * w.unsqueeze(0)

#         # 3. IFFT
#         out_ft = out_ft.permute(0, 2, 1) # Back to [B, Freq, C]
#         x = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')

#         return x

# class SpatialAttention(nn.Module):
#     """
#     [View 2] Spatial Domain Branch (Standard MHSA).
#     Captures Sparse Teleconnections and Non-periodic dependencies.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         # Standard MHSA
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class LocalMixer(nn.Module):
#     """
#     [View 3] Local Refinement Branch.
#     Uses 1D Conv over Latitude to capture local gradients.
#     """
#     def __init__(self, dim, kernel_size=5):
#         super().__init__()
#         self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

#     def forward(self, x):
#         # x: [B, N, C] -> [B, C, N]
#         x = x.transpose(1, 2)
#         x = self.conv(x)
#         x = x.transpose(1, 2)
#         return x

# class HoloBlock(nn.Module):
#     """
#     The Holographic Block (Static Version).
#     Features: Parallel disentangled processing with STATIC learnable weights.
#     """
#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             drop=0.,
#             attn_drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             seq_len=121
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)

#         # --- Dual-Domain Mixer ---
#         # 1. Global Periodic (Spectral)
#         self.spectral = SpectralGating(dim, h=seq_len)
#         # 2. Global Sparse (Spatial)
#         self.spatial = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # 3. Local Continuity
#         self.local = LocalMixer(dim)

#         # Learnable gating weights (Static Parameters)
#         self.w_spec = nn.Parameter(torch.ones(dim) * 0.5)
#         self.w_spat = nn.Parameter(torch.ones(dim) * 0.5)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         # x: [B, N, C]
#         shortcut = x
#         x_norm = self.norm1(x)

#         # Parallel Execution
#         feat_spectral = self.spectral(x_norm)
#         feat_spatial = self.spatial(x_norm)
#         feat_local = self.local(x_norm)

#         # Static Gated Fusion
#         # Note: This is the version WITHOUT AdaptiveRouter
#         mixed = (feat_spectral * self.w_spec) + (feat_spatial * self.w_spat) + feat_local

#         x = shortcut + self.drop_path(mixed)

#         # FFN
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class PatchEmbed(nn.Module):
#     """
#     Standard Patch Embedding from CirT (Aligned with Baseline).
#     Compresses Longitude (W) into Channels.
#     """
#     def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         # x: [B, V, H, W]
#         x = self.proj(x) # [B, Dim, H, 1]
#         x = x.flatten(2).transpose(1, 2) # [B, H, Dim]
#         x = self.norm(x)
#         return x

# # ==========================================
# # 2. Main Model Class
# # ==========================================

# class Model(nn.Module):
#     """
#     Holo-Former (Static Version).
#     Using CirT's efficient embedding but replacing the Attention block
#     with a Multi-View Holographic Block.
#     """
#     def __init__(
#         self,
#         img_size=[121, 240],
#         input_size=63,
#         patch_size=240, 
#         embed_dim=768, # CirT Default
#         depth=8,       # CirT Default
#         decoder_depth=2,
#         num_heads=16,
#         mlp_ratio=4.0,
#         drop_path=0.1,
#         drop_rate=0.1
#     ):
#         super().__init__()

#         self.img_size = img_size
#         self.input_size = input_size
#         self.patch_size = img_size[1] # 240

#         # 1. Embedding
#         self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
#         self.num_patches = 121 # Latitude points

#         # Learnable Positional Embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         trunc_normal_(self.pos_embed, std=0.02)
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # 2. Backbone
#         dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
#         self.blocks = nn.ModuleList([
#             HoloBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 drop=drop_rate,
#                 drop_path=dpr[i],
#                 norm_layer=nn.LayerNorm,
#                 seq_len=self.num_patches
#             )
#             for i in range(depth)
#         ])

#         self.norm = nn.LayerNorm(embed_dim)

#         # 3. Prediction Head (CirT Style)
#         self.head = nn.ModuleList()
#         for _ in range(decoder_depth):
#             self.head.append(nn.Linear(embed_dim, embed_dim))
#             self.head.append(nn.GELU())

#         # Output dim handles (2 weeks) * (Input Vars) * (Longitude Width)
#         output_dim = self.input_size * 2 * self.img_size[1]
#         self.head.append(nn.Linear(embed_dim, output_dim))
#         self.head = nn.Sequential(*self.head)

#         self.initialize_weights()

#     def initialize_weights(self):
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def unpatchify(self, x):
#         """
#         Reconstructs spatial dimensions from the flattened prediction.
#         x: (B, H, V * 2 * W) -> (B, 2, V, H, W)
#         """
#         B, H, D = x.shape
#         W = self.img_size[1]
#         V = self.input_size

#         # CirT's output format is essentially flattened longitude
#         x = x.reshape(B, H, 2, V, W)

#         # Permute to target: [B, 2, V, H, W]
#         x = x.permute(0, 2, 3, 1, 4)
#         return x

#     def forward_encoder(self, x):
#         # x: [B, V, H, W]
#         x = self.token_embeds(x) # [B, 121, 768]
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.norm(x)
#         return x

#     def forward(self, x):
#         # Input: [B, 63, 121, 240]
#         feats = self.forward_encoder(x) # [B, 121, 768]
#         preds = self.head(feats)        # [B, 121, 63*2*240]
#         preds = self.unpatchify(preds)  # [B, 2, 63, 121, 240]
#         return preds



import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import trunc_normal_
import torch.fft

# ==========================================
# 1. Theoretical Sub-Operators (子算子)
# ==========================================

class SpectralOperator(nn.Module):
    """
    Sub-Operator 1: Spectral Operator (S)
    Approximates the Linear Wave Propagation term in PDEs (e.g., Rossby waves).
    Operates in Frequency Domain via FFT.
    """
    def __init__(self, dim, h=121):
        super().__init__()
        self.dim = dim
        self.h = h
        self.freq_dim = h // 2 + 1 

        # Learnable Spectral Filter
        scale = (1 / (dim * dim))
        self.weights = nn.Parameter(
            scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
        )

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        # 1. FFT: Spatial -> Frequency
        # x: [B, C, N] after permute
        x = x.transpose(1, 2)
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho') # [B, C, Freq]

        # 2. Spectral Filtering (Global Convolution)
        w = torch.view_as_complex(self.weights) # [C, Freq]
        out_ft = x_ft * w.unsqueeze(0) # Element-wise mult broadcasting over batch

        # 3. IFFT: Frequency -> Spatial
        x = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        x = x.transpose(1, 2) # Back to [B, N, C]
        return x

class NonLocalOperator(nn.Module):
    """
    Sub-Operator 2: Non-local Integral Operator (K)
    Approximates Long-range Advection/Teleconnections via Self-Attention.
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DifferentialOperator(nn.Module):
    """
    Sub-Operator 3: Differential Operator (D)
    Approximates Diffusion/Local Gradient terms via Finite Difference (Conv1d).
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        # Depthwise conv simulates local Laplacian/Gradient operators
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

# ==========================================
# 2. Dynamic Lie-Trotter Gating (Theoretical Core)
# ==========================================

class DynamicOperatorGate(nn.Module):
    """
    Learns the State-Dependent Coefficients for Operator Splitting.
    Theory: \partial_t u = \lambda_s(u)S(u) + \lambda_k(u)K(u) + \lambda_d(u)D(u)
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.LayerNorm(dim // reduction),
            nn.GELU(),
            # Output 3 weights per channel: [alpha_s, alpha_k, alpha_d]
            nn.Linear(dim // reduction, dim * 3) 
        )

    def forward(self, x):
        # x: [B, N, C]
        # 1. Global Context Pooling (State descriptor)
        ctx = x.mean(dim=1) # [B, C]

        # 2. Predict Operator Coefficients
        weights = self.fc(ctx) # [B, 3*C]
        weights = weights.reshape(x.shape[0], 3, x.shape[2]) # [B, 3, C]

        # 3. Softmax ensures convex combination (stability) or Sigmoid (independence)
        # Using Softmax enforces a competition mechanism between operators
        weights = F.softmax(weights, dim=1).unsqueeze(2) # [B, 3, 1, C]
        return weights

class OperatorSplittingBlock(nn.Module):
    """
    Replaces 'HoloBlock' with a theoretically grounded name.
    Implements Dynamic Operator Splitting.
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

        # --- The Three Sub-Operators ---
        self.op_spectral = SpectralOperator(dim, h=seq_len)
        self.op_nonlocal = NonLocalOperator(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.op_diff = DifferentialOperator(dim)

        # --- Dynamic Gating (The Brain) ---
        # Replaces static w_spec/w_spat
        self.gate = DynamicOperatorGate(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        shortcut = x
        x_norm = self.norm1(x)

        # 1. Compute Sub-Operator Responses (Parallel Execution)
        # Represents solving sub-problems: d_t u = S(u), d_t u = K(u), d_t u = D(u)
        out_s = self.op_spectral(x_norm)
        out_k = self.op_nonlocal(x_norm)
        out_d = self.op_diff(x_norm)

        # Stack: [B, 3, N, C]
        stacked_ops = torch.stack([out_s, out_k, out_d], dim=1)

        # 2. Dynamic Weighting (Lie-Trotter Approximation)
        # weights: [B, 3, 1, C]
        weights = self.gate(x_norm)

        # Weighted Sum: \sum \lambda_i(u) * Op_i(u)
        mixed = torch.sum(stacked_ops * weights, dim=1)

        # 3. Time Stepping (Euler Integration)
        x = shortcut + self.drop_path(mixed)

        # FFN (Non-linear source term)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ==========================================
# 3. Infrastructure (Kept from CirT for fairness)
# ==========================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x) # [B, Dim, H, 1]
        x = x.flatten(2).transpose(1, 2) # [B, H, Dim]
        x = self.norm(x)
        return x

# ==========================================
# 4. Main Model
# ==========================================

class Model(nn.Module):
    """
    OST: Operator-Splitting Transformer.
    """
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        patch_size=240, 
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
        self.patch_size = img_size[1]

        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = 121

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Use the theoretically grounded OperatorSplittingBlock
        self.blocks = nn.ModuleList([
            OperatorSplittingBlock(
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

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())

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
        B, H, D = x.shape
        W = self.img_size[1]
        V = self.input_size
        x = x.reshape(B, H, 2, V, W)
        x = x.permute(0, 2, 3, 1, 4)
        return x

    def forward_encoder(self, x):
        x = self.token_embeds(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        feats = self.forward_encoder(x)
        preds = self.head(feats)
        preds = self.unpatchify(preds)
        return preds

if __name__ == "__main__":
    print("Initializing Operator-Splitting Transformer (OST)...")
    model = Model(
        img_size=[121, 240], 
        input_size=63, 
        embed_dim=128, 
        depth=2, 
        num_heads=4
    )
    model.eval()
    x = torch.randn(2, 63, 121, 240)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")