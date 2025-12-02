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
    [View 1] Frequency Domain Branch.
    Captures Global Periodic Patterns (Planetary Waves).
    """
    def __init__(self, dim, h=121):
        super().__init__()
        self.dim = dim
        self.h = h
        self.freq_dim = h // 2 + 1 

        scale = (1 / (dim * dim))
        self.weights = nn.Parameter(
            scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
        )

    def forward(self, x):
        B, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')

        w = torch.view_as_complex(self.weights) # [C, Freq]
        x_ft = x_ft.permute(0, 2, 1) # [B, C, Freq]

        # Spectral Filtering
        out_ft = x_ft * w.unsqueeze(0)

        out_ft = out_ft.permute(0, 2, 1)
        x = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')
        return x

class SpatialAttention(nn.Module):
    """
    [View 2] Spatial Domain Branch.
    Captures Sparse/Long-range Teleconnections.
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

class LocalMixer(nn.Module):
    """
    [View 3] Local Refinement Branch.
    Captures Local Gradients/Continuity.
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        # Depthwise conv for efficiency
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class AdaptiveRouter(nn.Module):
    """
    [Novelty Module] Adaptive Multi-View Fusion Router.
    Dynamically re-weights the three branches based on input context.
    Analogy: SK-Net / MoE Gating but for Inductive Biases.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        # Global Context Descriptor
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Lightweight Router Network
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * 3, bias=False) # Output 3 weights per channel
        )

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        # 1. Extract Global Context: [B, C]
        context = x.transpose(1, 2).mean(dim=2)

        # 2. Predict Weights: [B, 3*C] -> [B, 3, C]
        # We predict separate weights for Spectral, Spatial, and Local branches per channel
        attn = self.fc(context).reshape(B, 3, C)

        # 3. Softmax across the 3 branches
        attn = F.softmax(attn, dim=1) 

        # Returns: [B, 3, 1, C] for easy broadcasting
        return attn.unsqueeze(2)

class HoloBlock(nn.Module):
    """
    The Optimized HoloBlock with Adaptive Fusion.
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

        # --- Three Parallel Views ---
        self.spectral = SpectralGating(dim, h=seq_len)
        self.spatial = SpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.local = LocalMixer(dim)

        # --- Adaptive Router ---
        self.router = AdaptiveRouter(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        shortcut = x
        x_norm = self.norm1(x)

        # 1. Compute Branches
        feat_spectral = self.spectral(x_norm)
        feat_spatial = self.spatial(x_norm)
        feat_local = self.local(x_norm)

        # 2. Stack Branches: [B, 3, N, C]
        feats = torch.stack([feat_spectral, feat_spatial, feat_local], dim=1)

        # 3. Compute Adaptive Weights
        # weights: [B, 3, 1, C] (Channel-wise attention for each branch)
        weights = self.router(x_norm)

        # 4. Weighted Fusion
        # Element-wise multiplication and sum over branches
        mixed = torch.sum(feats * weights, dim=1)

        x = shortcut + self.drop_path(mixed)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """
    Standard Patch Embedding from CirT.
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
    Adaptive Holo-Former.

    Key Features:
    1. Inherits CirT's efficient embedding (Longitude collapse).
    2. Introduces Triple-View Modeling (Spectral, Spatial, Local).
    3. Adds Adaptive Routing (Dynamic Inductive Bias).
    """
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        patch_size=240, 
        embed_dim=768, # Keep CirT default
        depth=8,       # Keep CirT default
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

        # 1. Embedding
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = 121

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Backbone with Adaptive HoloBlocks
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

        # 3. Head
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
        """
        x: (B, H, V * 2 * W) -> (B, 2, V, H, W)
        """
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
