import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import trunc_normal_
import torch.fft

# ==========================================
# 1. Theoretical Sub-Operators (算子实现)
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Stabilizes operator learning by preserving vector direction (phase info)
    while normalizing energy (amplitude).
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, N, C]
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class SpectralOperator(nn.Module):
    """
    [Operator S] Spectral Dispersive Operator.
    Approximates linear wave propagation in frequency domain.
    Theory: \mathcal{F}(u) -> W \cdot \mathcal{F}(u)
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

        # 1. FFT
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')

        # 2. Spectral Filtering
        # Handle AMP safety by using real weights view as complex
        w = torch.view_as_complex(self.weights) # [C, Freq]

        # Permute for broadcasting: [B, N_freq, C] -> [B, C, N_freq]
        x_ft = x_ft.permute(0, 2, 1)

        # Global Convolution in Freq Domain
        out_ft = x_ft * w.unsqueeze(0)

        # 3. IFFT
        out_ft = out_ft.permute(0, 2, 1)
        x = torch.fft.irfft(out_ft, n=N, dim=1, norm='ortho')
        return x

class NonLocalOperator(nn.Module):
    """
    [Operator K] Non-local Integral Operator.
    Approximates long-range advection/teleconnection via Kernel integration (Attention).
    Theory: \int K(x, y) u(y) dy
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
    [Operator D] Local Differential Operator.
    Approximates local gradients/diffusion terms via Finite Difference (Convolution).
    Theory: \nabla \cdot (c \nabla u)
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        # Depthwise conv acts as a local laplacian/gradient approximator
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N]
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

    Contribution: 
    Instead of fixed splitting (u_{t+1} = S(K(D(u)))), we propose adaptive splitting:
    u_{t+1} = u_t + \lambda_s(u)S(u) + \lambda_k(u)K(u) + \lambda_d(u)D(u)

    This approximates a higher-order numerical scheme where step sizes adapt to local stiffness.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            RMSNorm(dim // reduction),
            nn.GELU(),
            # Output 3 weights per channel: [alpha_s, alpha_k, alpha_d]
            nn.Linear(dim // reduction, dim * 3) 
        )

    def forward(self, x):
        # x: [B, N, C]

        # 1. Global Context Pooling (State descriptor)
        ctx = x.mean(dim=1) # [B, C]

        # 2. Predict Operator Coefficients
        # weights: [B, C, 3] -> [B, 3, C]
        weights = self.fc(ctx) # [B, 3*C]
        weights = weights.reshape(x.shape[0], 3, x.shape[2]) 

        # 3. Sigmoid/Softmax
        # Use Softmax to enforce a competitive resource allocation (Conservation of Energy concept)
        # Or Sigmoid to allow independent scaling. Softmax is more "Splitting-like".
        weights = F.softmax(weights, dim=1).unsqueeze(2) # [B, 3, 1, C]

        return weights

class OperatorSplittingBlock(nn.Module):
    """
    OST Block: Operator Splitting Transformer Block.
    Replaces the generic 'HoloBlock'.
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
            norm_layer=nn.LayerNorm, # We will swap this to RMSNorm in Model init
            seq_len=121
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # --- The Three Sub-Operators ---
        self.op_spectral = SpectralOperator(dim, h=seq_len)
        self.op_nonlocal = NonLocalOperator(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.op_diff = DifferentialOperator(dim)

        # --- Dynamic Gating (The Brain) ---
        self.gate = DynamicOperatorGate(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        shortcut = x
        x_norm = self.norm1(x)

        # 1. Compute Sub-Operator Responses (Parallel Execution)
        # Represents solving sub-problems d_t u = S(u), d_t u = K(u), d_t u = D(u)
        out_s = self.op_spectral(x_norm)
        out_k = self.op_nonlocal(x_norm)
        out_d = self.op_diff(x_norm)

        # Coupling Trick (Optional but recommended for Performance): 
        # Allow slight non-linear interaction before summation (Simulates Operator Coupling)
        # E.g. Advection affects Diffusion.
        # This breaks simple linearity and adds expressivity without extra params.
        out_d = out_d * (1 + torch.tanh(out_k)) # Non-local flow modulates local diffusion

        # 2. Dynamic Weighting (Lie-Trotter Approximation)
        # Stack: [B, 3, N, C]
        stacked_ops = torch.stack([out_s, out_k, out_d], dim=1)

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
# 3. Infrastructure (Aligned with Baseline)
# ==========================================

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
# 4. Main Model
# ==========================================

class Model(nn.Module):
    """
    OST: Operator-Splitting Transformer (ICML Version).

    Theoretical Claim:
    A neural operator architecture that approximates the evolution operator 
    of chaotic systems via dynamic Lie-Trotter splitting.
    """
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        patch_size=124, 
        embed_dim=256,
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

        # 1. Embedding
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = 121

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Backbone (Operator Splitting Blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            OperatorSplittingBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=RMSNorm, # Use RMSNorm for better stability in Deep Operator Networks
                seq_len=self.num_patches
            )
            for i in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)

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
        elif isinstance(m, nn.LayerNorm): # Keep for compatibility
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

 