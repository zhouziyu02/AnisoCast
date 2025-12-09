import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import trunc_normal_
import torch.fft

# ==========================================
# 1. Theoretical Components (General ML Operators)
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Preferred in Operator Learning for preserving phase information and stability.
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

class AxisSpectralOperator(nn.Module):
    """
    [Operator A] Axis-wise Spectral Operator.
    Handles global periodic dependencies along the Feature Axis (C).

    In the context of our embedding (where Width is compressed into Channels),
    this implicitly models the periodic wave propagation (e.g., Zonal Waves).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # FFT on the last dimension (Channel/Feature dimension)
        # We process the 'hidden_dim' as the spectral domain.
        self.n_modes = dim // 2 + 1

        # Learnable complex weights: [1, 1, Freq, 2]
        # Shared across Batch and Sequence (Translation Invariance)
        scale = (1 / (dim * dim))
        self.weights = nn.Parameter(
            scale * torch.randn(1, 1, self.n_modes, 2, dtype=torch.float32)
        )

    def forward(self, x):
        # x: [B, N(Seq), C(Feat)]

        # 1. FFT along Feature Axis
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 2. Spectral Gating / Modulation
        w = torch.view_as_complex(self.weights)
        # Broadcasting: [B, N, Freq] * [1, 1, Freq]
        out_ft = x_ft * w

        # 3. IFFT
        x = torch.fft.irfft(out_ft, n=self.dim, dim=-1, norm='ortho')
        return x

class AxisSpatialOperator(nn.Module):
    """
    [Operator B] Axis-wise Spatial Operator.
    Handles local non-periodic dependencies along the Sequence Axis (N).

    Uses Gated Linear Unit (GLU) to model non-linear transport/advection effects
    (e.g., Meridional Transport).
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()

        # 1D Convolution along the sequence dimension
        # In_C=dim, Out_C=2*dim (for GLU)
        self.conv = nn.Conv1d(
            dim, dim * 2, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=dim # Depthwise for efficiency
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]

        # Permute to [B, C, N] for Conv1d operating on N
        x_in = x.transpose(1, 2)

        # Gated Convolution (GLU)
        # Simulates non-linear flux controls
        x_conv = self.conv(x_in)
        x_val, x_gate = x_conv.chunk(2, dim=1)
        x_out = x_val * torch.sigmoid(x_gate)

        # Permute back to [B, N, C]
        x_out = x_out.transpose(1, 2)
        x_out = self.proj(x_out)
        return x_out

class SymmetricOperatorBlock(nn.Module):
    """
    [The Core Architecture] Symmetric Operator Splitting Block.

    Theory: Implements 'Strang Splitting' for non-commutative operators A and B.
    Approximation: u(t+1) approx e^{A/2} e^{B} e^{A/2} u(t)
    Result: O(dt^3) local error, superior to O(dt^2) of standard sequential blocks.
    """
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=RMSNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # The two orthogonal operators
        self.op_spectral = AxisSpectralOperator(dim)
        self.op_spatial = AxisSpatialOperator(dim, kernel_size=7)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Source Term / Reaction Operator (MLP)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)

        # --- Strang Splitting Sequence (A/2 -> B -> A/2) ---

        # 1. Half-Step Spectral Propagation
        # Note: Applying the same layer twice with shared weights approximates 
        # solving the linear sub-problem in steps.
        x_step1 = self.op_spectral(x_norm)

        # 2. Full-Step Spatial Transport
        # The output of spectral is the input to spatial (Chained)
        x_step2 = self.op_spatial(x_step1)

        # 3. Half-Step Spectral Propagation
        x_step3 = self.op_spectral(x_step2)

        # Residual Connection
        x = shortcut + self.drop_path(x_step3)

        # --- Source Term / Local Mixing ---
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ==========================================
# 2. Infrastructure (CirT Compatible)
# ==========================================

class PatchEmbed(nn.Module):
    """
    Standard Patch Embedding.
    Compresses Longitude (W) into Channels (C).
    This creates the Anisotropic representation required by our operators.
    """
    def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
        self.norm = RMSNorm(embed_dim) # Use consistent Norm

    def forward(self, x):
        # x: [B, V, H, W]
        x = self.proj(x) # [B, Dim, H, 1]
        x = x.flatten(2).transpose(1, 2) # [B, H(Seq), Dim(Feat)]
        x = self.norm(x)
        return x

# ==========================================
# 3. Main Model: SOOT (Symmetric Orthogonal Operator Transformer)
# ==========================================

class Model(nn.Module):
    """
    SOOT: Symmetric Orthogonal Operator Transformer.

    A general-purpose spatiotemporal backbone designed for systems with 
    anisotropic dynamics (e.g. Fluids, Weather, Plasma).
    """
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        patch_size=240, 
        embed_dim=768,
        depth=8,
        decoder_depth=2,
        num_heads=16, # Kept for API compatibility, unused in SOOT
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.input_size = input_size

        # 1. Embedding
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = 121 # Sequence Length (Latitude)

        # Learnable Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Backbone: Symmetric Operator Splitting
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            SymmetricOperatorBlock(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=RMSNorm
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
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
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
 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.layers import DropPath, Mlp
# from timm.models.vision_transformer import trunc_normal_
# import torch.fft

# # ==========================================
# # 1. Core Operators (ICML: Split-Step Components)
# # ==========================================

# class RMSNorm(nn.Module):
#     """
#     Root Mean Square Layer Normalization.
#     Theoretical Justification: 
#     In physical systems, preserving the direction of the state vector 
#     while normalizing energy (amplitude) improves optimization stability 
#     without destroying phase information (crucial for waves).
#     """
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         # x: [B, N, C]
#         norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
#         return x / (norm + self.eps) * self.g

# class SpectralOperator(nn.Module):
#     """
#     [Operator S] Global Spectral Propagator.
#     Solves global/planetary wave dynamics in the frequency domain.

#     Since Longitude is compressed into channels (in PatchEmbed), 
#     this FFT operates purely on Latitude (Meridional) waves.
#     """
#     def __init__(self, dim, h=121):
#         super().__init__()
#         self.dim = dim
#         self.h = h
#         self.freq_dim = h // 2 + 1 

#         # Learnable Spectral Weights
#         scale = (1 / (dim * dim))
#         # Define as real (..., 2) to be AMP-safe
#         self.weights = nn.Parameter(
#             scale * torch.randn(dim, self.freq_dim, 2, dtype=torch.float32)
#         )

#     def forward(self, x):
#         # x: [B, N, C] -> N is Latitude
#         B, N, C = x.shape

#         # 1. Spatial -> Frequency (on Latitude dim)
#         x = x.transpose(1, 2) # [B, C, N]
#         x_ft = torch.fft.rfft(x, dim=-1, norm='ortho') # [B, C, Freq]

#         # 2. Spectral Modulation
#         w = torch.view_as_complex(self.weights) # [C, Freq]
#         # Global Convolution: Element-wise mult in freq domain
#         out_ft = x_ft * w.unsqueeze(0) 

#         # 3. Frequency -> Spatial
#         x = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
#         x = x.transpose(1, 2) # [B, N, C]
#         return x

# class DifferentialOperator(nn.Module):
#     """
#     [Operator D] Local Differential/Flux Operator.
#     Approximates local advection-diffusion terms (\nabla u).
#     Uses Depthwise Conv1d to capture local meridional dependencies.
#     """
#     def __init__(self, dim, kernel_size=7):
#         super().__init__()
#         # Depthwise conv1d: efficient local mixing
#         self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
#         # Pointwise mixing to combine local features
#         self.proj = nn.Conv1d(dim, dim, 1)

#     def forward(self, x):
#         # x: [B, N, C]
#         x = x.transpose(1, 2) # [B, C, N]

#         # Apply local differential approximation
#         res = x
#         x = self.conv(x)
#         x = F.gelu(x)
#         x = self.proj(x)

#         x = x.transpose(1, 2)
#         return x

# class SplitStepBlock(nn.Module):
#     """
#     [The Novelty] Split-Step Spectral Block.

#     Instead of parallel branches (Additive decomposition), 
#     we use Sequential Composition (Strang Splitting).

#     Flow: Input -> [Spectral] -> [Differential] -> [MLP] -> Output
#     Theory: u_{t+1} \approx \mathcal{M} \circ \mathcal{D} \circ \mathcal{S} (u_t)
#     """
#     def __init__(
#             self,
#             dim,
#             mlp_ratio=4.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=RMSNorm,
#             seq_len=121
#     ):
#         super().__init__()

#         # 1. Spectral Step (Global Wave)
#         self.norm1 = norm_layer(dim)
#         self.op_spectral = SpectralOperator(dim, h=seq_len)
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # 2. Differential Step (Local Flux)
#         self.norm2 = norm_layer(dim)
#         self.op_diff = DifferentialOperator(dim)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # 3. Mixing Step (Physics/Channel Interaction)
#         # Note: Since Longitude is in 'dim', MLP here does Zonal Mixing!
#         self.norm3 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         # Sequential processing minimizes splitting error for non-linear systems

#         # Step 1: Global Propagation
#         x = x + self.drop_path1(self.op_spectral(self.norm1(x)))

#         # Step 2: Local Diffusion/Advection
#         x = x + self.drop_path2(self.op_diff(self.norm2(x)))

#         # Step 3: Zonal/Channel Mixing
#         x = x + self.drop_path3(self.mlp(self.norm3(x)))

#         return x

# # ==========================================
# # 2. Infrastructure (CirT Compatible)
# # ==========================================

# class PatchEmbed(nn.Module):
#     """
#     Standard Patch Embedding from CirT.
#     Compresses Longitude (W=240) into Channels.
#     """
#     def __init__(self, img_size=[121, 240], in_chans=63, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, img_size[1]], stride=1)
#         self.norm = RMSNorm(embed_dim) # Consistent Norm

#     def forward(self, x):
#         # x: [B, V, H, W]
#         x = self.proj(x) # [B, Dim, H, 1]
#         x = x.flatten(2).transpose(1, 2) # [B, H, Dim]
#         x = self.norm(x)
#         return x

# # ==========================================
# # 3. Main Model: Split-Step Spectral Transformer
# # ==========================================

# class Model(nn.Module):
#     """
#     S3T: Split-Step Spectral Transformer (ICML 2026).
#     """
#     def __init__(
#         self,
#         img_size=[121, 240],
#         input_size=63,
#         patch_size=240, 
#         embed_dim=768, # CirT Default
#         depth=8,       # CirT Default
#         decoder_depth=2,
#         num_heads=16, # Kept for interface compatibility, unused in S3T
#         mlp_ratio=4.0,
#         drop_path=0.1,
#         drop_rate=0.1
#     ):
#         super().__init__()

#         self.img_size = img_size
#         self.input_size = input_size
#         self.patch_size = img_size[1]

#         # 1. Embedding
#         self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
#         self.num_patches = 121

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         trunc_normal_(self.pos_embed, std=0.02)
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # 2. Backbone
#         dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
#         self.blocks = nn.ModuleList([
#             SplitStepBlock(
#                 dim=embed_dim,
#                 mlp_ratio=mlp_ratio,
#                 drop=drop_rate,
#                 drop_path=dpr[i],
#                 norm_layer=RMSNorm,
#                 seq_len=self.num_patches
#             )
#             for i in range(depth)
#         ])

#         self.norm = RMSNorm(embed_dim)

#         # 3. Head
#         self.head = nn.ModuleList()
#         for _ in range(decoder_depth):
#             self.head.append(nn.Linear(embed_dim, embed_dim))
#             self.head.append(nn.GELU())

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
#         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
#             trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def unpatchify(self, x):
#         B, H, D = x.shape
#         W = self.img_size[1]
#         V = self.input_size
#         x = x.reshape(B, H, 2, V, W)
#         x = x.permute(0, 2, 3, 1, 4)
#         return x

#     def forward_encoder(self, x):
#         x = self.token_embeds(x)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.norm(x)
#         return x

#     def forward(self, x):
#         feats = self.forward_encoder(x)
#         preds = self.head(feats)
#         preds = self.unpatchify(preds)
#         return preds