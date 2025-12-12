import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import trunc_normal_
import torch.fft
import contextlib

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
        self.n_modes = dim // 2 + 1

        # Learnable complex weights stored as real/imag parts:
        # [1, 1, Freq, 2]
        # -----------------------------
        # MOD 1 (IMPORTANT):
        # Near-identity init for spectral gate
        # -----------------------------
        self.weights = nn.Parameter(torch.empty(1, 1, self.n_modes, 2, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        # Near-identity complex gate: w ≈ 1 + 0j + tiny noise
        with torch.no_grad():
            self.weights.zero_()
            self.weights[..., 0].fill_(1.0)  # real part = 1
            self.weights.add_(1e-3 * torch.randn_like(self.weights))

    def forward(self, x):
        # x: [B, N(Seq), C(Feat)]
        orig_dtype = x.dtype

        # -----------------------------
        # NEW FIX:
        # Avoid ComplexHalf in AMP by forcing FFT path to fp32
        # -----------------------------
        if x.is_cuda:
            amp_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            x_f = x.float()
            x_ft = torch.fft.rfft(x_f, dim=-1, norm='ortho')

            # Make sure weights are float32 before view_as_complex
            w = torch.view_as_complex(self.weights.float())
            out_ft = x_ft * w

            x_out = torch.fft.irfft(out_ft, n=self.dim, dim=-1, norm='ortho')

        return x_out.to(orig_dtype)


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
            padding=kernel_size // 2,
            groups=dim  # Depthwise for efficiency
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]

        # Permute to [B, C, N] for Conv1d operating on N
        x_in = x.transpose(1, 2)

        # Gated Convolution (GLU)
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
        layer_scale_init=1e-3,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # The two orthogonal operators
        self.op_spectral = AxisSpectralOperator(dim)
        self.op_spatial = AxisSpatialOperator(dim, kernel_size=7)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Source Term / Reaction Operator (MLP)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )

        # -----------------------------
        # MOD 2 (IMPORTANT):
        # LayerScale for residual branches
        # -----------------------------
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)

        # --- Strang Splitting Sequence (A/2 -> B -> A/2) ---

        # 1. Half-Step Spectral Propagation
        x_step1 = self.op_spectral(x_norm)

        # 2. Full-Step Spatial Transport
        x_step2 = self.op_spatial(x_step1)

        # 3. Half-Step Spectral Propagation
        x_step3 = self.op_spectral(x_step2)

        # Residual Connection with LayerScale
        x = shortcut + self.drop_path(self.gamma1 * x_step3)

        # --- Source Term / Local Mixing with LayerScale ---
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
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
    def __init__(self, img_size=(121, 240), in_chans=63, embed_dim=768):
        super().__init__()
        self.img_size = list(img_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, self.img_size[1]), stride=1)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        # x: [B, V, H, W]
        x = self.proj(x)                      # [B, Dim, H, 1]
        x = x.flatten(2).transpose(1, 2)      # [B, H(Seq), Dim(Feat)]
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
        img_size=(121, 240),
        input_size=63,
        patch_size=240,          # kept for API compatibility
        embed_dim=768,
        depth=8,
        decoder_depth=2,
        num_heads=16,            # kept for API compatibility, unused in SOOT
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1
    ):
        super().__init__()

        self.img_size = list(img_size)
        self.input_size = input_size

        # 1. Embedding
        self.token_embeds = PatchEmbed(self.img_size, input_size, embed_dim)
        self.num_patches = self.img_size[0]   # Sequence Length (Latitude)

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
                norm_layer=RMSNorm,
                layer_scale_init=1e-3
            )
            for i in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)

        # 3. Head
        head_layers = []
        for _ in range(decoder_depth):
            head_layers.append(nn.Linear(embed_dim, embed_dim))
            head_layers.append(nn.GELU())

        output_dim = self.input_size * 2 * self.img_size[1]
        head_layers.append(nn.Linear(embed_dim, output_dim))
        self.head = nn.Sequential(*head_layers)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # AxisSpectralOperator has customized reset_parameters()
        # Avoid overwriting it via global init.
        if isinstance(m, AxisSpectralOperator):
            return

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def unpatchify(self, x):
        # x: [B, H, output_dim]
        B, H, D = x.shape
        W = self.img_size[1]
        V = self.input_size
        # output_dim = 2 * V * W
        x = x.reshape(B, H, 2, V, W)
        x = x.permute(0, 2, 3, 1, 4)  # [B, 2, V, H, W]
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
        feats = self.forward_encoder(x)   # [B, H, C]
        preds = self.head(feats)          # [B, H, 2*V*W]
        preds = self.unpatchify(preds)    # [B, 2, V, H, W]
        return preds