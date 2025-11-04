import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchdiffeq import odeint


class SphericalHarmonicEmbedding(nn.Module):
    """
    Transforms the input data into spherical harmonic-inspired coefficients.
    Uses a learnable alternative to explicit spherical harmonics.
    """

    def __init__(self, max_degree=10, img_size=[121, 240], in_chans=63, embed_dim=256):
        super().__init__()
        self.max_degree = max_degree
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Make sure embed_dim is even for clean division
        assert embed_dim % 2 == 0, "embed_dim must be even to split between lat and lon embeddings"

        # Initialize position encodings non-in-place
        lat_embed = self._init_latitude_embeddings(img_size[0], embed_dim // 2)
        lon_embed = self._init_longitude_embeddings(img_size[1], embed_dim // 2)

        # Register as parameters
        self.lat_embed = nn.Parameter(lat_embed)
        self.lon_embed = nn.Parameter(lon_embed)

        # Use 2 groups since 256 is cleanly divisible by 2
        self.channel_groups = 2

        # Define sizes for each group to ensure exact match to embed_dim
        # For 63 channels split into 2 groups: 32 and 31 channels
        self.group_sizes = [32, 31]  # these sum to 63

        # For 256 embedding dimensions split into 2 groups: 128 and 128
        self.embed_sizes = [128, 128]  # these sum to 256

        # Learnable projections for each channel group
        self.channel_projections = nn.ModuleList([
            nn.Linear(self.group_sizes[i], self.embed_sizes[i])
            for i in range(self.channel_groups)
        ])

        # Final projection to embedding space
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _init_latitude_embeddings(self, size, dim):
        """Initialize latitude embeddings with sine/cosine values"""
        embeddings = torch.zeros(size, dim)
        lat_values = torch.linspace(-math.pi / 2, math.pi / 2, size)

        for i in range(dim // 2):
            embeddings[:, 2 * i] = torch.sin((i + 1) * lat_values)
            embeddings[:, 2 * i + 1] = torch.cos((i + 1) * lat_values)

        return embeddings

    def _init_longitude_embeddings(self, size, dim):
        """Initialize longitude embeddings with sine/cosine values"""
        embeddings = torch.zeros(size, dim)
        lon_values = torch.linspace(-math.pi, math.pi, size)

        for i in range(dim // 2):
            embeddings[:, 2 * i] = torch.sin((i + 1) * lon_values)
            embeddings[:, 2 * i + 1] = torch.cos((i + 1) * lon_values)

        return embeddings

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input spatial dims {H}x{W} don't match expected {self.img_size[0]}x{self.img_size[1]}"
        assert C == self.in_chans, f"Input has {C} channels, expected {self.in_chans}"

        # Process each channel group separately
        embeddings = []
        start_idx = 0

        for i in range(self.channel_groups):
            end_idx = start_idx + self.group_sizes[i]
            group_x = x[:, start_idx:end_idx]  # [B, group_size, H, W]
            lat_features = group_x.mean(dim=-1)  # [B, group_size, H]
            lat_features = lat_features.transpose(1, 2)  # [B, H, group_size]
            proj_features = self.channel_projections[i](lat_features)  # [B, H, embed_size[i]]
            embeddings.append(proj_features)
            start_idx = end_idx

        x_embedded = torch.cat(embeddings, dim=-1)  # [B, H, embed_dim]

        lat_pos = self.lat_embed.unsqueeze(0).expand(B, -1, -1)  # [B, H, embed_dim//2]
        lon_pos = self.lon_embed.mean(dim=0, keepdim=True).expand(B, H, -1)  # [B, H, embed_dim//2]
        pos_embed = torch.cat([lat_pos, lon_pos], dim=-1)  # [B, H, embed_dim]
        x_embedded = x_embedded + pos_embed
        x_embedded = self.proj(x_embedded)

        return x_embedded


class WaveletDecomposition(nn.Module):
    def __init__(self, n_levels=3, embed_dim=256):
        super().__init__()
        self.n_levels = n_levels
        self.embed_dim = embed_dim

        self.decompose_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim * 2)
            ) for _ in range(n_levels)
        ])

        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_levels + 1)
        ])

    def forward(self, x):
        B, H, D = x.shape
        outputs = []
        approximation = x
        for i in range(self.n_levels):
            decomposed = self.decompose_layers[i](approximation)
            next_approx, detail = torch.chunk(decomposed, 2, dim=-1)
            detail_proj = self.projections[i](detail)
            outputs.append(detail_proj)
            approximation = next_approx
        approx_proj = self.projections[-1](approximation)
        outputs.append(approx_proj)
        return outputs[::-1]


class PhysicsODEFunc(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.diffusion = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.001)
        self.advection = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.01)
        self.forcing = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, x):
        padded = F.pad(x, (0, 0, 1, 1), mode='constant', value=0.0)
        diffusion_term = self.diffusion * (padded[:, 2:, :] + padded[:, :-2, :] - 2 * x)
        advection_term = self.advection * (padded[:, 2:, :] - padded[:, :-2, :]) / 2
        physics_term = diffusion_term + advection_term + self.forcing
        correction = self.net(x)
        combined = physics_term + self.scale_factor * correction
        return torch.tanh(combined) * 0.5


class PhysicsODEBlock(nn.Module):
    def __init__(self, odefunc, integration_time=1.0):
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = integration_time

    def forward(self, x):
        integration_time = torch.tensor([0, self.integration_time]).float().to(x.device)
        try:
            out = odeint(
                self.odefunc,
                x,
                integration_time,
                method='euler',
                options={'step_size': 0.1}
            )
            return out[1]
        except Exception:
            with torch.no_grad():
                dx = self.odefunc(0, x)
            result = x + dx * 0.1
            return result


class TeleconnectionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_patterns = 5
        self.teleconnection_patterns = nn.Parameter(torch.randn(self.n_patterns, dim))
        self.pattern_weights = nn.Linear(dim, self.n_patterns)

    def forward(self, x):
        B, H, D = x.shape
        q = self.q_proj(x).reshape(B, H, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, H, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, H, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        global_repr = x.mean(dim=1, keepdim=True)
        pattern_weights = self.pattern_weights(global_repr).softmax(dim=-1)
        weighted_patterns = pattern_weights @ self.teleconnection_patterns
        tel_q = self.q_proj(weighted_patterns).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        tel_attn = (tel_q @ k.transpose(-2, -1)) * self.scale
        tel_attn = tel_attn.expand(-1, -1, H, -1)
        attn = attn + 0.2 * tel_attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TeleconnectionAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=4, n_levels=3, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.n_levels = n_levels
        self.transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop
                ) for _ in range(depth)
            ]) for _ in range(n_levels + 1)
        ])
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for _ in range(n_levels)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, multi_scale_x):
        outputs = []
        for i, x in enumerate(multi_scale_x):
            for block in self.transformers[i]:
                x = block(x)
            outputs.append(x)
        for i in range(self.n_levels):
            if i < len(outputs) - 1:
                lower_freq = outputs[i]
                higher_freq = outputs[i + 1]
                fused = torch.cat([lower_freq, higher_freq], dim=-1)
                outputs[i + 1] = self.cross_scale_fusion[i](fused)
        return [self.norm(x) for x in outputs]


class OutputLayer(nn.Module):
    def __init__(self, embed_dim=256, output_size=63, img_size=[121, 240]):
        super().__init__()
        self.img_size = img_size
        self.output_size = output_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 2 * output_size * img_size[1]),
        )

    def forward(self, x):
        B, H, D = x.shape
        output = self.head(x)
        output = output.reshape(B, H, 2, self.output_size, self.img_size[1])
        output = output.permute(0, 2, 3, 1, 4)
        return output


class Model(nn.Module):
    def __init__(
        self,
        img_size=[121, 240],
        input_size=63,
        output_size=63,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        wavelet_levels=3,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.input_size = input_size
        self.output_size = output_size

        self.embed = SphericalHarmonicEmbedding(
            img_size=img_size,
            in_chans=input_size,
            embed_dim=embed_dim
        )
        self.wavelet = WaveletDecomposition(
            n_levels=wavelet_levels,
            embed_dim=embed_dim
        )
        self.ode_func = PhysicsODEFunc(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 2
        )
        self.ode_block = PhysicsODEBlock(
            odefunc=self.ode_func,
            integration_time=1.0
        )
        self.transformer = HierarchicalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            n_levels=wavelet_levels,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate
        )
        self.output = OutputLayer(
            embed_dim=embed_dim,
            output_size=output_size,
            img_size=img_size
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        embedded = self.embed(x)
        multi_scale = self.wavelet(embedded)
        evolved_scales = []
        for scale in multi_scale:
            try:
                evolved_scale = self.ode_block(scale)
                evolved_scales.append(evolved_scale)
            except Exception:
                evolved_scales.append(scale)
        multi_scale = evolved_scales
        multi_scale = self.transformer(multi_scale)
        combined = torch.stack(multi_scale, dim=0).mean(dim=0)
        output = self.output(combined)
        return output


