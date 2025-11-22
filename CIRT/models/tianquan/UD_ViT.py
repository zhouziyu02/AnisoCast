import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange, repeat
import torch.utils.checkpoint

# Determine available attention mode
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'Attention mode is {ATTENTION_MODE}')


def LoadConstantMask(data_path, device, batch_size):
    """
    Load static geographic masks (land, soil, topography) and downsample them.
    Returns tensors of shape [batch_size, 1, h, w] on the given device.
    """
    # Sample indices to reduce resolution to 32x64
    H_idx = np.round(np.linspace(0, 720, 32, endpoint=True)).astype(int)
    W_idx = np.round(np.linspace(0, 1440 - 1, 64, endpoint=True)).astype(int)

    def load_and_repeat(fname):
        arr = np.load(os.path.join(data_path, fname))[H_idx][:, W_idx]
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        return t.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    land_mask = load_and_repeat("land_mask.npy")
    soil_type = load_and_repeat("soil_type.npy")
    topography = load_and_repeat("topography.npy")
    return land_mask, soil_type, topography


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal embeddings for timesteps.
    Args:
        timesteps: Tensor of shape [N], may be fractional.
        dim: output dimension.
        max_period: controls frequency range.
    Returns:
        [N, dim] embedding tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Diffusion(nn.Module):
    """Simple 1D convolutional diffusion block."""

    def __init__(self, planes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(planes, 2 * planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        B, L, D = x.shape
        y = x.view(B, D, L)
        y = self.net(y)
        return y.view(B, L, D)


class Attention(nn.Module):
    """
    Multi-head attention with optional local window or memory-efficient global.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'B L (k h d) -> k B h L d', k=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.window_size and self.window_size < L:
            out = self._local_attention(q, k, v)
        else:
            out = self._global_attention(q, k, v)

        out = rearrange(out, 'B h L d -> B L (h d)')
        out = self.proj(out)
        return self.proj_drop(out)

    def _local_attention(self, q, k, v):
        """Compute attention in non-overlapping local windows."""
        B, H, L, D = q.shape
        w = self.window_size
        # reshape to (B * windows)...
        q_w = rearrange(q, 'B H (n w) D -> (B n) H w D', w=w)
        k_w = rearrange(k, 'B H (n w) D -> (B n) H w D', w=w)
        v_w = rearrange(v, 'B H (n w) D -> (B n) H w D', w=w)

        attn = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_w = attn @ v_w  # (B*n, H, w, D)
        return rearrange(out_w, '(B n) H w D -> B H (n w) D', B=B)

    def _global_attention(self, q, k, v):
        """Chunked global attention to reduce memory usage."""
        B, H, L, D = q.shape
        chunk = 256
        out_chunks = []
        for i in range(0, L, chunk):
            q_c = q[:, :, i:i+chunk, :]
            attn = (q_c @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_chunks.append(attn @ v)
        return torch.cat(out_chunks, dim=2)


class Block(nn.Module):
    """Transformer block with optional skip and checkpointing."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads, qkv_bias, qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = Mlp(dim, mlp_hidden, act_layer=nn.GELU)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip_tensor=None):
        if self.skip_linear is not None and skip_tensor is not None:
            x = self.skip_linear(torch.cat([x, skip_tensor], dim=-1))
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PEDadd_UViT(nn.Module):
    """
    U-shaped Vision Transformer with additive diffusion at three stages.
    """

    def __init__(self, embed_dim=1536, depth=8, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, skip=True):
        super().__init__()
        self.embed_dim = embed_dim

        # diffusion modules
        self.diff1 = Diffusion(embed_dim)
        self.diff2 = Diffusion(embed_dim)
        self.diff3 = Diffusion(embed_dim)

        # encoder blocks
        half = depth // 2
        self.in_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  norm_layer, skip=False, use_checkpoint=use_checkpoint)
            for _ in range(half)
        ])

        self.mid_block = Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                               norm_layer, skip=False, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(half)
        ])

        self.norm   = norm_layer(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights: truncated normal for Linear, constant for LayerNorm."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, use_diffusion=True):
        """
        Forward pass with optional additive noise from diffusion modules.
        Returns output tensor of shape [B, L, D].
        """
        self.mus = []
        self.sigmas = []
        self.scales = []

        skips = []
        # encode
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # diffusion stage 1
        out = x
        sigma = self.diff1(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # middle block
        x = self.mid_block(x)

        # diffusion stage 2
        out = x
        sigma = self.diff2(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # decode
        for blk in self.out_blocks:
            skip = skips.pop()
            x = blk(x, skip)

        # diffusion stage 3
        out = x
        sigma = self.diff3(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # final norm+linear
        x = self.norm(x)
        return self.linear(x)

    def _record(self, mu, sigma):
        self.mus.append(mu)
        self.sigmas.append(sigma)
        self.scales.append(float(sigma.mean().detach()))
