from timm.models.vision_transformer import PatchEmbed
import torch
import numpy as np
from torch import nn
from einops import rearrange, repeat


class MultiFourierEncoder:
    """
    Fourier-based encoder for positional and area features on geospatial patches.
    """

    def __init__(self, embed_dim, patch_size, device=None):
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.device = device

    def set_device(self, device):
        """Set torch device for encoding tensors."""
        self.device = device

    def fourier_encode(self, x, embed_dim=None, lambda_min=0.01, lambda_max=10000):
        """
        Apply Fourier feature mapping to input x.
        Args:
            x (array-like): Scalar values to encode.
            embed_dim (int): Total output dimension (default self.embed_dim).
            lambda_min, lambda_max (float): Min/max wavelengths.
        Returns:
            Tensor of shape (..., embed_dim)
        """
        if embed_dim is None:
            embed_dim = self.embed_dim

        # Generate logarithmically spaced wavelengths
        indices = np.arange(embed_dim // 2)
        lambdas = np.exp(
            np.log(lambda_min)
            + indices * (np.log(lambda_max) - np.log(lambda_min)) / (embed_dim // 2 - 1)
        )

        # Convert to tensors on device
        x_t = torch.tensor(x, device=self.device).unsqueeze(-1).float()
        lam_t = torch.tensor(lambdas, device=self.device).float()

        # Compute sinusoidal embeddings
        encoded = torch.cat(
            [torch.cos(2 * np.pi * x_t / lam_t), torch.sin(2 * np.pi * x_t / lam_t)],
            dim=-1
        )
        return encoded

    def compute_patch_mean(self, data):
        """
        Compute mean value of each non-overlapping patch.
        Args:
            data (Tensor): 2D tensor of shape (H, W).
        Returns:
            Tensor of shape (H//ps, W//ps).
        """
        h, w = data.shape
        data = data.reshape(
            h // self.patch_size, self.patch_size,
            w // self.patch_size, self.patch_size
        )
        return data.mean(dim=(1, 3))

    def positional_encoding(self, latitudes, longitudes, lambda_min=0.01, lambda_max=720):
        """
        Generate positional encodings based on patch-mean latitude and longitude.
        Args:
            latitudes, longitudes (ndarray): 2D arrays of shape (H, W).
            lambda_min, lambda_max (float): Min and max wavelengths.
        Returns:
            Tensor of shape (H//ps, W//ps, embed_dim)
        """
        assert self.embed_dim % 2 == 0, "embed_dim must be even"

        lat_mean = self.compute_patch_mean(torch.from_numpy(latitudes).float())
        lon_mean = self.compute_patch_mean(torch.from_numpy(longitudes).float())

        lat_enc = self.fourier_encode(lat_mean.numpy(), embed_dim=self.embed_dim//2,
                                      lambda_min=lambda_min, lambda_max=lambda_max)
        lon_enc = self.fourier_encode(lon_mean.numpy(), embed_dim=self.embed_dim//2,
                                      lambda_min=lambda_min, lambda_max=lambda_max)

        return torch.cat([lat_enc, lon_enc], dim=-1)

    def area_encode(self, lats, lons, R=6371):
        """
        Encode each patch’s surface area on the sphere via Fourier features.
        Args:
            lats, lons (1D arrays): Full-resolution latitude and longitude vectors.
            R (float): Earth radius in kilometers.
        Returns:
            Tensor of shape (n_lat_patches, n_lon_patches, embed_dim)
        """
        lat_min = lats[::self.patch_size].reshape(-1, 1)
        lat_max = lats[1::self.patch_size].reshape(-1, 1)
        lon_min = lons[::self.patch_size].reshape(1, -1)
        lon_max = lons[1::self.patch_size].reshape(1, -1)

        phi1 = torch.sin(torch.deg2rad(torch.tensor(lat_min)))
        phi2 = torch.sin(torch.deg2rad(torch.tensor(lat_max)))
        theta1 = torch.deg2rad(torch.tensor(lon_min))
        theta2 = torch.deg2rad(torch.tensor(lon_max))

        areas = R**2 * (phi2 - phi1) * (theta2 - theta1)
        return self.fourier_encode(areas.numpy(), embed_dim=self.embed_dim,
                                   lambda_min=0.25, lambda_max=4*np.pi*R**2)


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer."""

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        b = x.shape[0]
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ResidualMLP(nn.Module):
    """Feed-forward network with residual connection."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class EncoderPerceiver(nn.Module):
    """Perceiver encoder with learnable latent queries."""

    def __init__(self, embed_dim, latent_query_dim, heads=8, dim_head=64):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, latent_query_dim, embed_dim))
        self.cross_attention = CrossAttention(embed_dim, heads, dim_head)
        self.residual_mlp = ResidualMLP(embed_dim, embed_dim * 4)

    def forward(self, x):
        b = x.shape[0]
        queries = repeat(self.latent_queries, '1 n d -> b n d', b=b)
        latents = self.cross_attention(queries, x)
        return self.residual_mlp(latents)


class DecoderPerceiver(nn.Module):
    """Perceiver decoder with fixed level queries."""

    def __init__(self, embed_dim, level_encodings, heads=8, dim_head=64):
        super().__init__()
        self.level_queries = level_encodings
        self.cross_attention = CrossAttention(embed_dim, heads, dim_head)
        self.residual_mlp = ResidualMLP(embed_dim, embed_dim * 4)

    def forward(self, x):
        b = x.shape[0]
        queries = repeat(self.level_queries.unsqueeze(0), '1 n d -> b n d', b=b).to(x.device)
        latents = self.cross_attention(queries, x)
        return self.residual_mlp(latents)


class PerceiverEncoder3D(nn.Module):
    """
    3D Perceiver encoder for multi-level atmospheric and surface patches.
    """

    def __init__(
        self,
        embed_dim,
        img_size,
        patch_size,
        levels,
        num_upper_vars,
        num_surface_vars,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.levels = levels

        self.upper_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=num_upper_vars, embed_dim=embed_dim
        )
        self.surface_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=num_surface_vars, embed_dim=embed_dim
        )

        self.level_linear = nn.Linear(embed_dim, embed_dim)
        self.surface_mlp = ResidualMLP(embed_dim, embed_dim * 4)
        self.perceiver = EncoderPerceiver(embed_dim, latent_query_dim=3, heads=heads, dim_head=dim_head)
        self.fourier_encoder = MultiFourierEncoder(embed_dim, patch_size)

        self.atmos_level_encodings = self.fourier_encoder.fourier_encode(levels)
        self.surface_encodings = nn.Parameter(
            torch.randn(1, img_size[0] // patch_size, img_size[1] // patch_size, embed_dim),
            requires_grad=True
        )

    def forward(self, upper, surface):
        B, L, V, H, W = upper.shape

        # Embed atmospheric levels
        ue = upper.flatten(0, 1)
        ue = self.upper_patch_embed(ue)
        ue = ue.unflatten(0, (B, L))
        ue = ue.unflatten(2, (H // self.patch_size, W // self.patch_size))
        ue = ue.permute(0, 2, 3, 1, 4)

        # Embed surface
        se = self.surface_patch_embed(surface)
        se = se.unflatten(1, (H // self.patch_size, W // self.patch_size))

        # Apply positional encodings
        ue = ue + self.atmos_level_encodings.to(ue.device)
        se = se + self.surface_encodings.to(se.device)

        b, h, w, c, d = ue.shape
        ue_flat = rearrange(ue, "b h w c d -> (b h w) c d")
        kv = self.level_linear(ue_flat)
        latents = self.perceiver(kv)
        atmos_latent = rearrange(latents, "(b h w) n d -> b h w n d", b=b, h=h, w=w)

        se_flat = rearrange(se, "b h w d -> (b h w) d")
        surf_latent = self.surface_mlp(se_flat).unsqueeze(1)
        surf_latent = rearrange(surf_latent, "(b h w) 1 d -> b h w 1 d", b=b, h=h, w=w)

        return torch.cat([atmos_latent, surf_latent], dim=3)


class PerceiverDecoder3D(nn.Module):
    """
    3D Perceiver decoder reconstructing atmospheric levels and surface patches.
    """

    def __init__(
        self,
        embed_dim,
        patch_size,
        levels,
        num_upper_vars,
        num_surface_vars,
        decode_depth,
        heads=8,
        dim_head=64
    ):
        super().__init__()
        self.level_linear = nn.Linear(embed_dim, embed_dim)
        self.patch_size = patch_size
        self.num_upper_vars = num_upper_vars
        self.num_surface_vars = num_surface_vars

        level_enc = MultiFourierEncoder(embed_dim, patch_size).fourier_encode(levels)
        self.perceiver = DecoderPerceiver(embed_dim, level_enc, heads=heads, dim_head=dim_head)

        # Upper patch head
        up_layers = []
        for _ in range(decode_depth):
            up_layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        up_layers.append(nn.Linear(embed_dim, num_upper_vars * patch_size**2))
        self.upper_head = nn.Sequential(*up_layers)

        # Surface patch head
        surf_layers = []
        for _ in range(decode_depth):
            surf_layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        surf_layers.append(nn.Linear(embed_dim, num_surface_vars * patch_size**2))
        self.surface_head = nn.Sequential(*surf_layers)

    def upper_unpatchify(self, x):
        b, h, w, levels, d = x.shape
        x = self.upper_head(x)
        x = x.view(b, h, w, levels, self.num_upper_vars, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        return x.view(b, levels, self.num_upper_vars, h * self.patch_size, w * self.patch_size)

    def surface_unpatchify(self, x):
        b, h, w, _, d = x.shape
        x = self.surface_head(x)
        x = x.view(b, h, w, self.num_surface_vars, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()
        return x.view(b, self.num_surface_vars, h * self.patch_size, w * self.patch_size)

    def forward(self, backbone_output):
        atmos, surf = torch.split(backbone_output, [-1, 1], dim=3)
        b, h, w, c, d = atmos.shape

        atm_flat = rearrange(atmos, "b h w c d -> (b h w) c d")
        kv = self.level_linear(atm_flat.float())
        dec_atm = self.perceiver(kv)
        atmos_dec = rearrange(dec_atm, "(b h w) n d -> b h w n d", b=b, h=h, w=w)

        surf_dec = self.surface_unpatchify(surf.float())
        up_dec = self.upper_unpatchify(atmos_dec)

        return up_dec, surf_dec
