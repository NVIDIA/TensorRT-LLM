from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange spatial dimensions into channels. Divides image into patch_size x patch_size blocks
    and moves pixels from each block into separate channels (space-to-depth).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, divides HxW into 4x4 blocks.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal patching).
    For 5D: (B, C, F, H, W) -> (B, Cx(patch_size_hw^2)x(patch_size_t), F/patch_size_t, H/patch_size_hw, W/patch_size_hw)
    Example: (B, 3, 33, 512, 512) with patch_size_hw=4, patch_size_t=1 -> (B, 48, 33, 128, 128)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def patchify_channels_last(x: torch.Tensor, patch_size_hw: int) -> torch.Tensor:
    """``patchify(x, patch_size_hw).permute(0, 2, 3, 4, 1)`` for a 5D ``x``, materialized contiguous.
    Same element mapping as that pair, in one copy instead of a copy plus the one a
    following ``Linear`` makes of the permuted view -- and a contiguous input lets the
    ``Linear`` fold its bias into the matmul. At ``patch_size_hw == 1`` ``rearrange`` alone
    returns a permuted view of ``x``.
    """
    return rearrange(x, "b c f (h q) (w r) -> b f h w (c r q)", q=patch_size_hw, r=patch_size_hw).contiguous()


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange channels back into spatial dimensions. Inverse of patchify - moves pixels from
    channels back into patch_size x patch_size blocks (depth-to-space).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, expands HxW by 4x.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal expansion).
    For 5D: (B, Cx(patch_size_hw^2)x(patch_size_t), F, H, W) -> (B, C, Fxpatch_size_t, Hxpatch_size_hw, Wxpatch_size_hw)
    Example: (B, 48, 33, 128, 128) with patch_size_hw=4, patch_size_t=1 -> (B, 3, 33, 512, 512)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statics is computed over the entire dataset and stored in model's checkpoint under VAE state_dict.
    Defaults are identity (std=1, mean=0) so models constructed without a checkpoint
    do not inherit allocator garbage / NaNs from ``torch.empty``.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)


def apply_rms_norm_uint32_safe(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """``norm(x)`` via :func:`rms_norm_uint32_safe`."""
    return rms_norm_uint32_safe(x, tuple(norm.normalized_shape), norm.weight, norm.eps)


def rms_norm_uint32_safe(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Last-dim RMSNorm that never feeds CUDA a volume of ``2**32``+ elements.
    PyTorch < 2.13 computes ``row * N`` in 32-bit in the layer_norm / RMSNorm
    kernels (pytorch/pytorch#181600). On 2.13+ this is ``F.rms_norm``.
    """
    torch_mm = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    max_elements = (1 << 32) if torch_mm < (2, 13) else (1 << 62)
    return rms_norm_chunked(x, normalized_shape, weight, eps, max_elements=max_elements)


def rms_norm_chunked(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    max_elements: int,
) -> torch.Tensor:
    """Last-dim RMSNorm with each launch strictly below ``max_elements``.
    Prefers slicing T (dim 1) on ``ndim >= 3``. If one T-plane is already at or
    above the limit, flattens to rows and chunks dim 0. Writes into a preallocated
    output so peak extra memory is one window, not every slice plus ``torch.cat``.
    """
    rms_args = (normalized_shape, weight, eps)
    if x.numel() < max_elements:
        return F.rms_norm(x, *rms_args)
    if x.ndim >= 3:
        t = int(x.shape[1])
        per_frame = x.numel() // max(t, 1)
        if per_frame < max_elements:
            chunk_t = _chunk_extent(t, per_frame, max_elements)
            if chunk_t >= t:
                return F.rms_norm(x, *rms_args)
            return _rms_norm_into(torch.empty_like(x), x, 1, chunk_t, *rms_args)
    if x.ndim < 2:
        return F.rms_norm(x, *rms_args)
    flat = x.reshape(-1, x.shape[-1])
    n = int(flat.shape[0])
    per_row = max(int(flat.shape[-1]), 1)
    chunk = _chunk_extent(n, per_row, max_elements)
    if chunk >= n:
        return F.rms_norm(x, *rms_args)
    out = _rms_norm_into(torch.empty_like(flat), flat, 0, chunk, *rms_args)
    return out if x.ndim == 2 else out.view(x.shape)


def _chunk_extent(length: int, per_item: int, max_elements: int) -> int:
    """Largest slice along ``length`` whose numel stays strictly below ``max_elements``."""
    if length <= 0:
        return 1
    return max(1, (max_elements - 1) // max(per_item, 1))


def _rms_norm_into(out: torch.Tensor, x: torch.Tensor, dim: int, chunk: int, *rms_args: object) -> torch.Tensor:
    """Last-dim RMSNorm along ``dim``, writing each slice into a preallocated ``out``."""
    n = int(x.shape[dim])
    idx: list[slice | int] = [slice(None)] * x.ndim
    for i in range(0, n, chunk):
        idx[dim] = slice(i, min(i + chunk, n))
        key = tuple(idx)
        out[key] = F.rms_norm(x[key], *rms_args)
    return out
