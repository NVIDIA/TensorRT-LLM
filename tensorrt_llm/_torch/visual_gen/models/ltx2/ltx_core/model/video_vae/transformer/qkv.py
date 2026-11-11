"""Split Q/K/V projections for Neighborhood Attention."""

from __future__ import annotations

import torch
from torch import nn


class QKVProjections(nn.Module):
    """Three Q/K/V linears (separate GEMMs).
    A fused ``nn.Linear(dim, 3*dim)`` + ``chunk`` looks cheaper, but on
    ``(B,T,H,W,3*C)`` the chunks are non-contiguous (q|k|v interleaved per
    spatial site). Making them contiguous while the fused buffer is alive
    peaks at ``6*C`` channels (~+2 GB at stage-5) vs ``3*C`` from three
    separate GEMMs -- and, unlike a fused tensor, Q/K/V here are independently
    owned tensors with different lifetimes (e.g. Q is consumed immediately by
    the score matmul while K/V live through the whole neighborhood gather),
    so each can be freed as soon as its last use completes instead of all
    three staying alive as long as the fused parent does. At production
    shape the runtime difference is ~20 ms/step; the memory win dominates.
    Checkpoint files still ship fused ``qkv.weight`` / ``qkv.bias``; those are
    split into ``to_q`` / ``to_k`` / ``to_v`` during load by
    ``DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER`` (``SDOps.with_kv_operation``).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)
