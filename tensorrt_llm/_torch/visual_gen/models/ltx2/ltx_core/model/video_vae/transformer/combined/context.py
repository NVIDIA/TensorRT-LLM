"""Combined context residual: project context half of ``context_and_x`` into ``x``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def combined(
    context_and_x: torch.Tensor,
    w_proj: torch.Tensor,
    b_proj: torch.Tensor | None,
) -> torch.Tensor:
    """Split ``context_and_x`` via ``w_proj.shape[1]`` (context channels), add projected ctx.
    Returns the updated ``x`` half (not the full concatenated buffer).
    ``w_proj`` is ``context_proj.weight`` with shape ``(dim, context_channels)``.
    """
    context_channels = w_proj.shape[1]
    latent_context = context_and_x[..., :context_channels]
    x = context_and_x[..., context_channels:]
    return x + F.linear(latent_context, w_proj, b_proj)
