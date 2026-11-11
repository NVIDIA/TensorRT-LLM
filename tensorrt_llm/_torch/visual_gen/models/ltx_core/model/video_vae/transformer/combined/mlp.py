"""Combined pathway MLP: out-of-place AdaLN SwiGLU residual."""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.ops import apply_rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.layers import modulate
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.swiglu import SwiGLUTileSpec, swiglu_tiled


def residual_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tile: SwiGLUTileSpec,
) -> torch.Tensor:
    """Combined*: ``x + swiglu_tiled(modulate(norm(x), scale, shift))``."""
    y = modulate(apply_rms_norm_uint32_safe(norm, x), scale, shift)
    if y.numel() == 0:
        return x
    return x + swiglu_tiled(y, mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight, tile)
