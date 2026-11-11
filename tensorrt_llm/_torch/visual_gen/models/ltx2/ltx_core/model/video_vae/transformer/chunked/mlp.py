"""Chunked pathway MLP: inplace fused ``x += swiglu(modulate(rms_norm(x)))``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import rms_norm_uint32_safe
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.swiglu import (
    SwiGLUTileSpec,
    _fused_gate_up_swiglu_triton,
    _fused_up_mul_torch,
    _fused_up_mul_triton,
    _resolve_use_triton,
    _tile_from_op_args,
    dual_gate_up_triton_eligible,
    token_intervals,
    triton_swiglu_available,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import DimensionInterval


def _channel_affine_bc(scale: torch.Tensor, shift: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = scale.reshape(-1, dim)
    sh = shift.reshape(-1, dim)
    if s.shape[0] != 1 or sh.shape[0] != 1:
        if s.shape[0] != sh.shape[0]:
            raise ValueError(f"scale/shift batch mismatch: {tuple(s.shape)} vs {tuple(sh.shape)}")
        if s.shape[0] != 1:
            raise ValueError(
                f"swiglu residual modulated expects broadcastable channel affine "
                f"(got scale {tuple(scale.shape)}); B>1 not supported"
            )
    return s, sh


def _chunked_residual_modulated_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    intervals: list[DimensionInterval],
    *,
    use_triton: bool,
    eps: float = 1e-6,
) -> None:
    """In-place: ``x += swiglu(modulate(rms_norm(x)))`` with peak O(chunk·hidden)."""
    if use_triton and not triton_swiglu_available():
        raise RuntimeError("use_triton=True but Triton/CUDA is unavailable")

    dim = x.shape[-1]
    if norm_weight.shape != (dim,):
        raise ValueError(f"norm_weight shape {tuple(norm_weight.shape)} != ({dim},)")
    x_flat = x.reshape(-1, dim)
    if not x_flat.is_contiguous():
        raise ValueError("x must be contiguous channels-last so reshape(-1, dim) is a writable view")
    n_tok = x_flat.shape[0]
    hidden = w_gate.shape[0]
    if w_up.shape[0] != hidden or w_gate.shape[1] != dim or w_up.shape[1] != dim:
        raise ValueError(
            f"weight shapes incompatible with x[..., {dim}]: gate={tuple(w_gate.shape)} up={tuple(w_up.shape)}"
        )
    if w_down.shape[1] != hidden or w_down.shape[0] != dim:
        raise ValueError(f"w_down {tuple(w_down.shape)} incompatible with hidden={hidden} dim={dim}")
    if n_tok == 0:
        return

    # Triton needs every pointer on the launch device; all-CPU falls back to torch.mm.
    use_triton = _resolve_use_triton(bool(use_triton), x_flat, w_gate, w_up, w_down)

    s, sh = _channel_affine_bc(scale, shift, dim)
    max_chunk = max((iv.end - iv.start for iv in intervals), default=0)
    workspace = torch.empty((max_chunk, hidden), device=x.device, dtype=x.dtype)
    y_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)
    out_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)
    use_dual = use_triton and dual_gate_up_triton_eligible(x, w_gate, w_up)
    w_gate_c = w_gate.contiguous() if use_dual else w_gate
    w_up_c = w_up.contiguous() if use_triton else w_up

    for iv in intervals:
        start, end = iv.start, iv.end
        if start >= end:
            continue
        n = end - start
        xc = x_flat[start:end]
        y = y_buf[:n]
        y.copy_(rms_norm_uint32_safe(xc, (dim,), norm_weight, eps))
        y.mul_(1.0 + s).add_(sh)

        ws = workspace[:n]
        if use_dual:
            _fused_gate_up_swiglu_triton(y, w_gate_c, w_up_c, ws)
        else:
            torch.mm(y, w_gate.t(), out=ws)
            F.silu(ws, inplace=True)
            if use_triton:
                _fused_up_mul_triton(y, w_up_c, ws)
            else:
                _fused_up_mul_torch(y, w_up, ws)
        torch.mm(ws, w_down.t(), out=out_buf[:n])
        xc.add_(out_buf[:n])


@torch.library.custom_op("ltx_core::swiglu_tiled_residual_modulated", mutates_args=("x",))
def _swiglu_tiled_residual_modulated_op(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> None:
    """Opaque op: chunked ``x += swiglu(modulate(rms_norm(x)))`` (compile-friendly)."""
    tile = _tile_from_op_args(tile_size, num_tiles)
    n_tok = x.reshape(-1, x.shape[-1]).shape[0]
    _chunked_residual_modulated_impl(
        x,
        norm_weight,
        scale,
        shift,
        w_gate,
        w_up,
        w_down,
        token_intervals(n_tok, tile),
        use_triton=use_triton,
    )


@_swiglu_tiled_residual_modulated_op.register_fake
def _swiglu_tiled_residual_modulated_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> None:
    del x, norm_weight, scale, shift, w_gate, w_up, w_down, tile_size, num_tiles, use_triton


def residual_modulating_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tile: SwiGLUTileSpec,
) -> torch.Tensor:
    """Inplace fused ``x += swiglu(modulate(rms_norm(x)))`` (Chunked path)."""
    if x.numel() == 0:
        return x
    if not x.is_contiguous():
        x = x.contiguous()

    tile_size = int(tile.tile_size) if tile.tile_size is not None else 0
    num_tiles = int(tile.num_tiles) if tile.num_tiles is not None else 0
    use_triton = triton_swiglu_available() and x.is_cuda

    _swiglu_tiled_residual_modulated_op(
        x,
        norm.weight,
        scale,
        shift,
        mlp.w_gate.weight,
        mlp.w_up.weight,
        mlp.w_down.weight,
        tile_size,
        num_tiles,
        bool(use_triton),
    )
    return x
