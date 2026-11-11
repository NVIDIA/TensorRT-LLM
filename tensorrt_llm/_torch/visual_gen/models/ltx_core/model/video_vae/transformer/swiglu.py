# ruff: noqa: ANN001, N803, ANN202, PLR0913

"""SwiGLU shell, tiled kernel, det plain residual, and shared tile/fuse helpers.
Pathway residuals:
  - Combined → ``combined.mlp.residual_mlp``
  - Chunked → ``chunked.mlp.residual_modulating_mlp``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import DimensionInterval, split_by_count, split_by_size

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

# Match visual_fidelity ``DEFAULT_CHUNK_TOKENS`` — hard token cap for peak VRAM.
DEFAULT_SWIGLU_TILE_SIZE: Final[int] = 16_384
DEFAULT_SWIGLU_TILES: Final[int] = 4


@dataclass(frozen=True)
class SwiGLUTileSpec:
    """Token tiling: provide exactly one of ``num_tiles`` or ``tile_size``."""

    num_tiles: int | None = None
    tile_size: int | None = None

    def __post_init__(self) -> None:
        if (self.num_tiles is None) == (self.tile_size is None):
            raise ValueError("Provide exactly one of num_tiles or tile_size")
        if self.num_tiles is not None and self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {self.tile_size}")

    @classmethod
    def by_count(cls, num_tiles: int = DEFAULT_SWIGLU_TILES) -> SwiGLUTileSpec:
        return cls(num_tiles=num_tiles)

    @classmethod
    def by_size(cls, tile_size: int = DEFAULT_SWIGLU_TILE_SIZE) -> SwiGLUTileSpec:
        return cls(tile_size=tile_size)


DEFAULT_SWIGLU_TILE_SPEC = SwiGLUTileSpec(tile_size=DEFAULT_SWIGLU_TILE_SIZE)


def triton_swiglu_available() -> bool:
    """True when a CUDA-capable Triton install can run the fused up-mul kernel."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _cuda_tensors_for_triton(*tensors: torch.Tensor) -> bool:
    """True when every tensor is CUDA and all share one device (Triton launch context)."""
    if not tensors or not tensors[0].is_cuda:
        return False
    device = tensors[0].device
    return all(t.is_cuda and t.device == device for t in tensors)


def _resolve_use_triton(use_triton: bool, *tensors: torch.Tensor) -> bool:
    """Enable Triton only when every tensor is on the same CUDA device; reject mixed devices."""
    if not use_triton:
        return False
    if _cuda_tensors_for_triton(*tensors):
        return True
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise ValueError(
            f"SwiGLU requires all tensors on one device for Triton or torch.mm; got {sorted(str(d) for d in devices)}"
        )
    return False


def token_intervals(n_tok: int, tile: SwiGLUTileSpec) -> list[DimensionInterval]:
    """Split ``n_tok`` tokens per ``tile`` (count or size). Overlap is always 0."""
    if n_tok < 0:
        raise ValueError(f"n_tok must be >= 0, got {n_tok}")
    if n_tok == 0:
        return []
    if tile.num_tiles is not None:
        tiles = min(tile.num_tiles, n_tok)
        if tiles <= 1:
            return [DimensionInterval(start=0, end=n_tok, left_ramp=0, right_ramp=0)]
        return list(split_by_count(num_tiles=tiles, overlap=0)(n_tok).intervals)
    assert tile.tile_size is not None
    if n_tok <= tile.tile_size:
        return [DimensionInterval(start=0, end=n_tok, left_ramp=0, right_ramp=0)]
    return list(split_by_size(tile.tile_size, overlap=0)(n_tok).intervals)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_up_mul_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_gm,
        stride_gn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``g = g * (x @ w.T)`` with ``w`` shaped ``(N, K)`` (nn.Linear layout)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)

        g = tl.load(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        out = g * acc.to(g.dtype)
        tl.store(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            out,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fused_gate_up_swiglu_kernel(
        x_ptr,
        w_gate_ptr,
        w_up_ptr,
        out_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_gate_n,
        stride_gate_k,
        stride_up_n,
        stride_up_k,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One x load → gate+up GEMMs → SiLU → gated product into ``out`` (BF16).
        Rounding matches the unfused BF16 path: SiLU is rounded to BF16 before the
        product, then ``(silu_f32 * up_f32).to(bf16)`` for the store.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                w_gate_ptr + offs_n[None, :] * stride_gate_n + offs_k[:, None] * stride_gate_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            up_w = tl.load(
                w_up_ptr + offs_n[None, :] * stride_up_n + offs_k[:, None] * stride_up_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_w)
            up_acc += tl.dot(x, up_w)

        silu_bf16 = (gate_acc * tl.sigmoid(gate_acc)).to(tl.bfloat16)
        product = (silu_bf16.to(tl.float32) * up_acc).to(tl.bfloat16)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            product,
            mask=mask_m[:, None] & mask_n[None, :],
        )


@dataclass(frozen=True)
class _GateUpLaunch:
    """Launch config of the fused gate+up kernel.
    Every config accumulates each output in the same K order, so all are bit-identical.
    """

    blocks: tuple[int, int, int]  # (BLOCK_M, BLOCK_N, BLOCK_K)
    num_warps: int
    num_stages: int


def _gate_up_launch(device: torch.device) -> _GateUpLaunch:
    """Fused gate+up kernel launch config for ``device``'s architecture."""
    major, _ = torch.cuda.get_device_capability(device)
    if major == 10:
        # Datacenter Blackwell. Swept at the DiffVAE det-stage MLP shapes (dim 512..2048,
        # 16k-row tiles) on sm_100.
        return _GateUpLaunch(blocks=(256, 128, 64), num_warps=8, num_stages=3)
    # Triton's default warps and stages.
    return _GateUpLaunch(blocks=(64, 64, 32), num_warps=4, num_stages=3)


_DUAL_GATE_UP_DIM_MIN: Final[int] = 256
_DUAL_GATE_UP_DIM_MAX: Final[int] = 2048


def dual_gate_up_triton_eligible(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> bool:
    """True when the fused gate+up SwiGLU Triton kernel can run.
    Requires BF16 activations/weights on the same CUDA device, ``hidden == 4 * dim``,
    and ``dim`` in ``[256, 2048]`` (DiffVAE MLP shapes).
    """
    if not triton_swiglu_available() or not _cuda_tensors_for_triton(x, w_gate, w_up):
        return False
    if x.dtype != torch.bfloat16 or w_gate.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        return False
    dim = x.shape[-1]
    hidden = w_gate.shape[0]
    if w_up.shape[0] != hidden or w_gate.shape[1] != dim or w_up.shape[1] != dim:
        return False
    if hidden != 4 * dim:
        return False
    return _DUAL_GATE_UP_DIM_MIN <= dim <= _DUAL_GATE_UP_DIM_MAX


def _fused_gate_up_swiglu_triton(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out = silu(x @ W_gateᵀ) * (x @ W_upᵀ)`` with one x-tile load (BF16)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not _cuda_tensors_for_triton(x, w_gate, w_up, out):
        raise ValueError(
            "fused gate+up SwiGLU Triton requires CUDA tensors on one device; "
            f"got x={x.device}, w_gate={w_gate.device}, w_up={w_up.device}, out={out.device}"
        )
    m, k = x.shape
    n = w_gate.shape[0]
    launch = _gate_up_launch(x.device)
    block_m, block_n, block_k = launch.blocks
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _fused_gate_up_swiglu_kernel[grid](
            x,
            w_gate,
            w_up,
            out,
            m,
            k,
            n,
            x.stride(0),
            x.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=launch.num_warps,
            num_stages=launch.num_stages,
        )


def _fused_up_mul_triton(x: torch.Tensor, w_up: torch.Tensor, silu_gate: torch.Tensor) -> None:
    """Inplace: ``silu_gate *= (x @ w_up.T)``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not _cuda_tensors_for_triton(x, w_up, silu_gate):
        raise ValueError(
            "fused up-mul SwiGLU Triton requires CUDA tensors on one device; "
            f"got x={x.device}, w_up={w_up.device}, silu_gate={silu_gate.device}"
        )
    m, k = x.shape
    n = w_up.shape[0]
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _fused_up_mul_kernel[grid](
            x,
            w_up,
            silu_gate,
            m,
            k,
            n,
            x.stride(0),
            x.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            silu_gate.stride(0),
            silu_gate.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )


def _fused_up_mul_torch(x: torch.Tensor, w_up: torch.Tensor, silu_gate: torch.Tensor) -> None:
    """PyTorch fallback: temporary ``up`` chunk, then inplace mul into ``silu_gate``."""
    up = F.linear(x, w_up)
    silu_gate.mul_(up)


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Reference: ``w_down(silu(x@W_gateᵀ) * (x@W_upᵀ))`` (full materialization)."""
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def _tile_from_op_args(tile_size: int, num_tiles: int) -> SwiGLUTileSpec:
    """Decode custom_op int knobs: exactly one of ``tile_size`` / ``num_tiles`` is > 0."""
    if (tile_size > 0) == (num_tiles > 0):
        raise ValueError(f"need exactly one of tile_size>0 or num_tiles>0, got {tile_size=}, {num_tiles=}")
    if tile_size > 0:
        return SwiGLUTileSpec(tile_size=tile_size)
    return SwiGLUTileSpec(num_tiles=num_tiles)


def _swiglu_chunked_impl(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    intervals: list[DimensionInterval],
    *,
    use_triton: bool,
) -> torch.Tensor:
    """Chunked SwiGLU: one reusable ``(chunk, hidden)`` workspace (runs via custom_op)."""
    if use_triton and not triton_swiglu_available():
        raise RuntimeError("use_triton=True but Triton/CUDA is unavailable")

    # RMSNorm under bf16 autocast can leave activations in float32 while Linear weights
    # stay bf16. Inside this custom_op autocast does not cast for ``torch.mm``.
    if x.dtype != w_gate.dtype:
        x = x.to(dtype=w_gate.dtype)

    leading = x.shape[:-1]
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim).contiguous()
    n_tok = x_flat.shape[0]
    hidden = w_gate.shape[0]
    if w_up.shape[0] != hidden or w_gate.shape[1] != dim or w_up.shape[1] != dim:
        raise ValueError(
            f"weight shapes incompatible with x[..., {dim}]: gate={tuple(w_gate.shape)} up={tuple(w_up.shape)}"
        )
    if w_down.shape[1] != hidden or w_down.shape[0] != dim:
        raise ValueError(f"w_down {tuple(w_down.shape)} incompatible with hidden={hidden} dim={dim}")
    if n_tok == 0:
        return x

    # Triton needs every pointer on the launch device; all-CPU falls back to torch.mm.
    use_triton = _resolve_use_triton(bool(use_triton), x_flat, w_gate, w_up, w_down)

    out_flat = torch.empty((n_tok, dim), device=x.device, dtype=x.dtype)
    max_chunk = max((iv.end - iv.start for iv in intervals), default=0)
    workspace = torch.empty((max_chunk, hidden), device=x.device, dtype=x.dtype)
    use_dual = use_triton and dual_gate_up_triton_eligible(x_flat, w_gate, w_up)
    w_gate_c = w_gate.contiguous() if use_dual else w_gate
    w_up_c = w_up.contiguous() if use_triton else w_up

    for iv in intervals:
        start, end = iv.start, iv.end
        if start >= end:
            continue
        xc = x_flat[start:end]
        ws = workspace[: end - start]
        if use_dual:
            _fused_gate_up_swiglu_triton(xc, w_gate_c, w_up_c, ws)
        else:
            torch.mm(xc, w_gate.t(), out=ws)
            F.silu(ws, inplace=True)
            if use_triton:
                _fused_up_mul_triton(xc, w_up_c, ws)
            else:
                _fused_up_mul_torch(xc, w_up, ws)
        torch.mm(ws, w_down.t(), out=out_flat[start:end])

    return out_flat.view(*leading, dim)


@torch.library.custom_op("ltx_core::swiglu_tiled", mutates_args=())
def _swiglu_tiled_op(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> torch.Tensor:
    """Opaque op: memory-efficient chunked SwiGLU (compile-friendly; no Dynamo unroll)."""
    tile = _tile_from_op_args(tile_size, num_tiles)
    n_tok = x.reshape(-1, x.shape[-1]).shape[0]
    return _swiglu_chunked_impl(x, w_gate, w_up, w_down, token_intervals(n_tok, tile), use_triton=use_triton)


@_swiglu_tiled_op.register_fake
def _swiglu_tiled_fake(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> torch.Tensor:
    del w_gate, w_up, w_down, tile_size, num_tiles, use_triton
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def swiglu_tiled(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile: SwiGLUTileSpec,
    *,
    use_triton: bool | None = None,
) -> torch.Tensor:
    """Memory-efficient SwiGLU on the full activation; tiling is internal (custom_op)."""
    tile_size = int(tile.tile_size) if tile.tile_size is not None else 0
    num_tiles = int(tile.num_tiles) if tile.num_tiles is not None else 0
    if use_triton is None:
        use_triton = triton_swiglu_available() and x.is_cuda
    return _swiglu_tiled_op(x, w_gate, w_up, w_down, tile_size, num_tiles, bool(use_triton))


def plain_mlp(x: torch.Tensor, mlp: nn.Module, norm: nn.Module, tile: SwiGLUTileSpec) -> torch.Tensor:
    """Det ``NABlock``: ``x + swiglu_tiled(norm(x))`` (no AdaLN); ``norm`` is the block's
    :class:`~ltx_core.model.video_vae.transformer.det_rms_norm.DetRMSNorm`."""
    y = norm(x)
    if y.numel() == 0:
        return x
    return x + swiglu_tiled(y, mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight, tile)


def plain_mlp_accumulate(x: torch.Tensor, mlp: nn.Module, norm: nn.Module, tile: SwiGLUTileSpec) -> torch.Tensor:
    """:func:`plain_mlp` with the residual added into the MLP's own fresh output: same values (addition
    commutes exactly), one volume fewer live."""
    y = norm(x)
    if y.numel() == 0:
        return x
    return swiglu_tiled(y, mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight, tile).add_(x)


class SwiGLU(nn.Module):
    """Gated MLP weights: ``w_down(silu(w_gate(x)) * w_up(x))``."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        tile: SwiGLUTileSpec = DEFAULT_SWIGLU_TILE_SPEC,
    ) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.tile = tile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled SwiGLU on ``x`` (tests / direct calls). Block residuals use pathway fns."""
        if x.numel() == 0:
            return x
        return swiglu_tiled(
            x,
            self.w_gate.weight,
            self.w_up.weight,
            self.w_down.weight,
            self.tile,
        )


def configure_swiglu_tile(
    module_root: nn.Module,
    *,
    num_tiles: int | None = None,
    tile_size: int | None = None,
) -> None:
    """Set tile spec on every ``SwiGLU`` under ``module_root``.
    Pass exactly one of ``num_tiles`` or ``tile_size``. Calling with both omitted
    is a no-op (leaves each module's existing ``tile``).
    """
    if num_tiles is None and tile_size is None:
        return
    tile = SwiGLUTileSpec(num_tiles=num_tiles, tile_size=tile_size)
    for module in module_root.modules():
        if isinstance(module, SwiGLU):
            module.tile = tile
