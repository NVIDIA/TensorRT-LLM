# ruff: noqa: ANN001, ANN201, ANN202, N803

"""Row RMSNorm for the DiffVAE in Triton: the in-place det-stage norms, and the norm the stage-5 tail fuses.
ATen's arithmetic (fp32 ``rstd = rsqrt(mean(x^2) + eps)``, ``y = x * rstd * w``, one rounding to the activation
dtype) at full bandwidth; the fp32 sum of squares may round differently (<1 bf16 ulp on a small fraction of rows).
:func:`rms_rstd` is its row reduction, shared by :func:`row_rms_norm_` and
:mod:`~ltx_core.model.video_vae.transformer.stage5_tail`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):  # pragma: no cover - platform without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

_MAX_WIDTH = 1024
_BLOCK_ELEMS = 8192


def row_rms_norm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Contiguous 2-D bf16 CUDA rows of width <= 1024, with a contiguous 1-D floating weight of that width on
    the same device, and Triton."""
    if triton is None or not x.is_cuda or x.dim() != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16:
        return False
    if weight.dim() != 1 or weight.shape[0] != x.shape[1] or weight.device != x.device:
        return False
    if not weight.is_floating_point() or not weight.is_contiguous():
        return False
    return 1 <= x.shape[1] <= _MAX_WIDTH


if triton is not None:

    @triton.jit
    def rms_rstd(x, N: tl.constexpr, eps):
        """``rsqrt(mean(x^2) + eps)`` per row of fp32 ``(ROWS, BLOCK_N)`` rows of width ``N`` (zero-padded past it).
        The caller scales: ``x * rstd[:, None] * w[None, :]``, then rounds.
        """
        return tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)

    @triton.jit
    def _row_rms_norm_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        n_rows,
        eps,
        scale,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROWS: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row0 = pid * ROWS
        rows = row0 + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_N)
        row_ok = rows < n_rows
        col_ok = cols < N
        mask = row_ok[:, None] & col_ok[None, :]
        offs = rows[:, None].to(tl.int64) * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
        y = x * rms_rstd(x, N, eps)[:, None] * w[None, :]
        if HAS_SCALE:
            # The caller's ``q.mul_(scale)`` on the bf16 norm output: round first, then scale, then round.
            y = y.to(tl.bfloat16).to(tl.float32) * scale
        tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)


def row_rms_norm_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, *, scale: float | None = None, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``rms_norm(x, weight, eps)`` (``* scale`` afterwards, as a second bf16 rounding) row by row.
    ``x`` is ``(rows, N)`` contiguous bf16 (see :func:`row_rms_norm_supported`). Written over ``x`` itself,
    or into ``out`` (same shape/dtype/contiguity, must not partially overlap ``x``). Returns the written tensor.
    """
    rows, n = x.shape
    y = x if out is None else out
    if out is not None and (out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous()):
        raise ValueError("out must be a contiguous tensor of x's shape and dtype")
    if rows == 0:
        return y
    block_n = triton.next_power_of_2(n)
    rows_per_cta = max(1, min(64, _BLOCK_ELEMS // block_n))
    grid = (triton.cdiv(rows, rows_per_cta),)
    w = weight if weight.dtype in (torch.bfloat16, torch.float32) else weight.to(torch.float32)
    _row_rms_norm_kernel[grid](
        x,
        y,
        w,
        rows,
        float(eps),
        float(scale) if scale is not None else 1.0,
        N=n,
        BLOCK_N=block_n,
        ROWS=rows_per_cta,
        HAS_SCALE=scale is not None,
        num_warps=4 if block_n <= 256 else 8,
    )
    return y
