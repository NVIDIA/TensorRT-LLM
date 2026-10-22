# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dense projection in FP8: the bf16 activation quantized to e4m3 with one dynamic
per-tensor scale, the weight stored in e4m3 with one per-tensor scale, cuBLASLt's FP8 GEMM
with both scales in its epilogue, bf16 out.

Tier 5 (Triton written for this target) for the activation quantization: two launches --
per-block partial maxima of `|x|`, then the quantization, each of whose programs reduces the
partials itself (no memset, no host sync) -- and `torch._scaled_mm` (cuBLASLt, tensor-wise
scales) for the GEMM. A producer that already wrote the partial maxima of what it stored
(`amax_partials`, filled by atomic maxima) hands them over and the first launch is skipped.
Tensor-wise scales keep cuBLASLt on its fast FP8 kernels: on GB300 the checkpoint's GDN input
projection runs ~1.9x bf16 at 16K rows and ~1.7x at 1-64 rows, where
row-wise activation scales fall back to kernels only ~1.4x bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

E4M3_MAX = 448.0
#: Floor of the activation maximum: an all-zero activation quantizes to zeros at this scale.
AMAX_FLOOR = 1e-12
#: Most programs of either launch; rows are grid-strided above it, and the quantization
#: reduces at most this many partial maxima.
_MAX_PROGRAMS = 1024
_ROWS = 4
#: Slots of a producer-filled partial-maximum buffer (`amax_partials`): a producer program
#: folds its maximum into slot `program % AMAX_PARTS`, spreading the atomics.
AMAX_PARTS = 1024


def amax_partials(device: torch.device) -> torch.Tensor:
    """A zeroed fp32 `[AMAX_PARTS]` buffer for a producer to fold `max(|out|)` into."""
    return torch.zeros(AMAX_PARTS, dtype=torch.float32, device=device)


def fp8_per_tensor_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(codes, scale)` of a bf16 `[N, K]` weight: e4m3 codes `[N, K]` (contiguous) and the
    fp32 `[1]` dequantization scale `amax / 448`."""
    scale = (weight.abs().amax().float().clamp_min(AMAX_FLOOR) / E4M3_MAX).reshape(1)
    codes = (weight.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.contiguous(), scale


@triton.jit
def _amax_kernel(
    x_ptr, part_ptr, M, K, stride_m, PROGRAMS, ROWS: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    acc = tl.zeros([ROWS, BLOCK_K], dtype=tl.float32)
    for r0 in range(pid * ROWS, M, PROGRAMS * ROWS):
        rows = r0 + tl.arange(0, ROWS)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            mask = (rows[:, None] < M) & (cols[None, :] < K)
            v = tl.load(
                x_ptr + rows[:, None].to(tl.int64) * stride_m + cols[None, :], mask=mask, other=0.0
            )
            acc = tl.maximum(acc, tl.abs(v.to(tl.float32)))
    tl.store(part_ptr + pid, tl.max(tl.max(acc, axis=1), axis=0))


@triton.jit
def _quant_kernel(
    x_ptr,
    part_ptr,
    q_ptr,
    scale_ptr,
    M,
    K,
    stride_m,
    PROGRAMS,
    NUM_PARTS,
    amax_floor,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    p = tl.arange(0, BLOCK_P)
    amax = tl.max(tl.load(part_ptr + p, mask=p < NUM_PARTS, other=0.0), axis=0)
    amax = tl.maximum(amax, amax_floor)
    scale = amax / 448.0
    inv = 448.0 / amax
    if pid == 0:
        tl.store(scale_ptr, scale)
    for r0 in range(pid * ROWS, M, PROGRAMS * ROWS):
        rows = r0 + tl.arange(0, ROWS)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            mask = (rows[:, None] < M) & (cols[None, :] < K)
            v = tl.load(
                x_ptr + rows[:, None].to(tl.int64) * stride_m + cols[None, :], mask=mask, other=0.0
            )
            q = tl.minimum(tl.maximum(v.to(tl.float32) * inv, -448.0), 448.0)
            tl.store(
                q_ptr + rows[:, None].to(tl.int64) * K + cols[None, :],
                q.to(tl.float8e4nv),
                mask=mask,
            )


def fp8_quantize_per_tensor(
    x: torch.Tensor, amax_parts: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(codes [M, K] e4m3 contiguous, scale [1] fp32)` of a bf16 `[M, K]` activation with
    unit column stride: scale = max(|x|) / 448 (floored), codes = round(x / scale).
    `amax_parts`, when given, holds partial maxima whose maximum is `max(|x|)`."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(1, dtype=torch.float32, device=x.device)
    programs = max(1, min(triton.cdiv(m, _ROWS), _MAX_PROGRAMS))
    block_k = min(1024, triton.next_power_of_2(k))
    part = amax_parts
    if part is None:
        part = torch.empty(programs, dtype=torch.float32, device=x.device)
        _amax_kernel[(programs,)](
            x, part, m, k, x.stride(0), programs, ROWS=_ROWS, BLOCK_K=block_k, num_warps=4
        )
    _quant_kernel[(programs,)](
        x,
        part,
        q,
        scale,
        m,
        k,
        x.stride(0),
        programs,
        part.shape[0],
        AMAX_FLOOR,
        ROWS=_ROWS,
        BLOCK_K=block_k,
        BLOCK_P=triton.next_power_of_2(part.shape[0]),
        num_warps=4,
    )
    return q, scale


def _ckpt(m: int, k: int, n: int) -> dict:
    return dict(m=m, k=k, n=n)


class Fp8PerTensorMm(OpWrapper):
    """`y = (q(x) * sx) @ (codes * sw)^T` in bf16, accumulated in fp32, where `sx =
    max(|x|) / 448` over the whole activation and `q(x) = round_e4m3(x / sx)`; `weight_t` is
    the `[K, N]` transpose view of the `[N, K]` e4m3 codes `fp8_per_tensor_weight` returns and
    `weight_scale` its `[1]` scale.

    Lossy against the bf16 projection: ~3.7% relative error per output on random operands
    (e4m3 keeps 3 mantissa bits on both operands). The activation scale is shared by every
    row of the call, so a row's rounding depends on the largest element of its batch.

    Fusion boundary: activation quantization + GEMM. The weight conversion
    (`fp8_per_tensor_weight`) is done once at load by the caller. `amax_parts` (fp32, at most
    `AMAX_PARTS` partial maxima whose maximum is `max(|x|)`, as a producer that stored `x`
    folds them) skips the maximum pass; the result is the same bits.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="GDN input projection, one decode row", spec=_ckpt(1, 2560, 16480)),
        Cell(why="GDN / attention output projection, 128 rows", spec=_ckpt(128, 6144, 2560)),
        Cell(
            why="attention input projection, a 16384-row prefill step",
            spec=_ckpt(16384, 2560, 14080),
        ),
    )

    note = """
    Metadata consumed: none. Call sites: the targets' GDN in/out and attention in/o
    projections when the dense FP8 mode is on (`TRTLLM_MODELING_V2_DENSE_FP8`).
    Preconditions: `x` bf16 `[M, K]` with unit column stride, K a multiple of 16; the weight
    `[K, N]` column-major view of contiguous `[N, K]` e4m3 codes with N a multiple of 16
    (cuBLASLt FP8). Three launches (two Triton, one cuBLASLt); CUDA-graph capturable: no host
    sync, the activation scale stays on the device.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        amax_parts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, scale = fp8_quantize_per_tensor(x, amax_parts)
        return torch._scaled_mm(
            q, weight_t, scale_a=scale, scale_b=weight_scale, out_dtype=torch.bfloat16
        )

    def reference(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        amax_parts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fp32 `[M, N]`: the activation quantized as the op does, both operands dequantized,
        an fp32 matmul."""
        scale = x.abs().amax().float().clamp_min(AMAX_FLOOR) / E4M3_MAX
        xq = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
        return xq @ (weight_t.float() * weight_scale.float())

    def is_valid(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        amax_parts: torch.Tensor | None = None,
    ) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
        assert weight_t.dtype == torch.float8_e4m3fn and weight_t.stride(0) == 1
        assert weight_t.shape[0] == x.shape[1], "K mismatch"
        assert x.shape[1] % 16 == 0 and weight_t.shape[1] % 16 == 0, "cuBLASLt FP8 needs K, N % 16"
        assert weight_scale.numel() == 1 and weight_scale.dtype == torch.float32
        assert amax_parts is None or (
            amax_parts.dtype == torch.float32 and 0 < amax_parts.numel() <= AMAX_PARTS
        ), "amax_parts: fp32, at most AMAX_PARTS partial maxima"


fp8_per_tensor_mm = Fp8PerTensorMm()
