# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 x MXFP8 dense GEMM in nn.Linear layout: act @ weight.T with 1x32 UE8M0 block scales.

Wraps the FlashInfer Python API `flashinfer.mm_mxfp8`, not a `torch.ops.trtllm`
op: its availability is a property of the environment (`flashinfer` importable),
not of the TensorRT-LLM build. `backend` selects one of the three certified
kernel families (cuDNN / cuBLASLt, CuTe DSL, CUTLASS); exactly one runs per call.
"""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer import mm_mxfp8

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

CERTIFIED_BACKENDS = ("cudnn", "cute-dsl", "cutlass")

_BLOCK = 32  # MXFP8: one UE8M0 scale per 32 contiguous elements along K

#: The DeepSeek-V4.1-Flash dense FP8 linears per rank, (N, K) in nn.Linear [out, in]
#: order. wq_b is [64 / tp * 512, 1280] and wo_b [5120, 8 / tp * 1024] (tp 1 under
#: attention DP).
_TARGET_SHAPES = (
    (1792, 5120, "wq_a||wkv"),
    (8192, 1280, "wq_b, attention TP4"),
    (32768, 1280, "wq_b, attention DP"),
    (5120, 2048, "wo_b, attention TP4"),
    (5120, 8192, "wo_b, attention DP"),
    (4608, 5120, "shared w1||w3"),
    (5120, 2304, "shared w2"),
    (4096, 1280, "indexer wq_b, wq_b TP8"),
    (25600, 6144, "engram wkv"),
    (16384, 1280, "wq_b, attention TP2"),
    (5120, 4096, "wo_b, attention TP2"),
    (5120, 1024, "wo_b, attention TP8"),
    (5120, 15360, "DSpark main_proj"),
)
#: decode (1..48 incl. DSpark verify rows 6 / 48), a mid batch, a prefill chunk.
_TARGET_ROWS = (1, 6, 8, 32, 48, 192, 4096)
#: The largest prefill chunk, at the widest and the deepest K.
_PREFILL_8192 = ((25600, 6144), (5120, 15360), (1792, 5120))


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _unswizzle(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The `[rows, cols]` UE8M0 bytes of a flat 128x4-swizzled scale buffer."""
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    idx = (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * _pad_up(cols, 4))
    )
    return sf.view(torch.uint8).reshape(-1)[idx]


def _dequant(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact fp32 value of an MXFP8 operand: e4m3 * 2^(byte - 127), one byte per 32 along K."""
    rows, k = q.shape
    scale = torch.exp2(_unswizzle(sf, rows, k // _BLOCK).float() - 127.0)
    return q.float() * scale.repeat_interleave(_BLOCK, dim=-1)


def _cell(why: str, n: int, k: int, m: int, backend: str, **overrides) -> Cell:
    spec = dict(
        N=n, K=k, M=m, backend=backend, out_dtype=torch.bfloat16, act_sf=(116, 124), out=False
    )
    spec.update(overrides)
    return Cell(why=why, spec=spec)


class Mxfp8Mxfp8Gemm(OpWrapper):
    """`out = a @ b.T` over two MXFP8 operands, fp32 accumulation, rounded once to `out_dtype`.

    `a[m, k] = e4m3(act[m, k]) * 2^(act_sf(m, k // 32) - 127)` and likewise `b`
    from `weight` / `weight_sf`; both scale buffers are 1-D uint8 in the 128x4
    swizzled order `torch.ops.trtllm.mxfp8_quantize(x, True)` emits and
    `torch.ops.trtllm.block_scale_interleave` produces. `weight` is the
    row-major `[N, K]` nn.Linear weight; FlashInfer gets its `.t()` view (the
    `[K, N]` column-major operand it expects), which launches nothing.

    Fusion boundary: both operands' block-scale dequantization, the fp32
    accumulation and the output cast are inside the call. The caller owns
    producing `act` / `act_sf` (an activation quantizer whose two outputs this op
    takes unchanged), the load-time weight-scale preparation, and any bias,
    activation or cross-rank reduction; there is no alpha, global scale or bias.
    Returns the `[M, N]` result: `out` itself when given, else a fresh tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            _cell(f"{be} M={m} {n}x{k}: {name}", n, k, m, be)
            for n, k, name in _TARGET_SHAPES
            for m in _TARGET_ROWS
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(f"{be} M=8192 {n}x{k}: the largest prefill chunk", n, k, 8192, be)
            for n, k in _PREFILL_8192
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(f"{be} M={m} 512x1024: act scale rows pad to 128", 512, 1024, m, be)
            for m in (1, 3, 7, 9, 127, 128, 129, 1000)
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(f"{be} M=8 160x1024: weight scale rows pad 160->256", 160, 1024, 8, be)
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(f"{be} M={m} 5120x576: K/32 = 18 scale cols pad to 20", 5120, 576, m, be)
            for m in (1, 8, 300)
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(
                f"{be} fp16 out M=48 1792x5120: act scales 2^-6 lower",
                1792,
                5120,
                48,
                be,
                out_dtype=torch.float16,
                act_sf=(110, 118),
            )
            for be in CERTIFIED_BACKENDS
        )
        + tuple(
            _cell(f"{be} out= M=48 1792x5120: written and returned", 1792, 5120, 48, be, out=True)
            for be in CERTIFIED_BACKENDS
        )
    )

    note = """
    Metadata consumed: FlashInfer's process-global autotuner cache, separate
    from TensorRT-LLM's autotuner. Nothing must be prepared: outside a
    `flashinfer.autotune()` context every call takes its backend's fallback
    tactic, the path certified here. `cute-dsl` JIT-compiles each variant on
    first use (about 0.5 s; a swap-AB kernel with N tile 8/16/32 at M <= 32,
    128x128 above); `cudnn` builds one plan per (N, K, M bucket) on the first
    call in the bucket (about 20 ms) and launches a kernel that still changes
    with M inside a bucket. If anything in the process tunes `mm_mxfp8` under
    `flashinfer.autotune()`, later calls take the tuned tactic: not certified.
    TensorRT-LLM's engine warmup does not tune it for a modeling_v2 target.

    Checkpoint weights store one UE8M0 byte per 32x32 tile; this op wants one
    byte per row and K block: `sf_rows = scale.view(torch.uint8)
    .repeat_interleave(32, dim=0)[:N]`, then `block_scale_interleave(sf_rows)`.
    Tiling instead (`scale.repeat(32, 1)`) is accepted and wrong. Scale buffers
    are exactly `pad_up(rows, 128) * pad_up(K / 32, 4)` uint8 bytes; padding
    bytes are ignored. A row-major scale buffer of the right length is accepted
    and silently wrong on every backend, and no guard can see it.

    Preconditions: `act` and `weight` 2-D float8_e4m3fn, contiguous (see
    `is_valid`); N >= 128 and K >= 128 (ValueError); N % 32 == 0 and K % 32 == 0
    are the certified domain; M >= 1 (M = 0 raises on every backend); `out_dtype`
    bf16 or fp16 (fp32 raises; fp16 overflow is inf); `out` `[M, N]` contiguous.
    Wrong scale lengths, a non-uint8 scale view and a non-e4m3 `act` raise.

    Backends: the target picks `cute-dsl` at M <= 32 and `cudnn` above (measured
    on sm_103 the fastest at 38 of 40 and 40 of 40 cells respectively).
    `cute-dsl` and `cutlass` are bitwise equal to each other at every certified
    cell; `cudnn` is held to the bound only: it differs in the last bits at
    M <= 48 for K >= 5120 (split-K plus a reduce kernel, two launches), and a
    row's bits can depend on the batch's M. Every backend is deterministic.
    Fused projections (`wq_a || wkv`, `w1 || w3`, the drafter's `wkv` x3) are
    within the bound of the separate GEMMs and bitwise equal on `cute-dsl` and
    `cutlass`; on `cudnn` they differ in the last bits in a window of M (at
    K = 5120: 129..640 for `wq_a || wkv`, 49..128 for `w1 || w3`, 193..640 for
    `wkv` x3). Eager host cost at M = 1 is about 187 / 93 / 82 us
    (cudnn / cute-dsl / cutlass); CUDA-graph replay removes it.

    CUDA graphs: the first call a process makes on a backend must be eager
    (capturing it fails); after one eager call capture works at row counts
    never run eagerly, and replay is bit-identical to eager. FlashInfer keeps
    one `mm_mxfp8_workspace` buffer per device; the first `cudnn` call grows it
    to 33,554,688 bytes at a new address, and no measured call moves it after
    that. A larger request (an M, shape or cuDNN not measured) would move it
    under every captured `cudnn` graph.

    Row counts certified at the twelve core linears: every M 1..32 on
    `cute-dsl` and `cudnn`; the bottom, interior and top of every FlashInfer
    bucket from 33 to 32768 and every dumped row count on all three backends;
    every M 33..N_max on `cudnn` on a column sample, and every `cudnn` kernel
    switch at full width. N_max is 32768 at the eleven decoder-layer shapes and
    16384 at the Engram `wkv`. The drafter's `main_proj` and fused `wkv` x3 run
    at every M 1..1280 on all three backends. Captured on the target's backend
    at G and 6G (5G where the drafter has the shape) over the batch sizes
    1..32, 40..128 step 8, 192, 256, and at every M 1..160 for the two drafter
    shapes; `cutlass` captured at M = 8 and 256 only.

    Not certified: M above N_max; on `cudnn` between 33 and N_max, errors
    confined to columns outside the sample except at the full-width values;
    `cute-dsl` and `cutlass` above 32 rows at M other than the bucket and dumped
    values; the drafter shapes above 1280 rows other than 4096 and 8192 at
    `main_proj`; capture at row counts or shapes other than those listed;
    non-finite inputs; a workspace request above 33,554,688 bytes; any tactic
    chosen by `flashinfer.autotune()`.
    """

    def raw_call(
        self,
        act: torch.Tensor,
        act_sf: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        backend: str = "cudnn",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return mm_mxfp8(
            act,
            weight.t(),
            act_sf,
            weight_sf,
            out=out,
            out_dtype=out_dtype,
            use_8x4_sf_layout=False,
            backend=backend,
        )

    def reference(
        self,
        act: torch.Tensor,
        act_sf: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        backend: str = "cudnn",
        out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The exact product of the dequantized operands in true fp32, and the bound `compare` gates on.

        Returns `(ref, bound)`, both fp32 `[M, N]`: `ref = a @ b.T` with `a` and
        `b` dequantized exactly from the e4m3 data and the swizzled scale bytes,
        and `bound = K * 2^-24 * sum_k |a_mk b_nk| + eps(out_dtype) * |ref|`.
        The product runs with TF32 off (`true_fp32_matmul`). `backend` and `out`
        are ignored; nothing is mutated.
        """
        a = _dequant(act, act_sf)
        b = _dequant(weight, weight_sf)
        with true_fp32_matmul():
            ref = a @ b.T
            l1 = a.abs() @ b.abs().T
        bound = act.shape[1] * 2.0**-24 * l1 + torch.finfo(out_dtype).eps * ref.abs()
        return ref, bound

    def is_valid(
        self,
        act: torch.Tensor,
        act_sf: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        backend: str = "cudnn",
        out: Optional[torch.Tensor] = None,
    ) -> None:
        # Every guard is pure metadata, and every guarded violation was observed on
        # this machine to be accepted and computed wrongly without raising: the
        # cute-dsl backend ignores the strides of a non-contiguous act or weight;
        # cute-dsl and cutlass write a non-contiguous `out` as if it were
        # contiguous (landing outside the view); FlashInfer's "trtllm" backend
        # reads the same operands in a shuffled layout they are not in.
        assert backend in CERTIFIED_BACKENDS, (
            f"backend must be one of {CERTIFIED_BACKENDS}; got {backend!r} (the 'trtllm' "
            "backend needs shuffled operands and returns wrong results on these)"
        )
        assert act.is_contiguous() and weight.is_contiguous(), (
            "act [M, K] and weight [N, K] must be contiguous; the cute-dsl backend "
            "ignores strides and returns wrong results"
        )
        assert out is None or out.is_contiguous(), (
            "out must be contiguous; the cute-dsl and cutlass backends write it as "
            "contiguous regardless of its strides"
        )

    def compare(self, out: torch.Tensor, ref: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Every element within the fp32-accumulation bound: `|out - ref| <= bound`.

        Both operands are exact in fp32 and every block scale is a power of two,
        so the kernel's only error sources are the order of its fp32
        accumulation over K, which the tensor-core MMA fixes and the reference
        cannot reproduce (bounded per element by the recursive-summation bound
        `K * 2^-24 * sum_k |a_mk b_nk|`), and one rounding of the fp32 sum to
        the output dtype (taken as a full `eps * |ref|`). torch's default bf16
        `assert_close` does not model the first: outputs that cancel to near
        zero carry an absolute error of that size, and it failed on 63 to 65 of
        80 measured shape and row cells for every candidate GEMM. Measured on
        sm_103 the largest `|err| / bound` over every certified check was
        0.4940, while a row-major weight or activation scale, a tiled 32x32
        expansion and every scale byte off by one land 101x to 5810x outside it.
        A single misplaced scale byte touches one column and can hide under the
        bound at M = 1.
        """
        expected, bound = ref
        assert out.dtype in (torch.bfloat16, torch.float16), out.dtype
        assert out.shape == expected.shape, (out.shape, expected.shape)
        err = (out.float() - expected).abs()
        ratio = (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        assert ratio <= 1.0, f"max |out - ref| / bound = {ratio:.3g} (> 1)"


mxfp8_mxfp8_gemm = Mxfp8Mxfp8Gemm()
