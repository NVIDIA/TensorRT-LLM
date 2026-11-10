# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-shaped bf16 projection: one kernel per call, its source picked per (M, N, K).

At decode M a dense bf16 projection streams its weight once, and FlashInfer's low-M CuTe-DSL
kernels (`dense_bf16_gemm_direct.run_direct_dense`, the register-prefetch kernel, and
`dense_bf16_gemm_warp_splitk.run_warp_splitk_dense`, cp.async + mma.sync warp split-K) beat
cuBLAS there. A per-(N, K) table by M bucket picks one and its tactic; every other shape and
every M above the table stays on the trtllm cuBLASLt op.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
from flashinfer.gemm.kernels import dense_bf16_gemm_direct as _fi_direct
from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as _fi_warp

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class _Plan(NamedTuple):
    """`kind` is "direct", "warp" or "cublas"; `tactic` is the kernel's own tactic tuple."""

    kind: str
    tactic: tuple = ()


def _d(block_size: int, outputs_per_block: int) -> tuple:
    """A direct-kernel choice; rows_per_block is the kernel's default for the runtime M."""
    return ("direct", block_size, outputs_per_block)


def _w(output_tile: int, token_tile: int, k_tile: int, stages: int, b_loaders: int) -> tuple:
    """A warp split-K choice."""
    return ("warp", output_tile, token_tile, k_tile, stages, b_loaders)


#: (N, K) -> ((M upper bound, choice), ...) in ascending M; other shapes and larger M run on
#: cuBLAS. The decode projections of a tp4/tep4 rank (1..16 rows) and a tp2/tep2 and tp1 rank's
#: attention in / out (3..16 rows), each raced on GB300 over the kernel's `autotune_tactics`.
_TABLE: dict[tuple[int, int], tuple[tuple[int, tuple], ...]] = {
    # Attention input projection [q|gate 6 x 256 | k | v | index q|k 640 | pad 128] = 4352.
    (4352, 2560): ((1, _d(64, 2)), (8, _w(16, 8, 256, 9, 2)), (16, _w(16, 16, 256, 6, 2))),
    # GDN input projection 4120, padded to 4128 (`weights.IN_PROJ_ROW_ALIGN`) so the warp split-K's
    # 16-wide output tile divides it.
    (4128, 2560): (
        (2, _d(64, 2)),
        (4, _w(16, 8, 256, 8, 2)),
        (8, _w(16, 8, 128, 11, 2)),
        (16, _w(16, 16, 256, 6, 2)),
    ),
    # Attention o and GDN out (1536 in), the draft layer's o too.
    (2560, 1536): ((1, _d(192, 2)), (2, _d(64, 2)), (16, _w(16, 8, 128, 11, 1))),
    # MoE router (512 experts), trunk and draft layer.
    (512, 2560): ((4, _d(64, 1)), (16, _w(16, 8, 256, 10, 1))),
    # Shared expert gate|up|gate logit|pad, 1288: the warp kernel cannot tile it.
    (1288, 2560): ((4, _d(64, 2)),),
    # Shared expert down projection.
    (2560, 640): ((4, _w(16, 8, 128, 5, 2)),),
    # Draft layer `fc_embedding` (M rows) and `fc_hidden` (4M rows, one per stream).
    (2560, 2560): ((1, _d(64, 2)), (4, _w(16, 8, 256, 9, 1)), (16, _w(32, 16, 256, 7, 1))),
    # Draft layer q|gate.
    (3072, 2560): ((1, _d(64, 2)), (4, _w(16, 8, 256, 9, 1))),
    # Draft layer k and v.
    (256, 2560): ((4, _d(64, 1)),),
    # Draft layer index q|k.
    (640, 2560): ((4, _d(64, 1)),),
    # tp2/tep2 attention input projection (7424). Its GDN input projection (8240) is not a whole
    # number of 32-wide tiles and stays on cuBLAS.
    (7424, 2560): ((8, _w(32, 8, 256, 5, 2)), (16, _w(32, 16, 256, 4, 2))),
    # tp2/tep2 attention o and GDN out (3072 in).
    (2560, 3072): ((8, _w(32, 8, 256, 11, 2)), (16, _w(32, 16, 256, 9, 2))),
    # tp1 attention input projection (14080); cuBLAS catches up from 12 rows, and GDN in (16480)
    # stays on cuBLAS.
    (14080, 2560): ((8, _w(32, 8, 256, 5, 2)),),
    # tp1 attention o and GDN out (6144 in).
    (2560, 6144): ((8, _w(32, 8, 256, 11, 2)), (16, _w(32, 16, 256, 9, 2))),
}


@functools.lru_cache(maxsize=None)
def plan(m: int, n: int, k: int) -> _Plan:
    """The kernel and tactic `LowMBf16Mm` launches for an [M, K] x [K, N] product."""
    for bound, choice in _TABLE.get((n, k), ()):
        if m <= bound:
            if choice[0] == "direct":
                t = _fi_direct.DirectTactic(
                    choice[1], choice[2], _fi_direct.default_tactic(m, n, k).rows_per_block
                )
                _fi_direct.validate_tactic(t, m, n, k)
                return _Plan("direct", tuple(choice[1:]) + (t.rows_per_block,))
            _fi_warp.validate_tactic(_fi_warp.WarpSplitKTactic(*choice[1:]), m, n, k)
            return _Plan("warp", tuple(choice[1:]))
    return _Plan("cublas")


def _cell(why: str, m: int, n: int, k: int, kind: str, pdl: bool = True) -> Cell:
    """`kind` is the branch `plan` must pick for this shape; the test holds it to that."""
    return Cell(why=why, spec=dict(m=m, n=n, k=k, kind=kind, pdl=pdl))


class LowMBf16Mm(OpWrapper):
    """`mat_a @ mat_b` in bf16 with fp32 accumulation: one kernel launch per call.

    A drop-in for `catalog/gemm/cublas_mm` without bias or `out_dtype`: `mat_a` dense row-major
    `[M, K]`, `mat_b` dense column-major `[K, N]` (a weight's `.t()`), the result a fresh
    `[M, N]` bf16 tensor. `plan(M, N, K)` picks the FlashInfer direct kernel (M <= 2, and up to 4
    on narrow outputs), its warp split-K kernel (M 2..16 where 16 divides N) or cuBLAS.

    `pdl` launches the FlashInfer kernels with programmatic dependent launch: each streams its
    weight before `griddepcontrol.wait` and reads `mat_a` and writes its output only after it,
    so it is safe after any kernel; it changes nothing on the cuBLAS branch.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per `_TABLE` tactic (PDL on), N=1288, both cuBLAS fallbacks and PDL off for each
    # FlashInfer kernel; `kind` pins the branch (the test's plan pins hold the other M buckets).
    CELLS: tuple[Cell, ...] = (
        _cell("tep4 attn in M=4 (MTP-3 verify, 1 request): warp split-K", 4, 4352, 2560, "warp"),
        _cell("tep4 attn in M=16 (MTP-3, 4 requests): warp 16-token tile", 16, 4352, 2560, "warp"),
        _cell("tep4 attn in M=1 (decode, no MTP): direct", 1, 4352, 2560, "direct"),
        _cell("tep4 attn in M=24: cuBLAS above the table", 24, 4352, 2560, "cublas"),
        _cell("tep4 GDN in (padded) M=8 (MTP-3, 2 requests): warp 128-K", 8, 4128, 2560, "warp"),
        _cell("tep4 GDN in (padded) M=1, PDL off: direct", 1, 4128, 2560, "direct", pdl=False),
        _cell("tep4 GDN in (padded) M=3, PDL off: warp", 3, 4128, 2560, "warp", pdl=False),
        _cell("tep4 o / GDN out M=3 (MTP-2 verify): warp, ragged tile", 3, 2560, 1536, "warp"),
        _cell("tep4 o / GDN out M=1: direct 192-thread", 1, 2560, 1536, "direct"),
        _cell("dep4 GDN in M=32: cuBLAS (not in the table)", 32, 16480, 2560, "cublas"),
        _cell("tep4 router M=4 (MTP-3 verify / draft extend): direct", 4, 512, 2560, "direct"),
        _cell("tep4 router M=16 (MTP-3, 4 requests): warp", 16, 512, 2560, "warp"),
        _cell("tep4 shared gate|up M=4: direct", 4, 1288, 2560, "direct"),
        _cell("tep4 shared down M=4: warp", 4, 2560, 640, "warp"),
        _cell("tep4 draft fc_embedding M=4 (draft extend): warp", 4, 2560, 2560, "warp"),
        _cell("tep4 draft fc_hidden M=16 (4 rows x 4 streams): warp 32x16", 16, 2560, 2560, "warp"),
        _cell("tep2 attn in M=3 (MTP-2 verify): warp 32-wide", 3, 7424, 2560, "warp"),
        _cell("tep2 attn in M=16 (4 requests x 4 rows): warp 16-token", 16, 7424, 2560, "warp"),
        _cell("tep2 o / GDN out M=12: warp 16-token", 12, 2560, 3072, "warp"),
        _cell("tp1 o / GDN out M=1: warp", 1, 2560, 6144, "warp"),
    )

    note = """
    Preconditions: bf16, `mat_a` dense row-major, `mat_b` dense column-major, both 32-byte
    aligned (the FlashInfer kernels refuse padded operands; cuBLAS reads them as dense, see
    `is_valid`). The FlashInfer kernels are JIT-compiled per (N, K, tactic, pdl; direct also per
    rows_per_block) on the first call, which must come before CUDA-graph capture, and cost tens
    of microseconds of host time a call: route only captured decode calls here.

    Certified on GB300 within the test's 4 / 2 bf16-ulp band at the cells: M 1, 3, 4, 8, 12, 16
    on the FlashInfer kernels, 24 and 32 on cuBLAS. Not certified: M above 16 on FlashInfer.
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> torch.Tensor:
        m, k = mat_a.shape
        n = mat_b.shape[1]
        p = plan(m, n, k)
        if p.kind == "cublas":
            return torch.ops.trtllm.cublas_mm(mat_a, mat_b, None, None, 0, None)
        out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
        if p.kind == "direct":
            return _fi_direct.run_direct_dense(
                mat_a, mat_b, out, pdl, _fi_direct.DirectTactic(*p.tactic)
            )
        return _fi_warp.run_warp_splitk_dense(
            mat_a, mat_b, out, pdl, _fi_warp.WarpSplitKTactic(*p.tactic)
        )

    def reference(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> torch.Tensor:
        """fp32-accumulated, TF32 off; `pdl` is a launch attribute and is ignored."""
        with true_fp32_matmul():
            return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)

    def is_valid(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> None:
        # The cuBLAS branch reads a row-padded mat_a or column-padded mat_b as dense: silently a
        # different product. The FlashInfer branches raise on those.
        assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16, "bf16 operands only"
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )


low_m_bf16_mm = LowMBf16Mm()
