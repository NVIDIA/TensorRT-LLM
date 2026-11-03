# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-shaped bf16 projection: one kernel per call, its source picked per (M, N, K).

Its table covers this target's decode shapes: at a tp4/tep4 rank the GDN input projection,
the attention input projection and the attention / GDN output projection, and on every rank
the replicated MoE router.

At decode M a dense bf16 projection streams its weight once. FlashInfer's Blackwell low-M
CuTe-DSL kernels -- the register-prefetch "direct" kernel and the tcgen05 split-K kernel, the
two `flashinfer.mm_bf16(backend="cute-dsl")` dispatches to and
`tensorrt_llm/_torch/modules/low_m_gemm.py` races -- are faster than cuBLAS at M <= 32 on
most such shapes. Which one, with which tactic, is a per-(N, K) table by M bucket; every
shape the table does not list, and every M above it, stays on the trtllm cuBLASLt op.

FlashInfer API: ``flashinfer.gemm.kernels.dense_bf16_gemm_direct.run_direct_dense`` and
``flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk.run_splitk_dense``.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
from flashinfer.gemm.kernels import dense_bf16_gemm_direct as _fi_direct
from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as _fi_splitk

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class _Plan(NamedTuple):
    """`kind` is "direct", "splitk" or "cublas"; `tactic` is the kernel's own tactic tuple."""

    kind: str
    tactic: tuple = ()


def _d(block_size: int, outputs_per_block: int) -> tuple:
    """A direct-kernel choice; rows_per_block is the kernel's default for the runtime M."""
    return ("direct", block_size, outputs_per_block)


def _s(mma_m: int, mma_n: int, split_k: int, ab_stages: int) -> tuple:
    return ("splitk", mma_m, mma_n, split_k, ab_stages)


#: (N, K) -> ((M upper bound, choice), ...) in ascending M. An M above the last
#: bound, and any (N, K) not listed, runs on cuBLAS. Every row is one of this target's decode
#: shapes. M=3 uses the M=4 choice, and so on.
_TABLE: dict[tuple[int, int], tuple[tuple[int, tuple], ...]] = {
    # GDN in_proj at a tp4/tep4 rank: q|k (4 heads x 128 each), v (12 x 128), z (12 x 128),
    # a, b (12 each) concatenated: 4120 columns.
    (4120, 2560): ((1, _d(64, 1)), (2, _d(64, 2)), (8, _s(64, 8, 2, 10))),
    # Attention input projection at a tp4/tep4 rank: q|gate (6 heads x 256 each), k, v (one
    # KV head x 256 each), the replicated index q|k (5 x 128) and one pad head of 128: 4352.
    (4352, 2560): ((2, _d(64, 2)), (8, _s(64, 8, 2, 10))),
    # Attention o and GDN out at a tp4/tep4 rank (6 q heads x 256 = 12 v heads x 128 = 1536
    # in). The target sends M <= 4 to `bf16_gemv_pdl__decode`; these rows are what this
    # entry runs at the same shape.
    (2560, 1536): ((4, _d(64, 2)), (8, _s(64, 8, 3, 10))),
    # The MoE router (512 experts), replicated on every rank.
    (512, 2560): ((2, _d(64, 2)), (8, _d(64, 1)), (16, _d(32, 1)), (32, _s(64, 8, 4, 10))),
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
            _fi_splitk.validate_tactic(_fi_splitk.SplitKTactic(*choice[1:]), m, n, k)
            return _Plan("splitk", tuple(choice[1:]))
    return _Plan("cublas")


def _cell(why: str, m: int, n: int, k: int, kind: str, pdl: bool = False) -> Cell:
    """`kind` is the branch `plan` must pick for this shape; the test holds it to that."""
    return Cell(why=why, spec=dict(m=m, n=n, k=k, kind=kind, pdl=pdl))


class LowMBf16Mm(OpWrapper):
    """`mat_a @ mat_b` in bf16 with fp32 accumulation: one kernel launch per call.

    A drop-in for `catalog/gemm/cublas_mm` without bias or `out_dtype`: `mat_a`
    is dense row-major `[M, K]`, `mat_b` dense column-major `[K, N]` (a stored
    `[N, K]` weight's `.t()`), the result a fresh row-major `[M, N]` bf16 tensor.
    `plan(M, N, K)` picks the kernel on the host from three sources:

    - `flashinfer.gemm.kernels.dense_bf16_gemm_direct.run_direct_dense` -- SIMT,
      the weight row's whole K prefetched into registers; wins at M <= 2 and on
      narrow N;
    - `flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk.run_splitk_dense` --
      tcgen05, swap-AB, split-K reduced across a cluster in shared memory (no
      extra launch); wins at M 4-32 on the wide shapes;
    - `torch.ops.trtllm.cublas_mm` -- everything else, including every M > 32.

    `pdl` launches the FlashInfer kernels with programmatic dependent launch:
    they wait on the previous grid (`griddepcontrol.wait`) before reading
    `mat_a`, so it is safe after any kernel; it changes nothing on the cuBLAS
    branch.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per branch `plan` takes on this target's rows; `kind` pins it. The test's
    # host-only plan pins hold the table's other M buckets.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "tp4 GDN in_proj M=4 (MTP3 verify, 1 request): split-K, pdl on",
            4,
            4120,
            2560,
            "splitk",
            pdl=True,
        ),
        _cell("tp4 GDN in_proj M=8 (2 requests): split-K, pdl off", 8, 4120, 2560, "splitk"),
        _cell("tp4 GDN in_proj M=2 (MTP1 verify): direct, 2-row tile", 2, 4120, 2560, "direct"),
        _cell("tp4 GDN in_proj M=1: direct", 1, 4120, 2560, "direct", pdl=True),
        _cell("tp4 GDN in_proj M=16: cuBLAS branch above the table", 16, 4120, 2560, "cublas"),
        _cell(
            "tp4 attention in M=4 (MTP3 verify, 1 request): split-K, pdl on",
            4,
            4352,
            2560,
            "splitk",
            pdl=True,
        ),
        _cell("tp4 attention in M=1: direct, 2-output tile, pdl on", 1, 4352, 2560, "direct", True),
        _cell("tp4 attention in M=16: cuBLAS branch above the table", 16, 4352, 2560, "cublas"),
        _cell(
            "tp4 o / GDN out M=8 (2 requests): split-K 3-way, pdl on", 8, 2560, 1536, "splitk", True
        ),
        _cell("tp4 o / GDN out M=4: direct, 2-output tile", 4, 2560, 1536, "direct"),
        _cell("router M=1: one decode token, direct 2-row tile", 1, 512, 2560, "direct", True),
        _cell("router M=8: an MTP-7 verify step, direct", 8, 512, 2560, "direct", True),
        _cell("router M=16: direct 32-thread tile", 16, 512, 2560, "direct", True),
        _cell("router M=32: split-K (M boundary)", 32, 512, 2560, "splitk", True),
        _cell("router M=64: cuBLAS branch above M=32", 64, 512, 2560, "cublas", True),
    )

    note = """
    Metadata consumed: none. Preconditions: bf16 operands, `mat_a` dense
    row-major, `mat_b` dense column-major, both 32-byte aligned (a freshly
    allocated activation and a `.t()` weight view are); the FlashInfer kernels
    raise ValueError on a layout or alignment they cannot read, the cuBLAS
    branch does not (see `is_valid`).

    Compilation: the FlashInfer kernels are CuTe-DSL, JIT-compiled in-process
    on the first call per (direct: M, N, K, tactic, pdl; split-K: tactic, pdl)
    and cached for the process. That first call must happen before CUDA-graph
    capture -- the engine's warmup forward at each captured batch size does it.

    Host cost: the FlashInfer branches spend tens of microseconds of Python per
    call (argument validation, DLPack conversion), which a CUDA graph removes
    and an eager step does not. Route only graph-captured decode calls here.

    Numerics: every branch accumulates in fp32 and rounds once; the branches
    differ from each other only in summation order.

    With PDL behind a producer the FlashInfer kernels stream the weight before the wait.

    Hard limits
      M    FlashInfer branches only at M <= 32 (both kernels' policy)  boundary 32  driven at
           the router's 1, 8, 16, 32 (cells) and 64 (cuBLAS cell)
      N*K  largest FlashInfer-branch weight 4352*2560 = 1.1e7 elements  boundary 2^31  not probed
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False) -> torch.Tensor:
        m, k = mat_a.shape
        n = mat_b.shape[1]
        p = plan(m, n, k)
        if p.kind == "direct":
            out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
            return _fi_direct.run_direct_dense(
                mat_a, mat_b, out, pdl, _fi_direct.DirectTactic(*p.tactic)
            )
        if p.kind == "splitk":
            out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
            return _fi_splitk.run_splitk_dense(
                mat_a, mat_b, None, out, pdl, _fi_splitk.SplitKTactic(*p.tactic)
            )
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, None, None, 0, None)

    def reference(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off; `pdl` is a launch attribute and is ignored."""
        with true_fp32_matmul():
            return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)

    def is_valid(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = False) -> None:
        # The cuBLAS branch ignores both leading dimensions: a row-padded
        # mat_a or a column-padded mat_b (a slice of a wider tensor) is read as
        # if dense -- silently a different product. The FlashInfer branches
        # raise on those. A dense row-major mat_b is outside the contract too
        # (cuBLAS and the direct kernel raise on it; split-K computes it).
        assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16, "bf16 operands only"
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )


low_m_bf16_mm = LowMBf16Mm()
