# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-shaped bf16 projection: one kernel per call, its source picked per (M, N, K).

At decode M a dense bf16 projection streams its weight once, and cuBLAS
reaches only 0.55-0.65 of HBM bandwidth on 0.25-63 MB weights (and ~4.5 us
flat on the small ones). FlashInfer's Blackwell low-M CuTe-DSL kernels -- the
register-prefetch "direct" kernel and the tcgen05 split-K kernel, the two
`flashinfer.mm_bf16(backend="cute-dsl")` dispatches to and
`tensorrt_llm/_torch/modules/low_m_gemm.py` races -- are faster at M <= 32 on
most such shapes. Which one, with which tactic, is a per-(N, K) table measured
per M bucket on GB300 (cold L2, CUDA-graph replay); every shape the table does
not list, and every M above it, stays on the trtllm cuBLASLt op. The rows
today are qwen3.8-flash-next-nvfp4's projections (hidden 2560, tp1).

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
#: bound, and any (N, K) not listed, runs on cuBLAS. Every row below is a
#: qwen3.8-flash-next-nvfp4 tp1 projection; a target with other shapes adds its
#: own rows after measuring them. Each choice beat cublas_mm by >= 5% at the
#: bound's M in a sweep of these shapes (GB300, cold L2, graph replay, pdl off).
#: M=3 uses the M=4 choice, and so on.
_TABLE: dict[tuple[int, int], tuple[tuple[int, tuple], ...]] = {
    # GDN in_qkv: 16 q + 16 k heads x 128, 48 v heads x 128
    (10240, 2560): ((1, _d(64, 1)), (2, _d(64, 2)), (8, _s(128, 8, 1, 6))),
    # GDN in_z
    (6144, 2560): ((1, _d(64, 1)), (2, _d(64, 2))),
    # GDN in_a / in_b (48 v heads)
    (48, 2560): ((32, _d(64, 1)),),
    # GDN out and attention o: [6144 -> 2560]
    (2560, 6144): (
        (2, _d(256, 2)),
        (8, _s(64, 8, 3, 10)),
        (16, _s(64, 16, 3, 8)),
        (32, _s(64, 32, 3, 8)),
    ),
    # attention q (q and its output gate interleaved per head)
    (12288, 2560): ((2, _d(64, 2)), (4, _s(128, 8, 1, 6)), (8, _s(128, 16, 1, 6))),
    # attention k / v (2 kv heads x 256) and the MoE router (512 experts)
    (512, 2560): ((2, _d(64, 2)), (8, _d(64, 1)), (16, _d(32, 1)), (32, _s(64, 8, 4, 10))),
    # QSA index projection: 4 index heads x 128 + 1 key x 128
    (640, 2560): ((4, _d(64, 1)), (8, _d(32, 1)), (16, _s(64, 8, 4, 10)), (32, _s(64, 16, 4, 8))),
    # shared expert gate_up, and the same with its gate row appended (padded to 1296)
    (1280, 2560): ((2, _d(64, 1)), (4, _d(64, 2)), (8, _s(64, 8, 4, 10))),
    (1296, 2560): ((2, _d(64, 1)), (4, _d(64, 2)), (8, _s(64, 8, 4, 10))),
    # shared expert gate [2560 -> 1]
    (1, 2560): ((32, _d(64, 1)),),
    # the four GDN input projections concatenated (10240 + 6144 + 48 + 48)
    (16480, 2560): ((1, _d(64, 2)),),
    # the attention input projections concatenated: q|gate (12288), k, v (512 each),
    # index (640), zero-padded to 55 heads of 256 (14080)
    (14080, 2560): ((2, _d(64, 2)), (8, _s(128, 8, 1, 6))),
    # PLE value projection (decoder layer 1 only); the PLE key projection is [10240, 2560]
    (2560, 2560): ((1, _d(64, 1)), (2, _d(64, 2)), (4, _d(32, 2)), (8, _s(64, 8, 2, 10))),
}
# Measured and left on cuBLAS: (2560, 640) shared-expert down (the direct kernel
# cannot tile K=640; split-K and TGV were within 4% of cuBLAS), the GDN concat
# (16480, 2560) at M >= 2 (direct 3.6-5.3% ahead at M=2 across two sweeps, behind
# from M=4), the attention concat (14080, 2560) at M >= 16, and M >= 64
# everywhere (outside both kernels' M <= 32 policy). Not servable at all: the
# hyper-connection mixer's first projection (fp32 output; both kernels write
# bf16 only) and its up projection [10240, 320] (K=320 tiles neither kernel).


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

    # Every cell is a qwen3.8-flash-next-nvfp4 tp1 decode projection at M >= 2 (one row
    # takes `bf16_gemv_pdl` in the target), one per branch; `kind` pins the branch `plan`
    # takes for it. The test's host-only plan pins hold the table's other shapes.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "qwen3.8 decode M=2 GDN in_z: direct, 2-row tile, pdl on",
            2,
            6144,
            2560,
            "direct",
            pdl=True,
        ),
        _cell(
            "qwen3.8 decode M=32 o_proj: split-K, mma_n 32, pdl on (M boundary)",
            32,
            2560,
            6144,
            "splitk",
            pdl=True,
        ),
        _cell("qwen3.8 decode M=128: cuBLAS branch above M=32", 128, 10240, 2560, "cublas"),
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

    Hard limits
      M    FlashInfer branches only at M <= 32 (both kernels' policy)  boundary 32  driven at 32
           (cells) and 128 (cuBLAS cell)
      N*K  largest FlashInfer-branch weight 16480*2560 = 4.2e7 elements  boundary 2^31  not probed (48x below it)
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
