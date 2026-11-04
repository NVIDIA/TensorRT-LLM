# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank's K-split partial of the MTP front's eh_proj, on CuTe DSL GEMMs per row count."""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.gemm import mtp_eh_proj_n6144_k3072_bf16 as _kernel

Runner = _kernel.GLMMtpEhProjAdapter

#: The row counts the runner holds a compiled GEMM for. The call site checks
#: membership and sends every other row count to the stock path.
SUPPORTED_M_VALUES = _kernel.SUPPORTED_M_VALUES
#: K of the local GEMM: one quarter of the 12288-wide concat.
LOCAL_INPUT_SIZE = _kernel.LOCAL_INPUT_SIZE


def build(weight: nn.Parameter) -> Runner:
    """Compile one GEMM per supported row count and bind them to `weight`.

    Call once, at load. Returns the runner the op takes as `runner`. `weight` is
    this rank's contiguous `[6144, 3072]` bf16 column slice of eh_proj
    (`W[:, r*3072 : (r+1)*3072]`), wrapped in an `nn.Parameter` (the kernel's
    builder refuses a plain tensor); the runner keeps a reference. Costs sixteen
    CuTe DSL GEMM compiles (7 low-latency for M <= 32, 9 persistent above).
    """
    return _kernel.make_trtllm_adapter(weight, None)


class CuteDslMtpEhProj(OpWrapper):
    """`input @ weight.T`: `[M, 3072] x [6144, 3072]^T -> [M, 6144]` bf16, a new tensor.

    This is the local, pre-collective partial of a TP4 row-parallel linear: the
    caller sums it across ranks. `input` is row-major with unit column stride
    and row stride 3072 (compact).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target runs this from the MTP front only at M in SUPPORTED_M_VALUES,
    # with the compact [M, 3072] output of cute_dsl_mtp_norm_shard as input.
    # M <= 32 runs a low-latency GEMM, M >= 48 a persistent one with a
    # per-M tile; M = 256 is the one bucket in the direct (unswapped)
    # orientation.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one draft row: low-latency GEMM, 8-column tiles", spec=dict(rows=1)),
        Cell(why="32 rows: the largest low-latency bucket, 32-column tiles", spec=dict(rows=32)),
        Cell(why="48 rows: the smallest persistent bucket", spec=dict(rows=48)),
        Cell(why="the full decode batch, 64 rows", spec=dict(rows=64)),
        Cell(why="256 rows: the one direct-orientation bucket", spec=dict(rows=256)),
        Cell(why="max_num_tokens, 8192 rows", spec=dict(rows=8192)),
    )

    note = """
    The target builds one runner at load from its rank's weight slice. It calls
    the op only when the row count is in SUPPORTED_M_VALUES and both MTP inputs
    are contiguous; every other step takes the stock path at the call site (two
    flashinfer RMSNorms, a concat, a cuBLAS GEMM over the full 12288 columns,
    no collective). A row count outside SUPPORTED_M_VALUES raises here.

    The kernel also accepts a row stride of 12288 (a quarter-column view of the
    full concat). The target never passes one, and no cell drives it.
    """

    def __call__(self, input: torch.Tensor, weight: nn.Parameter, runner: Runner) -> torch.Tensor:
        return runner(input)

    def reference(self, input: torch.Tensor, weight: nn.Parameter, runner: Runner) -> torch.Tensor:
        """The fp32-accumulated product, rounded once. `runner` is ignored.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, which would make the reference the less accurate side.
        """
        with true_fp32_matmul():
            return (input.float() @ weight.float().t()).to(input.dtype)

    def is_valid(self, input: torch.Tensor, weight: nn.Parameter, runner: Runner) -> None:
        # The runner multiplies by the weight it was built with and never looks
        # at `weight`: a runner built on another rank's slice, or on a stale
        # copy, returns a plausible wrong partial.
        assert runner.weight is weight, (
            "runner was built on a different weight tensor; it ignores the one passed"
        )
        # Built with a collective, the runner returns the all-reduced sum rather
        # than this rank's partial, and the caller's own reduce then adds the
        # ranks a second time.
        assert runner.all_reduce is None, "runner must be built without a collective"


cute_dsl_mtp_eh_proj = CuteDslMtpEhProj()
