# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum a tensor across the tensor-parallel group, via the trtllm allreduce op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

#: The dtypes whose NCCL mapping sums values (`getDtypeMap` in
#: cpp/tensorrt_llm/common/opUtils.cpp). float8 and bool both map to ncclInt8,
#: so their raw bytes would be added.
_SUMMABLE_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
    }
)

#: Read only by the fused-norm patterns; a plain sum ignores it.
_EPS_UNUSED = 1e-6

HIDDEN = 8192


def _cell(why: str, rows: int, strategy: AllReduceStrategy, kernel: str) -> Cell:
    """`kernel` is the implementation the cell certifies actually runs.

    The matrix tells them apart by their arithmetic, so a strategy that
    silently falls back to NCCL -- a missed NVLink, an undersized workspace --
    fails the cell instead of passing it on NCCL's answer.
    """
    return Cell(
        why=why,
        spec=dict(
            rows=rows,
            hidden=HIDDEN,
            dtype=torch.bfloat16,
            strategy=strategy,
            kernel=kernel,
        ),
    )


class Allreduce(OpWrapper):
    """Sum `input` over every rank of `group`; every rank gets the same bits back.

    A plain sum (fusion op NONE) of a dense tensor. The result is a fresh
    tensor of `input`'s shape and dtype on every rank; `input` is untouched.

    `strategy` is an `AllReduceStrategy` and picks the implementation:

    * NCCL -- `ncclAllReduce`. Needs no workspace. Sums in the input dtype,
      rounding at every hop.
    * ONESHOT -- a one-shot kernel over NVLink peer buffers. Sums in fp32 in
      ascending rank order and rounds once, which is exactly `reference`.
    * TWOSHOT -- reduce-scatter then all-gather inside one kernel, same
      arithmetic as ONESHOT; with fewer rows than ranks it runs ONESHOT.
    * AUTO -- chosen per call from the op's SM100 table, which sm_107 reads
      too. Which of the three it picks depends on the group size and the row
      count (measured on sm_107 at four ranks and hidden 8192: ONESHOT up to
      256 rows, TWOSHOT from 512 to 2048, NCCL from 4096); the cells below
      name the kernel each one certifies.

    ONESHOT, TWOSHOT and AUTO need `workspace`, the engine's IPC buffer table:
    `tensorrt_llm._torch.distributed.ops.get_allreduce_workspace(mapping)`,
    the same tensor `AllReduce(mapping).workspace` holds. Fetch it once,
    outside capture. All three fall back to NCCL, correctly and silently, when
    the message exceeds that workspace (64 MiB unless
    TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE says otherwise: more than 4096
    rows at hidden 8192 bf16) or the group lacks NVLink peer access.

    `process_group` selects the communicator flavor, mirroring the product
    router in `_torch/distributed/ops.py`. None runs the MPI-session symbol --
    the LLM API's default worker mode, where `group=mapping.tp_group` is all
    it needs: the NCCL communicator is built from the MPI session on the
    group's first call, which must not be inside a CUDA graph capture (the
    engine's warm-up forward makes it). A boxed c10d ProcessGroup
    (`mapping.tp_group_pg.boxed()`, under TLLM_DISABLE_MPI=1) runs the `_pg`
    symbol and also needs `rank`, this process's global rank, plus a prior
    `tensorrt_llm.bindings.internal.process_group.init_pg(...)` -- the engine's
    TorchDist makes that call; without it the op dereferences a null process
    group and the process dies (SIGSEGV, measured), which no Python-side guard
    can see.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_107})

    # One all-reduce per layer of the two Nemotron-3 Ultra TP targets that call
    # it: the attention o_proj partial, the mamba out_proj partial, and the MoE
    # latent-plus-shared partial, all bf16 [T, 8192] over the TP ranks -- four
    # in nemotron3_ultra_nvfp4__sm_107__tep4, eight in
    # nemotron3_ultra_bf16__sm_100__tep8. The row counts are a decode token, the
    # largest captured decode batch and a full prefill chunk; AUTO adds the
    # middle of its TWOSHOT band, since it changes kernel with T.
    CELLS: tuple[Cell, ...] = (
        _cell("NCCL, decode: 1 row, no workspace", 1, AllReduceStrategy.NCCL, "nccl"),
        _cell("NCCL, decode batch: 32 rows", 32, AllReduceStrategy.NCCL, "nccl"),
        _cell("NCCL, prefill chunk: 8192 rows, 128 MiB", 8192, AllReduceStrategy.NCCL, "nccl"),
        _cell(
            "ONESHOT, decode: 1 row over the IPC workspace", 1, AllReduceStrategy.ONESHOT, "oneshot"
        ),
        _cell("ONESHOT, decode batch: 32 rows", 32, AllReduceStrategy.ONESHOT, "oneshot"),
        _cell("AUTO at 1 row: its table picks ONESHOT", 1, AllReduceStrategy.AUTO, "oneshot"),
        _cell("AUTO at 32 rows: its table picks ONESHOT", 32, AllReduceStrategy.AUTO, "oneshot"),
        _cell(
            "AUTO at 2048 rows: its table picks TWOSHOT", 2048, AllReduceStrategy.AUTO, "twoshot"
        ),
        _cell(
            "AUTO at 8192 rows: 128 MiB exceeds the workspace, NCCL runs",
            8192,
            AllReduceStrategy.AUTO,
            "nccl",
        ),
    )

    note = """
    Numerics, measured on every cell and certified by the matrix: every rank
    receives bit-identical results (which is what keeps TP-replicated
    activations replicated) and repeated calls are bitwise deterministic.
    ONESHOT / TWOSHOT equal `reference` bit for bit. NCCL differs from it in
    about a third of the elements of a random payload (0.329 to 0.330 on every
    NCCL cell) -- its per-hop bf16 rounding -- and stays inside the
    sequential-summation bound gamma_{n-1} * sum|x_r| (bf16 unit roundoff
    2^-8), the worst element at 0.90 of it on sm_100 (the 8192-row cell; 0.61
    at one row) and at 0.95 on sm_107. The custom kernels sit at 0.33 of it
    (sm_100). Both architectures were driven at 4 ranks inside one node --
    sm_100 on a GB200 node, sm_107 over NVLink; the 8-rank, one-node case is
    not covered by any run. On payloads whose partial sums are all exact,
    every strategy returns the exact sum.

    The one-shot kernel treats a 32-bit word equal to fp32 -0.0 as its
    empty-slot marker and rewrites it to +0.0 on the way in, so a bf16 -0.0 at
    an odd index whose even neighbour is +0.0 is summed as +0.0. Only the sign
    of a zero can change.

    CUDA graphs: capturable on every certified path, including a group's
    first-ever call under the `_pg` symbol (its communicator exists from
    init_process_group). The workspace is read through device pointers, so
    eager calls between graph replays share it safely (the matrix runs eager
    NCCL, one-shot and two-shot calls between replays of one-shot graphs).

    Measured per call inside a CUDA graph, 4 x sm_107 over NVLink, MPI
    symbol: NCCL 15 / 16 / 268 us at 1 / 32 / 8192 rows; ONESHOT 3.9 / 6.1
    us at 1 / 32 rows; TWOSHOT 97 us at 2048 rows. Perf is not gated.

    Not certified: other dtypes (allowed by the guard, exercised by nothing),
    MIN_LATENCY / NCCL_SYMMETRIC / UB / LOWPRECISION, the fusion operands
    (residual, norm weight, quant scale) this entry does not expose, and the
    MPI-session symbol itself: the matrix drives the `_pg` symbol, the same
    kernels behind a different communicator, and the MPI symbol was measured
    separately with the same results on every strategy.
    """

    def __call__(
        self,
        input: torch.Tensor,
        group: List[int],
        strategy: AllReduceStrategy,
        workspace: Optional[torch.Tensor] = None,
        process_group: Optional[object] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        if process_group is None:
            return torch.ops.trtllm.allreduce(
                input,
                None,
                None,
                None,
                None,
                workspace,
                group,
                int(strategy),
                int(AllReduceFusionOp.NONE),
                _EPS_UNUSED,
                True,
            )[0]
        return torch.ops.trtllm.allreduce_pg(
            input,
            None,
            None,
            None,
            None,
            workspace,
            group,
            rank,
            process_group,
            int(strategy),
            int(AllReduceFusionOp.NONE),
            _EPS_UNUSED,
            True,
        )[0]

    def reference(self, per_rank_inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """The sum every rank should receive.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds. It takes the group's
        inputs in ascending rank order, sums them in fp32 (int64 for integer
        dtypes) in that order and rounds once -- which is bit for bit what the
        ONESHOT and TWOSHOT kernels compute, and the exact sum NCCL is gated
        against.
        """
        first = per_rank_inputs[0]
        acc_dtype = torch.float32 if first.dtype.is_floating_point else torch.int64
        acc = torch.zeros(first.shape, dtype=acc_dtype, device=first.device)
        for x in per_rank_inputs:
            acc += x.to(acc_dtype)
        return acc.to(first.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        group: List[int],
        strategy: AllReduceStrategy,
        workspace: Optional[torch.Tensor] = None,
        process_group: Optional[object] = None,
        rank: Optional[int] = None,
    ) -> None:
        # Each guard is a violation measured to return a plausible wrong result
        # rather than raise.
        assert input.is_contiguous(), (
            "input must be contiguous: NCCL under the MPI symbol and the ONESHOT / "
            "TWOSHOT kernels read it as packed memory, so a strided view is summed "
            "from the wrong elements"
        )
        assert input.dtype in _SUMMABLE_DTYPES, (
            f"{input.dtype} is not summed by this op: under the MPI-session symbol float8 "
            f"and bool map to ncclInt8 and their raw bytes are added (measured for float8). "
            f"Reduce in one of {sorted(str(d) for d in _SUMMABLE_DTYPES)} and cast afterwards"
        )


allreduce = Allreduce()
