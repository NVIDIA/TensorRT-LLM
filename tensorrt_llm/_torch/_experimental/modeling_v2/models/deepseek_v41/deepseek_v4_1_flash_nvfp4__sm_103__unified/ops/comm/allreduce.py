# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum a tensor across a group of ranks; every rank gets the whole sum back.

The op is `torch.ops.trtllm.allreduce` (MPI-session communicator) or
`torch.ops.trtllm.allreduce_pg` (a c10d ProcessGroup), both registered from
`cpp/tensorrt_llm/thop/allreduceOp.cpp`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HIDDEN = 5120
_ENGRAM_ROW = (24, 256)
_ENGRAM_HIDDEN = 6144

#: Strategies the op implements for a plain sum, by the integer it takes.
_STRATEGY_NAMES = {
    0: "NCCL",
    8: "NCCL_SYMMETRIC",
    3: "AUTO",
    4: "ONESHOT",
    5: "TWOSHOT",
    1: "MIN_LATENCY",
}
#: The strategies that read the IPC workspace; 0 and 8 take `workspace=None`.
_WORKSPACE_STRATEGIES = (3, 4, 5, 1)
_AUTO, _NCCL = 3, 0

#: (layout, world size, group size, nodes). Groups are the contiguous blocks a
#: pipeline-parallel Mapping makes (`Mapping.tp_group`), all running at once.
#: ws4 is tep4/tp4; ws2 one pp2 x tep2/tp2 stage alone; ws4_groups2 the pp2 x
#: tep2/tp2 layout; ws8 tep8 over two 4-GPU NVL72 trays; ws8_groups4 pp2 x
#: tep4/tp4, one group per tray.
_LAYOUTS = (
    ("ws4", 4, 4, 1),
    ("ws2", 2, 2, 1),
    ("ws4_groups2", 4, 2, 1),
    ("ws8", 8, 8, 2),
    ("ws8_groups4", 8, 4, 2),
)

#: Row counts: decode graph sizes, prefill chunks, TWOSHOT's fewer-rows-than-
#: ranks corner (1..3), and both sides of the 64 MiB edge at which a workspace
#: strategy hands an fp32 [T, 5120] call to NCCL (3276 fits, 3277 does not).
_FP32_ROWS = (1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 1024, 2048, 3276, 3277, 4096, 8192)
_BF16_ROWS = (1, 3, 8, 32, 256, 4096, 6553, 6554, 8192)
_ENGRAM_ROWS = (1, 8, 32, 4096)

#: The core target's two call forms, at every row count: fp32 [T, 5120] up to
#: the decoder pass's 128 x 256 rows, bf16 [T, 6144] engram rows up to
#: max_num_tokens.
_CORE_FP32_ROWS = range(1, 32768 + 1)
_CORE_ENGRAM_ROWS = range(1, 16384 + 1)

#: (suite, dtype, trailing dims, row counts, label). The suite is the process
#: a form runs in: fp32 and bf16 never share a workspace's one-shot path.
_FORMS = (
    ("fp32", torch.float32, (_HIDDEN,), _FP32_ROWS, "fp32 [T, 5120]"),
    ("bf16", torch.bfloat16, (_HIDDEN,), _BF16_ROWS, "bf16 [T, 5120]"),
    ("bf16", torch.bfloat16, _ENGRAM_ROW, _ENGRAM_ROWS, "bf16 [T, 24, 256]"),
)


def _group_on_one_node(world_size: int, group_size: int, nodes: int) -> bool:
    """A group with an IPC workspace: all of its ranks on one node."""
    return group_size <= world_size // nodes


def _layout_cells() -> tuple[Cell, ...]:
    cells = []
    for layout, world_size, group_size, nodes in _LAYOUTS:
        one_node = _group_on_one_node(world_size, group_size, nodes)
        base = dict(layout=layout, world_size=world_size, group_size=group_size, nodes=nodes)
        strategies = [s for s in _STRATEGY_NAMES if one_node or s not in _WORKSPACE_STRATEGIES]
        for strategy in strategies:
            for suite, dtype, trailing, rows, label in _FORMS:
                cells.append(
                    Cell(
                        why=f"{layout} {_STRATEGY_NAMES[strategy]} {label}: rows {rows}",
                        spec=dict(
                            base,
                            suite=suite,
                            dtype=dtype,
                            trailing=trailing,
                            rows=rows,
                            strategy=strategy,
                        ),
                    )
                )
        core_fp32 = _AUTO if one_node else _NCCL
        cells.append(
            Cell(
                why=(
                    f"{layout} core fp32 [T, 5120] on {_STRATEGY_NAMES[core_fp32]}: every T "
                    "1..32768, the target's strategy for this group"
                ),
                spec=dict(
                    base,
                    suite="core",
                    dtype=torch.float32,
                    trailing=(_HIDDEN,),
                    rows=_CORE_FP32_ROWS,
                    strategy=core_fp32,
                ),
            )
        )
        cells.append(
            Cell(
                why=f"{layout} core bf16 [T, 6144] on NCCL: every T 1..16384, the engram rows",
                spec=dict(
                    base,
                    suite="core",
                    dtype=torch.bfloat16,
                    trailing=(_ENGRAM_HIDDEN,),
                    rows=_CORE_ENGRAM_ROWS,
                    strategy=_NCCL,
                ),
            )
        )
    return tuple(cells)


class Allreduce(OpWrapper):
    """All-reduce (sum) `input` over `group` with the `op` fusion epilogue.

    Every rank in `group` passes a tensor of the same shape and dtype and gets
    back `[reduced]` (for `op = 0`, the certified value): a freshly allocated
    tensor of `input`'s shape and dtype holding the elementwise sum over the
    group, the same bytes on every rank. `input` is untouched.

    Fusion boundary: the collective and the output allocation are inside the
    call. Everything that produced `input` (a row-parallel GEMM's or the routed
    experts' partial sums), any cast into or out of fp32 around the sum, any
    bias added after it, residual add and normalization are the caller's.
    `strategy` picks the transport; the workspace strategies need the IPC
    `workspace` (see `note`). Returns the op's list unchanged.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Layout x strategy x call form. A cell's `rows` are all run; the core
    # cells are the core target's two forms at every row count, on the strategy
    # it picks for the group (AUTO with the workspace on one node, NCCL across
    # the trays). The ws8 / ws8_groups4 cells span two nodes: see `note`.
    CELLS: tuple[Cell, ...] = _layout_cells()

    note = """
    Metadata consumed: none. Process state: the communicator -- trtllm's MPI
    session communicator when `process_group` is None (`group` names its
    ranks), else the boxed c10d group, whose global ranks `group` lists -- and,
    for strategies 1, 3, 4 and 5, the `[3 * len(group) + 2]` int64 IPC
    `workspace` from `get_allreduce_workspace(Mapping(...))`, built collectively
    on every rank of the group, cached per Mapping, 64 MiB base. A group larger
    than gpus_per_node gets no IPC buffers, so a group spanning the two trays
    runs strategies 0 and 8 only, with `workspace=None`. On the `process_group`
    path strategies 0 and 8 both run the group's own allreduce: the NCCL
    window-buffer form of 8 exists only on the MPI path.

    Preconditions:
    - Every rank of the group calls in the same order with the same shape,
      dtype, `strategy` and `op`; nothing cross-checks ranks.
    - One element size per workspace's one-shot path. The one-shot kernel sizes
      the reset of its previous call's buffer in the current call's element
      size: fp32, then bf16, then fp32 on one workspace returned wrong values on
      every on-time rank, and bf16 [4096, 24, 256] then fp32 never completed.
      AUTO (fp32 up to 1023 rows in a group of 4), TWOSHOT below len(group)
      rows and MIN_LATENCY up to 128 rows all reach that kernel. A forward that
      sums both dtypes keeps one off the workspace strategies; DeepSeek-V4.1
      sums fp32 on AUTO and its bf16 engram rows, as a 2-D [T, 6144] view, on
      NCCL.
    - A 3-D input on a workspace strategy stays within 64 MiB of real size: the
      size check is `shape[0] * shape[-1] * element_size`, and bf16
      [8192, 24, 256] on ONESHOT never completed. Pass a 2-D view.
    - A group's first call is made eagerly, before any CUDA-graph capture.
    - `workspace=None` on 1, 3, 4 or 5 raises `bad optional access`; strategies
      2, 6, 7 and 9 raise at `op = 0`.

    Resolution, measured on sm_103 from the op's debug log: AUTO on fp32
    [T, 5120] is ONESHOT for T 1..1023, TWOSHOT 1024..3276, NCCL from 3277 in a
    group of 4, and ONESHOT 1..3276 then NCCL in a group of 2. Every workspace
    strategy hands a message over 64 MiB to NCCL; TWOSHOT falls back to ONESHOT
    below len(group) rows.

    Numerics: exact whenever the true sum is representable in the dtype (the
    matrix's payloads are built so). Unrestricted fp32 partials are summed in
    fp32 but not correctly rounded and not bit-identical across strategies
    (within assert_close's fp32 defaults, the same bits on every rank); bf16
    sums with cancellation differ by strategy, by up to 0.0625 at 4096 rows.
    Rows owned by one rank (zeros elsewhere) come back bitwise on every
    strategy.

    CUDA graphs are matrix checks, not cells: fp32 on every strategy of the
    layout with bf16 sites on NCCL at 1..128 rows, and the core forward's 82
    sites at the 84 decode row counts T = G, 5G, 6G of the capture batch set.
    Strategy 8 under capture only on payloads whose sums are exact.

    Not certified: `op` other than 0; layouts other than the cells' five;
    groups that are not contiguous blocks of the world; a workspace base other
    than 64 MiB; rows above the cells' ranges; capture at other row counts; a
    group's first call captured.

    The ws8 and ws8_groups4 cells span two 4-GPU nodes of one NVLink domain and
    cannot be collected on one node. They were certified on the MPI-session
    symbol; the `process_group` path at world 8 has not been run. Run each
    suite by hand from this target's comm test directory under
    tests/unittest/_torch/modeling_v2/models/deepseek_v41/:
      TLLM_DISABLE_MPI=1 PYTHONPATH=<repo root> torchrun --nnodes 2
        --nproc-per-node 4 --node-rank <0|1> --master-addr <node 0>
        --master-port <port> _allreduce_op_matrix.py --rank-worker <suite>
        [--group-size 4]
    with <suite> each of fp32, bf16 and core, plus mixed and mixed_core with
    --group-size 4.
    """

    def raw_call(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        process_group: Optional[object] = None,
    ) -> List[torch.Tensor]:
        """`process_group` selects the communicator flavor, mirroring the
        product router in `_torch/distributed/ops.py`: None runs the
        MPI-session symbol, and a boxed c10d ProcessGroup runs the `_pg`
        symbol the non-MPI backends use. That symbol also takes this process's
        global rank, which must sit at the group's own rank in sorted `group`.
        """
        if process_group is None:
            return torch.ops.trtllm.allreduce(
                input,
                residual,
                norm_weight,
                scale,
                bias,
                workspace,
                group,
                strategy,
                op,
                eps,
                trigger_completion_at_end,
            )
        return torch.ops.trtllm.allreduce_pg(
            input,
            residual,
            norm_weight,
            scale,
            bias,
            workspace,
            group,
            torch.distributed.get_rank(),
            process_group,
            strategy,
            op,
            eps,
            trigger_completion_at_end,
        )

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        group: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """The sum the collective is supposed to make, as the op's `[reduced]`.

        The documented exception to mirroring `raw_call`: a collective's output
        is not a function of what the calling rank holds, so this takes the
        group's inputs in ascending rank order instead of `input`. Summed in
        fp64 and rounded once to the inputs' dtype. Covers `op = 0` only;
        `residual`, `norm_weight`, `scale`, `bias`, `eps` are unread there, and
        `workspace`, `strategy`, `trigger_completion_at_end` and
        `process_group` choose the transport, not the value, so none is taken.
        """
        parts = list(per_rank_inputs)
        total = parts[0].double()
        for part in parts[1:]:
            total = total + part.double()
        return [total.to(parts[0].dtype)]

    def compare(self, out: List[torch.Tensor], ref: List[torch.Tensor]) -> None:
        """A gate of exactly zero, per returned tensor.

        The certified payloads are built so that every sum is exact in the
        input dtype (fp32 multiples of 1/256 below 4096 in magnitude, bf16
        multiples of 1/8 of at most 4), and on those every strategy returned
        the correctly rounded fp64 sum bit for bit. On such a payload one
        rank missing or counted twice, the mean instead of the sum, and fp32
        partials rounded to bf16 first each differ on more than 90% of
        elements. Unrestricted payloads are not gated here: their sums are
        not correctly rounded (see `note`).
        """
        assert len(out) == len(ref), (len(out), len(ref))
        for got, want in zip(out, ref):
            assert got.dtype == want.dtype, (got.dtype, want.dtype)
            assert got.shape == want.shape, (got.shape, want.shape)
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def is_valid(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
        process_group: Optional[object] = None,
    ) -> None:
        # Pure-metadata guards, each for a violation measured on this machine to
        # return a wrong answer without raising.
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently reduced over the wrong elements"
        )
        assert input.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the op but summed as raw unsigned bytes, "
            "not as floats, so the result is meaningless; reduce in fp32/bf16 and "
            "quantize afterwards"
        )


allreduce = Allreduce()
