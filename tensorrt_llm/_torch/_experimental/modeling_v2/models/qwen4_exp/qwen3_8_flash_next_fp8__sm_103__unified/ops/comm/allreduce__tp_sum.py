# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks, then add the residual and RMS-normalize, via the trtllm allreduce op.

This target's local variant of the shared `catalog/comm/allreduce` entry, at the site
`tp_sum`: `Qwen4ExpUnifiedModelingV2._tp_sum_op.tp_sum`, the plain sum (op `NONE`) that
completes a block's per-rank partial output under attention TP (tp2, tep2, tp4, tep4) --
after o_proj, the GDN out_proj, the routed expert window and the draft layer's o_proj and
MoE. Copied from the shared entry as it stood; the class, the instance name and
`allreduce_group_args` are unchanged."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

#: The plain sum this target's `tp_sum` calls: one summed tensor, nothing fused.
NONE = int(AllReduceFusionOp.NONE)

#: The fusion the shared entry certifies: `residual_out = sum(input) + residual`,
#: `norm_out = rmsnorm(residual_out) * norm_weight`.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The strategy the default targets pass. The op picks a kernel by message
#: size and topology: one-shot or two-shot over the IPC workspace inside one
#: NVLink domain, NCCL plus a separate norm kernel when the message outgrows
#: the workspace or the ranks cannot reach each other's memory.
AUTO = int(AllReduceStrategy.AUTO)

#: Strategies that never read the IPC workspace.
_NO_WORKSPACE = frozenset(
    {
        int(AllReduceStrategy.NCCL),
        int(AllReduceStrategy.NCCL_SYMMETRIC),
        int(AllReduceStrategy.UB),
    }
)


def allreduce_group_args(mapping, strategy: int = AUTO) -> dict:
    """The communicator side of the call for this rank's TP group.

    A target calls this once in `__init__` and passes `**args` at every call
    site; the values do not change for the life of the mapping.

    The IPC workspace is this op's own fixture: one per mapping, shared with
    every other user of that mapping in the process, allocated on first use.
    It exists only for strategies that read it.

    Mirrors the router in `_torch/distributed/ops.py`: under MPI the plain
    symbol is called and `process_group` stays None; with MPI disabled the
    `_pg` symbol needs the TP process group and this rank's global rank.
    """
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm._utils import mpi_disabled

    workspace = None if strategy in _NO_WORKSPACE else get_allreduce_workspace(mapping)
    process_group, rank = None, None
    if mpi_disabled():
        process_group = mapping.tp_group_pg.boxed()
        rank = torch.distributed.get_rank()
    return dict(
        workspace=workspace,
        group=list(mapping.tp_group),
        strategy=strategy,
        process_group=process_group,
        rank=rank,
    )


class Allreduce(OpWrapper):
    """Sum `input` over every rank in `group`; with `op=RESIDUAL_RMS_NORM` also
    add `residual` and RMS-normalize the sum.

    Returns fresh tensors and leaves `input` and `residual` untouched -- unlike
    `flashinfer_fused_add_rmsnorm`, which writes both of its arguments. For
    `RESIDUAL_RMS_NORM` the return value is `(norm_out, residual_out)`, the
    order a call site reads as `x, residual = op(...)`; for `NONE` it is the
    one summed tensor.

    This is the raw op, not the `AllReduce` module. The module also tries
    MNNVL and symmetric-memory kernels before it reaches this op, and runs an
    autotuner over its strategies. A target that calls this entry therefore
    does not get those paths: on GB200/GB300 the module's AUTO prefers MNNVL,
    and across trays this op falls back to NCCL with an unfused norm.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The shared entry's cells for the op's fused form, kept at the two and
    # four ranks of one GB300 tray and re-driven here; its eight-rank cells
    # (one B200/B300 node) stay with the shared entry. The gpt-oss default
    # target calls the fused form twice per layer, after o_proj and after the
    # MoE, on the full token set every rank holds.
    _GPT_OSS = dict(
        hidden=2880, dtype=torch.bfloat16, eps=1e-5, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )
    # The deepseek default target at attention TP2, TP4 and TP8: the same two
    # calls per layer at deepseek's hidden width.
    _DEEPSEEK = dict(
        hidden=7168, dtype=torch.bfloat16, eps=1e-6, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )
    # This target's tp_sum: Qwen3.8-Flash-Next's hidden 2560, op NONE. `eps` is the
    # checkpoint's rms_norm_eps, which the call site passes and NONE never reads.
    _TP_SUM = dict(hidden=2560, dtype=torch.bfloat16, eps=1e-6, op=NONE, strategy=AUTO)

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token, tp4", spec=dict(world=4, rows=1, **_GPT_OSS)),
        Cell(why="a decode batch, tp4", spec=dict(world=4, rows=64, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp4: the message outgrows the one-shot kernel",
            spec=dict(world=4, rows=2048, **_GPT_OSS),
        ),
        Cell(
            why="a full 8192-token prefill chunk, tp4: the widest error measured",
            spec=dict(world=4, rows=8192, **_GPT_OSS),
        ),
        Cell(why="decode, one token, tp2", spec=dict(world=2, rows=1, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp2",
            spec=dict(world=2, rows=2048, **_GPT_OSS),
        ),
        Cell(why="deepseek, decode, one token, tp4", spec=dict(world=4, rows=1, **_DEEPSEEK)),
        Cell(why="deepseek, a decode batch, tp4", spec=dict(world=4, rows=64, **_DEEPSEEK)),
        Cell(why="deepseek, prefill chunk, tp4", spec=dict(world=4, rows=2048, **_DEEPSEEK)),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp4",
            spec=dict(world=4, rows=8192, **_DEEPSEEK),
        ),
        Cell(why="deepseek, decode, one token, tp2", spec=dict(world=2, rows=1, **_DEEPSEEK)),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp2",
            spec=dict(world=2, rows=8192, **_DEEPSEEK),
        ),
        # This target's `tp_sum` (`Qwen4ExpUnifiedModelingV2._tp_sum_op`): the plain sum of a
        # block's per-rank partial under attention TP, at the checkpoint's hidden 2560. Every
        # rank holds every row, so the row count is the step's token count: one decode
        # token, the largest captured decode batch, the MTP verify rows at tep4 (128
        # requests x 4 for MTP3, x 8 for MTP7) and a max_num_tokens prefill chunk.
        Cell(why="tp_sum decode, one token, tp2/tep2", spec=dict(world=2, rows=1, **_TP_SUM)),
        # The captured decode batches tp2/tep2 and tp4/tep4 measurably run (19 sizes, 1, 2, 4,
        # 8, then 16..128 by 8), sampled between the ones above: each is the row count of one
        # captured graph.
        Cell(
            why="tp_sum at a captured decode batch, 24 rows, tp2/tep2",
            spec=dict(world=2, rows=24, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at a captured decode batch, 72 rows, tp2/tep2",
            spec=dict(world=2, rows=72, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at a captured decode batch, 120 rows, tp2/tep2",
            spec=dict(world=2, rows=120, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at the largest captured decode batch, 128 rows, tp2/tep2",
            spec=dict(world=2, rows=128, **_TP_SUM),
        ),
        # World 2's prefill chunks between 2048 and max_num_tokens: messages of 15 and 30 MiB,
        # inside the 64 MiB IPC workspace, where AUTO still answers from its lookup table.
        Cell(
            why="tp_sum over a 3072-row prefill chunk, tp2/tep2",
            spec=dict(world=2, rows=3072, **_TP_SUM),
        ),
        Cell(
            why="tp_sum over a 6144-row prefill chunk, tp2/tep2",
            spec=dict(world=2, rows=6144, **_TP_SUM),
        ),
        Cell(
            why="tp_sum over a max_num_tokens prefill chunk, 8192 rows, tp2/tep2: AUTO takes NCCL",
            spec=dict(world=2, rows=8192, **_TP_SUM),
        ),
        Cell(why="tp_sum decode, one token, tp4/tep4", spec=dict(world=4, rows=1, **_TP_SUM)),
        Cell(
            why="tp_sum at a captured decode batch, 40 rows, tp4/tep4",
            spec=dict(world=4, rows=40, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at a captured decode batch, 96 rows, tp4/tep4",
            spec=dict(world=4, rows=96, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at the largest captured decode batch, 128 rows, tp4/tep4",
            spec=dict(world=4, rows=128, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at an MTP3 verify step of 128 requests, 512 rows, tep4",
            spec=dict(world=4, rows=512, **_TP_SUM),
        ),
        Cell(
            why="tp_sum at an MTP7 verify step of 128 requests, 1024 rows, tep4",
            spec=dict(world=4, rows=1024, **_TP_SUM),
        ),
        Cell(
            why="tp_sum over a max_num_tokens prefill chunk, 8192 rows, tp4/tep4: AUTO takes NCCL",
            spec=dict(world=4, rows=8192, **_TP_SUM),
        ),
    )

    note = """
    Two forms are certified: `NONE`, the plain sum this target's `tp_sum`
    calls (hidden 2560), and `RESIDUAL_RMS_NORM`, the fusion the gpt-oss and
    deepseek default targets call (hidden 2880 / 7168). The quantizing fusions
    (FP8, NVFP4 outputs) are not modelled.

    Source: compiled-in C++, `torch.ops.trtllm.allreduce` /
    `torch.ops.trtllm.allreduce_pg` (`AllreduceOp` in
    `cpp/tensorrt_llm/thop/allreduceOp.cpp`); nothing is JIT-compiled. AUTO
    picks the kernel per call in `AllreduceOp::selectImplementation`: NCCL when
    the message outgrows the 64 MiB IPC workspace or the ranks lack P2P/NVLink,
    otherwise `selectStrategyLookUpTable` (`common/customAllReduceUtils.h`;
    sm_103 reads the SM100 table) by world, op, floor(log2(hidden)) and
    floor(log2(rows)). For tp_sum (op NONE, hidden 2560) that table answers
    ONESHOT up to 8191 rows at world 2 and up to 2047 rows at world 4, NCCL
    from there; TWOSHOT is never taken. Recorded with `TLLM_LOG_LEVEL=DEBUG` on
    GB300, the same on both symbols: world 2 takes ONESHOT at every tp_sum cell
    from 1 to 6144 rows (5 KiB to 30 MiB) and NCCL at 8192 rows (40 MiB); world
    4 takes ONESHOT from 1 to 1024 rows (5 KiB to 5 MiB) and NCCL at 8192 rows.

    Measured on GB300 (sm_103), against the fp32 sum rounded once, in bf16 ulp
    (element against the row max / relative RMS), inside the test's 5 / 1.25
    band, identical on both symbols. tp_sum: 0.00 / 0.000, every element
    bit-exact, at every ONESHOT cell (world 2 at 1, 24, 72, 120, 128, 3072 and
    6144 rows; world 4 at 1, 40, 96, 128, 512 and 1024 rows) and at world 2 x
    8192 rows (with two ranks NCCL rounds one two-term sum, as the reference
    does); world 4 x 8192 rows (NCCL, which rounds each partial sum) 2.53 /
    0.759, max abs 6.25e-2, 5.71e-3 of the largest |sum|, 67.1% of elements
    bit-exact. Fused form: at most 3.71 / 0.968 (world 4, 8192 rows). A
    reference missing one rank sits 192.7 / 148.4 ulp away at world 4 and
    343.1 / 255.1 at world 2.

    Every rank gets bit-identical sums at every tp_sum cell, eager and under
    CUDA-graph replay at the 19 captured decode batches (1, 2, 4, 8, then
    16..128 by 8) and at 512 and 1024 rows, on both symbols -- the property the
    target's replicated hidden state and its routed expert windows rely on.

    The matrix runs every check over both symbols `allreduce_group_args`
    routes to: torch.distributed (`allreduce_pg` on the TP process group, MPI
    disabled) and an MPI session under mpirun (`allreduce`, `process_group`
    None, the IPC workspace exchanged by the engine's `MPIDist`), which is what
    a served engine under `mpirun` + `trtllm-llmapi-launch` calls. A counting
    check holds that the call reached exactly that transport's symbol, once.

    `trigger_completion_at_end` only changes when the next kernel may launch
    under PDL. It is not a cell axis. `eps` is not read under `NONE`.

    Not certified here: tp_sum at 256 to 511 rows (the ONESHOT bucket between
    the 128 and 512 cells, at either world); at world 4, 1025 to 2047 rows
    (ONESHOT, the 1024 cell's bucket) and 2048 to 8191 rows (NCCL, measured
    only at 8192); the quantizing fusions; the shared entry's eight-rank cells;
    more than one node (across trays this op falls back to NCCL); sm_100.
    """

    def __call__(
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
        rank: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """`process_group` selects the symbol: None runs the MPI one, a boxed
        c10d ProcessGroup runs `allreduce_pg`, which also needs `rank`.
        `allreduce_group_args` builds the communicator arguments for a mapping.
        """
        if process_group is None:
            outs = torch.ops.trtllm.allreduce(
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
        else:
            outs = torch.ops.trtllm.allreduce_pg(
                input,
                residual,
                norm_weight,
                scale,
                bias,
                workspace,
                group,
                rank,
                process_group,
                strategy,
                op,
                eps,
                trigger_completion_at_end,
            )
        return outs[0] if len(outs) == 1 else tuple(outs)

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        eps: float,
        op: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """What every rank should get back.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so this takes every
        rank's `input`, in rank order.

        Summed in fp32 and rounded once. For `RESIDUAL_RMS_NORM` the residual
        is added to the fp32 sum, the norm reads that fp32 value, and both
        outputs are rounded to the input dtype at the end.
        """
        acc = torch.zeros(inputs_in_rank_order[0].shape, dtype=torch.float32, device="cuda")
        for x in inputs_in_rank_order:
            acc += x.float()
        dtype = inputs_in_rank_order[0].dtype
        if op == int(AllReduceFusionOp.NONE):
            return acc.to(dtype)
        assert op == RESIDUAL_RMS_NORM, f"reference covers NONE and RESIDUAL_RMS_NORM, not {op}"
        h = acc + residual.float()
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * norm_weight.float()).to(dtype), h.to(dtype)

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
        rank: Optional[int] = None,
    ) -> None:
        # The kernels take raw pointers and read every operand as packed rows.
        for name, tensor in (("input", input), ("residual", residual)):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; the kernels read it as packed rows"
                )
        if op == RESIDUAL_RMS_NORM:
            assert residual is not None and norm_weight is not None, (
                "RESIDUAL_RMS_NORM reads residual and norm_weight; a None pointer is not rejected"
            )
            assert residual.shape == input.shape, (residual.shape, input.shape)
            assert norm_weight.shape == (input.shape[-1],), (norm_weight.shape, input.shape)
        assert strategy in _NO_WORKSPACE or workspace is not None, (
            f"strategy {strategy} reads the IPC workspace; build one with allreduce_group_args"
        )


allreduce = Allreduce()
