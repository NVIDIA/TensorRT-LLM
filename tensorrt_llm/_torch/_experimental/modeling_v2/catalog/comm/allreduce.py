# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks, optionally adding the residual and RMS-normalizing, via the trtllm allreduce op."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

from .._op import Arch, Cell, OpWrapper

#: The plain sum: one output, `sum(input)` over the group.
NONE = int(AllReduceFusionOp.NONE)

#: The fusion: `residual_out = sum(input) + residual`,
#: `norm_out = rmsnorm(residual_out) * norm_weight`.
RESIDUAL_RMS_NORM = int(AllReduceFusionOp.RESIDUAL_RMS_NORM)

#: The strategy the default targets pass. The op picks a kernel by message
#: size and topology: one-shot or two-shot over the IPC workspace inside one
#: NVLink domain, NCCL plus a separate norm kernel when the message outgrows
#: the workspace or the ranks cannot reach each other's memory.
AUTO = int(AllReduceStrategy.AUTO)

#: NCCL alone: no IPC workspace, so no Lamport slot. The strategy an fp32 sum
#: takes (see `is_valid`).
NCCL = int(AllReduceStrategy.NCCL)

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
    one summed tensor, in the input's dtype.

    This is the raw op, not the `AllReduce` module. The module also tries
    MNNVL and symmetric-memory kernels before it reaches this op, and runs an
    autotuner over its strategies. A target that calls this entry therefore
    does not get those paths: on GB200/GB300 the module's AUTO prefers MNNVL,
    and across trays this op falls back to NCCL with an unfused norm.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The first twenty cells are gpt-oss's and deepseek's; the Step-5-Preview
    # cells follow them.
    #
    # The gpt-oss default target at attention TP2, TP4 and TP8, inside one
    # node: up to four ranks in a GB200/GB300 tray, eight in a B200/B300 node.
    # It calls this twice per layer, after o_proj and after the MoE, on the
    # full token set every rank holds.
    _GPT_OSS = dict(
        hidden=2880, dtype=torch.bfloat16, eps=1e-5, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )
    # The deepseek default target at attention TP2, TP4 and TP8: the same two
    # calls per layer at deepseek's hidden width.
    _DEEPSEEK = dict(
        hidden=7168, dtype=torch.bfloat16, eps=1e-6, op=RESIDUAL_RMS_NORM, strategy=AUTO
    )
    # Step-5-Preview at attention TP4 (its tp4 and tep4 members): a plain sum
    # with no fusion, because the residual stream is fp32 and its norm runs on
    # that stream separately. bf16 carries the o_proj partials and the
    # dense-MLP down partials; fp32 carries the MoE layers' routed plus
    # shared-expert partials. Rows are one decode token, the 128-request decode
    # batch (`max_batch_size`) and one 8192-token prefill chunk
    # (`max_num_tokens`, the envelope's largest row count). NONE reads no
    # `eps`, so these specs carry none. The fp32 sum runs under NCCL: under a
    # workspace strategy it corrupts the next 16-bit one-shot call that reuses
    # its Lamport slot (`is_valid`, and the note).
    _STEP5_BF16 = dict(hidden=4096, dtype=torch.bfloat16, op=NONE, strategy=AUTO)
    _STEP5_FP32 = dict(hidden=4096, dtype=torch.float32, op=NONE, strategy=NCCL)

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
        # Eight ranks in one node is a B200/B300 layout only: a GB200/GB300
        # tp8 spans two trays and takes `comm/mnnvl_allreduce` instead.
        Cell(why="decode, one token, tp8 in one node", spec=dict(world=8, rows=1, **_GPT_OSS)),
        Cell(why="a decode batch, tp8 in one node", spec=dict(world=8, rows=64, **_GPT_OSS)),
        Cell(
            why="prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=2048, **_GPT_OSS),
        ),
        Cell(
            why="a full 8192-token prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=8192, **_GPT_OSS),
        ),
        Cell(
            why="deepseek, decode, one token, tp8 in one node",
            spec=dict(world=8, rows=1, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, a decode batch, tp8 in one node",
            spec=dict(world=8, rows=64, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=2048, **_DEEPSEEK),
        ),
        Cell(
            why="deepseek, a full 8192-token prefill chunk, tp8 in one node",
            spec=dict(world=8, rows=8192, **_DEEPSEEK),
        ),
        Cell(
            why="step5 bf16 plain sum, tp4, one decode token",
            spec=dict(world=4, rows=1, **_STEP5_BF16),
        ),
        Cell(
            why="step5 bf16 plain sum, tp4, 128-request batch",
            spec=dict(world=4, rows=128, **_STEP5_BF16),
        ),
        Cell(
            why="step5 bf16 plain sum, tp4, 8192-token chunk",
            spec=dict(world=4, rows=8192, **_STEP5_BF16),
        ),
        Cell(
            why="step5 fp32 MoE sum via NCCL, tp4, one token",
            spec=dict(world=4, rows=1, **_STEP5_FP32),
        ),
        Cell(
            why="step5 fp32 MoE sum via NCCL, tp4, 128 rows",
            spec=dict(world=4, rows=128, **_STEP5_FP32),
        ),
        Cell(
            why="step5 fp32 MoE sum via NCCL, tp4, 8192 rows",
            spec=dict(world=4, rows=8192, **_STEP5_FP32),
        ),
    )

    note = """
    Certified fusions: `RESIDUAL_RMS_NORM` in bf16 (the gpt-oss and deepseek
    cells) and `NONE` in bf16 under AUTO and in fp32 under NCCL (the Step-5
    cells). The quantizing fusions (FP8, NVFP4 outputs) are not modelled.

    Lamport slot hazard (engine defect, refused by `is_valid`). The one-shot
    kernel (`allreduce_fusion_kernel_oneshot_lamport` in
    `allReduceFusionKernels.cu`) records the size it wrote as
    `params.size * NRanks` elements of its own dtype (`LamportComm::update`),
    and the next one-shot kernel clears that many elements of *its* dtype
    (`clear_size / kElemsPerAccess<DType>`). After an fp32 call the next
    16-bit call clears half of the fp32 call's bytes, and the call after it
    that reuses the slot (three calls on, triple buffering) reads the stale
    half as already arrived. Measured on GB300, world 4, hidden 4096: fp32
    128 rows, bf16 1 row, bf16 1 row, then bf16 1000 rows with rank 0's
    launch delayed: ranks 1 to 3 got 2440 of 5000 rows wrong over 5 trials
    (rows 512 to 999 every time), against 0 when the first call is bf16 at
    the same byte count (256 rows); with no delay, 1301 to 1548 of 5000. The
    Step-5 step boundary (fp32 MoE sum at n rows, two bf16 sums, the next
    step's bf16 sum at m rows), no delay, 20 trials: n 8 -> m 520 up to 306
    wrong rows of 10400, n 32 -> m 128 up to 1131 of 2560, n 128 -> m 1000
    up to 6312 of 20000. A 4-byte input is therefore refused under any
    strategy that reads the IPC workspace; an fp32 sum passes NCCL.

    ARCHS: the gpt-oss and deepseek cells were driven on sm_100 and sm_103.
    The Step-5 cells (`step5 ...`) were driven on sm_103 only and are not
    certified on sm_100 until they run there. The world-8 cells need eight
    GPUs in one node (B200/B300); a four-GPU GB300 tray skips them.

    The cells run the `_pg` symbol, because the certification matrix starts its
    ranks without MPI. A served engine under `mpirun` or
    `trtllm-llmapi-launch` runs the MPI symbol. Both dispatch to the same
    `AllreduceOp` in C++; only the communicator differs, and on the NCCL path
    that means the matrix reduces through the c10d NCCL group while the engine
    reduces through the op's own NCCL communicator.

    AUTO's kernel at Step-5's shape (world 4, hidden 4096, NONE) comes from
    `selectStrategyLookUpTable` in `customAllReduceUtils.h`, indexed by
    floor(log2(rows)) and blind to dtype: one-shot Lamport for 1 to 1023 rows,
    two-shot for 1024 to 4095, NCCL from 4096, and NCCL for any message over
    the 64 MiB IPC workspace. The op's debug log ("AllReduceOp runtime
    strategy") confirmed that split on GB300 in bf16 and fp32. The Step-5
    bf16 cells hold the one-shot kernel (1 and 128 rows) and NCCL (8192
    rows). The two-shot range, which a prefill chunk or mixed batch of 1024
    to 4095 tokens takes, is not a cell: a probe at 1024, 1029, 2048 and 4095
    rows measured it bit-equal to the correctly rounded sum in bf16. The same
    probe ran fp32 under AUTO (one-shot, two-shot, NCCL) within 1.37 fp32 ulp
    of the row scale; that is the accuracy of each call alone, before the
    hazard above.

    16-bit inputs accumulate in fp32 in the one-shot and two-shot kernels
    unless `ALL_REDUCE_FUSION_KERNEL_ACC_FP32=0`. The one-shot kernel replaces
    a negative-zero input word with +0 (its Lamport sentinel), so an all
    negative-zero sum can come back as +0.

    Hard limits: the fusion kernels index elements and 16-byte accesses as
    int32, which reaches 2^31 elements at 524288 rows of 4096; the envelope
    stops at 8192 rows.

    `trigger_completion_at_end` only changes when the next kernel may launch
    under PDL. It is not a cell axis.
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

        Summed in a type wider than the input and rounded once: fp32 for 16-bit
        inputs, fp64 for fp32 inputs, so the reference is the correctly
        rounded sum rather than one more fp32 summation order. For
        `RESIDUAL_RMS_NORM` the residual is added to the wide sum, the norm
        reads that value, and both outputs are rounded to the input dtype at
        the end.
        """
        dtype = inputs_in_rank_order[0].dtype
        wide = torch.float64 if dtype == torch.float32 else torch.float32
        acc = torch.zeros(inputs_in_rank_order[0].shape, dtype=wide, device="cuda")
        for x in inputs_in_rank_order:
            acc += x.to(wide)
        if op == NONE:
            return acc.to(dtype)
        assert op == RESIDUAL_RMS_NORM, f"reference covers NONE and RESIDUAL_RMS_NORM, not {op}"
        h = acc + residual.to(wide)
        normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * norm_weight.to(wide)).to(dtype), h.to(dtype)

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
        # The fp32 call answers right; the next 16-bit one-shot call to reuse
        # its Lamport slot does not (the note's "Lamport slot hazard").
        assert input.element_size() <= 2 or strategy in _NO_WORKSPACE, (
            f"a {input.dtype} input under strategy {strategy} leaves its one-shot Lamport slot "
            "half-cleared for the next 16-bit call on the same workspace; pass strategy NCCL "
            "(allreduce_group_args(mapping, NCCL)) for a 4-byte sum"
        )


allreduce = Allreduce()
