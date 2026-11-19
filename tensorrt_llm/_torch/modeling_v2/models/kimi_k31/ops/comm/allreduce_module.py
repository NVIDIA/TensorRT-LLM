# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 tensor-parallel sum, optionally RMS-normalized, through the engine AllReduce module."""

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

#: Kimi K3.1's hidden size, routed-expert latent width and RMSNorm epsilon.
_HIDDEN = 7168
_LATENT = 3584
_EPS = 1e-5
_NONE = int(AllReduceFusionOp.NONE)
_RMS_NORM = int(AllReduceFusionOp.RMS_NORM)
#: The two fusions the target calls.
_CERTIFIED_FUSIONS = frozenset({_NONE, _RMS_NORM})
#: The strategy a served engine passes (model_config.allreduce_strategy).
_AUTO = int(AllReduceStrategy.AUTO)


class AllreduceModule(OpWrapper):
    """Sum `input` over `all_reduce`'s TP group; with `fusion_op=RMS_NORM` also normalize the sum.

    `all_reduce` is the engine `AllReduce` module the target built for that group: it carries the
    mapping, the strategy and the dtype its workspaces were sized for, and its forward picks the
    kernel (symmetric memory, MNNVL, or the trtllm allreduce op's one-shot / two-shot / NCCL by
    message size, autotuned under MPI). `fusion_op` is `NONE` (`norm_weight` and `eps` unused) or
    `RMS_NORM`: `rmsnorm(sum) * norm_weight` with `eps`, no residual.

    Returns a new tensor shaped like `input` in `input.dtype`, identical on every rank of the group.
    A group of one rank returns `input` itself.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 {site} ({fusion}, {hidden}), {strategy}, M={rows}",
                spec=dict(
                    world=4,
                    rows=rows,
                    hidden=hidden,
                    fusion_op=fusion_op,
                    strategy=int(getattr(AllReduceStrategy, strategy)),
                    eps=_EPS,
                ),
            )
            for site, fusion, fusion_op, hidden in (
                ("attention / MLP output reduce", "NONE", _NONE, _HIDDEN),
                ("routed latent reduce + norm", "RMS_NORM", _RMS_NORM, _LATENT),
            )
            for strategy, rows_list in (
                # What a served engine passes, at decode, verify and prefill row counts.
                ("AUTO", (1, 5, 8, 40, 2048)),
                # What AUTO resolves to once the group spans trays (attention TP8 and wider).
                ("MNNVL", (1, 5, 8, 40, 2048)),
                # The kernels AUTO's autotuner chooses among, each forced.
                ("ONESHOT", (1, 40)),
                ("TWOSHOT", (8, 40)),
                ("NCCL", (8, 2048)),
            )
            for rows in rows_list
        ),
        *(
            Cell(
                why=f"K3.1 8192-token prefill ({fusion}, {hidden}), AUTO: past the one-shot sizes",
                spec=dict(
                    world=4,
                    rows=8192,
                    hidden=hidden,
                    fusion_op=fusion_op,
                    strategy=_AUTO,
                    eps=_EPS,
                ),
            )
            for fusion, fusion_op, hidden in (
                ("NONE", _NONE, _HIDDEN),
                ("RMS_NORM", _RMS_NORM, _LATENT),
            )
        ),
    )
    note = """
    Engine-module-internal: this entry does not choose a kernel, the module
    does. Its forward returns `input` for a one-rank group, then tries the
    symmetric-memory kernel (SYMM_MEM strategy, NONE only), then MNNVL (built
    when the strategy is MNNVL, or AUTO on a group that spans nodes -- GB300
    trays -- and the module was given a dtype), then the trtllm allreduce op,
    which picks one-shot, two-shot or NCCL by message size (through the
    autotuner when the strategy is AUTO under MPI, choosing among NCCL,
    NCCL_SYMMETRIC, ONESHOT and TWOSHOT). So the same call can land on
    different kernels for different token counts, and the strategy is a cell
    axis: AUTO as served, ONESHOT / TWOSHOT / NCCL forced so a tuned choice is
    covered whichever it is, and MNNVL. On one GB300 tray AUTO never builds
    MNNVL; the MNNVL cells construct the module with that strategy explicitly,
    which is what AUTO resolves to at attention TP8 and wider. Those wider
    groups themselves are not driven, nor is NCCL_SYMMETRIC.

    Kimi K3.1 calls it with bf16 rows: NONE at 7168 for the MLA o_proj
    partials of a head-sharded group (the MLA runtime, and `oproj_allreduce_push`
    when its push kernel does not serve) and for the row-parallel BF16 linears
    (`K31Linear`: the dense and shared-expert down_proj); RMS_NORM at 3584 for
    the routed latent under direct MoE TP (norm weight
    `routed_expert_norm.weight`, eps 1e-5). Every rank passes the same row
    count. Only NONE and RMS_NORM are certified; the residual and quantizing
    fusions are not modelled by `reference`.

    No residual is added and `input` is not written; every rank gets the
    same bits. The sum is formed in fp32 from the bf16 inputs: one-shot,
    two-shot and MNNVL NONE results equal the fp32 sum rounded once, NCCL's
    bf16 reduction steps differ by up to ~2 ulp; RMS_NORM paths round the
    normalized value before the weight product and differ from the once-rounded
    reference by up to ~2.8 ulp of the row maximum. Under MPI the AUTO path can
    return an NCCL-window tensor, which must be released inside an NCCL window
    tensor scope (the target's decoder layer is `nccl_window_tensor_scoped_module`;
    a test uses `discard_nccl_window_tensor_outputs`). Workspaces (IPC, MNNVL
    Lamport buffers) belong to the module and its mapping, are grown on first
    use for a larger message, and must be grown before CUDA-graph capture
    (`MNNVLAllReduce.prescale_workspace`); TRTLLM_ENABLE_PDL and
    `trigger_completion_at_end` change launch ordering, not results.

    ARCHS lists only what was driven: the SM_107 cells ran on four VR200 GPUs.
    """

    def __call__(
        self,
        input: torch.Tensor,
        all_reduce: AllReduce,
        fusion_op: int,
        norm_weight: Optional[torch.Tensor],
        eps: Optional[float],
    ) -> torch.Tensor:
        if fusion_op == _NONE:
            return all_reduce(input)
        return all_reduce(
            input,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp(fusion_op), norm_weight=norm_weight, eps=eps
            ),
        )

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        fusion_op: int,
        norm_weight: Optional[torch.Tensor],
        eps: Optional[float],
    ) -> torch.Tensor:
        """What every rank should get back.

        The documented collective exception to mirroring `__call__`: the output
        is not a function of what the calling rank holds, so this takes every
        rank's `input`, in rank order, and no module. Summed in fp32; for
        RMS_NORM the fp32 sum is normalized and scaled by the weight in fp32;
        rounded once to the input dtype at the end.
        """
        acc = torch.zeros(inputs_in_rank_order[0].shape, dtype=torch.float32, device="cuda")
        for x in inputs_in_rank_order:
            acc += x.float()
        dtype = inputs_in_rank_order[0].dtype
        if fusion_op == _NONE:
            return acc.to(dtype)
        assert fusion_op == _RMS_NORM, f"reference covers NONE and RMS_NORM, not {fusion_op}"
        normed = acc * torch.rsqrt(acc.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * norm_weight.float()).to(dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        all_reduce: AllReduce,
        fusion_op: int,
        norm_weight: Optional[torch.Tensor],
        eps: Optional[float],
    ) -> None:
        assert int(fusion_op) in _CERTIFIED_FUSIONS, (
            f"only NONE and RMS_NORM are certified; got fusion_op {int(fusion_op)}"
        )
        if int(fusion_op) == _RMS_NORM and norm_weight is not None:
            # The kernels read hidden-size elements through the weight's raw pointer; a shorter
            # weight is not rejected (a missing one is).
            assert norm_weight.numel() == input.shape[-1], (
                f"RMS_NORM reads {input.shape[-1]} norm_weight elements; got {norm_weight.numel()}"
            )


allreduce_module = AllreduceModule()

#: The ``fusion_op`` values of :data:`allreduce_module`, so model code does not import the engine enum.
FUSION_NONE = _NONE
FUSION_RMS_NORM = _RMS_NORM


def uses_mnnvl(all_reduce: AllReduce) -> bool:
    """Whether ``all_reduce`` (a :func:`tp_allreduce` handle) built the MNNVL path, whose Lamport workspace the
    MNNVL gather / push ops use."""
    return getattr(all_reduce, "mnnvl_allreduce", None) is not None


def mnnvl_group(all_reduce: Optional[AllReduce]) -> Optional[tuple[Mapping, torch.dtype]]:
    """``(mapping, dtype)`` of ``all_reduce``'s MNNVL Lamport workspace: the TP group it spans and the dtype it is
    sized for; None for no handle or a handle without the MNNVL path."""
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    return None if mnnvl is None else (mnnvl.mapping, mnnvl.dtype)


def tp_allreduce(mapping, strategy=None, dtype: torch.dtype = torch.bfloat16) -> AllReduce:
    """The engine ``AllReduce`` module for ``mapping``'s TP group: the handle :data:`allreduce_module` takes.

    ``strategy`` is the model config's ``allreduce_strategy`` (an ``AllReduceStrategy`` or its int value; ``AUTO``
    when None) and ``dtype`` the dtype its workspaces are sized for. Built here so that model code builds no engine
    communication module itself.
    """
    return AllReduce(
        mapping=mapping,
        strategy=AllReduceStrategy.AUTO if strategy is None else AllReduceStrategy(strategy),
        dtype=dtype,
    )
