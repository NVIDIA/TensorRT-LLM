# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""wide_moe_allreduce: the MoE output all-reduce of 129..1024-row decode steps on MNNVL two-shot.

At TEP4 the built-in MoE finalizes a step of up to 128 tokens with its fused
finalize + all-reduce. A larger step (the decode steps of 17 to 128 requests, 8
rows each; above 64 requests, a generation worker's) combines the expert rows
unfused, casts the FP32 combine to BF16 and reduces it over
``self.all_reduce``, which the built-in pins to NCCL: a ring
all-reduce at these sizes, much slower than the MNNVL two-shot the attention
output projection's all-reduce runs on for the same ``[tokens, 4096]`` BF16 shape.

This path puts a ``WideMoeAllReduce`` in every layer's ``feed_forward.all_reduce``,
all sharing one ``WideMoeAllReduces`` switch that ``DecodeTarget`` arms for one
decode forward of ``MIN_TOKENS..MAX_TOKENS`` rows. While it is armed, calls of
``MIN_TOKENS..MAX_TOKENS`` contiguous BF16 rows of width ``HIDDEN`` with no fused
op run the MNNVL all-reduce directly (``MNNVLAllReduce``, the module the built-in
``AllReduce`` builds for ``AllReduceStrategy.MNNVL``); every other call, and every
call of a prefill or mixed step, runs the layer's own NCCL module as before. The
MNNVL module is called directly, not through ``AllReduce``, so a call it declines
goes back to NCCL and never to the AUTO strategy's tunable all-reduce. Every rank
routes a call alike: the choice depends on the step's kind and the call's shape.

The same switch widens the built-in router / pool-projection overlap of an armed
forward. ``MuseSparkOpenMoE.prepare_moe_inputs`` runs the router on the main
stream and the pool projection on the MoE's side stream for calls of up to
``_prepare_overlap_max_tokens`` tokens: 128, or 512 with the built-in opt-in
``MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP``. While armed, every MoE layer whose
bound is the opt-in's 512 takes ``MAX_TOKENS`` instead, so the decode steps of
65 to 128 requests overlap too; the bound returns on any exit. An armed step of
up to 512 rows is under both bounds, so it overlaps exactly as without the
widening; without the opt-in nothing is widened; and prefill and mixed steps
never arm the switch, so they keep the built-in's bound.

Numerics: NCCL's ring sums the four BF16 partials hop by hop and rounds each
hop's sum to BF16; the MNNVL two-shot sums them in FP32 in rank order and rounds
once. Not bitwise the NCCL result (at most one BF16 rounding per hop apart);
checked on GSM8K. The overlap places the same kernels on two streams: bitwise
the serial prepare.
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Iterator, Optional

import torch

from tensorrt_llm._torch.distributed.ops import (
    AllReduce,
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.models.modeling_musespark_open import _LARGE_PREPARE_OVERLAP_MAX_TOKENS
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger

#: The calls this path takes: the steps the fused finalize does not serve, up to
#: 128 requests of 8 rows (a generation worker of ``max_batch_size`` up to 128).
MIN_TOKENS = 129
MAX_TOKENS = 1024
#: The width of the MoE output the built-in reduces.
HIDDEN = 4096


class WideMoeAllReduces:
    """The switch every layer's ``WideMoeAllReduce`` reads, the MNNVL module they share and
    the MoE layers whose prepare overlap it widens.

    Off except inside ``armed``, which ``DecodeTarget`` enters for one decode forward
    of ``takes`` rows. Host state only: under a CUDA graph it is a per-capture constant.
    """

    def __init__(self, mnnvl: MNNVLAllReduce, overlaps: Iterable[torch.nn.Module] = ()) -> None:
        self.mnnvl = mnnvl
        self.active = False
        #: The MoE modules (``MuseSparkOpenMoE``) whose ``_prepare_overlap_max_tokens`` an
        #: armed forward widens from the built-in opt-in's bound to ``MAX_TOKENS``.
        self.overlaps = tuple(overlaps)

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows is armed."""
        return MIN_TOKENS <= rows <= MAX_TOKENS

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Route the layers' MoE all-reduces and widen their prepare overlap for one decode
        forward; disarm and restore every bound on any exit."""
        widened = [
            moe
            for moe in self.overlaps
            if moe._prepare_overlap_max_tokens == _LARGE_PREPARE_OVERLAP_MAX_TOKENS
        ]
        self.active = True
        try:
            for moe in widened:
                moe._prepare_overlap_max_tokens = MAX_TOKENS
            yield
        finally:
            self.active = False
            for moe in widened:
                moe._prepare_overlap_max_tokens = _LARGE_PREPARE_OVERLAP_MAX_TOKENS


class WideMoeAllReduce(torch.nn.Module):
    """A layer's MoE all-reduce: while armed, ``MIN_TOKENS..MAX_TOKENS`` rows on MNNVL; else ``nccl``."""

    def __init__(self, nccl: AllReduce, switch: WideMoeAllReduces) -> None:
        super().__init__()
        self.nccl = nccl
        self.switch = switch

    @staticmethod
    def takes(input: torch.Tensor) -> bool:
        """Whether a call on ``input`` runs the MNNVL all-reduce."""
        return (
            input.dim() == 2
            and MIN_TOKENS <= input.shape[0] <= MAX_TOKENS
            and input.shape[1] == HIDDEN
            and input.dtype == torch.bfloat16
            and input.is_contiguous()
        )

    def forward(
        self, input: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        if all_reduce_params is None and self.switch.active and self.takes(input):
            output = self.switch.mnnvl(input, AllReduceParams())
            if output is not None:
                return output
        return self.nccl(input, all_reduce_params=all_reduce_params)


def refusal(model: torch.nn.Module) -> Optional[str]:
    """Why ``install`` cannot take the model's MoE all-reduces, or None. Changes nothing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (10, 7),
    ):
        return "it is written for sm_103 and sm_107"
    mapping = model.layers[0].mapping
    if mapping.tp_size != 4 or mapping.is_multi_node():
        return f"it serves TP4 on one node, got TP{mapping.tp_size}"
    if mapping.enable_attention_dp or mapping.moe_tp_size != 1:
        return "the MoE output is not all-reduced over the TP group (attention DP or MoE TP)"
    for layer in model.layers:
        reduce = getattr(layer.feed_forward, "all_reduce", None)
        if type(reduce) is not AllReduce or reduce.strategy != AllReduceStrategy.NCCL:
            return f"layer {layer.layer_idx}'s MoE all-reduce is not the built-in NCCL module"
    if not MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True):
        return "MNNVL all-reduce is not supported here"
    return None


def check(model: torch.nn.Module) -> None:
    """Raise ``ValueError`` with ``refusal``'s reason, if any. Changes nothing."""
    reason = refusal(model)
    if reason is not None:
        raise ValueError(f"[musespark_open] wide_moe_allreduce cannot run: {reason}")


def install(model: torch.nn.Module) -> WideMoeAllReduces:
    """Put a ``WideMoeAllReduce`` in every layer's MoE all-reduce (collective over the TP group).

    Call it on every rank once ``check`` has passed on every rank. Builds (or
    reuses, sized for ``MAX_TOKENS`` rows) the group's MNNVL workspace, the one the
    attention output projection's MNNVL all-reduce uses: two-shot buffers of
    ``2 * MAX_TOKENS * HIDDEN`` BF16 elements, no larger than the built-in creates a
    workspace with, so taking up to 1024 rows grows no workspace. Hands the switch every
    MoE layer with a prepare side stream (the overlap it widens while armed). Returns the
    switch ``DecodeTarget`` arms.
    """
    mapping = model.layers[0].mapping
    overlaps = [
        layer.feed_forward
        for layer in model.layers
        if getattr(layer.feed_forward, "_prepare_stream", None) is not None
        and hasattr(layer.feed_forward, "_prepare_overlap_max_tokens")
    ]
    switch = WideMoeAllReduces(MNNVLAllReduce(mapping, torch.bfloat16), overlaps)
    get_or_scale_allreduce_mnnvl_workspace(
        mapping,
        torch.bfloat16,
        buffer_size_bytes=MNNVLAllReduce.get_required_workspace_size(
            MAX_TOKENS, HIDDEN, mapping.tp_size, torch.bfloat16
        ),
    )
    for layer in model.layers:
        moe = layer.feed_forward
        moe.all_reduce = WideMoeAllReduce(moe.all_reduce, switch)
    widens = sum(
        moe._prepare_overlap_max_tokens == _LARGE_PREPARE_OVERLAP_MAX_TOKENS for moe in overlaps
    )
    logger.info(
        f"[musespark_open] wide_moe_allreduce: the MoE output all-reduce of all {len(model.layers)} "
        f"layers runs on the MNNVL two-shot on decode steps of {MIN_TOKENS}..{MAX_TOKENS} rows; "
        f"NCCL otherwise; the router / pool overlap of {widens} layers covers those steps up to "
        f"{MAX_TOKENS} rows"
    )
    return switch
