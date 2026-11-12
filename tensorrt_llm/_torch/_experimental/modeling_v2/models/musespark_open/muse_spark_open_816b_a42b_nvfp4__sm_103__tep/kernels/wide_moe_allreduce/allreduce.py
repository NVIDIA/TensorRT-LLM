# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""wide_moe_allreduce: the MoE output all-reduce of 129..1024-row decode steps on MNNVL two-shot.

At TEP4 (one tray of four) and TEP8 (two trays of four in one NVLink domain) the
built-in MoE finalizes a step of up to 128 tokens with its fused finalize +
all-reduce (at TEP8 the fabric finalize). A larger step (the decode steps of 17
to 128 requests, 8 rows each; above 64 requests, a generation worker's) combines
the expert rows unfused, casts the FP32 combine to BF16 and reduces it over
``self.all_reduce``, which the built-in pins to NCCL: a ring
all-reduce at these sizes, much slower than the MNNVL two-shot the attention
output projection's all-reduce runs on for the same ``[tokens, 4096]`` BF16 shape.

The TP group is four ranks on one node, or eight over two nodes of four GPUs in one
NVLink domain (``dispatch.TWO_TRAYS_OF_4``, the topology the sm103 TEP8 composition
is validated on, as the TP8 fabric finalize and the multi-request post-attention hop
are). Across the two trays the MNNVL workspace is a fabric allocation every rank
maps (``get_or_scale_allreduce_mnnvl_workspace`` takes fabric handles once the group
spans nodes); its construction is collective and fails on every rank alike when
the trays share no fabric.

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

Numerics: NCCL's ring sums the TP group's BF16 partials (four at TEP4, eight at
TEP8) hop by hop and rounds each hop's sum to BF16; the MNNVL two-shot sums them
in FP32 in rank order and rounds once. Not bitwise the NCCL result (at most one
BF16 rounding per hop apart); checked on GSM8K, and at eight ranks against the
float64 sum of the partials. The overlap places the same kernels on two streams: bitwise the serial
prepare.
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

from ... import dispatch as _dispatch

#: The calls this path takes: the steps the fused finalize does not serve, up to
#: 128 requests of 8 rows (a generation worker of ``max_batch_size`` up to 128).
MIN_TOKENS = 129
MAX_TOKENS = 1024
#: The width of the MoE output the built-in reduces.
HIDDEN = 4096
#: The most rows the built-in MoE finalizes fused without a chunking workspace.
FUSED_FINALIZE_ROWS = 128


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


def topology_refusal(mapping) -> Optional[str]:
    """Why ``mapping``'s TP group is not one this path serves, or None. Changes nothing.

    TP4: on one node, as the path was first written (refused once the mapping spans
    nodes). TP8: over exactly two nodes of four GPUs (``dispatch.TWO_TRAYS_OF_4``), the
    two trays of one NVLink domain the sm103 TEP8 composition runs on; the group spans
    nodes there, so the TP4 clause would refuse it. Whether MNNVL itself is usable is
    ``refusal``'s last check.
    """
    if mapping.tp_size == 4:
        if mapping.is_multi_node():
            return "it serves TP4 on one node, got TP4 over more than one node"
        return None
    if mapping.tp_size == 8:
        if not _dispatch.TWO_TRAYS_OF_4.holds(mapping):
            return (
                f"it serves TP8 over {_dispatch.TWO_TRAYS_OF_4.description}, got "
                f"{_dispatch.describe_topology(mapping)}"
            )
        return None
    return f"it serves TP4 on one node or TP8 over two nodes of four, got TP{mapping.tp_size}"


def refusal(model: torch.nn.Module) -> Optional[str]:
    """Why ``install`` cannot take the model's MoE all-reduces, or None. Changes nothing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        return "it is written for sm_103"
    mapping = model.layers[0].mapping
    reason = topology_refusal(mapping)
    if reason is not None:
        return reason
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


def fused_finalize_rows(model: torch.nn.Module) -> int:
    """The most rows a layer's fused finalize takes, which never reach its MoE all-reduce.

    The built-in MoE finalizes 1..128 rows fused when it has a finalize workspace, and up to
    its ``_finalize_ar_max_tokens`` when that workspace chunks (``finalize_chunked``, as
    ``finalize_two_shot`` provides); 0 without one. Read at install, for the log line only.
    Changes nothing.
    """
    most = 0
    for layer in model.layers:
        moe = layer.feed_forward
        workspace = getattr(moe, "_finalize_ar_workspace", None)
        if workspace is None:
            continue
        rows = FUSED_FINALIZE_ROWS
        if getattr(workspace, "finalize_chunked", None) is not None:
            rows = max(rows, int(getattr(moe, "_finalize_ar_max_tokens", FUSED_FINALIZE_ROWS)))
        most = max(most, rows)
    return most


def install(model: torch.nn.Module) -> WideMoeAllReduces:
    """Put a ``WideMoeAllReduce`` in every layer's MoE all-reduce (collective over the TP group).

    Call it on every rank once ``check`` has passed on every rank. Builds (or
    reuses, sized for ``MAX_TOKENS`` rows) the group's MNNVL workspace, the one the
    attention output projection's MNNVL all-reduce uses: two-shot buffers of
    ``2 * MAX_TOKENS * HIDDEN`` BF16 elements (1024 rows split evenly over 4 or 8 ranks:
    16 MiB at either), no larger than the built-in creates a workspace with, so taking up
    to 1024 rows grows no workspace and moves no buffer another module captured. Hands the
    switch every MoE layer with a prepare side stream (the overlap it widens while armed).
    Returns the switch ``DecodeTarget`` arms. Logs the row band, the rank count and the
    workspace it shares.
    """
    mapping = model.layers[0].mapping
    overlaps = [
        layer.feed_forward
        for layer in model.layers
        if getattr(layer.feed_forward, "_prepare_stream", None) is not None
        and hasattr(layer.feed_forward, "_prepare_overlap_max_tokens")
    ]
    shared = mapping in MNNVLAllReduce.allreduce_mnnvl_workspaces
    switch = WideMoeAllReduces(MNNVLAllReduce(mapping, torch.bfloat16), overlaps)
    needed = MNNVLAllReduce.get_required_workspace_size(
        MAX_TOKENS, HIDDEN, mapping.tp_size, torch.bfloat16
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping, torch.bfloat16, buffer_size_bytes=needed
    )
    for layer in model.layers:
        moe = layer.feed_forward
        moe.all_reduce = WideMoeAllReduce(moe.all_reduce, switch)
    widens = sum(
        moe._prepare_overlap_max_tokens == _LARGE_PREPARE_OVERLAP_MAX_TOKENS for moe in overlaps
    )
    fused = fused_finalize_rows(model)
    kept = (
        f" (the fused finalize keeps steps of up to {fused} rows, so the two-shot acts on "
        f"{fused + 1}..{MAX_TOKENS})"
        if fused >= MIN_TOKENS
        else ""
    )
    logger.info(
        f"[musespark_open] wide_moe_allreduce at {MIN_TOKENS}..{MAX_TOKENS} rows over "
        f"{mapping.tp_size} ranks ({_dispatch.tp_group_nodes(mapping)} node(s)): the MoE output "
        f"all-reduce of all {len(model.layers)} layers runs on the MNNVL two-shot on decode steps "
        f"of {MIN_TOKENS}..{MAX_TOKENS} rows{kept}, NCCL otherwise; it needs "
        f"{needed / 2**20:.1f} MiB of the {'shared' if shared else 'new'} MNNVL workspace's "
        f"{workspace['buffer_size_bytes'] / 2**20:.1f} MiB buffers; the router / pool overlap of "
        f"{widens} layers covers those steps up to {MAX_TOKENS} rows"
    )
    return switch
