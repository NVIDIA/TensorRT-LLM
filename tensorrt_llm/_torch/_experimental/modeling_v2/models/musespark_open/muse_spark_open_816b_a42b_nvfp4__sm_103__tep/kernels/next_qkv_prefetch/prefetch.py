# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""next_qkv_prefetch: the next layer's QKV weight is asked of the L2 while the MoE epilogue runs.

At 8 rows the QKV projection (flashinfer split-K (64,8,4,12), ``[8, 8192] x
[2304, 8192]^T``) is launched early by the boundary early release
(``boundary_early_release``): the post-FFN boundary of the previous layer triggers it at entry, and the GEMM
stages the first 12 of each CTA's 16 weight k-tiles before its dependency wait.
Behind the MoE epilogue (expert kernel -> fused finalize all-reduce -> unpool
GEMM -> boundary) the weight still comes from HBM, and the QKV stage after the
boundary is the one dense projection whose cost moves with L2 residency.

What this fast path does, on decode steps of the row counts where the boundary
early release launches QKV early (its kernel stages its weight before its wait,
and the dense GEMM prefetch's unpooler prefetch of the same weight is dropped there):

* Each layer ``l < L - 1`` asks the L2 for layer ``l + 1``'s QKV weight from a
  side stream, forked right after its expert call is launched (the ring-buddy
  stacked call: the persistent expert kernel on the NVFP4 layers, the BF16 leaf
  on layer 0) and before the fused finalize; the last layer forks none. Layer
  0's QKV weight is asked for at the start of the decode forward (the embedding
  chain runs in front of it).
* The fork is the dense GEMM plan's named fork ``qkv_prefetch_qkv`` (``dense_gemms.NamedFork``), joined
  only by the QKV dispatchers: the unpooler's dispatcher and the boundary in
  between do not wait for it.
* The prefetch asks for the whole weight (one bulk prefetch per 32 KiB).

Graph structure (captured decode graph). The fork records an event on the
current stream right after the expert kernel's launch and makes the fork's own
stream wait on it, so the prefetch kernel ``P(l + 1)`` is a plain launch with a
full edge from the expert kernel: it starts once the expert kernel completed
and runs beside the finalize. The QKV dispatcher's join makes ``P(l + 1)`` a
second predecessor of ``QKV(l + 1)`` beside the boundary; because QKV is
launched with programmatic dependent launch, both edges are programmatic.
``P`` never triggers its dependents, so its trigger is the exit of its last
CTA: QKV still launches at the boundary's entry trigger once every ``P`` CTA
has exited, and its dependency wait returns once the boundary and ``P``
completed. ``P`` writes nothing anyone reads and reads no memory, so no value
of the step depends on it.

No ``P`` kernel is resident when a full-attention layer's FMHA launches, for
two reasons, each sufficient alone: (1) every CTA of ``P(l + 1)`` has exited
before ``QKV(l + 1)`` launches, and every later kernel of layer ``l + 1``
launches after QKV; (2) ``P(l + 1)`` completes before ``QKV(l + 1)``'s wait
returns, which is before QKV triggers its dependents: its weight warp triggers
after its last k-tile, and each CTA's K slice has more k-tiles than the ring
has stages, so the last ones are issued only after the wait. ``install`` forks
only at the row counts whose QKV kernel has such post-wait k-tiles
(``dense_gemm.post_wait_columns`` is not ``None``), so (2) holds whatever
kernel the plan names. And ``P(l + 1)`` starts after the
expert kernel of layer ``l`` completed, after layer ``l``'s FMHA.
"""

from __future__ import annotations

from typing import Callable, FrozenSet, Optional

import torch

from tensorrt_llm.logger import logger

from ...ops.gemm.dense_gemm import QKV_SHAPE, l2_prefetch, post_wait_columns

#: The fork's name in the dense GEMM plan and the site whose dispatchers join it.
FORK_NAME = "qkv_prefetch_qkv"
JOIN_ROLE = "qkv"


class QkvPrefetch:
    """The prefetch forks of one model: what to ask for, at which row counts, on which fork.

    ``rows``: the row counts whose QKV kernel stages its weight before its wait
    (and whose unpooler prefetch the boundary early release dropped). ``fork``: the plan's named fork.
    """

    def __init__(self, fork, rows: FrozenSet[int]) -> None:
        self.fork = fork
        self.rows = frozenset(rows)

    def launches(self, weight: torch.Tensor, rows: int) -> Callable[[], None]:
        """The launch the fork runs for ``weight`` at ``rows``: the whole weight."""
        fork = self.fork
        return lambda: l2_prefetch(weight, fork.sink)

    #: Layer 0's QKV weight, asked for at the start of the decode forward (``graph_start``).
    first_weight: Optional[Callable[[], torch.Tensor]] = None

    def ask(self, weight: torch.Tensor, rows: int) -> None:
        """Fork the prefetch of ``weight`` after the current stream's work, at ``rows``."""
        if rows in self.rows:
            # The launch is built after the fork made its sink (first eager fork).
            self.fork.fork(lambda: self.launches(weight, rows)(), device=weight.device)

    def graph_start(self, rows: int) -> None:
        """Layer 0's prefetch, forked by the decode forward before its embedding."""
        if self.first_weight is not None:
            self.ask(self.first_weight(), rows)


class AfterExperts:
    """Called by the ring-buddy stacked expert call right after it launched the experts (``ring_buddy``)."""

    def __init__(self, prefetch: QkvPrefetch, next_weight: Callable[[], torch.Tensor]) -> None:
        self.prefetch = prefetch
        self.next_weight = next_weight

    def __call__(self, rows: int) -> None:
        self.prefetch.ask(self.next_weight(), rows)


def converted_layers(model) -> FrozenSet[int]:
    """The layers whose post-FFN boundary the boundary early release set to release the next QKV at entry."""
    layers = list(model.layers)
    return frozenset(
        index
        for index, layer in enumerate(layers[:-1])
        if getattr(getattr(layer, "add_ffn", None), "_boundary_trigger", "exit") != "exit"
    )


def fork_rows(kernels) -> FrozenSet[int]:
    """The row counts at which a prefetch is forked: those whose QKV kernel (``kernels[rows -
    1]``) stages its weight before its wait and loads its last k-tiles only after it."""
    from ..boundary_early_release.triggers import weights_before_wait_rows

    return frozenset(
        rows
        for rows in weights_before_wait_rows(kernels)
        if post_wait_columns(kernels[rows - 1], QKV_SHAPE[0]) is not None
    )


def install(model, plan) -> QkvPrefetch:
    """Register the fork, give every layer whose next QKV edge the early release converted its hook.

    Runs after ``dense_gemms`` (whose plan carries the fork and whose QKV
    dispatchers join it) and ``boundary_early_release`` (which converts the
    edges). Layer ``l``'s MoE module gets ``_qkv_prefetch_after_experts`` (read
    by ``ring_buddy`` at call time, so the replicated experts' later install
    needs nothing from here); a layer whose next QKV edge keeps the dense GEMM
    prefetch's unpooler prefetch (a capture layer without ``capture_fold``) gets none.

    Raises:
        ValueError: if the plan does not launch with programmatic dependent
            launch, if no QKV kernel stages its weight before its wait and
            loads its last k-tiles after it (``fork_rows``), or if the boundary
            early release converted no QKV edge.
    """
    if plan is None or not plan.enable_pdl:
        raise ValueError(
            "[musespark_open] next_qkv_prefetch prefetches for the QKV GEMM that boundary_early_release "
            "launches early, which needs the dense_gemms plan with programmatic dependent launch on"
        )
    rows = fork_rows(plan.kernels["qkv"])
    if not rows:
        raise ValueError(
            "[musespark_open] next_qkv_prefetch: no QKV kernel of the dense_gemms plan stages its weight "
            "before its wait and loads its last k-tiles after it"
        )
    converted = converted_layers(model)
    if not converted:
        raise ValueError(
            "[musespark_open] next_qkv_prefetch: boundary_early_release converted no boundary -> QKV edge"
        )
    fork = plan.add_fork(FORK_NAME, JOIN_ROLE)
    prefetch = QkvPrefetch(fork, rows)
    layers = list(model.layers)
    for index in sorted(converted):
        following = layers[index + 1].self_attn
        layers[index].feed_forward._qkv_prefetch_after_experts = AfterExperts(
            prefetch, lambda a=following: a.qkv_proj.weight
        )
    prefetch.first_weight = lambda a=layers[0].self_attn: a.qkv_proj.weight
    logger.info(
        "[musespark_open] next_qkv_prefetch: the next layer's QKV weight L2-prefetched (full) on a side "
        f"stream forked after the expert launch on {len(converted)} of {len(layers) - 1} "
        f"layers, layer 0's at the start of the decode forward; joined at the QKV GEMM; "
        f"rows {sorted(rows)}"
    )
    return prefetch
