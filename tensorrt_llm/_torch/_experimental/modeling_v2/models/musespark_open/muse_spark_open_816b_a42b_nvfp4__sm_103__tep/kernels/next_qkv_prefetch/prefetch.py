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
  side stream, forked right after its expert call is launched (whatever expert
  call the layer runs: the persistent expert kernel on the NVFP4 layers, the BF16
  leaf on layer 0, or the built-in TRTLLM-gen call) and before the fused finalize;
  the last layer forks none. Layer 0's QKV weight is asked for at the start of the
  decode forward (the embedding chain runs in front of it). The fork is made by
  an outermost wrapper of the experts' ``forward`` (``ExpertsForward``, installed
  by ``wrap_expert_forwards`` once the expert paths are installed), so it does not
  depend on ``replicated_experts`` and runs at every tensor-parallel size.
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

Decode steps of two to four requests (9..32 rows, ``wide_dense_gemms``). Their QKV
kernels are the plan's wide table (``DenseGemmPlan.wide["qkv"]``: split-K 2 at
16 / 11 and 32 / 9 rows / stages, 32 k-tiles per CTA), and the fused boundary in
front of them triggers at entry on the same converted layers. By default these
rows keep the dense GEMM prefetch's unpooler prefetch of the next QKV weight,
whose join holds the early-released QKV launch until that prefetch has exited.
``install``'s ``wide`` setting (``check_wide``) moves it:

* ``WIDE_OFF`` (the default): the row sets above, rows 1..8 only;
* ``WIDE_MOVE`` (the ``wide_qkv_prefetch`` lever): the wide rows whose QKV kernel stages its weight before its wait and
  loads its last k-tiles after it (``wide_fork_rows``) fork the prefetch after
  the experts, as above, and ``boundary_early_release`` drops the unpooler's
  prefetch at the same rows (``drop_rows``), on the converted layers only;
* ``WIDE_DROP`` (a diagnostic, reached by direct calls only): the unpooler's
  prefetch is dropped at those rows and nothing is forked after the experts (the
  QKV kernel streams its weight from HBM before its wait).

The two arguments for the absence of a resident ``P`` at an FMHA launch hold at
these rows unchanged: the wide QKV calls run through the same dispatcher, whose
join makes ``P(l + 1)`` a predecessor of ``QKV(l + 1)``, and ``wide_fork_rows``
keeps only kernels with post-wait k-tiles. Either setting changes only when the
L2 is asked for an immutable weight, never a value.
"""

from __future__ import annotations

from typing import Callable, FrozenSet, Optional

import torch

from tensorrt_llm.logger import logger

from ...ops.gemm.dense_gemm import MAX_ROWS, QKV_SHAPE, l2_prefetch, post_wait_columns

#: The fork's name in the dense GEMM plan and the site whose dispatchers join it.
FORK_NAME = "qkv_prefetch_qkv"
JOIN_ROLE = "qkv"

#: Where the next-QKV prefetch of a 9..32-row decode step runs (``check_wide``): the
#: unpooler's prefetch (default), moved after the experts, dropped.
WIDE_OFF, WIDE_MOVE, WIDE_DROP = "off", "move", "drop"


def check_wide(wide: str) -> str:
    """``wide`` when it is ``WIDE_OFF``, ``WIDE_MOVE`` or ``WIDE_DROP``.

    Raises:
        ValueError: on any other value.
    """
    if wide in (WIDE_OFF, WIDE_MOVE, WIDE_DROP):
        return wide
    raise ValueError(
        f"[musespark_open] next_qkv_prefetch: the wide setting must be one of "
        f"{[WIDE_OFF, WIDE_MOVE, WIDE_DROP]}, got {wide!r}"
    )


def wide_fork_rows(plan) -> FrozenSet[int]:
    """The 9..32 row counts whose wide QKV kernel (``plan.wide["qkv"][rows - 9]``) stages its
    weight before its wait and loads its last k-tiles after it; empty without a wide table."""
    from ..boundary_early_release.triggers import weights_before_wait_rows

    kernels = (getattr(plan, "wide", None) or {}).get("qkv")
    if not kernels:
        return frozenset()
    return frozenset(
        MAX_ROWS + index
        for index in weights_before_wait_rows(kernels)
        if post_wait_columns(kernels[index - 1], QKV_SHAPE[0]) is not None
    )


def drop_rows(plan, wide: str = WIDE_OFF) -> FrozenSet[int]:
    """The 9..32 row counts at which ``boundary_early_release`` drops the unpooler's prefetch of
    the next QKV weight on a converted layer: ``wide_fork_rows`` unless ``wide`` is off.

    Raises:
        ValueError: for a ``wide`` ``check_wide`` refuses.
    """
    return frozenset() if check_wide(wide) == WIDE_OFF else wide_fork_rows(plan)


def _spans(rows: FrozenSet[int]) -> str:
    """``9-32`` / ``4-8,16`` for a set of row counts; ``none`` when empty."""
    if not rows:
        return "none"
    spans, ordered = [], sorted(rows)
    start = previous = ordered[0]
    for value in ordered[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        spans.append(f"{start}" if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(spans)


class QkvPrefetch:
    """The prefetch forks of one model: what to ask for, at which row counts, on which fork.

    ``rows``: the row counts whose QKV kernel stages its weight before its wait
    (and whose unpooler prefetch the boundary early release dropped). ``fork``: the plan's named fork.
    """

    def __init__(self, fork, rows: FrozenSet[int]) -> None:
        self.fork = fork
        self.rows = frozenset(rows)
        #: The wide row counts (above 8) whose first fork was logged.
        self._logged_wide: FrozenSet[int] = frozenset()

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
            forked = self.fork.fork(lambda: self.launches(weight, rows)(), device=weight.device)
            if forked and rows > MAX_ROWS and rows not in self._logged_wide:
                self._logged_wide = self._logged_wide | {rows}
                logger.info(
                    f"[musespark_open] next_qkv_prefetch: a {rows}-row decode step forks the next QKV "
                    "weight's L2 prefetch after the expert launch"
                )

    def graph_start(self, rows: int) -> None:
        """Layer 0's prefetch, forked by the decode forward before its embedding."""
        if self.first_weight is not None:
            self.ask(self.first_weight(), rows)


class AfterExperts:
    """Called by ``ExpertsForward`` right after the layer's expert call launched the experts."""

    def __init__(self, prefetch: QkvPrefetch, next_weight: Callable[[], torch.Tensor]) -> None:
        self.prefetch = prefetch
        self.next_weight = next_weight

    def __call__(self, rows: int) -> None:
        self.prefetch.ask(self.next_weight(), rows)


class ExpertsForward:
    """A MoE layer's ``experts.forward``: the expert forward installed under it, then the hook.

    The outermost wrapper of the experts' forward (``wrap_expert_forwards``): it
    calls ``inner`` (the built-in forward, or the stacked call ``ring_buddy``
    patched in, whichever the layer runs) and, on a ``do_finalize=False`` call,
    the layer's ``_qkv_prefetch_after_experts`` with the call's row count, so the
    fork follows the expert launches on the current stream.
    """

    def __init__(self, inner: Callable, moe) -> None:
        self.inner = inner
        self.moe = moe

    def __call__(self, x, router_logits, *args, do_finalize=True, **kwargs):
        outputs = self.inner(x, router_logits, *args, do_finalize=do_finalize, **kwargs)
        if not do_finalize:
            after = getattr(self.moe, "_qkv_prefetch_after_experts", None)
            if after is not None:
                after(int(x.shape[0]))
        return outputs


def wrap_expert_forwards(model) -> int:
    """Give every layer with a prefetch hook the ``ExpertsForward`` wrapper; returns how many.

    Call it after every expert path has patched ``experts.forward``
    (``weights.install_loaded_features`` does, last), so the wrapper is
    outermost. Idempotent: a layer already wrapped keeps its wrapper.
    """
    wrapped = 0
    for layer in model.layers:
        moe = layer.feed_forward
        if getattr(moe, "_qkv_prefetch_after_experts", None) is None:
            continue
        experts = moe.experts
        if isinstance(vars(experts).get("forward"), ExpertsForward):
            continue
        experts.forward = ExpertsForward(experts.forward, moe)
        wrapped += 1
    return wrapped


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


def install(model, plan, wide: str = WIDE_OFF) -> QkvPrefetch:
    """Register the fork, give every layer whose next QKV edge the early release converted its hook.

    Runs after ``dense_gemms`` (whose plan carries the fork and whose QKV
    dispatchers join it) and ``boundary_early_release`` (which converts the
    edges). Layer ``l``'s MoE module gets ``_qkv_prefetch_after_experts``, read
    at call time by the ``ExpertsForward`` wrapper that ``wrap_expert_forwards``
    puts around the experts' forward once the weights and the expert paths are
    installed; a layer whose next QKV edge keeps the dense GEMM prefetch's
    unpooler prefetch (a capture layer without ``capture_fold``) gets none.
    ``wide`` places the prefetch of the 9..32-row steps (module docstring); the
    caller passes the same setting to ``drop_converted_prefetches``.

    Raises:
        ValueError: if the plan does not launch with programmatic dependent
            launch, if no QKV kernel stages its weight before its wait and
            loads its last k-tiles after it (``fork_rows``), if the boundary
            early release converted no QKV edge, or for a ``wide`` ``check_wide``
            refuses.
    """
    setting = check_wide(wide)
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
    wide = wide_fork_rows(plan)
    fork = plan.add_fork(FORK_NAME, JOIN_ROLE)
    prefetch = QkvPrefetch(fork, rows | wide if setting == WIDE_MOVE else rows)
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
    if setting != WIDE_OFF:
        if not wide:
            logger.info(
                f"[musespark_open] next_qkv_prefetch: wide setting {setting}: the dense_gemms plan has no "
                "wide QKV kernel that stages its weight before its wait, so the 9..32-row steps are "
                "unchanged"
            )
        elif setting == WIDE_MOVE:
            logger.info(
                f"[musespark_open] next_qkv_prefetch: wide_qkv_prefetch: wide rows {_spans(wide)} forked "
                f"after the experts on {len(converted)} layers; the unpooler's prefetch of the next "
                "QKV weight dropped there on the same layers"
            )
        else:
            logger.info(
                f"[musespark_open] next_qkv_prefetch: wide setting {setting}: wide rows {_spans(wide)}: the "
                f"unpooler's prefetch of the next QKV weight dropped on {len(converted)} layers, "
                "nothing forked after the experts"
            )
    return prefetch
