# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``boundary_early_release``: dependents launched early by the kernels before two weight GEMMs.

flashinfer's split-K GEMM (``ops/gemm/dense_gemm``'s ``splitk`` kernels, which
``dense_gemms`` pins for the QKV projection) issues the TMA loads of its weight -- immutable, read by nothing
else -- before ``griddepcontrol.wait``, and loads its activation after it. A
predecessor that lets it launch only at its own exit leaves that pre-wait
window empty. This fast path moves the trigger forward in the two predecessors this
target owns, so the GEMM streams its weight while the predecessor still runs:

* QKV projection of layer ``i + 1`` <- the fused unpool gather + post-FFN
  boundary of layer ``i`` (``ops/comm/unpool_gather_boundary``): the trigger
  moves from after the kernel's last store to its entry, before its own
  ``griddepcontrol.wait``. The QKV GEMM launches once every boundary CTA is
  resident -- the unpool GEMM, which lets the boundary launch early, may still
  be finishing -- and streams its weight through the boundary's wait,
  peer polls and normalization. Only where the unpool site is fused
  (every layer but the last); with the built-in gather and boundary nothing
  changes. Nor on the layers whose output DFlash captures for its drafter
  (``spec_config.target_layer_ids``): the built-in decoder layer copies that
  output into the capture buffer right after the boundary, a plain copy
  launch between the boundary and the QKV GEMM, so the GEMM cannot start
  before the copy has completed and an early trigger buys nothing there.
* attention output projection <- the attention output's per-head norm and
  sigmoid gate. The built-in ``_attention_norm_gate_kernel`` is a plain Triton
  launch; under this fast path a decode step runs ``ops/attention/norm_gate``, the
  same kernel statement for statement, launched with programmatic dependent
  launch and triggering right after its own wait. With the built-in cuBLAS
  projection that hides only the launch latency.

Both changes are launch properties: every value is computed by the same
kernels in the same order, so this fast path is bit-identical.

One mechanism per GEMM. ``dense_gemm_prefetch`` prefetches the next projection's weight into
L2 from a side stream, and its join makes the prefetch kernel a predecessor of
the GEMM it serves: a programmatic edge in the captured graph when the GEMM is
launched with programmatic dependent launch, but the prefetch kernel never
triggers early, so the GEMM still launches only once every prefetch CTA has
exited. On an edge converted here that prefetch is dropped: the
unpooler's prefetch of the next QKV weight. It is dropped only at
the row counts whose consumer kernel loads its weight before its wait (the
split-K kernel; the ``dense_gemms`` direct kernel waits before any load), and only with
programmatic dependent launch on (``TRTLLM_ENABLE_PDL``, a precondition of
the composition).

Memory order (why an early trigger is safe here). A dependent's
``griddepcontrol.wait`` returns only once the whole primary grid has completed
and its stores are visible. The split-K GEMM reads before its wait only its
weight; the norm-gate kernel and the fused boundary read nothing before theirs.
Every kernel launched with programmatic dependent launch between two calls of
the same inbox (the boundary's per-row tags and slots) waits on its
predecessor before it completes, so call ``n + 1`` of the boundary still
starts after call ``n`` completed. With the boundary's trigger at its entry,
three grids can be in flight (unpool GEMM, boundary, QKV GEMM); a dependent
launches only once every CTA of its predecessor has started, so each earlier
grid already holds its SMs, and none of them waits on a later one. The
norm-gate kernel triggers right after its own wait.

The built-in norm-gate is a plain launch, a full barrier in every layer; this
fast path's copy is not. The post-attention hop and the fused finalize
all-reduce reuse their buffers on the condition that call ``n`` has completed
before call ``n + 1`` reads them, which the plain launches still between two
of their calls carry (the built-in router projection, the built-in routing
kernel and the replicated experts' ``ring_buddy_assign``: in the captured graph no
programmatic edge enters any of the three), together with the waits of the
PDL kernels between them. A stream fork or join is not such a barrier: when
the kernel after it is launched with programmatic dependent launch, the
captured edge is programmatic too (the prepare stream's pool GEMM after the
post-attention hop, the persistent experts after the fused pool gather). A change that makes the last
of the three plain launches PDL needs a negative control of the all-PDL chain
(a late writer and a sleeping rank, as in
``comm/test_modeling_v2_unpool_gather_boundary_trigger.py``). ``moe_prologue`` is such a change: on
the NVFP4-expert layers' decode steps of at most 8 rows none of the three runs,
and the order rests on the waits of every PDL kernel between two calls (its
router shard, the pool GEMM and its routed pool kernel among them; each waits in
every CTA before it can exit). The schedule test's transitivity census lists
those kernels per pair of calls, and its host-delay arms run that all-PDL chain
under a late rank.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ...attention import (
    MuseSparkOpenTepAttention,
    MuseSparkOpenTepDecodeAttention,
    MuseSparkOpenTepPrepDecodeAttention,
)
from ...ops.attention import norm_gate as _norm_gate

#: When the fused boundary lets the next QKV GEMM launch: at its entry.
BOUNDARY_TRIGGER = "entry"
#: When the norm-gate kernel lets the output projection launch: right after its wait.
NORM_GATE_TRIGGER = "after_wait"


class _NormGatePdl:
    """The output gate of a decode step through ``ops/attention/norm_gate``.

    A mixin for the attention classes this target re-classes to; never used
    alone. A call carrying ``norm_gate_pdl`` (a trigger name; only
    ``DecodeTarget`` passes it) runs the PDL copy of the built-in kernel where
    the built-in ``apply_output_gate`` would launch that kernel for a decode
    step (at most 8 rows); on a decode step of 33..512 rows it asks
    ``batch_dense_gemms`` (``_batch_dense_gemms``, set on the attention when that
    path is composed); every other call, prefill among them, runs the built-in
    method.
    """

    _norm_gate_trigger: Optional[str]

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        *args,
        norm_gate_pdl: Optional[str] = None,
        **kwargs,
    ):
        self._norm_gate_trigger = norm_gate_pdl
        try:
            return super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._norm_gate_trigger = None

    def apply_output_gate(self, attention_output, gate):
        trigger = self._norm_gate_trigger
        if trigger is None:
            return super().apply_output_gate(attention_output, gate)
        if not self._fits_norm_gate(attention_output, gate):
            # Larger decode steps: ``batch_dense_gemms`` (when composed) runs the PDL copy of
            # whichever kernel the built-in would launch, or declines.
            batch = getattr(self, "_batch_dense_gemms", None)
            out = None if batch is None else batch.norm_gate(self, attention_output, gate, trigger)
            return super().apply_output_gate(attention_output, gate) if out is None else out
        return _norm_gate.attention_norm_gate(
            attention_output, gate, self.o_head_norm.variance_epsilon, trigger
        )

    def _fits_norm_gate(self, attention_output, gate) -> bool:
        """Exactly when the built-in method would launch its Triton kernel, at 1-8 rows."""
        return (
            not isinstance(attention_output, Fp4QuantizedTensor)
            and attention_output.is_cuda
            and attention_output.dtype == torch.bfloat16
            and attention_output.ndim == 2
            and attention_output.stride(-1) == 1
            and 0 < attention_output.shape[0] <= _norm_gate.MAX_ROWS
            and not torch.compiler.is_compiling()
            and self.head_dim == _norm_gate.HEAD_DIM
            and attention_output.shape[1] % self.head_dim == 0
            and gate.is_cuda
            and gate.dtype == torch.bfloat16
            and gate.ndim in (2, 3)
            and gate.stride(-1) == 1
            and gate.shape[0] == attention_output.shape[0]
            and gate.numel() == attention_output.numel()
            and (gate.ndim == 2 or gate.shape[-1] == self.head_dim)
        )

    @classmethod
    def _check_norm_gate(cls, attention: MuseSparkOpenAttention) -> None:
        if attention.head_dim != _norm_gate.HEAD_DIM:
            raise ValueError(
                f"layer {attention.layer_idx}: the PDL norm-gate kernel is written for "
                f"head_dim {_norm_gate.HEAD_DIM}, got {attention.head_dim}"
            )


def _builtin_check(attention: MuseSparkOpenAttention) -> None:
    if type(attention) is not MuseSparkOpenAttention:
        raise TypeError(
            f"boundary_early_release adopts the built-in MuseSparkOpenAttention, got {type(attention).__name__}"
        )


class MuseSparkOpenTepNormGateAttention(_NormGatePdl, MuseSparkOpenAttention):
    """``boundary_early_release`` alone on an attention: the built-in attention with the PDL norm-gate."""

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        _builtin_check(attention)
        cls._check_norm_gate(attention)

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepNormGateAttention":
        cls.check(attention)
        attention.__class__ = cls
        attention._norm_gate_trigger = None
        return attention


class MuseSparkOpenTepPrepNormGateAttention(_NormGatePdl, MuseSparkOpenTepAttention):
    """``preattn_prep`` + ``boundary_early_release``: the fused prep, the built-in attention, the PDL norm-gate."""

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        MuseSparkOpenTepAttention.check(attention)
        cls._check_norm_gate(attention)

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepPrepNormGateAttention":
        cls.check(attention)
        attention.__class__ = cls
        attention._preattn_prep = None
        attention._norm_gate_trigger = None
        return attention


class MuseSparkOpenTepDecodeNormGateAttention(_NormGatePdl, MuseSparkOpenTepDecodeAttention):
    """``swa_paged_decode`` + ``boundary_early_release``: the fused decode attention, the PDL norm-gate."""

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        MuseSparkOpenTepDecodeAttention.check(attention)
        cls._check_norm_gate(attention)

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepDecodeNormGateAttention":
        cls.check(attention)
        attention.__class__ = cls
        attention._decode_attention = None
        attention._norm_gate_trigger = None
        return attention


class MuseSparkOpenTepPrepDecodeNormGateAttention(
    _NormGatePdl, MuseSparkOpenTepPrepDecodeAttention
):
    """``preattn_prep`` + ``swa_paged_decode`` + ``boundary_early_release``: prep, decode attention, norm-gate.

    Each of the three overrides different methods (``preprocess_qkv``,
    ``_attn_impl``, ``apply_output_gate``) and each ``forward`` takes its own
    keyword and hands the rest on, so the method resolution order changes no
    call of any of them.
    """

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        MuseSparkOpenTepPrepDecodeAttention.check(attention)
        cls._check_norm_gate(attention)

    @classmethod
    def adopt(
        cls, attention: MuseSparkOpenAttention
    ) -> "MuseSparkOpenTepPrepDecodeNormGateAttention":
        cls.check(attention)
        attention.__class__ = cls
        attention._preattn_prep = None
        attention._decode_attention = None
        attention._norm_gate_trigger = None
        return attention


#: The class carrying the norm-gate for each class ``modeling._attention_class`` picks.
NORM_GATE_CLASSES = {
    None: MuseSparkOpenTepNormGateAttention,
    MuseSparkOpenTepAttention: MuseSparkOpenTepPrepNormGateAttention,
    MuseSparkOpenTepDecodeAttention: MuseSparkOpenTepDecodeNormGateAttention,
    MuseSparkOpenTepPrepDecodeAttention: MuseSparkOpenTepPrepDecodeNormGateAttention,
}


def weights_before_wait_rows(kernels: Sequence[str]) -> FrozenSet[int]:
    """The row counts whose kernel (``kernels[rows - 1]``) loads its weight before its wait.

    flashinfer's split-K kernel issues the weight's TMA loads before
    ``griddepcontrol.wait``; the ``dense_gemms`` direct kernel and cuBLAS load nothing
    before theirs, so an early predecessor hides only their launch latency.
    """
    from ...ops.gemm.dense_gemm import SPLITK, parse_kernel

    return frozenset(
        rows for rows, kernel in enumerate(kernels, start=1) if parse_kernel(kernel)[0] == SPLITK
    )


def drop_converted_prefetches(model, plan, qkv: bool) -> Dict[str, Tuple[int, FrozenSet[int]]]:
    """Drop the ``dense_gemm_prefetch`` prefetch of the weights whose GEMM edge this fast path converts.

    ``qkv``: the unpooler's prefetch of the next layer's QKV weight, on every
    layer whose boundary triggers early. It is dropped only
    at the row counts whose consumer kernel loads its weight before its wait
    (``weights_before_wait_rows``; the unpooler's and QKV's calls of
    one step have the same rows), and only when the plan launches with
    programmatic dependent launch: without it the consumer cannot start early,
    and the prefetch is the only thing that streams its weight. Returns, per
    site, how many layers dropped it and at which row counts, for the log line.
    """
    if plan is None or not plan.prefetch or not plan.enable_pdl:
        return {}
    none: FrozenSet[int] = frozenset()
    rows = {"qkv": weights_before_wait_rows(plan.kernels["qkv"]) if qkv else none}
    dropped = {site: 0 for site in rows}
    layers = list(model.layers)
    for index, layer in enumerate(layers):
        if rows["qkv"] and index + 1 < len(layers):
            if getattr(getattr(layer, "add_ffn", None), "_boundary_trigger", "exit") != "exit":
                layer.feed_forward.unpooler._low_m_gemm_dispatcher.site.skip_prefetch_rows = rows[
                    "qkv"
                ]
                dropped["qkv"] += 1
    return {site: (count, rows[site]) for site, count in dropped.items() if count}


def capture_layers(model) -> FrozenSet[int]:
    """The layers whose output DFlash captures for its drafter (empty without DFlash).

    ``spec_config.target_layer_ids``, which the speculative config resolves
    from the drafter checkpoint before the model is built; the spec metadata
    captures exactly these layers (``DFlashSpecMetadata.layers_to_capture``).
    """
    spec_config = getattr(getattr(model, "model_config", None), "spec_config", None)
    return frozenset(int(index) for index in getattr(spec_config, "target_layer_ids", None) or ())


def install_boundary_triggers(
    model, captured: FrozenSet[int] = frozenset()
) -> Tuple[List[int], List[int]]:
    """Have the fused unpool boundaries that precede a QKV GEMM directly trigger it at entry.

    Only layers whose post-FFN join is the fused projection-gather class: with the built-in
    gather and boundary (``projection_gathers`` off, or its unpool site kept on the built-in
    path) there is nothing this target owns to move. A layer in ``captured``
    keeps the exit trigger: the capture copy after its boundary is a plain
    launch, so the next QKV GEMM starts only once that copy completed, and the
    layer keeps the ``dense_gemm_prefetch`` prefetch of that GEMM's weight
    (``drop_converted_prefetches`` drops it only behind an early trigger).

    Returns:
        The layers whose boundary now triggers at entry, and the fused layers
        left at exit because their output is captured.
    """
    from ..projection_gathers.gathers import MuseSparkOpenTepGatedResidual

    converted, kept = [], []
    layers = list(model.layers)
    for index, layer in enumerate(layers[:-1]):
        join = getattr(layer, "add_ffn", None)
        if isinstance(join, MuseSparkOpenTepGatedResidual) and join._gather_inbox is not None:
            if index in captured:
                kept.append(index)
            else:
                join._boundary_trigger = BOUNDARY_TRIGGER
                converted.append(index)
    return converted, kept


def _spans(rows: FrozenSet[int]) -> str:
    """``3-8`` / ``1,3-8`` for a set of row counts."""
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


def log_installed(
    layers: int,
    converted: List[int],
    kept: List[int],
    norm_gate: bool,
    dropped: Mapping[str, Tuple[int, FrozenSet[int]]],
) -> None:
    parts = [
        f"fused boundary -> QKV on {len(converted)} of {layers - 1} layer edges"
        f" (trigger {BOUNDARY_TRIGGER})"
    ]
    if kept:
        parts.append(
            f"DFlash capture layers {kept} keep the exit trigger and the QKV weight prefetch "
            "(a plain capture copy precedes their QKV)"
        )
    if norm_gate:
        parts.append(f"PDL norm-gate -> o_proj on {layers} layers (trigger {NORM_GATE_TRIGGER})")
    if dropped:
        parts.append(
            "QKV weight prefetch dropped on converted edges: "
            + ", ".join(
                f"{site} x{count} at rows {_spans(rows)}"
                for site, (count, rows) in sorted(dropped.items())
            )
        )
    logger.info(
        f"[musespark_open] boundary_early_release: early dependent launches ({'; '.join(parts)})"
    )
