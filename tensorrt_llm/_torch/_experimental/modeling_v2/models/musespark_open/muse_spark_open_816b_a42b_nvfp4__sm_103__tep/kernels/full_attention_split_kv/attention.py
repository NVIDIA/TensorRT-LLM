# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The full-attention layers' decode FMHA as ``full_attention_split_kv`` partials,
merged in the attention norm-gate (``combine_norm_gate``).

Under ``full_attention_prep_append`` a decode step of one sequence and at most 8 rows runs, on each of the 16
full-attention (NoPE) layers, the folded prep (``preattn_prep_append``: the step's K/V into
the paged FP8 cache, Q, the FMHA scale words), then the built-in attention, whose generation
preprocess stands down for the prep (``attention_forward.eliding``) and which then launches
flashinfer's trtllm-gen decode FMHA. This path keeps that call chain and replaces only its
last launch: while a full layer's built-in attention runs after a folded prep,
``flashinfer`` as seen by ``flashinfer_trtllm_gen`` is a stand-in whose
``decode.trtllm_batch_decode_with_kv_cache`` checks the call against the plain case the
kernel is written for and then launches ``full_attention_split_kv`` on the prep's BF16 fused operand
and the call's own cache view, block table and sequence lengths, instead of the FMHA. The
kernel leaves one partial per (row, split); the attention output tensor of the call is not
written. The layer's output gate (``apply_output_gate``) then runs ``combine_norm_gate``,
which merges the partials of the splits that hold keys (keyed by the call's ``seq_lens``) and
applies the built-in norm-gate, in place of the norm-gate on the attention output, with
the ``boundary_early_release`` norm-gate trigger when the layer carries that class.

Any call that is not the plain case -- the built-in preprocess ran instead of standing down,
a sliding window, sinks, a scaled or multi-token query layout, another cache layout, a
context beyond the kernel's per-split page limit, a build not yet compiled while a CUDA
graph is captured -- launches the real FMHA with the same arguments, which is then exactly
the ``full_attention_prep_append`` path, and is recorded with its reason. The partials of a call are consumed by
the same forward's output gate; a forward that ended with partials pending raises, since
its attention output was never written.

Numerics: not bit-identical to the built-in FMHA (``full_attention_split_kv``'s module docstring: Q as
two E4M3 terms, P in E4M3 against a running maximum, FP16 partials, a fixed-order merge);
deterministic; no cache byte changes. ``SplitKvCensus`` logs, once per captured decode graph, how
many full-attention layers run the partials and merge, and warns about the ones that do not
at a step the fold covers.

Rank shapes (``install``): a tep8 rank (16 query / 2 KV heads) launches the kernel's one-pair
build on one CTA per SM; a tep4 rank (32 query / 4 KV heads) launches its two-pair build on
``SMs / 2`` splits per pair, again one CTA per SM. Any other rank shape is refused.

The spill-free build (``install_spill_free``, after ``install``): those layers launch the
kernel with ``rescale_cols = full_attention_split_kv.SPILL_FREE_RESCALE_COLS``: the rare O
rescale runs in 16-column chunks, so the softmax warps do not spill part of S on every tile.
Bit-identical to the default build.
"""

from __future__ import annotations

import collections
import functools
from dataclasses import dataclass, field
from typing import ClassVar, Deque, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale
from tensorrt_llm.logger import logger

from ...attention_forward import decode_reachable, intercepting
from ...ops.attention import combine_norm_gate as _cng
from ...ops.attention import full_attention_split_kv as _v4
from ..full_attention_prep_append import attention as _prep_append

#: The keyword arguments ``run_generation`` passes to the decode FMHA (and no others).
DECODE_KWARGS = frozenset(
    {
        "query",
        "kv_cache",
        "workspace_buffer",
        "block_tables",
        "seq_lens",
        "max_seq_len",
        "bmm1_scale",
        "bmm2_scale",
        "window_left",
        "out",
        "sinks",
        "kv_layout",
        "enable_pdl",
        "backend",
        "q_len_per_req",
        "max_q_len",
        "cum_seq_lens_q",
        "kv_cache_sf",
        "uses_shared_paged_kv_idx",
        "bmm1_scale_log2",
        "multi_ctas_kv_counter_buffer",
    }
)
#: The softmax scale the kernel applies: 1 / sqrt(64) (``q_scaling`` 1).
SOFTMAX_SCALE = 0.125
#: The op's build: no page prefetch before the dependency wait, TMA loads evict-first.
PREFETCH = False
EVICT_FIRST = True

#: ``SplitKvCensus`` outcome kinds.
V4 = "full_attention_split_kv"
NO_FOLD = "no folded prep"
BUILTIN = "built-in FMHA"
NOT_RUN = "not run in this forward"


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class SplitKvCall:
    """One full layer's decode call: what the stand-in needs, and what it did."""

    #: The fast path the stand-in names when the call runs the built-in FMHA.
    PATH: ClassVar[str] = "full_attention_split_kv"
    folded: "_prep_append.FoldedPreprocess"
    kv_scale: torch.Tensor
    splits: int
    rescale_cols: int = _v4.HEAD_DIM
    kv_heads: int = _v4.KV_HEADS
    launched: bool = False
    fallback: Optional[str] = None
    pending: Optional[tuple] = None

    def decline(self, kwargs: dict) -> Optional[str]:
        """Why this decode FMHA call is not the plain case ``full_attention_split_kv`` serves, or ``None``.

        Host-side reads only.
        """
        folded = self.folded
        if not folded.consumed or getattr(folded, "fallback", None) is not None:
            return "the built-in preprocess ran"
        if set(kwargs) != DECODE_KWARGS:
            return f"the FMHA was called with {sorted(kwargs)}"
        pool = kwargs["kv_cache"][0] if isinstance(kwargs["kv_cache"], tuple) else None
        tables = kwargs["block_tables"]
        out = kwargs["out"]
        seq_lens = kwargs["seq_lens"]
        rows = folded.rows
        requests = folded.requests
        kv_heads = self.kv_heads
        checks = (
            (pool is None or kwargs["kv_cache"][1] is not pool, "not one K/V pool"),
            (kwargs["window_left"] != -1, "a sliding window"),
            (kwargs["sinks"] is not None, "attention sinks"),
            (kwargs["kv_cache_sf"] is not None, "a scaled KV cache"),
            (kwargs["kv_layout"] != "HND", "another KV layout"),
            (kwargs["backend"] != "trtllm-gen", "another backend"),
            (kwargs["q_len_per_req"] != rows // requests, "another query length"),
            (
                kwargs["max_q_len"] is not None or kwargs["cum_seq_lens_q"] is not None,
                "a multi-token query layout",
            ),
            (kwargs["uses_shared_paged_kv_idx"], "shared K/V page ids"),
            (folded.bmm1_host != SOFTMAX_SCALE, "another softmax scale"),
            (
                not isinstance(seq_lens, torch.Tensor)
                or seq_lens.data_ptr() != folded.seq_lens_ptr,
                "other sequence lengths",
            ),
            (
                pool is not None
                and (
                    pool.dtype != torch.float8_e4m3fn
                    or pool.dim() != 4
                    or tuple(pool.shape[1:]) != (kv_heads, _v4.PAGE, _v4.HEAD_DIM)
                    or not pool.is_contiguous()
                ),
                "another cache view",
            ),
            (
                tables.dim() != 3 or tables.shape[0] != requests or tables.shape[1] != 2,
                "not one sequence's K/V page table"
                if requests == 1
                else "not the step's sequences' K/V page tables",
            ),
            (
                requests == 1
                and tables.dim() == 3
                and -(-tables.shape[2] // self.splits) > _v4.cta_pages(kv_heads),
                "more pages per split than the kernel stages",
            ),
            (
                requests > 1
                and tables.dim() == 3
                and (
                    self.splits <= requests
                    or -(-requests * tables.shape[2] // (self.splits - requests))
                    > _v4.MULTI_CTA_PAGES_FACTOR * _v4.cta_pages(kv_heads)
                    or tables.stride(0) != 2 * tables.stride(1)
                    or tables.stride(-1) != 1
                ),
                "more pages per split than the multi-sequence kernel stages",
            ),
            (
                out.dtype != torch.bfloat16
                or out.numel() != rows * _v4.q_heads(kv_heads) * _v4.HEAD_DIM
                or not out.is_contiguous(),
                "another output",
            ),
            (folded.qkv_out.shape[1] != _v4.qkv_width(kv_heads), "another fused operand"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        key = _v4.compiled_key(
            folded.qkv_out.device.index,
            bool(kwargs["enable_pdl"]),
            PREFETCH,
            EVICT_FIRST,
            self.rescale_cols,
            kv_heads,
            requests,
        )
        if _capturing() and key not in _v4._compiled_keys:
            return "full_attention_split_kv was not compiled before this capture"
        return None

    def launch(self, kwargs: dict) -> None:
        folded = self.folded
        if folded.requests == 1:
            part_o, part_lse = _v4.full_attention_split_kv(
                folded.qkv_out,
                kwargs["kv_cache"][0],
                kwargs["block_tables"][0],
                kwargs["seq_lens"],
                self.kv_scale,
                self.splits,
                bool(kwargs["enable_pdl"]),
                prefetch=PREFETCH,
                evict_first=EVICT_FIRST,
                rescale_cols=self.rescale_cols,
            )
        else:
            part_o, part_lse = _v4.full_attention_split_kv_multi(
                folded.qkv_out,
                kwargs["kv_cache"][0],
                kwargs["block_tables"],
                kwargs["seq_lens"],
                self.kv_scale,
                self.splits,
                bool(kwargs["enable_pdl"]),
                evict_first=EVICT_FIRST,
                rescale_cols=self.rescale_cols,
            )
        self.launched = True
        self.pending = (
            part_o,
            part_lse,
            folded.rows,
            kwargs["out"].data_ptr(),
            kwargs["seq_lens"],
            folded.requests,
        )


@dataclass
class SplitKvCensus:
    """Which full-attention layers ran the partials and merge in one forward, logged once per
    captured decode graph (as ``full_attention_prep_append.attention.FoldCensus`` does for the fold)."""

    layers: Tuple[int, ...]
    outcomes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    captures: int = 0
    #: ``(capture number, layers on full_attention_split_kv, layers, level, line)`` of the latest captures.
    reports: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=64))

    def begin(self) -> None:
        self.outcomes = {}

    def record(self, layer_idx: int, kind: str, detail: str = "") -> None:
        self.outcomes[layer_idx] = (kind, detail)

    def end(self) -> Optional[tuple]:
        if not _capturing():
            return None
        self.captures += 1
        groups: Dict[Tuple[str, str], list] = {}
        for layer_idx in self.layers:
            groups.setdefault(self.outcomes.get(layer_idx, (NOT_RUN, "")), []).append(layer_idx)
        served = len(groups.get((V4, ""), ()))
        line = (
            f"[musespark_open] full_attention_split_kv: CUDA graph capture {self.captures}: "
            f"{served} of {len(self.layers)} "
            "full-attention layers run full_attention_split_kv and combine_norm_gate"
        )
        rest = [
            f"{kind}{f' ({detail})' if detail else ''} on layers {', '.join(map(str, layers))}"
            for (kind, detail), layers in sorted(groups.items())
            if kind != V4
        ]
        if rest:
            line += "; " + "; ".join(rest)
        if any(kind == BUILTIN for kind, _ in groups):
            level = "warning"
            logger.warning(line + " -- these layers run the built-in FMHA in this graph")
        elif all(kind == NO_FOLD for kind, _ in groups):
            level = "silent"
        else:
            level = "info"
            logger.info(line)
        report = (self.captures, served, len(self.layers), level, line)
        self.reports.append(report)
        return report


class FullAttnSplitKvAttention:
    """Mixin re-classed in front of the prep-append class on the full-attention layers.

    A call that ``full_attention_prep_append`` folded runs the built-in attention with the decode FMHA call
    intercepted (``intercepting``): on the plain case the stand-in launches ``full_attention_split_kv``
    and the output gate merges its partials (``combine_norm_gate``); otherwise the real FMHA
    runs. Any other call is exactly the class behind it. Adds two plain attributes: the
    partials of the current forward and the model's ``SplitKvCensus``.
    """

    _split_kv_pending: Optional[tuple] = None
    _split_kv_census: Optional[SplitKvCensus] = None
    _split_kv_splits: int = 0
    #: The kernel build's ``rescale_cols``: 64 by default, 16 in the spill-free build.
    _split_kv_rescale_cols: int = _v4.HEAD_DIM
    #: The rank's KV heads, which pick the kernel build: 2 at tep8, 4 at tep4.
    _split_kv_kv_heads: int = _v4.KV_HEADS

    def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        census = self._split_kv_census
        if census is not None:
            if self.layer_idx == census.layers[0]:
                census.begin()
            census.record(self.layer_idx, NO_FOLD)
        self._split_kv_pending = None
        result = super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        pending, self._split_kv_pending = self._split_kv_pending, None
        if pending is not None:
            raise RuntimeError(
                f"[musespark_open] full_attention_split_kv: layer {self.layer_idx} left "
                "full_attention_split_kv partials unmerged; its attention output was never written"
            )
        if census is not None and self.layer_idx == census.layers[-1]:
            census.end()
        return result

    def _attn_impl(self, *args, **kwargs):
        folded = self._prep_append_folded
        if folded is None:
            return super()._attn_impl(*args, **kwargs)
        call = SplitKvCall(
            folded=folded,
            kv_scale=self.attn.kv_scale_orig_quant,
            splits=self._split_kv_splits,
            rescale_cols=self._split_kv_rescale_cols,
            kv_heads=self._split_kv_kv_heads,
        )
        with intercepting(call):
            result = super()._attn_impl(*args, **kwargs)
        census = self._split_kv_census
        if call.launched:
            self._split_kv_pending = call.pending
            if census is not None:
                census.record(self.layer_idx, V4)
        elif census is not None:
            census.record(self.layer_idx, BUILTIN, call.fallback or "the FMHA was not called")
        return result

    def apply_output_gate(self, attention_output, gate):
        pending = self._split_kv_pending
        if pending is None:
            return super().apply_output_gate(attention_output, gate)
        self._split_kv_pending = None
        part_o, part_lse, rows, out_ptr, seq_lens, requests = pending
        if (
            gate is None
            or not isinstance(attention_output, torch.Tensor)
            or attention_output.data_ptr() != out_ptr
            or attention_output.shape[0] != rows
        ):
            raise RuntimeError(
                f"[musespark_open] full_attention_split_kv: layer {self.layer_idx}'s output gate did "
                "not get the attention output whose partials are pending"
            )
        trigger = getattr(self, "_norm_gate_trigger", None) or "exit"
        return _cng.combine_norm_gate(
            part_o,
            part_lse,
            rows,
            gate,
            self.o_head_norm.variance_epsilon,
            trigger,
            seq_lens=seq_lens,
            heads=_v4.q_heads(self._split_kv_kv_heads),
            **({"requests": requests} if requests > 1 else {}),
        )


@functools.lru_cache(maxsize=None)
def _split_kv_class(base: type) -> type:
    """``base`` with ``FullAttnSplitKvAttention`` in front, one class per base."""
    return type(f"{base.__name__}V4", (FullAttnSplitKvAttention, base), {})


def rank_tp(attention) -> Optional[int]:
    """The tensor-parallel size whose kernel build serves this attention's per-rank heads: 8
    for 16 query / 2 KV heads, 4 for 32 query / 4 KV heads, else None (no build)."""
    for tp, kv_heads in _v4.RANK_KV_HEADS.items():
        if attention.num_key_value_heads == kv_heads and attention.num_heads == _v4.q_heads(
            kv_heads
        ):
            return tp
    return None


def install(model: torch.nn.Module) -> int:
    """Re-class every full-attention layer's attention in front of the prep-append class.

    Run after ``full_attention_prep_append.attention.install`` and its options. Every layer is checked
    before any is re-classed. Returns the number of layers re-classed.

    Raises:
        ValueError: if ``intercepting`` would not reach the built-in decode FMHA
            (``decode_reachable``), the device is not sm_103 or sm_107, or
            a full-attention layer does not carry the prep-append mixin, has no output gate, or is
            not a tep8 rank's (16 query / 2 KV heads) or a tep4 rank's (32 query / 4 KV heads)
            heads of width 64 with the plain softmax scale, or the layers differ in rank shape.
    """
    reason = decode_reachable()
    if reason is not None:
        raise ValueError(
            f"[musespark_open] full_attention_split_kv cannot replace the full-attention decode FMHA: "
            f"{reason}"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (10, 7),
    ):
        raise ValueError(
            "[musespark_open] the full_attention_split_kv kernel is built for sm_103 and sm_107 only"
        )
    full = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = []
    for attention in full:
        reasons = [
            why
            for failed, why in (
                (
                    not isinstance(attention, _prep_append.FullAppendAttention),
                    "full_attention_prep_append's folded prep is not installed",
                ),
                (not getattr(attention, "attn_output_gate", False), "it has no output gate"),
                (attention.head_dim != _v4.HEAD_DIM, f"head_dim is not {_v4.HEAD_DIM}"),
                (
                    rank_tp(attention) is None,
                    f"{attention.num_heads} query / {attention.num_key_value_heads} KV heads per "
                    "rank: neither a tep8 rank's (16 / 2) nor a tep4 rank's (32 / 4)",
                ),
                (
                    get_bmm1_scale(attention.attn) != SOFTMAX_SCALE,
                    "its softmax scale is not 1 / sqrt(64)",
                ),
            )
            if failed
        ]
        if reasons:
            refused.append(f"layer {attention.layer_idx}: " + "; ".join(reasons))
    tps = {rank_tp(attention) for attention in full}
    if not refused and len(tps) > 1:
        refused.append(f"the layers mix rank shapes of tp {sorted(tps)}")
    if refused:
        raise ValueError(
            "[musespark_open] full_attention_split_kv cannot replace the full-attention decode FMHA: "
            + " | ".join(refused)
        )
    if not full:
        return 0
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    (tp,) = tps
    if tp == 8:
        kv_heads = _v4.RANK_KV_HEADS[8]
    elif tp == 4:
        kv_heads = _v4.RANK_KV_HEADS[4]
    else:
        raise ValueError(f"[musespark_open] full_attention_split_kv has no build for tp {tp}")
    splits = _v4.default_splits(kv_heads, sms)
    census = SplitKvCensus(tuple(attention.layer_idx for attention in full))
    for attention in full:
        attention.__class__ = _split_kv_class(type(attention))
        attention._split_kv_pending = None
        attention._split_kv_census = census
        attention._split_kv_splits = splits
        attention._split_kv_kv_heads = kv_heads
    logger.info(
        f"[musespark_open] full_attention_split_kv: the decode FMHA of {len(full)} full-attention "
        f"layers runs as full_attention_split_kv partials (the tep{tp} build: "
        f"{kv_heads} KV heads per rank, {kv_heads // _v4.KV_HEADS} pair(s) x {splits} splits), "
        "merged in combine_norm_gate, on the steps full_attention_prep_append folds"
    )
    return len(full)


def install_spill_free(model: torch.nn.Module) -> int:
    """Every full-attention layer ``install`` re-classed launches the kernel's
    spill-free build (``full_attention_split_kv.SPILL_FREE_RESCALE_COLS``). Run after ``install``. Every
    layer is checked before any is changed. Returns the number of layers.

    Raises:
        ValueError: if a full-attention layer does not carry ``FullAttnSplitKvAttention`` (``install``).
    """
    full = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = [
        attention.layer_idx
        for attention in full
        if not isinstance(attention, FullAttnSplitKvAttention)
    ]
    if refused or not full:
        raise ValueError(
            "[musespark_open] the spill-free build of the full_attention_split_kv kernel needs "
            f"full_attention_split_kv, but full-attention layers {refused or 'none'} do not run "
            "full_attention_split_kv; install it first"
        )
    for attention in full:
        attention._split_kv_rescale_cols = _v4.SPILL_FREE_RESCALE_COLS
    logger.info(
        f"[musespark_open] full_attention_split_kv: the kernel of {len(full)} full-attention layers "
        f"rescales O in {_v4.SPILL_FREE_RESCALE_COLS}-column chunks (spill-free build; "
        "bit-identical to the default build)"
    )
    return len(full)
