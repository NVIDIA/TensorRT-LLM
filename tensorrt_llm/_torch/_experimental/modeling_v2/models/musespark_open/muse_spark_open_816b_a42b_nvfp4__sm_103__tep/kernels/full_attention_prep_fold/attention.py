# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The full-attention layers' folded prep runs inside the split-KV decode attention kernel.

Under ``full_attention_prep_append`` + ``full_attention_split_kv`` a decode step of one
sequence and at most 8 rows runs, on each of the 16 full-attention (NoPE) layers, the folded
prep (``preattn_prep_append``: Q / K norms, the step's K/V into the paged FP8 cache, the fused
operand and the gate copy), then the built-in attention call chain, whose generation
preprocess stands down for the prep and whose decode FMHA call launches
``full_attention_split_kv`` on the prep's fused operand; the output gate merges the partials
(``combine_norm_gate``). ``full_attention_prep_fold`` takes the prep's launch out of that
chain: such a step runs ``ops/attention/full_attention_prep_fold`` -- the same kernel reading
the raw projection, normalizing Q itself and appending the step's K/V from two of its idle
warps -- and the output gate reads the gate as a view of the projection. The partials and the
cache bytes are bit for bit those of the prep followed by ``full_attention_split_kv``.

Routing is decided once per layer and step, where the prep would run (``preprocess_qkv``): a
step is *fold-routed* when ``full_attention_prep_append`` would fold it
(``_fits_preattn_prep``, ``_prep_append_decline``) and ``full_attention_prep_fold`` serves it
as ``full_attention_split_kv`` would serve the prep's call (one K/V pool view of the plain
layout, one sequence's page table within the per-split page limit, the plain softmax scale,
the fold build compiled before a capture), ``forward_impl`` reaches ``_attn_impl`` (no
torch.compile / breakable-graph custom op, no sparse attention hooks, no helix context
parallelism), and the built-in chain would run the step as causal attention over its rows
with a BF16 output (no multi-token generation or spec-decoding tree mask, no quantized
output, no chunked attention: ``_prep_fold_builtin_mode``). A fold-routed step skips the prep
and returns the raw projection as the query and the gate view; this class's ``_attn_impl``
then launches the fold kernel itself and never calls the built-in ``_attn_impl``, so neither
the built-in preprocess nor the trtllm-gen FMHA can run on that layer (a stale prep output
never exists to be read). Any other step runs the folded prep and the split-KV attention
unchanged. A fold-routed call whose ``_attn_impl`` arguments are not the plain ones this
model's attention passes, or that ``_attn_impl`` never saw, raises: the prep was skipped, and
no fallback could compute it.

Ordering (``full_attention_prep_fold``'s record): the kernel's predecessor is the QKV
projection; it reads the projection and writes the cache only after its
``griddepcontrol.wait``, and every one of its threads waits before it exits, so the norm-gate
after it (which reads the gate from the projection after its own wait) and every kernel the
collectives' PDL chains reach see the projection complete.

Census: the prep append's fold census records the layer as folded and the split-KV census as
running ``full_attention_split_kv``, so their per-capture lines keep their meaning;
``FoldRouteCensus`` logs, once per captured decode graph, how many layers are fold-routed, and
warns about layers a step of the folded shape left on the folded-prep + split-KV route.
"""

from __future__ import annotations

import collections
import functools
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
)
from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.attention import full_attention_prep_fold as _fold
from ...ops.attention import full_attention_split_kv as _v4
from ..full_attention_prep_append import attention as _prep_append
from ..full_attention_split_kv import attention as _split_kv

#: The op's build: evict-first TMA loads (``split_kv_attention.EVICT_FIRST``).
EVICT_FIRST = _split_kv.EVICT_FIRST

#: ``FoldRouteCensus`` outcome kinds.
FOLD = "full_attention_prep_fold"
NO_DECODE_PREP = "no decode prep"
NOT_RUN = "not run in this forward"


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class FoldCall:
    """One fold-routed layer's decode step: what ``_attn_impl`` launches."""

    qkv: torch.Tensor
    kv_pool: torch.Tensor
    block_tables: torch.Tensor  # the sequence's [2, pages] row
    seq_lens: torch.Tensor
    kv_scale: torch.Tensor
    splits: int
    enable_pdl: bool
    eps: float
    q_scale: float
    permuted: bool
    rescale_cols: int = _v4.HEAD_DIM
    launched: bool = False

    @property
    def rows(self) -> int:
        return self.qkv.shape[0]

    def unplain(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Why an ``_attn_impl`` call is not the plain one this fold serves, or ``None``.

        This model's attention calls it as ``forward_impl`` does: the query (here the projection),
        no separate K / V, the causal mask, no rotary or mask data, no window on a
        full-attention layer, no output buffers, sinks, bias or LoRA.
        """
        names = (
            "q",
            "k",
            "v",
            "attn_metadata",
            "attention_mask",
            "mrope_rotary_cos_sin",
            "mrope_position_deltas",
            "attention_window_size",
            "attention_mask_data",
            "output",
            "output_sf",
            "attention_sinks",
            "relative_attention_bias",
            "relative_attention_max_distance",
            "has_lora",
        )
        if len(args) > len(names) or set(kwargs) - set(names[len(args) :]):
            return f"called with {len(args)} positional and {sorted(kwargs)} keyword arguments"
        a = dict(zip(names, args))
        a.update(kwargs)
        q = a.get("q")
        checks = (
            (
                not isinstance(q, torch.Tensor)
                or q.data_ptr() != self.qkv.data_ptr()
                or tuple(q.shape) != tuple(self.qkv.shape),
                "the query is not the step's projection",
            ),
            (a.get("k") is not None or a.get("v") is not None, "separate K / V"),
            (a.get("attention_mask") != PredefinedAttentionMask.CAUSAL, "not the causal mask"),
            (
                a.get("mrope_rotary_cos_sin") is not None
                or a.get("mrope_position_deltas") is not None,
                "rotary metadata",
            ),
            (a.get("attention_window_size") is not None, "an attention window"),
            (a.get("attention_mask_data") is not None, "mask data"),
            (a.get("output") is not None or a.get("output_sf") is not None, "an output buffer"),
            (a.get("attention_sinks") is not None, "attention sinks"),
            (a.get("relative_attention_bias") is not None, "a relative attention bias"),
            (bool(a.get("has_lora", False)), "LoRA"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        return None


@dataclass
class FoldRouteCensus:
    """Which full-attention layers were fold-routed in one forward, logged once per captured
    decode graph (as ``full_attention_prep_append.attention.FoldCensus`` does for the folded
    prep); eager decode steps are counted (host-side counters only)."""

    layers: Tuple[int, ...]
    outcomes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    captures: int = 0
    #: ``(capture number, layers fold-routed, layers, level, line)`` of the latest captures.
    reports: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=64))
    eager_steps: int = 0
    eager_unfolded_steps: int = 0
    eager_reasons: Dict[str, int] = field(default_factory=dict)

    def begin(self) -> None:
        self.outcomes = {}

    def record(self, layer_idx: int, kind: str, detail: str = "") -> None:
        self.outcomes[layer_idx] = (kind, detail)

    def _groups(self) -> Dict[Tuple[str, str], list]:
        groups: Dict[Tuple[str, str], list] = {}
        for layer_idx in self.layers:
            groups.setdefault(self.outcomes.get(layer_idx, (NOT_RUN, "")), []).append(layer_idx)
        return groups

    def end(self) -> Optional[tuple]:
        groups = self._groups()
        decode = any(kind not in (NO_DECODE_PREP, NOT_RUN) for kind, _ in groups)
        if not _capturing():
            if decode:
                self.eager_steps += 1
                left = [
                    f"{kind}{f' ({detail})' if detail else ''}"
                    for (kind, detail), layers in groups.items()
                    if kind not in (FOLD, NO_DECODE_PREP, NOT_RUN)
                    for _ in layers
                ]
                if left:
                    self.eager_unfolded_steps += 1
                    for reason in left:
                        self.eager_reasons[reason] = self.eager_reasons.get(reason, 0) + 1
            return None
        self.captures += 1
        folded = len(groups.get((FOLD, ""), ()))
        line = (
            f"[musespark_open] full_attention_prep_fold: CUDA graph capture {self.captures}: "
            f"{folded} of {len(self.layers)} "
            "full-attention layers run the prep inside full_attention_prep_fold"
        )
        rest = [
            f"{kind}{f' ({detail})' if detail else ''} on layers {', '.join(map(str, layers))}"
            for (kind, detail), layers in sorted(groups.items())
            if kind != FOLD
        ]
        if rest:
            line += "; " + "; ".join(rest)
        if not decode:
            level = "silent"
        elif any(kind not in (FOLD, NO_DECODE_PREP, NOT_RUN) for kind, _ in groups):
            level = "warning"
            logger.warning(
                line
                + " -- these layers run the folded prep and full_attention_split_kv in this graph"
            )
        else:
            level = "info"
            logger.info(line)
        report = (self.captures, folded, len(self.layers), level, line)
        self.reports.append(report)
        return report


class FullAttnSplitKvFoldAttention:
    """Mixin re-classed in front of the split-KV attention class on the full-attention layers.

    A decode step that ``full_attention_prep_append`` would fold and ``full_attention_prep_fold``
    serves skips the prep and runs that kernel from ``_attn_impl``, without the built-in call chain; any other call is
    exactly the class behind it. Adds three plain attributes: the current step's ``FoldCall``,
    the model's ``FoldRouteCensus`` and the split count (the split-KV attention's).
    """

    _prep_fold_call: Optional[FoldCall] = None
    _prep_fold_census: Optional[FoldRouteCensus] = None

    def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        census = self._prep_fold_census
        if census is not None:
            if self.layer_idx == census.layers[0]:
                census.begin()
            census.record(self.layer_idx, NO_DECODE_PREP)
        self._prep_fold_call = None
        try:
            result = super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            call, self._prep_fold_call = self._prep_fold_call, None
        if call is not None and not call.launched:
            raise RuntimeError(
                f"[musespark_open] full_attention_prep_fold: layer {self.layer_idx} skipped its prep for "
                "full_attention_prep_fold, and the attention never launched it"
            )
        if census is not None and self.layer_idx == census.layers[-1]:
            census.end()
        return result

    def _prep_fold_record(self, kind: str, detail: str = "") -> None:
        if self._prep_fold_census is not None:
            self._prep_fold_census.record(self.layer_idx, kind, detail)

    def _prep_fold_route(self, qkv: torch.Tensor, md) -> Tuple[Optional[str], Optional[tuple]]:
        """``(None, (kv_pool, block_tables))`` for a fold-routed step, else ``(reason, None)``.

        Host-side reads only; called for steps ``full_attention_prep_append`` would fold.
        """
        permuted = self._qkv_layout_inverse is not None
        mapping = getattr(self, "mapping", None)
        checks = (
            (
                qkv.dim() != 2
                or qkv.shape[1] != _fold.RAW_WIDTH
                or qkv.stride(-1) != 1
                or qkv.stride(0) % 8 != 0
                or qkv.data_ptr() % 16 != 0,
                "not a raw projection the fold reads",
            ),
            (not getattr(self, "attn_output_gate", False), "no output gate"),
            (is_torch_compiling() or is_in_breakable_cuda_graph(), "the custom attention op"),
            (getattr(self, "sparse_attn_hooks", None) is not None, "sparse attention hooks"),
            (
                mapping is not None and mapping.has_cp_helix(),
                "helix context parallelism",
            ),
            (get_bmm1_scale(self.attn) != _split_kv.SOFTMAX_SCALE, "another softmax scale"),
            (self.attn.kv_scale_orig_quant is None, "no KV quant scale"),
        )
        for failed, reason in checks:
            if failed:
                return reason, None
        reason = self._prep_fold_builtin_mode(md)
        if reason is not None:
            return reason, None
        attn = self.attn
        layer_idx = attn.get_local_layer_idx(md)
        total_num_blocks = self._prep_append_fmha_lib()._get_total_num_blocks(md)
        kv_pool, block_tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            layer_idx,
            self.num_key_value_heads,
            md.tokens_per_block,
            self.head_dim,
            2,
            total_num_blocks,
            attn.quant_mode,
            0,
            1,
            qkv.dtype,
        )
        splits = self._split_kv_splits
        checks = (
            (
                kv_pool.dtype != torch.float8_e4m3fn
                or kv_pool.dim() != 4
                or tuple(kv_pool.shape[1:]) != (_v4.KV_HEADS, _v4.PAGE, _v4.HEAD_DIM)
                or not kv_pool.is_contiguous(),
                "another cache view",
            ),
            (
                block_tables.dim() != 3 or block_tables.shape[0] != 1 or block_tables.shape[1] != 2,
                "not one sequence's K/V page table",
            ),
            (
                block_tables.dim() == 3 and -(-block_tables.shape[2] // splits) > _v4.MAX_CTA_PAGES,
                "more pages per split than the kernel stages",
            ),
            (
                _capturing()
                and _fold.compiled_key(
                    qkv.device.index,
                    LAUNCH_PDL,
                    EVICT_FIRST,
                    self.q_norm.variance_epsilon,
                    self.scale_query_by,
                    permuted,
                    self._split_kv_rescale_cols,
                )
                not in _v4._compiled_keys,
                "full_attention_prep_fold was not compiled before this capture",
            ),
        )
        for failed, reason in checks:
            if failed:
                return reason, None
        return None, (kv_pool, block_tables[0])

    def _prep_fold_builtin_mode(self, md) -> Optional[str]:
        """Why the built-in chain would run this step as something other than causal attention
        over the step's rows with a BF16 output -- the only mode the fold implements -- or
        ``None``. Host-side reads only.

        Each condition is read as the built-in chain reads it:

        - multi-token generation: the generation preprocess is handed the spec-decoding
          generation lengths (``phased.py``: spec-dec mode on and ``predicted_tokens_per_seq``
          > 1), on which the folded prep's stand-in runs the real preprocess
          (``FoldedPreprocess.mismatch``);
        - spec-dec tree masks: the metadata carries a spec-decoding mask, on which the
          sliding-window paged decode declines (``modeling._paged_decode_decline``) and the
          trtllm-gen FMHA refuses a tree;
          on sm_103 ``update_spec_dec_param`` turns spec-dec mode on only for a dynamic tree;
        - a quantized output: ``_attn_impl`` hands the attention an output scale;
        - chunked attention: a non-zero attention chunk size.

        None of these holds on the served verify step of a linear draft chain (DFlash, K = 7):
        on sm_103 its spec-dec mode stays off, and ``predicted_tokens_per_seq`` (an MLA
        parameter) is 1 on this model.
        """
        attn = self.attn
        spec_dec = getattr(md, "is_spec_decoding_enabled", False)
        checks = (
            (
                spec_dec
                and int(attn.predicted_tokens_per_seq) > 1
                and getattr(md, "spec_decoding_generation_lengths", None) is not None,
                "multi-token generation",
            ),
            (
                spec_dec
                or getattr(md, "is_spec_dec_tree", False)
                or getattr(md, "spec_decoding_bl_tree_mask", None) is not None,
                "spec-dec tree masks",
            ),
            (self._use_quantize_output(), "a quantized output"),
            (get_attention_chunk_size(attn) != 0, "chunked attention"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        return None

    def preprocess_qkv(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]):
        md = self._prep_append_metadata
        if (
            md is None
            or getattr(self, "_preattn_prep", None) is None
            or not self._fits_preattn_prep(qkv, position_ids)
            or self._prep_append_decline(qkv, md) is not None
        ):
            # Not a step the prep append folds: its own path decides (and records) what runs.
            return super().preprocess_qkv(qkv, position_ids)
        reason, views = self._prep_fold_route(qkv, md)
        if reason is not None:
            self._prep_fold_record("declined", reason)
            logger.warning_once(
                f"[musespark_open] full_attention_prep_fold: a decode step runs the folded prep and "
                f"full_attention_split_kv ({reason})",
                key=f"musespark_open.prep_fold.decline_{reason}",
            )
            return super().preprocess_qkv(qkv, position_ids)
        kv_pool, block_tables = views
        permuted = self._qkv_layout_inverse is not None
        self._prep_fold_call = FoldCall(
            qkv=qkv,
            kv_pool=kv_pool,
            block_tables=block_tables,
            seq_lens=md.kv_lens_cuda_runtime,
            kv_scale=self.attn.kv_scale_orig_quant,
            splits=self._split_kv_splits,
            rescale_cols=self._split_kv_rescale_cols,
            enable_pdl=LAUNCH_PDL,
            eps=self.q_norm.variance_epsilon,
            q_scale=self.scale_query_by,
            permuted=permuted,
        )
        return qkv, None, None, _fold.gate_view(qkv, permuted)

    def _attn_impl(self, *args, **kwargs):
        call = self._prep_fold_call
        if call is None:
            return super()._attn_impl(*args, **kwargs)
        if call.launched:
            raise RuntimeError(
                f"[musespark_open] full_attention_prep_fold: layer {self.layer_idx}'s attention ran twice "
                "on one fold-routed step"
            )
        reason = call.unplain(args, kwargs)
        if reason is not None:
            raise RuntimeError(
                f"[musespark_open] full_attention_prep_fold: layer {self.layer_idx} skipped its prep for "
                "full_attention_prep_fold, but "
                f"its attention call is not the plain one the fold serves ({reason})"
            )
        part_o, part_lse = _fold.full_attention_prep_fold(
            call.qkv,
            call.eps,
            call.q_scale,
            call.permuted,
            call.kv_pool,
            call.block_tables,
            call.seq_lens,
            call.kv_scale,
            call.splits,
            call.enable_pdl,
            evict_first=EVICT_FIRST,
            rescale_cols=call.rescale_cols,
        )
        call.launched = True
        rows = call.rows
        # The attention output the output gate is handed: never written (the gate merges the
        # partials instead, as with the split-KV attention).
        out = torch.empty(
            (rows, _v4.Q_HEADS * _v4.HEAD_DIM), dtype=torch.bfloat16, device=call.qkv.device
        )
        self._split_kv_pending = (part_o, part_lse, rows, out.data_ptr(), call.seq_lens)
        self._prep_fold_record(FOLD)
        if self._split_kv_census is not None:
            self._split_kv_census.record(self.layer_idx, _split_kv.V4)
        self._prep_append_record(_prep_append.FOLDED)
        return out, None


@functools.lru_cache(maxsize=None)
def _fold_class(base: type) -> type:
    """``base`` with ``FullAttnSplitKvFoldAttention`` in front, one class per base."""
    return type(f"{base.__name__}Fold", (FullAttnSplitKvFoldAttention, base), {})


def install(model: torch.nn.Module) -> int:
    """Re-class every full-attention layer's attention in front of the split-KV attention class.

    Run after ``full_attention_split_kv.attention.install``. Every layer is checked before any is
    re-classed. Returns the number of layers re-classed.

    Raises:
        ValueError: if a full-attention layer does not carry the split-KV and the prep-append mixins,
            has no output gate, or lacks the norm epsilon, query pre-scale or projection layout
            the fold bakes in.
    """
    full = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = []
    for attention in full:
        reasons = [
            why
            for failed, why in (
                (
                    not isinstance(attention, _split_kv.FullAttnSplitKvAttention),
                    "full_attention_split_kv is not installed",
                ),
                (
                    not isinstance(attention, _prep_append.FullAppendAttention),
                    "full_attention_prep_append's folded prep is not installed",
                ),
                (not getattr(attention, "attn_output_gate", False), "it has no output gate"),
                (
                    not hasattr(attention, "q_norm")
                    or not hasattr(attention, "scale_query_by")
                    or not hasattr(attention, "_qkv_layout_inverse"),
                    "no norm epsilon, query pre-scale or projection layout",
                ),
                (attention.head_dim != _v4.HEAD_DIM, f"head_dim is not {_v4.HEAD_DIM}"),
            )
            if failed
        ]
        if reasons:
            refused.append(f"layer {attention.layer_idx}: " + "; ".join(reasons))
    if refused:
        raise ValueError(
            "[musespark_open] full_attention_prep_fold cannot fold the prep into full_attention_split_kv: "
            + " | ".join(refused)
        )
    census = FoldRouteCensus(tuple(attention.layer_idx for attention in full))
    for attention in full:
        attention.__class__ = _fold_class(type(attention))
        attention._prep_fold_call = None
        attention._prep_fold_census = census
    logger.info(
        f"[musespark_open] full_attention_prep_fold: the decode prep of {len(full)} full-attention layers "
        "runs inside full_attention_prep_fold (the step's K/V append and the Q norm in the attention "
        "kernel; no prep launch) on the steps full_attention_prep_append folds and "
        "full_attention_split_kv serves"
    )
    return len(full)
