# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every layer's decode preprocess folded into the decode prep on steps of two or more requests.

On a decode step (no context rows) of ``B >= 2`` sequences, every target layer's
built-in attention runs the ``preattn_prep`` decode prep, then
``FlashInferTrtllmGenFmha.run_generation``: the generation preprocess
``thop.trtllm_gen_generation_preprocess`` -- two kernels here, the decoder-info
kernel ``computeSeqAndPaddingOffsets`` and ``applyBiasRopeUpdateKVCacheV2``
(no rotation: the prep applied RoPE) -- then flashinfer's trtllm-gen decode
FMHA. With this fold the prep writes what the FMHA reads
(``ops/attention/multi_request_prep_append.py``: the K/V append, the E4M3 query,
the scale words, the cumulative query lengths) and the preprocess call returns
the views the real call would return without launching anything: two launches
fewer per layer, on all 62 layers.

What the real preprocess computes on this path, and where each piece comes from
under the fold (``cpp/tensorrt_llm/thop/trtllmGenQKVProcessOp.cpp``):

* the decoder-info kernel (``B > 1`` and more than one query row): cumulative
  query lengths -- written by the prep, for fixed query lengths; cumulative key
  lengths and the per-token table -- not written: without multi-token
  generation (``spec_decoding_generation_lengths`` absent, checked below) the
  append kernel does not read them and the FMHA takes ``q_len_per_req``;
* ``applyBiasRopeUpdateKVCacheV2``: K/V cache bytes, E4M3 Q, ``bmm1`` (2 words)
  and ``bmm2`` (1 word) -- written by the prep;
* the paged cache view and block tables -- ``build_trtllm_gen_kv_cache_metadata``
  (the same C++ function, host-side);
* the query, scale, cumulative-length and FMHA workspace views -- the generation
  workspace layout (``get_trtllm_gen_generation_workspace_layout``) of ``B``
  sequences over the same workspace;
* ``max_q_len``, ``max_kv_len``, ``window_left``, ``is_multi_token_gen`` --
  host integers computed from the arguments as the C++ computes them.

The preprocess call is reached through the same stand-in the one-sequence fold
uses (``attention_forward.eliding``): while a layer's built-in attention runs
after a folded prep, ``run_generation`` sees a ``thop`` whose generation
preprocess checks the call against what the prep wrote
(``MultiRequestFoldedPreprocess.mismatch``) and returns the views on a match;
on any mismatch it runs the real call -- which rewrites the very bytes the prep
wrote from the same operand -- and logs why once.

Which steps: ``DecodeTarget`` decode steps (the call carries the
``preattn_prep`` op) with no context rows, ``min_seqs <= B <= 128`` sequences
(``install``; 2 unless the composition folds the append of smaller multi-request
steps in its own paths) of one width ``Q <= 8`` and at most ``preattn_prep``'s
1024 rows. A step of one
sequence -- the 8-row decode step of a single request -- returns to the class behind this mixin
before anything else is read, so the one-sequence fast paths
(``full_attention_prep_append``, ``full_attention_split_kv``,
``full_attention_prep_fold``, ``swa_block``) and their kernels are unchanged.
Prefill and mixed steps carry no ``preattn_prep`` and are never folded.

Ordering: the FMHA is the prep's immediate successor on the stream, launched
with programmatic dependent launch, and reads the cache, the query and the
scale words only after its ``griddepcontrol.wait``; the prep never triggers
early, so the FMHA cannot launch before every prep CTA has exited.

``MultiRequestCensus`` records what each layer did in one forward and logs one
line per captured decode graph of two or more sequences: the number of layers
folded, or a warning naming the layers left on the built-in path.
"""

from __future__ import annotations

import collections
import functools
import re
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from ... import attention as _target_attention
from ...attention_forward import PREPROCESS_ARGS, eliding, preprocess_reachable
from ...dispatch import LAUNCH_PDL
from ...ops.attention.multi_request_prep_append import (
    MAX_ROWS_PER_SEQ,
    MAX_SEQS,
    PACKED_GEOMETRY,
    launch_kind,
    multi_request_prep_append,
)

#: ``PositionEmbeddingType.learned_absolute``: what the attention op of a layer
#: whose RoPE runs in the prep passes; the append kernel then applies no rotation.
_NO_ROTARY = 0
#: ``RotaryScalingType.none``.
_NO_ROTARY_SCALING = 0


def _ptr(t: Optional[torch.Tensor]) -> Optional[int]:
    return None if t is None else t.data_ptr()


@functools.lru_cache(maxsize=256)
def generation_layout(
    dtype: torch.dtype,
    num_seqs: int,
    rows: int,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
) -> dict:
    """The C++ generation workspace layout of ``num_seqs`` sequences and ``rows`` query rows."""
    return dict(
        thop.get_trtllm_gen_generation_workspace_layout(
            dtype, num_seqs, rows, num_heads, 64, rotary_dim, num_kv_heads
        )
    )


def _workspace_view(workspace: torch.Tensor, offset: int, size: int, dtype: torch.dtype):
    """``WorkspaceAccessor::tensor``: ``size`` bytes at ``offset`` as a 1-D ``dtype`` tensor."""
    raw = workspace.view(torch.uint8).reshape(-1)
    return raw[offset : offset + size].view(dtype)


def _window_left(cyclic: int, max_past: int) -> int:
    """``computeWindowLeft`` of the C++ preprocess (chunked attention is refused before)."""
    return cyclic - 1 if 0 < cyclic < max_past else -1


#: Fewest sequences of a step the fold takes by default (``install``'s ``min_seqs``).
MIN_SEQS = 2


def takes_step(attn_metadata, min_seqs: int = MIN_SEQS) -> bool:
    """Whether a step is one this path may fold: no context rows, ``min_seqs``..128 sequences, no beams.

    Host-side reads only (per-capture constants under a CUDA graph).
    """
    return (
        getattr(attn_metadata, "num_contexts", 1) == 0
        and min_seqs <= getattr(attn_metadata, "num_seqs", 0) <= MAX_SEQS
        and getattr(attn_metadata, "num_generations", 0) == getattr(attn_metadata, "num_seqs", -1)
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


@dataclass
class MultiRequestFoldedPreprocess:
    """What one folded prep wrote, and the preprocess call that may skip its launch."""

    qkv_out: torch.Tensor
    rows: int
    num_seqs: int
    rows_per_seq: int
    num_heads: int
    num_kv_heads: int
    workspace_ptr: int
    layout: dict
    seq_lens_ptr: int
    block_offsets_ptr: int
    pool_pointers: torch.Tensor
    pool_mapping: torch.Tensor
    orig_quant_ptr: int
    quant_orig_ptr: int
    layer_idx: int
    tokens_per_block: int
    total_num_blocks: int
    quant_mode: int
    bmm1_host: float
    consumed: bool = field(default=False)
    #: Why a preprocess call in this layer's window ran the real op, if one did.
    fallback: Optional[str] = field(default=None)

    def mismatch(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Why this preprocess call is not the one the prep stood in for, or ``None``."""
        if kwargs or len(args) != len(PREPROCESS_ARGS):
            return f"called with {len(args)} positional and {sorted(kwargs)} keyword arguments"
        a = dict(zip(PREPROCESS_ARGS, args))
        qkv, workspace = a["qkv_input"], a["workspace"]
        mode = QuantMode(int(a["kv_cache_quant_mode"]))
        checks = (
            (
                qkv.data_ptr() != self.qkv_out.data_ptr()
                or tuple(qkv.shape) != tuple(self.qkv_out.shape),
                "not the prep's fused operand",
            ),
            (workspace.data_ptr() != self.workspace_ptr, "another workspace"),
            (
                generation_layout(
                    qkv.dtype,
                    int(a["batch_beam"]),
                    int(a["num_tokens"]),
                    int(a["num_heads"]),
                    int(a["num_kv_heads"]),
                    int(a["rotary_embedding_dim"]),
                )
                != self.layout,
                "another workspace layout",
            ),
            (workspace.nbytes < self.layout["total_size"], "a workspace below its layout"),
            (_ptr(a["sequence_lengths"]) != self.seq_lens_ptr, "other sequence lengths"),
            (_ptr(a["kv_cache_block_offsets"]) != self.block_offsets_ptr, "other block offsets"),
            (a["host_kv_cache_pool_pointers"] is not self.pool_pointers, "other pool pointers"),
            (a["host_kv_cache_pool_mapping"] is not self.pool_mapping, "another pool mapping"),
            (_ptr(a["kv_scale_orig_quant"]) != self.orig_quant_ptr, "another quant scale"),
            (_ptr(a["kv_scale_quant_orig"]) != self.quant_orig_ptr, "another dequant scale"),
            (a["attention_output_orig_quant"] is not None, "a quantized output"),
            (
                a["spec_decoding_generation_lengths"] is not None
                and int(a["predicted_tokens_per_seq"]) > 1,
                "multi-token generation",
            ),
            (a["mrope_position_deltas"] is not None, "mRoPE"),
            (int(a["layer_idx"]) != self.layer_idx, "another layer"),
            (int(a["seq_offset"]) != 0, "a batch offset"),
            (
                int(a["num_heads"]) != self.num_heads
                or int(a["num_kv_heads"]) != self.num_kv_heads
                or int(a["head_size"]) != 64,
                "other head counts",
            ),
            (int(a["tokens_per_block"]) != self.tokens_per_block, "another page size"),
            (int(a["kv_cache_quant_mode"]) != self.quant_mode, "another KV cache mode"),
            (not mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache(), "not an FP8 KV cache"),
            (
                int(a["num_tokens"]) != self.rows
                or int(a["input_seq_length"]) != self.rows_per_seq
                or int(a["batch_beam"]) != self.num_seqs,
                "other rows or sequences",
            ),
            (int(a["position_embedding_type"]) != _NO_ROTARY, "a rotary append"),
            (int(a["rotary_embedding_scale_type"]) != _NO_ROTARY_SCALING, "rotary scaling"),
            (float(a["bmm1_scale"]) != self.bmm1_host, "another softmax scale"),
            (not a["fp8_context_fmha"], "a BF16 query"),
            (int(a["attention_chunk_size"]) != 0, "chunked attention"),
            (int(a["total_num_blocks"]) != self.total_num_blocks, "another pool bound"),
            (int(a["kv_factor"]) != 2, "not a K/V cache"),
            (not a["need_build_kv_cache_metadata"], "no cache view asked for"),
            (a["cross_attention"], "cross attention"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        return None

    def outputs(self, args: tuple) -> tuple:
        """What the real call returns, built host-side from the same arguments."""
        a = dict(zip(PREPROCESS_ARGS, args))
        workspace, layout = a["workspace"], self.layout
        kv_pool, block_tables, kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            a["host_kv_cache_pool_pointers"],
            a["host_kv_cache_pool_mapping"],
            a["kv_cache_block_offsets"],
            int(a["layer_idx"]),
            int(a["num_kv_heads"]),
            int(a["tokens_per_block"]),
            int(a["head_size"]),
            int(a["kv_factor"]),
            int(a["total_num_blocks"]),
            int(a["kv_cache_quant_mode"]),
            int(a["seq_offset"]),
            int(a["batch_beam"]),
            a["qkv_input"].dtype,
        )
        dtype = a["qkv_input"].dtype
        q_buf = _workspace_view(workspace, layout["q_buf_offset"], layout["q_buf_size"], dtype)
        bmm1 = _workspace_view(
            workspace, layout["bmm1_scale_offset"], layout["bmm1_scale_size"], torch.float32
        )
        bmm2 = _workspace_view(
            workspace, layout["bmm2_scale_offset"], layout["bmm2_scale_size"], torch.float32
        )
        # The decoder-info kernel runs (and its cumulative query lengths are returned) only
        # for more than one query row per sequence.
        cu_seqlens = (
            _workspace_view(
                workspace, layout["cu_seqlens_offset"], layout["cu_seqlens_size"], torch.int32
            )
            if self.rows_per_seq > 1
            else None
        )
        if layout["trtllm_gen_workspace_size"] == 0:
            fmha_workspace = torch.empty(0, dtype=torch.uint8, device=workspace.device)
        else:
            fmha_workspace = _workspace_view(
                workspace,
                layout["trtllm_gen_workspace_offset"],
                layout["trtllm_gen_workspace_size"],
                torch.uint8,
            )
        return (
            q_buf.view(self.rows, self.num_heads, 64),
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1,
            bmm2,
            fmha_workspace,
            cu_seqlens,
            int(a["input_seq_length"]),
            int(a["max_past_kv_length"]),
            _window_left(int(a["cyclic_attention_window_size"]), int(a["max_past_kv_length"])),
            False,
        )


#: ``MultiRequestCensus`` outcome kinds. The first two are what a step is meant to do: the
#: fold, and a call that is not a multi-request decode prep (one sequence, a prefill).
FOLDED = "folded"
NOT_TAKEN = "not a multi-request decode prep"
#: ... and the outcomes that leave a multi-request decode step on the built-in path.
DECLINED = "declined"
PREPROCESS_RAN = "built-in preprocess ran"
NOT_REACHED = "preprocess not reached"
NOT_RUN = "not run in this forward"
_EXPECTED = frozenset({FOLDED, NOT_TAKEN})


def _capturing() -> bool:
    """Whether this thread's current CUDA stream is capturing a graph."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class MultiRequestCensus:
    """What each layer did in one forward, logged once per captured multi-request graph.

    One per model, shared by its layers (``install``): the first layer opens a
    forward's record, each records its outcome, the last closes it -- in a forward
    whose first layer takes a multi-request decode step; any other forward (one
    sequence, prefill, mixed) leaves the census alone after its first layer. A forward
    under CUDA-graph capture of a multi-request decode step builds the graph the
    server replays at every step of that batch size, so its record is logged:
    one info line with the number of layers folded, or a warning naming the
    layers left on the built-in path. Eager forwards are counted (host-side
    counters, no log line).
    """

    #: The model's layer indices, in forward order.
    layers: Tuple[int, ...]
    outcomes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    sequences: int = 0
    captures: int = 0
    #: ``(capture number, sequences, layers folded, layers, level, line)`` of the latest 64.
    reports: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=64))
    eager_steps: int = 0
    eager_fallback_steps: int = 0
    eager_reasons: Dict[str, int] = field(default_factory=dict)

    def begin(self, sequences: int) -> None:
        self.outcomes = {}
        self.sequences = sequences

    def record(self, layer_idx: int, kind: str, detail: str = "") -> None:
        self.outcomes[layer_idx] = (kind, detail)

    def end(self) -> Optional[tuple]:
        """Log and return the record of a captured multi-request forward; ``None`` otherwise."""
        outcomes = [self.outcomes.get(layer_idx, (NOT_RUN, "")) for layer_idx in self.layers]
        if all(kind in (NOT_TAKEN, NOT_RUN) for kind, _ in outcomes):
            return None
        left = [(i, k, d) for i, (k, d) in zip(self.layers, outcomes) if k not in _EXPECTED]
        if not _capturing():
            self.eager_steps += 1
            if left:
                self.eager_fallback_steps += 1
                for _, kind, detail in left:
                    reason = f"{kind}{f' ({detail})' if detail else ''}"
                    self.eager_reasons[reason] = self.eager_reasons.get(reason, 0) + 1
            return None
        self.captures += 1
        folded = sum(1 for kind, _ in outcomes if kind == FOLDED)
        line = (
            f"[musespark_open] multi_request_prep_append: CUDA graph capture {self.captures} "
            f"({self.sequences} sequences): {folded} of {len(self.layers)} layers fold the K/V "
            "append into the decode prep (generation preprocess launch skipped)"
        )
        groups: Dict[Tuple[str, str], list] = {}
        for layer_idx, kind, detail in left:
            groups.setdefault((kind, detail), []).append(layer_idx)
        if groups:
            line += "; " + "; ".join(
                f"{kind}{f' ({detail})' if detail else ''} on layers {', '.join(map(str, ids))}"
                for (kind, detail), ids in sorted(groups.items())
            )
            level = "warning"
            logger.warning(line + " -- these layers serve the built-in path in this graph")
        else:
            level = "info"
            logger.info(line)
        report = (self.captures, self.sequences, folded, len(self.layers), level, line)
        self.reports.append(report)
        return report


def _is_shape_decline(reason: str) -> bool:
    """Whether a decline only says the step is not a uniform multi-request decode step."""
    return re.fullmatch(r"(\d+ rows .*|.* sequences)", reason) is not None


class MultiRequestPrepAppendAttention:
    """Mixin re-classed in front of every layer's attention class.

    A decode call (one carrying the ``preattn_prep`` kwarg) of a multi-request step
    whose projection, metadata and attention backend fit runs
    ``multi_request_prep_append`` in place of the ``preattn_prep`` prep and lets the
    built-in attention's generation preprocess stand down for it (``eliding``);
    any other call is exactly the class behind it. Adds two plain attributes set
    for the duration of one forward, and two set by ``install``: the model's
    ``MultiRequestCensus`` and the fewest sequences of a step it folds.
    """

    _mr_append_metadata: Optional[object]
    _mr_append_folded: Optional[MultiRequestFoldedPreprocess]
    _mr_append_census: Optional[MultiRequestCensus] = None
    _mr_append_min_seqs: int = MIN_SEQS

    def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        census = self._mr_append_census
        takes = kwargs.get("preattn_prep") is not None and takes_step(
            attn_metadata, self._mr_append_min_seqs
        )
        if census is not None and self.layer_idx == census.layers[0]:
            census.begin(int(getattr(attn_metadata, "num_seqs", 0)) if takes else 0)
        # Only a forward the first layer opened as a multi-request decode step is counted.
        counted = census is not None and census.sequences > 0
        if counted:
            census.record(self.layer_idx, NOT_TAKEN)
        self._mr_append_metadata = attn_metadata if takes else None
        self._mr_append_folded = None
        try:
            result = super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._mr_append_metadata = None
            self._mr_append_folded = None
        if counted and self.layer_idx == census.layers[-1]:
            census.end()
        return result

    def _mr_append_record(self, kind: str, detail: str = "") -> None:
        if self._mr_append_census is not None:
            self._mr_append_census.record(self.layer_idx, kind, detail)

    def preprocess_qkv(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]):
        md = self._mr_append_metadata
        if md is None or getattr(self, "_preattn_prep", None) is None:
            return super().preprocess_qkv(qkv, position_ids)
        if not self._fits_preattn_prep(qkv, position_ids):
            self._mr_append_record(DECLINED, "the projection does not fit preattn_prep")
            return super().preprocess_qkv(qkv, position_ids)
        decline = self._mr_append_decline(qkv, md)
        if decline is not None:
            self._mr_append_record(NOT_TAKEN if _is_shape_decline(decline) else DECLINED, decline)
            logger.warning_once(
                f"[musespark_open] multi_request_prep_append: a decode step did not fit the folded "
                f"prep ({decline}); running the preattn_prep prep and the built-in preprocess",
                key=f"musespark_open.mr_prep_append.decline_{decline}",
            )
            return super().preprocess_qkv(qkv, position_ids)
        qkv_out, gate, folded = self._mr_append_prep(qkv, position_ids, md)
        self._mr_append_folded = folded
        self._mr_append_record(NOT_REACHED)
        return qkv_out, None, None, gate

    def _attn_impl(self, *args, **kwargs):
        folded = self._mr_append_folded
        if folded is None:
            return super()._attn_impl(*args, **kwargs)
        self._mr_append_folded = None
        with eliding(folded):
            result = super()._attn_impl(*args, **kwargs)
        fallback = getattr(folded, "fallback", None)
        if folded.consumed and fallback is None:
            self._mr_append_record(FOLDED)
        elif fallback is not None:
            self._mr_append_record(PREPROCESS_RAN, fallback)
        else:
            logger.warning_once(
                "[musespark_open] multi_request_prep_append: the built-in attention did not reach "
                "the trtllm-gen generation preprocess after a folded prep; it ran its own path",
                key="musespark_open.mr_prep_append.unconsumed",
            )
        return result

    def _mr_append_fmha_lib(self):
        """The flashinfer trtllm-gen FMHA library of this layer's FMHA manager, or None."""
        return _target_attention.trtllm_gen_fmha(self.attn)

    def _mr_append_decline(self, qkv: torch.Tensor, md) -> Optional[str]:
        """Why this call cannot run the folded prep, or ``None``. Host-side reads only."""
        attn = self.attn
        mode = QuantMode(int(getattr(attn, "quant_mode", 0)))
        rows = qkv.shape[0]
        seqs = int(md.num_seqs)
        checks = (
            (md.num_tokens != rows, f"{rows} rows other than the step's tokens"),
            (rows % seqs != 0, f"{rows} rows over {seqs} sequences"),
            (
                rows % seqs == 0 and not 1 <= rows // seqs <= MAX_ROWS_PER_SEQ,
                f"{rows} rows of {seqs} sequences",
            ),
            (getattr(attn, "position_embedding_type", None) != _NO_ROTARY, "a rotary append"),
            (not mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache(), "the KV cache is not FP8"),
            (getattr(md, "kv_cache_manager", None) is None, "no KV cache manager"),
            (getattr(md, "kv_cache_block_offsets", None) is None, "no paged KV cache"),
            (getattr(md, "is_cross", False), "cross attention"),
            (getattr(md, "is_spec_decoding_enabled", False), "multi-token generation"),
            (getattr(md, "effective_workspace", None) is None, "no attention workspace"),
            (self._mr_append_fmha_lib() is None, "no trtllm-gen FMHA library"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        tokens_per_block = md.tokens_per_block
        if not tokens_per_block or tokens_per_block & (tokens_per_block - 1):
            return f"{tokens_per_block}-token pages"
        # The attention grows its workspace before the preprocess call when it is below
        # what it asks for (``prepare_workspace``); a prep that wrote into the old one
        # would only be redone, so such a step runs the built-in path.
        workspace = md.effective_workspace
        required = _fi_gen._get_workspace_size(
            dtype=qkv.dtype,
            num_tokens=max(rows, md.max_context_length),
            num_gen_tokens=max(rows, md.max_num_requests),
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            max_num_requests=md.max_num_requests,
            rotary_embedding_dim=attn.rope_dim,
            fp8_context_fmha=True,
        )
        layout = generation_layout(
            qkv.dtype, seqs, rows, self.num_heads, self.num_key_value_heads, attn.rope_params.dim
        )
        if workspace.numel() * workspace.element_size() < max(required, layout["total_size"]):
            return "a workspace the attention grows first"
        return None

    def _mr_append_prep(
        self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor], md
    ) -> Tuple[torch.Tensor, torch.Tensor, MultiRequestFoldedPreprocess]:
        attn = self.attn
        rows = qkv.shape[0]
        seqs = int(md.num_seqs)
        layer_idx = attn.get_local_layer_idx(md)
        total_num_blocks = self._mr_append_fmha_lib()._get_total_num_blocks(md)
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
            seqs,
            qkv.dtype,
        )
        workspace = md.effective_workspace
        layout = generation_layout(
            qkv.dtype, seqs, rows, self.num_heads, self.num_key_value_heads, attn.rope_params.dim
        )
        q_buf = _workspace_view(workspace, layout["q_buf_offset"], layout["q_buf_size"], qkv.dtype)
        bmm1 = _workspace_view(
            workspace, layout["bmm1_scale_offset"], layout["bmm1_scale_size"], torch.float32
        )
        bmm2 = _workspace_view(
            workspace, layout["bmm2_scale_offset"], layout["bmm2_scale_size"], torch.float32
        )
        cu_seqlens = _workspace_view(
            workspace, layout["cu_seqlens_offset"], layout["cu_seqlens_size"], torch.int32
        )
        seq_lens = md.kv_lens_cuda_runtime
        bmm1_host = get_bmm1_scale(attn)
        rope = not self.skip_rope
        batch = getattr(self, "_batch_dense_gemms", None)
        prefetch = batch.prep_prefetch(self, rows) if batch is not None else None
        qkv_out, gate = multi_request_prep_append(
            qkv,
            self.rotary_emb.rotary_cos_sin if rope else None,
            position_ids if rope else None,
            self.num_heads,
            self.num_key_value_heads,
            self.q_norm.variance_epsilon,
            self.scale_query_by,
            self._qkv_layout_inverse is not None,
            seqs,
            kv_pool,
            block_tables,
            seq_lens,
            attn.kv_scale_orig_quant,
            attn.kv_scale_quant_orig,
            q_buf,
            bmm1,
            bmm2,
            cu_seqlens,
            bmm1_host,
            1.0,
            LAUNCH_PDL,
            prefetch=prefetch,
        )
        folded = MultiRequestFoldedPreprocess(
            qkv_out=qkv_out,
            rows=rows,
            num_seqs=seqs,
            rows_per_seq=rows // seqs,
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            workspace_ptr=workspace.data_ptr(),
            layout=layout,
            seq_lens_ptr=seq_lens.data_ptr(),
            block_offsets_ptr=md.kv_cache_block_offsets.data_ptr(),
            pool_pointers=md.host_kv_cache_pool_pointers,
            pool_mapping=md.host_kv_cache_pool_mapping,
            orig_quant_ptr=attn.kv_scale_orig_quant.data_ptr(),
            quant_orig_ptr=attn.kv_scale_quant_orig.data_ptr(),
            layer_idx=layer_idx,
            tokens_per_block=md.tokens_per_block,
            total_num_blocks=total_num_blocks,
            quant_mode=int(attn.quant_mode),
            bmm1_host=bmm1_host,
        )
        return qkv_out, gate, folded


@functools.lru_cache(maxsize=None)
def _folded_class(base: type) -> type:
    """``base`` with ``MultiRequestPrepAppendAttention`` in front, one class per base."""
    return type(f"{base.__name__}MultiRequestAppend", (MultiRequestPrepAppendAttention, base), {})


def install(model: torch.nn.Module, min_seqs: int = MIN_SEQS) -> int:
    """Re-class every layer's attention with ``MultiRequestPrepAppendAttention`` in front.

    Run after every other attention re-class: the mixin goes in front of whatever
    class each layer carries by then, and acts only on decode steps of ``min_seqs``
    or more sequences (a composition whose own paths fold the append of smaller
    multi-request steps passes the first count they leave to this one).
    Every layer is checked before any is re-classed. Idempotent: a model whose
    layers all carry the mixin is left alone. Returns the number of layers
    re-classed.

    Raises:
        ValueError: if ``eliding`` would not reach the built-in preprocess
            (``preprocess_reachable``), or a layer's attention is not a
            ``preattn_prep`` class or is not on the TRTLLM backend over fused QKV
            with 64-wide heads.
    """
    reason = preprocess_reachable()
    if reason is not None:
        raise ValueError(
            "[musespark_open] multi_request_prep_append cannot skip the built-in preprocess: "
            f"{reason}"
        )
    attentions = [layer.self_attn for layer in model.layers]
    if all(isinstance(attention, MultiRequestPrepAppendAttention) for attention in attentions):
        return 0
    refused = []
    for attention in attentions:
        reasons = [
            why
            for failed, why in (
                (
                    not hasattr(attention, "_preattn_prep"),
                    "its attention is not the preattn_prep class",
                ),
                (attention.attn_backend != "TRTLLM", "the attention backend is not TRTLLM"),
                (not attention.support_fused_qkv, "the attention backend takes split Q/K/V"),
                (attention.head_dim != 64, "head_dim is not 64"),
            )
            if failed
        ]
        if reasons:
            refused.append(f"layer {attention.layer_idx}: " + "; ".join(reasons))
    if refused:
        raise ValueError(
            "[musespark_open] multi_request_prep_append cannot fold the decode preprocess: "
            + " | ".join(refused)
        )
    if not 2 <= min_seqs <= MAX_SEQS:
        raise ValueError(
            f"[musespark_open] multi_request_prep_append folds steps of 2..{MAX_SEQS} sequences, "
            f"not from {min_seqs}"
        )
    census = MultiRequestCensus(tuple(attention.layer_idx for attention in attentions))
    for attention in attentions:
        attention.__class__ = _folded_class(type(attention))
        attention._mr_append_metadata = None
        attention._mr_append_folded = None
        attention._mr_append_census = census
        attention._mr_append_min_seqs = min_seqs
    grid = (
        "packed grid (last rows, items per program): "
        f"{[(last, items) for last, items, _, _ in PACKED_GEOMETRY]}, one item per program on "
        "o_proj-prefetching steps"
        if not torch.cuda.is_available() or launch_kind(torch.device("cuda")) == "packed"
        else "grid of one program per (row, head)"
    )
    logger.info(
        f"[musespark_open] multi_request_prep_append: the K/V append, E4M3 query and FMHA scales "
        f"of {len(attentions)} layers run in the decode prep on steps of {min_seqs}..{MAX_SEQS} sequences "
        f"of up to {MAX_ROWS_PER_SEQ} rows each; their trtllm-gen generation preprocess launch "
        f"(decoder-info + append) is skipped; {grid}"
    )
    return len(attentions)
