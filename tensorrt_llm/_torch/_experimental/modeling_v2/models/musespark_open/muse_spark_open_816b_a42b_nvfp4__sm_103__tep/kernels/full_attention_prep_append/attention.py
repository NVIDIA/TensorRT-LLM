# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The full-attention layers' decode preprocess folded into the ``preattn_prep`` decode prep.

On a decode step the built-in attention of a full-attention (NoPE) layer runs
the ``preattn_prep`` decode prep, then ``FlashInferTrtllmGenFmha.run_generation``: first the
generation preprocess ``thop.trtllm_gen_generation_preprocess``, whose only
kernel on this path is ``applyBiasRopeUpdateKVCacheV2`` -- it writes the step's
K/V into the paged FP8 cache, Q in E4M3 into the attention workspace's query
buffer and the FMHA's two scale words into the workspace -- and then flashinfer's
trtllm-gen decode FMHA, which reads exactly those. With this fold the prep writes
all of it (``ops/attention/preattn_prep_append.py``) and the preprocess call
returns the views the real call would return without launching anything, so the
FMHA follows the prep directly: one launch fewer per full-attention layer.

What the real preprocess computes on this path, and where each piece comes from
under the fold (``cpp/tensorrt_llm/thop/trtllmGenQKVProcessOp.cpp``):

* the decoder-info kernel: not launched for one sequence without a mask, a
  dynamic RoPE or device scales (``isBuildDecoderInfoKernelNeeded``), so no
  ``cu_seqlens`` -- required by the check below;
* ``applyBiasRopeUpdateKVCacheV2``: K/V cache bytes, E4M3 Q, ``bmm1`` (2 words)
  and ``bmm2`` (1 word) -- written by the prep (its tile-counter reset is not
  armed: the counter pointer is null);
* the paged cache view and block tables -- ``build_trtllm_gen_kv_cache_metadata``
  (the same C++ function, host-side);
* the query, scale and FMHA workspace views -- the generation workspace layout
  (``get_trtllm_gen_generation_workspace_layout``) over the same workspace;
* ``max_q_len``, ``max_kv_len``, ``window_left``, ``is_multi_token_gen`` --
  host integers computed from the arguments as the C++ computes them.

How the launch is skipped without editing the built-in attention:
``run_generation`` looks ``thop`` up as a global of its module. While a full
layer's built-in attention runs after the folded prep, that global is an
``_ElidingThop`` whose ``trtllm_gen_generation_preprocess`` checks the call
against what the prep wrote (``FoldedPreprocess.mismatch``): the prep's fused
operand and rows, the workspace and its layout, the cache pointers, tables,
sequence lengths and scales, one sequence, NoPE, FP8 KV and FP8 Q, no
multi-token or cross path. On a match it returns the views; on any mismatch it
runs the real call, which rewrites the very bytes the prep wrote (its input is
the same operand), and logs why once. Every other attribute is the real
``thop``. ``install`` refuses the feature when ``run_generation`` does not read
``thop`` from that module global or the binding does not name the arguments
``PREPROCESS_ARGS`` (``preprocess_reachable``). A ``run_generation`` that calls
the preprocess some other way runs the real call, which rewrites the bytes the
prep wrote; ``FoldCensus`` then names the layers left on the built-in path.

Ordering: the FMHA is the prep's immediate successor on the stream, launched
with programmatic dependent launch, and reads the cache, the query and the
scale words only after its ``griddepcontrol.wait`` -- the property the built-in
path relies on for the append kernel, which triggers its dependents before it
exits. With this fold alone the prep never triggers early, so the FMHA cannot
even launch before every prep CTA has exited.

A decode graph is captured once and replayed for every step, so a fold the
capture declined is one the server never runs. ``FoldCensus`` records what each
full-attention layer did in one forward and, for a forward under CUDA-graph
capture, logs one line per captured decode graph: the number of layers folded,
or a warning naming the layers left on the built-in path at a step the fold
covers (a prefill graph, where no layer sees a decode prep, is not logged).

``drop_o_proj_prefetch`` removes the ``dense_gemm_prefetch`` L2 prefetch of the
attention output weight from these layers' QKV projection: without the append launch it would
issue under the FMHA.
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
from ...ops.attention.preattn_prep_append import MAX_REQUESTS, MAX_ROWS, preattn_prep_append

#: ``PositionEmbeddingType.learned_absolute``: what a layer without RoPE passes; the
#: append kernel then applies no rotation.
_NO_ROTARY = 0
#: ``RotaryScalingType.none``.
_NO_ROTARY_SCALING = 0


def _ptr(t: Optional[torch.Tensor]) -> Optional[int]:
    return None if t is None else t.data_ptr()


@functools.lru_cache(maxsize=64)
def generation_layout(
    dtype: torch.dtype,
    rows: int,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    batch: int = 1,
) -> dict:
    """The C++ generation workspace layout of ``batch`` sequences of ``rows`` query rows in all."""
    return dict(
        thop.get_trtllm_gen_generation_workspace_layout(
            dtype, batch, rows, num_heads, 64, rotary_dim, num_kv_heads
        )
    )


def _workspace_view(workspace: torch.Tensor, offset: int, size: int, dtype: torch.dtype):
    """``WorkspaceAccessor::tensor``: ``size`` bytes at ``offset`` as a 1-D ``dtype`` tensor."""
    raw = workspace.view(torch.uint8).reshape(-1)
    return raw[offset : offset + size].view(dtype)


def _window_left(cyclic: int, max_past: int, chunk: int) -> int:
    """``computeWindowLeft`` of the C++ preprocess (chunked attention is refused before)."""
    return cyclic - 1 if 0 < cyclic < max_past else -1


@dataclass
class FoldedPreprocess:
    """What one folded prep wrote, and the preprocess call that may skip its launch."""

    qkv_out: torch.Tensor
    rows: int
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
    rotary_dim: int
    #: The step's sequences (``wide_full_attention``: 2..4 of ``rows / requests`` rows each).
    requests: int = 1
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
                    int(a["num_tokens"]),
                    int(a["num_heads"]),
                    int(a["num_kv_heads"]),
                    int(a["rotary_embedding_dim"]),
                    *((int(a["batch_beam"]),) if self.requests > 1 else ()),
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
                or int(a["input_seq_length"]) != self.rows // self.requests,
                "other rows",
            ),
            (
                int(a["batch_beam"]) != self.requests,
                "not one sequence" if self.requests == 1 else "other sequences",
            ),
            (int(a["position_embedding_type"]) != _NO_ROTARY, "a rotary layer"),
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
        rows, heads = self.rows, self.num_heads
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
            q_buf.view(rows, heads, 64),
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1,
            bmm2,
            fmha_workspace,
            None,
            int(a["input_seq_length"]),
            int(a["max_past_kv_length"]),
            _window_left(
                int(a["cyclic_attention_window_size"]),
                int(a["max_past_kv_length"]),
                int(a["attention_chunk_size"]),
            ),
            False,
        )


#: ``FoldCensus`` outcome kinds. The first three are what a step is meant to do: the
#: fold, a call without a decode prep (a prefill, or a step the ``preattn_prep``
#: prep does not take), and a decline that only says the step is not the shape the fold covers.
FOLDED = "folded"
NO_DECODE_PREP = "no decode prep"
SHAPE = "not the folded shape"
#: ... and the outcomes that leave a step of the folded shape on the built-in path.
DECLINED = "declined"
PREPROCESS_RAN = "built-in preprocess ran"
NOT_REACHED = "preprocess not reached"
NOT_RUN = "not run in this forward"
_EXPECTED = frozenset({FOLDED, NO_DECODE_PREP, SHAPE})

_NOT_ONE_SEQUENCE = "not one generation sequence"
_BEAM_SEARCH = "beam search"


def _is_shape_decline(reason: str) -> bool:
    """Whether a decline only says the step has more rows or sequences than the fold takes."""
    return (
        reason in (_NOT_ONE_SEQUENCE, _BEAM_SEARCH) or re.fullmatch(r"\d+ rows", reason) is not None
    )


def _capturing() -> bool:
    """Whether this thread's current CUDA stream is capturing a graph."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class FoldCensus:
    """What each full-attention layer did in one forward, logged once per captured graph.

    One per model, shared by its full-attention layers (``install``): the first
    of them opens a forward's record, each records its outcome, the last closes
    it. A forward under CUDA-graph capture builds the graph the server replays at
    every step, so its record is logged: one info line with the number of layers
    that fold, or a warning naming the layers a step of the folded shape leaves
    on the built-in path (a decline other than the shape, a preprocess call that
    did not match, no preprocess call). A capture in which no layer saw a decode
    prep (a prefill graph) is recorded but not logged. Eager steps are counted
    (``count_eager``: host-side counters, no device synchronization, no log line);
    their once-per-reason warnings stay as they are. A step that replays a captured
    graph runs no Python, so it does what that graph's record says.
    """

    #: The model's full-attention layer indices, in forward order.
    layers: Tuple[int, ...]
    outcomes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    captures: int = 0
    #: ``(capture number, layers folded, layers, level, line)`` of the latest 64 captures,
    #: level ``"warning"``, ``"info"``, or ``"silent"`` (no decode prep: not logged).
    reports: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=64))
    #: Eager forwards in which a full-attention layer saw a decode prep, those of them
    #: that left a layer of the folded shape on the built-in path, and why (per layer).
    eager_steps: int = 0
    eager_fallback_steps: int = 0
    eager_reasons: Dict[str, int] = field(default_factory=dict)

    def begin(self) -> None:
        self.outcomes = {}

    def record(self, layer_idx: int, kind: str, detail: str = "") -> None:
        self.outcomes[layer_idx] = (kind, detail)

    def count_eager(self) -> None:
        """Count this eager forward if a layer saw a decode prep (host-side only)."""
        outcomes = [self.outcomes.get(layer_idx, (NOT_RUN, "")) for layer_idx in self.layers]
        if all(kind in (NO_DECODE_PREP, NOT_RUN) for kind, _ in outcomes):
            return
        self.eager_steps += 1
        left = [
            f"{kind}{f' ({detail})' if detail else ''}"
            for kind, detail in outcomes
            if kind not in _EXPECTED
        ]
        if left:
            self.eager_fallback_steps += 1
            for reason in left:
                self.eager_reasons[reason] = self.eager_reasons.get(reason, 0) + 1

    def end(self) -> Optional[tuple]:
        """Log and return the record of a captured forward; ``None`` when not capturing
        (an eager forward is counted instead, ``count_eager``)."""
        if not _capturing():
            self.count_eager()
            return None
        self.captures += 1
        groups: Dict[Tuple[str, str], list] = {}
        for layer_idx in self.layers:
            outcome = self.outcomes.get(layer_idx, (NOT_RUN, ""))
            groups.setdefault(outcome, []).append(layer_idx)
        folded = len(groups.get((FOLDED, ""), ()))
        line = (
            f"[musespark_open] full_attention_prep_append: CUDA graph capture {self.captures}: "
            f"{folded} of {len(self.layers)} full-attention layers fold the KV append "
            "(generation preprocess launch skipped)"
        )
        rest = [
            f"{kind}{f' ({detail})' if detail else ''} on layers {', '.join(map(str, layers))}"
            for (kind, detail), layers in sorted(groups.items())
            if kind != FOLDED
        ]
        if rest:
            line += "; " + "; ".join(rest)
        if not all(kind in _EXPECTED for kind, _ in groups):
            level = "warning"
            logger.warning(line + " -- these layers serve the built-in path in this graph")
        elif all(kind == NO_DECODE_PREP for kind, _ in groups):
            level = "silent"
        else:
            level = "info"
            logger.info(line)
        report = (self.captures, folded, len(self.layers), level, line)
        self.reports.append(report)
        return report


class FullAppendAttention:
    """Mixin re-classed in front of a full-attention layer's ``preattn_prep`` attention class.

    A decode call (one carrying the ``preattn_prep`` kwarg) whose projection,
    metadata and attention backend fit runs ``preattn_prep_append`` in place
    of the ``preattn_prep`` prep and lets the built-in attention's generation preprocess
    stand down for it (``eliding``); any other call is exactly the class
    behind it. Adds two plain attributes set for the duration of one forward,
    and one set by ``install``: the model's ``FoldCensus``.
    """

    _prep_append_metadata: Optional[object]
    _prep_append_folded: Optional[FoldedPreprocess]
    _prep_append_census: Optional[FoldCensus] = None
    #: ``wide_full_attention`` (``install_multi``): fold decode steps of 2..``MAX_REQUESTS``
    #: generation sequences of equal row count too.
    _prep_append_multi: bool = False

    def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        census = self._prep_append_census
        if census is not None:
            if self.layer_idx == census.layers[0]:
                census.begin()
            census.record(self.layer_idx, NO_DECODE_PREP)
        self._prep_append_metadata = (
            attn_metadata if kwargs.get("preattn_prep") is not None else None
        )
        self._prep_append_folded = None
        try:
            result = super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._prep_append_metadata = None
            self._prep_append_folded = None
        if census is not None and self.layer_idx == census.layers[-1]:
            census.end()
        return result

    def _prep_append_record(self, kind: str, detail: str = "") -> None:
        if self._prep_append_census is not None:
            self._prep_append_census.record(self.layer_idx, kind, detail)

    def preprocess_qkv(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]):
        md = self._prep_append_metadata
        if md is None or getattr(self, "_preattn_prep", None) is None:
            return super().preprocess_qkv(qkv, position_ids)
        if not self._fits_preattn_prep(qkv, position_ids):
            return super().preprocess_qkv(qkv, position_ids)
        decline = self._prep_append_decline(qkv, md)
        if decline is not None:
            self._prep_append_record(SHAPE if _is_shape_decline(decline) else DECLINED, decline)
            logger.warning_once(
                f"[musespark_open] full_attention_prep_append: a decode step did not fit the folded prep "
                f"({decline}); running the preattn_prep prep and the built-in preprocess",
                key=f"musespark_open.prep_append.decline_{decline}",
            )
            return super().preprocess_qkv(qkv, position_ids)
        qkv_out, gate, folded = self._prep_append_prep(qkv, md)
        self._prep_append_folded = folded
        self._prep_append_record(NOT_REACHED)
        return qkv_out, None, None, gate

    def _attn_impl(self, *args, **kwargs):
        folded = self._prep_append_folded
        if folded is None:
            return super()._attn_impl(*args, **kwargs)
        self._prep_append_folded = None
        with eliding(folded):
            result = super()._attn_impl(*args, **kwargs)
        fallback = getattr(folded, "fallback", None)
        if folded.consumed and fallback is None:
            self._prep_append_record(FOLDED)
        elif fallback is not None:
            self._prep_append_record(PREPROCESS_RAN, fallback)
        else:
            logger.warning_once(
                "[musespark_open] full_attention_prep_append: the built-in attention did not reach "
                "the trtllm-gen generation preprocess after a folded prep; it ran its own path",
                key="musespark_open.prep_append.unconsumed",
            )
        return result

    def _prep_append_requests(self, md) -> int:
        """The sequences of this step the fold takes in one call: 1, or under
        ``_prep_append_multi`` the 2..``MAX_REQUESTS`` generation sequences of a step whose host
        ``seq_lens`` are all ``rows / sequences`` (at most ``MAX_ROWS``). Host-side reads only."""
        if not self._prep_append_multi:
            return 1
        n = md.num_seqs
        rows = md.num_tokens
        seq_lens = getattr(md, "seq_lens", None)
        if (
            not 2 <= n <= MAX_REQUESTS
            or md.num_generations != n
            or rows % n
            or rows // n > MAX_ROWS
            or seq_lens is None
            or seq_lens.is_cuda
            or seq_lens.numel() < n
            or bool((seq_lens[:n] != rows // n).any())
        ):
            return 1
        return n

    def _prep_append_fmha_lib(self):
        """The flashinfer trtllm-gen FMHA library of this layer's FMHA manager, or None."""
        return _target_attention.trtllm_gen_fmha(self.attn)

    def _prep_append_decline(self, qkv: torch.Tensor, md) -> Optional[str]:
        """Why this call cannot run the folded prep, or ``None``. Host-side reads only."""
        attn = self.attn
        mode = QuantMode(int(getattr(attn, "quant_mode", 0)))
        requests = self._prep_append_requests(md)
        checks = (
            (not 1 <= qkv.shape[0] // requests <= MAX_ROWS, f"{qkv.shape[0]} rows"),
            (not getattr(self, "skip_rope", False), "a RoPE layer"),
            (getattr(attn, "position_embedding_type", None) != _NO_ROTARY, "a rotary append"),
            (not mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache(), "the KV cache is not FP8"),
            (getattr(md, "kv_cache_manager", None) is None, "no KV cache manager"),
            (getattr(md, "kv_cache_block_offsets", None) is None, "no paged KV cache"),
            (getattr(md, "is_cross", False), "cross attention"),
            (getattr(md, "beam_width", 1) != 1, _BEAM_SEARCH),
            (
                md.num_seqs != requests or md.num_generations != requests,
                _NOT_ONE_SEQUENCE,
            ),
            (md.num_tokens != qkv.shape[0], "rows other than the step's tokens"),
            (getattr(md, "effective_workspace", None) is None, "no attention workspace"),
            (self._prep_append_fmha_lib() is None, "no trtllm-gen FMHA library"),
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
        rows = qkv.shape[0]
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
            qkv.dtype,
            rows,
            self.num_heads,
            self.num_key_value_heads,
            attn.rope_params.dim,
            *((requests,) if requests > 1 else ()),
        )
        if workspace.numel() * workspace.element_size() < max(required, layout["total_size"]):
            return "a workspace the attention grows first"
        return None

    def _prep_append_prep(
        self, qkv: torch.Tensor, md
    ) -> Tuple[torch.Tensor, torch.Tensor, FoldedPreprocess]:
        attn = self.attn
        rows = qkv.shape[0]
        requests = self._prep_append_requests(md)
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
            requests,
            qkv.dtype,
        )
        workspace = md.effective_workspace
        layout = generation_layout(
            qkv.dtype,
            rows,
            self.num_heads,
            self.num_key_value_heads,
            attn.rope_params.dim,
            *((requests,) if requests > 1 else ()),
        )
        q_buf = _workspace_view(workspace, layout["q_buf_offset"], layout["q_buf_size"], qkv.dtype)
        bmm1 = _workspace_view(
            workspace, layout["bmm1_scale_offset"], layout["bmm1_scale_size"], torch.float32
        )
        bmm2 = _workspace_view(
            workspace, layout["bmm2_scale_offset"], layout["bmm2_scale_size"], torch.float32
        )
        seq_lens = md.kv_lens_cuda_runtime
        bmm1_host = get_bmm1_scale(attn)
        qkv_out, gate = preattn_prep_append(
            qkv,
            self.num_heads,
            self.num_key_value_heads,
            self.q_norm.variance_epsilon,
            self.scale_query_by,
            self._qkv_layout_inverse is not None,
            kv_pool,
            block_tables[0] if requests == 1 else block_tables[:requests],
            seq_lens,
            attn.kv_scale_orig_quant,
            attn.kv_scale_quant_orig,
            q_buf,
            bmm1,
            bmm2,
            bmm1_host,
            1.0,
            LAUNCH_PDL,
        )
        folded = FoldedPreprocess(
            qkv_out=qkv_out,
            rows=rows,
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
            rotary_dim=attn.rope_params.dim,
            requests=requests,
        )
        return qkv_out, gate, folded


@functools.lru_cache(maxsize=None)
def _folded_class(base: type) -> type:
    """``base`` with ``FullAppendAttention`` in front, one class per base."""
    return type(f"{base.__name__}FullAppend", (FullAppendAttention, base), {})


def install(model: torch.nn.Module) -> int:
    """Re-class every full-attention layer's attention in front of its current class.

    Run after the attention lattice (``preattn_prep`` / ``swa_paged_decode`` /
    ``boundary_early_release``) has re-classed the layers; every layer is checked before any
    is re-classed. Returns the number of layers re-classed.

    Raises:
        ValueError: if ``eliding`` would not reach the built-in preprocess
            (``preprocess_reachable``), or a full-attention
            layer's attention is not a ``preattn_prep`` class, has RoPE or is not on the
            TRTLLM backend over fused QKV with 64-wide heads.
    """
    reason = preprocess_reachable()
    if reason is not None:
        raise ValueError(
            f"[musespark_open] full_attention_prep_append cannot skip the built-in preprocess: {reason}"
        )
    full = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = []
    for attention in full:
        reasons = [
            why
            for failed, why in (
                (
                    not hasattr(attention, "_preattn_prep"),
                    "its attention is not the preattn_prep class",
                ),
                (not attention.skip_rope, "it applies RoPE"),
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
            "[musespark_open] full_attention_prep_append cannot fold the full-attention preprocess: "
            + " | ".join(refused)
        )
    census = FoldCensus(tuple(attention.layer_idx for attention in full))
    for attention in full:
        attention.__class__ = _folded_class(type(attention))
        attention._prep_append_metadata = None
        attention._prep_append_folded = None
        attention._prep_append_census = census
    logger.info(
        f"[musespark_open] full_attention_prep_append: the KV append and FP8 query of {len(full)} "
        "full-attention layers run in the preattn_prep prep on decode steps of one sequence and at most "
        f"{MAX_ROWS} rows; their trtllm-gen generation preprocess launch is skipped"
    )
    return len(full)


def install_multi(model: torch.nn.Module) -> int:
    """``wide_full_attention``: the fold also takes decode steps of 2..``MAX_REQUESTS`` sequences.

    Run after ``install``. Sets ``_prep_append_multi`` on every full-attention layer; returns
    how many.

    Raises:
        ValueError: if a full-attention layer does not carry the fold (``install``).
    """
    full = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = [a.layer_idx for a in full if not isinstance(a, FullAppendAttention)]
    if refused:
        raise ValueError(
            "[musespark_open] wide_full_attention extends full_attention_prep_append, which "
            f"full-attention layers {refused} do not carry"
        )
    for attention in full:
        attention._prep_append_multi = True
    logger.info(
        f"[musespark_open] wide_full_attention: the folded prep of {len(full)} full-attention "
        f"layers also takes decode steps of 2..{MAX_REQUESTS} sequences of up to {MAX_ROWS} rows"
    )
    return len(full)


def drop_o_proj_prefetch(model: torch.nn.Module) -> int:
    """The full-attention layers' QKV projection forks no attention output prefetch.

    The ``dense_gemm_prefetch`` QKV site forks an L2 prefetch of the layer's attention
    output weight after the QKV GEMM. Behind the folded prep the FMHA starts sooner,
    while the prefetch is still issuing, so its HBM traffic runs under the
    FMHA's KV stream, and at long contexts that stream evicts the prefetched lines
    before the attention output GEMM reads them. Clearing the site's next weight
    removes the fork; the attention output site's join then finds nothing pending.
    The sliding-window layers are not touched. Every site is checked before any is
    changed. Returns the number of sites changed.

    Raises:
        ValueError: if a full-attention layer's QKV projection has no ``dense_gemms``
            dispatcher with the prefetch (``dense_gemms`` not installed, or installed without
            ``dense_gemm_prefetch``)
            or its site does not prefetch the layer's attention output weight.
    """
    from ..dense_gemms import dispatcher as dense_gemms

    sites = []
    refused = []
    for layer in model.layers:
        attention = layer.self_attn
        if attention.is_sliding:
            continue
        dispatcher = vars(attention.qkv_proj).get("_low_m_dispatcher")
        if not isinstance(dispatcher, dense_gemms.DenseGemmDispatcher):
            refused.append(
                f"layer {attention.layer_idx}: its QKV projection has no dense_gemms dispatcher"
            )
            continue
        site = dispatcher.site
        if not site.plan.prefetch:
            refused.append(f"layer {attention.layer_idx}: dense_gemms runs without the prefetch")
        elif site.role != "qkv" or site.next_weight is None:
            refused.append(
                f"layer {attention.layer_idx}: its QKV site forks no attention output prefetch"
            )
        elif site.next_weight() is not attention.o_proj.weight:
            refused.append(f"layer {attention.layer_idx}: its QKV site prefetches another weight")
        else:
            sites.append(site)
    if refused:
        raise ValueError(
            "[musespark_open] drop_o_proj_prefetch cannot drop the full-attention attention output prefetch: "
            + " | ".join(refused)
        )
    for site in sites:
        site.next_weight = None
    logger.info(
        f"[musespark_open] drop_o_proj_prefetch: the QKV projection of {len(sites)} full-attention "
        "layers forks no L2 prefetch of the attention output weight"
    )
    return len(sites)
