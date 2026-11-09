# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode FMHA of steps with two or more requests as ``multi_request_split_kv``.

On a decode step (no context rows) of two or more sequences, every layer's built-in attention
reaches flashinfer's trtllm-gen generation FMHA through ``run_generation``: the generation
preprocess appends the step's K/V and, with the FP8 KV cache, quantizes Q to E4M3 and writes
the FMHA scales as fp32 device scalars (``fp8_context_fmha``; a ``generation_only`` call keeps
Q in BF16 with host scales), then ``flashinfer.decode.trtllm_batch_decode_with_kv_cache``
attends. This path keeps that call
chain and replaces only the last launch: while such a layer's built-in attention runs,
``flashinfer`` as seen by ``flashinfer_trtllm_gen`` is a stand-in
(``attention_forward.intercepting``) whose decode call checks the call against the plain case
the kernel is written for and then launches ``ops/attention/multi_request_split_kv`` on the
call's own query, cache view, block tables and sequence lengths, writing the call's ``out``.
Everything after the FMHA (the output norm-gate, the output projection) reads ``out`` as before.

Which layers: the full-attention layers, and the sliding-window layers when installed with
``sliding=True`` (the target installs with ``SLIDING_LAYERS``). A sliding-window layer's call
always runs the 2048-key window build: its ``window_left`` is 2047, or -1 when the call's
``max_seq_len`` does not exceed the window, in which case no key lies outside the window and the
window changes nothing.

Which steps: ``num_contexts == 0`` and at least two sequences, read on the host from the
metadata (per-capture constants under a CUDA graph). A step of one sequence -- the 8-row
decode graph of a single request -- returns to the class behind this mixin before anything else
is read, so its kernels and what the 8-row fast paths do are unchanged. Prefill and mixed steps are never
intercepted. ``install``'s ``min_requests`` raises the floor where another path owns the smallest
multi-request steps (``wide_full_attention`` takes 2..4 sequences on sm107 TEP4): steps below it
are the class behind this mixin.

Any call that is not the plain case -- positional arguments, other keywords, sinks, a scaled
cache, another layout, a query length above 8 or ragged query lengths, scales that are neither
an E4M3 query's device scalars nor a BF16 query's host softmax scale, more than 128 requests,
a build not compiled before a CUDA-graph capture (the 128-request build, which batches of
65..128 requests take, compiles on the first such call: a decode graph's warmup forward) --
launches the real FMHA with the same arguments and is recorded with its reason
(``MultiRequestCensus``, logged once per captured graph).

Numerics: not bit-identical to the built-in FMHA (``multi_request_split_kv``'s module
docstring: the same E4M3 Q, K and V as the built-in, with P rounded to E4M3 and FP16
partials); deterministic; no cache byte changes.
"""

from __future__ import annotations

import collections
import functools
from dataclasses import dataclass, field
from typing import ClassVar, Deque, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale
from tensorrt_llm.logger import logger

from ...attention import trtllm_gen_fmha
from ...attention_forward import decode_reachable, intercepting
from ...ops.attention import multi_request_split_kv as _mr

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
#: The softmax scale the layers apply: 1 / sqrt(64).
SOFTMAX_SCALE = 0.125
#: The sliding-window layers' window.
WINDOW = 2048
#: Whether the target installs the path on the sliding-window layers too. Their 2048-key reads
#: are short enough that the built-in FMHA's one launch is the faster choice there.
SLIDING_LAYERS = False

#: ``MultiRequestCensus`` outcome kinds.
SERVED = "multi_request_split_kv"
BUILTIN = "built-in FMHA"
NOT_RUN = "not run in this forward"


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def takes_step(attn_metadata, min_requests: int = 2) -> bool:
    """Whether a step is one this path intercepts: no context rows and at least
    ``min_requests`` (at least two) sequences.

    Host-side reads only.
    """
    return (
        getattr(attn_metadata, "num_contexts", 1) == 0
        and getattr(attn_metadata, "num_seqs", 0) >= max(2, min_requests)
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


def _is_device_scalar(value, device) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.numel() >= 1
        and value.device == device
    )


def _scale_reason(kwargs: dict, q_fp8: bool) -> Optional[str]:
    """Why the call's FMHA scales are not one of the two cases the builds take, or ``None``:
    a BF16 query with host scales (the softmax scale and a float ``bmm2_scale``), or an E4M3
    query with the preprocess's fp32 device scales (read by the kernel and the combine on the
    device; ``bmm1_scale_log2``, the preprocess's log2 copy, is not read). Host-side only."""
    bmm1, bmm2, log2 = kwargs["bmm1_scale"], kwargs["bmm2_scale"], kwargs["bmm1_scale_log2"]
    if not q_fp8:
        if not isinstance(bmm1, float) or not isinstance(bmm2, float) or log2 is not None:
            return "device FMHA scales with a BF16 query"
        if bmm1 != SOFTMAX_SCALE:
            return "another softmax scale"
        return None
    device = kwargs["query"].device
    if not _is_device_scalar(bmm1, device) or not _is_device_scalar(bmm2, device):
        return "an E4M3 query without fp32 device FMHA scales"
    if log2 is not None and not _is_device_scalar(log2, device):
        return "another bmm1_scale_log2"
    return None


@dataclass
class MultiRequestCall:
    """One layer's decode call: what the stand-in needs, and what it did."""

    #: The fast path the stand-in names when the call runs the built-in FMHA.
    PATH: ClassVar[str] = "multi_request_split_kv"
    kv_heads: int
    window: int
    launched: bool = False
    fallback: Optional[str] = None

    def decline(self, kwargs: dict) -> Optional[str]:
        """Why this decode FMHA call is not the plain case, or ``None``. Host-side reads only."""
        if set(kwargs) != DECODE_KWARGS:
            return f"the FMHA was called with {sorted(kwargs)}"
        kv_cache = kwargs["kv_cache"]
        pool = kv_cache[0] if isinstance(kv_cache, tuple) and len(kv_cache) == 2 else None
        query = kwargs["query"]
        out = kwargs["out"]
        tables = kwargs["block_tables"]
        seq_lens = kwargs["seq_lens"]
        q_len = kwargs["q_len_per_req"]
        window_left = kwargs["window_left"]
        heads = _mr.Q_PER_KV * self.kv_heads
        requests = seq_lens.numel() if isinstance(seq_lens, torch.Tensor) else 0
        q_fp8 = isinstance(query, torch.Tensor) and query.dtype == torch.float8_e4m3fn
        checks = (
            (pool is None or kv_cache[1] is not pool, "not one K/V pool"),
            (
                pool is not None
                and (
                    pool.dtype != torch.float8_e4m3fn
                    or pool.dim() != 4
                    or tuple(pool.shape[1:]) != (self.kv_heads, _mr.PAGE, _mr.HEAD_DIM)
                    or not pool.is_contiguous()
                ),
                "another cache view",
            ),
            (
                window_left != -1 and not (self.window and window_left == self.window - 1),
                f"window_left {window_left}",
            ),
            (kwargs["sinks"] is not None, "attention sinks"),
            (kwargs["kv_cache_sf"] is not None, "a scaled KV cache"),
            (kwargs["kv_layout"] != "HND", "another KV layout"),
            (kwargs["backend"] != "trtllm-gen", "another backend"),
            (
                kwargs["max_q_len"] is not None or kwargs["cum_seq_lens_q"] is not None,
                "ragged query lengths",
            ),
            (not isinstance(q_len, int) or not 1 <= q_len <= _mr.MAX_ROWS, f"q_len {q_len}"),
            (kwargs["uses_shared_paged_kv_idx"], "shared K/V page ids"),
            (_scale_reason(kwargs, q_fp8) is not None, _scale_reason(kwargs, q_fp8)),
            (
                not isinstance(seq_lens, torch.Tensor)
                or seq_lens.dtype != torch.int32
                or seq_lens.dim() != 1
                or not 2 <= requests <= _mr.MAX_REQUESTS,
                f"{requests} sequences",
            ),
            (
                not isinstance(query, torch.Tensor)
                or query.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                or query.dim() != 3
                or tuple(query.shape[1:]) != (heads, _mr.HEAD_DIM)
                or not isinstance(q_len, int)
                or query.shape[0] != q_len * requests,
                "another query",
            ),
            (
                not isinstance(tables, torch.Tensor)
                or tables.dtype != torch.int32
                or tables.dim() != 3
                or tables.shape[0] < requests
                or tables.shape[1] != 2,
                "another block table",
            ),
            (
                not isinstance(out, torch.Tensor)
                or out.dtype != torch.bfloat16
                or isinstance(query, torch.Tensor)
                and out.numel() != query.numel(),
                "another output",
            ),
        )
        for failed, reason in checks:
            if failed:
                return reason
        if _capturing() and (
            _mr.compiled_key(
                query.device.index,
                bool(kwargs["enable_pdl"]),
                self.kv_heads,
                self.window,
                q_fp8=q_fp8,
                device_scale=q_fp8,
                requests=requests,
            )
            not in _mr._compiled_keys
        ):
            return "multi_request_split_kv was not compiled before this capture"
        return None

    def launch(self, kwargs: dict) -> None:
        query = kwargs["query"]
        rows = query.shape[0]
        _mr.multi_request_split_kv(
            query.view(rows, -1),
            kwargs["kv_cache"][0],
            kwargs["block_tables"],
            kwargs["seq_lens"],
            kwargs["bmm1_scale"],
            kwargs["bmm2_scale"],
            kwargs["out"].view(rows, -1),
            window=self.window,
            enable_pdl=bool(kwargs["enable_pdl"]),
        )
        self.launched = True


@dataclass
class MultiRequestCensus:
    """Which layers ran the op in one forward, logged once per captured decode graph."""

    layers: Tuple[int, ...]
    outcomes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    captures: int = 0
    #: ``(capture number, layers on the op, layers, level, line)`` of the latest captures.
    reports: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=64))

    def begin(self) -> None:
        self.outcomes = {}

    def record(self, layer_idx: int, kind: str, detail: str = "") -> None:
        self.outcomes[layer_idx] = (kind, detail)

    def end(self) -> Optional[tuple]:
        if not _capturing() or not self.outcomes:
            return None
        self.captures += 1
        groups: Dict[Tuple[str, str], list] = {}
        for layer_idx in self.layers:
            groups.setdefault(self.outcomes.get(layer_idx, (NOT_RUN, "")), []).append(layer_idx)
        served = len(groups.get((SERVED, ""), ()))
        line = (
            f"[musespark_open] multi_request_split_kv: CUDA graph capture {self.captures}: "
            f"{served} of {len(self.layers)} layers run multi_request_split_kv"
        )
        rest = [
            f"{kind}{f' ({detail})' if detail else ''} on layers {', '.join(map(str, layers))}"
            for (kind, detail), layers in sorted(groups.items())
            if kind != SERVED
        ]
        if rest:
            line += "; " + "; ".join(rest)
        if any(kind == BUILTIN for kind, _ in groups):
            level = "warning"
            logger.warning(line + " -- these layers run the built-in FMHA in this graph")
        else:
            level = "info"
            logger.info(line)
        report = (self.captures, served, len(self.layers), level, line)
        self.reports.append(report)
        return report


class MultiRequestSplitKvAttention:
    """Mixin re-classed in front of a layer's attention class.

    A call on a step ``takes_step`` accepts runs the class behind it with the decode FMHA call
    intercepted (``intercepting``): on the plain case the stand-in launches
    ``multi_request_split_kv`` into the call's output; otherwise the real FMHA runs. Any other
    call is exactly the class behind it. Adds plain attributes only.
    """

    _mr_window: int = 0
    _mr_kv_heads: int = 4
    _mr_min_requests: int = 2
    _mr_census: Optional[MultiRequestCensus] = None

    def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        census = self._mr_census
        if census is not None and self.layer_idx == census.layers[0]:
            census.begin()
        result = super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        if census is not None and self.layer_idx == census.layers[-1]:
            census.end()
        return result

    def _attn_impl(self, q, k, v, attn_metadata, *args, **kwargs):
        if not takes_step(attn_metadata, self._mr_min_requests):
            return super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        call = MultiRequestCall(kv_heads=self._mr_kv_heads, window=self._mr_window)
        with intercepting(call):
            result = super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        census = self._mr_census
        if census is not None:
            if call.launched:
                census.record(self.layer_idx, SERVED)
            else:
                census.record(self.layer_idx, BUILTIN, call.fallback or "the FMHA was not called")
        return result


@functools.lru_cache(maxsize=None)
def _mr_class(base: type) -> type:
    """``base`` with ``MultiRequestSplitKvAttention`` in front, one class per base."""
    return type(f"{base.__name__}MR", (MultiRequestSplitKvAttention, base), {})


def install(model: torch.nn.Module, sliding: bool, min_requests: int = 2) -> int:
    """Re-class the full-attention layers' attention (and, with ``sliding``, the sliding-window
    layers') in front of whatever class they carry, so that their decode steps of at least
    ``min_requests`` (at least two) sequences take this path, and compile the kernel build their decode
    calls launch: an E4M3 query with fp32 device scales, which the generation preprocess makes
    with the FP8 KV cache unless the call is ``generation_only`` (which this target's layers
    never pass). The BF16-query build is not compiled here; a call that needs it runs the
    built-in FMHA inside a CUDA-graph capture (``MultiRequestCall.decline``) and compiles it on
    its first call outside one.

    Run after every other attention install. Every layer is checked before any is re-classed.
    Returns the number of layers re-classed.

    Raises:
        ValueError: if ``intercepting`` would not reach the built-in decode FMHA, the device is
            not sm_103 or sm_107, or a chosen layer's attention is not a tep4 rank's (32 query
            / 4 KV heads) or a tep8 rank's (16 / 2) heads of width 64 with the plain softmax
            scale and the trtllm-gen FMHA library, or a sliding-window layer's window is not
            2048, or the layers differ in rank shape.
    """
    reason = decode_reachable()
    if reason is not None:
        raise ValueError(
            f"[musespark_open] multi_request_split_kv cannot replace the decode FMHA: {reason}"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (10, 7),
    ):
        raise ValueError(
            "[musespark_open] the multi_request_split_kv kernel is built for sm_103 and sm_107 only"
        )
    chosen = [
        layer.self_attn for layer in model.layers if sliding or not layer.self_attn.is_sliding
    ]
    refused = []
    for attention in chosen:
        kv_heads = attention.num_key_value_heads
        reasons = [
            why
            for failed, why in (
                (
                    kv_heads not in _mr.RANK_KV_HEADS.values()
                    or attention.num_heads != _mr.Q_PER_KV * kv_heads,
                    f"{attention.num_heads} query / {kv_heads} KV heads per rank",
                ),
                (attention.head_dim != _mr.HEAD_DIM, f"head_dim is not {_mr.HEAD_DIM}"),
                (
                    get_bmm1_scale(attention.attn) != SOFTMAX_SCALE,
                    "its softmax scale is not 1 / sqrt(64)",
                ),
                (
                    trtllm_gen_fmha(attention.attn) is None,
                    "no trtllm-gen FMHA library in its FMHA manager",
                ),
                (
                    attention.is_sliding and attention.attention_window_size != WINDOW,
                    f"sliding window {attention.attention_window_size}, not {WINDOW}",
                ),
            )
            if failed
        ]
        if reasons:
            refused.append(f"layer {attention.layer_idx}: " + "; ".join(reasons))
    shapes = {attention.num_key_value_heads for attention in chosen}
    if not refused and len(shapes) > 1:
        refused.append(f"the layers mix {sorted(shapes)} KV heads per rank")
    if refused:
        raise ValueError(
            "[musespark_open] multi_request_split_kv cannot replace the decode FMHA: "
            + " | ".join(refused)
        )
    if not chosen:
        return 0
    (kv_heads,) = shapes
    device_index = torch.cuda.current_device()
    pdl = {bool(trtllm_gen_fmha(attention.attn)._enable_pdl) for attention in chosen}
    windows = {WINDOW if attention.is_sliding else 0 for attention in chosen}
    for enable_pdl in sorted(pdl):
        for window in sorted(windows):
            _mr.precompile(
                device_index, enable_pdl, kv_heads, window, q_fp8=True, device_scale=True
            )
    census = MultiRequestCensus(tuple(attention.layer_idx for attention in chosen))
    for attention in chosen:
        attention.__class__ = _mr_class(type(attention))
        attention._mr_window = WINDOW if attention.is_sliding else 0
        attention._mr_kv_heads = kv_heads
        attention._mr_census = census
        attention._mr_min_requests = max(2, min_requests)
    full = sum(1 for attention in chosen if not attention.is_sliding)
    logger.info(
        f"[musespark_open] multi_request_split_kv: on decode steps of {max(2, min_requests)} or "
        f"more sequences the "
        f"decode FMHA of {full} full-attention and {len(chosen) - full} sliding-window layers "
        f"runs multi_request_split_kv ({kv_heads} KV heads per rank, builds "
        f"pdl={sorted(pdl)} window={sorted(windows)} q=E4M3 + device scales)"
    )
    return len(chosen)
