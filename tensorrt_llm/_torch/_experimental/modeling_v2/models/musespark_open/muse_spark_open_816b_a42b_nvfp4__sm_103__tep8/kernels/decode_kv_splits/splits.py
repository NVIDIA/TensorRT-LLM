# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""decode_kv_splits: the multi-request decode FMHA over an over-subscribed KV split.

A decode step of two or more requests (8 rows per request) runs every
layer's attention through the built-in ``FlashInferTrtllmGenFmha.run_generation``,
which launches flashinfer's trtllm-gen decode FMHA. flashinfer picks the KV split
on the host from the SM count it reads (``flashinfer.decode`` calls
``get_device_sm_count``): one CTA per (request, KV head, split) with
``splits = SMs // (KV heads * requests)``, and the persistent one-CTA-per-request
kernel when that is 1. On a TEP4 rank (4 KV heads, 152 SMs) every step of 20 or
more requests runs the persistent kernel, and every step's time is set by its
longest request: a CTA holds a whole request's keys (or half of them at 13..19
requests) while the CTAs of the short requests of the step have long finished.

This path keeps the call chain and the kernel family and changes only the SM count
the selection reads, for the duration of one call: ``DecodeKvSplits.armed`` swaps
``flashinfer`` as seen by ``flashinfer_trtllm_gen`` for a stand-in for one decode
forward, and the stand-in's ``decode.trtllm_batch_decode_with_kv_cache`` presents
``splits * KV heads * requests`` SMs to flashinfer's selection, with ``splits``
chosen so that the grid holds about ``TARGET_CTAS_PER_SM`` CTAs per SM. The
over-subscribed grid keeps multi-CTA KV on at every batch size and lets the block
scheduler hand the splits of long requests to the SMs that short requests free.
flashinfer sizes the multi-CTA partials and counters from the same SM count
(``mMultiProcessorCount * stepQ`` partial rows, ``max(requests * heads, SMs)``
counters). The counters of a split call live in a buffer of this path's own,
zeroed once at install and sized for the largest armed call (the backend's
buffer holds ``max(heads * max_batch_size, SMs)`` counters, too few for the
presented count whenever ``max_batch_size`` is below about 24); the kernel
resets the counters it used at the end of every launch, so the calls of every
layer and step share it, as they share the backend's. The presented count is
checked against the call's workspace; a call that would not fit runs as it was.

Calls this path leaves alone: positional arguments, another backend, a query that
is not 8 rows per request, attention sinks, a scaled KV cache, and steps outside
72..512 rows (9 to 64 requests, ``takes``); sliding-window layers unless
``SLIDING_WINDOW`` (their keys are bounded by the window, so the persistent kernel
is not the bottleneck there). Up to 8 requests flashinfer already splits every
request's keys at least four times; there the over-subscribed grid only pays off
when the step's lengths are uneven, and costs time when they are even, so those
steps keep flashinfer's split.

Numerics: the same trtllm-gen kernel family as the steps of 2..19 requests run
without it (multi-CTA KV with the in-kernel fp32 merge of the split partials); only the
KV partition of each request changes, so every output element is the same
attention over the same FP8 keys and values summed in another fp32 order. Not
bitwise the persistent kernel's output; bounded against an fp32 reference and
checked on GSM8K.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from typing import Iterator, Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm.logger import logger

from ...attention_forward import decode_reachable
from ...dispatch import DRAFT_LEN

#: Rows of one request on a decode step: its last accepted token and the drafts.
ROWS_PER_REQUEST = DRAFT_LEN + 1
#: The decode steps this path arms: 9 to 64 requests.
MIN_ROWS = 9 * ROWS_PER_REQUEST
MAX_ROWS = 64 * ROWS_PER_REQUEST
#: CTAs per SM the split aims for.
TARGET_CTAS_PER_SM = 4
#: Whether the sliding-window layers' decode calls take the split too.
SLIDING_WINDOW = False
#: The query heads per KV head and the head width the selection rule is written for.
Q_PER_KV = 8
HEAD_DIM = 64
#: Bytes of multi-CTA scratch flashinfer carves per presented SM: the partial
#: softmax stats (one float2 per query row of a 64-row step) and the fp32 partial
#: output rows of width 64.
SCRATCH_BYTES_PER_SM = 64 * (8 + 4 * HEAD_DIM)
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


def splits_for(requests: int, kv_heads: int, sms: int) -> int:
    """The KV splits per (request, KV head): a grid of about ``TARGET_CTAS_PER_SM`` CTAs per SM."""
    return max(1, math.ceil(TARGET_CTAS_PER_SM * sms / (kv_heads * requests)))


def counter_bytes(requests: int, heads: int, sms: int) -> int:
    """flashinfer's multi-CTA counter bytes for ``requests`` x ``heads`` at ``sms`` SMs."""
    slots = max(requests * heads, sms)
    return (slots + 7) // 8 * 8 * 4


def counter_capacity(sms: int, kv_heads: int, q_heads: int) -> int:
    """Counter bytes of the largest call the rule splits, over every armed request count."""
    return max(
        counter_bytes(requests, q_heads, splits_for(requests, kv_heads, sms) * kv_heads * requests)
        for requests in range(MIN_ROWS // ROWS_PER_REQUEST, MAX_ROWS // ROWS_PER_REQUEST + 1)
    )


def selection_reachable() -> Optional[str]:
    """Why the stand-in could not present an SM count to flashinfer's selection, or ``None``.

    ``run_generation`` must read ``flashinfer`` as a global of its own module
    (``attention_forward.decode_reachable``), and flashinfer's
    ``trtllm_batch_decode_with_kv_cache`` must read ``get_device_sm_count`` as a global
    of ``flashinfer.decode``, the name the stand-in replaces for one call.
    """
    reason = decode_reachable()
    if reason is not None:
        return reason
    decode = getattr(_fi_gen.flashinfer, "decode", None)
    fn = getattr(decode, "trtllm_batch_decode_with_kv_cache", None)
    if fn is None or not callable(getattr(decode, "get_device_sm_count", None)):
        return "flashinfer.decode has no trtllm_batch_decode_with_kv_cache or get_device_sm_count"
    fn = inspect.unwrap(fn)
    if getattr(fn, "__globals__", None) is not vars(decode):
        return "flashinfer's trtllm_batch_decode_with_kv_cache is not flashinfer.decode's function"
    if "get_device_sm_count" not in fn.__code__.co_names:
        return "flashinfer's trtllm_batch_decode_with_kv_cache does not read get_device_sm_count"
    return None


class DecodeKvSplits:
    """The split rule of one model's decode steps and the stand-in that applies it.

    ``sms``: the device's SM count; ``kv_heads`` / ``q_heads``: a rank's attention heads;
    ``device``: where the split calls' counter buffer (``counter``) lives. Build it
    before any CUDA graph capture: the buffer is allocated here.
    """

    def __init__(
        self, sms: int, kv_heads: int, q_heads: int, device: Optional[torch.device] = None
    ) -> None:
        self.sms = sms
        self.kv_heads = kv_heads
        self.q_heads = q_heads
        self.fallback: Optional[str] = None
        #: Decode calls that reached the armed stand-in: how many were split, how many kept
        #: flashinfer's own split.
        self.split_calls = 0
        self.kept_calls = 0
        #: The multi-CTA counters of every call the rule splits (zeroed once; the
        #: kernel resets what it used at the end of each launch).
        self.counter = torch.zeros(
            counter_capacity(sms, kv_heads, q_heads), dtype=torch.uint8, device=device
        )

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows is armed (9 to 64 requests)."""
        return MIN_ROWS <= rows <= MAX_ROWS and rows % ROWS_PER_REQUEST == 0

    def presented_sm_count(self, kwargs: dict) -> Optional[int]:
        """The SM count to present for this decode FMHA call, or ``None`` to run it as it is.

        Host-side reads only: under a CUDA graph each is a per-capture constant.
        """
        if set(kwargs) != DECODE_KWARGS:
            return None
        query = kwargs["query"]
        kv_cache = kwargs["kv_cache"]
        if (
            kwargs["backend"] != "trtllm-gen"
            or kwargs["q_len_per_req"] != ROWS_PER_REQUEST
            or kwargs["sinks"] is not None
            or kwargs["kv_cache_sf"] is not None
            or (kwargs["window_left"] != -1 and not SLIDING_WINDOW)
            or not isinstance(kv_cache, tuple)
            or query.dim() != 3
            or query.shape[1] != self.q_heads
        ):
            return None
        rows = query.shape[0]
        if not self.takes(rows):
            return None
        requests = rows // ROWS_PER_REQUEST
        default = max(1, self.sms // (self.kv_heads * requests))
        splits = splits_for(requests, self.kv_heads, self.sms)
        if splits <= default:
            return None
        presented = splits * self.kv_heads * requests
        workspace = kwargs["workspace_buffer"]
        if workspace.numel() * workspace.element_size() < presented * SCRATCH_BYTES_PER_SM:
            self._decline(f"the FMHA workspace is too small for {presented} SMs")
            return None
        window = "full" if kwargs["window_left"] == -1 else "sliding-window"
        logger.info_once(
            f"[musespark_open] decode_kv_splits: {window} decode FMHA of {requests} requests over "
            f"{splits} KV splits per request and KV head ({presented} CTAs; flashinfer alone: "
            f"{default} split{'s' if default > 1 else ''})",
            key=f"musespark_open.decode_kv_splits.{window}.{requests}",
        )
        return presented

    def _decline(self, reason: str) -> None:
        if self.fallback is None:
            self.fallback = reason
        logger.warning_once(
            "[musespark_open] decode_kv_splits: a decode FMHA call keeps flashinfer's split "
            f"({reason})",
            key=f"musespark_open.decode_kv_splits.decline_{reason}",
        )

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Have every trtllm-gen decode FMHA call of the forward go through the split rule."""
        real = _fi_gen.flashinfer
        _fi_gen.flashinfer = _FlashinferStandIn(real, self)
        try:
            yield
        finally:
            _fi_gen.flashinfer = real


class _DecodeStandIn:
    """``flashinfer.decode`` whose trtllm-gen decode call sees the rule's SM count."""

    def __init__(self, real, splits: DecodeKvSplits) -> None:
        self._real = real
        self._splits = splits

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def trtllm_batch_decode_with_kv_cache(self, *args, **kwargs):
        presented = None if args else self._splits.presented_sm_count(kwargs)
        if presented is None:
            self._splits.kept_calls += 1
            return self._real.trtllm_batch_decode_with_kv_cache(*args, **kwargs)
        self._splits.split_calls += 1
        kwargs["multi_ctas_kv_counter_buffer"] = self._splits.counter
        decode = self._real
        device_sm_count = decode.get_device_sm_count
        decode.get_device_sm_count = lambda device: presented
        try:
            return decode.trtllm_batch_decode_with_kv_cache(**kwargs)
        finally:
            decode.get_device_sm_count = device_sm_count


class _FlashinferStandIn:
    """``flashinfer`` whose ``decode`` is ``_DecodeStandIn`` while a forward is armed."""

    def __init__(self, real, splits: DecodeKvSplits) -> None:
        self._real = real
        self.decode = _DecodeStandIn(real.decode, splits)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def install(model: torch.nn.Module) -> DecodeKvSplits:
    """The split rule for ``model``'s decode steps, after checking it can apply.

    Raises:
        ValueError: if the stand-in would not reach flashinfer's selection
            (``selection_reachable``), the device is not sm_103, or the layers' attention
            is not ``Q_PER_KV`` query heads per KV head of width ``HEAD_DIM`` with one
            rank shape.
    """
    reason = selection_reachable()
    if reason is not None:
        raise ValueError(
            "[musespark_open] decode_kv_splits cannot reach flashinfer's KV-split selection: "
            f"{reason}"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("[musespark_open] decode_kv_splits is written for sm_103")
    shapes = {
        (layer.self_attn.num_heads, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)
        for layer in model.layers
    }
    if len(shapes) != 1:
        raise ValueError(f"[musespark_open] decode_kv_splits: the layers mix rank shapes {shapes}")
    ((q_heads, kv_heads, head_dim),) = shapes
    if head_dim != HEAD_DIM or q_heads != Q_PER_KV * kv_heads:
        raise ValueError(
            f"[musespark_open] decode_kv_splits: {q_heads} query / {kv_heads} KV heads of width "
            f"{head_dim} per rank, not {Q_PER_KV} query heads per KV head of width {HEAD_DIM}"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    splits = DecodeKvSplits(sms, kv_heads, q_heads, device)
    layers = "every" if SLIDING_WINDOW else "the full-attention"
    logger.info(
        f"[musespark_open] decode_kv_splits: decode steps of {MIN_ROWS}..{MAX_ROWS} rows run "
        f"{layers} layers' trtllm-gen decode FMHA over about {TARGET_CTAS_PER_SM} CTAs per SM "
        f"({sms} SMs, {kv_heads} KV heads per rank; {splits.counter.numel()} bytes of own "
        f"multi-CTA counters)"
    )
    return splits
