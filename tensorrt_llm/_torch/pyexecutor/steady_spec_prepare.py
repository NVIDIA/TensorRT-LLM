# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state one-model MTP decode input preparation.

A decode step whose batch is the previous step's (same requests and slots, each verifying the previous
step's device drafts, same CUDA graph metadata, no padding) needs only a few new integers per request. The
preparer records a full ``DecoderInputPreparer.prepare_inputs`` pass of such a step (``record_batch``); later
matching steps write those integers to a pinned host slab (3-slot ring, fenced by events) and one Triton
kernel stages every device input from it, including the CUDA graph's static inputs. The attention
metadata's ``prepare_steady_decode_step`` stands in for its ``prepare()`` where it supports it.

A model with ``steady_decode_prepare`` stages its steps with a kernel that issues its loads before its
stores and also takes CUDA graph padding rows, the previous step's output rows and, with attention TP on
SM100/SM103, the graph's Engram pre-replay refresh (hook positions, history scatter, every layer's hashes;
the replay then skips its hooks). Its passes with no previous step on the device are recorded too, and a
step whose requests all join on the recorded metadata and padding rows (their first step on this worker:
the last token and the drafts from the host, no previous step to correct) is staged the same way
(``admits``): the slab also carries those tokens, and the record then describes the joined batch.

Whether a full pass qualifies is decided when it is recorded (``DecoderInputPreparer._record_steady_mtp``);
a later step matches when it runs the recorded requests on the recorded slots and metadata. Any other step
(batch or slot change, context rows, attention-DP dummies, multimodal, LoRA, guided decoding, beams, helix
CP, warmup) takes the full pass, which records again when it qualifies.

With ``stage_rng`` (``RuntimeHooks.pack_spec_rng_uploads``) and zero-copy slabs the same kernel also writes the
step's Philox state for sampling (``SpecMetadata.request_seeds`` / ``request_offsets`` / ``seeds`` /
``offsets``) from a second pinned ring, instead of four host-to-device copies on the critical stream. The host
computes the same values and advances the same offset windows (``SpecMetadata._populate_request_rng_state``); only
the delivery changes.
"""

import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.kernels.blackwell.engram.steady_refresh import scatter_and_hash_decode_rows
from tensorrt_llm._utils import current_cuda_stream, is_blackwell_datacenter, prefer_pinned
from tensorrt_llm.logger import logger

from ..attention.backends.interface import AttentionMetadata
from ..attention.backends.trtllm import TrtllmAttentionMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec, StepOutputRows
from .llm_request import get_draft_token_length

if TYPE_CHECKING:
    from ..modules.engram.engram import DecodeRefreshPlan, EngramHashProvider
    from ..speculative.interface import SpecMetadata
    from .cuda_graph_runner import CUDAGraphRunner
    from .engine.runners.decoder.inputs import _SteadyStepInputs
    from .llm_request import LlmRequest

__all__ = ["SteadyMtpPrepare", "SteadyMtpRecord", "stage_steady_inputs", "steady_glue_enabled"]

# Depth of the host slab ring.
_RING = 3


def _staging_words(num_rows: int, num_tokens: int) -> int:
    """Worst-case slab words before seed cells: slots, KV and prompt lengths,
    history rows, positions, and tokens for an admission."""
    return 4 * num_rows + 2 * num_tokens


# Integer args unspecialized, pointers not specialized on alignment, one BLOCK for every batch size: one variant
# per engine, compiled during warmup instead of at each new batch size.
_STAGE_INT_ARGS = [
    "nnt_tok_stride",
    "nnt_slot_stride",
    "nd_slot_stride",
    "mrope_axis_stride",
    "static_pos_axis_stride",
    "rng_span",
    "B",
]
_STAGE_PTR_ARGS = [
    "slab",
    "next_new_tokens",
    "next_drafts",
    "new_lens",
    "input_ids",
    "draft_tokens",
    "prev_batch_idx",
    "prev_pos_idx",
    "prev_pos_offsets",
    "prev_kv_offsets",
    "positions",
    "mrope_positions",
    "gather_ids",
    "kv_lens_out",
    "static_ids",
    "static_pos",
    "rng_slab",
    "req_seeds_out",
    "req_offsets_out",
    "seeds_out",
    "offsets_out",
]
_DECODE_STAGE_INT_ARGS = [
    *_STAGE_INT_ARGS,
    "nd_col_stride",
    "B_ALL",
    "history_columns",
    "lookup_max",
    "pad_id",
    "dead",
    "hash_rows",
    "num_seeds",
]
_DECODE_STAGE_PTR_ARGS = [
    *_STAGE_PTR_ARGS,
    "prompt_lens_out",
    "hook_pos",
    "rows",
    "lookup",
    "history",
    "multipliers",
    "moduli",
    "hashes",
]


def steady_glue_enabled(enable_attention_dp: bool) -> bool:
    """The staging launch also does the Engram pre-replay refresh: attention TP on SM100/SM103."""
    return not enable_attention_dp and is_blackwell_datacenter()


def _stage_jit(**kwargs):
    try:
        return triton.jit(**kwargs)
    except TypeError:  # a Triton without do_not_specialize_on_alignment
        kwargs.pop("do_not_specialize_on_alignment", None)
        return triton.jit(**kwargs)


@_stage_jit(do_not_specialize=_STAGE_INT_ARGS, do_not_specialize_on_alignment=_STAGE_PTR_ARGS)
def _steady_stage_kernel(
    slab,  # int32 [slots (B) | positions (B * T) | kv lens (B, HAS_KV)]
    next_new_tokens,  # [>= T, num_slots, 1] previous step's next-step input tokens
    nnt_tok_stride,
    nnt_slot_stride,
    next_drafts,  # [num_slots, >= D]
    nd_slot_stride,
    new_lens,  # [num_slots] previous step's accepted-token counts
    input_ids,
    draft_tokens,
    prev_batch_idx,
    prev_pos_idx,
    prev_pos_offsets,
    prev_kv_offsets,
    positions,
    mrope_positions,  # [3, 1, max_tokens] or a dummy when HAS_MROPE is False
    mrope_axis_stride,
    gather_ids,
    kv_lens_out,  # [B] kv_lens_cuda (HAS_KV): slab[2B + B*T + r]
    static_ids,  # graph static input ids (HAS_STATIC)
    static_pos,  # graph static positions, [3, 1, M] (mrope) or [1, M]
    static_pos_axis_stride,
    rng_slab,  # int64 [2 * B] (HAS_RNG): each request's Philox seed, then its offset-window base
    req_seeds_out,  # int64 [B] SpecMetadata.request_seeds (HAS_RNG)
    req_offsets_out,  # int64 [B] SpecMetadata.request_offsets (HAS_RNG)
    seeds_out,  # int64 [B * T] SpecMetadata.seeds, one per logits row (HAS_RNG)
    offsets_out,  # int64 [B * T] SpecMetadata.offsets (HAS_RNG)
    rng_span,  # Philox offsets between a request's consecutive rows (_RNG_SLOT_SPAN)
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    HAS_KV: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    HAS_RNG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    # token rows: request r = i // T, position j = i % T
    tok = i < B * T
    r = i // T
    j = i % T
    slot = tl.load(slab + r, mask=tok, other=0)
    pos = tl.load(slab + B + i, mask=tok, other=0)
    ids = tl.load(next_new_tokens + j * nnt_tok_stride + slot * nnt_slot_stride, mask=tok, other=0)
    tl.store(input_ids + i, ids, mask=tok)
    tl.store(positions + i, pos, mask=tok)
    if HAS_STATIC:
        tl.store(static_ids + i, ids, mask=tok)
        tl.store(static_pos + i, pos, mask=tok)
        if HAS_MROPE:
            tl.store(static_pos + static_pos_axis_stride + i, pos, mask=tok)
            tl.store(static_pos + 2 * static_pos_axis_stride + i, pos, mask=tok)
    if HAS_MROPE:
        tl.store(mrope_positions + i, pos, mask=tok)
        tl.store(mrope_positions + mrope_axis_stride + i, pos, mask=tok)
        tl.store(mrope_positions + 2 * mrope_axis_stride + i, pos, mask=tok)
    tl.store(gather_ids + i, i, mask=tok)
    tl.store(prev_pos_idx + i, slot, mask=tok)
    lens = tl.load(new_lens + slot, mask=tok, other=0)
    tl.store(prev_pos_offsets + i, lens, mask=tok)
    # request rows
    req = i < B
    slot_r = tl.load(slab + i, mask=req, other=0)
    tl.store(prev_batch_idx + i, slot_r, mask=req)
    lens_r = tl.load(new_lens + slot_r, mask=req, other=0)
    tl.store(prev_kv_offsets + i, lens_r - T, mask=req)
    if HAS_KV:
        tl.store(kv_lens_out + i, tl.load(slab + B + B * T + i, mask=req, other=0), mask=req)
    if HAS_RNG:
        # The layout SpecMetadata._populate_request_rng_state uploads: request_* per request, and per logits
        # row (request r = i // T, row j = i % T) the request's seed and offset_r + j * rng_span.
        tl.store(req_seeds_out + i, tl.load(rng_slab + i, mask=req, other=0), mask=req)
        tl.store(req_offsets_out + i, tl.load(rng_slab + B + i, mask=req, other=0), mask=req)
        row_seed = tl.load(rng_slab + r, mask=tok, other=0)
        row_off = tl.load(rng_slab + B + r, mask=tok, other=0)
        tl.store(seeds_out + i, row_seed, mask=tok)
        tl.store(offsets_out + i, row_off + j.to(tl.int64) * rng_span, mask=tok)
    # draft rows: request i // D, draft i % D
    drf = i < B * D
    slot_d = tl.load(slab + i // D, mask=drf, other=0)
    dval = tl.load(next_drafts + slot_d * nd_slot_stride + i % D, mask=drf, other=0)
    tl.store(draft_tokens + i, dval, mask=drf)


# The staging kernel of a model with steady_decode_prepare: also padding rows, first steps from the
# host, the previous step's output rows and the Engram refresh, with every load before the stores.
@_stage_jit(
    do_not_specialize=_DECODE_STAGE_INT_ARGS,
    do_not_specialize_on_alignment=_DECODE_STAGE_PTR_ARGS,
)
def _steady_decode_stage_kernel(
    slab,  # int32 [slots (B) | positions (B_ALL*T) | kv lens (B_ALL, HAS_KV) | tokens (B*T, FROM_HOST) | prompt
    # lengths (B_ALL, HAS_PROMPT) | history rows (B) and seeds (4 x num_seeds) (GLUE and FROM_HOST)]
    # The previous step's next-step input tokens [>= T, num_slots, 1], drafts [num_slots, >= D] and
    # accepted-token counts [num_slots]; with BY_ROW its forward outputs, one row per request instead.
    next_new_tokens,
    nnt_tok_stride,
    nnt_slot_stride,
    next_drafts,
    nd_slot_stride,
    nd_col_stride,
    new_lens,
    input_ids,
    draft_tokens,
    prev_batch_idx,
    prev_pos_idx,
    prev_pos_offsets,
    prev_kv_offsets,
    positions,
    mrope_positions,  # [3, 1, max_tokens] or a dummy when HAS_MROPE is False
    mrope_axis_stride,
    gather_ids,
    kv_lens_out,  # [B_ALL] kv_lens_cuda (HAS_KV): slab[B + B_ALL*T + r]
    prompt_lens_out,  # [B_ALL] prompt_lens_cuda (HAS_PROMPT)
    static_ids,  # graph static input ids (HAS_STATIC)
    static_pos,  # graph static positions, [3, 1, M] (mrope) or [1, M]
    static_pos_axis_stride,
    rng_slab,  # int64 [2 * B_ALL] (HAS_RNG): each request's Philox seed, then its offset-window base
    req_seeds_out,  # int64 [B_ALL] SpecMetadata.request_seeds (HAS_RNG)
    req_offsets_out,  # int64 [B_ALL] SpecMetadata.request_offsets (HAS_RNG)
    seeds_out,  # int64 [B_ALL * T] SpecMetadata.seeds, one per logits row (HAS_RNG)
    offsets_out,  # int64 [B_ALL * T] SpecMetadata.offsets (HAS_RNG)
    rng_span,  # Philox offsets between a request's consecutive rows (_RNG_SLOT_SPAN)
    B,  # requests verifying their previous drafts (FROM_HOST: their host drafts)
    B_ALL,  # B plus the CUDA graph padding rows after them
    hook_pos,  # GLUE: the pre-replay hooks' positions [>= B_ALL * T]
    rows,  # GLUE: int64 Engram history row of each token
    lookup,  # GLUE: int64 compressed-token lookup table
    history,  # GLUE: int32 [history rows, history_columns]
    history_columns,
    lookup_max,
    pad_id,
    dead,
    multipliers,  # GLUE: int64 [NUM_LAYERS, MAX_NGRAM]
    moduli,  # GLUE: int64 [NUM_LAYERS, MAX_NGRAM - 1, N_HEAD]
    hashes,  # GLUE: int64 [NUM_LAYERS, hash_rows, (MAX_NGRAM - 1) * N_HEAD]
    hash_rows,
    num_seeds,  # GLUE and FROM_HOST: history seed cells
    T: tl.constexpr,
    D: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    HAS_KV: tl.constexpr,
    HAS_PROMPT: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    HAS_RNG: tl.constexpr,
    GLUE: tl.constexpr,
    FROM_HOST: tl.constexpr,
    BY_ROW: tl.constexpr,
    N_HEAD: tl.constexpr,
    MAX_NGRAM: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
    # Powers of two covering MAX_NGRAM, NUM_LAYERS and NUM_LAYERS * (MAX_NGRAM - 1) (GLUE).
    BLOCK_NGRAM: tl.constexpr,
    BLOCK_LAYERS: tl.constexpr,
    BLOCK_ORDERS: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    # token rows: request r = i // T, position j = i % T; padding rows read token 0 and no offsets
    tok = i < B_ALL * T
    real = i < B * T
    r = i // T
    j = i % T
    # request rows; draft rows: request i // D, draft i % D
    req = i < B_ALL
    req_real = i < B
    drf = i < B * D
    # Loads precede the stores they might alias, except an admission's history reads after its seeds.
    slot = tl.load(slab + r, mask=real, other=0)
    pos = tl.load(slab + B + i, mask=tok, other=0)
    host_tokens = slab + B + B_ALL * T
    if HAS_KV:
        kv_lens = tl.load(slab + B + B_ALL * T + i, mask=req, other=0)
        host_tokens += B_ALL
    if HAS_PROMPT:
        prompt_lens = tl.load(host_tokens + B * T + i, mask=req, other=0)
    if FROM_HOST:
        ids = tl.load(host_tokens + i, mask=real, other=0)
        # No previous step: zero offsets, and the previous-step indices stay unread.
        lens = tl.zeros([BLOCK], dtype=tl.int32)
        dval = tl.load(host_tokens + (i // D) * T + 1 + i % D, mask=drf, other=0)
    else:
        slot_r = tl.load(slab + i, mask=req_real, other=0)
        # Where each row reads the previous step: its request's output row, or its slot.
        if BY_ROW:
            src = r
            src_r = i
            src_d = i // D
        else:
            src = slot
            src_r = slot_r
            src_d = tl.load(slab + i // D, mask=drf, other=0)
        ids = tl.load(
            next_new_tokens + j * nnt_tok_stride + src * nnt_slot_stride, mask=real, other=0
        )
        lens = tl.load(new_lens + src, mask=real, other=0)
        lens_r = tl.load(new_lens + src_r, mask=req_real, other=T)
        dval = tl.load(
            next_drafts + src_d * nd_slot_stride + (i % D) * nd_col_stride, mask=drf, other=0
        )
    if HAS_RNG:
        req_seed = tl.load(rng_slab + i, mask=req, other=0)
        req_offset = tl.load(rng_slab + B_ALL + i, mask=req, other=0)
        row_seed = tl.load(rng_slab + r, mask=tok, other=0)
        row_offset = tl.load(rng_slab + B_ALL + r, mask=tok, other=0)
    if GLUE:
        # The hook positions, then the Engram refresh as EngramDecodeRefreshKernel computes it
        # (static ids and hook positions as the token inputs; no padding rows, B == B_ALL).
        ks = tl.arange(0, BLOCK_NGRAM)
        hooked = pos + lens.to(tl.int32)
        position = hooked.to(tl.int64)
        if FROM_HOST:
            # Joining requests: each token's history row (kept as the batch's row map) and the
            # history seeds (row, position, raw id, text flag).
            joined = host_tokens + B * T
            if HAS_PROMPT:
                joined += B_ALL
            row = tl.load(joined + r, mask=real, other=0).to(tl.int64)
            seeds = joined + B
            seeded = i < num_seeds
            seed_row = tl.load(seeds + i, mask=seeded, other=0).to(tl.int64)
            seed_position = tl.load(seeds + num_seeds + i, mask=seeded, other=0).to(tl.int64)
            seed_raw = tl.load(seeds + 2 * num_seeds + i, mask=seeded, other=0).to(tl.int64)
            seed_text = tl.load(seeds + 3 * num_seeds + i, mask=seeded, other=0)
            seed_lookup = tl.load(lookup + seed_raw, mask=seeded, other=0)
        else:
            row = tl.load(rows + i, mask=real, other=0)
        cells = history + row * history_columns
        raw = tl.minimum(tl.maximum(ids.to(tl.int32).to(tl.int64), 0), lookup_max)
        compressed = tl.load(lookup + raw, mask=real, other=0)
        # Each token's n-gram cells: offset k is token j - k of this step (j >= k), else an earlier step's cell.
        back = j[:, None] - ks[None, :]
        in_window = real[:, None] & (ks[None, :] < MAX_NGRAM)
        here = in_window & (back >= 0)
        if FROM_HOST:
            window_ids = tl.load(host_tokens + i[:, None] - ks[None, :], mask=here, other=0)
        else:
            window_ids = tl.load(
                next_new_tokens + back * nnt_tok_stride + src[:, None] * nnt_slot_stride,
                mask=here,
                other=0,
            )
        window_raw = tl.minimum(tl.maximum(window_ids.to(tl.int32).to(tl.int64), 0), lookup_max)
        window_compressed = tl.load(lookup + window_raw, mask=here, other=0)
        cell = position[:, None] - ks[None, :]
        in_range = (cell >= 0) & (cell < history_columns)
        earlier = in_window & (back < 0) & in_range
        if not FROM_HOST:
            window_history = tl.load(
                cells[:, None] + tl.maximum(cell, 0), mask=earlier, other=dead
            ).to(tl.int64)
    tl.store(input_ids + i, ids, mask=tok)
    tl.store(positions + i, pos, mask=tok)
    if HAS_STATIC:
        tl.store(static_ids + i, ids, mask=tok)
        tl.store(static_pos + i, pos, mask=tok)
        if HAS_MROPE:
            tl.store(static_pos + static_pos_axis_stride + i, pos, mask=tok)
            tl.store(static_pos + 2 * static_pos_axis_stride + i, pos, mask=tok)
    if HAS_MROPE:
        tl.store(mrope_positions + i, pos, mask=tok)
        tl.store(mrope_positions + mrope_axis_stride + i, pos, mask=tok)
        tl.store(mrope_positions + 2 * mrope_axis_stride + i, pos, mask=tok)
    tl.store(gather_ids + i, i, mask=tok)
    if not FROM_HOST:
        tl.store(prev_pos_idx + i, slot, mask=real)
    tl.store(prev_pos_offsets + i, lens, mask=tok)
    if FROM_HOST:
        tl.store(prev_kv_offsets + i, lens, mask=req)
    else:
        tl.store(prev_batch_idx + i, slot_r, mask=req_real)
        tl.store(prev_kv_offsets + i, lens_r - T, mask=req)
    if HAS_KV:
        tl.store(kv_lens_out + i, kv_lens, mask=req)
    if HAS_PROMPT:
        tl.store(prompt_lens_out + i, prompt_lens, mask=req)
    if HAS_RNG:
        tl.store(req_seeds_out + i, req_seed, mask=req)
        tl.store(req_offsets_out + i, req_offset, mask=req)
        tl.store(seeds_out + i, row_seed, mask=tok)
        tl.store(offsets_out + i, row_offset + j.to(tl.int64) * rng_span, mask=tok)
    tl.store(draft_tokens + i, dval, mask=drf)
    if GLUE:
        tl.store(hook_pos + i, hooked, mask=tok)
        if FROM_HOST:
            tl.store(rows + i, row, mask=real)
            seed_value = tl.where(seed_text != 0, seed_lookup, dead)
            tl.store(
                history + seed_row * history_columns + seed_position,
                seed_value.to(tl.int32),
                mask=seeded,
            )
            # The earlier cells include the seeds stored above.
            tl.debug_barrier()
            window_history = tl.load(
                cells[:, None] + tl.maximum(cell, 0), mask=earlier, other=dead
            ).to(tl.int64)
        # This step's cells as the history holds them (int32); dead out of range.
        window = tl.where(
            here,
            tl.where(in_range, window_compressed.to(tl.int32).to(tl.int64), dead),
            window_history,
        )
        scatter_and_hash_decode_rows(
            cells,
            history_columns,
            position,
            compressed,
            window,
            multipliers,
            moduli,
            hashes,
            hash_rows,
            i,
            real,
            pad_id,
            dead,
            N_HEAD,
            MAX_NGRAM,
            NUM_LAYERS,
            BLOCK_HEAD,
            BLOCK_NGRAM,
            BLOCK_LAYERS,
            BLOCK_ORDERS,
        )


def stage_steady_inputs(
    slab: torch.Tensor,
    b: int,
    t: int,
    new_tensors_device: Any,
    *,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    prev_batch_idx: torch.Tensor,
    prev_pos_idx: torch.Tensor,
    prev_pos_offsets: torch.Tensor,
    prev_kv_offsets: torch.Tensor,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor | None,
    gather_ids: torch.Tensor,
    kv_lens_cuda: torch.Tensor | None = None,
    static_input_ids: torch.Tensor | None = None,
    static_position_ids: torch.Tensor | None = None,
    block: int | None = None,
    rng_slab: torch.Tensor | None = None,
    rng_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    rng_span: int = 0,
    loads_first: bool = False,
    num_rows: int | None = None,
    hook_positions: torch.Tensor | None = None,
    engram: "DecodeRefreshPlan | None" = None,
    from_host: bool = False,
    prompt_lens_cuda: torch.Tensor | None = None,
    num_seeds: int = 0,
    rows: "StepOutputRows | None" = None,
) -> None:
    """One launch of the steady staging kernel.

    Args:
        slab: int32 [b + num_rows * t (+ num_rows with ``kv_lens_cuda``) (+ b * t with
            ``from_host``) (+ num_rows with ``prompt_lens_cuda``) (+ b + 4 * num_seeds with
            ``engram`` and ``from_host``)]: the sequence slots of the requests, the positions and
            the KV lengths of every row, each request's last token and drafts, the prompt lengths,
            and the joining requests' Engram history rows and seed cells; a device tensor or a
            pinned host tensor (read through its device mapping).
        b: Number of requests verifying their previous drafts (with ``from_host``: their host drafts).
        t: Tokens per request (draft length + 1).
        new_tensors_device: The previous step's device sampler tensors (``new_tokens``
            [>= t, num_slots, 1], ``next_draft_tokens`` [num_slots, >= t - 1],
            ``new_tokens_lens`` [num_slots]); not read with ``from_host``.
        input_ids, draft_tokens, prev_batch_idx, prev_pos_idx, prev_pos_offsets, prev_kv_offsets,
            positions, gather_ids: The decoder input buffers the full pass writes.
        mrope_positions: The [3, 1, N] mrope position buffer, or None without mrope.
        kv_lens_cuda: ``TrtllmAttentionMetadata.kv_lens_cuda`` when the slab carries the KV lengths.
        static_input_ids, static_position_ids: The CUDA graph's static inputs, when staged here.
        block: The BLOCK of the launch when it covers the batch (one variant for all batch sizes),
            else the batch's own power of two.
        rng_slab: int64 [2 * num_rows]: each request's Philox seed, then its offset-window base (a pinned
            host tensor read through its device mapping), or None to leave the Philox state alone.
        rng_buffers: ``SpecMetadata``'s (request_seeds, request_offsets, seeds, offsets), written
            when ``rng_slab`` is given.
        rng_span: Philox offsets between consecutive rows of a request (``_RNG_SLOT_SPAN``).
        loads_first: Launch the kernel of a model with ``steady_decode_prepare``, which issues every
            load before its stores. The arguments below need it.
        num_rows: ``b`` plus the CUDA graph padding rows after them (default ``b``), which take
            input token 0, their slab positions and KV lengths, and zero acceptance offsets.
        hook_positions, engram: The hooks' position buffer and a ``DecodeRefreshPlan``. With both
            (and the static inputs, ``num_rows == b``) the launch also writes the hooks' positions,
            static positions plus ``prev_pos_offsets``, and runs the plan on the static ids.
        from_host: The requests' first step: the tokens come from the slab, the previous-step
            offsets are zero and the previous-step indices are left as they are. With ``engram``
            the launch also writes the plan's row map and ``num_seeds`` history seed cells.
        prompt_lens_cuda: ``TrtllmAttentionMetadata.prompt_lens_cuda`` (with ``from_host``) when
            the slab carries the prompt lengths.
        rows: The previous step's forward outputs for these requests in order, read instead of
            ``new_tensors_device`` (int32, at least ``t`` tokens and ``t - 1`` drafts per row).
    """
    d = t - 1
    num_rows = b if num_rows is None else num_rows
    need = max(num_rows * t, b * d, num_rows, 1)
    if block is None or block < need:
        block = triton.next_power_of_2(need)
    has_mrope = mrope_positions is not None
    has_rng = rng_slab is not None
    if has_rng:
        req_seeds_out, req_offsets_out, seeds_out, offsets_out = rng_buffers
    else:
        # Untouched without HAS_RNG; one dtype for every launch keeps a single variant.
        rng_slab = req_seeds_out = req_offsets_out = seeds_out = offsets_out = gather_ids
    if not loads_first:
        assert num_rows == b and not from_host and engram is None and rows is None
        assert prompt_lens_cuda is None
        new_tokens = new_tensors_device.new_tokens
        next_drafts = new_tensors_device.next_draft_tokens
        _steady_stage_kernel[(1,)](
            slab,
            new_tokens,
            new_tokens.stride(0),
            new_tokens.stride(1),
            next_drafts,
            next_drafts.stride(0),
            new_tensors_device.new_tokens_lens,
            input_ids,
            draft_tokens,
            prev_batch_idx,
            prev_pos_idx,
            prev_pos_offsets,
            prev_kv_offsets,
            positions,
            mrope_positions if has_mrope else positions,
            mrope_positions.stride(0) if has_mrope else 0,
            gather_ids,
            kv_lens_cuda if kv_lens_cuda is not None else gather_ids,
            static_input_ids if static_input_ids is not None else gather_ids,
            static_position_ids if static_position_ids is not None else gather_ids,
            static_position_ids.stride(0) if (static_position_ids is not None and has_mrope) else 0,
            rng_slab,
            req_seeds_out,
            req_offsets_out,
            seeds_out,
            offsets_out,
            rng_span,
            b,
            T=t,
            D=d,
            HAS_MROPE=has_mrope,
            HAS_KV=kv_lens_cuda is not None,
            HAS_STATIC=static_input_ids is not None,
            HAS_RNG=has_rng,
            BLOCK=block,
        )
        return
    if from_host:
        # Stand-ins for the unread previous-step tensors: one kernel variant with or without them.
        previous = (slab, 0, 0, slab, 0, 0, slab)
    elif rows is not None:
        tokens, drafts = rows.next_new_tokens, rows.next_draft_tokens
        previous = (
            tokens,
            tokens.stride(1),
            tokens.stride(0),
            drafts,
            drafts.stride(0),
            drafts.stride(1),
            rows.new_tokens_lens,
        )
    else:
        new_tokens = new_tensors_device.new_tokens
        next_drafts = new_tensors_device.next_draft_tokens
        previous = (
            new_tokens,
            new_tokens.stride(0),
            new_tokens.stride(1),
            next_drafts,
            next_drafts.stride(0),
            next_drafts.stride(1),
            new_tensors_device.new_tokens_lens,
        )
    glue = engram is not None
    assert (prompt_lens_cuda is None and not num_seeds) or from_host
    if glue:
        assert hook_positions is not None and num_rows == b and static_input_ids is not None
        history = engram.history
        glue_args = (
            hook_positions,
            engram.rows,
            engram.lookup,
            history,
            history.shape[1],
            engram.lookup.shape[0] - 1,
            engram.pad_id,
            engram.dead,
            engram.multipliers,
            engram.moduli,
            engram.hashes,
            engram.hashes.shape[1],
            num_seeds,
        )
        layers = engram.multipliers.shape[0]
        glue_constants = dict(
            N_HEAD=engram.n_head,
            MAX_NGRAM=engram.max_ngram,
            NUM_LAYERS=layers,
            BLOCK_HEAD=triton.next_power_of_2(engram.n_head),
            BLOCK_NGRAM=triton.next_power_of_2(engram.max_ngram),
            BLOCK_LAYERS=triton.next_power_of_2(layers),
            BLOCK_ORDERS=triton.next_power_of_2(layers * (engram.max_ngram - 1)),
        )
    else:
        glue_args = (gather_ids, gather_ids, gather_ids, gather_ids, 1, 0, 0, 0)
        glue_args += (gather_ids, gather_ids, gather_ids, 0, 0)
        glue_constants = dict(
            N_HEAD=1,
            MAX_NGRAM=2,
            NUM_LAYERS=1,
            BLOCK_HEAD=1,
            BLOCK_NGRAM=2,
            BLOCK_LAYERS=1,
            BLOCK_ORDERS=1,
        )
    _steady_decode_stage_kernel[(1,)](
        slab,
        *previous,
        input_ids,
        draft_tokens,
        prev_batch_idx,
        prev_pos_idx,
        prev_pos_offsets,
        prev_kv_offsets,
        positions,
        mrope_positions if has_mrope else positions,
        mrope_positions.stride(0) if has_mrope else 0,
        gather_ids,
        kv_lens_cuda if kv_lens_cuda is not None else gather_ids,
        prompt_lens_cuda if prompt_lens_cuda is not None else gather_ids,
        static_input_ids if static_input_ids is not None else gather_ids,
        static_position_ids if static_position_ids is not None else gather_ids,
        static_position_ids.stride(0) if (static_position_ids is not None and has_mrope) else 0,
        rng_slab,
        req_seeds_out,
        req_offsets_out,
        seeds_out,
        offsets_out,
        rng_span,
        b,
        num_rows,
        *glue_args,
        T=t,
        D=d,
        HAS_MROPE=has_mrope,
        HAS_KV=kv_lens_cuda is not None,
        HAS_PROMPT=prompt_lens_cuda is not None,
        HAS_STATIC=static_input_ids is not None,
        HAS_RNG=has_rng,
        GLUE=glue,
        FROM_HOST=from_host,
        BY_ROW=rows is not None,
        BLOCK=block,
        **glue_constants,
        # Tiles of 32 rows or more run faster on eight warps.
        num_warps=8 if block >= 32 else 4,
    )


@dataclass
class SteadyMtpRecord:
    """Prepared state keyed by the recorded or most recently admitted batch.

    Request IDs and lengths include padding rows. Slots describe real requests only.
    """

    request_ids: list[int]
    slots: list[int]
    prompt_lengths: list[int]
    sequence_lengths: list[int]
    tokens_per_step: int  # T = runtime draft length + 1
    runtime_draft_len: int
    num_extra_kv_tokens: int
    use_full_generation_page_table: bool
    attn_metadata: AttentionMetadata
    spec_metadata: "SpecMetadata"
    # Whether attn_metadata.prepare_steady_decode_step replaces its prepare() (and the slab carries the
    # KV lengths).
    attn_steady: bool
    # The CUDA graph's static input ids / positions the staging kernel writes, or None.
    static_ids: torch.Tensor | None
    static_pos: torch.Tensor | None
    # The CUDA graph padding rows after the requests (one dummy request, repeated), as recorded.
    padding_request: "LlmRequest | None" = None
    num_padding: int = 0
    padding_num_tokens: int = 0
    padding_positions: np.ndarray | None = None
    padding_num_cached: list[int] = field(default_factory=list)
    padding_kv_lens: np.ndarray | None = None
    # The requests' slots as a slab segment, and the KV length each adds to its cached tokens.
    slots_array: np.ndarray | None = None
    seq_lens_kv_array: np.ndarray | None = None
    # The preparer's _SteadyStepInputs of the recorded batch.
    step_inputs: "_SteadyStepInputs | None" = None
    # The Engram refresh its steady steps do in the staging launch (None: unknown yet, False: none).
    glue_plan: "DecodeRefreshPlan | bool | None" = None


class SteadyMtpPrepare:
    """Record, host slab ring and staging-kernel launch of the steady one-model MTP decode prepare.

    Owned by ``DecoderInputPreparer``, which decides when a full pass is recorded.

    Args:
        max_batch_size: The engine's maximum number of requests per step.
        max_num_tokens: The engine's maximum number of tokens per step.
        max_tokens_per_step: Tokens per request at the maximum draft length.
        use_mrope: Whether the model reads the [3, 1, N] mrope positions.
        has_static_inputs: Whether decode steps replay CUDA graphs (to compile the kernel variant
            that also writes their static inputs).
        stage_rng: Whether the staging kernel also writes the step's Philox state.
        graph_runner: The CUDA graph runner whose Engram hook the staging launch may take over
            (with ``steady_decode_prepare``).
        steady_decode_prepare: The model's ``steady_decode_prepare`` opt-in: the loads-first staging
            kernel, which can also read the previous step's output rows.
    """

    # With stage_rng: Philox offsets between a request's consecutive rows, and the pinned ring of the step's Philox
    # state (fenced by the slab events); 0 / empty otherwise.
    rng_span: int = 0
    host_rng_slabs: tuple[torch.Tensor, ...] = ()
    host_rng_views: tuple[np.ndarray, ...] = ()

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_tokens_per_step: int,
        use_mrope: bool,
        has_static_inputs: bool,
        stage_rng: bool = False,
        graph_runner: "CUDAGraphRunner | None" = None,
        steady_decode_prepare: bool = False,
    ) -> None:
        self.record: SteadyMtpRecord | None = None
        self.use_mrope = use_mrope
        self.steady_decode_prepare = steady_decode_prepare
        max_requests = max(1, int(max_batch_size))
        # One staging-kernel BLOCK for every batch size of this engine, and its variant compiled now.
        self.block: int | None = None
        self.glue_runner: "CUDAGraphRunner | None" = None
        self.glue_provider: "EngramHashProvider | None" = None
        if max_tokens_per_step >= 2:
            self.block = triton.next_power_of_2(max_requests * max_tokens_per_step)
            if (
                steady_decode_prepare
                and graph_runner is not None
                and has_static_inputs
                and not use_mrope
            ):
                hook = graph_runner.fusible_engram_hook
                if hook is not None:
                    from ..modules.engram.engram import DECODE_REFRESH_MAX_ROWS

                    # The block covers the engine's configured maximum batch and draft width.
                    if self.block <= DECODE_REFRESH_MAX_ROWS:
                        self.glue_runner, self.glue_provider = graph_runner, hook.provider
                    else:
                        logger.info(
                            "Engram refresh uses the replay hook: the engine's staging block %d "
                            "exceeds the fused refresh limit %d",
                            self.block,
                            DECODE_REFRESH_MAX_ROWS,
                        )
        # The joining requests' seed cells carry four words each.
        seed_words = 0
        if self.glue_provider is not None:
            seed_words = 4 * max_requests * (self.glue_provider.config.max_ngram_size - 1)
        self.slab_capacity = _staging_words(max_requests, int(max_num_tokens)) + seed_words
        # With pinned memory the kernel reads the host slab through its device mapping (no H2D copy); otherwise
        # the slab goes through a device copy.
        self.zero_copy = prefer_pinned()
        self.host_slabs = [
            torch.zeros(self.slab_capacity, dtype=torch.int32, pin_memory=self.zero_copy)
            for _ in range(_RING)
        ]
        self.host_views = [slab.numpy() for slab in self.host_slabs]
        self.slab_events: list[torch.cuda.Event | None] = [None] * _RING
        self.device_slab = (
            None
            if self.zero_copy
            else torch.zeros(self.slab_capacity, dtype=torch.int32, device="cuda")
        )
        self.ring_pos = 0
        # With stage_rng the step's Philox state rides in a second pinned ring (int64, fenced by the same
        # events). Zero-copy only: the device-copy fallback has no mirror for this ring.
        if stage_rng and self.zero_copy and max_tokens_per_step >= 2:
            from ..speculative.interface import _RNG_SLOT_SPAN

            self.rng_span = int(_RNG_SLOT_SPAN)
            self.host_rng_slabs = tuple(
                torch.zeros(2 * max_requests, dtype=torch.int64, pin_memory=self.zero_copy)
                for _ in range(_RING)
            )
            self.host_rng_views = tuple(slab.numpy() for slab in self.host_rng_slabs)
        if self.block is not None:
            self._warm_stage_kernel(max_tokens_per_step, has_static_inputs)

    def _warm_stage_kernel(self, t: int, has_static_inputs: bool) -> None:
        """Compile the staging kernel variants the steady steps launch (same constexprs,
        unspecialized ints and alignments) with B = 0 launches, which read and write nothing."""
        if torch.cuda.is_current_stream_capturing():
            return
        dummy = torch.zeros(1, dtype=torch.int32, device="cuda")
        rng_variants = [{}]
        if self.rng_span:
            dummy64 = torch.zeros(1, dtype=torch.int64, device="cuda")
            rng_variants.append(
                dict(
                    rng_slab=dummy64,
                    rng_buffers=(dummy64, dummy64, dummy64, dummy64),
                    rng_span=self.rng_span,
                )
            )
        previous = types.SimpleNamespace(
            new_tokens=dummy.view(1, 1, 1),
            next_draft_tokens=dummy.view(1, 1),
            new_tokens_lens=dummy,
        )
        buffers = dict(
            input_ids=dummy,
            draft_tokens=dummy,
            prev_batch_idx=dummy,
            prev_pos_idx=dummy,
            prev_pos_offsets=dummy,
            prev_kv_offsets=dummy,
            positions=dummy,
            gather_ids=dummy,
            kv_lens_cuda=dummy,
            block=self.block,
        )
        static = dict(
            mrope_positions=dummy.view(1, 1, 1) if self.use_mrope else None,
            static_input_ids=dummy if has_static_inputs else None,
            static_position_ids=(
                (dummy.view(1, 1, 1) if self.use_mrope else dummy.view(1, 1))
                if has_static_inputs
                else None
            ),
        )
        if not self.steady_decode_prepare:
            for rng_kwargs in rng_variants:
                stage_steady_inputs(dummy, 0, t, previous, **buffers, **static, **rng_kwargs)
            return
        engram = None
        if self.glue_provider is not None:
            config = self.glue_provider.config
            dummy64 = torch.zeros((1, 1, 1), dtype=torch.long, device="cuda")
            layers = len(config.layer_ids)
            engram = types.SimpleNamespace(
                rows=dummy64.view(1),
                lookup=dummy64.view(1),
                history=dummy.view(1, 1),
                multipliers=torch.zeros(
                    (layers, config.max_ngram_size), dtype=torch.long, device="cuda"
                ),
                moduli=torch.ones(
                    (layers, config.max_ngram_size - 1, config.n_head_per_ngram),
                    dtype=torch.long,
                    device="cuda",
                ),
                hashes=dummy64,
                pad_id=0,
                dead=0,
                n_head=config.n_head_per_ngram,
                max_ngram=config.max_ngram_size,
            )
        rows = StepOutputRows(
            next_new_tokens=dummy.view(1, 1),
            new_tokens_lens=dummy,
            next_draft_tokens=dummy.view(1, 1),
            slots=(),
        )
        for from_host, by_row in ((False, None), (False, rows), (True, None)):
            variant = dict(
                loads_first=True,
                from_host=from_host,
                prompt_lens_cuda=dummy if from_host else None,
                rows=by_row,
            )
            for rng_kwargs in rng_variants:
                if engram is not None:
                    stage_steady_inputs(
                        dummy,
                        0,
                        t,
                        previous,
                        **buffers,
                        **variant,
                        mrope_positions=None,
                        static_input_ids=dummy,
                        static_position_ids=dummy.view(1, 1),
                        hook_positions=dummy,
                        engram=engram,
                        **rng_kwargs,
                    )
                stage_steady_inputs(
                    dummy, 0, t, previous, **buffers, **static, **variant, **rng_kwargs
                )

    def rng_rows(self) -> int | None:
        """Logits rows per request when this stager writes the step's Philox state (``stage_rng``),
        else None. The caller hands it to ``SpecMetadata`` before populating the sampling parameters."""
        if not self.rng_span or self.record is None:
            return None
        return self.record.tokens_per_step

    def invalidate(self) -> None:
        """Drop the record (every full pass does, before it may record again)."""
        self.record = None

    def record_batch(
        self,
        *,
        requests: list["LlmRequest"],
        prompt_lengths: list[int],
        sequence_lengths: list[int],
        tokens_per_step: int,
        runtime_draft_len: int,
        num_extra_kv_tokens: int,
        use_full_generation_page_table: bool,
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata",
        static_tensors: dict[str, torch.Tensor] | None,
        num_padding: int = 0,
        step_inputs: "_SteadyStepInputs | None" = None,
    ) -> None:
        """Record a full pass that prepared ``requests``, all verifying their previous device drafts
        (or all without a previous step on the device) but the last ``num_padding`` padding rows.

        The caller has checked the batch-level conditions (see ``DecoderInputPreparer._record_steady_mtp``).
        """
        b = len(requests)
        num_real = b - num_padding
        n_tokens = b * tokens_per_step
        if _staging_words(b, n_tokens) > self.slab_capacity:
            return
        attn_steady = (
            isinstance(attn_metadata, TrtllmAttentionMetadata)
            and attn_metadata.supports_steady_decode_step
            and attn_metadata.num_seqs == b
        )
        static_ids = static_pos = None
        static = static_tensors or {}
        if (
            "input_ids" in static
            and "position_ids" in static
            and static["input_ids"].shape[0] >= n_tokens
            and static["position_ids"].shape[-1] >= n_tokens
            and static["position_ids"].dim() == (3 if self.use_mrope else 2)
            and static["input_ids"].dtype == torch.int32
            and static["position_ids"].dtype == torch.int32
        ):
            static_ids, static_pos = static["input_ids"], static["position_ids"]
        if attn_steady:
            # The first steady step skips the block-offset copy this pass made when nothing changed.
            attn_metadata.key_prepared_block_offsets()
        seq_lens_kv = np.asarray(attn_metadata.seq_lens_kv[:b].tolist(), dtype=np.int64)
        padding = requests[num_real] if num_padding else None
        padding_num_tokens = 0
        padding_positions = padding_kv_lens = None
        padding_num_cached = []
        if padding is not None:
            # A padding row has no previous step: positions from its last token, cache before it.
            padding_num_tokens = padding.max_beam_num_tokens
            past_seen = padding_num_tokens - 1
            padding_positions = np.tile(
                np.arange(past_seen, past_seen + tokens_per_step, dtype=np.int32), num_padding
            )
            cached = past_seen - padding.py_num_compressed_tokens
            padding_num_cached = [cached] * num_padding
            padding_kv_lens = cached + seq_lens_kv[num_real:]
        slots = [request.py_seq_slot for request in requests[:num_real]]
        self.record = SteadyMtpRecord(
            request_ids=[request.py_request_id for request in requests],
            slots=slots,
            prompt_lengths=list(prompt_lengths),
            sequence_lengths=list(sequence_lengths),
            tokens_per_step=tokens_per_step,
            runtime_draft_len=runtime_draft_len,
            num_extra_kv_tokens=num_extra_kv_tokens,
            use_full_generation_page_table=use_full_generation_page_table,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            attn_steady=attn_steady,
            static_ids=static_ids,
            static_pos=static_pos,
            padding_request=padding,
            num_padding=num_padding,
            padding_num_tokens=padding_num_tokens,
            padding_positions=padding_positions,
            padding_num_cached=padding_num_cached,
            padding_kv_lens=padding_kv_lens,
            slots_array=np.asarray(slots, dtype=np.int32),
            seq_lens_kv_array=seq_lens_kv[:num_real],
            step_inputs=step_inputs,
        )

    def matches(
        self,
        requests: list["LlmRequest"],
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata | None",
        new_tensors_device: Any,
        runtime_draft_len: int,
    ) -> bool:
        """Whether the generation-only batch ``requests`` is the recorded one, one step later."""
        rec = self.record
        if (
            rec is None
            or attn_metadata is not rec.attn_metadata
            or spec_metadata is not rec.spec_metadata
            or not attn_metadata.is_cuda_graph
            or runtime_draft_len != rec.runtime_draft_len
            or not isinstance(new_tensors_device, SampleStateTensorsSpec)
            or new_tensors_device.next_draft_tokens is None
            or new_tensors_device.new_tokens_lens is None
            or len(requests) != len(rec.request_ids)
            or not self._padding_matches(requests)
        ):
            return False
        # zip stops after the requests: rec.slots has no padding rows.
        return all(
            request.py_request_id == rid
            and request.py_seq_slot == slot
            and request.py_batch_idx == slot
            and self.request_qualifies(request)
            for request, rid, slot in zip(requests, rec.request_ids, rec.slots)
        )

    def admits(
        self,
        requests: list["LlmRequest"],
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata | None",
        runtime_draft_len: int,
    ) -> bool:
        """Whether the generation-only batch ``requests`` joins on the recorded metadata and padding rows:
        each request's first step on this worker (no batch index yet) with every draft slot filled."""
        rec = self.record
        if (
            rec is None
            or attn_metadata is not rec.attn_metadata
            or spec_metadata is not rec.spec_metadata
            or not attn_metadata.is_cuda_graph
            or runtime_draft_len != rec.runtime_draft_len
            or len(requests) != len(rec.request_ids)
            or not self._padding_matches(requests)
        ):
            return False
        num_drafts = rec.tokens_per_step - 1
        return all(
            request.py_batch_idx is None
            and request.py_seq_slot is not None
            and get_draft_token_length(request) == num_drafts
            and self.request_qualifies(request)
            for request in requests[: len(rec.slots)]
        )

    def admit(self, requests: list["LlmRequest"]) -> None:
        """Make the record describe ``requests``, which join on it (``admits``) this step."""
        rec = self.record
        rec.request_ids = [request.py_request_id for request in requests]
        rec.slots = [request.py_seq_slot for request in requests[: len(rec.slots)]]
        rec.slots_array = np.asarray(rec.slots, dtype=np.int32)
        rec.prompt_lengths = [request.py_prompt_len for request in requests]
        rec.glue_plan = None

    def _padding_matches(self, requests: list["LlmRequest"]) -> bool:
        """Whether the rows after the requests are the recorded CUDA graph padding rows."""
        rec = self.record
        if not rec.num_padding:
            return True
        padding = rec.padding_request
        return padding.max_beam_num_tokens == rec.padding_num_tokens and all(
            request is padding for request in requests[len(rec.slots) :]
        )

    @staticmethod
    def request_qualifies(request: "LlmRequest") -> bool:
        """Whether a request's step can be staged: a real, text-only request that needs no one-hot draft
        probabilities this step."""
        return (
            not request.is_dummy
            and request.py_multimodal_data is None
            and not request.py_needs_onehot_draft_probs
        )

    def _next_slab(self) -> int:
        k = self.ring_pos
        self.ring_pos = (k + 1) % _RING
        event = self.slab_events[k]
        if event is not None and not event.query():
            # The kernel launched from this slot _RING steps ago has not run yet; with the
            # overlap loop's per-step sync this does not happen, but never overwrite it early.
            event.synchronize()
        return k

    def _glue_plan(self, rec: SteadyMtpRecord, num_tokens: int) -> "DecodeRefreshPlan | None":
        """The Engram refresh the staging launch does in place of the replay's hook, or None."""
        provider = self.glue_provider
        plan = rec.glue_plan
        if plan is False or provider is None or not self._glue_applies(rec, num_tokens):
            return None
        if plan is not None and provider.decode_refresh_plan_current(plan):
            return plan
        plan = provider.decode_refresh_plan(
            rec.request_ids,
            rec.attn_metadata.seq_lens,
            num_tokens,
            getattr(rec.attn_metadata.kv_cache_manager, "max_seq_len", None),
        )
        # Looked up once per record: a batch the refresh did not record takes the hook every step.
        rec.glue_plan = plan if plan is not None else False
        return plan

    def _glue_applies(self, rec: SteadyMtpRecord, num_tokens: int) -> bool:
        """Whether the replay's only hook is the provider's refresh of the recorded batch."""
        assert self.glue_runner is not None
        hook = self.glue_runner.fusible_engram_hook
        return (
            rec.num_padding == 0
            and rec.static_ids is not None
            and hook is not None
            and hook.provider is self.glue_provider
            and rec.attn_metadata.num_tokens == num_tokens
        )

    def _admission_plan(
        self, rec: SteadyMtpRecord, num_tokens: int, free_words: int
    ) -> "tuple[DecodeRefreshPlan, list[int], tuple[list[int], list[int], list[int], list[bool]]] | None":
        """The Engram refresh of joining requests (with their history rows and seed cells) that the
        staging launch does in place of the replay's hook, or None."""
        if self.glue_provider is None or not self._glue_applies(rec, num_tokens):
            return None
        return self.glue_provider.admission_refresh_plan(
            rec.request_ids,
            rec.attn_metadata.seq_lens,
            num_tokens,
            getattr(rec.attn_metadata.kv_cache_manager, "max_seq_len", None),
            min(self.block, (free_words - len(rec.slots)) // 4),
        )

    def stage(
        self,
        requests: list["LlmRequest"],
        new_tensors_device: Any,
        buffers: Any,
        attn_metadata: AttentionMetadata,
        rng_state: tuple[list[int], list[int]] | None = None,
        rng_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        from_host: bool = False,
    ) -> tuple[list[int], list[int], int, np.ndarray, bool]:
        """Write this step's device inputs (one slab write, one kernel launch).

        Args:
            requests: The matched generation requests, padding rows included.
            new_tensors_device: The previous step's device sampler tensors.
            buffers: The preparer's DecoderInputBuffers.
            attn_metadata: The recorded attention metadata.
            rng_state: What ``SpecMetadata.populate_sampling_params_for_one_model`` returns (each request's Philox
                seed and offset-window base, in batch order), or None. Written only when this stager has the Philox
                ring.
            rng_buffers: ``SpecMetadata``'s (request_seeds, request_offsets, seeds, offsets).
            from_host: The requests joined this step (``admit``): their last token and drafts come
                from the host, and nothing past their last token is cached.

        Returns:
            The per-row cached-token counts (``KVCacheParams.num_cached_tokens_per_seq``) and KV
            lengths (without the extra spec tokens), the number of tokens staged in the CUDA graph's
            static inputs (0: none), the host positions, and whether the launch did the hooks' work.
        """
        rec = self.record
        b = len(rec.slots)
        num_rows = len(requests)
        t = rec.tokens_per_step
        n = num_rows * t
        k = self._next_slab()
        view = self.host_views[k]
        past_seen = np.fromiter(
            (request.max_beam_num_tokens for request in requests[:b]), dtype=np.int64, count=b
        )
        past_seen -= 1
        compressed = np.fromiter(
            (request.py_num_compressed_tokens for request in requests[:b]), dtype=np.int64, count=b
        )
        view[:b] = rec.slots_array
        positions = view[b : b + n]
        np.add(
            past_seen[:, None],
            np.arange(t, dtype=np.int64),
            out=positions[: b * t].reshape(b, t),
            casting="unsafe",
        )
        # A verifying request's KV length assumes all its drafts accepted; the kernel corrects it.
        num_cached_real = past_seen - compressed if from_host else past_seen + (t - compressed)
        kv_lens_real = num_cached_real + rec.seq_lens_kv_array
        if rec.num_padding:
            positions[b * t :] = rec.padding_positions
        used = b + n
        if rec.attn_steady:
            view[used : used + b] = kv_lens_real
            if rec.num_padding:
                view[used + b : used + num_rows] = rec.padding_kv_lens
            used += num_rows
        prompt_staged = from_host and rec.attn_steady
        plan, num_seeds = None, 0
        if from_host:
            tokens = view[used : used + b * t].reshape(b, t)
            tokens[:, 0] = [request.get_last_tokens(0) for request in requests[:b]]
            tokens[:, 1:] = [request.py_draft_tokens for request in requests[:b]]
            used += b * t
            if prompt_staged:
                view[used : used + num_rows] = rec.prompt_lengths
                used += num_rows
            admission = self._admission_plan(rec, n, self.slab_capacity - used)
            if admission is not None:
                plan, rows, cells = admission
                num_seeds = len(cells[0])
                view[used : used + b] = rows
                used += b
                for values in cells:
                    view[used : used + num_seeds] = values
                    used += num_seeds
        else:
            plan = self._glue_plan(rec, n)
        assert used <= self.slab_capacity
        host_positions = positions.copy()
        if self.zero_copy:
            slab = self.host_slabs[k]
        else:
            self.device_slab[:used].copy_(self.host_slabs[k][:used], non_blocking=True)
            slab = self.device_slab
        rng_kwargs = {}
        if rng_state is not None and self.rng_span:
            seeds, offsets = rng_state
            assert len(seeds) == num_rows and len(offsets) == num_rows and rng_buffers is not None
            rng_view = self.host_rng_views[k]
            rng_view[:num_rows] = seeds
            rng_view[num_rows : 2 * num_rows] = offsets
            rng_kwargs = dict(
                rng_slab=self.host_rng_slabs[k], rng_buffers=rng_buffers, rng_span=self.rng_span
            )

        stores_ready = getattr(new_tensors_device, "stores_ready", None)
        rows = None
        if self.steady_decode_prepare and not from_host:
            rows = self._previous_rows(new_tensors_device, rec, t)
        if stores_ready is not None and rows is None and not from_host:
            # The kernel reads the stores, which the sampler's side stream updates.
            current_cuda_stream().wait_event(stores_ready)
        stage_steady_inputs(
            slab,
            b,
            t,
            new_tensors_device,
            input_ids=buffers.input_ids_cuda,
            draft_tokens=buffers.draft_tokens_cuda,
            prev_batch_idx=buffers.previous_batch_indices_cuda,
            prev_pos_idx=buffers.previous_pos_indices_cuda,
            prev_pos_offsets=buffers.previous_pos_id_offsets_cuda,
            prev_kv_offsets=buffers.previous_kv_lens_offsets_cuda,
            positions=buffers.position_ids_cuda,
            mrope_positions=buffers.mrope_position_ids_cuda if self.use_mrope else None,
            gather_ids=buffers.gather_ids_cuda,
            kv_lens_cuda=attn_metadata.kv_lens_cuda if rec.attn_steady else None,
            static_input_ids=rec.static_ids,
            static_position_ids=rec.static_pos,
            block=self.block,
            **rng_kwargs,
            loads_first=self.steady_decode_prepare,
            num_rows=num_rows,
            hook_positions=None if plan is None else self.glue_runner.pre_replay_hook_positions(n),
            engram=plan,
            from_host=from_host,
            prompt_lens_cuda=attn_metadata.prompt_lens_cuda if prompt_staged else None,
            num_seeds=num_seeds,
            rows=rows,
        )
        if plan is not None:
            self.glue_provider.adopt_decode_refresh(plan)
            rec.glue_plan = plan
        # The host slots (and the Philox slot) may be rewritten once their reader (the copy, or the kernel
        # itself) has run.
        event = self.slab_events[k]
        if event is None:
            event = self.slab_events[k] = torch.cuda.Event()
        event.record(current_cuda_stream())
        if stores_ready is not None and (rows is not None or from_host):
            # The replay overwrites the outputs the side stream's store update reads.
            current_cuda_stream().wait_event(stores_ready)
        num_cached = num_cached_real.tolist()
        kv_lens = kv_lens_real.tolist()
        if rec.num_padding:
            num_cached += rec.padding_num_cached
            kv_lens += rec.padding_kv_lens.tolist()
        staged = n if rec.static_ids is not None else 0
        return num_cached, kv_lens, staged, host_positions, plan is not None

    @staticmethod
    def _previous_rows(
        new_tensors_device: Any, rec: SteadyMtpRecord, t: int
    ) -> "StepOutputRows | None":
        """The previous step's output rows when they are this batch's requests in order and wide
        enough for the kernel (the stores hold the same values then), else None."""
        rows = getattr(new_tensors_device, "rows", None)
        b = len(rec.slots)
        if (
            rows is None
            or rows.slots[:b] != tuple(rec.slots)
            or rows.next_new_tokens.shape[1] < t
            or rows.next_draft_tokens.shape[1] < t - 1
        ):
            return None
        return rows

    def prepare_attention(
        self, attn_metadata: AttentionMetadata, kv_lens: list[int], admitted: bool = False
    ) -> None:
        """The attention metadata's prepare() of a staged step (``admitted``: one that ``admit``
        re-keyed the record for, whose prompt lengths changed)."""
        if self.record.attn_steady:
            attn_metadata.prepare_steady_decode_step(
                kv_lens, self.record.prompt_lengths if admitted else None
            )
        else:
            attn_metadata.prepare()
