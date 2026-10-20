# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady one-model MTP decode prepare (``TRTLLM_Q35_STEADY_PREPARE=1``, default off).

Under the overlap scheduler a one-model speculative decode step whose batch is the previous
step's (same requests in the same order, every request verifying the previous step's device
drafts from an unchanged sequence slot, same CUDA graph metadata, no padding) needs only a few
new integers per request from ``PyTorchModelEngine._prepare_tp_inputs``: the base position, the
host's accepted-draft count and the cached KV length. Everything else the full pass builds --
the request / prompt / sequence-length lists, the gather ids, the previous-slot indices -- is
what the last full pass left. With the flag the model engine records the full pass of such a
step and prepares the following ones from it:

- the per-request integers (previous slots, accepted counts, the ``b * (draft + 1)`` positions)
  go to the device in one pinned-slab upload (3-slot ring, each reuse fenced by an event);
- one Triton kernel writes everything the full pass writes on the device from the slab and the
  previous step's device sampler tensors: input ids, draft tokens, accepted counts, previous
  batch / position indices, the overlap position and KV-length corrections, the scalar and the
  (3, 1, N) mrope positions and the gather ids (the full pass needs ~12 H2D copies and small
  kernels for these);
- the host-side metadata (KV cache params, request ids, prompt lengths, spec metadata fields,
  iteration stats) is set from the record; ``attn_metadata.prepare()`` and the spec metadata
  prepare / sampling-parameter population run as in the full pass.

Any other step (batch or slot change, a request without a previous device tensor, context rows,
dummies / padding, mrope multimodal spans, LoRA, guided decoding, beams, attention DP, CP,
warmup) takes the full path, which records again when it qualifies.

``TRTLLM_Q35_STEADY_PREPARE_VERIFY=N`` checks every N-th fast step against the full pass: the
full pass runs first, the device buffers and host fields it produced are snapshotted, the fast
path runs on the same state, and both must agree bit for bit; a mismatch is logged and turns the
fast path off (``TRTLLM_Q35_STEADY_PREPARE_VERIFY_STRICT=1`` raises instead).
``TRTLLM_Q35_STEADY_PREPARE_STATS=N`` logs the fast / full step counts every N steps.
"""

import os
import types
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from ..metadata import KVCacheParams
from ..utils import set_per_request_piecewise_cuda_graph_flag

STEADY_PREPARE_ENV = "TRTLLM_Q35_STEADY_PREPARE"
VERIFY_ENV = "TRTLLM_Q35_STEADY_PREPARE_VERIFY"
VERIFY_STRICT_ENV = "TRTLLM_Q35_STEADY_PREPARE_VERIFY_STRICT"
STATS_ENV = "TRTLLM_Q35_STEADY_PREPARE_STATS"
# With the steady prepare on, also replace TrtllmAttentionMetadata.prepare() of a steady step by
# _steady_attn_prepare (kv_lens_cuda from the staging kernel, no prompt-length / request-type uploads).
ATTN_ENV = "TRTLLM_Q35_STEADY_PREPARE_ATTN"
# With the steady prepare on, the staging kernel also writes the CUDA graph's static input ids and
# positions, and the graph replay that follows skips its two staging copies.
STATIC_ENV = "TRTLLM_Q35_STEADY_PREPARE_STATIC"
# With the steady attention prepare on, skip the block-offset table copies while no request of the batch
# changed its block count or the pages of its last blocks since the last copy.
SKIP_BLOCKS_ENV = "TRTLLM_Q35_STEADY_PREPARE_SKIP_BLOCKS"
# With the steady prepare on, the staging kernel reads the pinned host slab through its device mapping instead of
# a device copy of it: one H2D copy fewer between the decode graph replays (the slab ring's event now fences the
# kernel, the slab's reader).
ZEROCOPY_ENV = "TRTLLM_Q35_STEADY_PREPARE_ZEROCOPY"
_RING = 3


def steady_prepare_enabled() -> bool:
    return os.environ.get(STEADY_PREPARE_ENV, "0") == "1"


# Integer arguments (B, strides) are not specialized, pointers not on alignment, and SteadyMtpPrepare launches
# every batch size with one BLOCK: one compiled variant per engine, compiled during the engine warmup
# (SteadyMtpPrepare.__init__) instead of a JIT stall (up to ~0.3 s) at the first steady step of each new batch size.
_STAGE_INT_ARGS = ["nnt_tok_stride", "nnt_slot_stride", "nd_slot_stride", "mrope_axis_stride",
                   "static_pos_axis_stride", "B"]
_STAGE_PTR_ARGS = ["slab", "next_new_tokens", "next_drafts", "new_lens", "input_ids", "draft_tokens",
                   "num_accepted", "prev_batch_idx", "prev_pos_idx", "prev_pos_offsets", "prev_kv_offsets",
                   "positions", "mrope_positions", "gather_ids", "kv_lens_out", "static_ids", "static_pos"]


def _stage_jit(**kwargs):
    try:
        return triton.jit(**kwargs)
    except TypeError:  # a Triton without do_not_specialize_on_alignment
        kwargs.pop("do_not_specialize_on_alignment", None)
        return triton.jit(**kwargs)


@_stage_jit(do_not_specialize=_STAGE_INT_ARGS, do_not_specialize_on_alignment=_STAGE_PTR_ARGS)
def _steady_stage_kernel(
    slab,  # int32 [slots (B) | accepted (B) | positions (B * T)]
    next_new_tokens,  # [>= T, num_slots, 1] previous step's next-step input tokens
    nnt_tok_stride,
    nnt_slot_stride,
    next_drafts,  # [num_slots, >= D]
    nd_slot_stride,
    new_lens,  # [num_slots] previous step's accepted-token counts
    input_ids,
    draft_tokens,
    num_accepted,
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
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    HAS_KV: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    # token rows: request r = i // T, position j = i % T
    tok = i < B * T
    r = i // T
    j = i % T
    slot = tl.load(slab + r, mask=tok, other=0)
    pos = tl.load(slab + 2 * B + i, mask=tok, other=0)
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
    tl.store(num_accepted + i, tl.load(slab + B + i, mask=req, other=0), mask=req)
    tl.store(prev_batch_idx + i, slot_r, mask=req)
    lens_r = tl.load(new_lens + slot_r, mask=req, other=0)
    tl.store(prev_kv_offsets + i, lens_r - T, mask=req)
    if HAS_KV:
        tl.store(kv_lens_out + i, tl.load(slab + 2 * B + B * T + i, mask=req, other=0), mask=req)
    # draft rows: request i // D, draft i % D
    drf = i < B * D
    slot_d = tl.load(slab + i // D, mask=drf, other=0)
    dval = tl.load(next_drafts + slot_d * nd_slot_stride + i % D, mask=drf, other=0)
    tl.store(draft_tokens + i, dval, mask=drf)


def stage_steady_inputs(device_slab: torch.Tensor, b: int, t: int, new_tensors_device, *,
                        input_ids: torch.Tensor, draft_tokens: torch.Tensor,
                        num_accepted: torch.Tensor, prev_batch_idx: torch.Tensor,
                        prev_pos_idx: torch.Tensor, prev_pos_offsets: torch.Tensor,
                        prev_kv_offsets: torch.Tensor, positions: torch.Tensor,
                        mrope_positions: Optional[torch.Tensor], gather_ids: torch.Tensor,
                        kv_lens_cuda: Optional[torch.Tensor] = None,
                        static_input_ids: Optional[torch.Tensor] = None,
                        static_position_ids: Optional[torch.Tensor] = None,
                        block: Optional[int] = None) -> None:
    """One launch of the steady staging kernel (slab layout: slots, accepted, positions[, kv lens]).

    ``block``: the BLOCK of the launch when it covers the batch (one variant for all batch sizes), else the
    batch's own power of two.
    """
    d = t - 1
    n = b * t
    need = max(n, b * d, b, 1)
    if block is None or block < need:
        block = triton.next_power_of_2(need)
    new_tokens = new_tensors_device.new_tokens
    next_drafts = new_tensors_device.next_draft_tokens
    has_mrope = mrope_positions is not None
    _steady_stage_kernel[(1, )](
        device_slab,
        new_tokens,
        new_tokens.stride(0),
        new_tokens.stride(1),
        next_drafts,
        next_drafts.stride(0),
        new_tensors_device.new_tokens_lens,
        input_ids,
        draft_tokens,
        num_accepted,
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
        b,
        T=t,
        D=d,
        HAS_MROPE=has_mrope,
        HAS_KV=kv_lens_cuda is not None,
        HAS_STATIC=static_input_ids is not None,
        BLOCK=block,
    )


# Trailing blocks whose page indices key the block-offset skip: a decode step appends at most one block and the
# rewind of its rejected drafts frees at most one (both < tokens_per_block tokens), so only the last block can be
# re-allocated (on another page) at an unchanged block count; two cover any step of fewer than 2 blocks' tokens.
_TAIL_BLOCKS = 2


def _block_counts(kv_cache_manager, request_ids: List[int]) -> Optional[Tuple[Any, ...]]:
    """Key of the batch's block-offset table rows (KV cache manager v2), or None when unavailable.

    Per request: the block count and the page indices of its last ``_TAIL_BLOCKS`` blocks in every pool, read from
    the manager's host table (which the KV caches write and copy_batch_block_offsets copies from). The count alone
    is not enough: update_resources rewinds the rejected drafts after every step, which can free the last block, and
    the next step re-allocates it, possibly on another page (a stale table the VERIFY mode detects).
    """
    kv_map = getattr(kv_cache_manager, "kv_cache_map", None)
    table = getattr(kv_cache_manager, "host_kv_cache_block_offsets", None)
    mapper = getattr(kv_cache_manager, "index_mapper", None)
    if (kv_map is None or table is None or mapper is None
            or getattr(kv_cache_manager, "enable_swa_scratch_reuse", False)
            or getattr(kv_cache_manager, "enable_block_reuse", False)):
        return None
    host = getattr(kv_cache_manager, "_q35_host_table_np", None)
    if host is None or host.shape != tuple(table.shape):
        host = kv_cache_manager._q35_host_table_np = table.numpy()
    beam = int(getattr(kv_cache_manager, "max_beam_width", 1))
    keys = []
    for rid in request_ids:
        cache = kv_map.get(rid)
        if cache is None:
            return None
        n = cache.num_blocks
        row = mapper.get_index(rid) * beam
        keys.append((n, host[:, row, 0, max(0, n - _TAIL_BLOCKS):n].tobytes()))
    return tuple(keys)


def _steady_attn_prepare(attn_metadata, kv_lens: List[int], num_extra_kv_tokens: int,
                         state: "SteadyMtpPrepare") -> None:
    """TrtllmAttentionMetadata.prepare() of a steady decode step.

    The request ids, prompt lengths and request types are those the recording full pass (or the
    previous steady step) uploaded, and nothing else writes these buffers in between (any other
    batch is a full pass, which drops the record); kv_lens_cuda[:b] was written by the staging
    kernel. What is left is the host state prepare() leaves: the mamba metadata, the global
    metadata reference, the host kv lengths (with the extra spec tokens), the host totals, the
    block offsets and the runtime views.
    """
    from ..attention_backend.interface import AttentionMetadata
    from ..utils import get_global_attrs, get_model_extra_attrs

    AttentionMetadata.prepare(attn_metadata)  # TrtllmAttentionMetadata.prepare's super().prepare()
    if get_model_extra_attrs() is None:
        get_global_attrs().attention_metadata = weakref.ref(attn_metadata)
    n = len(kv_lens)
    host = state.kv_lens_host
    host_np = state.kv_lens_host_np
    host_np[:n] = kv_lens
    attn_metadata.kv_lens[:n].copy_(host[:n] + num_extra_kv_tokens)
    attn_metadata.host_total_kv_lens[0] = 0
    attn_metadata.host_total_kv_lens[1] = sum(kv_lens)
    max_kv_len = max(kv_lens) + num_extra_kv_tokens
    kv_cache_manager = attn_metadata.kv_cache_manager
    assert max_kv_len <= kv_cache_manager.max_seq_len, (
        f"The max KV cache length of input sequences ({max_kv_len}) exceeds the KV cache "
        f"manager's maximum supported length ({kv_cache_manager.max_seq_len}).")
    # The block-offset table rows of a steady batch change only at their tails (a block appended, or the last
    # block freed by the draft rewind and re-allocated; no block reuse / SWA on this path): skip the copies while
    # every request's block count and tail pages are those the last copy saw (the recording full pass or an
    # earlier steady step).
    rec = state.record
    counts = _block_counts(kv_cache_manager, attn_metadata.request_ids)
    state.skipped_block_copy = not (counts is None or counts != rec.block_counts)
    if not state.skipped_block_copy:
        kv_cache_manager.copy_batch_block_offsets(attn_metadata.kv_cache_block_offsets,
                                                  attn_metadata.request_ids,
                                                  attn_metadata.beam_width,
                                                  attn_metadata.num_contexts, n, max_blocks=None)
        rec.block_counts = counts if state.skip_block_copies else None
    draft = attn_metadata.draft_kv_cache_manager
    if draft is not None:
        draft_counts = _block_counts(draft, attn_metadata.request_ids)
        if draft_counts is None or draft_counts != rec.draft_block_counts:
            draft.copy_batch_block_offsets(attn_metadata.draft_kv_cache_block_offsets,
                                           attn_metadata.request_ids, attn_metadata.beam_width,
                                           attn_metadata.num_contexts, n, max_blocks=None)
            rec.draft_block_counts = draft_counts if state.skip_block_copies else None
    attn_metadata._bind_runtime_views(
        kv_lens_cuda=attn_metadata.kv_lens_cuda[:n],
        kv_lens=host[:n].clone(),
        prompt_lens_cuda=attn_metadata.prompt_lens_cuda[:n],
        prompt_lens_cpu=attn_metadata.prompt_lens_cpu[:n],
        host_request_types=attn_metadata.host_request_types[:n],
    )


@dataclass
class _Record:
    """What the last qualifying full pass left, keyed by the batch it ran."""
    request_ids: List[int]
    slots: List[int]
    prompt_lengths: List[int]
    sequence_lengths: List[int]
    draft_lens: List[int]
    tokens_per_step: int  # T = runtime_draft_len + 1
    runtime_draft_len: int
    num_extra_kv_tokens: int
    attn_metadata: Any
    spec_metadata: Any
    resource_manager: Any
    kv_cache_manager: Any
    seq_lens_kv: List[int]
    attn_steady: bool
    static_ids: Optional[torch.Tensor]
    static_pos: Optional[torch.Tensor]
    block_counts: Optional[Tuple[Any, ...]]  # _block_counts keys of the last block-offset copy
    draft_block_counts: Optional[Tuple[Any, ...]]
    has_mamba_chunk_size: bool
    mamba_chunk_size: Optional[int]
    use_mrope: bool


class SteadyMtpPrepare:
    """Per-engine state of the steady decode prepare (see the module docstring)."""

    def __init__(self, engine) -> None:
        self.record: Optional[_Record] = None
        self.disabled = False
        self.verify_every = int(os.environ.get(VERIFY_ENV, "0") or 0)
        self.verify_strict = os.environ.get(VERIFY_STRICT_ENV, "0") == "1"
        self.stats_every = int(os.environ.get(STATS_ENV, "0") or 0)
        self.skip_block_copies = os.environ.get(SKIP_BLOCKS_ENV, "0") == "1"
        self.zero_copy = os.environ.get(ZEROCOPY_ENV, "0") == "1" and prefer_pinned()
        self.skipped_block_copy = False
        self.n_fast = 0
        self.n_full = 0
        self.n_verified = 0
        self.n_realloc_stale = 0  # verified steps where a block-count-only skip key would have left a stale table
        max_requests = max(1, int(engine.batch_size))
        max_tokens = int(engine.max_num_tokens)
        self.slab_capacity = 3 * max_requests + max_tokens
        self.host_slabs = [
            torch.zeros(self.slab_capacity, dtype=torch.int32, pin_memory=prefer_pinned())
            for _ in range(_RING)
        ]
        self.host_views = [slab.numpy() for slab in self.host_slabs]
        self.slab_events: List[Optional[torch.cuda.Event]] = [None] * _RING
        self.device_slab = torch.zeros(self.slab_capacity, dtype=torch.int32, device="cuda")
        self.kv_lens_host = torch.zeros(max_requests, dtype=torch.int32)
        self.kv_lens_host_np = self.kv_lens_host.numpy()
        self.ring_pos = 0
        # One staging-kernel BLOCK for every batch size of this engine, and its variant compiled now.
        self.block: Optional[int] = None
        tokens_per_step = int(engine.get_runtime_tokens_per_gen_step(engine.runtime_draft_len))
        if tokens_per_step >= 2:
            self.block = triton.next_power_of_2(max_requests * tokens_per_step)
            self._warm_stage_kernel(engine, tokens_per_step)

    def _warm_stage_kernel(self, engine, t: int) -> None:
        """Compile the staging kernel variant the fast path will launch (same constexprs, unspecialized ints and
        alignments) with a B = 0 launch, which reads and writes nothing."""
        if torch.cuda.is_current_stream_capturing():
            return
        dummy = self.device_slab[:1]
        use_mrope = bool(engine.use_mrope)
        stage_steady_inputs(
            self.device_slab, 0, t,
            types.SimpleNamespace(new_tokens=dummy.view(1, 1, 1), next_draft_tokens=dummy.view(1, 1),
                                  new_tokens_lens=dummy),
            input_ids=dummy, draft_tokens=dummy, num_accepted=dummy, prev_batch_idx=dummy,
            prev_pos_idx=dummy, prev_pos_offsets=dummy, prev_kv_offsets=dummy, positions=dummy,
            mrope_positions=dummy.view(1, 1, 1) if use_mrope else None, gather_ids=dummy,
            kv_lens_cuda=dummy if os.environ.get(ATTN_ENV, "0") == "1" else None,
            static_input_ids=dummy if os.environ.get(STATIC_ENV, "0") == "1" else None,
            static_position_ids=(dummy.view(1, 1, 1) if use_mrope else dummy.view(1, 1))
            if os.environ.get(STATIC_ENV, "0") == "1" else None,
            block=self.block)

    # ------------------------------------------------------------------ eligibility
    @staticmethod
    def engine_supported(engine) -> bool:
        spec_config = engine.spec_config
        return (spec_config is not None and engine.enable_spec_decode
                and not engine.is_draft_model
                and spec_config.spec_dec_mode.use_one_engine()
                and spec_config.is_linear_tree and engine.guided_decoder is None
                and not engine.enable_attention_dp and not engine.use_beam_search
                and engine.max_beam_width == 1
                and not engine._is_encoder_decoder_model()
                and not engine.mapping.has_cp_helix()
                and not engine._disable_overlap_scheduler
                and getattr(engine, "cuda_graph_lora_manager", None) is None)

    def _batch_matches(self, engine, scheduled_requests, attn_metadata, spec_metadata,
                       new_tensors_device, promoted_context_request_ids) -> bool:
        rec = self.record
        if (rec is None or self.disabled or engine.is_warmup or promoted_context_request_ids
                or scheduled_requests.num_context_requests != 0
                or attn_metadata is not rec.attn_metadata
                or spec_metadata is not rec.spec_metadata
                or not attn_metadata.is_cuda_graph
                or engine.runtime_draft_len != rec.runtime_draft_len
                or new_tensors_device is None
                or getattr(new_tensors_device, "next_draft_tokens", None) is None
                or getattr(new_tensors_device, "new_tokens_lens", None) is None):
            return False
        gen = scheduled_requests.generation_requests
        if len(gen) != len(rec.request_ids):
            return False
        for request, rid, slot in zip(gen, rec.request_ids, rec.slots):
            if (request.py_request_id != rid or request.py_seq_slot != slot
                    or request.py_batch_idx != slot or request.is_dummy
                    or request.py_multimodal_data is not None
                    or getattr(request, "py_needs_onehot_draft_probs", False)):
                return False
        return True

    # ------------------------------------------------------------------ record
    def maybe_record(self, engine, *, scheduled_requests, attn_metadata, spec_metadata,
                     resource_manager, kv_cache_manager, request_ids, prompt_lengths,
                     sequence_lengths, draft_lens, previous_batch_indices, num_tokens,
                     num_draft_tokens, extend_requests, extend_dummy_requests,
                     generation_requests, first_draft_requests, padded, mrope_lists_empty,
                     has_lora, promoted_context_request_ids) -> None:
        """Called at the end of a full _prepare_tp_inputs pass; keeps the record if it qualifies."""
        self.n_full += 1
        self.record = None
        if self.disabled or engine.is_warmup or not self.engine_supported(engine):
            return
        b = len(extend_requests)
        from .resource_manager import ResourceManagerType

        if (resource_manager is None or resource_manager.get_resource_manager(
                ResourceManagerType.PEFT_CACHE_MANAGER) is not None):
            return
        if (b == 0 or promoted_context_request_ids or scheduled_requests.num_context_requests
                or extend_dummy_requests or generation_requests or first_draft_requests
                or num_tokens or num_draft_tokens or padded or not mrope_lists_empty or has_lora
                or len(previous_batch_indices) != b or spec_metadata is None
                or not attn_metadata.is_cuda_graph
                or engine._get_position_id_offset() != 0):
            return
        tokens_per_step = engine.get_runtime_tokens_per_gen_step(engine.runtime_draft_len)
        if (sequence_lengths != [tokens_per_step] * b
                or draft_lens != [tokens_per_step - 1] * b or tokens_per_step < 2
                or 2 * b + b * tokens_per_step > self.slab_capacity):
            return
        gen = scheduled_requests.generation_requests
        slots = [request.py_seq_slot for request in gen]
        if list(previous_batch_indices) != slots or [r.py_request_id for r in gen] != request_ids:
            return
        if any(r.is_dummy or r.py_multimodal_data is not None for r in gen):
            return
        from ..speculative.eagle3 import Eagle3SpecMetadata
        from ..speculative.utils import get_num_extra_kv_tokens

        if isinstance(spec_metadata, Eagle3SpecMetadata):
            return

        from ..attention_backend.trtllm import TrtllmAttentionMetadata

        seq_lens_kv = attn_metadata.seq_lens_kv[:b].tolist()
        attn_steady = (os.environ.get(ATTN_ENV, "0") == "1"
                       and type(attn_metadata) is TrtllmAttentionMetadata
                       and attn_metadata.kv_cache_manager is not None
                       and not attn_metadata.enable_flash_mla
                       and not attn_metadata.enable_helix
                       and not attn_metadata.enable_context_mla_with_cached_kv
                       and attn_metadata.num_seqs == b
                       and 3 * b + b * tokens_per_step <= self.slab_capacity)
        static_ids = static_pos = None
        runner = getattr(engine, "cuda_graph_runner", None)
        static = getattr(runner, "shared_static_tensors", None) or {}
        n_tokens = b * tokens_per_step
        if (os.environ.get(STATIC_ENV, "0") == "1" and "input_ids" in static
                and "position_ids" in static and static["input_ids"].shape[0] >= n_tokens
                and static["position_ids"].shape[-1] >= n_tokens
                and static["position_ids"].dim() == (3 if engine.use_mrope else 2)
                and static["input_ids"].dtype == torch.int32
                and static["position_ids"].dtype == torch.int32):
            static_ids, static_pos = static["input_ids"], static["position_ids"]
        skip_blocks = self.skip_block_copies and attn_steady
        draft_manager = getattr(attn_metadata, "draft_kv_cache_manager", None)
        self.record = _Record(
            block_counts=_block_counts(kv_cache_manager, list(request_ids)) if skip_blocks else None,
            draft_block_counts=(_block_counts(draft_manager, list(request_ids))
                                if skip_blocks and draft_manager is not None else None),
            static_ids=static_ids,
            static_pos=static_pos,
            seq_lens_kv=seq_lens_kv,
            attn_steady=attn_steady,
            request_ids=list(request_ids),
            slots=slots,
            prompt_lengths=list(prompt_lengths),
            sequence_lengths=list(sequence_lengths),
            draft_lens=list(draft_lens),
            tokens_per_step=tokens_per_step,
            runtime_draft_len=engine.runtime_draft_len,
            num_extra_kv_tokens=get_num_extra_kv_tokens(engine.spec_config),
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            resource_manager=resource_manager,
            kv_cache_manager=kv_cache_manager,
            has_mamba_chunk_size=hasattr(engine.model.model_config.pretrained_config,
                                         "chunk_size"),
            mamba_chunk_size=getattr(engine.model.model_config.pretrained_config, "chunk_size",
                                     None),
            use_mrope=bool(engine.use_mrope),
        )

    # ------------------------------------------------------------------ fast path
    def try_prepare(self, engine, scheduled_requests, kv_cache_manager, attn_metadata,
                    spec_metadata, new_tensors_device, resource_manager,
                    promoted_context_request_ids, full_prepare_fn
                    ) -> Optional[Tuple[Dict[str, Any], torch.Tensor]]:
        if not self._batch_matches(engine, scheduled_requests, attn_metadata, spec_metadata,
                                   new_tensors_device, promoted_context_request_ids):
            return None
        if (self.verify_every > 0 and (self.n_fast + 1) % self.verify_every == 0):
            return self._verify(engine, scheduled_requests, kv_cache_manager, attn_metadata,
                                spec_metadata, new_tensors_device, resource_manager,
                                full_prepare_fn)
        return self._fast(engine, scheduled_requests, kv_cache_manager, attn_metadata,
                          spec_metadata, new_tensors_device, resource_manager)

    def _next_slab(self) -> int:
        k = self.ring_pos
        self.ring_pos = (k + 1) % _RING
        event = self.slab_events[k]
        if event is not None and not event.query():
            # The upload issued from this slot _RING steps ago has not run yet; with the
            # overlap loop's per-step sync this does not happen, but never overwrite it early.
            event.synchronize()
        return k

    def _fast(self, engine, scheduled_requests, kv_cache_manager, attn_metadata, spec_metadata,
              new_tensors_device, resource_manager):
        rec = self.record
        gen = scheduled_requests.generation_requests
        b = len(gen)
        t = rec.tokens_per_step
        d = t - 1
        n = b * t
        padded_num_tokens, can_run_piecewise, _ = engine._get_padding_params(
            n, 0, engine._get_all_rank_num_tokens(attn_metadata))
        if padded_num_tokens != n:
            return None
        # Host integers of this step (the full pass computes the same values).
        k = self._next_slab()
        view = self.host_views[k]
        num_cached = []
        kv_lens = []
        for i, request in enumerate(gen):
            past_seen = request.max_beam_num_tokens - 1
            view[i] = rec.slots[i]
            view[b + i] = request.py_num_accepted_draft_tokens
            base = 2 * b + i * t
            view[base:base + t] = np.arange(past_seen, past_seen + t, dtype=np.int32)
            num_cached.append(past_seen + t - request.py_num_compressed_tokens)
            kv_lens.append(num_cached[-1] + rec.seq_lens_kv[i])
            request.cached_tokens = past_seen + t
            request.py_batch_idx = request.py_seq_slot
        used = 2 * b + n
        if rec.attn_steady:
            view[used:used + b] = kv_lens
            used += b
        if self.zero_copy:
            slab = self.host_slabs[k]  # read by the kernel through the pinned buffer's device mapping
        else:
            self.device_slab[:used].copy_(self.host_slabs[k][:used], non_blocking=True)
            slab = self.device_slab

        stage_steady_inputs(
            slab, b, t, new_tensors_device,
            input_ids=engine.input_ids_cuda,
            draft_tokens=engine.draft_tokens_cuda,
            num_accepted=engine.num_accepted_draft_tokens_cuda,
            prev_batch_idx=engine.previous_batch_indices_cuda,
            prev_pos_idx=engine.previous_pos_indices_cuda,
            prev_pos_offsets=engine.previous_pos_id_offsets_cuda,
            prev_kv_offsets=engine.previous_kv_lens_offsets_cuda,
            positions=engine.position_ids_cuda,
            mrope_positions=engine.mrope_position_ids_cuda if rec.use_mrope else None,
            gather_ids=engine.gather_ids_cuda,
            kv_lens_cuda=attn_metadata.kv_lens_cuda if rec.attn_steady else None,
            static_input_ids=rec.static_ids,
            static_position_ids=rec.static_pos,
            block=self.block)
        # The host slot may be rewritten once its reader (the copy, or the kernel itself) has run.
        event = self.slab_events[k]
        if event is None:
            event = self.slab_events[k] = torch.cuda.Event()
        event.record()
        if rec.static_ids is not None:
            # The replay that follows finds its static inputs staged (CUDAGraphRunner.replay).
            engine.cuda_graph_runner._q35_static_staged = n

        attn_metadata.beam_width = 1
        attn_metadata.request_ids = rec.request_ids
        attn_metadata.prompt_lens = rec.prompt_lengths
        attn_metadata.num_contexts = 0
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached,
            num_extra_kv_tokens=rec.num_extra_kv_tokens)
        attn_metadata.kv_cache_manager = kv_cache_manager
        if rec.has_mamba_chunk_size:
            attn_metadata.mamba_chunk_size = rec.mamba_chunk_size
        if rec.attn_steady:
            _steady_attn_prepare(attn_metadata, kv_lens, rec.num_extra_kv_tokens, self)
        else:
            attn_metadata.prepare()
        set_per_request_piecewise_cuda_graph_flag(can_run_piecewise)
        attn_metadata.padded_num_tokens = None

        if rec.use_mrope:
            final_position_ids = engine.mrope_position_ids_cuda[:, :, :n]
        else:
            final_position_ids = engine.position_ids_cuda[:n].unsqueeze(0)
        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': engine.input_ids_cuda[:n],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            'multimodal_params': [],
            'resource_manager': resource_manager,
        }
        spec_metadata.draft_tokens = engine.draft_tokens_cuda[:b * d]
        spec_metadata.request_ids = rec.request_ids
        spec_metadata.gather_ids = engine.gather_ids_cuda[:n]
        spec_metadata.num_generations = b
        spec_metadata.num_tokens = n
        spec_metadata.seq_lens = rec.sequence_lengths
        spec_metadata.num_accepted_draft_tokens = engine.num_accepted_draft_tokens_cuda[:b]
        spec_metadata.populate_sampling_params_for_one_model(scheduled_requests.all_requests())
        spec_metadata.prepare()
        spec_metadata.write_padding_onehot_draft_probs([], engine.runtime_draft_len)
        inputs['spec_metadata'] = spec_metadata

        engine.iter_states['num_ctx_requests'] = 0
        engine.iter_states['num_ctx_tokens'] = 0
        engine.iter_states['num_generation_tokens'] = n
        engine.iter_states['cached_kv_tokens'] = sum(num_cached)
        engine.previous_request_ids = list(rec.request_ids)
        engine.has_previous_device_draft = True
        self.n_fast += 1
        if self.stats_every and (self.n_fast + self.n_full) % self.stats_every == 0:
            logger.info(f"[q35_steady_prepare] fast={self.n_fast} full={self.n_full} "
                        f"verified={self.n_verified} realloc_stale={self.n_realloc_stale}")
        return inputs, engine.gather_ids_cuda[:n]

    # ------------------------------------------------------------------ verification
    @staticmethod
    def _snapshot(engine, attn_metadata, spec_metadata, b, t) -> Dict[str, Any]:
        n = b * t
        snap = {
            "input_ids": engine.input_ids_cuda[:n].clone(),
            "draft_tokens": engine.draft_tokens_cuda[:b * (t - 1)].clone(),
            "num_accepted": engine.num_accepted_draft_tokens_cuda[:b].clone(),
            "prev_batch_idx": engine.previous_batch_indices_cuda[:b].clone(),
            "prev_pos_idx": engine.previous_pos_indices_cuda[:n].clone(),
            "prev_pos_offsets": engine.previous_pos_id_offsets_cuda.clone(),
            "prev_kv_offsets": engine.previous_kv_lens_offsets_cuda.clone(),
            "positions": engine.position_ids_cuda[:n].clone(),
            "gather_ids": engine.gather_ids_cuda[:n].clone(),
            "kv_lens_cuda": attn_metadata.kv_lens_cuda[:b].clone(),
            "kv_lens": attn_metadata.kv_lens[:b].clone(),
            "num_cached": list(attn_metadata.kv_cache_params.num_cached_tokens_per_seq),
            "extra": attn_metadata.kv_cache_params.num_extra_kv_tokens,
            "request_ids": list(attn_metadata.request_ids),
            "prompt_lens": list(attn_metadata.prompt_lens),
            "num_contexts": attn_metadata.num_contexts,
            "host_total_kv_lens": attn_metadata.host_total_kv_lens.clone(),
            "prompt_lens_cuda": attn_metadata.prompt_lens_cuda[:b].clone(),
            "prompt_lens_cpu": attn_metadata.prompt_lens_cpu[:b].clone(),
            "host_request_types": attn_metadata.host_request_types[:b].clone(),
            "kv_lens_runtime": attn_metadata.kv_lens_runtime.clone(),
            "kv_lens_cuda_runtime_ptr": attn_metadata.kv_lens_cuda_runtime.data_ptr(),
            "kv_lens_cuda_runtime_n": attn_metadata.kv_lens_cuda_runtime.shape[0],
            "block_offsets": attn_metadata.kv_cache_block_offsets.clone(),
            "spec_seq_lens": list(spec_metadata.seq_lens),
            "spec_num_tokens": spec_metadata.num_tokens,
            "spec_num_generations": spec_metadata.num_generations,
            "iter_states": dict(engine.iter_states),
        }
        if engine.use_mrope:
            snap["mrope"] = engine.mrope_position_ids_cuda[:, :, :n].clone()
        # Mamba metadata and the hybrid manager's buffers (TRTLLM_Q35_STEADY_MAMBA_INDICES skips their uploads; the
        # GDN replay kernels read the slot indices, the dummy mask and the replay work items).
        mamba = getattr(attn_metadata, "mamba_metadata", None)
        if mamba:
            for name in ("state_indices", "state_indices_long"):
                value = getattr(mamba, name, None)
                if isinstance(value, torch.Tensor):
                    snap["mamba." + name] = value[:b].clone()
            for name, value in vars(mamba).items():
                if name.startswith("replay_"):
                    snap["mamba." + name] = value.clone() if isinstance(value, torch.Tensor) else value
        manager = attn_metadata.kv_cache_manager
        for name in ("cuda_state_indices", "_dummy_request_mask"):
            value = getattr(manager, name, None)
            if isinstance(value, torch.Tensor):
                snap["manager." + name] = value.clone()
        return snap

    @staticmethod
    def _mamba_ground_truth(attn_metadata, request_ids: List[int]) -> List[str]:
        """Fields whose Mamba state indices / dummy mask differ from what the hybrid KV cache manager (v2) says for
        this batch right now. Both the full pass and the fast path read the indices through the same cached uploads
        (TRTLLM_Q35_STEADY_MAMBA_INDICES), so their agreement alone would not catch a stale upload."""
        manager = attn_metadata.kv_cache_manager
        mamba = getattr(attn_metadata, "mamba_metadata", None)
        kv_map = getattr(manager, "kv_cache_map", None)
        group = getattr(manager, "ssm_layer_group_id", None)
        if (not mamba or kv_map is None or group is None
                or getattr(manager, "local_num_mamba_layers", 0) == 0):
            return []
        b = len(request_ids)
        expected = [kv_map[rid].get_ssm_block_base_index(group) for rid in request_ids]
        bad = []
        indices = getattr(manager, "cuda_state_indices", None)
        if isinstance(indices, torch.Tensor) and indices[:b].tolist() != expected:
            bad.append("manager.cuda_state_indices vs the KV caches' SSM base blocks")
        for name in ("state_indices", "state_indices_long"):
            value = getattr(mamba, name, None)
            if isinstance(value, torch.Tensor) and value[:b].tolist() != expected:
                bad.append(f"mamba.{name} vs the KV caches' SSM base blocks")
        mask = getattr(manager, "_dummy_request_mask", None)
        flags = getattr(manager, "_request_id_to_is_dummy", None)
        if isinstance(mask, torch.Tensor) and flags is not None:
            want = [bool(flags.get(rid, False)) for rid in request_ids]
            want += [False] * (mask.shape[0] - b)
            if [bool(v) for v in mask.tolist()] != want:
                bad.append("manager._dummy_request_mask vs the batch's dummy flags")
        return bad

    @staticmethod
    def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        bad = []
        for key in a:
            x, y = a[key], b.get(key)
            if isinstance(x, torch.Tensor):
                same = (isinstance(y, torch.Tensor) and x.shape == y.shape
                        and x.dtype == y.dtype and torch.equal(x.cpu(), y.cpu()))
            else:
                same = x == y
            if not same:
                bad.append(key)
        return bad

    def _verify(self, engine, scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, resource_manager, full_prepare_fn):
        rec = self.record
        b, t = len(rec.request_ids), rec.tokens_per_step
        # The block-offset tables as the previous steps left them, and whether the fast path would skip their
        # copies at this step (block keys still those of the last copy): a skipped copy must find the full pass's
        # fresh table equal to the old one. Decided before the full pass, which re-records the keys.
        tables = []
        if rec.attn_steady and self.skip_block_copies:
            managers = [("block_offsets", kv_cache_manager, "kv_cache_block_offsets", rec.block_counts)]
            if attn_metadata.draft_kv_cache_manager is not None:
                managers.append(("draft_block_offsets", attn_metadata.draft_kv_cache_manager,
                                 "draft_kv_cache_block_offsets", rec.draft_block_counts))
            for name, manager, attr, last_key in managers:
                key = _block_counts(manager, rec.request_ids)
                if key is None or last_key is None:
                    continue
                if key == last_key:
                    tables.append((name, attr, getattr(attn_metadata, attr).clone()))
                elif [k[0] for k in key] == [k[0] for k in last_key]:
                    # Same block counts, other tail pages: a count-only key would have skipped this copy.
                    tables.append(("realloc:" + name, attr, getattr(attn_metadata, attr).clone()))
        full_inputs, full_gather = full_prepare_fn()
        # The full pass re-recorded (same batch); keep its record for the fast path.
        if self.record is None:
            return full_inputs, full_gather
        reference = self._snapshot(engine, attn_metadata, spec_metadata, b, t)
        fresh_tables = [(name, old, getattr(attn_metadata, attr).clone()) for name, attr, old in tables]
        fast = self._fast(engine, scheduled_requests, kv_cache_manager, attn_metadata,
                          spec_metadata, new_tensors_device, resource_manager)
        if fast is None:
            return full_inputs, full_gather
        fast_inputs, fast_gather = fast
        torch.cuda.current_stream().synchronize()
        mismatch = self._diff(reference, self._snapshot(engine, attn_metadata, spec_metadata,
                                                        b, t))
        mismatch += self._mamba_ground_truth(attn_metadata, rec.request_ids)
        for name, old, fresh in fresh_tables:
            old, fresh = old.cpu(), fresh.cpu()
            if name.startswith("realloc:"):
                self.n_realloc_stale += int(not torch.equal(old, fresh))
                continue
            if not torch.equal(old, fresh):
                where = (old != fresh).nonzero()
                first = tuple(where[0].tolist())
                mismatch.append(f"skipped_{name}_copy ({where.shape[0]} entries, first {first}: "
                                f"{int(old[first])} -> {int(fresh[first])})")
        if rec.static_ids is not None:
            n = b * t
            if not torch.equal(rec.static_ids[:n].cpu(), reference["input_ids"].cpu()):
                mismatch.append("static_input_ids")
            ref_pos = reference["mrope"] if engine.use_mrope else reference["positions"].view(1, -1)
            if not torch.equal(rec.static_pos[..., :n].cpu(), ref_pos.cpu()):
                mismatch.append("static_position_ids")
        if (set(full_inputs) != set(fast_inputs)
                or fast_inputs['input_ids'].data_ptr() != full_inputs['input_ids'].data_ptr()
                or tuple(fast_inputs['position_ids'].shape) != tuple(
                    full_inputs['position_ids'].shape)
                or fast_gather.data_ptr() != full_gather.data_ptr()
                or fast_gather.shape != full_gather.shape):
            mismatch.append("inputs")
        self.n_verified += 1
        if mismatch:
            message = (f"[q35_steady_prepare] fast path differs from the full pass in {mismatch} "
                       f"(batch {rec.request_ids}); steady prepare disabled")
            if self.verify_strict:
                raise RuntimeError(message)
            logger.error(message)
            self.disabled = True
            self.record = None
            engine.cuda_graph_runner._q35_static_staged = None
            # The full pass's values were overwritten by identical-shape writes only where the
            # fast path ran; redo the full pass so this step uses its own result.
            return full_prepare_fn()
        return fast_inputs, fast_gather
