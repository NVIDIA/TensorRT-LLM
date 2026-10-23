# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state one-model MTP decode input preparation (Qwen3.5 fast paths, ``q35_env.fast_paths_enabled()``).

A decode step whose batch is the previous step's (same requests and slots, each verifying the previous
step's device drafts, same CUDA graph metadata, no padding) needs only a few new integers per request. The
preparer records a full ``DecoderInputPreparer.prepare_inputs`` pass of such a step (``record_batch``); later
matching steps write those integers to a pinned host slab (3-slot ring, fenced by events) and one Triton
kernel stages every device input from it, including the CUDA graph's static inputs.
``_steady_attn_prepare`` replaces ``TrtllmAttentionMetadata.prepare()``.

Any other step (batch or slot change, context rows, dummies, multimodal, LoRA, guided decoding, beams,
attention DP, helix CP, warmup) takes the full pass, which records again when it qualifies.
"""

import types
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

from ..attention.backends.interface import AttentionMetadata
from ..attention.backends.trtllm import TrtllmAttentionMetadata
from ..utils import get_global_attrs, get_model_extra_attrs

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from .cuda_graph_runner import CUDAGraphRunner
    from .llm_request import LlmRequest

__all__ = ["SteadyMtpPrepare", "SteadyMtpRecord", "stage_steady_inputs"]

# Depth of the host slab ring.
_RING = 3

# Integer args unspecialized, pointers not specialized on alignment, one BLOCK for every batch size: one variant
# per engine, compiled during warmup instead of a JIT stall (up to ~0.3 s) at each new batch size.
_STAGE_INT_ARGS = [
    "nnt_tok_stride",
    "nnt_slot_stride",
    "nd_slot_stride",
    "mrope_axis_stride",
    "static_pos_axis_stride",
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
]


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
    # draft rows: request i // D, draft i % D
    drf = i < B * D
    slot_d = tl.load(slab + i // D, mask=drf, other=0)
    dval = tl.load(next_drafts + slot_d * nd_slot_stride + i % D, mask=drf, other=0)
    tl.store(draft_tokens + i, dval, mask=drf)


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
) -> None:
    """One launch of the steady staging kernel.

    Args:
        slab: int32 [b + b * t (+ b with ``kv_lens_cuda``)]: the sequence slots, the positions
            and the KV lengths of the batch; a device tensor or a pinned host tensor (read
            through its device mapping).
        b: Number of requests.
        t: Tokens per request (draft length + 1).
        new_tensors_device: The previous step's device sampler tensors (``new_tokens``
            [>= t, num_slots, 1], ``next_draft_tokens`` [num_slots, >= t - 1],
            ``new_tokens_lens`` [num_slots]).
        input_ids, draft_tokens, prev_batch_idx, prev_pos_idx, prev_pos_offsets, prev_kv_offsets,
            positions, gather_ids: The decoder input buffers the full pass writes.
        mrope_positions: The [3, 1, N] mrope position buffer, or None without mrope.
        kv_lens_cuda: ``TrtllmAttentionMetadata.kv_lens_cuda`` when the slab carries the KV lengths.
        static_input_ids, static_position_ids: The CUDA graph's static inputs, when staged here.
        block: The BLOCK of the launch when it covers the batch (one variant for all batch sizes),
            else the batch's own power of two.
    """
    d = t - 1
    n = b * t
    need = max(n, b * d, b, 1)
    if block is None or block < need:
        block = triton.next_power_of_2(need)
    new_tokens = new_tensors_device.new_tokens
    next_drafts = new_tensors_device.next_draft_tokens
    has_mrope = mrope_positions is not None
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


def _block_counts(kv_cache_manager: Any, request_ids: list[int]) -> tuple[Any, ...] | None:
    """Key of the batch's block-offset table rows (KVCacheManagerV2), or None when unavailable.

    Per request: the block count and the page indices of its last ``_TAIL_BLOCKS`` blocks (the draft rewind can
    re-allocate the last block on another page at an unchanged count). None when the device table is not a plain
    copy of the host table (per-layer page tables, sparse metadata) or blocks can change otherwise (block reuse).
    """
    kv_map = getattr(kv_cache_manager, "kv_cache_map", None)
    table = getattr(kv_cache_manager, "host_kv_cache_block_offsets", None)
    mapper = getattr(kv_cache_manager, "index_mapper", None)
    if (
        kv_map is None
        or table is None
        or mapper is None
        or getattr(kv_cache_manager, "enable_swa_scratch_reuse", False)
        or getattr(kv_cache_manager, "enable_block_reuse", False)
        or getattr(kv_cache_manager, "_use_per_layer_page_tables", False)
        or getattr(kv_cache_manager, "sparse_metadata_batch", None) is not None
    ):
        return None
    host = table.numpy()
    beam = int(getattr(kv_cache_manager, "max_beam_width", 1))
    keys = []
    for rid in request_ids:
        cache = kv_map.get(rid)
        if cache is None:
            return None
        n = int(cache.num_blocks)
        row = mapper.get_index(rid) * beam
        keys.append((n, host[:, row, 0, max(0, n - _TAIL_BLOCKS) : n].tobytes()))
    return tuple(keys)


@dataclass
class SteadyMtpRecord:
    """What the last qualifying full pass left, keyed by the batch it ran."""

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
    seq_lens_kv: list[int]
    # Whether _steady_attn_prepare replaces attn_metadata.prepare() (and the slab carries the KV lengths).
    attn_steady: bool
    # The CUDA graph's static input ids / positions the staging kernel writes, or None.
    static_ids: torch.Tensor | None
    static_pos: torch.Tensor | None
    # _block_counts keys of the last block-offset copy of the target / draft KV cache manager.
    block_counts: tuple[Any, ...] | None
    draft_block_counts: tuple[Any, ...] | None


def _steady_attn_prepare(
    attn_metadata: TrtllmAttentionMetadata,
    kv_lens: list[int],
    record: SteadyMtpRecord,
    state: "SteadyMtpPrepare",
) -> None:
    """``TrtllmAttentionMetadata.prepare()`` of a steady decode step.

    Request ids, prompt lengths and request types are still what the recording pass uploaded (any other
    batch is a full pass, which drops the record) and the staging kernel wrote ``kv_lens_cuda[:b]``; this
    sets only the host state prepare() leaves.
    """
    AttentionMetadata.prepare(attn_metadata)  # TrtllmAttentionMetadata.prepare's super().prepare()
    attn_metadata._invalidate_mla_scheduler_buffers()
    if get_model_extra_attrs() is None:
        get_global_attrs().attention_metadata = weakref.ref(attn_metadata)
    n = len(kv_lens)
    num_extra_kv_tokens = record.num_extra_kv_tokens
    host = state.kv_lens_host
    state.kv_lens_host_np[:n] = kv_lens
    attn_metadata.kv_lens[:n].copy_(host[:n] + num_extra_kv_tokens)
    attn_metadata.host_total_kv_lens[0] = 0
    attn_metadata.host_total_kv_lens[1] = sum(kv_lens)
    max_kv_len = max(kv_lens) + num_extra_kv_tokens
    kv_cache_manager = attn_metadata.kv_cache_manager
    assert max_kv_len <= kv_cache_manager.max_seq_len, (
        f"The max KV cache length of input sequences ({max_kv_len}) exceeds the KV cache "
        f"manager's maximum supported length ({kv_cache_manager.max_seq_len})."
    )
    # Rows of a steady batch change only at their tails: skip the copies while every block count and tail page is
    # what the last copy saw. Speculative decoding copies the full table width (max_blocks=None), as prepare() does.
    counts = _block_counts(kv_cache_manager, attn_metadata.request_ids)
    if counts is None or counts != record.block_counts:
        kv_cache_manager.copy_batch_block_offsets(
            attn_metadata.kv_cache_block_offsets,
            attn_metadata.request_ids,
            attn_metadata.beam_width,
            attn_metadata.num_contexts,
            n,
            max_blocks=None,
        )
        record.block_counts = counts
    draft = attn_metadata.draft_kv_cache_manager
    if draft is not None:
        draft_counts = _block_counts(draft, attn_metadata.request_ids)
        if draft_counts is None or draft_counts != record.draft_block_counts:
            draft.copy_batch_block_offsets(
                attn_metadata.draft_kv_cache_block_offsets,
                attn_metadata.request_ids,
                attn_metadata.beam_width,
                attn_metadata.num_contexts,
                n,
                max_blocks=None,
            )
            record.draft_block_counts = draft_counts
    attn_metadata._bind_runtime_views(
        kv_lens_cuda=attn_metadata.kv_lens_cuda[:n],
        kv_lens=host[:n].clone(),
        prompt_lens_cuda=attn_metadata.prompt_lens_cuda[:n],
        prompt_lens_cpu=attn_metadata.prompt_lens_cpu[:n],
        host_request_types=attn_metadata.host_request_types[:n],
    )


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
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_tokens_per_step: int,
        use_mrope: bool,
        has_static_inputs: bool,
    ) -> None:
        self.record: SteadyMtpRecord | None = None
        self.use_mrope = use_mrope
        max_requests = max(1, int(max_batch_size))
        self.slab_capacity = 3 * max_requests + int(max_num_tokens)
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
        self.kv_lens_host = torch.zeros(max_requests, dtype=torch.int32)
        self.kv_lens_host_np = self.kv_lens_host.numpy()
        self.ring_pos = 0
        # One staging-kernel BLOCK for every batch size of this engine, and its variant compiled now.
        self.block: int | None = None
        if max_tokens_per_step >= 2:
            self.block = triton.next_power_of_2(max_requests * max_tokens_per_step)
            self._warm_stage_kernel(max_tokens_per_step, has_static_inputs)

    def _warm_stage_kernel(self, t: int, has_static_inputs: bool) -> None:
        """Compile the staging kernel variant the steady steps launch (same constexprs, unspecialized ints and
        alignments) with a B = 0 launch, which reads and writes nothing."""
        if torch.cuda.is_current_stream_capturing():
            return
        dummy = torch.zeros(1, dtype=torch.int32, device="cuda")
        stage_steady_inputs(
            dummy,
            0,
            t,
            types.SimpleNamespace(
                new_tokens=dummy.view(1, 1, 1),
                next_draft_tokens=dummy.view(1, 1),
                new_tokens_lens=dummy,
            ),
            input_ids=dummy,
            draft_tokens=dummy,
            prev_batch_idx=dummy,
            prev_pos_idx=dummy,
            prev_pos_offsets=dummy,
            prev_kv_offsets=dummy,
            positions=dummy,
            mrope_positions=dummy.view(1, 1, 1) if self.use_mrope else None,
            gather_ids=dummy,
            kv_lens_cuda=dummy,
            static_input_ids=dummy if has_static_inputs else None,
            static_position_ids=(
                (dummy.view(1, 1, 1) if self.use_mrope else dummy.view(1, 1))
                if has_static_inputs
                else None
            ),
            block=self.block,
        )

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
        kv_cache_manager: Any,
        attn_metadata: AttentionMetadata,
        spec_metadata: "SpecMetadata",
        static_tensors: dict[str, torch.Tensor] | None,
    ) -> None:
        """Record a full pass that prepared ``requests``, all verifying their previous device drafts.

        The caller has checked the batch-level conditions (see ``DecoderInputPreparer._record_steady_mtp``).
        """
        b = len(requests)
        n_tokens = b * tokens_per_step
        if 3 * b + n_tokens > self.slab_capacity:
            return
        attn_steady = (
            type(attn_metadata) is TrtllmAttentionMetadata
            and attn_metadata.kv_cache_manager is not None
            and not attn_metadata.enable_flash_mla
            and not attn_metadata.enable_helix
            and not attn_metadata.enable_context_mla_with_cached_kv
            and getattr(attn_metadata, "fp4_mla_state", None) is None
            and not use_full_generation_page_table
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
        request_ids = [request.py_request_id for request in requests]
        draft_manager = getattr(attn_metadata, "draft_kv_cache_manager", None)
        self.record = SteadyMtpRecord(
            request_ids=request_ids,
            slots=[request.py_seq_slot for request in requests],
            prompt_lengths=list(prompt_lengths),
            sequence_lengths=list(sequence_lengths),
            tokens_per_step=tokens_per_step,
            runtime_draft_len=runtime_draft_len,
            num_extra_kv_tokens=num_extra_kv_tokens,
            use_full_generation_page_table=use_full_generation_page_table,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            seq_lens_kv=attn_metadata.seq_lens_kv[:b].tolist(),
            attn_steady=attn_steady,
            static_ids=static_ids,
            static_pos=static_pos,
            block_counts=_block_counts(kv_cache_manager, request_ids) if attn_steady else None,
            draft_block_counts=(
                _block_counts(draft_manager, request_ids)
                if attn_steady and draft_manager is not None
                else None
            ),
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
            or new_tensors_device is None
            or getattr(new_tensors_device, "next_draft_tokens", None) is None
            or getattr(new_tensors_device, "new_tokens_lens", None) is None
            or len(requests) != len(rec.request_ids)
        ):
            return False
        for request, rid, slot in zip(requests, rec.request_ids, rec.slots):
            if (
                request.py_request_id != rid
                or request.py_seq_slot != slot
                or request.py_batch_idx != slot
                or request.is_dummy
                or request.py_multimodal_data is not None
                or getattr(request, "py_needs_onehot_draft_probs", False)
            ):
                return False
        return True

    def _next_slab(self) -> int:
        k = self.ring_pos
        self.ring_pos = (k + 1) % _RING
        event = self.slab_events[k]
        if event is not None and not event.query():
            # The kernel launched from this slot _RING steps ago has not run yet; with the
            # overlap loop's per-step sync this does not happen, but never overwrite it early.
            event.synchronize()
        return k

    def stage(
        self,
        requests: list["LlmRequest"],
        new_tensors_device: Any,
        buffers: Any,
        attn_metadata: AttentionMetadata,
        cuda_graph_runner: "CUDAGraphRunner | None",
    ) -> tuple[list[int], list[int]]:
        """Write this step's device inputs (one slab write, one kernel launch).

        Args:
            requests: The matched generation requests.
            new_tensors_device: The previous step's device sampler tensors.
            buffers: The preparer's DecoderInputBuffers.
            attn_metadata: The recorded attention metadata.
            cuda_graph_runner: The runner whose next replay finds its static inputs staged.

        Returns:
            The cached-token counts (``KVCacheParams.num_cached_tokens_per_seq``) and the KV lengths
            (without the extra spec tokens) of the requests.
        """
        rec = self.record
        b = len(requests)
        t = rec.tokens_per_step
        n = b * t
        k = self._next_slab()
        view = self.host_views[k]
        num_cached = []
        kv_lens = []
        for i, request in enumerate(requests):
            past_seen = request.max_beam_num_tokens - 1
            view[i] = rec.slots[i]
            base = b + i * t
            view[base : base + t] = np.arange(past_seen, past_seen + t, dtype=np.int32)
            num_cached.append(past_seen + t - request.py_num_compressed_tokens)
            kv_lens.append(num_cached[-1] + rec.seq_lens_kv[i])
        used = b + n
        if rec.attn_steady:
            view[used : used + b] = kv_lens
            used += b
        if self.zero_copy:
            slab = self.host_slabs[k]
        else:
            self.device_slab[:used].copy_(self.host_slabs[k][:used], non_blocking=True)
            slab = self.device_slab

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
        )
        # The host slot may be rewritten once its reader (the copy, or the kernel itself) has run.
        event = self.slab_events[k]
        if event is None:
            event = self.slab_events[k] = torch.cuda.Event()
        event.record()
        if rec.static_ids is not None and cuda_graph_runner is not None:
            cuda_graph_runner.set_static_inputs_staged(n)
        return num_cached, kv_lens

    def prepare_attention(self, attn_metadata: AttentionMetadata, kv_lens: list[int]) -> None:
        """The attention metadata's prepare() of a staged steady step."""
        rec = self.record
        if rec.attn_steady:
            _steady_attn_prepare(attn_metadata, kv_lens, rec, self)
        else:
            attn_metadata.prepare()
