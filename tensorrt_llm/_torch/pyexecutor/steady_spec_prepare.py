# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state one-model MTP decode input preparation.

A decode step whose batch is the previous step's (same requests and slots, each verifying the previous
step's device drafts, same CUDA graph metadata, no padding) needs only a few new integers per request. The
preparer records a full ``DecoderInputPreparer.prepare_inputs`` pass of such a step (``record_batch``); later
matching steps write those integers to a pinned host slab (3-slot ring, fenced by events) and one Triton
kernel stages every device input from it, including the CUDA graph's static inputs. The attention
metadata's ``prepare_steady_decode_step`` stands in for its ``prepare()`` where it supports it.

Whether a full pass qualifies is decided when it is recorded (``DecoderInputPreparer._record_steady_mtp``);
a later step matches when it runs the recorded requests on the recorded slots and metadata. Any other step
(batch or slot change, context rows, dummies, multimodal, LoRA, guided decoding, beams, attention DP, helix
CP, warmup) takes the full pass, which records again when it qualifies.
"""

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

from ..attention.backends.interface import AttentionMetadata
from ..attention.backends.trtllm import TrtllmAttentionMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from .llm_request import LlmRequest

__all__ = ["SteadyMtpPrepare", "SteadyMtpRecord", "stage_steady_inputs"]

# Depth of the host slab ring.
_RING = 3

# Integer args unspecialized, pointers not specialized on alignment, one BLOCK for every batch size: one variant
# per engine, compiled during warmup instead of at each new batch size.
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
    # Whether attn_metadata.prepare_steady_decode_step replaces its prepare() (and the slab carries the
    # KV lengths).
    attn_steady: bool
    # The CUDA graph's static input ids / positions the staging kernel writes, or None.
    static_ids: torch.Tensor | None
    static_pos: torch.Tensor | None


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
        self.record = SteadyMtpRecord(
            request_ids=[request.py_request_id for request in requests],
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
        ):
            return False
        return all(
            request.py_request_id == rid
            and request.py_seq_slot == slot
            and request.py_batch_idx == slot
            and self.request_qualifies(request)
            for request, rid, slot in zip(requests, rec.request_ids, rec.slots)
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

    def stage(
        self,
        requests: list["LlmRequest"],
        new_tensors_device: Any,
        buffers: Any,
        attn_metadata: AttentionMetadata,
    ) -> tuple[list[int], list[int], int]:
        """Write this step's device inputs (one slab write, one kernel launch).

        Args:
            requests: The matched generation requests.
            new_tensors_device: The previous step's device sampler tensors.
            buffers: The preparer's DecoderInputBuffers.
            attn_metadata: The recorded attention metadata.

        Returns:
            The cached-token counts (``KVCacheParams.num_cached_tokens_per_seq``), the KV lengths
            (without the extra spec tokens) of the requests, and the number of tokens written to the
            CUDA graph's static inputs (0 when none were).
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
        return num_cached, kv_lens, n if rec.static_ids is not None else 0

    def prepare_attention(self, attn_metadata: AttentionMetadata, kv_lens: list[int]) -> None:
        """The attention metadata's prepare() of a staged steady step."""
        if self.record.attn_steady:
            attn_metadata.prepare_steady_decode_step(kv_lens)
        else:
            attn_metadata.prepare()
