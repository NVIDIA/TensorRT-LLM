# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import itertools
import math
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
    CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._utils import prefer_pinned

REPLAY_WORK_POSITION_IN_DECODE_BATCH = 0
REPLAY_WORK_CACHE_SLOT = 1
REPLAY_WORK_PNAT = 2
REPLAY_WORK_CACHE_BUF_IDX = 3
REPLAY_WORK_ITEM_WIDTH = 4
_FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE = 256


@triton.jit
def _prepare_gdn_replay_work_items_kernel(
    state_indices,
    prev_num_accepted_tokens,
    cache_buf_idx,
    work_items,
    n_writes_output,
    num_decodes,
    replay_step_width: tl.constexpr,
    replay_history_size: tl.constexpr,
    work_item_width: tl.constexpr,
    position_field: tl.constexpr,
    cache_slot_field: tl.constexpr,
    pnat_field: tl.constexpr,
    cache_buf_idx_field: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build the write-first GDN replay partition in one launch."""
    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < num_decodes
    slots = tl.load(state_indices + offsets, mask=active, other=0)
    pnat = tl.load(prev_num_accepted_tokens + slots, mask=active, other=0)
    active_buffer = tl.load(cache_buf_idx + slots, mask=active, other=0)
    writes = active & (pnat + replay_step_width > replay_history_size)
    writes_i32 = writes.to(tl.int32)
    inclusive_write_offsets = tl.cumsum(writes_i32, axis=0)
    write_offsets = inclusive_write_offsets - writes_i32
    n_writes = tl.sum(writes_i32, axis=0)
    no_write_offsets = offsets - write_offsets
    output_offsets = tl.where(writes, write_offsets,
                              n_writes + no_write_offsets)
    output_base = work_items + output_offsets * work_item_width
    tl.store(output_base + position_field, offsets, mask=active)
    tl.store(output_base + cache_slot_field, slots, mask=active)
    tl.store(output_base + pnat_field, pnat, mask=active)
    tl.store(output_base + cache_buf_idx_field, active_buffer, mask=active)
    tl.store(n_writes_output, n_writes)


def _build_replay_work_items_triton(state_indices, prev_num_accepted_tokens,
                                    cache_buf_idx, work_items, n_writes,
                                    replay_step_width, replay_history_size):
    """Single-launch build of the write-first replay partition.

    Interchangeable with :func:`_build_replay_work_items_torch`; the caller
    picks between them. Kept to one CTA because the write-first offsets come
    from an in-block ``tl.cumsum``.
    """
    num_decodes = state_indices.shape[0]
    _prepare_gdn_replay_work_items_kernel[(1, )](
        state_indices,
        prev_num_accepted_tokens,
        cache_buf_idx,
        work_items,
        n_writes,
        num_decodes,
        replay_step_width=replay_step_width,
        replay_history_size=replay_history_size,
        work_item_width=REPLAY_WORK_ITEM_WIDTH,
        position_field=REPLAY_WORK_POSITION_IN_DECODE_BATCH,
        cache_slot_field=REPLAY_WORK_CACHE_SLOT,
        pnat_field=REPLAY_WORK_PNAT,
        cache_buf_idx_field=REPLAY_WORK_CACHE_BUF_IDX,
        BLOCK_SIZE=triton.next_power_of_2(num_decodes),
        num_warps=4,
    )


@torch.compile(dynamic=True, fullgraph=True)
def _build_replay_work_items_torch(state_indices, prev_num_accepted_tokens,
                                   cache_buf_idx, work_items, n_writes,
                                   replay_step_width, replay_history_size):
    """Same partition as :func:`_build_replay_work_items_triton`, in ATen ops."""
    num_decodes = state_indices.shape[0]
    position_in_decode_batch = torch.arange(num_decodes,
                                            dtype=torch.int32,
                                            device=state_indices.device)
    cache_slot_idx = state_indices.to(torch.long)
    pnat = prev_num_accepted_tokens[cache_slot_idx].to(torch.int32)
    active_cache_buf_idx = cache_buf_idx[cache_slot_idx].to(torch.int32)

    writes = (pnat + replay_step_width > replay_history_size)
    writes_i32 = writes.to(torch.int32)
    write_offsets = torch.cumsum(writes_i32, dim=0) - writes_i32
    batch_n_writes = torch.sum(writes_i32, dim=0, keepdim=True).to(torch.int32)
    no_write_offsets = position_in_decode_batch - write_offsets
    output_offsets = torch.where(writes, write_offsets,
                                 batch_n_writes + no_write_offsets)
    output_offsets = output_offsets.to(torch.long)

    decode_work_items = work_items[:num_decodes]
    decode_work_items[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH].scatter_(
        0, output_offsets, position_in_decode_batch)
    decode_work_items[:,
                      REPLAY_WORK_CACHE_SLOT].scatter_(0, output_offsets,
                                                       state_indices)
    decode_work_items[:, REPLAY_WORK_PNAT].scatter_(0, output_offsets, pnat)
    decode_work_items[:, REPLAY_WORK_CACHE_BUF_IDX].scatter_(
        0, output_offsets, active_cache_buf_idx)
    n_writes.copy_(batch_n_writes)


@triton.jit
def _cu_seqlens_triton_kernel(
    cu_seqlens_ptr,  # [num_seqs + 1]
    chunk_indices_ptr,  # [N] output
    chunk_offsets_ptr,  # [N] output
    num_seqs,
    chunk_size: tl.constexpr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Computes chunk_indices and chunk_offsets in a single kernel launch."""
    pid = tl.program_id(0)
    chunk_start = pid * BLOCK_SIZE
    offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    chunk_indices = offsets.to(tl.int64)
    chunk_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    p = 0
    for seq_idx in range(num_seqs - 1):
        seq_start = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int64)
        seq_end = tl.load(cu_seqlens_ptr + seq_idx + 2).to(tl.int64)
        is_misaligned = (seq_start % chunk_size) > 0
        p = p + is_misaligned
        s_chunk = seq_start // chunk_size + p
        e_chunk = seq_end // chunk_size + p + ((seq_end % chunk_size) > 0)
        in_range = (offsets >= s_chunk) & (offsets < e_chunk)
        chunk_indices = tl.where(in_range & mask, chunk_indices - p,
                                 chunk_indices)
        is_start = (offsets == s_chunk)
        chunk_offsets = tl.where(is_start & mask, seq_start % chunk_size,
                                 chunk_offsets)

    tl.store(chunk_indices_ptr + offsets, chunk_indices.to(tl.int32), mask=mask)
    tl.store(chunk_offsets_ptr + offsets, chunk_offsets.to(tl.int32), mask=mask)


def compute_extra_chunks_cpu(seq_lens, num_seqs: int, chunk_size: int) -> int:
    """Count extra chunks caused by misaligned sequence boundaries.

    Computes from CPU seq_lens to avoid GPU->CPU synchronization.
    """
    cumsum = 0
    extra = 0
    for i in range(num_seqs - 1):
        cumsum += int(seq_lens[i])
        if cumsum % chunk_size != 0:
            extra += 1
    return extra


def cu_seqlens_to_chunk_indices_offsets_triton(
        cu_seqlens: torch.Tensor,
        chunk_size: int,
        total_seqlens: int = -1,
        extra_chunks: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized version of cu_seqlens_to_chunk_indices_offsets.

    Args:
        total_seqlens: If provided (>= 0), avoids a GPU->CPU sync to read
            cu_seqlens[-1].  Callers that already know the total number of
            context tokens should pass it here.
        extra_chunks: If provided (>= 0), avoids a GPU->CPU sync to compute
            the number of extra chunks from misaligned sequence boundaries.
    """
    device = cu_seqlens.device
    num_seqs = cu_seqlens.numel() - 1

    if num_seqs == 0:
        return (torch.empty(0, dtype=torch.int, device=device),
                torch.empty(0, dtype=torch.int, device=device))

    cu = cu_seqlens.to(dtype=torch.int64)
    if total_seqlens < 0:
        total_seqlens = cu[-1].item()

    if num_seqs == 1:
        # Fast path for single sequence (no boundaries to process)
        N = (total_seqlens + chunk_size - 1) // chunk_size
        return (torch.arange(N, device=device, dtype=torch.int),
                torch.zeros(N, device=device, dtype=torch.int))

    if extra_chunks < 0:
        seq_starts = cu[1:-1]
        misaligned = ((seq_starts % chunk_size) > 0).to(torch.int64)
        p = torch.cumsum(misaligned, dim=0)
        extra_chunks = p[-1].item() if p.numel() > 0 else 0
    N = (total_seqlens + chunk_size - 1) // chunk_size + extra_chunks
    chunk_indices = torch.empty(N, device=device, dtype=torch.int)
    chunk_offsets = torch.empty(N, device=device, dtype=torch.int)

    BLOCK_SIZE = 256
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _cu_seqlens_triton_kernel[grid](
        cu,
        chunk_indices,
        chunk_offsets,
        num_seqs=num_seqs,
        chunk_size=chunk_size,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return chunk_indices, chunk_offsets


def cu_seqlens_to_chunk_indices_offsets(
        cu_seqlens: torch.Tensor,
        chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cu_seqlens (torch.Tensor): 1D tensor of cumulative sequence lengths,
            shape (num_seqs + 1,). The first element should be 0. Each entry
            represents the starting index of a sequence in the flattened token
            array.
        chunk_size (int): The size of each physical mamba chunk (number of tokens per chunk).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - chunk_indices (torch.Tensor): 1D tensor of indices indicating the physical chunk for each logical chunk.
            - chunk_offsets (torch.Tensor): 1D tensor of offsets indicating
              the starting index of each logical chunk within its physical
              chunk.

    This function computes the chunk indices and offsets for the given cu_seqlens and chunk_size.
    Both are tensors of integers with length N, where N is the number of logical (pseudo) chunks.
    A logical chunk is a sequence of tokens that are all part of the same sequence
    and are all in the same physical mamba chunk.
    In other words, a logical chunk changes every time we cross a sequence boundary or a physical mamba chunk boundary.
    Logical chunks are needed to handle batched requests with initial states
    (see _state_passing_fwd and _chunk_scan_fwd).
    The chunk_indices tensor contains the index of the physical chunk for each logical chunk.
    The chunk_offsets tensor contains the offset (AKA starting index) of the logical chunk in the physical chunk.

    Example:
    cu_seqlens = [0, 5, 10]
    chunk_size = 8
    -> chunk_indices = [0, 0, 1]
    -> chunk_offsets = [0, 5, 0]

    In this example, we have 2 sequences, each with 5 tokens. The physical chunk size is 8 tokens.
    We have three logical chunks:
    - the first logical chunk starts at token 0 in the first physical chunk and
      contains all 5 tokens from the first sequence
    - the second logical chunk starts at token 5 in the first physical chunk
      and contains first 3 tokens from the second sequence
    - the third logical chunk starts at token 0 in the second physical chunk
      and contains the remaining 2 tokens from the second sequence
    """

    total_seqlens = cu_seqlens[-1]
    cu_seqlens = cu_seqlens[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size
                                                 > 0).sum()
    chunk_indices = torch.arange(N, dtype=torch.int, device=cu_seqlens.device)
    chunk_offsets = torch.zeros((N, ),
                                dtype=torch.int,
                                device=cu_seqlens.device)

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += (s % chunk_size > 0)

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


class InforwardSnapshotSplit(NamedTuple):
    """Index tensors of one step's in-forward Mamba snapshot split.

    See ``Mamba2Metadata._prepare_inforward_split``. ``k`` requests are split;
    ``num_tail_tokens`` tokens lie in their tails.
    """
    k: int
    num_tail_tokens: int
    cu: torch.Tensor
    gather: torch.Tensor
    live: torch.Tensor
    main_rows: torch.Tensor
    split_rows: torch.Tensor
    stage_rows: torch.Tensor
    tail_tok: torch.Tensor
    cu_tail: torch.Tensor
    tail_slots: torch.Tensor
    conv_win: torch.Tensor


# Fields of InforwardSnapshotSplit, in upload order.
_INFORWARD_SPLIT_FIELDS = ("cu", "gather", "live", "main_rows", "split_rows",
                           "stage_rows", "tail_tok", "cu_tail", "tail_slots",
                           "conv_win")


class Mamba2Metadata:

    # Warmup-only knob: when set via ``force_initial_states_for_warmup``,
    # ``prepare()`` forces ``has_initial_states_cpu[:num_contexts]`` to True so
    # the ``HAS_INITSTATES=True`` variants of the SSD Triton kernels compile
    # during warmup. Class-scoped (not env-var) so it cannot leak into real
    # inference from a stray shell export or a forked worker.
    _warmup_force_initial_states: bool = False

    # Warmup-only knob, set via ``inforward_split_for_warmup``: ``prepare()``
    # plans an in-forward snapshot split for the context requests, so warmup
    # captures the CUDA graphs of the split GDN prefill core.
    _warmup_inforward_split: bool = False

    # Set by a subclass whose prefill kernel compiles a separate variant
    # depending on whether every context length is a multiple of this value.
    # Warmup reads it to cover both variants.
    prefill_chunk_alignment: Optional[int] = None

    @classmethod
    @contextlib.contextmanager
    def force_initial_states_for_warmup(cls):
        prev = cls._warmup_force_initial_states
        cls._warmup_force_initial_states = True
        try:
            yield
        finally:
            cls._warmup_force_initial_states = prev

    @classmethod
    @contextlib.contextmanager
    def inforward_split_for_warmup(cls):
        prev = cls._warmup_inforward_split
        cls._warmup_inforward_split = True
        try:
            yield
        finally:
            cls._warmup_inforward_split = prev

    def __init__(
        self,
        max_batch_size: int,
        chunk_size: int,
        max_num_tokens: int | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size
        self.max_num_tokens = max_num_tokens

        # cumulative sequence lengths for prefill requests [batch_size+1]
        self.cu_seqlens = torch.zeros(max_batch_size + 1,
                                      dtype=torch.int,
                                      device="cuda")

        # sequence index for prefill requests [num_prefill_tokens] - specifies which request each token belongs to
        self.seq_idx: torch.Tensor = None

        # helper tensors for chunked prefill
        self.has_initial_states_cpu = torch.zeros(max_batch_size,
                                                  dtype=torch.bool,
                                                  pin_memory=prefer_pinned())
        self.has_initial_states = torch.zeros(max_batch_size,
                                              dtype=torch.bool,
                                              device="cuda")
        self.use_initial_states = False
        self.chunk_indices: torch.Tensor = None
        self.chunk_offsets: torch.Tensor = None

        self.state_indices_cpu = torch.zeros(max_batch_size,
                                             dtype=torch.int32,
                                             pin_memory=prefer_pinned())
        self.state_indices = torch.zeros(max_batch_size,
                                         dtype=torch.int32,
                                         device="cuda")
        self._state_indices_buffer = DeviceCopyBuffer(
            self.state_indices, pin_memory=prefer_pinned())
        # Stable data_ptr() of the CUDA tensor we alias (if any) — used to
        # detect cache-manager buffer reallocation that would silently break
        # CUDA graph replays.
        self._state_indices_aliased_ptr = None

        self.replay_work_items = torch.zeros(max_batch_size,
                                             REPLAY_WORK_ITEM_WIDTH,
                                             dtype=torch.int32,
                                             device="cuda")
        self.replay_n_writes = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.replay_num_decodes = 0

        # Pre-allocated buffers.
        self._arange_buffer = torch.arange(max_batch_size + 1,
                                           dtype=torch.int,
                                           device="cuda")
        self._arange_buffer_long = self._arange_buffer.to(torch.long)

        # In-forward Mamba snapshot split of this step (None: no split), and
        # the buffer its index tensors live in at fixed field offsets, so they
        # keep their addresses across steps (sized on the first split from the
        # cache manager's staging capacity).
        self.inforward_split: Optional[InforwardSnapshotSplit] = None
        self._inforward_split_buf: Optional[torch.Tensor] = None
        self._inforward_split_offsets: list[int] = []
        self._cu_seqlens_long = torch.zeros(max_batch_size + 1,
                                            dtype=torch.long,
                                            device="cuda")

    def _prepare_replay_work_items(self, kv_cache_manager, batch_size: int,
                                   num_contexts: int):
        self.replay_num_decodes = 0
        get_replay_metadata = getattr(kv_cache_manager,
                                      'get_replay_state_update_metadata', None)
        replay_metadata = get_replay_metadata(
        ) if get_replay_metadata is not None else None
        if replay_metadata is None:
            return
        num_decodes = batch_size - num_contexts
        self.replay_num_decodes = num_decodes
        if num_decodes == 0:
            return
        # FlashInfer ring replay folds in-kernel and never reads work items.
        if getattr(kv_cache_manager, "use_gdn_flashinfer_ring_replay", False):
            return
        use_gdn_all_layer_commit = getattr(
            kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False)
        # The fused small-batch GDN kernel commits its checkpoint in-layer
        # and indexes cache metadata directly. Work items are only consumed
        # by the partitioned replay + all-layer commit path.
        if (use_gdn_all_layer_commit
                and not kv_cache_manager.gdn_replay_partitioned(num_decodes)):
            return

        prev_num_accepted_tokens = replay_metadata.prev_num_accepted_tokens
        cache_buf_idx = replay_metadata.cache_buf_idx
        replay_step_width = replay_metadata.replay_step_width
        replay_history_size = replay_metadata.replay_history_size

        if (use_gdn_all_layer_commit
                and num_decodes <= _FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE):
            build_work_items = _build_replay_work_items_triton
        else:
            build_work_items = _build_replay_work_items_torch
        build_work_items(
            self.state_indices[num_contexts:batch_size],
            prev_num_accepted_tokens,
            cache_buf_idx,
            self.replay_work_items,
            self.replay_n_writes,
            replay_step_width,
            replay_history_size,
        )

    def _prepare_inforward_split(self, attn_metadata: AttentionMetadata,
                                 kv_cache_manager, plan, num_contexts: int,
                                 batch_size: int) -> None:
        """Index tensors of the in-forward Mamba snapshot split.

        A planned request's chunk [start, start + len) becomes two chunked
        prefill sequences, [start, start + before) and the tail [start + before,
        start + len); both gather the request's live state. The GDN layer
        writes the first one's final state (the state at P) to the live slot,
        recomputes the tail from it with the outputs written back at
        ``tail_tok``, and stages the tail's final state (the state at the
        prompt end) in the request's row. ``conv_win`` holds the token
        positions of the convolution window that ends at P.
        """
        request_ids = attn_metadata.request_ids[:num_contexts]
        entries = [plan.get(rid) for rid in request_ids]
        k = sum(entry is not None for entry in entries)
        if k == 0:
            return
        if batch_size != num_contexts:
            raise RuntimeError("In-forward Mamba snapshot planned for a batch "
                               "with generation requests")
        width = kv_cache_manager.conv_state_shape[-1]
        rows_cap = kv_cache_manager.inforward_snapshot_rows
        tail_cap = rows_cap * kv_cache_manager.inforward_snapshot_max_tail
        lens = attn_metadata.seq_lens[:num_contexts].tolist()
        vals = {name: [] for name in _INFORWARD_SPLIT_FIELDS}
        vals["cu"].append(0)
        vals["cu_tail"].append(0)
        start = 0
        for rid, length, entry in zip(request_ids, lens, entries):
            slot = kv_cache_manager.state_index(rid)
            vals["live"].append(slot)
            vals["main_rows"].append(len(vals["gather"]))
            vals["gather"].append(slot)
            if entry is None:
                vals["cu"].append(start + length)
            else:
                before, after = entry.num_before, entry.num_after
                if before + after != length or before < width:
                    raise RuntimeError(
                        f"In-forward Mamba snapshot: request {rid} chunk of "
                        f"{length} tokens does not match the plan ({before} + "
                        f"{after}, window {width})")
                vals["split_rows"].append(len(vals["gather"]) - 1)
                vals["stage_rows"].append(entry.row)
                vals["cu"].append(start + before)
                # The tail as a sequence of its own: its outputs are recomputed
                # from the state at P, its final state is dropped.
                vals["gather"].append(slot)
                vals["cu"].append(start + length)
                vals["tail_tok"].extend(range(start + before, start + length))
                vals["cu_tail"].append(vals["cu_tail"][-1] + after)
                vals["tail_slots"].append(slot)
                vals["conv_win"].extend(
                    range(start + before - width, start + before))
            start += length
        num_tail_tokens = len(vals["tail_tok"])
        if k > rows_cap or num_tail_tokens > tail_cap:
            raise RuntimeError(
                f"In-forward Mamba snapshot: {k} requests / {num_tail_tokens} "
                "tail tokens exceed the staging capacity")
        if self._inforward_split_buf is None:
            batch_cap = self.max_batch_size
            capacity = (batch_cap + rows_cap + 1, batch_cap + rows_cap,
                        batch_cap, batch_cap, rows_cap, rows_cap, tail_cap,
                        rows_cap + 1, rows_cap, width * rows_cap)
            self._inforward_split_offsets = [0, *itertools.accumulate(capacity)]
            self._inforward_split_buf = torch.zeros(
                self._inforward_split_offsets[-1],
                dtype=torch.long,
                device="cuda")
        offsets = self._inforward_split_offsets
        flat = [0] * offsets[-1]
        for name, start in zip(_INFORWARD_SPLIT_FIELDS, offsets):
            flat[start:start + len(vals[name])] = vals[name]
        # One upload through a fresh pinned tensor: no later host write can
        # race the copy.
        self._inforward_split_buf.copy_(torch.tensor(
            flat, dtype=torch.long, pin_memory=prefer_pinned()),
                                        non_blocking=True)
        views = {
            name: self._inforward_split_buf[start:start + len(vals[name])]
            for name, start in zip(_INFORWARD_SPLIT_FIELDS, offsets)
        }
        self.inforward_split = InforwardSnapshotSplit(
            k=k, num_tail_tokens=num_tail_tokens, **views)

    @staticmethod
    def _warmup_inforward_split_plan(attn_metadata: AttentionMetadata,
                                     kv_cache_manager,
                                     num_contexts: int) -> dict:
        """Split each context request with room for the longest tail, up to the
        staging rows (``inforward_split_for_warmup``)."""
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import \
            InforwardSnapshotEntry
        tail = kv_cache_manager.inforward_snapshot_max_tail
        plan = {}
        for rid, length in zip(attn_metadata.request_ids[:num_contexts],
                               attn_metadata.seq_lens[:num_contexts].tolist()):
            if (len(plan) < kv_cache_manager.inforward_snapshot_rows
                    and length - tail
                    >= kv_cache_manager.inforward_snapshot_min_main):
                plan[rid] = InforwardSnapshotEntry(len(plan), length - tail,
                                                   length - tail, tail)
        return plan

    def prepare(self, attn_metadata: AttentionMetadata):
        batch_size = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        context_lens = attn_metadata.seq_lens_cuda[:num_contexts]
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        kv_cache_manager = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids

        if (kv_cache_manager is not None
                and hasattr(kv_cache_manager, 'get_state_indices')
                and request_ids is not None):
            batch_request_ids = request_ids[:batch_size]
            max_draft_len = getattr(kv_cache_manager,
                                    "speculative_num_draft_tokens", 0) or 0
            is_padding = [
                CUDA_GRAPH_DUMMY_REQUEST_ID - max_draft_len <= req_id <=
                CUDA_GRAPH_DUMMY_REQUEST_ID for req_id in batch_request_ids
            ]
            indices = kv_cache_manager.get_state_indices(
                batch_request_ids, is_padding)
            if isinstance(indices,
                          torch.Tensor) and indices.device.type == 'cuda':
                # Alias the cache manager's CUDA buffer directly instead of
                # copying. Iterating a CUDA tensor and assigning each 0-d
                # slice to a CPU tensor would trigger one cudaMemcpyAsync +
                # cudaStreamSynchronize per element.
                #
                # Safe under CUDA graphs only when the source buffer has a
                # stable data pointer across all calls. If a cache manager
                # reallocates this buffer between iterations, captured kernels
                # would still read from the address seen at capture time, so
                # we assert stability here.
                if self._state_indices_aliased_ptr is None:
                    self._state_indices_aliased_ptr = indices.data_ptr()
                else:
                    assert indices.data_ptr(
                    ) == self._state_indices_aliased_ptr, (
                        "kv_cache_manager.get_state_indices() must return a "
                        "buffer with a stable data pointer when CUDA graphs "
                        "are used; got a different address than the first "
                        "call.")
                self.state_indices = indices
                self._state_indices_buffer = kv_cache_manager._state_indices_buffer
            elif isinstance(indices, torch.Tensor):
                # CPU tensor → bulk H2D
                self.state_indices_cpu[:batch_size].copy_(indices[:batch_size])
                self._state_indices_buffer.update(
                    self.state_indices_cpu[:batch_size])
            else:
                # indices is a Python sequence (e.g. List[int]); data
                # already lives on host, CPU staging is fine. One bulk
                # conversion instead of a per-element tensor write.
                assert len(indices) == batch_size, (
                    f"get_state_indices() returned {len(indices)} entries for "
                    f"a batch of {batch_size} requests.")
                self.state_indices_cpu[:batch_size].copy_(
                    torch.as_tensor(indices,
                                    dtype=self.state_indices_cpu.dtype))
                self._state_indices_buffer.update(
                    self.state_indices_cpu[:batch_size])

        self._prepare_replay_work_items(kv_cache_manager, batch_size,
                                        num_contexts)

        if num_contexts > 0:
            torch.cumsum(context_lens,
                         dim=0,
                         dtype=torch.int,
                         out=self.cu_seqlens[1:num_contexts + 1])
            torch.add(self.cu_seqlens[num_contexts],
                      self._arange_buffer[1:batch_size - num_contexts + 1],
                      out=self.cu_seqlens[num_contexts + 1:batch_size + 1])
            # Need both `query_start_loc` and `query_start_loc_long` because `causal_conv1d_fn`
            # accepts only `int32` while `chunk_gated_delta_rule` accepts only `long`.
            self.query_start_loc = self.cu_seqlens[:batch_size + 1]
            self._cu_seqlens_long[:batch_size + 1].copy_(self.query_start_loc)
            self.query_start_loc_long = self._cu_seqlens_long[:batch_size + 1]
            self.seq_idx = torch.repeat_interleave(
                self._arange_buffer[:num_contexts],
                repeats=context_lens,
                output_size=num_ctx_tokens).unsqueeze(0)

            # Build "has initial state" flags on CPU first, then issue a
            # single async H2D copy from the pinned staging buffer.
            num_cached_tokens_per_seq = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
            if isinstance(num_cached_tokens_per_seq, torch.Tensor):
                # Keep this as a CPU bool view/tensor to avoid introducing an
                # implicit sync point while reading per-sequence cache status.
                initial_states_cpu = num_cached_tokens_per_seq[:num_contexts].to(
                    dtype=torch.bool, device='cpu')
            else:
                # Fallback when cache metadata is provided as a Python sequence.
                initial_states_cpu = torch.tensor([
                    num_cached_tokens_per_seq[i] > 0
                    for i in range(num_contexts)
                ],
                                                  dtype=torch.bool,
                                                  device='cpu')

            self.has_initial_states_cpu[:num_contexts].copy_(initial_states_cpu)
            # Warmup-only override: force HAS_INITSTATES=True path so the
            # HAS_INITSTATES=True variants of _state_passing_fwd_kernel,
            # _chunk_scan_fwd_kernel, and _chunk_state_varlen_kernel compile
            # during warmup instead of the first real-request iter that hits
            # chunked prefill with cached tokens. Gate is a class-scoped
            # context manager (see ``force_initial_states_for_warmup``) so it
            # cannot silently affect real inference.
            if Mamba2Metadata._warmup_force_initial_states:
                self.has_initial_states_cpu[:num_contexts].fill_(True)
            # Mirror CPU staging flags to the CUDA-side buffer asynchronously.
            self.has_initial_states[:num_contexts].copy_(
                self.has_initial_states_cpu[:num_contexts], non_blocking=True)
            # Keep a host boolean gate for chunk metadata construction.
            self.use_initial_states = bool(
                self.has_initial_states_cpu[:num_contexts].any())

            if self.use_initial_states:
                _extra = compute_extra_chunks_cpu(attn_metadata.seq_lens,
                                                  num_contexts, self.chunk_size)

                self.chunk_indices, self.chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
                    self.cu_seqlens[:num_contexts + 1],
                    self.chunk_size,
                    total_seqlens=num_ctx_tokens,
                    extra_chunks=_extra)
            else:
                self.chunk_indices = None
                self.chunk_offsets = None
        else:
            self.query_start_loc = None
            self.query_start_loc_long = self._arange_buffer_long[:batch_size +
                                                                 1]

        self.inforward_split = None
        plan = getattr(kv_cache_manager, "inforward_snapshot_plan", None)
        if (Mamba2Metadata._warmup_inforward_split and num_contexts > 0
                and request_ids is not None):
            plan = self._warmup_inforward_split_plan(attn_metadata,
                                                     kv_cache_manager,
                                                     num_contexts)
        if (isinstance(plan, dict) and plan and num_contexts > 0
                and request_ids is not None):
            self._prepare_inforward_split(attn_metadata, kv_cache_manager, plan,
                                          num_contexts, batch_size)

        # Complete any deferred recurrent-state block onboards scheduled by
        # CppMambaHybridCacheManager.prepare_resources(). prepare_resources
        # only enqueues the async cudaMemcpyAsync calls and sets a pending
        # flag; we sync the onboard stream here, so CPU-side prep
        # work in DecoderInputPreparer.prepare_inputs overlaps with the in-flight transfers.
        # Cheap no-op on cache managers without this method or when no
        # transfers were scheduled this iteration.
        flush = getattr(kv_cache_manager, "flush_state_transfers", None)
        if flush is not None:
            flush()
