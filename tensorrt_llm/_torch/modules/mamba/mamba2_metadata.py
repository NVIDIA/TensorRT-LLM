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
import math
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.fla.flashinfer_chunk import \
    invalidate_int32_cu_seqlens_cache
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


def _build_replay_work_items_torch(state_indices, prev_num_accepted_tokens,
                                   cache_buf_idx, work_items, n_writes,
                                   replay_step_width, replay_history_size):
    """Same partition as :func:`_build_replay_work_items_triton`, in ATen ops.

    Keep field order and write-first partitioning in sync with the AutoDeploy
    replay metadata path in shim/interface.py.
    """
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


def build_fold_segments(
    ctx_seq_lens: List[int],
    ctx_state_indices: List[int],
    folds: List[Optional[Tuple[int, int]]],
    decode_state_indices: List[int],
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int],
           List[int]]:
    """Segment layout of one batch for the linear-attention scan when some
    context chunks fold the save-last snapshot point into a single chunk.

    Returns ``(scan_cu_seqlens, scan_state_indices, s1, s2, conv_tok, b_rows,
    b_cu_seqlens)``:

    * ``scan_cu_seqlens`` / ``scan_state_indices``: the scan's first launch.
      Every context request contributes one segment with its own slot, except
      a folded one, which contributes segment A ``[pos, pos+off)`` on its
      primary slot S1 (the snapshot block; final state = the snapshot) and
      segment B ``[pos+off, end)`` on the terminal slot S2 whose result is
      discarded (overwritten by the second launch). Decode requests follow
      with one token each, as in the unfolded layout.
    * ``s1`` / ``s2``: the two slots of every folded request, in batch order.
    * ``conv_tok``: token index (into the packed prefill tokens) of the fold
      point, i.e. the first token of segment B; the conv state of S1 is the
      ``d_conv - 1`` raw inputs right before it.
    * ``b_rows`` / ``b_cu_seqlens``: the packed token rows of all B segments
      and their cumulative lengths, for the second launch (initial state from
      S1, final state into S2).
    """
    scan_cu = [0]
    scan_idx: List[int] = []
    s1: List[int] = []
    s2: List[int] = []
    conv_tok: List[int] = []
    b_rows: List[int] = []
    b_cu = [0]
    start = 0
    for length, slot, fold in zip(ctx_seq_lens, ctx_state_indices, folds):
        if fold is None:
            scan_cu.append(start + length)
            scan_idx.append(slot)
        else:
            off, terminal = fold
            if not (0 < off < length):
                raise ValueError(
                    f"fold offset {off} must lie strictly inside the chunk "
                    f"(length {length})")
            scan_cu.append(start + off)
            scan_idx.append(slot)
            scan_cu.append(start + length)
            scan_idx.append(terminal)
            s1.append(slot)
            s2.append(terminal)
            conv_tok.append(start + off)
            b_rows.extend(range(start + off, start + length))
            b_cu.append(b_cu[-1] + (length - off))
        start += length
    for slot in decode_state_indices:
        start += 1
        scan_cu.append(start)
        scan_idx.append(slot)
    return scan_cu, scan_idx, s1, s2, conv_tok, b_rows, b_cu


def fold_b_ranges(b_rows: List[int], b_cu: List[int]) -> List[Tuple[int, int]]:
    """``(first row, length)`` of every folded tail in the packed token space.

    Each tail [fold point, chunk end) is a contiguous row range of the packed
    prefill tokens; ``b_rows`` lists them back to back and ``b_cu`` delimits
    the folds. The fused single-fold path slices q/k/v/g/beta and the output
    with these instead of gathering rows.
    """
    return [(b_rows[b_cu[i]], b_cu[i + 1] - b_cu[i])
            for i in range(len(b_cu) - 1)]


class Mamba2Metadata:

    # Warmup-only knob: when set via ``force_initial_states_for_warmup``,
    # ``prepare()`` forces ``has_initial_states_cpu[:num_contexts]`` to True so
    # the ``HAS_INITSTATES=True`` variants of the SSD Triton kernels compile
    # during warmup. Class-scoped (not env-var) so it cannot leak into real
    # inference from a stray shell export or a forked worker.
    _warmup_force_initial_states: bool = False

    @classmethod
    @contextlib.contextmanager
    def force_initial_states_for_warmup(cls):
        prev = cls._warmup_force_initial_states
        cls._warmup_force_initial_states = True
        try:
            yield
        finally:
            cls._warmup_force_initial_states = prev

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
        # Host gate for the GDN state reset (see gdn_mixer._reset_prefill_states):
        # True when a context request of this iteration starts without a
        # recurrent state; state_reset_done flips once the first GDN layer has
        # cleared those slots for every layer.
        self.prefill_needs_state_reset = True
        self.state_reset_done = False
        self.chunk_indices: torch.Tensor = None
        self.chunk_offsets: torch.Tensor = None

        self.state_indices_cpu = torch.zeros(max_batch_size,
                                             dtype=torch.int32,
                                             pin_memory=prefer_pinned())
        self.state_indices = torch.zeros(max_batch_size,
                                         dtype=torch.int32,
                                         device="cuda")
        # The decode requests' slots of a mixed iteration in their own buffer:
        # the FlashInfer decode kernel wants an aligned base pointer, which the
        # tail slice ``state_indices[num_contexts:]`` only has when the number
        # of context requests is a multiple of 4 (see _mixed_decode_recurrent).
        self.state_indices_decode = torch.zeros(max_batch_size,
                                                dtype=torch.int32,
                                                device="cuda")
        # Stable data_ptr() of the CUDA tensor we alias (if any) — used to
        # detect cache-manager buffer reallocation that would silently break
        # CUDA graph replays.
        self._state_indices_aliased_ptr = None
        # Per-iteration GDN call context (Qwen3NextGatedDeltaNet.forward_core):
        # the first GDN layer of an iteration builds the metadata-derived
        # kwargs (batch split, slot / initial-state views, scan layout) once and
        # the other layers reuse them. Cleared by prepare().
        self.gdn_iteration_kwargs = None
        # True when state_indices_cpu mirrors state_indices for the current
        # batch (list / CPU-tensor sources); the fold layout needs host values.
        self._state_indices_host_valid = False

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
        self._cu_seqlens_long = torch.zeros(max_batch_size + 1,
                                            dtype=torch.long,
                                            device="cuda")

        # Folded save-last prefill (see build_fold_segments). ``fold_count``
        # is 0 on every iteration without a folded chunk, in which case the
        # scan_* views alias the unfolded layout above.
        self.fold_count = 0
        self.scan_cu_seqlens_long: torch.Tensor = None
        self.scan_state_indices: torch.Tensor = None
        self.fold_s1: torch.Tensor = None
        self.fold_s2: torch.Tensor = None
        self.fold_conv_tok: torch.Tensor = None
        self.fold_b_rows: torch.Tensor = None
        self.fold_b_cu_seqlens_long: torch.Tensor = None
        self._scan_cu_seqlens_long_buf = torch.zeros(2 * max_batch_size + 1,
                                                     dtype=torch.long,
                                                     device="cuda")
        self._scan_state_indices_buf = torch.zeros(2 * max_batch_size,
                                                   dtype=torch.int32,
                                                   device="cuda")
        self._fold_s1_buf = torch.zeros(max_batch_size,
                                        dtype=torch.int32,
                                        device="cuda")
        self._fold_s2_buf = torch.zeros(max_batch_size,
                                        dtype=torch.int32,
                                        device="cuda")
        self._fold_conv_tok_buf = torch.zeros(max_batch_size,
                                              dtype=torch.long,
                                              device="cuda")
        self._fold_b_cu_seqlens_long_buf = torch.zeros(max_batch_size + 1,
                                                       dtype=torch.long,
                                                       device="cuda")
        self._fold_b_rows_buf: torch.Tensor = None  # grown on demand
        # ``[0, len_i]`` pairs, one per folded tail: the cu_seqlens of the
        # per-fold tail scan launches (fold_scan_tails), staged once per
        # iteration.
        self._fold_tail_cu_buf = torch.zeros(2 * max_batch_size,
                                             dtype=torch.long,
                                             device="cuda")
        # S1 / S2 slot of every fold as its own 32-byte aligned int32 element
        # (row stride 8): the FlashInfer recurrent tail kernel takes one slot
        # per launch and requires an aligned index pointer, which a
        # ``fold_s1[i:i + 1]`` slice only has for i == 0.
        self._fold_slot_aligned_buf = torch.zeros(2,
                                                  max_batch_size,
                                                  8,
                                                  dtype=torch.int32,
                                                  device="cuda")
        # Per-iteration constants of the fused fold bookkeeping (gdn_mixer
        # fold_conv_tail / fold_commit_conv_states / fold_scan_tails): the
        # per-layer path used to rebuild them with 5 launches per GDN layer.
        self.fold_b_ranges_host: List[Tuple[int, int]] = []
        self._fold_s1_host: List[int] = []
        self._fold_s2_host: List[int] = []
        self._fold_conv_tok_host: List[int] = []
        self._fold_s1_long: torch.Tensor = None
        self._fold_s2_long: torch.Tensor = None
        self._fold_conv_tail_idx: torch.Tensor = None
        self._fold_conv_tail_width = 0
        self._fold_s1_long_buf = torch.zeros(max_batch_size,
                                             dtype=torch.long,
                                             device="cuda")
        self._fold_s2_long_buf = torch.zeros(max_batch_size,
                                             dtype=torch.long,
                                             device="cuda")
        self._fold_conv_tail_idx_buf: torch.Tensor = None  # grown on demand

    def arange_long(self, n: int) -> torch.Tensor:
        """``[0, 1, ..., n]`` as int64 on the device: the cu_seqlens of ``n`` one-token sequences."""
        return self._arange_buffer_long[:n + 1]

    def fold_tail_cu_seqlens_long(self, i: int) -> torch.Tensor:
        """``[0, len_i]`` (int64, device): cu_seqlens of the ``i``-th folded tail scanned alone."""
        return self._fold_tail_cu_buf[2 * i:2 * i + 2]

    def fold_s1_aligned(self, i: int) -> torch.Tensor:
        """``[S1_i]`` (int32, device, 32-byte aligned): the snapshot slot of the ``i``-th fold."""
        return self._fold_slot_aligned_buf[0, i, :1]

    def fold_s2_aligned(self, i: int) -> torch.Tensor:
        """``[S2_i]`` (int32, device, 32-byte aligned): the terminal slot of the ``i``-th fold."""
        return self._fold_slot_aligned_buf[1, i, :1]

    @property
    def fold_s1_long(self) -> torch.Tensor:
        """``fold_s1`` as int64 (one H2D copy per iteration instead of a
        ``.long()`` launch per GDN layer)."""
        if self._fold_s1_long is None:
            n = self.fold_count
            self._fold_s1_long_buf[:n].copy_(torch.tensor(self._fold_s1_host,
                                                          dtype=torch.long),
                                             non_blocking=True)
            self._fold_s1_long = self._fold_s1_long_buf[:n]
        return self._fold_s1_long

    @property
    def fold_s2_long(self) -> torch.Tensor:
        if self._fold_s2_long is None:
            n = self.fold_count
            self._fold_s2_long_buf[:n].copy_(torch.tensor(self._fold_s2_host,
                                                          dtype=torch.long),
                                             non_blocking=True)
            self._fold_s2_long = self._fold_s2_long_buf[:n]
        return self._fold_s2_long

    def conv_tail_index(self, width: int) -> torch.Tensor:
        """Packed-token indices of the ``width`` pre-conv inputs before each
        fold point, flat ``[fold_count * width]`` int64 on the device.

        Built on the host once per iteration (the fold points are host
        values) and shared by every GDN layer; the per-layer path spent an
        arange + sub + add launch per layer on the same tensor.
        """
        if (self._fold_conv_tail_idx is None
                or self._fold_conv_tail_width != width):
            need = self.fold_count * width
            if (self._fold_conv_tail_idx_buf is None
                    or self._fold_conv_tail_idx_buf.numel() < need):
                self._fold_conv_tail_idx_buf = torch.zeros(
                    max(need, self.max_batch_size * max(width, 1)),
                    dtype=torch.long,
                    device=self._fold_conv_tok_buf.device)
            host = torch.tensor([
                tok - width + j for tok in self._fold_conv_tok_host
                for j in range(width)
            ],
                                dtype=torch.long)
            self._fold_conv_tail_idx_buf[:need].copy_(host, non_blocking=True)
            self._fold_conv_tail_idx = self._fold_conv_tail_idx_buf[:need]
            self._fold_conv_tail_width = width
        return self._fold_conv_tail_idx

    def prepare_steady_gen_step(self, attn_metadata: AttentionMetadata):
        """``prepare()`` for a decode-only batch whose request layout (ids,
        order, state slots, padding rows) is unchanged since the last
        ``prepare()``: the state indices and the dummy-request mask are still
        in place, so their host walk and H2D copies are skipped; only the
        per-step bookkeeping is refreshed."""
        self.prefill_needs_state_reset = False
        self.state_reset_done = False
        self.gdn_iteration_kwargs = None
        batch_size = attn_metadata.seq_lens.shape[0]
        kv_cache_manager = attn_metadata.kv_cache_manager

        self._prepare_replay_work_items(kv_cache_manager, batch_size, 0)
        self._prepare_fold_segments(kv_cache_manager, attn_metadata,
                                    batch_size, 0)

        self.query_start_loc = None
        self.query_start_loc_long = self._arange_buffer_long[:batch_size + 1]

        flush = getattr(kv_cache_manager, "flush_state_transfers", None)
        if flush is not None:
            flush()

    def _prepare_replay_work_items(self, kv_cache_manager, batch_size: int,
                                   num_contexts: int):
        self.replay_num_decodes = 0
        if not getattr(kv_cache_manager, 'use_replay_state_update', False):
            return
        num_decodes = batch_size - num_contexts
        self.replay_num_decodes = num_decodes
        if num_decodes == 0:
            return
        use_gdn_all_layer_commit = getattr(
            kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False)
        if use_gdn_all_layer_commit:
            from tensorrt_llm._torch.modules.fla.cached_replay import \
                CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

            # The fused small-batch GDN kernel commits its checkpoint in-layer
            # and indexes cache metadata directly. Work items are only consumed
            # by the partitioned replay + all-layer commit path.
            if num_decodes < CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE:
                return

        if not hasattr(kv_cache_manager, 'get_replay_state_update_metadata'):
            raise RuntimeError(
                "Replay state update is enabled, but the KV cache manager "
                "does not expose replay state update metadata.")

        replay_metadata = kv_cache_manager.get_replay_state_update_metadata()
        if replay_metadata is None:
            raise RuntimeError(
                "Replay state update is enabled for a decode batch, but the "
                "KV cache manager returned no replay state update metadata.")

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

    def _prepare_fold_segments(self, kv_cache_manager, attn_metadata,
                               batch_size: int, num_contexts: int) -> None:
        """Layout for the folded save-last prefill (see build_fold_segments).

        Cheap no-op (``fold_count = 0``) unless the cache manager reports a
        context chunk that carries its save-last snapshot point.
        """
        self.fold_count = 0
        self.scan_cu_seqlens_long = None
        self.scan_state_indices = None
        self.fold_b_ranges_host = []
        self._fold_s1_host = []
        self._fold_s2_host = []
        self._fold_conv_tok_host = []
        self._fold_s1_long = None
        self._fold_s2_long = None
        self._fold_conv_tail_idx = None
        get_fold_info = getattr(kv_cache_manager, "get_fold_info", None)
        if get_fold_info is None or num_contexts == 0:
            return
        request_ids = attn_metadata.request_ids
        if request_ids is None or not self._state_indices_host_valid:
            # The cache manager may have chosen the snapshot slot as a folded
            # chunk's primary slot; running such a chunk unfolded would leave
            # the terminal slot unwritten. Fail loudly instead.
            if getattr(kv_cache_manager, "_request_id_to_fold", None):
                raise RuntimeError(
                    "folded save-last chunks are pending but the mamba metadata "
                    "cannot build their layout (request_ids or host state "
                    "indices unavailable)")
            return
        folds = get_fold_info(list(request_ids[:num_contexts]))
        if not any(f is not None for f in folds):
            return
        seq_lens = attn_metadata.seq_lens[:batch_size].tolist()
        state_indices = self.state_indices_cpu[:batch_size].tolist()
        (scan_cu, scan_idx, s1, s2, conv_tok, b_rows,
         b_cu) = build_fold_segments(seq_lens[:num_contexts],
                                     state_indices[:num_contexts], folds,
                                     state_indices[num_contexts:])
        n = len(s1)
        nseg = len(scan_idx)
        dev = self._scan_cu_seqlens_long_buf.device
        self._scan_cu_seqlens_long_buf[:nseg + 1].copy_(torch.tensor(
            scan_cu, dtype=torch.long),
                                                        non_blocking=True)
        self._scan_state_indices_buf[:nseg].copy_(torch.tensor(
            scan_idx, dtype=torch.int32),
                                                  non_blocking=True)
        self._fold_s1_buf[:n].copy_(torch.tensor(s1, dtype=torch.int32),
                                    non_blocking=True)
        self._fold_s2_buf[:n].copy_(torch.tensor(s2, dtype=torch.int32),
                                    non_blocking=True)
        self._fold_conv_tok_buf[:n].copy_(torch.tensor(conv_tok,
                                                       dtype=torch.long),
                                          non_blocking=True)
        self._fold_b_cu_seqlens_long_buf[:n + 1].copy_(torch.tensor(
            b_cu, dtype=torch.long),
                                                       non_blocking=True)
        nb = len(b_rows)
        if self._fold_b_rows_buf is None or self._fold_b_rows_buf.numel() < nb:
            self._fold_b_rows_buf = torch.zeros(max(nb,
                                                    32 * self.max_batch_size),
                                                dtype=torch.long,
                                                device=dev)
        self._fold_b_rows_buf[:nb].copy_(torch.tensor(b_rows, dtype=torch.long),
                                         non_blocking=True)
        tail_cu = []
        for i in range(n):
            tail_cu += (0, b_cu[i + 1] - b_cu[i])
        self._fold_tail_cu_buf[:2 * n].copy_(torch.tensor(tail_cu,
                                                          dtype=torch.long),
                                             non_blocking=True)
        self.fold_count = n
        self.scan_cu_seqlens_long = self._scan_cu_seqlens_long_buf[:nseg + 1]
        self.scan_state_indices = self._scan_state_indices_buf[:nseg]
        self.fold_s1 = self._fold_s1_buf[:n]
        self.fold_s2 = self._fold_s2_buf[:n]
        self._fold_slot_aligned_buf[0, :n, 0].copy_(self.fold_s1)
        self._fold_slot_aligned_buf[1, :n, 0].copy_(self.fold_s2)
        self.fold_conv_tok = self._fold_conv_tok_buf[:n]
        self.fold_b_rows = self._fold_b_rows_buf[:nb]
        self.fold_b_cu_seqlens_long = self._fold_b_cu_seqlens_long_buf[:n + 1]
        # Host copies for the fused bookkeeping (see the lazy properties).
        self._fold_s1_host = s1
        self._fold_s2_host = s2
        self._fold_conv_tok_host = conv_tok
        self.fold_b_ranges_host = fold_b_ranges(b_rows, b_cu)

    def prepare(self, attn_metadata: AttentionMetadata):
        # The cu_seqlens buffers below are rewritten in place: drop the GDN
        # launch path's per-iteration int32 casts of them first.
        invalidate_int32_cu_seqlens_cache()
        # Every iteration starts with the reset undone; the context branch below
        # decides whether any request needs it at all.
        self.prefill_needs_state_reset = False
        self.state_reset_done = False
        self.gdn_iteration_kwargs = None
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
                self._state_indices_host_valid = False
            elif isinstance(indices, torch.Tensor):
                # CPU tensor → bulk H2D
                self.state_indices_cpu[:batch_size].copy_(indices[:batch_size])
                self.state_indices[:batch_size].copy_(
                    self.state_indices_cpu[:batch_size], non_blocking=True)
                self._state_indices_host_valid = True
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
                self.state_indices[:batch_size].copy_(
                    self.state_indices_cpu[:batch_size], non_blocking=True)
                self._state_indices_host_valid = True

        num_decodes = batch_size - num_contexts
        if num_contexts > 0 and num_decodes > 0:
            self.state_indices_decode[:num_decodes].copy_(
                self.state_indices[num_contexts:batch_size],
                non_blocking=True)

        self._prepare_replay_work_items(kv_cache_manager, batch_size,
                                        num_contexts)

        self._prepare_fold_segments(kv_cache_manager, attn_metadata, batch_size,
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
            # Only context requests without a state need their slots zeroed;
            # continuation chunks (the common case under prefix reuse) skip the
            # reset launches in every GDN layer. Warmup forces the flag on so the
            # reset kernel is compiled before the first real iteration.
            self.prefill_needs_state_reset = bool(
                Mamba2Metadata._warmup_force_initial_states
                or not self.has_initial_states_cpu[:num_contexts].all())

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

        # Complete any deferred recurrent-state block onboards scheduled by
        # CppMambaHybridCacheManager.prepare_resources(). prepare_resources
        # only enqueues the async cudaMemcpyAsync calls and sets a pending
        # flag; we sync the onboard stream here, so CPU-side prep
        # work in _prepare_tp_inputs overlaps with the in-flight transfers.
        # Cheap no-op on cache managers without this method or when no
        # transfers were scheduled this iteration.
        flush = getattr(kv_cache_manager, "flush_state_transfers", None)
        if flush is not None:
            flush()
