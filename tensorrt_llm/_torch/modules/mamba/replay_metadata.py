# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba metadata of the Mamba2 and GDN replay states: replay slots and replay work items.

A hybrid cache manager whose replay state the kernels address by replay slots of
its own names this class as its ``mamba_metadata_cls``. The generic
``Mamba2Metadata`` carries none of it.
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._utils import prefer_pinned

from .mamba2_metadata import Mamba2Metadata

REPLAY_WORK_POSITION_IN_DECODE_BATCH = 0
REPLAY_WORK_CACHE_SLOT = 1
REPLAY_WORK_PNAT = 2
REPLAY_WORK_CACHE_BUF_IDX = 3
REPLAY_WORK_REPLAY_SLOT = 4
REPLAY_WORK_ITEM_WIDTH = 5
_FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE = 256


# Neither the decode count nor the alignment of the decode slices of the state
# and replay slots (they start after the step's context requests) is
# specialized: a mixed step would otherwise reach a new variant, and compile
# it, while serving.
@triton.jit(
    do_not_specialize=["num_decodes"],
    do_not_specialize_on_alignment=["state_indices", "replay_indices"],
)
def _prepare_gdn_replay_work_items_kernel(
    state_indices,
    replay_indices,
    prev_num_accepted_tokens,
    cache_buf_idx,
    work_items,
    n_writes_output,
    num_decodes,
    stride_pnat,
    stride_cache_buf_idx,
    replay_step_width: tl.constexpr,
    replay_history_size: tl.constexpr,
    work_item_width: tl.constexpr,
    position_field: tl.constexpr,
    cache_slot_field: tl.constexpr,
    pnat_field: tl.constexpr,
    cache_buf_idx_field: tl.constexpr,
    replay_slot_field: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Build the write-first GDN replay partition in one launch."""
    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < num_decodes
    slots = tl.load(state_indices + offsets, mask=active, other=0)
    replay_slots = tl.load(replay_indices + offsets, mask=active, other=0)
    pnat = tl.load(prev_num_accepted_tokens + replay_slots * stride_pnat, mask=active, other=0)
    active_buffer = tl.load(
        cache_buf_idx + replay_slots * stride_cache_buf_idx, mask=active, other=0
    )
    writes = active & (pnat + replay_step_width > replay_history_size)
    writes_i32 = writes.to(tl.int32)
    inclusive_write_offsets = tl.cumsum(writes_i32, axis=0)
    write_offsets = inclusive_write_offsets - writes_i32
    n_writes = tl.sum(writes_i32, axis=0)
    no_write_offsets = offsets - write_offsets
    output_offsets = tl.where(writes, write_offsets, n_writes + no_write_offsets)
    output_base = work_items + output_offsets * work_item_width
    tl.store(output_base + position_field, offsets, mask=active)
    tl.store(output_base + cache_slot_field, slots, mask=active)
    tl.store(output_base + pnat_field, pnat, mask=active)
    tl.store(output_base + cache_buf_idx_field, active_buffer, mask=active)
    tl.store(output_base + replay_slot_field, replay_slots, mask=active)
    tl.store(n_writes_output, n_writes)


def _build_replay_work_items_triton(
    state_indices,
    prev_num_accepted_tokens,
    cache_buf_idx,
    work_items,
    n_writes,
    replay_step_width,
    replay_history_size,
    replay_indices=None,
):
    """Single-launch build of the write-first replay partition.

    Interchangeable with :func:`_build_replay_work_items_torch` for up to
    ``_FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE`` decodes; the caller picks
    between them. Kept to one CTA because the write-first offsets come from an
    in-block ``tl.cumsum``. The block always spans the largest batch and the
    kernel specializes neither on the decode count nor on the alignment of
    ``state_indices``, so one compiled kernel serves every batch and offset.
    PNAT and the active buffer are read at ``replay_indices``, which default
    to ``state_indices``.
    """
    if replay_indices is None:
        replay_indices = state_indices
    num_decodes = state_indices.shape[0]
    assert num_decodes <= _FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE
    _prepare_gdn_replay_work_items_kernel[(1,)](
        state_indices,
        replay_indices,
        prev_num_accepted_tokens,
        cache_buf_idx,
        work_items,
        n_writes,
        num_decodes,
        prev_num_accepted_tokens.stride(0),
        cache_buf_idx.stride(0),
        replay_step_width=replay_step_width,
        replay_history_size=replay_history_size,
        work_item_width=REPLAY_WORK_ITEM_WIDTH,
        position_field=REPLAY_WORK_POSITION_IN_DECODE_BATCH,
        cache_slot_field=REPLAY_WORK_CACHE_SLOT,
        pnat_field=REPLAY_WORK_PNAT,
        cache_buf_idx_field=REPLAY_WORK_CACHE_BUF_IDX,
        replay_slot_field=REPLAY_WORK_REPLAY_SLOT,
        BLOCK_SIZE=_FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE,
        num_warps=4,
    )


@torch.compile(dynamic=True, fullgraph=True)
def _build_replay_work_items_torch(
    state_indices,
    prev_num_accepted_tokens,
    cache_buf_idx,
    work_items,
    n_writes,
    replay_step_width,
    replay_history_size,
    replay_indices=None,
):
    """Same partition as :func:`_build_replay_work_items_triton`, in ATen ops."""
    if replay_indices is None:
        replay_indices = state_indices
    num_decodes = state_indices.shape[0]
    position_in_decode_batch = torch.arange(
        num_decodes, dtype=torch.int32, device=state_indices.device
    )
    replay_slot_idx = replay_indices.to(torch.long)
    pnat = prev_num_accepted_tokens[replay_slot_idx].to(torch.int32)
    active_cache_buf_idx = cache_buf_idx[replay_slot_idx].to(torch.int32)

    writes = pnat + replay_step_width > replay_history_size
    writes_i32 = writes.to(torch.int32)
    write_offsets = torch.cumsum(writes_i32, dim=0) - writes_i32
    batch_n_writes = torch.sum(writes_i32, dim=0, keepdim=True).to(torch.int32)
    no_write_offsets = position_in_decode_batch - write_offsets
    output_offsets = torch.where(writes, write_offsets, batch_n_writes + no_write_offsets)
    output_offsets = output_offsets.to(torch.long)

    decode_work_items = work_items[:num_decodes]
    decode_work_items[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH].scatter_(
        0, output_offsets, position_in_decode_batch
    )
    decode_work_items[:, REPLAY_WORK_CACHE_SLOT].scatter_(0, output_offsets, state_indices)
    decode_work_items[:, REPLAY_WORK_PNAT].scatter_(0, output_offsets, pnat)
    decode_work_items[:, REPLAY_WORK_CACHE_BUF_IDX].scatter_(
        0, output_offsets, active_cache_buf_idx
    )
    decode_work_items[:, REPLAY_WORK_REPLAY_SLOT].scatter_(
        0, output_offsets, replay_indices.to(torch.int32)
    )
    n_writes.copy_(batch_n_writes)


class ReplayMamba2Metadata(Mamba2Metadata):
    """Mamba metadata plus each request's replay slot and the replay work items.

    ``replay_state_indices`` is a static buffer the manager's replay slots are
    copied into, or ``state_indices`` itself when the replay buffers share the
    recurrent-state slots. The replay work items order the generation rows
    write-first for the partitioned cached replay.
    """

    def __init__(
        self,
        max_batch_size: int,
        chunk_size: int,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self.replay_state_indices = self.state_indices
        self._replay_state_indices = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self._replay_state_indices_cpu = torch.zeros(
            max_batch_size, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._replay_state_indices_buffer = DeviceCopyBuffer(
            self._replay_state_indices, pin_memory=prefer_pinned()
        )
        self.replay_work_items = torch.zeros(
            max_batch_size, REPLAY_WORK_ITEM_WIDTH, dtype=torch.int32, device="cuda"
        )
        self.replay_n_writes = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.replay_num_decodes = 0

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        super().prepare(attn_metadata)
        batch_size = attn_metadata.seq_lens.shape[0]
        kv_cache_manager = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids
        if (
            kv_cache_manager is not None
            and hasattr(kv_cache_manager, "get_state_indices")
            and request_ids is not None
        ):
            self._prepare_replay_state_indices(kv_cache_manager, request_ids[:batch_size])
        else:
            self.replay_state_indices = self.state_indices
        self._prepare_replay_work_items(kv_cache_manager, batch_size, attn_metadata.num_contexts)

    def prepare_steady_gen_step(self, attn_metadata: AttentionMetadata) -> None:
        # The replay slots stay in place with the state slots; the work items follow the
        # accepted tokens of the previous step.
        super().prepare_steady_gen_step(attn_metadata)
        batch_size = attn_metadata.seq_lens.shape[0]
        self._prepare_replay_work_items(attn_metadata.kv_cache_manager, batch_size, 0)

    def _prepare_replay_state_indices(self, kv_cache_manager, batch_request_ids) -> None:
        """Fill the static replay-slot buffer, or alias the state slots."""
        if getattr(kv_cache_manager, "replay_layer_group_id", None) is None:
            self.replay_state_indices = self.state_indices
            return
        batch_size = len(batch_request_ids)
        indices = kv_cache_manager.get_replay_state_indices(batch_request_ids)
        assert len(indices) == batch_size, (
            f"get_replay_state_indices() returned {len(indices)} entries for "
            f"a batch of {batch_size} requests."
        )
        self._replay_state_indices_cpu[:batch_size].copy_(
            torch.as_tensor(indices, dtype=self._replay_state_indices_cpu.dtype)
        )
        self._replay_state_indices_buffer.update(self._replay_state_indices_cpu[:batch_size])
        self.replay_state_indices = self._replay_state_indices

    def _prepare_replay_work_items(self, kv_cache_manager, batch_size: int, num_contexts: int):
        self.replay_num_decodes = 0
        get_replay_metadata = getattr(kv_cache_manager, "get_replay_state_update_metadata", None)
        replay_metadata = get_replay_metadata() if get_replay_metadata is not None else None
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
            kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False
        )
        # The fused small-batch GDN kernel commits its checkpoint in-layer
        # and indexes cache metadata directly. Work items are only consumed
        # by the partitioned replay + all-layer commit path.
        if use_gdn_all_layer_commit and not kv_cache_manager.gdn_replay_partitioned(num_decodes):
            return

        prev_num_accepted_tokens = replay_metadata.prev_num_accepted_tokens
        cache_buf_idx = replay_metadata.cache_buf_idx
        replay_step_width = replay_metadata.replay_step_width
        replay_history_size = replay_metadata.replay_history_size

        # The kernel reads the accepted counts and active buffers through their
        # strides: a manager that keeps them in its KV cache pool hands out
        # strided views of the per-slot replay metadata rows.
        if num_decodes <= _FUSED_GDN_REPLAY_WORK_ITEMS_MAX_BATCH_SIZE:
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
            self.replay_state_indices[num_contexts:batch_size],
        )


__all__ = [
    "REPLAY_WORK_CACHE_BUF_IDX",
    "REPLAY_WORK_CACHE_SLOT",
    "REPLAY_WORK_ITEM_WIDTH",
    "REPLAY_WORK_PNAT",
    "REPLAY_WORK_POSITION_IN_DECODE_BATCH",
    "REPLAY_WORK_REPLAY_SLOT",
    "ReplayMamba2Metadata",
]
