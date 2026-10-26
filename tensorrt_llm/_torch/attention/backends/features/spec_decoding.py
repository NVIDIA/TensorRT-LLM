# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding (multi-query generation) state of the TRTLLM metadata."""

import functools
import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version

from ..extensions import MetadataExtension

if TYPE_CHECKING:
    from ....speculative.interface import SpecMetadata
    from ....speculative.spec_tree_manager import SpecTreeManager


def is_trtllm_gen_sm(sm: int) -> bool:
    """Whether the TRTLLM attention op runs the trtllm-gen kernels on ``sm``."""
    return not (sm < 100 or sm in [120, 121])


@functools.cache
def generate_spec_decoding_position_offsets(max_num_requests: int, draft_len: int) -> torch.Tensor:
    width = draft_len + 1
    row = torch.arange(width, dtype=torch.int, device="cuda")
    return row.unsqueeze(0).expand(max_num_requests, -1).contiguous()


@functools.cache
def generate_spec_decoding_packed_mask(max_num_requests: int, draft_len: int) -> torch.Tensor:
    width = draft_len + 1
    num_blocks = math.ceil(width / 32)
    mask = torch.zeros([max_num_requests, width, num_blocks], dtype=torch.int, device="cuda")
    remaining = width
    for blk in range(num_blocks):
        if remaining <= 0:
            break
        n = min(32, remaining)
        vals = (torch.pow(2, torch.arange(n) + 1) - 1).int()
        mask[:, blk * 32 : blk * 32 + n, blk] = vals
        remaining -= 32
    return mask


def set_use_spec_decoding(metadata, use: bool) -> None:
    """Toggle spec-dec mode for this step on metadata whose backend has one."""
    spec = metadata.find_ext(SpecDecodingMetadata)
    if spec is not None:
        spec.use_spec_decoding = use


def find_spec_decoding(metadata) -> Optional["SpecDecodingMetadata"]:
    """The spec-decoding extension of ``metadata``, or None.

    Accepts the duck-typed metadata views that may carry no extensions.
    """
    find_ext = getattr(metadata, "find_ext", None)
    return find_ext(SpecDecodingMetadata) if find_ext is not None else None


class SpecDecodingMetadata(MetadataExtension):
    """Spec-dec mode flags and the draft-token buffers the attention op reads.

    The buffers stay None until ``update`` enables spec-dec mode; CUDA graph
    copies share them with the metadata they were copied from.
    """

    def __init__(self, metadata) -> None:
        super().__init__(metadata)
        # Spec-dec mode (multi-query mode) of the TRTLLM XQA kernels, supported
        # for the entire runtime.
        self.is_spec_decoding_enabled = False
        # Whether the attention layer runs in spec-dec mode at this step/layer.
        self.use_spec_decoding = False
        # Whether the spec-dec tree is a tree or a chain (linear tree).
        self.is_spec_dec_tree = False
        # A tree that never changes needs its mask computed only once.
        self.is_spec_dec_dynamic_tree = False
        self.force_prepare_tree_mask = False
        self.max_total_draft_tokens: Optional[int] = None
        self.position_offsets: Optional[torch.Tensor] = None
        # The C++ attention op requires a 2-D position_offsets tensor and
        # reads sizes()[1] as the generation length / packed-mask row stride.
        self.position_offsets_cpp: Optional[torch.Tensor] = None
        # Compact Hopper C++ row stride for 1D dynamic-tree offsets.
        self.position_offsets_stride = 0
        self.packed_mask: Optional[torch.Tensor] = None
        self.generation_lengths: Optional[torch.Tensor] = None
        self.bl_tree_mask_offset: Optional[torch.Tensor] = None
        self.bl_tree_mask: Optional[torch.Tensor] = None
        self.bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None

    @property
    def position_offsets_for_cpp(self) -> Optional[torch.Tensor]:
        """``position_offsets`` reshaped to the 2D layout the C++ kernel expects."""
        offsets = self.position_offsets
        if offsets is not None and offsets.dim() == 1:
            if self.position_offsets_cpp is not None and not is_trtllm_gen_sm(get_sm_version()):
                return self.position_offsets_cpp
            return offsets.view(self.metadata.max_num_requests, -1)
        return offsets

    def update_position_offsets_for_cpp(self, query_len: int) -> None:
        """Refresh the C++ view of spec-dec position offsets."""
        offsets = self.position_offsets
        max_num_requests = self.metadata.max_num_requests
        if offsets is None or offsets.dim() != 1:
            self.position_offsets_cpp = offsets
            self.position_offsets_stride = 0
            return

        if max_num_requests > 0 and query_len > 0:
            self.position_offsets_stride = query_len
            total = max_num_requests * query_len
            self.position_offsets_cpp = offsets[:total].view(max_num_requests, query_len)
        else:
            self.position_offsets_cpp = offsets
            self.position_offsets_stride = 0

    def prepare_for_blackwell(self) -> None:
        """Prepare the Blackwell parameters of the generation-phase spec-dec kernels.

        Uses persistent buffers (only allocated if None) for CUDA graph
        compatibility.
        """
        md = self.metadata
        if self.bl_tree_mask_offset is None:
            self.bl_tree_mask_offset = torch.zeros(
                [md.max_num_requests],
                dtype=torch.int64,
                device="cuda",
            )

        if self.bl_tree_first_sparse_mask_offset_kv is None:
            self.bl_tree_first_sparse_mask_offset_kv = torch.zeros(
                [md.max_num_requests],
                dtype=torch.int32,
                device="cuda",
            )

        if self.bl_tree_mask is None:
            seqLenQ = self.packed_mask.shape[1] if self.packed_mask is not None else 1
            mask_size = torch.ops.trtllm.spec_dec_tree_custom_mask_capacity(
                md.max_num_requests, seqLenQ, md.max_num_heads_per_kv
            )
            self.bl_tree_mask = torch.zeros(
                mask_size,
                dtype=torch.uint32,
                device="cuda",
            )

    def update_blackwell_first_sparse_mask_offset(self) -> None:
        """Fill gen slots [0:ng) with (kv_lens_cuda - seq_lens); in-place for CUDA graph."""
        md = self.metadata
        nc = md.num_contexts
        n = md.num_seqs
        ng = n - nc
        if ng > 0:
            gen_offset = (md.kv_lens_cuda[nc:n] - md.seq_lens_cuda[nc:n]).to(torch.int32)
            self.bl_tree_first_sparse_mask_offset_kv[:ng].copy_(gen_offset)

    def fill_generation_lengths(self, runtime_draft_len: int) -> None:
        self.generation_lengths[: self.metadata.max_num_requests].fill_(runtime_draft_len + 1)

    def update(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        spec_metadata: Optional["SpecMetadata"] = None,
        spec_tree_manager: Optional["SpecTreeManager"] = None,
        num_contexts: int = 0,
    ) -> None:
        """
        Update the spec-dec parameters for the TRTLLM attention layer.
        Args:
            batch_size: int, the number of requests in the batch.
            is_spec_decoding_enabled: bool, whether the attention need to be spec_decoding mode,
                which is determined by attention_need_spec_dec_mode() function.
            is_spec_dec_tree: bool, whether the spec-dec mode is a tree, i.e., dynamic tree.
                For linear-tree, it is always False.
            is_spec_dec_dynamic_tree: bool, whether using dynamic tree.
            max_draft_len: int, the number of the draft layers.
            max_total_draft_tokens: int, the number of all nodes in the tree (except the root).
            spec_metadata: Optional['SpecMetadata'] = None, the metadata of the spec-dec.
            spec_tree_manager: Optional['SpecTreeManager'] = None, the spec_tree_manager for draft token tree.
            num_contexts: int = 0, the number of context (prefill) requests in the
                batch.  The slot-storage layout is ``[ctx | gen]``; dynamic-tree
                metadata only describes the gen rows, so we must source from
                ``[num_contexts:batch_size]`` rather than ``[:batch_size]``.
        """
        md = self.metadata
        trtllm_gen = is_trtllm_gen_sm(get_sm_version())

        # Blackwell trtllm-gen spec-dec is enabled only for dynamic-tree masks.
        self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
            not trtllm_gen or is_spec_dec_dynamic_tree
        )

        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled
        self.is_spec_dec_tree = is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = is_spec_dec_dynamic_tree
        # A hybrid model's first executed attention layer can have a nonzero
        # cache-local index because recurrent layers precede it.  Do not rely
        # on the C++ ``layer_idx == 0`` fallback to rebuild the target mask:
        # the dynamic draft loop clears that mask before the next target step.
        self.force_prepare_tree_mask = is_spec_dec_dynamic_tree
        # Forward the tree length to FMHA kernel selection.
        self.max_total_draft_tokens = max_total_draft_tokens

        if not self.is_spec_decoding_enabled:
            return

        # These buffers are accessed more like removing input padding,
        # rather than using max_total_draft_tokens + 1 as the offset between different requests.
        if is_spec_dec_tree and self.position_offsets is None:
            # Dynamic tree: use _internal_buf_dim which may be larger
            # than max_total_draft_tokens+1 to accommodate K*max_draft_len.
            # 1D layout for flexible view() in the drafting loop.
            buf_dim = spec_tree_manager._internal_buf_dim
            self.position_offsets = torch.empty(
                (md.max_num_requests * buf_dim,),
                dtype=torch.int,
                device="cuda",
            )
        if is_spec_dec_tree and self.packed_mask is None:
            buf_dim = spec_tree_manager._internal_buf_dim
            # Zero-init: dynamic-tree dst has inner dim
            # ceil(buf_dim/32) but only ceil((max_total_draft_tokens+1)/32)
            # is written each step. Unwritten cols would otherwise feed the
            # C++ XQA kernel as live mask bits.
            self.packed_mask = torch.zeros(
                [md.max_num_requests, buf_dim, math.ceil(buf_dim / 32)],
                dtype=torch.int,
                device="cuda",
            )

        if self.generation_lengths is None:
            self.generation_lengths = torch.empty(
                [md.max_num_requests],
                dtype=torch.int,
                device="cuda",
            )

        if trtllm_gen:
            self.prepare_for_blackwell()
        else:
            self.bl_tree_mask_offset = None
            self.bl_tree_mask = None
            self.bl_tree_first_sparse_mask_offset_kv = None

        cpp_query_len = 0

        # Case 1: dynamic tree — copy per-request params from spec_tree_manager.
        if self.is_spec_dec_dynamic_tree:
            assert spec_tree_manager is not None, "spec_tree_manager is required for dynamic tree"
            n_dt = spec_tree_manager.max_total_draft_tokens + 1
            buf_dim = spec_tree_manager._internal_buf_dim

            assert buf_dim >= n_dt, (
                f"Dynamic-tree copy requires buf_dim >= n_dt; got buf_dim={buf_dim}, n_dt={n_dt}."
            )

            slot_storage = spec_tree_manager.slot_storage

            num_gens = batch_size - num_contexts
            if num_gens > 0:
                slot_ids = slot_storage.all_ids_buf[num_contexts:batch_size]

                pos_src = torch.index_select(slot_storage.position_offsets, 0, slot_ids)[:, :n_dt]
                pos_dst = self.position_offsets[: num_gens * n_dt].view(num_gens, n_dt)
                pos_dst.copy_(pos_src, non_blocking=True)

                actual_mask_width = math.ceil(n_dt / 32)
                mask_src = torch.index_select(slot_storage.packed_mask, 0, slot_ids)[
                    :, :, :actual_mask_width
                ]
                if trtllm_gen:
                    # Blackwell reads the padded 3D mask layout.
                    self.packed_mask[:num_gens, :n_dt, :actual_mask_width].copy_(
                        mask_src, non_blocking=True
                    )
                else:
                    # Hopper XQA reads a compact flat prefix.
                    total = num_gens * n_dt * actual_mask_width
                    self.packed_mask.view(-1)[:total].copy_(mask_src.reshape(-1), non_blocking=True)

            self.generation_lengths[:batch_size].fill_(n_dt)
            cpp_query_len = n_dt

        # Case 2: linear tree
        else:
            # Currently dynamic draft length is only supported for linear tree
            # Dynamic draft length needs position offsets and packed mask to be shaped for each runtime draft length.
            # So we create cache for position offsets and packed mask for each draft length to avoid reallocation.
            assert max_draft_len == max_total_draft_tokens, (
                "max_draft_len should be equal to max_total_draft_tokens for linear tree"
            )
            # For algos other than PARD, this equals runtime_draft_len (K); for PARD it's 2K-1.
            runtime_draft_token_buffer_width = (
                spec_metadata.runtime_tokens_per_gen_step - 1
                if spec_metadata is not None
                else max_draft_len
            )
            self.fill_generation_lengths(runtime_draft_len=runtime_draft_token_buffer_width)
            self.position_offsets = generate_spec_decoding_position_offsets(
                md.max_num_requests, runtime_draft_token_buffer_width
            )
            self.packed_mask = generate_spec_decoding_packed_mask(
                md.max_num_requests, runtime_draft_token_buffer_width
            )

        self.update_position_offsets_for_cpp(cpp_query_len)
