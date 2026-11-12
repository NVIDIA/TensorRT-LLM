# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding (multi-query generation) state of the TRTLLM metadata."""

import functools
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SpecDecodingConfig:
    """Speculative-decoding settings fixed for the lifetime of a metadata.

    A metadata takes them at creation (``spec_decoding``) and sizes its
    draft-token buffers from them; every CUDA graph copy is created with the
    same settings.
    """

    #: Whether the attention runs in multi-query (spec-dec) mode; see
    #: ``SpeculativeDecodingMode.attention_need_spec_dec_mode``.
    attention_needs_spec_dec_mode: bool = False
    #: Whether the draft tokens form a tree, and whether it changes per step.
    is_spec_dec_tree: bool = False
    is_spec_dec_dynamic_tree: bool = False
    #: The number of draft layers and of draft tokens (tree nodes but the root).
    max_draft_len: int = 0
    max_total_draft_tokens: int = 0
    #: Per-request row width of the tree buffers (the tree manager's internal
    #: dimension); 0 for linear drafts.
    tree_buffer_dim: int = 0


def set_use_spec_decoding(metadata, use: bool) -> None:
    """Toggle spec-dec mode for this step on metadata whose backend has one."""
    spec = metadata.find_ext(SpecDecodingMetadata)
    if spec is not None:
        spec.use_spec_decoding = use


class SpecDecodingMetadata(MetadataExtension):
    """Spec-dec mode flags and the draft-token buffers the attention op reads.

    The buffers stay None until ``update`` enables spec-dec mode; CUDA graph
    copies share them with the metadata they were copied from.
    """

    # A draft binds the target's spec-dec settings and step flags. Its loop
    # rewrites the masks, position offsets and generation lengths in place,
    # so it keeps buffers of its own for them, copied from the target's on
    # device (not the buffers the target shares with its CUDA graph copies).
    adopts_target_state = True
    # Every draft takes it at its draft-loop start, also one that derives the
    # rest: it depends on neither the KV cache nor other extensions (the
    # engine sets it on the target with update_spec_dec_param).
    adopts_target_state_always = True
    draft_bound = frozenset(
        {
            "spec_decoding",
            "max_num_heads_per_kv",
            "is_spec_decoding_enabled",
            "use_spec_decoding",
            "is_spec_dec_tree",
            "is_spec_dec_dynamic_tree",
            "force_prepare_spec_dec_tree_mask",
            "max_total_draft_tokens",
        }
    )
    draft_written = frozenset(
        {
            "spec_decoding_query_len",
            "spec_decoding_position_offsets",
            "spec_decoding_packed_mask",
            "spec_decoding_generation_lengths",
            "spec_decoding_bl_tree_mask_offset",
            "spec_decoding_bl_tree_mask",
            "spec_bl_tree_first_sparse_mask_offset_kv",
        }
    )

    fields = frozenset(
        {
            "spec_decoding",
            "max_num_heads_per_kv",
            "spec_bl_tree_first_sparse_mask_offset_kv",
            "spec_decoding_bl_tree_mask",
            "spec_decoding_bl_tree_mask_offset",
            "force_prepare_spec_dec_tree_mask",
            "spec_decoding_generation_lengths",
            "is_spec_dec_dynamic_tree",
            "is_spec_dec_tree",
            "is_spec_decoding_enabled",
            "max_total_draft_tokens",
            "spec_decoding_packed_mask",
            "spec_decoding_position_offsets",
            "position_offsets_for_cpp",
            "spec_decoding_query_len",
            "use_spec_decoding",
        }
    )
    methods = frozenset(
        {
            "fill_generation_lengths",
            "spec_decoding_param_prepare_for_blackwell",
            "update",
            "set_spec_decoding_query_len",
            "update_blackwell_first_sparse_mask_offset",
        }
    )

    init_fields = {
        # Upper bound on query heads per KV head over the target model's layers
        # (unsharded); sizes the spec-dec tree mask.
        "max_num_heads_per_kv": 1,
        # The speculative-decoding settings this metadata was created with.
        "spec_decoding": SpecDecodingConfig(),
    }
    # The dynamic tree writes these each step on the metadata the engine
    # prepares; its CUDA graph copies read them through the shared tensors.
    shared = frozenset(
        {
            "spec_decoding_position_offsets",
            "spec_decoding_packed_mask",
            "spec_decoding_generation_lengths",
            "spec_decoding_bl_tree_mask",
            "spec_decoding_bl_tree_mask_offset",
            "spec_bl_tree_first_sparse_mask_offset_kv",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        # Spec-dec mode (multi-query mode) of the TRTLLM XQA kernels, supported
        # for the entire runtime.
        self.is_spec_decoding_enabled = False
        # Whether the attention layer runs in spec-dec mode at this step/layer.
        self.use_spec_decoding = False
        # Whether the spec-dec tree is a tree or a chain (linear tree).
        self.is_spec_dec_tree = False
        # A tree that never changes needs its mask computed only once.
        self.is_spec_dec_dynamic_tree = False
        self.force_prepare_spec_dec_tree_mask = False
        self.max_total_draft_tokens: Optional[int] = None
        self.spec_decoding_position_offsets: Optional[torch.Tensor] = self.shared_or(
            "spec_decoding_position_offsets", lambda: None
        )
        # Current query width of the compact prefix in a 1-D dynamic-tree
        # offsets buffer.
        self.spec_decoding_query_len = 0
        self.spec_decoding_packed_mask: Optional[torch.Tensor] = self.shared_or(
            "spec_decoding_packed_mask", lambda: None
        )
        self.spec_decoding_generation_lengths: Optional[torch.Tensor] = self.shared_or(
            "spec_decoding_generation_lengths", lambda: None
        )
        self.spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = self.shared_or(
            "spec_decoding_bl_tree_mask_offset", lambda: None
        )
        self.spec_decoding_bl_tree_mask: Optional[torch.Tensor] = self.shared_or(
            "spec_decoding_bl_tree_mask", lambda: None
        )
        self.spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = self.shared_or(
            "spec_bl_tree_first_sparse_mask_offset_kv", lambda: None
        )
        config = self.spec_decoding
        md = self.metadata
        trtllm_gen = is_trtllm_gen_sm(get_sm_version())
        # Blackwell trtllm-gen spec-dec is enabled only for dynamic-tree masks.
        self.is_spec_decoding_enabled = config.attention_needs_spec_dec_mode and (
            not trtllm_gen or config.is_spec_dec_dynamic_tree
        )
        self.is_spec_dec_tree = config.is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = config.is_spec_dec_dynamic_tree
        # A hybrid model's first executed attention layer can have a nonzero
        # cache-local index because recurrent layers precede it.  Do not rely
        # on the C++ ``layer_idx == 0`` fallback to rebuild the target mask:
        # the dynamic draft loop clears that mask before the next target step.
        self.force_prepare_spec_dec_tree_mask = config.is_spec_dec_dynamic_tree
        # Forward the tree length to FMHA kernel selection.
        if config.max_draft_len or config.max_total_draft_tokens:
            self.max_total_draft_tokens = config.max_total_draft_tokens
        if not self.is_spec_decoding_enabled:
            return

        # These buffers are accessed more like removing input padding,
        # rather than using max_total_draft_tokens + 1 as the offset between different requests.
        if config.is_spec_dec_tree and self.spec_decoding_position_offsets is None:
            # Dynamic tree: the tree manager's internal dimension may be
            # larger than max_total_draft_tokens+1 to accommodate
            # K*max_draft_len. 1D layout for flexible view() in the drafting loop.
            self.spec_decoding_position_offsets = torch.empty(
                (md.max_num_requests * config.tree_buffer_dim,),
                dtype=torch.int,
                device="cuda",
            )
        if config.is_spec_dec_tree and self.spec_decoding_packed_mask is None:
            buf_dim = config.tree_buffer_dim
            # Zero-init: dynamic-tree dst has inner dim
            # ceil(buf_dim/32) but only ceil((max_total_draft_tokens+1)/32)
            # is written each step. Unwritten cols would otherwise feed the
            # C++ XQA kernel as live mask bits.
            self.spec_decoding_packed_mask = torch.zeros(
                [md.max_num_requests, buf_dim, math.ceil(buf_dim / 32)],
                dtype=torch.int,
                device="cuda",
            )
        if self.spec_decoding_generation_lengths is None:
            self.spec_decoding_generation_lengths = torch.empty(
                [md.max_num_requests],
                dtype=torch.int,
                device="cuda",
            )
        if trtllm_gen:
            self.spec_decoding_param_prepare_for_blackwell()

    @property
    def position_offsets_for_cpp(self) -> Optional[torch.Tensor]:
        """``spec_decoding_position_offsets`` as the [requests, query width] view the attention op reads.

        A 1-D (dynamic-tree) buffer is viewed at its current compact query
        width on Hopper, which can be smaller than the buffer's capacity, and
        at its full row width on trtllm-gen.
        """
        offsets = self.spec_decoding_position_offsets
        if offsets is None or offsets.dim() != 1:
            return offsets
        max_num_requests = self.metadata.max_num_requests
        if is_trtllm_gen_sm(get_sm_version()):
            return offsets.view(max_num_requests, -1)
        query_len = self.spec_decoding_query_len
        if query_len <= 0:
            raise ValueError("1-D speculative position offsets require a positive query length.")
        return offsets[: max_num_requests * query_len].view(max_num_requests, query_len)

    def set_spec_decoding_query_len(self, query_len: int) -> None:
        """Set the query width of the compact prefix of the 1-D offsets buffer.

        A dynamic-tree draft loop narrows it between its draft forwards.
        """
        self.spec_decoding_query_len = query_len

    def spec_decoding_param_prepare_for_blackwell(self) -> None:
        """Prepare the Blackwell parameters of the generation-phase spec-dec kernels.

        Uses persistent buffers (only allocated if None) for CUDA graph
        compatibility.
        """
        md = self.metadata
        if self.spec_decoding_bl_tree_mask_offset is None:
            self.spec_decoding_bl_tree_mask_offset = torch.zeros(
                [md.max_num_requests],
                dtype=torch.int64,
                device="cuda",
            )

        if self.spec_bl_tree_first_sparse_mask_offset_kv is None:
            self.spec_bl_tree_first_sparse_mask_offset_kv = torch.zeros(
                [md.max_num_requests],
                dtype=torch.int32,
                device="cuda",
            )

        if self.spec_decoding_bl_tree_mask is None:
            seqLenQ = (
                self.spec_decoding_packed_mask.shape[1]
                if self.spec_decoding_packed_mask is not None
                else 1
            )
            mask_size = torch.ops.trtllm.spec_dec_tree_custom_mask_capacity(
                md.max_num_requests, seqLenQ, self.max_num_heads_per_kv
            )
            self.spec_decoding_bl_tree_mask = torch.zeros(
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
            self.spec_bl_tree_first_sparse_mask_offset_kv[:ng].copy_(gen_offset)

    def fill_generation_lengths(self, runtime_draft_len: int) -> None:
        self.spec_decoding_generation_lengths[: self.metadata.max_num_requests].fill_(
            runtime_draft_len + 1
        )

    def update(
        self,
        batch_size,
        spec_metadata: Optional["SpecMetadata"] = None,
        spec_tree_manager: Optional["SpecTreeManager"] = None,
        num_contexts: int = 0,
    ) -> None:
        """Write this step's draft-token layout into the spec-dec buffers.

        Args:
            batch_size: int, the number of requests in the batch.
            spec_metadata: Optional['SpecMetadata'] = None, the metadata of the spec-dec.
            spec_tree_manager: Optional['SpecTreeManager'] = None, the spec_tree_manager for draft token tree.
            num_contexts: int = 0, the number of context (prefill) requests in the
                batch.  The slot-storage layout is ``[ctx | gen]``; dynamic-tree
                metadata only describes the gen rows, so we must source from
                ``[num_contexts:batch_size]`` rather than ``[:batch_size]``.
        """
        md = self.metadata
        trtllm_gen = is_trtllm_gen_sm(get_sm_version())
        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled
        if not self.is_spec_decoding_enabled:
            return
        max_draft_len = self.spec_decoding.max_draft_len
        max_total_draft_tokens = self.spec_decoding.max_total_draft_tokens

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
                pos_dst = self.spec_decoding_position_offsets[: num_gens * n_dt].view(
                    num_gens, n_dt
                )
                pos_dst.copy_(pos_src, non_blocking=True)

                actual_mask_width = math.ceil(n_dt / 32)
                mask_src = torch.index_select(slot_storage.packed_mask, 0, slot_ids)[
                    :, :, :actual_mask_width
                ]
                if trtllm_gen:
                    # Blackwell reads the padded 3D mask layout.
                    self.spec_decoding_packed_mask[:num_gens, :n_dt, :actual_mask_width].copy_(
                        mask_src, non_blocking=True
                    )
                else:
                    # Hopper XQA reads a compact flat prefix.
                    total = num_gens * n_dt * actual_mask_width
                    self.spec_decoding_packed_mask.view(-1)[:total].copy_(
                        mask_src.reshape(-1), non_blocking=True
                    )

            self.spec_decoding_generation_lengths[:batch_size].fill_(n_dt)
            self.spec_decoding_query_len = n_dt

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
            self.spec_decoding_position_offsets = generate_spec_decoding_position_offsets(
                md.max_num_requests, runtime_draft_token_buffer_width
            )
            self.spec_decoding_packed_mask = generate_spec_decoding_packed_mask(
                md.max_num_requests, runtime_draft_token_buffer_width
            )
            self.spec_decoding_query_len = runtime_draft_token_buffer_width + 1
