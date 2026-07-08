# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import math
import os
import weakref
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from ..speculative.spec_tree_manager import SpecTreeManager

from tensorrt_llm._torch.attention_backend.fmha import (
    Fmha, get_enabled_fmha_lib_classes)
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory, prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..utils import (compute_swizzled_sf_shape, get_global_attrs,
                     get_model_extra_attrs)
from .interface import (AttentionBackend, AttentionForwardArgs,
                        AttentionInputType, AttentionMask, AttentionMetadata,
                        KVCacheParams, MLAParams, PositionalEmbeddingParams,
                        PredefinedAttentionMask, RopeParams,
                        merge_attention_forward_args)
from .sparse.params import SparseParams
from .sparse.skip_softmax import SkipSoftmaxParams


@functools.cache
def generate_spec_decoding_position_offsets(max_num_requests: int,
                                            draft_len: int) -> torch.Tensor:
    width = draft_len + 1
    row = torch.arange(width, dtype=torch.int, device='cuda')
    return row.unsqueeze(0).expand(max_num_requests, -1).contiguous()


@functools.cache
def generate_spec_decoding_packed_mask(max_num_requests: int,
                                       draft_len: int) -> torch.Tensor:
    width = draft_len + 1
    num_blocks = math.ceil(width / 32)
    mask = torch.zeros([max_num_requests, width, num_blocks],
                       dtype=torch.int,
                       device='cuda')
    remaining = width
    for blk in range(num_blocks):
        if remaining <= 0:
            break
        n = min(32, remaining)
        vals = (torch.pow(2, torch.arange(n) + 1) - 1).int()
        mask[:, blk * 32:blk * 32 + n, blk] = vals
        remaining -= 32
    return mask


@dataclass(kw_only=True)
class TrtllmAttentionMetadata(AttentionMetadata):
    workspace: Optional[torch.Tensor] = None
    cuda_graph_workspace: Optional[torch.Tensor] = None

    # TrtllmAttention needs to know the beam width to access to the cache indirection buffer,
    # when beam search is enabled.
    beam_width: int = 1

    @property
    def effective_beam_width(self) -> int:
        # Only use this for the fallback kernel's beam_width argument.
        # Cross-attention reads request-scoped encoder K/V that is written once
        # and reused unchanged by every decoder beam. Metadata preparation still
        # uses beam_width to expand cross block-offset rows to decoder-sequence
        # scope, but the fallback kernel should treat the cross K/V cache as
        # non-beam-packed.
        return 1 if self.is_cross else self.beam_width

    # TrtllmAttention needs to know the max sequence length.
    # Implemented as a property to support no cache mode.
    max_seq_len: Optional[int]

    # Storage for internal max_seq_len value
    _max_seq_len_storage: Optional[int] = field(default=None,
                                                init=True,
                                                repr=False)

    # Encoder CUDA graph compatibility: overrides host-side max_context_q_len
    # so FMHA kernel launch params are stable across graph capture/replay even
    # when actual per-batch sequence lengths vary. Only set by
    # EncoderCUDAGraphRunner; None elsewhere.
    max_context_q_len_override: Optional[int] = None

    # Flags to enable spec-dec mode (multi-query mode) in TRTLLM XQA Kernels
    # spec decoding mode can be enabled for non-TRTLLM-gen kernels (pre-Blackwell XQA kernels)
    # is_spec_decoding_enabled specifies if spec-dec mode is supported for the entire runtime.
    is_spec_decoding_enabled: bool = False
    # use_spec_decoding determines if the attention layer should be run in spec-dec mode at the specific step / layer.
    use_spec_decoding: bool = False

    # if spec-dec tree is a tree or a chain (linear tree)
    is_spec_dec_tree: bool = False
    # if spec-dec tree wouldn't be changed at all, the mask won't be computed every step.
    is_spec_dec_dynamic_tree: bool = False

    # parameters required for spec-dec mode
    max_total_draft_tokens: Optional[int] = None
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    # C++ attention op requires a 2-D position_offsets tensor and reads
    # sizes()[1] as the generation length / packed-mask row stride.
    spec_decoding_position_offsets_cpp: Optional[torch.Tensor] = None
    # Compact Hopper C++ row stride for 1D dynamic-tree offsets.
    position_offsets_stride: int = 0
    spec_decoding_packed_mask: Optional[torch.Tensor] = None
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None

    # TRTLLM-Gen FMHA JIT warmup controls.
    trtllm_gen_jit_warmup: bool = False

    # Flag to enable helix parallelism.
    enable_helix: bool = False

    # Global position ids of tokens for each sequence in the batch. Given
    # each helix rank has only a subset of tokens for a sequence, we compute
    # a global position id for each token here.
    helix_position_offsets: Optional[torch.Tensor] = None
    helix_position_offsets_cpu: Optional[torch.Tensor] = None

    # Whether the current rank is inactive for helix parallelism.
    # In helix parallelism, only the active rank appends KV cache for the query token
    # and attends to the previously cached tokens as well as the query token. Inactive
    # ranks do not append KV cache for the query token and attend to the previously
    # cached tokens only.
    helix_is_inactive_rank: Optional[torch.Tensor] = None
    helix_is_inactive_rank_cpu: Optional[torch.Tensor] = None

    # Block offsets for the target and draft KV caches
    kv_cache_block_offsets: Optional[torch.Tensor] = None
    host_kv_cache_block_offsets: Optional[torch.Tensor] = None
    draft_kv_cache_block_offsets: Optional[torch.Tensor] = None
    # Block IDs per sequence; populated in __post_init__ when a KV cache
    # manager is present. Declared here so encoder-only metadata (no KV cache)
    # still exposes the attribute.
    block_ids_per_seq: Optional[torch.Tensor] = None
    kv_block_ids_per_seq: Optional[torch.Tensor] = None

    # Pre-computed FlashMLA tile-scheduler metadata and num_splits.
    # Computed once per forward pass in TrtllmAttention.forward() and reused across layers.
    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None
    flash_mla_num_splits: Optional[torch.Tensor] = None
    _flash_mla_metadata_valid: bool = field(default=False,
                                            init=False,
                                            repr=False)

    use_paged_context_fmha: bool = field(init=False, default=False, repr=False)

    # `DSAtrtllmAttentionMetadata` overrides this; the dense path keeps 0.
    num_sparse_topk: int = 0

    @property
    def effective_workspace(self) -> Optional[torch.Tensor]:
        """Attention-kernel workspace, switching to the CUDA-graph copy under capture."""
        return self.cuda_graph_workspace if self.is_cuda_graph else self.workspace

    @property
    def spec_decoding_position_offsets_for_cpp(self) -> Optional[torch.Tensor]:
        """``spec_decoding_position_offsets`` reshaped to the 2D layout the C++
        kernel expects."""
        offsets = self.spec_decoding_position_offsets
        if offsets is not None and offsets.dim() == 1:
            if (self.spec_decoding_position_offsets_cpp is not None
                    and not self.is_sm_version_trtllm_gen_kernel(
                        sm=get_sm_version())):
                return self.spec_decoding_position_offsets_cpp
            return offsets.view(self.max_num_requests, -1)
        return offsets

    @property
    def max_context_length(self) -> int:
        """
        Upper bound for a single context window.
        Required max_seq_len for context-only attention cases like visual gen
        """
        return min(self.max_seq_len, self.max_num_tokens)

    @property
    def max_seq_len(self) -> int:
        """
        Returns the max sequence length.
        If the attention uses KV cache, it will return max_seq_len from the KV cache manager.
        If the attention is no cache, max_seq_len should be set manually by user.
        """
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.max_seq_len
        else:
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no kv cache attention"
            return self._max_seq_len_storage

    @max_seq_len.setter
    def max_seq_len(self, value: int) -> None:
        """
        Set the max sequence length for no cache attention.
        """
        self._max_seq_len_storage = value

    @property
    def tokens_per_block(self) -> Optional[int]:
        """
        Returns the number of tokens per block from the KV cache manager.
        """
        return self.kv_cache_manager.tokens_per_block if self.kv_cache_manager is not None else None

    @property
    def host_kv_cache_pool_pointers(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool pointers from the KV cache manager if KV cache manager is not None.
        """
        return self.kv_cache_manager.kv_cache_pool_pointers if self.kv_cache_manager is not None else None

    @property
    def host_kv_cache_pool_mapping(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool mapping from the KV cache manager if KV cache manager is not None.
        """
        return self.kv_cache_manager.kv_cache_pool_mapping if self.kv_cache_manager is not None else None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.enable_helix = self.mapping.has_cp_helix(
        ) if self.mapping is not None else False
        self.use_paged_context_fmha = (
            self.runtime_features.chunked_prefill
            or self.runtime_features.cache_reuse
            or self.runtime_features.has_speculative_draft_tokens
        ) if self.runtime_features is not None else False
        self._post_init_with_buffers(self.cuda_graph_buffers)

    def update_position_offsets_for_cpp(self, query_len: int) -> None:
        """Refresh the C++ view of spec-dec position offsets."""
        offsets = self.spec_decoding_position_offsets
        if offsets is None or offsets.dim() != 1:
            self.spec_decoding_position_offsets_cpp = offsets
            self.position_offsets_stride = 0
            return

        if self.max_num_requests > 0 and query_len > 0:
            self.position_offsets_stride = query_len
            total = self.max_num_requests * query_len
            self.spec_decoding_position_offsets_cpp = offsets[:total].view(
                self.max_num_requests, query_len)
        else:
            self.spec_decoding_position_offsets_cpp = offsets
            self.position_offsets_stride = 0

    def _post_init_with_buffers(self, buffers) -> None:

        # Set a default value, as max_num_sequences is not always set.
        if self.max_num_sequences is None:
            self.max_num_sequences = self.max_num_requests

        capture_graph = self.is_cuda_graph

        self.prompt_lens_cuda = self.get_empty(
            buffers,
            (self.max_num_sequences, ),
            cache_name="prompt_lens_cuda",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.kv_lens_cuda = self.get_empty_like(
            buffers,
            self.prompt_lens_cuda,
            cache_name="kv_lens_cuda",
            capture_graph=capture_graph,
        )
        self.kv_lens = torch.empty_like(self.kv_lens_cuda,
                                        device='cpu',
                                        pin_memory=prefer_pinned())
        self.host_total_kv_lens = torch.empty(2, device='cpu', dtype=torch.int)
        self.host_request_types = torch.empty_like(self.prompt_lens_cpu)

        if self.workspace is None:
            self.workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )

        if self.cuda_graph_workspace is None:
            self.cuda_graph_workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )

        if self.kv_cache_manager is not None:
            num_attention_op_pools = getattr(self.kv_cache_manager,
                                             "num_attention_op_pools",
                                             self.kv_cache_manager.num_pools)
            self.kv_cache_block_offsets = self.get_empty(
                buffers,
                [
                    num_attention_op_pools, self.max_num_sequences, 2,
                    self.kv_cache_manager.max_blocks_per_seq
                ],
                cache_name="kv_cache_block_offsets",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_kv_cache_block_offsets = self.kv_cache_manager.host_kv_cache_block_offsets
            self.block_ids_per_seq = None
            self.kv_block_ids_per_seq = None

            # Allocate separate block offset tensors for draft KV cache manager
            # Used in one-model speculative decoding with different KV cache layouts
            if self.draft_kv_cache_manager is not None:
                num_draft_attention_op_pools = getattr(
                    self.draft_kv_cache_manager, "num_attention_op_pools",
                    self.draft_kv_cache_manager.num_pools)
                self.draft_kv_cache_block_offsets = self.get_empty(
                    buffers,
                    [
                        num_draft_attention_op_pools, self.max_num_sequences, 2,
                        self.draft_kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="draft_kv_cache_block_offsets",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )

            if self.enable_flash_mla:
                self.block_ids_per_seq = self.get_empty(
                    buffers,
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="block_ids_per_seq",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
                self.kv_block_ids_per_seq = self.get_empty(
                    buffers,
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="kv_block_ids_per_seq",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
                # Allocate fixed-size buffers for pre-computed FlashMLA metadata.
                # These are pre-allocated so their GPU addresses are stable across CUDA graph captures.
                sm_count = torch.cuda.get_device_properties(
                    torch.cuda.current_device()).multi_processor_count
                self.flash_mla_tile_scheduler_metadata = self.get_empty(
                    buffers,
                    [sm_count * 8],  # TileSchedulerMetaDataSize = 8
                    cache_name="flash_mla_tile_scheduler_metadata",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
                self.flash_mla_num_splits = self.get_empty(
                    buffers,
                    [self.kv_cache_manager.max_batch_size + 1],
                    cache_name="flash_mla_num_splits",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
            if self.enable_context_mla_with_cached_kv:
                # for kv cache reuse/chunked context in MLA
                self.ctx_cached_token_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_cached_token_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_cached_token_indptr = torch.zeros_like(
                    self.ctx_cached_token_indptr,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.ctx_uncached_token_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_uncached_token_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_uncached_token_indptr = torch.zeros_like(
                    self.ctx_uncached_token_indptr,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                # context full seqlens include cached tokens and uncached tokens
                self.ctx_kv_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_kv_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_kv_indptr = torch.zeros_like(
                    self.ctx_kv_indptr,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )

        # Allocate static buffers for helix parallelism support.
        if self.enable_helix:
            self.helix_position_offsets = self.get_empty(
                buffers,
                (self.max_num_tokens, ),
                cache_name="helix_position_offsets",
                dtype=torch.int,
                capture_graph=capture_graph,
            )
            self.helix_position_offsets_cpu = torch.empty_like(
                self.helix_position_offsets,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
            self.helix_is_inactive_rank = self.get_empty(
                buffers,
                (self.max_num_sequences, ),
                cache_name="helix_is_inactive_rank",
                dtype=torch.bool,
                capture_graph=capture_graph,
            )
            self.helix_is_inactive_rank_cpu = torch.empty_like(
                self.helix_is_inactive_rank,
                device='cpu',
                pin_memory=prefer_pinned(),
            )

    def on_update_kv_lens(self):
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadata.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        if self.enable_flash_mla:
            self._flash_mla_metadata_valid = False

    def update_for_spec_dec(self) -> None:
        # MTP updates kv_lens_cuda in-place between sub-steps, which changes
        # cache_seq_lens seen by the C++ attention op.  Invalidate the metadata
        # so that forward() recomputes it for the next sub-step.
        if self.enable_flash_mla:
            self._flash_mla_metadata_valid = False

    def update_helix_param(
        self,
        helix_position_offsets: List[int],
        helix_is_inactive_rank: List[bool],
    ) -> None:
        """
        Update helix parameters by copying into static buffers for CUDA graph compatibility.

        Args:
            helix_position_offsets: Position offsets for helix parallelism with shape (num_tokens,).
            helix_is_inactive_rank: Whether the current rank is inactive with shape (batch_size,).
        """
        if helix_position_offsets is not None and self.helix_position_offsets is not None:
            num_tokens = len(helix_position_offsets)
            self.helix_position_offsets_cpu[:num_tokens].copy_(
                torch.tensor(helix_position_offsets, dtype=torch.int))
            self.helix_position_offsets[:num_tokens].copy_(
                self.helix_position_offsets_cpu[:num_tokens], non_blocking=True)

        if helix_is_inactive_rank is not None and self.helix_is_inactive_rank is not None:
            batch_size = len(helix_is_inactive_rank)
            self.helix_is_inactive_rank_cpu[:batch_size].copy_(
                torch.tensor(helix_is_inactive_rank, dtype=torch.bool))
            self.helix_is_inactive_rank[:batch_size].copy_(
                self.helix_is_inactive_rank_cpu[:batch_size], non_blocking=True)

    def _bind_runtime_views(
        self,
        *,
        kv_lens_cuda: torch.Tensor,
        kv_lens: torch.Tensor,
        prompt_lens_cuda: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        host_request_types: torch.Tensor,
    ) -> None:
        """Bind the per-forward ``*_runtime`` views the FMHA kernels read.

        Shared by ``prepare``, ``prepare_encoder_only`` and the encoder CUDA
        graph binding so the set of runtime views stays in sync across paths.
        """
        self.kv_lens_cuda_runtime = kv_lens_cuda
        self.kv_lens_runtime = kv_lens
        self.prompt_lens_cuda_runtime = prompt_lens_cuda
        self.prompt_lens_cpu_runtime = prompt_lens_cpu
        self.host_request_types_runtime = host_request_types

    def prepare(self) -> None:
        super().prepare()
        extra_attrs = get_model_extra_attrs()
        # If model extra attrs is set, attention_metadata is setup in executor.
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self)
        if self.kv_cache_manager is None:
            # Convert the attention metadata to a TRT-LLM no cache attention metadata.
            assert self.kv_cache_manager is None, "no cache attention should not have KV cache manager"
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no cache attention"

            # setting kv cache params
            self.kv_cache_params = KVCacheParams(use_cache=False, )

            # trtllm attn metadata prepare() requires this
            self.prompt_lens = self.context_lens

            self.kv_cache_block_offsets = None
            self.block_ids_per_seq = None

        prompt_lens = torch.as_tensor(
            self.prompt_lens,
            dtype=torch.int,
            device='cpu',
        )
        self.prompt_lens_cpu[:self.num_seqs].copy_(prompt_lens)
        self.prompt_lens_cuda[:self.num_seqs].copy_(
            self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        ) if self.kv_cache_params.use_cache else None

        if self.enable_flash_mla:
            self.prepare_flash_mla()

        # number of tokens needed in the kv cache for each sequence after the next pass.
        if self.enable_helix:
            # If helix is inactive, attend to the previously cached tokens only.
            assert cached_token_lens is not None, "cached_token_lens should be set for helix"
            active_rank = ~self.helix_is_inactive_rank_cpu[:self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + \
                self.seq_lens_kv if cached_token_lens is not None else self.seq_lens_kv

        # self.kv_lens is the valid kv cache length, while the self.kv_lens_cuda is
        # the sequence length including the cached tokens and the input tokens.
        self.kv_lens[:self.num_seqs].copy_(
            kv_lens + self.kv_cache_params.num_extra_kv_tokens)
        self.kv_lens_cuda[:self.num_seqs].copy_(maybe_pin_memory(
            kv_lens[:self.num_seqs]),
                                                non_blocking=True)
        # total kv lens for context requests and generation requests, without extra tokens
        self.host_total_kv_lens[0] = kv_lens[:self.num_contexts].sum().item()
        self.host_total_kv_lens[1] = kv_lens[self.num_contexts:self.
                                             num_seqs].sum().item()
        self.host_request_types[:self.num_contexts].fill_(0)
        self.host_request_types[self.num_contexts:self.num_seqs].fill_(1)

        # prepare for kv cache reuse/chunked context in MLA
        if self.enable_context_mla_with_cached_kv:
            self.prepare_context_mla_with_cached_kv(cached_token_lens, kv_lens)

        # kv block offsets
        assert self.request_ids is not None
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.copy_batch_block_offsets(
                self.kv_cache_block_offsets, self.request_ids, self.beam_width,
                self.num_contexts, self.num_seqs)

            error_message = (
                f"The max KV cache length of input sequences ({self.kv_lens[:self.num_seqs].max()}) "
                f"exceeds the KV cache manager's maximum supported length "
                f"({self.kv_cache_manager.max_seq_len}).")

            assert self.kv_lens[:self.num_seqs].max(
            ) <= self.kv_cache_manager.max_seq_len, error_message

            # Also prepare draft KV cache block offsets if draft_kv_cache_manager exists
            if self.draft_kv_cache_manager is not None:
                # Use the wrapper method which works for both V1 and V2
                self.draft_kv_cache_manager.copy_batch_block_offsets(
                    self.draft_kv_cache_block_offsets, self.request_ids,
                    self.beam_width, self.num_contexts, self.num_seqs)

        # Don't pass self.kv_lens as kv_lens here because it includes extra
        # tokens. Use the actual KV length (without extra tokens) for
        # kv_lens_runtime, which becomes host_past_key_value_lengths and
        # eventually mMaxSeqLenKv.
        self._bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:self.num_seqs],
            kv_lens=kv_lens[:self.num_seqs],
            prompt_lens_cuda=self.prompt_lens_cuda[:self.num_seqs],
            prompt_lens_cpu=self.prompt_lens_cpu[:self.num_seqs],
            host_request_types=self.host_request_types[:self.num_seqs],
        )

    def prepare_encoder_only(self) -> None:
        """Fast path for encoder-only forward (eager + CUDA graph capture)."""
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self)

        # Encoder batches run without a KV cache. The block-offset / block-id
        # attributes default to None (declared as dataclass fields).
        self.kv_cache_params = KVCacheParams(use_cache=False)

        # For encoder batches every request is a context request, so total
        # kv-tokens equals total q-tokens.
        self.host_total_kv_lens[0] = self._num_tokens

        # `host_request_types` is allocated with `torch.empty_like` (not
        # zeroed), and the full `prepare()` is what normally fills it. This
        # fast path skips `prepare()`, so explicitly mark every request as a
        # context request (type 0) here -- otherwise the FMHA reads whatever
        # garbage the buffer happened to hold and may treat a context segment
        # as a generation step.
        n = self.num_seqs
        self.host_request_types[:n].fill_(0)

        # Graph metadata binds these views once per key; eager refreshes them
        # because batch shape can vary between calls.
        if not self.is_cuda_graph:
            self._bind_runtime_views(
                kv_lens_cuda=self._seq_lens_cuda[:n],
                kv_lens=self._seq_lens[:n],
                prompt_lens_cuda=self._seq_lens_cuda[:n],
                prompt_lens_cpu=self._seq_lens[:n],
                host_request_types=self.host_request_types[:n],
            )

    def bind_encoder_cuda_graph_seq_lens(self, seq_lens_host: torch.Tensor,
                                         padded_batch_size: int) -> None:
        """Bind stable seq_lens storage for one encoder CUDA graph key."""
        self._seq_lens = seq_lens_host[:padded_batch_size]
        self._num_contexts = padded_batch_size
        self._num_generations = 0

        self.host_request_types[:padded_batch_size].fill_(0)
        self._bind_runtime_views(
            kv_lens_cuda=self._seq_lens_cuda[:padded_batch_size],
            kv_lens=self._seq_lens,
            prompt_lens_cuda=self._seq_lens_cuda[:padded_batch_size],
            prompt_lens_cpu=self._seq_lens,
            host_request_types=self.host_request_types[:padded_batch_size],
        )
        self.host_total_kv_lens[1] = 0

    def prepare_encoder_cuda_graph_replay(self, seq_lens: List[int],
                                          padded_num_tokens: int) -> None:
        """Update per-replay encoder CUDA graph metadata in-place."""
        self._seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int))
        self._num_tokens = padded_num_tokens
        self._num_ctx_tokens = padded_num_tokens
        self.host_total_kv_lens[0] = padded_num_tokens

    def prepare_flash_mla(self) -> None:
        # Invalidate the pre-computed metadata so that forward() recomputes it
        # for this forward pass before the first attention layer runs.
        self._flash_mla_metadata_valid = False
        block_ids_per_seq = maybe_pin_memory(
            self.kv_cache_manager.get_block_ids_per_seq(self.request_ids))
        num_blocks = block_ids_per_seq.shape[1]
        self.kv_block_ids_per_seq.fill_(0)
        self.kv_block_ids_per_seq[:self.num_seqs, :num_blocks].copy_(
            block_ids_per_seq, non_blocking=True)
        self.block_ids_per_seq.fill_(0)
        self.block_ids_per_seq[:self.num_generations, :num_blocks].copy_(
            block_ids_per_seq[self.num_contexts:], non_blocking=True)

    def pre_process_for_chunked_prefill(
        self,
        chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.
        Tensor,  # [chunked_loop_num + 1, num_contexts]
        cu_chunked_seq_len: torch.Tensor,
        merge_op_tensor: torch.Tensor,
        max_chunk_len_per_loop: list[int],
        chunked_loop_num: int,
    ) -> None:
        """
        Pre-process the MLA layer for chunked prefill.
        This method is called before the forward pass to prepare the MLA layer for chunked prefill.
        """
        num_contexts = self.num_contexts
        chunk_size = self.runtime_features.chunk_size
        chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
        total_chunk_size = chunk_size * chunk_batch_size
        remain_buffer_len = total_chunk_size
        current_batch_idx = 0
        max_chunk_len_per_loop.clear()
        max_chunk_len = 0
        # cal chunked_seq_len
        for batch_idx in range(num_contexts):
            cached_kv_len = self.kv_cache_params.num_cached_tokens_per_seq[
                batch_idx]
            while cached_kv_len > 0:
                used_buffer_len = min(remain_buffer_len, cached_kv_len)
                chunked_seq_len[current_batch_idx, batch_idx] = used_buffer_len
                max_chunk_len = max(max_chunk_len, used_buffer_len)
                remain_buffer_len -= used_buffer_len
                cached_kv_len -= used_buffer_len
                chunked_global_offset[
                    current_batch_idx + 1, batch_idx] = chunked_global_offset[
                        current_batch_idx,
                        batch_idx] + chunked_seq_len[current_batch_idx,
                                                     batch_idx]
                if remain_buffer_len == 0:
                    current_batch_idx += 1
                    remain_buffer_len = total_chunk_size
                    max_chunk_len_per_loop.append(max_chunk_len)
                    max_chunk_len = 0
        if len(max_chunk_len_per_loop) < chunked_loop_num:
            max_chunk_len_per_loop.append(max_chunk_len)
        assert len(
            max_chunk_len_per_loop
        ) == chunked_loop_num, f"max_chunk_len_per_loop size {len(max_chunk_len_per_loop)} != chunked_loop_num {chunked_loop_num}"
        for loop_idx in range(chunked_loop_num):
            cu_chunked_seq_len[loop_idx, 0] = 0
            torch.cumsum(chunked_seq_len[loop_idx, :num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=cu_chunked_seq_len[loop_idx, 1:num_contexts + 1])
            for s in range(num_contexts):
                if chunked_seq_len[loop_idx, s] > 0 and (
                        loop_idx == 0 or chunked_seq_len[loop_idx - 1, s] == 0):
                    merge_op_tensor[loop_idx, s] = 2  # copy only
                elif chunked_seq_len[loop_idx, s] > 0:
                    merge_op_tensor[loop_idx, s] = 1  # merge
                else:
                    merge_op_tensor[loop_idx, s] = 0  # skip

        # set merge op for last attn
        for s in range(num_contexts):
            if self.kv_cache_params.num_cached_tokens_per_seq[s] == 0:
                merge_op_tensor[chunked_loop_num, s] = 2  # copy only
            else:
                merge_op_tensor[chunked_loop_num, s] = 1  # merge

    def prepare_context_mla_with_cached_kv(self,
                                           cached_token_lens: torch.Tensor,
                                           kv_lens: torch.Tensor) -> None:
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_cached_token_len = cached_token_lens[:self.
                                                              num_contexts].max(
                                                              ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # determine the number of loop
            # currently we assume that the chunk size is the same as the max_num_tokens
            if self.runtime_features.chunked_prefill:
                chunk_size = self.runtime_features.chunk_size
                chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
                total_chunk_size = chunk_size * chunk_batch_size
                self.chunked_loop_num = math.ceil(self.num_ctx_cached_tokens /
                                                  total_chunk_size)
                self.chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_seqs),
                    dtype=torch.int,
                    device='cuda',
                )
                self.host_chunked_seq_len = torch.zeros_like(
                    self.chunked_seq_len,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.cu_chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_contexts + 1),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_cu_chunked_seq_len = torch.zeros_like(
                    self.cu_chunked_seq_len,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.chunked_global_offset = torch.zeros(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_chunked_global_offset = torch.zeros_like(
                    self.chunked_global_offset,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.max_chunk_len_per_loop = []
                # For last chunk we use the uncached kv
                self.merge_op_tensor = torch.empty(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_merge_op_tensor = torch.empty_like(
                    self.merge_op_tensor,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )

                self.pre_process_for_chunked_prefill(
                    chunked_seq_len=self.host_chunked_seq_len,
                    chunked_global_offset=self.host_chunked_global_offset,
                    cu_chunked_seq_len=self.host_cu_chunked_seq_len,
                    merge_op_tensor=self.host_merge_op_tensor,
                    max_chunk_len_per_loop=self.max_chunk_len_per_loop,
                    chunked_loop_num=self.chunked_loop_num)
                self.chunked_seq_len.copy_(self.host_chunked_seq_len,
                                           non_blocking=True)
                self.cu_chunked_seq_len.copy_(self.host_cu_chunked_seq_len,
                                              non_blocking=True)
                self.merge_op_tensor.copy_(self.host_merge_op_tensor,
                                           non_blocking=True)
                self.chunked_global_offset.copy_(
                    self.host_chunked_global_offset, non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_cached_token_len = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0
        torch.cumsum(cached_token_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_cached_token_indptr[1:self.num_contexts +
                                                           1])
        self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(
            self.seq_lens[:self.num_contexts],
            dim=0,
            dtype=torch.int64,
            out=self.host_ctx_uncached_token_indptr[1:self.num_contexts + 1])
        self.ctx_uncached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_uncached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(kv_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
        self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_kv_indptr[:self.num_contexts + 1], non_blocking=True)

    def compute_max_num_custom_mask_tiles_kv_upper_bound(
            self, max_seq_len_kv, min_first_sparse_mask_offset_kv,
            tile_size_kv_per_cta) -> int:
        """
        Compute the conservative upper bound of numCustomMaskTilesKv.

        Args:
            max_seq_len_kv (int): The maximum seqLenKv in the batch
            min_first_sparse_mask_offset_kv (int): The minimum firstSparseMaskOffsetKv in the batch
            tile_size_kv_per_cta (int): tileSizeKvPerCta value
        """
        first_sparse_tile_offset = min_first_sparse_mask_offset_kv // tile_size_kv_per_cta
        num_tiles_kv_total = math.ceil(max_seq_len_kv / tile_size_kv_per_cta)
        max_num_custom_mask_tiles_kv = num_tiles_kv_total - first_sparse_tile_offset
        return max_num_custom_mask_tiles_kv

    def spec_decoding_param_prepare_for_blackwell(self) -> None:
        """
        Prepare the blackwell parameters for the speculative decoding (Medusa and Eagle) generation-phase attention kernels.
        Uses persistent buffers (only allocate if None) for CUDA graph compatibility.
        """
        if self.spec_decoding_bl_tree_mask_offset is None:
            self.spec_decoding_bl_tree_mask_offset = torch.zeros(
                [self.max_num_requests],
                dtype=torch.int64,
                device='cuda',
            )

        if self.spec_bl_tree_first_sparse_mask_offset_kv is None:
            self.spec_bl_tree_first_sparse_mask_offset_kv = torch.zeros(
                [self.max_num_requests],
                dtype=torch.int32,
                device='cuda',
            )

        if self.spec_decoding_bl_tree_mask is None:
            # Custom mask covers the tree region (seqLenQ x seqLenQ).
            # The mask buffer needs extra room for tile boundary crossing:
            # when firstSparse is not aligned to stepKv (=tileSizeKv*numInstsKv),
            # the mask can span one extra KV tile. Adding (stepKv - 1) to
            # max_kv_len guarantees the upper-bound tile count is correct.
            seqLenQ = self.spec_decoding_packed_mask.shape[
                1] if self.spec_decoding_packed_mask is not None else 1
            tile_size_kv = 128
            tile_size_q = 128
            num_instances_q = 1
            num_instances_kv = 2
            tile_size_kv_per_cta = tile_size_kv * num_instances_kv
            max_kv_len = seqLenQ + tile_size_kv_per_cta - 1
            tile_size_q_per_cta = tile_size_q * num_instances_q
            max_num_custom_mask_tiles_kv = self.compute_max_num_custom_mask_tiles_kv_upper_bound(
                max_kv_len, 0, tile_size_kv_per_cta)
            max_num_tiles_q = math.ceil(
                (seqLenQ * self.num_heads_per_kv) / tile_size_q_per_cta)
            mask_size = int(self.max_num_requests * max_num_tiles_q *
                            max_num_custom_mask_tiles_kv * num_instances_q *
                            num_instances_kv * tile_size_q * tile_size_kv / 32)
            self.spec_decoding_bl_tree_mask = torch.zeros(
                mask_size,
                dtype=torch.uint32,
                device='cuda',
            )

    def update_blackwell_first_sparse_mask_offset(self) -> None:
        """Fill gen slots [0:ng) with (kv_lens_cuda - seq_lens); in-place for CUDA graph."""
        nc = self.num_contexts
        n = self.num_seqs
        ng = n - nc
        if ng > 0:
            gen_offset = (self.kv_lens_cuda[nc:n] -
                          self._seq_lens_cuda[nc:n]).to(torch.int32)
            self.spec_bl_tree_first_sparse_mask_offset_kv[:ng].copy_(gen_offset)

    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        model_is_wrapped: bool = False,
        spec_metadata: Optional['SpecMetadata'] = None,
        spec_tree_manager: Optional['SpecTreeManager'] = None,
        num_contexts: int = 0,
    ) -> None:
        '''
        Update the spec-dec parameters for the TRTLLM attention layer.
        Args:
            batch_size: int, the number of requests in the batch.
            is_spec_decoding_enabled: bool, whether the attention need to be spec_decoding mode, which is determined by attention_need_spec_dec_mode() function.
            is_spec_dec_tree: bool, whether the spec-dec mode is a tree, i.e., static tree or dynamic tree. For linear-tree, it is always False.
            is_spec_dec_dynamic_tree: bool, whether using dynamic tree.
            max_draft_len: int, the number of the draft layers.
            max_total_draft_tokens: int, the number of all nodes in the tree (except the root).
            model_is_wrapped: Optional[bool] = False, whether the drafter model is wrapped (i.e, CDL).
            spec_metadata: Optional['SpecMetadata'] = None, the metadata of the spec-dec.
            spec_tree_manager: Optional['SpecTreeManager'] = None, the spec_tree_manager for draft token tree.
            num_contexts: int = 0, the number of context (prefill) requests in the
                batch.  The slot-storage layout is ``[ctx | gen]``; dynamic-tree
                metadata only describes the gen rows, so we must source from
                ``[num_contexts:batch_size]`` rather than ``[:batch_size]``.
        '''

        # Blackwell trtllm-gen spec-dec is enabled only for dynamic-tree masks.
        self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
            not self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version())
            or is_spec_dec_dynamic_tree)

        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled
        self.is_spec_dec_tree = is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = is_spec_dec_dynamic_tree
        # Forward static tree length to FMHA kernel selection.
        self.max_total_draft_tokens = max_total_draft_tokens

        # Parameters can be fixed and not changed during runtime if the
        if self.is_spec_decoding_enabled:
            # Skip pre-allocating position_offsets and packed_mask when dynamic draft length is enabled.
            # We will use per-draft-len cached tensors for position_offsets and packed_mask instead.
            # Currently dynamic draft length is only supported for linear tree (not is_spec_dec_tree).

            # These buffers are accessed more like removing input padding,
            # rather than using max_total_draft_tokens + 1 as the offset between different requests.
            if is_spec_dec_tree and self.spec_decoding_position_offsets is None:
                if spec_tree_manager is not None and spec_tree_manager.use_dynamic_tree:
                    # Dynamic tree: use _internal_buf_dim which may be larger
                    # than max_total_draft_tokens+1 to accommodate K*max_draft_len
                    buf_dim = spec_tree_manager._internal_buf_dim
                    # Dynamic tree: 1D layout for flexible view() in drafting loop
                    self.spec_decoding_position_offsets = torch.empty(
                        (self.max_num_requests * buf_dim, ),
                        dtype=torch.int,
                        device='cuda',
                    )
                else:
                    # Static tree: keep 2D layout
                    self.spec_decoding_position_offsets = torch.empty(
                        [self.max_num_requests, max_total_draft_tokens + 1],
                        dtype=torch.int,
                        device='cuda',
                    )
            if is_spec_dec_tree and self.spec_decoding_packed_mask is None:
                if spec_tree_manager is not None and spec_tree_manager.use_dynamic_tree:
                    buf_dim = spec_tree_manager._internal_buf_dim
                else:
                    buf_dim = max_total_draft_tokens + 1
                # Zero-init: dynamic-tree dst has inner dim
                # ceil(buf_dim/32) but only ceil((max_total_draft_tokens+1)/32)
                # is written each step. Unwritten cols would otherwise feed the
                # C++ XQA kernel as live mask bits.
                self.spec_decoding_packed_mask = torch.zeros(
                    [self.max_num_requests, buf_dim,
                     math.ceil(buf_dim / 32)],
                    dtype=torch.int,
                    device='cuda',
                )

            if self.spec_decoding_generation_lengths is None:
                self.spec_decoding_generation_lengths = torch.empty(
                    [self.max_num_requests],
                    dtype=torch.int,
                    device='cuda',
                )

            if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
                self.spec_decoding_param_prepare_for_blackwell()
            else:
                self.spec_decoding_bl_tree_mask_offset = None
                self.spec_decoding_bl_tree_mask = None
                self.spec_bl_tree_first_sparse_mask_offset_kv = None

            cpp_query_len = 0

            # Case 1: dynamic tree — copy per-request params from spec_tree_manager.
            if self.is_spec_dec_dynamic_tree:
                assert spec_tree_manager is not None, "spec_tree_manager is required for dynamic tree"
                n_dt = spec_tree_manager.max_total_draft_tokens + 1
                buf_dim = spec_tree_manager._internal_buf_dim

                assert buf_dim >= n_dt, (
                    f"Dynamic-tree copy requires buf_dim >= n_dt; got "
                    f"buf_dim={buf_dim}, n_dt={n_dt}.")

                slot_storage = spec_tree_manager.slot_storage

                num_gens = batch_size - num_contexts
                if num_gens > 0:
                    slot_ids = slot_storage.all_ids_buf[num_contexts:batch_size]

                    pos_src = torch.index_select(slot_storage.position_offsets,
                                                 0, slot_ids)[:, :n_dt]
                    pos_dst = self.spec_decoding_position_offsets[:num_gens *
                                                                  n_dt].view(
                                                                      num_gens,
                                                                      n_dt)
                    pos_dst.copy_(pos_src, non_blocking=True)

                    actual_mask_width = math.ceil(n_dt / 32)
                    mask_src = torch.index_select(
                        slot_storage.packed_mask, 0,
                        slot_ids)[:, :, :actual_mask_width]
                    if self.is_sm_version_trtllm_gen_kernel(
                            sm=get_sm_version()):
                        # Blackwell reads the padded 3D mask layout.
                        self.spec_decoding_packed_mask[:num_gens, :n_dt, :
                                                       actual_mask_width].copy_(
                                                           mask_src,
                                                           non_blocking=True)
                    else:
                        # Hopper XQA reads a compact flat prefix.
                        total = num_gens * n_dt * actual_mask_width
                        self.spec_decoding_packed_mask.view(-1)[:total].copy_(
                            mask_src.reshape(-1), non_blocking=True)

                self.spec_decoding_generation_lengths[:batch_size].fill_(n_dt)
                cpp_query_len = n_dt

            # Case 2/3: static tree
            elif self.is_spec_dec_tree and not self.is_spec_dec_dynamic_tree and spec_metadata is not None:
                assert spec_metadata.spec_dec_mode.is_eagle3(
                ), "Tree decoding is only supported for Eagle3 now"

                is_target_model = not getattr(spec_metadata, 'is_draft_model',
                                              False)

                # Case 2: static tree and target model
                if is_target_model:
                    # For the target model, we update the spec-dec parameters with the spec_tree_manager, which is prepared in advance.
                    self.spec_decoding_position_offsets[:batch_size, :].copy_(
                        spec_tree_manager.spec_dec_position_offsets[0, :],
                        non_blocking=True)
                    self.spec_decoding_packed_mask[:batch_size, :, :].copy_(
                        spec_tree_manager.spec_dec_packed_mask[0, :, :],
                        non_blocking=True)
                    self.spec_decoding_generation_lengths[:batch_size].fill_(
                        spec_tree_manager.max_total_draft_tokens + 1)

                # Case 3: static tree and the first drafter layer
                else:
                    assert model_is_wrapped == True, "The drafter model should be wrapped"
                    # The first drafter layer will take the padded tokens as input (padding to the max_draft_len + 1)
                    # But the spec-dec parameters are still in the shape of max_total_draft_tokens + 1.
                    # Considering that these spec-dec params are accessed consecutively (without padding) in the attention Op,
                    # we need to write them consecutively when setting them.
                    # For the next drafter layers, we will prepare these spec-dec params in the drafting loops.

                    # position_offsets
                    position_offset = torch.arange(
                        max_draft_len + 1,
                        dtype=torch.int,
                        device='cpu',
                        pin_memory=prefer_pinned()).repeat(batch_size)
                    self.spec_decoding_position_offsets.reshape(
                        -1)[:(max_draft_len + 1) * batch_size].copy_(
                            position_offset, non_blocking=True)
                    # packed_mask
                    dummy_idx = torch.arange(max_draft_len + 1)
                    spec_decoding_packed_mask = torch.pow(
                        2, dummy_idx + 1) - 1  # [max_draft_len + 1]
                    spec_decoding_packed_mask = spec_decoding_packed_mask.repeat(
                        batch_size)  # [batch_size * (max_draft_len + 1)]
                    self.spec_decoding_packed_mask.reshape(
                        -1)[:(max_draft_len + 1) * batch_size].copy_(
                            spec_decoding_packed_mask, non_blocking=True)
                    self.generate_spec_decoding_generation_length(
                        runtime_draft_len=max_draft_len)
                    if (self.spec_decoding_position_offsets is not None
                            and self.spec_decoding_position_offsets.dim() == 1):
                        cpp_query_len = max_draft_len + 1

            # Case 4: linear tree
            else:
                # Currently dynamic draft length is only supported for linear tree
                # Dynamic draft length needs position offsets and packed mask to be shaped for each runtime draft length.
                # So we create cache for position offsets and packed mask for each draft length to avoid reallocation.
                assert max_draft_len == max_total_draft_tokens, "max_draft_len should be equal to max_total_draft_tokens for linear tree"
                # For algos other than PARD, this equals runtime_draft_len (K); for PARD it's 2K-1.
                runtime_draft_token_buffer_width = (
                    spec_metadata.runtime_tokens_per_gen_step -
                    1 if spec_metadata is not None else max_draft_len)
                self.generate_spec_decoding_generation_length(
                    runtime_draft_len=runtime_draft_token_buffer_width)
                self.spec_decoding_position_offsets = generate_spec_decoding_position_offsets(
                    self.max_num_requests, runtime_draft_token_buffer_width)
                self.spec_decoding_packed_mask = generate_spec_decoding_packed_mask(
                    self.max_num_requests, runtime_draft_token_buffer_width)

            self.update_position_offsets_for_cpp(cpp_query_len)

    def generate_spec_decoding_generation_length(self, runtime_draft_len):
        self.spec_decoding_generation_lengths[:self.max_num_requests].fill_(
            runtime_draft_len + 1)

    def is_sm_version_trtllm_gen_kernel(self, sm):
        return not (sm < 100 or sm in [120, 121])


class TrtllmAttention(AttentionBackend[TrtllmAttentionMetadata]):

    Metadata = TrtllmAttentionMetadata

    @staticmethod
    def is_sm_version_trtllm_gen_kernel(sm):
        return not (sm < 100 or sm in [120, 121])

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[SparseParams] = None,
        **kwargs,
    ):
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
            q_scaling (float): Scaling factor for QK. Defaults to 1.0 if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
                                                         If None, positional embedding should be applied by the model before calling the backend.
                                                         Otherwise, the backend is in-charge of applying positional embedding and may cache K without embedding it first.
            mla_params (MLAParams): Optional parameters for MLA. If None, MLA is not enabled.
        """
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)
        self.sparse_params = sparse_params

        self.is_mla_enable = mla_params is not None
        self.mla_params = mla_params or MLAParams()
        self.v_head_dim = self.mla_params.v_head_dim if self.is_mla_enable else head_dim

        rope_params = None
        if pos_embd_params is not None:
            rope_params = pos_embd_params.rope
        self.rope_params = rope_params or RopeParams()

        self.q_scaling = q_scaling or 1.0
        self.predicted_tokens_per_seq = self.mla_params.predicted_tokens_per_seq
        self.attention_chunk_size = attention_chunk_size

        if self.is_mla_enable:
            self.q_lora_rank = self.mla_params.q_lora_rank
            self.kv_lora_rank = self.mla_params.kv_lora_rank
            self.qk_nope_head_dim = self.mla_params.qk_nope_head_dim
            self.qk_rope_head_dim = self.mla_params.qk_rope_head_dim
            self.rope_append = self.mla_params.rope_append
        else:
            self.q_lora_rank = None
            self.kv_lora_rank = None
            self.qk_nope_head_dim = None
            self.qk_rope_head_dim = None
            self.rope_append = None

        self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params(
        )
        self.position_embedding_type = int(
            pos_embd_params.type) if pos_embd_params is not None else 0
        self.skip_softmax_stat = torch.zeros(2,
                                             dtype=torch.uint32,
                                             device='cuda')
        self.print_skip_softmax_stat = os.environ.get(
            "TRTLLM_PRINT_SKIP_SOFTMAX_STAT", "0") == "1"

        # Layer-level fp8 KV-cache scales. Stay at 1.0 (no-op) for the
        # PyTorch backend, which never overrides them. They guarantee the
        # kernel always receives a valid pointer, since several non-MLA
        # XQA kernels (cpp/kernels/xqa/mha.cu, mha_sm90.cu) deref
        # `kvCacheScale[0]` whenever `isKVCacheQuantized` is true and
        # do not check for nullptr. `modules/attention.py` only assigns
        # `forward_args.kv_scale_*` for fp4 KV cache, so without this
        # fallback the kernel takes nullptr on fp8-KV models → illegal
        # memory access.
        self.kv_cache_scaling_factor = torch.ones(1,
                                                  dtype=torch.float32,
                                                  device='cuda')
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor

        self.local_layer_idx: Optional[int] = None
        self.fmha_libs: List[Fmha] = []
        if not skip_create_weights_in_init:
            self.update_quant_config(self.quant_config)

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config or QuantConfig()
        self.quant_mode = int(self.quant_config.layer_quant_mode)

        self.has_fp8_qdq = self.has_fp8_kv_cache = self.has_nvfp4 = False
        if self.quant_config is not None:
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )
            self.has_fp4_kv_cache = self.quant_config.layer_quant_mode.has_fp4_kv_cache(
            )

            self.has_fp8_qdq = self.quant_config.layer_quant_mode.has_fp8_qdq()
            self.has_fp8_block_wise = self.quant_config.layer_quant_mode.has_fp8_block_scales(
            )
            self.has_fp8_rowwise = self.quant_config.layer_quant_mode.has_fp8_rowwise(
            )
            self.has_nvfp4 = self.quant_config.layer_quant_mode.has_nvfp4()
            self.has_w4a8_nvfp4_fp8 = self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
            )
        self.create_fmha_libs()

    def get_local_layer_idx(self, metadata: TrtllmAttentionMetadata) -> int:
        if self.local_layer_idx is not None:
            return self.local_layer_idx
        if metadata.kv_cache_manager is None:
            # Uncached: recomputed each call until a cache manager appears.
            return self.layer_idx
        self.local_layer_idx = metadata.kv_cache_manager.layer_offsets[
            self.layer_idx]
        return self.local_layer_idx

    def use_nvfp4_output(
        self,
        metadata: TrtllmAttentionMetadata,
        attention_mask: AttentionMask,
    ) -> bool:
        # Not running NVFP4
        if not self.has_nvfp4:
            return False

        # Default enabled, but allow manual disabling through `TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT=0`
        if not os.environ.get("TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT",
                              "1") == "1":
            return False

        use_paged_context_fmha = (
            metadata.runtime_features.chunked_prefill
            or metadata.runtime_features.cache_reuse
            or metadata.runtime_features.has_speculative_draft_tokens
        ) if metadata.runtime_features else False

        # This is a workaround for https://nvbugs/5624818
        # Paged context FMHA is forced on SM90 for correctness
        if get_sm_version() == 90:
            use_paged_context_fmha = True

        return self._is_nvfp4_output_kernel_available(
            tokens_per_block=metadata.tokens_per_block,
            attention_mask=attention_mask,
            use_paged_context_fmha=use_paged_context_fmha,
            is_mla_enable=self.is_mla_enable,
        )

    def get_quantize_output_dtype(
            self, use_nvfp4_output: bool) -> Optional[torch.dtype]:
        if use_nvfp4_output:
            # Use UINT8 as the container dtype for NVFP4.
            return torch.uint8
        elif (self.has_fp8_qdq or self.has_nvfp4 or self.has_fp8_block_wise
              or self.has_fp8_rowwise
              or self.has_w4a8_nvfp4_fp8) and (self.has_fp8_kv_cache
                                               or self.has_fp4_kv_cache):
            return torch.float8_e4m3fn
        return None

    def _compute_flash_mla_metadata(self,
                                    metadata: TrtllmAttentionMetadata) -> None:
        num_generations = metadata.num_generations
        if num_generations <= 0:
            return

        generation_slice = slice(metadata.num_contexts,
                                 metadata.num_contexts + num_generations)
        generation_kv_lens = metadata.kv_lens_cuda_runtime[generation_slice]
        generation_num_splits = metadata.flash_mla_num_splits[:num_generations +
                                                              1]

        s_q = int(metadata.seq_lens[metadata.num_contexts].item())

        thop.compute_flash_mla_metadata(
            generation_kv_lens,
            metadata.flash_mla_tile_scheduler_metadata,
            generation_num_splits,
            num_generations,
            s_q,
            self.num_heads,
            1,
            self.kv_lora_rank,
        )

    def _ensure_rope_table_size(self, required_max_positions: int) -> None:
        if required_max_positions > self.rope_params.max_positions:
            self.rope_params.max_positions = required_max_positions
            self.rotary_inv_freq, self.rotary_cos_sin = (
                self.rope_params.create_rope_const_params())

    def _is_nvfp4_output_kernel_available(
        self,
        *,
        tokens_per_block: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        use_paged_context_fmha: bool = False,
        is_mla_enable: bool = False,
    ) -> bool:
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            mask_type = AttentionMaskType.causal
        elif attention_mask == PredefinedAttentionMask.FULL:
            mask_type = AttentionMaskType.padding
        else:
            raise ValueError("Unexpected attention mask type")

        return torch.ops.trtllm.attention_supports_nvfp4_output(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            tokens_per_block,
            int(mask_type),
            self.quant_mode,
            use_paged_context_fmha,
            is_mla_enable,
        )

    def create_output(self, q, *, is_quantize_output: bool,
                      metadata: TrtllmAttentionMetadata,
                      attention_mask: AttentionMask, is_gen_only: bool,
                      **kwargs) -> List[torch.Tensor]:
        use_nvfp4_output = False
        out_dtype = None
        if is_quantize_output:
            use_nvfp4_output = self.use_nvfp4_output(metadata, attention_mask)
            out_dtype = self.get_quantize_output_dtype(use_nvfp4_output)

        num_tokens = q.size(0)
        if out_dtype is None:
            out_dtype = q.dtype
        v_head_size = self.head_dim
        if self.is_mla_enable:
            if is_gen_only:
                v_head_size = self.kv_lora_rank if self.rope_append else (
                    self.kv_lora_rank + self.qk_rope_head_dim)
            else:
                v_head_size = self.v_head_dim
        if use_nvfp4_output:
            num_nvfp4_elements_per_container = 2
            scaling_vector_size = 16
            size_per_token = self.num_heads * v_head_size
            output = q.new_empty(
                (num_tokens,
                 size_per_token // num_nvfp4_elements_per_container),
                dtype=torch.uint8)
            padded_row, padded_col = compute_swizzled_sf_shape(
                num_tokens, size_per_token // scaling_vector_size)
            output_sf = q.new_empty(padded_row * padded_col, dtype=torch.uint8)
            return [output, output_sf]
        return [
            q.new_empty((num_tokens, self.num_heads * v_head_size),
                        dtype=out_dtype)
        ]

    @property
    def rope_dim(self) -> int:
        return self.rope_params.dim

    @property
    def rope_base(self) -> float:
        return self.rope_params.theta

    @property
    def rope_scale_type(self) -> int:
        return int(self.rope_params.scale_type)

    @property
    def rope_scale(self) -> float:
        return self.rope_params.scale

    @property
    def rope_short_m_scale(self) -> float:
        return self.rope_params.short_m_scale

    @property
    def rope_long_m_scale(self) -> float:
        return self.rope_params.long_m_scale

    @property
    def rope_max_positions(self) -> int:
        return self.rope_params.max_positions

    @property
    def rope_original_max_positions(self) -> int:
        return self.rope_params.original_max_positions

    def create_fmha_libs(self) -> None:
        self.fmha_libs = []
        for fmha_cls in get_enabled_fmha_lib_classes():
            if fmha_cls.is_available(self):
                self.fmha_libs.append(fmha_cls(self))

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute the TRTLLM attention backend."""
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
        # Cross-attention uses the THOP path; the trtllm-gen backend API does
        # not carry encoder K/V tensors yet.

        # cpp/tensorrt_llm/thop/attentionOp.cpp enables mFP8ContextFMHA for an
        # FP8 KV cache only when use_paged_context_fmha is true. Force paged
        # context so QKV preprocessing and context FMHA use the FP8 path.
        if self.has_fp8_kv_cache:
            metadata.use_paged_context_fmha = True

        # SM90 forces `use_paged_context_fmha` on for correctness
        # (https://nvbugs/5624818).
        if get_sm_version() == 90:
            metadata.use_paged_context_fmha = True

        # Sparse mqa/gqa attention uses generation kernel which reads Q from qPtr (separate buffer).
        # Force paged context FMHA so QKV preprocessing writes Q to q_buf_2_.
        if (self.sparse_params is not None and getattr(
                self.sparse_params, 'algorithm', None) == 'mqa_gqa'):
            metadata.use_paged_context_fmha = True

        if self.is_mla_enable:
            # Context MLA uses separate qkv instead of paged_context_fmha
            metadata.use_paged_context_fmha = False

        if forward_args.output is None:
            is_gen_only = (forward_args.attention_input_type ==
                           AttentionInputType.generation_only)
            outputs = self.create_output(
                q,
                is_quantize_output=forward_args.out_scale is not None,
                metadata=metadata,
                attention_mask=forward_args.attention_mask,
                use_paged_context_fmha=metadata.use_paged_context_fmha,
                is_mla_enable=self.is_mla_enable,
                is_gen_only=is_gen_only,
            )
            forward_args.output = outputs[0]
            forward_args.output_sf = outputs[1] if len(outputs) == 2 else None

        forward_args.is_fused_qkv = not metadata.is_cross and k is None
        forward_args.update_kv_cache = not metadata.is_cross or k is not None
        has_fused_qkv = forward_args.is_fused_qkv and k is None and v is None
        has_unfused_kv = (not forward_args.is_fused_qkv and k is not None
                          and v is not None)
        uses_cached_cross_kv = (metadata.is_cross
                                and not forward_args.update_kv_cache
                                and k is None and v is None)
        assert has_fused_qkv or has_unfused_kv or uses_cached_cross_kv
        # `quant_scale_qkv` only makes sense paired with `quant_q_buffer`: the
        # C++ op interprets the buffer as the destination of a pre-quantized
        # FP8 Q for the DSv4 fused norm+RoPE path. `quant_q_buffer` alone is
        # still valid for the regular FP8-KV-cache path.
        assert (forward_args.quant_scale_qkv is None
                or forward_args.quant_q_buffer is not None), (
                    "quant_scale_qkv requires quant_q_buffer to be set")
        if forward_args.cu_q_seqlens is None:
            forward_args.cu_q_seqlens = metadata.cu_q_seqlens
        if forward_args.cu_kv_seqlens is None:
            forward_args.cu_kv_seqlens = metadata.cu_kv_seqlens

        # Testing only: ``mla_rope_generation`` normally rotates q_pe, appends the
        # new latent to the paged cache, and fills the trtllm-gen scheduler
        # buffers (cumulative q/kv seqlens + the FMHA scheduler counter). When the
        # harness sets ``skip_mla_rope_generation`` it feeds a pre-RoPE'd fused_q,
        # so we skip only the RoPE and do the append + scheduler init here: the
        # generation FMHA only reads the cache, and the fallback path needs the
        # scheduler buffers (the flashinfer trtllm-gen decode kernel ignores them).
        if (self.is_mla_enable and forward_args.skip_mla_rope_generation
                and forward_args.attention_input_type
                == AttentionInputType.generation_only):
            num_ctx = metadata.num_contexts
            n_gen = metadata.num_generations
            # Use the GPU-resident length tensors (no host->device copy) so this
            # stays CUDA-graph-capturable.
            gen_q_lens = metadata.seq_lens_cuda[num_ctx:num_ctx + n_gen].to(
                torch.int32)
            gen_kv_lens = metadata.kv_lens_cuda_runtime[num_ctx:num_ctx +
                                                        n_gen].to(torch.int32)
            cu_q = torch.zeros(n_gen + 1, dtype=torch.int32, device=q.device)
            cu_kv = torch.zeros(n_gen + 1, dtype=torch.int32, device=q.device)
            cu_q[1:] = torch.cumsum(gen_q_lens, dim=0).to(
                torch.int32) * self.num_heads
            cu_kv[1:] = torch.cumsum(gen_kv_lens, dim=0).to(torch.int32)
            forward_args.cu_q_seqlens = cu_q
            forward_args.cu_kv_seqlens = cu_kv
            if forward_args.fmha_scheduler_counter is None:
                forward_args.fmha_scheduler_counter = torch.zeros(
                    1, dtype=torch.uint32, device=q.device)
            else:
                forward_args.fmha_scheduler_counter.zero_()
            assert forward_args.latent_cache is not None
            from .utils import append_mla_latent_cache
            append_mla_latent_cache(
                metadata.kv_cache_manager,
                self.get_local_layer_idx(metadata),
                metadata.request_ids,
                metadata.seq_lens.tolist(),
                metadata.kv_cache_params.num_cached_tokens_per_seq,
                forward_args.latent_cache,
                kv_layout=metadata.kv_layout,
                seq_start=num_ctx,
            )

        # RocketKV and DSA predict which blocks to keep, so build their sparse
        # index tensors here. Skip-softmax needs no prediction.
        sparse_params = self.sparse_params
        if (sparse_params is not None
                and not isinstance(sparse_params, SkipSoftmaxParams)):
            kv_idx, kv_off = self.sparse_kv_predict(q, k, metadata,
                                                    forward_args)
            at_idx, at_off = self.sparse_attn_predict(q, k, metadata,
                                                      forward_args)
            forward_args.sparse_prediction = replace(
                forward_args.sparse_prediction,
                sparse_kv_indices=kv_idx,
                sparse_kv_offsets=kv_off,
                sparse_attn_indices=at_idx,
                sparse_attn_offsets=at_off,
                sparse_attn_indices_block_size=sparse_params.indices_block_size,
            )

        # Compute FlashMLA tile-scheduler metadata once per forward pass.
        # The flag is reset in prepare_flash_mla() and update_for_spec_dec() to trigger
        # recomputation when cache_seq_lens change. The metadata must always match the
        # compacted generation sub-batch, which is also the layout used by block_ids_per_seq.
        if (metadata.enable_flash_mla and forward_args.attention_input_type
                != AttentionInputType.context_only
                and metadata.num_generations > 0
                and not metadata._flash_mla_metadata_valid):
            self._compute_flash_mla_metadata(metadata)
            metadata._flash_mla_metadata_valid = True

        # Blackwell first_sparse: refresh at layer 0 before kernel launch.
        if self.get_local_layer_idx(metadata) == 0 and (
                metadata.spec_bl_tree_first_sparse_mask_offset_kv is not None
                and metadata._seq_lens_cuda is not None):
            metadata.update_blackwell_first_sparse_mask_offset()

        if metadata.is_cross:
            if k is not None and v is not None:
                k_flat = k.contiguous().view(k.shape[0], -1)
                v_flat = v.contiguous().view(v.shape[0], -1)
                forward_args.cross_kv = torch.cat([k_flat, v_flat],
                                                  dim=1).contiguous()

            q_hidden_size = self.num_heads * self.head_dim
            kv_hidden_size = self.num_kv_heads * self.head_dim
            qkv_hidden_size = q_hidden_size + 2 * kv_hidden_size
            if q.shape[1] == q_hidden_size:
                fused_q = q.new_zeros((q.shape[0], qkv_hidden_size))
                fused_q[:, :q_hidden_size].copy_(q)
                q = fused_q
            else:
                assert q.shape[1] == qkv_hidden_size
            k = None
            v = None
            forward_args.is_fused_qkv = True

        attention_input_type = forward_args.attention_input_type
        if not self.is_mla_enable:
            if forward_args.is_fused_qkv:
                qkv_hidden_size = (self.num_heads +
                                   2 * self.num_kv_heads) * self.head_dim
                assert q.shape[1] == qkv_hidden_size
            else:
                q_hidden_size = self.num_heads * self.head_dim
                assert q.shape[1] == q_hidden_size
                if forward_args.update_kv_cache:
                    kv_hidden_size = self.num_kv_heads * self.head_dim
                    assert k.shape[1] == kv_hidden_size
                    assert v.shape[1] == kv_hidden_size
            num_tokens = q.shape[0]
            if k is not None and not metadata.is_cross:
                assert k.shape[0] == num_tokens
                assert v.shape[0] == num_tokens
        else:
            is_sparse_attn = forward_args.sparse_prediction.sparse_attn_indices is not None and forward_args.sparse_prediction.sparse_attn_indices.numel(
            ) > 0
            if attention_input_type == AttentionInputType.context_only and is_sparse_attn:
                assert forward_args.is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.context_only:
                assert not forward_args.is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.qk_nope_head_dim +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.generation_only:
                assert forward_args.is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            else:
                raise ValueError(
                    "In MLA, TrtllmAttention can only support context_only or generation_only, not mixed."
                )
            assert q.shape[
                1] == qkv_hidden_size, f"q.shape[1] must be equal to qkv_hidden_size, got {q.shape[1]=}, {qkv_hidden_size=}"

        batch_size = metadata.kv_lens_cuda_runtime.shape[0]
        assert metadata.kv_lens_runtime.shape[0] == batch_size
        assert metadata.prompt_lens_cuda_runtime.shape[0] == batch_size
        assert metadata.prompt_lens_cpu_runtime.shape[0] == batch_size
        assert metadata.host_request_types_runtime.shape[0] == batch_size

        self._ensure_rope_table_size(metadata.max_seq_len)

        # Prime `self.local_layer_idx` so FMHA implementations read a
        # populated int rather than the `None` placeholder.
        # The call is a fast cache hit after the first forward.
        self.local_layer_idx = self.get_local_layer_idx(metadata)
        if metadata.spec_decoding_bl_tree_mask is not None and self.local_layer_idx == 0:
            metadata.spec_decoding_bl_tree_mask.zero_()

        if self.print_skip_softmax_stat:
            self.skip_softmax_stat.zero_()

        # Propagate the KV cache manager's SWA window to the FMHA kernel when
        # the model does not specify one, so paged-context attention does not
        # read stale page-table entries (BAD_PAGE_INDEX).
        if forward_args.attention_window_size is None and metadata.kv_cache_manager is not None:
            window_vec = getattr(metadata.kv_cache_manager,
                                 'max_attention_window_vec', None)
            if window_vec:
                window = window_vec[self.local_layer_idx % len(window_vec)]
                if window is not None:
                    forward_args.attention_window_size = window
        if forward_args.attention_window_size is None:
            forward_args.attention_window_size = metadata.max_seq_len

        # Promote `out_scale_sf` -> `out_scale` for the NVFP4-output path
        # (kernel reads a single `out_scale` and interprets it as the SF
        # quant scale when `output_sf` is allocated). `output_sf` is
        # populated by `create_output` in `forward` above, so the
        # decision is correct only here, not at the modules/attention.py
        # call site where `output_sf` is always `None`.
        if forward_args.output_sf is not None and forward_args.out_scale_sf is not None:
            forward_args.out_scale = forward_args.out_scale_sf

        # Default `forward_args.kv_scale_*` to the layer-level mirrors when
        # the caller didn't populate them. `modules/attention.py` only sets
        # these for fp4 KV cache; fp8-KV models leave them `None`. Several
        # XQA kernels (mha.cu, mha_sm90.cu) deref `kvCacheScale[0]` when
        # `isKVCacheQuantized` is true and don't check for nullptr, so
        # passing `None` crashes with illegal memory access.
        if forward_args.kv_scale_orig_quant is None:
            forward_args.kv_scale_orig_quant = self.kv_scale_orig_quant
        if forward_args.kv_scale_quant_orig is None:
            forward_args.kv_scale_quant_orig = self.kv_scale_quant_orig

        sparse_params = self.sparse_params
        if isinstance(sparse_params, SkipSoftmaxParams):
            forward_args.skip_softmax_kernel_params = (
                sparse_params.scheduler.get_kernel_params(
                    timestep=forward_args.timestep))

        # max_context_q_len_override is only set when encoder CUDA graphs are enabled.
        if metadata.max_context_q_len_override is not None:
            assert metadata.is_cuda_graph
            assert metadata.num_generations == 0
            assert metadata.kv_cache_manager is None
            assert metadata.num_contexts == metadata.num_seqs

        if not self.fmha_libs:
            self.create_fmha_libs()

        for fmha in self.fmha_libs:
            if fmha.is_supported(q, k, v, metadata, forward_args):
                fmha.forward(q, k, v, metadata, forward_args)
                break
        else:
            raise RuntimeError(
                "No TRT-LLM attention FMHA library supports this request.")

        if self.print_skip_softmax_stat:
            total_blocks, skipped_blocks = self.skip_softmax_stat
            if total_blocks != 0:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped_blocks} / {total_blocks}"
                    f" = {skipped_blocks / total_blocks * 100: .2f}%")

        if forward_args.output_sf is None:
            return forward_args.output
        else:
            return forward_args.output, forward_args.output_sf

    @classmethod
    def support_fused_rope(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True

    @classmethod
    def support_mla(cls) -> bool:
        return True

    def has_cached_kv_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0)

    def is_chunked_prefill_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0
                and metadata.runtime_features.chunked_prefill)

    def has_cached_kv_for_mla_context_warmup(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        """KV cache reuse / chunked prefill MLA context check for warmup, do not check num_ctx_cached_tokens."""
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv)

    def load_paged_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        out_dtype: torch.dtype,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None
        assert metadata.max_ctx_kv_len > 0
        assert metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens == metadata.host_ctx_kv_indptr[
            metadata.num_contexts]

        beam_width = 1

        compressed_kv, k_pe = torch.ops.trtllm.load_paged_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens,
            metadata.max_ctx_kv_len,
            metadata.ctx_kv_indptr,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_quant_orig
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )

        return compressed_kv, k_pe

    def load_chunked_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        num_ctx_cached_tokens: int,
        cu_chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.Tensor,
        chunked_max_seq_len: int,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        if metadata.max_ctx_cached_token_len == 0:
            empty_kv = torch.empty((0, self.mla_params.kv_lora_rank),
                                   dtype=out_dtype,
                                   device=cu_chunked_seq_len.device)
            empty_k_pe = torch.empty((0, self.mla_params.qk_rope_head_dim),
                                     dtype=out_dtype,
                                     device=cu_chunked_seq_len.device)
            return empty_kv, empty_k_pe

        beam_width = 1

        output_kv, output_k_pe = torch.ops.trtllm.load_chunked_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            num_ctx_cached_tokens,
            cu_chunked_seq_len,
            chunked_global_offset,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_quant_orig
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            chunked_max_seq_len,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )
        return output_kv, output_k_pe

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        # Ensure RoPE cos/sin table covers the sequence length before the
        # kernel reads it.
        self._ensure_rope_table_size(metadata.kv_cache_manager.max_seq_len)

        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            metadata.num_contexts,
            metadata.ctx_cached_token_indptr,
            metadata.ctx_kv_indptr,
            metadata.max_ctx_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )

    def merge_attention_for_mla(
        self,
        merged_attn: torch.Tensor,
        temp_attn: torch.Tensor,
        softmax_stats: torch.Tensor,
        temp_softmax_stats: torch.Tensor,
        merge_op: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged_attn,
            temp_attn,
            softmax_stats,
            temp_softmax_stats,
            metadata.num_contexts,
            metadata.ctx_uncached_token_indptr,  # cu_q_seq_len
            metadata.max_ctx_seq_len,  # max_q_seq_len
            merge_op,
            self.num_heads,
            self.mla_params.v_head_dim,
        )

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Predict sparse kv indices. It's implemented in the derived class.
        """
        raise NotImplementedError

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Predict sparse attn indices. It's implemented in the derived class.
        """
        raise NotImplementedError

    def mla_rope_generation(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """
            fused_q (torch.Tensor): The tensor to store the fused q, with shape (num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            q_pe (torch.Tensor): The tensor to store the q_pe, with shape (num_tokens, num_heads, qk_rope_head_dim) on GPU.
            latent_cache (torch.Tensor): The tensor to store the latent cache, with shape (num_tokens, kv_lora_rank + qk_rope_head_dim) on GPU.
            cu_q_seqlens (torch.Tensor): The tensor to store the cu_q_seqlens, with shape (num_seqs + 1) on GPU.
            cu_kv_seqlens (torch.Tensor): The tensor to store the cu_kv_seqlens, with shape (num_seqs + 1) on GPU.
            fmha_scheduler_counter (torch.Tensor): The tensor to store the fmha_scheduler_counter, with shape (1) on GPU.
            mla_bmm1_scale (torch.Tensor): The tensor to store the mla_bmm1_scale, with shape (2) on GPU.
            mla_bmm2_scale (torch.Tensor): The tensor to store the mla_bmm2_scale, with shape (1) on GPU.
            quant_q_buffer (torch.Tensor): The tensor to store the quant_q_buffer, with shape (tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            helix_position_offsets (torch.Tensor): The tensor to store the helix position offsets, with shape (num_tokens) on GPU.
            out_scale (torch.Tensor): The tensor to store the out_scale, with shape (1) on GPU.
        """

        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        # Ensure RoPE cos/sin table covers the sequence length before the
        # kernel reads it.
        self._ensure_rope_table_size(metadata.max_seq_len)

        helix_tensor_params = [
            metadata.helix_position_offsets, metadata.helix_is_inactive_rank
        ]

        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            self.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,  # sequence_length
            metadata.kv_lens_runtime,  # host_past_key_value_lengths
            metadata.prompt_lens_cpu_runtime,  # host_context_lengths,
            metadata.num_contexts,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            out_scale,
            metadata.block_ids_per_seq,
            helix_tensor_params,
            self.predicted_tokens_per_seq,
            self.get_local_layer_idx(metadata),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.max_seq_len,  # attention_window_size
            metadata.beam_width,
            self.quant_mode,
            self.q_scaling,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.rope_append,
        )
