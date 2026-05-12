import functools
import math
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ..speculative.interface import SpecMetadata
    from ..speculative.spec_tree_manager import SpecTreeManager

from tensorrt_llm._torch.attention_backend import trtllm_gen
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory, prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..utils import (compute_swizzled_sf_shape, get_global_attrs,
                     get_model_extra_attrs)
from .interface import (AttentionBackend, AttentionForwardArgs,
                        AttentionInputType, AttentionMask, AttentionMetadata,
                        KVCacheParams, MLAParams, PositionalEmbeddingParams,
                        PredefinedAttentionMask, RopeParams,
                        merge_attention_forward_args)
from .trtllm_gen import trtllm_gen_attention

# Enable TRTLLM-Gen attention backend via environment variable (default: off).
_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION = (os.environ.get(
    "TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION", "0") == "1")


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

    # TrtllmAttention needs to know the max sequence length.
    # Implemented as a property to support no cache mode.
    max_seq_len: Optional[int]

    # Storage for internal max_seq_len value
    _max_seq_len_storage: Optional[int] = field(default=None,
                                                init=True,
                                                repr=False)

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
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    spec_decoding_packed_mask: Optional[torch.Tensor] = None
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None

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

    # Pre-computed FlashMLA tile-scheduler metadata and num_splits.
    # Computed once per forward pass in TrtllmAttention.forward() and reused across layers.
    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None
    flash_mla_num_splits: Optional[torch.Tensor] = None
    _flash_mla_metadata_valid: bool = field(default=False,
                                            init=False,
                                            repr=False)

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
        self._post_init_with_buffers(self.cuda_graph_buffers)

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
            self.kv_cache_block_offsets = self.get_empty(
                buffers,
                [
                    self.kv_cache_manager.num_pools, self.max_num_sequences, 2,
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
                self.draft_kv_cache_block_offsets = self.get_empty(
                    buffers,
                    [
                        self.draft_kv_cache_manager.num_pools,
                        self.max_num_sequences, 2,
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

        prompt_lens = torch.tensor(
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

        self.kv_lens_cuda_runtime = self.kv_lens_cuda[:self.num_seqs]
        # Don't use self.kv_lens here because it includes extra tokens.
        # Use actual KV length (without extra tokens) for kv_lens_runtime,
        # which becomes host_past_key_value_lengths and eventually mMaxSeqLenKv.
        self.kv_lens_runtime = kv_lens[:self.num_seqs]
        self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.num_seqs]
        self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]
        self.host_request_types_runtime = self.host_request_types[:self.
                                                                  num_seqs]

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
        '''

        # Disable spec decoding on Blackwell (sm100+). The trtllmGen FMHA
        # kernels do not yet support speculative decoding mode.
        self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
            not self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()))

        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled
        self.is_spec_dec_tree = is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = is_spec_dec_dynamic_tree

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
                self.spec_decoding_packed_mask = torch.empty(
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

            # Case 1: dynamic tree — copy per-request params from spec_tree_manager.
            if self.is_spec_dec_dynamic_tree:
                assert spec_tree_manager is not None, "spec_tree_manager is required for dynamic tree"
                n_dt = spec_tree_manager.max_total_draft_tokens + 1
                mask_width = spec_tree_manager.spec_dec_packed_mask.shape[-1]

                self.spec_decoding_position_offsets[:batch_size * n_dt].view(
                    batch_size,
                    n_dt).copy_(spec_tree_manager.
                                spec_dec_position_offsets[:batch_size],
                                non_blocking=True)

                if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
                    self.spec_decoding_packed_mask[:batch_size].zero_()
                    self.spec_decoding_packed_mask[:batch_size, :n_dt, :].copy_(
                        spec_tree_manager.spec_dec_packed_mask[:batch_size],
                        non_blocking=True)
                else:
                    total = batch_size * n_dt * mask_width
                    self.spec_decoding_packed_mask.view(-1)[:total].copy_(
                        spec_tree_manager.spec_dec_packed_mask[:batch_size].
                        reshape(-1),
                        non_blocking=True)

                self.spec_decoding_generation_lengths[:batch_size].fill_(n_dt)

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

            # Case 4: linear tree
            else:
                # Currently dynamic draft length is only supported for linear tree
                # Dynamic draft length needs position offsets and packed mask to be shaped for each runtime draft length.
                # So we create cache for position offsets and packed mask for each draft length to avoid reallocation.
                assert max_draft_len == max_total_draft_tokens, "max_draft_len should be equal to max_total_draft_tokens for linear tree"
                runtime_draft_len = (spec_metadata.runtime_draft_len
                                     if spec_metadata is not None else
                                     max_draft_len)
                self.generate_spec_decoding_generation_length(
                    runtime_draft_len=runtime_draft_len)
                self.spec_decoding_position_offsets = generate_spec_decoding_position_offsets(
                    self.max_num_requests, runtime_draft_len)
                self.spec_decoding_packed_mask = generate_spec_decoding_packed_mask(
                    self.max_num_requests, runtime_draft_len)

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

        self.kv_cache_scaling_factor = torch.ones(1,
                                                  dtype=torch.float32,
                                                  device='cuda')
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor
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

    def get_local_layer_idx(self, metadata: TrtllmAttentionMetadata) -> int:
        if metadata.kv_cache_manager is None:
            return self.layer_idx
        else:
            return metadata.kv_cache_manager.layer_offsets[self.layer_idx]

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

    def _get_mask_type(self,
                       attention_mask: AttentionMask) -> AttentionMaskType:
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            return AttentionMaskType.causal
        if attention_mask == PredefinedAttentionMask.FULL:
            return AttentionMaskType.padding
        raise ValueError("Unexpected attention mask type")

    def _run(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
        use_paged_context_fmha: bool,
        sparse_kv_indices: Optional[torch.Tensor],
        sparse_kv_offsets: Optional[torch.Tensor],
        sparse_attn_indices: Optional[torch.Tensor],
        sparse_attn_offsets: Optional[torch.Tensor],
        sparse_attn_indices_block_size: int,
        num_sparse_topk: int,
        sparse_mla_topk_lens: Optional[torch.Tensor],
        compressed_kv_cache_pool_ptr: Optional[int],
        skip_softmax_threshold_scale_factor_prefill: Optional[float],
        skip_softmax_threshold_scale_factor_decode: Optional[float],
    ) -> None:
        is_fused_qkv = not metadata.is_cross and k is None
        update_kv_cache = not metadata.is_cross or k is not None
        assert (is_fused_qkv and k is None
                and v is None) or (not is_fused_qkv and k is not None
                                   and v is not None)

        attention_input_type = forward_args.attention_input_type
        if not self.is_mla_enable:
            if is_fused_qkv:
                qkv_hidden_size = (self.num_heads +
                                   2 * self.num_kv_heads) * self.head_dim
                assert q.shape[1] == qkv_hidden_size
            else:
                q_hidden_size = self.num_heads * self.head_dim
                assert q.shape[1] == q_hidden_size
                if update_kv_cache:
                    kv_hidden_size = self.num_kv_heads * self.head_dim
                    assert k.shape[1] == kv_hidden_size
                    assert v.shape[1] == kv_hidden_size
            num_tokens = q.shape[0]
            if k is not None:
                assert k.shape[0] == num_tokens
                assert v.shape[0] == num_tokens
        else:
            is_sparse_attn = sparse_attn_indices is not None and sparse_attn_indices.numel(
            ) > 0
            if attention_input_type == AttentionInputType.context_only and is_sparse_attn:
                assert is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.context_only:
                assert not is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.qk_nope_head_dim +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.generation_only:
                assert is_fused_qkv
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

        mask_type = self._get_mask_type(forward_args.attention_mask)
        self._ensure_rope_table_size(metadata.max_seq_len)

        rotary_embedding_dim = self.rope_params.dim
        rotary_embedding_base = self.rope_params.theta
        rotary_embedding_scale_type = int(self.rope_params.scale_type)
        rotary_embedding_scales = [
            self.rope_params.scale, self.rope_params.short_m_scale,
            self.rope_params.long_m_scale
        ]
        rotary_embedding_max_position_info = [
            self.rope_params.max_positions,
            self.rope_params.original_max_positions
        ]
        spec_decoding_bool_params = [
            metadata.is_spec_decoding_enabled, metadata.use_spec_decoding,
            metadata.is_spec_dec_tree
        ]
        position_offsets_for_cpp = metadata.spec_decoding_position_offsets
        if (metadata.spec_decoding_position_offsets is not None
                and metadata.spec_decoding_position_offsets.dim() == 1):
            position_offsets_for_cpp = metadata.spec_decoding_position_offsets.view(
                metadata.max_num_requests, -1)

        spec_decoding_tensor_params = [
            metadata.spec_decoding_generation_lengths, position_offsets_for_cpp,
            metadata.spec_decoding_packed_mask
        ]
        if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
            spec_decoding_tensor_params.append(
                metadata.spec_decoding_bl_tree_mask_offset)
            spec_decoding_tensor_params.append(
                metadata.spec_decoding_bl_tree_mask)
            spec_decoding_tensor_params.append(
                metadata.spec_bl_tree_first_sparse_mask_offset_kv)
        helix_tensor_params = [
            metadata.helix_position_offsets, metadata.helix_is_inactive_rank
        ]

        layer_idx = self.get_local_layer_idx(metadata)
        if metadata.spec_decoding_bl_tree_mask is not None and layer_idx == 0:
            metadata.spec_decoding_bl_tree_mask.zero_()

        if self.print_skip_softmax_stat:
            self.skip_softmax_stat.zero_()

        use_nvfp4_output = output_sf is not None
        out_scale = (forward_args.out_scale_sf
                     if use_nvfp4_output else forward_args.out_scale)
        kv_scale_orig_quant = (self.kv_scale_orig_quant
                               if forward_args.kv_scales_sf_inv is None else
                               forward_args.kv_scales_sf_inv)
        kv_scale_quant_orig = (self.kv_scale_quant_orig
                               if forward_args.kv_scales_sf is None else
                               forward_args.kv_scales_sf)
        mrope_rotary_cos_sin = forward_args.mrope_config.get(
            'mrope_rotary_cos_sin'
        ) if forward_args.mrope_config is not None else None
        mrope_position_deltas = forward_args.mrope_config.get(
            'mrope_position_deltas'
        ) if forward_args.mrope_config is not None else None
        workspace = metadata.workspace if not metadata.is_cuda_graph else metadata.cuda_graph_workspace
        flash_mla_tile_scheduler_metadata = (
            metadata.flash_mla_tile_scheduler_metadata
            if metadata.enable_flash_mla else None)
        flash_mla_num_splits = metadata.flash_mla_num_splits if metadata.enable_flash_mla else None
        attention_window_size = (forward_args.attention_window_size
                                 or metadata.max_seq_len)
        max_context_length = min(metadata.max_seq_len - 1,
                                 metadata.max_num_tokens)

        helix_active = metadata.helix_position_offsets is not None
        use_sage_attn = (forward_args.sage_attn_num_elts_per_blk_q > 0
                         or forward_args.sage_attn_num_elts_per_blk_k > 0
                         or forward_args.sage_attn_num_elts_per_blk_v > 0)
        if _TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION and not helix_active and not use_sage_attn and trtllm_gen.is_supported(
                q=q,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                out_dtype=output.dtype,
                mask_type=int(mask_type),
                has_alibi=(self.position_embedding_type == 4
                           or self.position_embedding_type == 5),
                is_padded=False,
                use_paged_kv_cache=(metadata.kv_cache_block_offsets
                                    is not None),
                tokens_per_block=metadata.tokens_per_block,
                beam_width=metadata.beam_width,
                position_shift_enabled=False,
                sink_token_length=0,
                cross_attention=False,
                is_spec_decoding=metadata.is_spec_decoding_enabled,
                is_mla_enable=self.is_mla_enable,
                is_fused_qkv=is_fused_qkv,
                update_kv_cache=update_kv_cache,
                has_cross_kv=False,
                quant_config=self.quant_config,
                kv_cache_manager=metadata.kv_cache_manager,
                skip_softmax_threshold_scale_factor_prefill=
                skip_softmax_threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_decode=
                skip_softmax_threshold_scale_factor_decode,
        )[0]:
            trtllm_gen_attention(
                q,
                k,
                v,
                output,
                output_sf,
                workspace,
                metadata.kv_lens_cuda_runtime,
                metadata.kv_lens_runtime,
                metadata.host_total_kv_lens,
                metadata.prompt_lens_cuda_runtime,
                metadata.prompt_lens_cpu_runtime,
                metadata.host_request_types_runtime,
                metadata.kv_cache_block_offsets,
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                metadata.cache_indirection,
                kv_scale_orig_quant,
                kv_scale_quant_orig,
                out_scale,
                self.rotary_inv_freq,
                self.rotary_cos_sin,
                forward_args.latent_cache,
                forward_args.q_pe,
                metadata.block_ids_per_seq,
                forward_args.attention_sinks,
                is_fused_qkv,
                update_kv_cache,
                self.predicted_tokens_per_seq,
                layer_idx,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                metadata.tokens_per_block,
                metadata.max_num_requests,
                max_context_length,
                attention_window_size,
                0,
                metadata.beam_width,
                int(mask_type),
                self.quant_mode,
                self.q_scaling,
                self.position_embedding_type,
                rotary_embedding_dim,
                rotary_embedding_base,
                rotary_embedding_scale_type,
                rotary_embedding_scales,
                rotary_embedding_max_position_info,
                use_paged_context_fmha,
                int(attention_input_type),
                self.is_mla_enable,
                forward_args.chunked_prefill_buffer_batch_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                helix_tensor_params,
                self.attention_chunk_size,
                forward_args.softmax_stats_tensor,
                spec_decoding_bool_params,
                spec_decoding_tensor_params,
                sparse_kv_indices,
                sparse_kv_offsets,
                sparse_attn_indices,
                sparse_attn_offsets,
                sparse_attn_indices_block_size,
                num_sparse_topk,
                skip_softmax_threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_decode,
                self.skip_softmax_stat,
                forward_args.cu_q_seqlens,
                forward_args.cu_kv_seqlens,
                forward_args.fmha_scheduler_counter,
                forward_args.mla_bmm1_scale,
                forward_args.mla_bmm2_scale,
                forward_args.quant_q_buffer,
                self.quant_config,
                metadata.kv_cache_manager,
                metadata.num_contexts,
                metadata.num_ctx_tokens,
                global_layer_idx=self.layer_idx,
            )
        else:
            thop.attention(
                q,
                k,
                v,
                output,
                output_sf,
                workspace,
                metadata.kv_lens_cuda_runtime,
                metadata.kv_lens_runtime,
                metadata.host_total_kv_lens,
                metadata.prompt_lens_cuda_runtime,
                metadata.prompt_lens_cpu_runtime,
                metadata.host_request_types_runtime,
                metadata.kv_cache_block_offsets,
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                metadata.cache_indirection,
                kv_scale_orig_quant,
                kv_scale_quant_orig,
                out_scale,
                self.rotary_inv_freq,
                self.rotary_cos_sin,
                forward_args.latent_cache,
                forward_args.q_pe,
                metadata.block_ids_per_seq,
                forward_args.attention_sinks,
                is_fused_qkv,
                update_kv_cache,
                self.predicted_tokens_per_seq,
                layer_idx,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                metadata.tokens_per_block,
                metadata.max_num_requests,
                max_context_length,
                attention_window_size,
                0,
                metadata.beam_width,
                int(mask_type),
                self.quant_mode,
                self.q_scaling,
                self.position_embedding_type,
                rotary_embedding_dim,
                rotary_embedding_base,
                rotary_embedding_scale_type,
                rotary_embedding_scales,
                rotary_embedding_max_position_info,
                use_paged_context_fmha,
                int(attention_input_type),
                self.is_mla_enable,
                forward_args.chunked_prefill_buffer_batch_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.rope_append,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                helix_tensor_params,
                self.attention_chunk_size,
                forward_args.softmax_stats_tensor,
                spec_decoding_bool_params,
                spec_decoding_tensor_params,
                sparse_kv_indices,
                sparse_kv_offsets,
                sparse_attn_indices,
                sparse_attn_offsets,
                sparse_attn_indices_block_size,
                num_sparse_topk,
                sparse_mla_topk_lens,
                skip_softmax_threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_decode,
                self.skip_softmax_stat,
                forward_args.cu_q_seqlens,
                forward_args.cu_kv_seqlens,
                forward_args.fmha_scheduler_counter,
                forward_args.mla_bmm1_scale,
                forward_args.mla_bmm2_scale,
                forward_args.quant_q_buffer,
                flash_mla_tile_scheduler_metadata,
                flash_mla_num_splits,
                forward_args.sage_attn_num_elts_per_blk_q,
                forward_args.sage_attn_num_elts_per_blk_k,
                forward_args.sage_attn_num_elts_per_blk_v,
                forward_args.sage_attn_qk_int8,
                num_contexts=metadata.num_contexts,
                num_ctx_tokens=metadata.num_ctx_tokens,
                compressed_kv_cache_pool_ptr=compressed_kv_cache_pool_ptr,
            )

        if self.print_skip_softmax_stat:
            total_blocks, skipped_blocks = self.skip_softmax_stat
            if total_blocks != 0:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped_blocks} / {total_blocks}"
                    f" = {skipped_blocks / total_blocks * 100: .2f}%")

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
        assert not metadata.is_cross, "TRT-LLM Attention does not support cross attention yet."

        use_paged_context_fmha = (
            metadata.runtime_features.chunked_prefill
            or metadata.runtime_features.cache_reuse
            or metadata.runtime_features.has_speculative_draft_tokens
        ) if metadata.runtime_features else False

        # This is a workaround for https://nvbugs/5624818
        # Paged context FMHA is forced on SM90 for correctness
        if get_sm_version() == 90:
            use_paged_context_fmha = True

        # Sparse mqa/gqa attention uses generation kernel which reads Q from qPtr (separate buffer).
        # Force paged context FMHA so QKV preprocessing writes Q to q_buf_2_.
        if (self.sparse_attention_config is not None and getattr(
                self.sparse_attention_config, 'algorithm', None) == 'mqa_gqa'):
            use_paged_context_fmha = True

        if self.is_mla_enable:
            # Context MLA uses separate qkv instead of paged_context_fmha
            use_paged_context_fmha = False

        output = forward_args.output
        output_sf = forward_args.output_sf
        if output is None:
            # Output is not provided.
            is_gen_only = (forward_args.attention_input_type ==
                           AttentionInputType.generation_only)
            outputs = self.create_output(
                q,
                is_quantize_output=forward_args.out_scale is not None,
                metadata=metadata,
                attention_mask=forward_args.attention_mask,
                use_paged_context_fmha=use_paged_context_fmha,
                is_mla_enable=self.is_mla_enable,
                is_gen_only=is_gen_only,
            )

            output = outputs[0]
            output_sf = outputs[1] if len(outputs) == 2 else None

        sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets = None, None, None, None
        sparse_attn_indices_block_size = 1
        skip_softmax_threshold_scale_factor_prefill = None
        skip_softmax_threshold_scale_factor_decode = None
        num_sparse_topk = getattr(metadata, 'num_sparse_topk', 0)
        sparse_mla_topk_lens = None
        compressed_kv_cache_pool_ptr = None
        if self.sparse_attention_config is not None:
            if isinstance(self.sparse_attention_config,
                          SkipSoftmaxAttentionConfig):
                skip_softmax_threshold_scale_factor_prefill = self.sparse_attention_config.threshold_scale_factor_prefill
                skip_softmax_threshold_scale_factor_decode = self.sparse_attention_config.threshold_scale_factor_decode

            else:
                sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(
                    q, k, metadata, forward_args)
                sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
                    q, k, metadata, forward_args)
                sparse_attn_indices_block_size = self.sparse_attention_config.get_indices_block_size(
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
        layer_idx = self.get_local_layer_idx(metadata)
        if layer_idx == 0 and (metadata.spec_bl_tree_first_sparse_mask_offset_kv
                               is not None
                               and metadata._seq_lens_cuda is not None):
            metadata.update_blackwell_first_sparse_mask_offset()

        self._run(q, k, v, output, output_sf, metadata, forward_args,
                  use_paged_context_fmha, sparse_kv_indices, sparse_kv_offsets,
                  sparse_attn_indices, sparse_attn_offsets,
                  sparse_attn_indices_block_size, num_sparse_topk,
                  sparse_mla_topk_lens, compressed_kv_cache_pool_ptr,
                  skip_softmax_threshold_scale_factor_prefill,
                  skip_softmax_threshold_scale_factor_decode)

        if output_sf is None:
            return output
        else:
            return output, output_sf

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

    def is_chunked_prefill_mla_context_for_warmup(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        """Chunked prefill MLA context check for warmup; does not check num_ctx_cached_tokens."""
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.runtime_features.chunked_prefill)

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

        sink_token_length = 0
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
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
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

        sink_token_length = 0
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
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            chunked_max_seq_len,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
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

        sink_token_length = 0
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
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
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
        sink_token_length = 0

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
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
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
            sink_token_length,
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
