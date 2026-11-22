# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 DSA indexer: FP8 index keys, an exact Top-K in a fixed row order (the deterministic GVR on
SM 10.3, the CUDA radix elsewhere), a position-free (NoPE) indexer and BF16 ``wq_b`` / ``wk`` / ``weights_proj``.

Every kernel runs through a family op. The host-side planning (prefill chunking, slot mappings, CuTe DSL logits
split) is ``dsa_indexer_plan``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention.backends.interface import MLAParams
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (
    supports as dsa_topk_order_remap_supports,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import dsa_prefill_share
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (
    convert_req_index_to_global,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (
    cute_dsl_fp8_paged_mqa_logits,
    cute_dsl_fp8_paged_mqa_logits_pdl,
    logits_row_stride,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_index_key_norm import (  # noqa: E501
    indexer_k_norm,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    _DG_SCHEDULE_BLOCK_KV,
    _INDEXER_MQA_LOGITS_ELEM_BUDGET,
    _MAX_RADIX_BLOCKS_PER_ROW,
    K31IndexerParams,
    K31IndexerPrefillChunkMetadata,
    _compute_slot_mappings,
    _packed_logits_elem_budget,
    _prefill_q_split_threshold,
    _split_prefill_queries,
    compute_cu_seqlen_kv_bounds_with_cache,
    plan_mla_chunked_prefill_chunks,
    select_indexer_chunk_size,
    split_prefill_chunks,
    use_cute_dsl_paged_mqa_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import K31DSAParams
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_topk_order_remap import (
    dsa_topk_order_remap,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fp8_mqa_logits import (
    fp8_mqa_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (
    fused_cat_fp8,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (
    k31_gather_k_cache_paged,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (
    get_paged_mqa_logits_metadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_k_cache_scatter_op import (
    indexer_k_cache_scatter_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode import (
    indexer_topk_decode,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (
    indexer_topk_decode_gvr,
    indexer_topk_decode_gvr_paged,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (
    indexer_topk_prefill,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill_gvr import (
    indexer_topk_prefill_gvr,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import (
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31Linear,
    bf16_linear,
)
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
        K31DSAtrtllmAttentionMetadata,
    )


#: The compute capability the deterministic GVR Top-K is certified on (its op ARCHS); on any other
#: architecture the indexer keeps the CUDA radix Top-K.
_GVR_TOPK_CAPABILITY = (10, 3)


@functools.lru_cache(maxsize=None)
def gvr_topk_device(device_index: Optional[int]) -> bool:
    """Whether the indexer takes the deterministic GVR Top-K on this device (SM 10.3)."""
    return torch.cuda.get_device_capability(device_index) == _GVR_TOPK_CAPABILITY


def deterministic_topk_order(topk_indices: torch.Tensor) -> torch.Tensor:
    """Each Top-K row in ascending index order, its unused (-1) slots last.

    The radix Top-K kernels write a row's indices in atomic order, which varies from run to run,
    and the sparse MLA accumulates the selected keys in row order: a fixed order makes the
    attention output independent of that order. The set of indices is unchanged.
    """
    unused = torch.iinfo(topk_indices.dtype).max
    keyed = torch.where(topk_indices < 0, unused, topk_indices)
    ordered = keyed.sort(dim=-1).values
    return torch.where(ordered == unused, -1, ordered)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "K31DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()
    page_index_scale, layer_offset = (
        attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(layer_idx)
    )

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    tokens_per_block = attn_metadata._cached_tokens_per_block
    if (
        getattr(attn_metadata, "_k31_perf", False)
        and dsa_topk_order_remap_supports(topk_indices)
        and tokens_per_block & (tokens_per_block - 1) == 0
    ):
        # Row order + pool remap in one launch, bitwise the torch sort + convert below.
        global_indices = dsa_topk_order_remap(
            req_idx,
            block_table,
            topk_indices,
            tokens_per_block,
            topk_indices.shape[1],
            page_index_scale * tokens_per_block,
            layer_offset,
        )
        return global_indices, attn_metadata._cached_pool_view

    topk_indices = deterministic_topk_order(topk_indices)
    global_indices = convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        page_index_scale * attn_metadata._cached_tokens_per_block,
        layer_offset,
    )

    return global_indices, attn_metadata._cached_pool_view


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor, s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


class _KeyNormWeights(nn.Module):
    """The index-key LayerNorm's ``weight``, ``bias`` and ``variance_epsilon``; the family ``index_key_norm`` op
    runs the norm."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps


class K31Indexer(nn.Module):
    """The Kimi K3.1 DSA indexer: selects the top-K cached keys of each query token.

    Parameters (``wq_b``, ``wk``, ``weights_proj``, ``k_norm``) plus the glue that calls the family DSA ops in the
    engine indexer's order; not an engine class.
    """

    #: Set by ``K31MLAAttention`` for the SM 10.3 perf targets: their key-sharded decode Top-K, the
    #: pool-row decode Top-K and the PDL native-tile logits. Every target on SM 10.3 takes the
    #: deterministic GVR decode / prefill Top-K (``gvr_topk``; the GVR keeps the smaller column on an
    #: exact tie at the k-th score).
    _k31_perf = False
    #: With ``_k31_perf``: native-tile logits launch with PDL (the attention-DP perf target turns it
    #: off: slower at its 224-row decode).
    _k31_logits_pdl = True
    #: Set by ``K31MLAAttention`` for one generation call: (block_table, tokens_per_block,
    #: stride_factor, layer_offset) of the MLA pool; the indexer then reports in
    #: _gen_rows_paged whether its rows are already ordered pool rows.
    _gen_pool_remap = None
    _gen_rows_paged = False

    def __init__(
        self,
        mla_params: MLAParams,
        sparse_params: K31DSAParams,
        dtype: torch.dtype,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        """The indexer's BF16 projections (``dtype``) and key norm. Routing admits FP8 index keys and a
        position-free indexer (no rotated index channels) only."""
        super().__init__()
        assert sparse_params.indexer_k_dtype == "fp8", sparse_params.indexer_k_dtype
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.n_heads = sparse_params.index_n_heads  # 64
        self.head_dim = sparse_params.index_head_dim  # 128
        self.index_topk = sparse_params.index_topk  # 2048
        self.layer_idx = layer_idx

        self.wq_b = K31Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            dtype=dtype,
            use_custom_cublas_mm=True,
        )
        # wk and weights_proj run in the model dtype (BF16): one native GEMM
        # over their fused weight, the precision of the reference indexer.
        self.wk = K31Linear(self.hidden_size, self.head_dim, dtype=dtype)
        k_norm_eps = sparse_params.index_k_norm_eps
        self.k_norm = _KeyNormWeights(self.head_dim, 1e-6 if k_norm_eps is None else k_norm_eps)
        self.weights_proj = K31Linear(self.hidden_size, self.n_heads, dtype=dtype)
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

        self.softmax_scale = self.head_dim**-0.5
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.use_cute_dsl_paged_mqa_logits = use_cute_dsl_paged_mqa_logits(
            sparse_params.use_cute_dsl_paged_mqa_logits
        )
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        # Exact Top-K in both phases (the deterministic GVR on SM 10.3, the CUDA radix elsewhere); the
        # heuristic Top-K settings of the sparse-attention config select nothing here.
        if sparse_params.enable_heuristic_topk:
            logger.info_once(
                "the Kimi K3.1 DSA indexer runs an exact Top-K; enable_heuristic_topk is ignored",
                key="k31_dsa_indexer_heuristic_topk_ignored",
            )

    def cache_derived_state(self) -> None:
        """Fuse wk and weights_proj into one weight for a single GEMM."""
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def gvr_topk(self, device: torch.device) -> bool:
        """Whether this indexer's prefill and decode Top-K are the deterministic GVR: always for the perf
        targets, and on SM 10.3 for every target; elsewhere the CUDA radix (whose picks among scores
        tied at the k-th value vary from run to run)."""
        return self._k31_perf or gvr_topk_device(device.index)

    @staticmethod
    def prepare_one_prefill_chunk(
        metadata: K31DSAtrtllmAttentionMetadata,
        chunk_specs: List[Tuple[int, int, int, int]],
    ) -> K31IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - token_start_in_req, token_end_in_req are indices into current batch context tokens
                        - For multi-request: multiple specs from different requests (full requests)
                        - For intra-request: single spec from one request's Q-block

        Note: Cached token counts are derived from metadata.host_ctx_cached_token_indptr
        """
        device = metadata.cu_seqlen_ks.device
        # One index key per token: KV positions are token positions.
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1]
                - metadata.host_ctx_cached_token_indptr[req_idx]
            ).item()

            # Total KV tokens for this request
            req_kv_len = num_cached + token_end_in_req

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:req_kv_len]
            cu_seqlen_ks = torch.zeros(num_q_tokens, dtype=torch.int32, device="cpu")
            cu_seqlen_ke = (
                torch.arange(
                    token_start_in_req + 1, token_end_in_req + 1, dtype=torch.int32, device="cpu"
                )
                + num_cached
            )

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping
            # For req_idx=0 the offset is 0; for other indices the cumulative KV offset
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                for j in range(req_idx)
            )
            total_kv_for_req = req_kv_len
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_num_past_tokens = []
            req_kv_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_past_tokens = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1]
                    - metadata.host_ctx_cached_token_indptr[req_idx]
                ).item()
                req_num_past_tokens.append(num_past_tokens)
                req_kv_lens.append(num_past_tokens + req_seq_lens[-1])

            req_seq_lens_tensor = torch.tensor(req_seq_lens, dtype=torch.int32, device="cpu")
            req_num_past_tokens_tensor = torch.tensor(
                req_num_past_tokens, dtype=torch.int32, device="cpu"
            )
            req_kv_lens_tensor = torch.tensor(req_kv_lens, dtype=torch.int32, device="cpu")
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor,
                len(chunk_specs),
                num_q_tokens,
                req_num_past_tokens_tensor,
                req_kv_lens_tensor,
            )

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                for j in range(first_req_idx)
            )
            total_kv_len = sum(req_kv_lens)
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, (
            "K31Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: "
            f"{cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        )
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, (
            "K31Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: "
            f"{cu_seqlen_ke.shape[0]} != {num_q_tokens}"
        )

        return K31IndexerPrefillChunkMetadata(
            cu_seqlen_ks=maybe_pin_memory(cu_seqlen_ks).to(device, non_blocking=True),
            cu_seqlen_ke=maybe_pin_memory(cu_seqlen_ke).to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def build_indexer_params(metadata: K31DSAtrtllmAttentionMetadata) -> Optional[K31IndexerParams]:
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "index_head_dim"):
            return None

        head_dim = metadata.indexer_head_dim
        return K31IndexerParams(
            num_contexts=metadata.num_contexts,
            num_generations=metadata.num_generations,
            num_ctx_tokens=metadata.num_ctx_tokens,
            head_dim=head_dim,
            quant_block_size=metadata.indexer_quant_block_size,
            tokens_per_block=metadata._tokens_per_block,
            request_ids=metadata.request_ids,
            num_past_tokens=metadata.kv_cache_params.num_cached_tokens_per_seq,
            seq_lens=metadata.seq_lens,
        )

    @staticmethod
    def recompute_slot_mappings(
        metadata: K31DSAtrtllmAttentionMetadata, indexer_params: Optional[K31IndexerParams] = None
    ):
        """Recompute slot_mapping_fp8/scale from current block offsets.

        This is the part of prepare() that maps each new KV token to its
        flat cache position for the cache write. It is safe to call in
        isolation after a caller has swapped the active KV cache manager and
        block-offset buffers, such as during draft KV-cache replay.
        """
        if indexer_params is None:
            indexer_params = K31Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        batch_size = indexer_params.batch_size
        tokens_per_block = indexer_params.tokens_per_block
        head_dim = indexer_params.head_dim
        new_kv_tokens = indexer_params.new_kv_tokens
        total_new_kv_tokens = new_kv_tokens.sum().item()

        # Compute global positions for all kv tokens in the batch (fully vectorized)
        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device="cpu"), new_kv_tokens
        )
        # Vectorized token_offsets: arange(total) - cumulative start per request
        cu_new_kv = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
        cu_new_kv[1:] = new_kv_tokens.to(torch.int64).cumsum(0)
        token_offsets = torch.arange(
            total_new_kv_tokens, dtype=torch.int64, device="cpu"
        ) - cu_new_kv[:-1].repeat_interleave(new_kv_tokens)
        global_positions = indexer_params.cached_kv_tokens[req_indices] + token_offsets

        # Block indices/pos for all kv tokens in the batch
        block_indices_in_seq = global_positions // tokens_per_block

        max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            indexer_params.quant_block_size,
        )

        metadata.host_slot_mapping_fp8[:total_new_kv_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_new_kv_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_new_kv_tokens], non_blocking=True
        )
        metadata.slot_mapping_scale[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_new_kv_tokens], non_blocking=True
        )

    @staticmethod
    def recompute_context_kv_gather_mappings(
        metadata: K31DSAtrtllmAttentionMetadata,
        indexer_params: Optional[K31IndexerParams] = None,
    ) -> None:
        """Recompute context KV-cache gather mappings for the active cache manager."""
        if not metadata.enable_context_mla_with_cached_kv:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        if indexer_params is None:
            indexer_params = K31Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        num_contexts = indexer_params.num_contexts
        if num_contexts == 0:
            # Generation does not consume full-KV gather mappings. In
            # particular, avoid allocating temporary zero-length tensors
            # while capturing generation CUDA graphs.
            return

        if metadata.skip_indexer_for_ctx_reqs:
            # The dense short-context path does not gather from the indexer K
            # cache, so full-KV mappings would be allocated but never consumed.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        cached_kv_tokens = indexer_params.cached_kv_tokens[:num_contexts]
        if cached_kv_tokens.count_nonzero().item() == 0:
            # Without cached tokens, the regular mappings cover every context
            # KV token in the same order and remain equivalent when
            # on_update_kv_lens() refreshes them in place.
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale
            return

        # Built on the device from the device copy of the block-offset table, which
        # prepare_for_indexer_k_cache() refreshed and committed on this stream: the host
        # work is a few launches per request rather than arithmetic over every cached and
        # new token plus an H2D copy of the result.
        block_offsets = metadata.indexer_k_cache_block_offsets
        kv_positions = []
        req_indices = []
        for req_idx, kv_len in enumerate(indexer_params.kv_lens[:num_contexts].tolist()):
            kv_positions.append(
                torch.arange(kv_len, dtype=torch.int64, device=block_offsets.device)
            )
            req_indices.append(
                torch.full((kv_len,), req_idx, dtype=torch.int64, device=block_offsets.device)
            )

        (
            metadata.slot_mapping_fp8_fullkv,
            metadata.slot_mapping_scale_fullkv,
        ) = _compute_slot_mappings(
            torch.cat(kv_positions),
            block_offsets,
            torch.cat(req_indices),
            indexer_params.head_dim,
            indexer_params.tokens_per_block,
            indexer_params.quant_block_size,
        )

    @staticmethod
    def prepare_for_chunked_prefill(
        metadata: K31DSAtrtllmAttentionMetadata, indexer_params: K31IndexerParams
    ):
        """
        Prepare indexer for the chunked prefill.
        """
        num_contexts = indexer_params.num_contexts
        seq_lens = indexer_params.seq_lens
        metadata.indexer_prefill_q_split = None

        # When MLA chunked prefill is active, it already bounds the new tokens per
        # request, so requests are not split by length here. From
        # q_split_threshold context query tokens on, the batch's queries are
        # split evenly across TP ranks and each rank plans chunks for its share
        # only. Within that share, requests are grouped so that a chunk's packed
        # logits stay bounded: with cached KV, a request's KV can be far longer
        # than its new tokens.
        has_mla_chunked_prefill = (
            metadata.enable_context_mla_with_cached_kv and metadata.runtime_features.chunked_prefill
        )
        if has_mla_chunked_prefill:
            q_lens = seq_lens[:num_contexts].tolist()
            cached_indptr = metadata.host_ctx_cached_token_indptr[: num_contexts + 1].tolist()
            kv_lens = [
                cached_indptr[i + 1] - cached_indptr[i] + q_lens[i] for i in range(num_contexts)
            ]
            q_begin, q_end, q_sizes = _split_prefill_queries(
                sum(q_lens), metadata.mapping, _prefill_q_split_threshold(metadata)
            )
            if q_sizes is not None:
                metadata.indexer_prefill_q_split = (q_begin, q_end, q_sizes)
            metadata.indexer_prefill_chunks = [
                K31Indexer.prepare_one_prefill_chunk(metadata, chunk_specs)
                for chunk_specs in plan_mla_chunked_prefill_chunks(
                    q_lens, kv_lens, _packed_logits_elem_budget(metadata), q_begin, q_end
                )
            ]
        else:
            # Use indexer's chunking to avoid quadratic indexer MQA logits
            # computation for long sequences.
            # This is only used when MLA chunked prefill is not enabled.
            # Adapt chunk size to the batch's largest index-key length (see
            # select_indexer_chunk_size).
            max_kv_len = (
                int(indexer_params.kv_lens[:num_contexts].max().item()) if num_contexts > 0 else 0
            )
            effective_chunk_size = select_indexer_chunk_size(
                metadata.indexer_max_chunk_size, max_kv_len
            )
            chunk_groups = split_prefill_chunks(
                seq_lens[:num_contexts],
                effective_chunk_size,
                start_idx=0,
            )

            if len(chunk_groups) > 1 or metadata.enable_context_mla_with_cached_kv:
                metadata.indexer_prefill_chunks = [
                    K31Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                    for chunk_specs in chunk_groups
                ]
            else:
                metadata.indexer_prefill_chunks = None

        # Chunked prefill and KV-cache reuse require gather mappings that cover
        # all cached and newly appended indexer tokens.
        K31Indexer.recompute_context_kv_gather_mappings(metadata, indexer_params)

    @staticmethod
    def prepare_scheduler_metadata(metadata: K31DSAtrtllmAttentionMetadata):
        """Per-SM schedules of the paged MQA logits kernels, written into the metadata's persistent buffers."""
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        gen_seq_lens = metadata.get_indexer_kv_lens(
            metadata.kv_lens_cuda_runtime[num_contexts : num_contexts + num_generations]
        )
        metadata.gen_indexer_kv_lens_cuda_runtime = gen_seq_lens
        scheduler_context_lens = metadata.update_indexer_kv_lens_2d(gen_seq_lens)
        get_paged_mqa_logits_metadata(
            gen_seq_lens.view(-1, 1),
            _DG_SCHEDULE_BLOCK_KV,
            metadata.num_sms,
            metadata.scheduler_metadata_buffer,
        )
        if metadata.max_draft_tokens > 0:
            get_paged_mqa_logits_metadata(
                scheduler_context_lens.contiguous(),
                _DG_SCHEDULE_BLOCK_KV,
                metadata.num_sms,
                metadata.scheduler_metadata_buffer_full_next_n,
            )

    @staticmethod
    def prepare(metadata: K31DSAtrtllmAttentionMetadata) -> None:
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """

        # Skip indexer preparation for a KV cache manager without the indexer K cache (a drafter's).
        indexer_params = K31Indexer.build_indexer_params(metadata)
        if indexer_params is None:
            return

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        # Store the KV token count of the context requests
        metadata.num_ctx_kv_tokens = indexer_params.new_kv_tokens[:num_contexts].sum().item()

        # Prepare for update_k_cache
        K31Indexer.recompute_slot_mappings(metadata, indexer_params)

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            host_num_past_tokens = indexer_params.cached_kv_tokens[:num_contexts]
            host_kv_lens = indexer_params.kv_lens[:num_contexts]
            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens,
                num_contexts,
                num_ctx_tokens,
                host_num_past_tokens,
                host_kv_lens,
            )

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ks), non_blocking=True
            )
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ke), non_blocking=True
            )
            K31Indexer.prepare_for_chunked_prefill(metadata, indexer_params)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            # from the device KV lengths.
            K31Indexer.prepare_scheduler_metadata(metadata)

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: K31DSAtrtllmAttentionMetadata
    ) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        if k_scale.element_size() == 1:
            # The op expects 4 byte elements.
            k_scale = k_scale.view(torch.int32)
        indexer_k_cache_scatter_op(
            k_fp8,
            k_scale,
            k_cache,
            metadata.slot_mapping_fp8,
            metadata.slot_mapping_scale,
            num_tokens,
        )

    def forward_prepared(
        self,
        q_data: torch.Tensor,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        tier_rows: Optional[int] = None,
    ) -> torch.Tensor:
        """Run shared dense MQA and prefill Top-K on prequantized inputs.

        Row bounds index the shared K matrix. Output preserves DSA's row-local
        unsorted offsets. Only those row windows are read by Top-K, so the
        logits outside them are not cleaned. ``tier_rows``: the
        GVR Top-K takes the plan of a ``tier_rows``-row tile, for
        a slice of that tile (the row order the whole tile gives).
        """
        if output_indices.shape != (q_data.shape[0], self.index_topk):
            raise ValueError("Prepared output must match query rows and configured Top-K")
        if q_data.shape[0] == 0 or k_data.shape[0] == 0:
            return output_indices.fill_(-1)
        logits = self._call_mqa_logits(q_data, k_data, k_scale, weights, row_starts, row_ends)
        # The deterministic GVR on every tile wider than the Top-K (SM 10.3), else the CUDA radix.
        if self.gvr_topk(logits.device):
            indexer_topk_prefill_gvr(
                logits, row_starts, row_ends, output_indices, self.index_topk, tier_rows=tier_rows
            )
        else:
            indexer_topk_prefill(logits, row_starts, row_ends, output_indices, self.index_topk)
        return output_indices

    def _prefill_key_gather(self, metadata: K31DSAtrtllmAttentionMetadata):
        """``gather(k_token_start, num_keys)``: a prefill chunk's index keys and key scales from this
        layer's index-K cache. The paged gather kernel reads the manager's page-strided view
        directly."""
        k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        head_bytes = self.head_dim

        def gather(k_token_start: int, num_keys: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return k31_gather_k_cache_paged(
                k_cache_4d,
                metadata.slot_mapping_fp8_fullkv,
                metadata.slot_mapping_scale_fullkv,
                k_token_start,
                num_keys,
                head_bytes,
            )

        return gather

    def prefill_q_tile(self, q_rows: int, num_k_tokens: int) -> int:
        """Query rows per fp8_mqa_logits call of a prefill chunk (logits element budget)."""
        return max(1, min(q_rows, _INDEXER_MQA_LOGITS_ELEM_BUDGET // max(1, num_k_tokens)))

    def _call_mqa_logits(
        self,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        """DeepGEMM fp8_mqa_logits over the FP8 index keys, without its clean-logits pass (the -inf fill
        outside each row's [ks, ke) window): the prefill Top-K kernels read only inside the window."""
        return fp8_mqa_logits(
            q_fp8, k_fp8, k_scale.reshape(-1), weights, cu_seqlen_ks, cu_seqlen_ke
        )

    def _call_paged_mqa_logits(
        self,
        q_decode: torch.Tensor,
        k_cache: torch.Tensor,
        weights_decode: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        scheduler_metadata_buffer: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """DeepGEMM fp8_paged_mqa_logits over the FP8 index-K cache."""
        return fp8_paged_mqa_logits(
            q_decode,
            k_cache,
            weights_decode,
            context_lens,
            block_table,
            scheduler_metadata_buffer,
            int(max_seq_len),
        )

    def sparse_attn_indexer(
        self,
        metadata: K31DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for one phase or the full batch.

        When ``is_generation`` is ``None``, the inputs contain the full mixed
        batch. Otherwise they contain only the selected context or generation
        phase.
        """
        # FP8 index keys: one 4-byte scale word per token at index_head_dim=128.
        assert (
            metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size == 128
        ), (
            "Unexpected quant_block_size "
            f"{metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        )
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        num_gen_tokens = num_tokens - num_ctx_tokens
        if is_generation is None:
            has_prefill = num_contexts > 0
            has_decode = num_generations > 0
            token_offset = num_ctx_tokens
            cache_name = "indexer_topk_out_buffer"
        else:
            has_prefill = not is_generation and num_contexts > 0
            has_decode = is_generation and num_generations > 0
            token_offset = 0
            expected_tokens = num_gen_tokens if is_generation else num_ctx_tokens
            assert hidden_states.shape[0] == expected_tokens, (
                "Phase-specific DSA prediction received "
                f"{hidden_states.shape[0]} tokens, expected {expected_tokens}."
            )
            cache_name = (
                "indexer_topk_out_buffer_gen" if is_generation else "indexer_topk_out_buffer_ctx"
            )

        topk_indices_buffer = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (hidden_states.shape[0], self.index_topk),
            cache_name=cache_name,
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )

        # Prefill share (DEP perf target): this step's long-prefix chunks taken for scoring across the
        # attention-DP ranks (every rank runs this context indexer then; see dsa_prefill_share).
        share_plan = (
            getattr(metadata, "indexer_share_plan", None) if is_generation is not True else None
        )
        shared_chunks = (
            frozenset(share_plan.local_chunks) if share_plan is not None else frozenset()
        )
        share_pending = (
            dsa_prefill_share.start_shared_indexer(
                self,
                metadata,
                q_fp8,
                weights,
                self._prefill_key_gather(metadata),
            )
            if share_plan is not None
            else None
        )

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                q_split_threshold = _prefill_q_split_threshold(metadata)
                # Chunks planned for this rank's share of the queries only are
                # never split again; their top-k rows are gathered after the loop.
                batch_q_split = metadata.indexer_prefill_q_split
                # A token-sharded MLA prefill (the SM 10.3 tep target's ShardQ) reads only this rank's
                # query rows: it sets this for the call and the split rows are not gathered.
                skip_q_split_gather = getattr(metadata, "indexer_skip_q_split_gather", False)
                gather_chunk_keys = self._prefill_key_gather(metadata)

                for chunk_idx, chunk in enumerate(metadata.indexer_prefill_chunks):
                    if chunk_idx in shared_chunks:
                        continue  # scored across the ranks (dsa_prefill_share)
                    # Skip chunks with no KV tokens.
                    if chunk.k_token_start >= chunk.k_token_end:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :].fill_(-1)
                        continue
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = gather_chunk_keys(
                        chunk.k_token_start, num_k_tokens
                    )

                    chunk_num_token = chunk.token_end - chunk.token_start
                    if batch_q_split is None:
                        chunk_q_start, chunk_q_end, q_sizes = _split_prefill_queries(
                            chunk_num_token, metadata.mapping, q_split_threshold
                        )
                    else:
                        chunk_q_start, chunk_q_end, q_sizes = 0, chunk_num_token, None

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    # Tile the query dimension so each fp8_mqa_logits call
                    # allocates at most [q_tile x num_k_tokens] instead of the
                    # full [local_q x num_k_tokens] (which can reach tens of GB
                    # on a long context and stall cuMemCreate under
                    # expandable_segments -> engine hang; see
                    # _INDEXER_MQA_LOGITS_ELEM_BUDGET). Results are identical:
                    # each query row's logits/top-k are independent and the KV
                    # (chunk_k_fp8) is unchanged across tiles, so the per-call
                    # allocation is the same size and the caching allocator
                    # reuses one block (peak ~= one tile, no extra sync).
                    local_q_len = chunk_q_end - chunk_q_start
                    q_tile = self.prefill_q_tile(local_q_len, num_k_tokens)
                    for c0 in range(chunk_q_start, chunk_q_end, q_tile):
                        c1 = min(c0 + q_tile, chunk_q_end)
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        self.forward_prepared(
                            q_fp8[g0:g1, ...],
                            chunk_k_fp8,
                            chunk_k_scale,
                            weights[g0:g1, ...],
                            chunk.cu_seqlen_ks[c0:c1],
                            chunk.cu_seqlen_ke[c0:c1],
                            topk_indices_buffer[g0:g1, :],
                        )

                    if q_sizes is not None and not skip_q_split_gather:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :] = (
                            allgather_module(
                                topk_indices_buffer[global_q_start:global_q_end, :],
                                metadata.mapping,
                                0,
                                q_sizes,
                            )
                        )

                if batch_q_split is not None and not skip_q_split_gather:
                    local_q_begin, local_q_end, q_sizes = batch_q_split
                    topk_indices_buffer[: sum(q_sizes), :] = allgather_module(
                        topk_indices_buffer[local_q_begin:local_q_end, :],
                        metadata.mapping,
                        0,
                        q_sizes,
                    )
            elif metadata.num_ctx_kv_tokens == 0:
                # No KV tokens: fill with -1 (no valid indices)
                topk_indices_buffer[:num_ctx_tokens, :].fill_(-1)
            else:
                # Single pass: no MLA chunked prefill and the whole context batch fits one indexer chunk
                # (prepare_for_chunked_prefill planned no chunks).
                num_ctx_kv_tokens = metadata.num_ctx_kv_tokens
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                self.forward_prepared(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_kv_tokens, ...],
                    k_scale[:num_ctx_kv_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices_buffer[:num_ctx_tokens, :],
                )

        if share_pending is not None:
            dsa_prefill_share.finish_shared_indexer(
                self, metadata, share_pending, topk_indices_buffer
            )

        if has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = metadata.topk_indices_buffer[
                :num_ctx_tokens, :
            ]

        if has_decode and not metadata.skip_indexer_for_gen_reqs:
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
            max_decode_len = gen_seq_lens.max().item()
            min_decode_len = gen_seq_lens.min().item()
            assert max_decode_len == min_decode_len, (
                "max_decode_len != min_decode_len, we need padding"
            )

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[token_offset : token_offset + num_gen_tokens, ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Blackwell's paged MQA logits take every next_n natively: [batch_size, next_n, ...].
            q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
            # 2D context_lens from the pre-allocated buffer, in q_decode's (batch, next_n) layout.
            context_lens = metadata.kv_lens_cuda_2d[:num_generations, :next_n].contiguous()
            block_table = metadata.indexer_k_cache_block_offsets[
                num_contexts : num_contexts + num_generations
            ]
            # The 2D-context_lens schedule encodes next_n (num_next_n_atoms): a verify step
            # (next_n == 1 + max_draft_tokens) and a next_n == 1 call each read the buffer built for
            # their next_n. The DSL path uses its own schedule buffer (built with num_next_n_atoms=1
            # via a (num_gen, 1) input shape) and overrides this below.
            if next_n == 1:
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
            else:
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[token_offset : token_offset + num_gen_tokens, ...]

            # Get k cache and call the paged MQA logits with prepared decode metadata.
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            indexer_max_seq_len = metadata.get_indexer_max_seq_len()

            # The attention-TP perf target's key-sharded decode Top-K (ops/attention/dsa_key_shard) on the steps
            # it serves: each TP rank scores 1/TP of the keys; the same rows as the replicated path below.
            key_shard = getattr(metadata, "_k31_key_shard", None) if self._k31_perf else None
            if key_shard is not None:
                gen_rows = topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :]
                pool_remap = (
                    self._gen_pool_remap
                    if is_generation and metadata.shared_topk_indices is None
                    else None
                )
                if key_shard.topk(
                    q_fp8[token_offset : token_offset + num_gen_tokens],
                    k_cache,
                    weights_decode,
                    gen_rows,
                    num_generations,
                    next_n,
                    pool_remap,
                ):
                    self._gen_rows_paged = pool_remap is not None
                    return topk_indices_buffer

            if self.use_cute_dsl_paged_mqa_logits:
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `K31Indexer.prepare()`
                # with a (num_gen, 1) input shape, which makes DeepGEMM's wrapper
                # compute `num_next_n_atoms = 1`. (DeepGEMM uses the same buffer
                # for its own next_n=1 kernel; DSL piggy-backs on it for all
                # real next_n values.) All next_n positions of a batch share
                # the same KV length on this path (kv_lens_cuda_2d broadcasts),
                # so passing the 1D contiguous kv_lens slice for context_lens
                # avoids materializing a 2D contiguous tensor per call.
                dsl_context_lens = metadata.gen_indexer_kv_lens_cuda_runtime
                assert dsl_context_lens is not None
                # Wave-aware atom-split: the picker in `_pick_dsl_expand` caches
                # (factor, atom) on metadata with invariant
                # `factor * atom == 1 + max_draft_tokens` (the target/verify-time
                # next_n). The reshape `(num_gen, next_n, ...) ->
                # (num_gen*factor, atom, ...)` is only valid when the caller
                # actually supplies next_n == factor * atom tokens; other calls
                # (next_n=1) take the kernel-native path.
                dsl_atom_split = (
                    metadata.dsl_expand_factor > 1
                    and next_n == metadata.dsl_expand_factor * metadata.dsl_atom
                )
                # In-kernel uneven split (next_n with no exact atom divisor):
                # native Q / context_lens / block_table, schedule over the
                # staggered per-atom rows. Same full-window gate as above.
                dsl_kernel_split = (
                    metadata.dsl_kernel_atoms > 1 and next_n == metadata.dsl_kernel_next_n
                )
                dsl_split_kwargs: dict = {}
                if dsl_kernel_split:
                    dsl_split_kwargs = dict(
                        num_next_n_atoms=metadata.dsl_kernel_atoms, atom=metadata.dsl_atom
                    )
                # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                # Atom-split raises SM utilization for small batches at the
                # cost of factor x KV HBM re-reads.
                dsl_q = q_decode
                fp8_ctx_lens = dsl_context_lens
                fp8_block_table = block_table
                fp8_schedule_meta = metadata.scheduler_metadata_buffer
                if dsl_atom_split:
                    factor = metadata.dsl_expand_factor
                    atom = metadata.dsl_atom
                    exp_B = num_generations * factor
                    dsl_q = q_decode.reshape(exp_B, atom, self.n_heads, q_decode.shape[-1])
                    fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                    fp8_block_table = metadata.block_table_expanded[:exp_B]
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                elif dsl_kernel_split:
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                # The logits land in the persistent arena buffer the registered op also writes
                # (stable address across CUDA-graph replays), sliced to the kernel's row stride.
                logits_out = get_memory_buffers().get_buffer(
                    [dsl_q.shape[0] * dsl_q.shape[1], logits_row_stride(indexer_max_seq_len)],
                    torch.float32,
                    buffer_name="cute_dsl_mqa_logits",
                    reserve_buffer=torch.cuda.is_current_stream_capturing(),
                )[:, :indexer_max_seq_len]
                # The perf targets launch a native tile as a programmatic dependent launch.
                native_tile = not dsl_atom_split and not dsl_kernel_split
                logits_op = (
                    cute_dsl_fp8_paged_mqa_logits_pdl
                    if self._k31_perf and self._k31_logits_pdl and native_tile
                    else cute_dsl_fp8_paged_mqa_logits
                )
                logits_decode = logits_op(
                    dsl_q,
                    k_cache,
                    weights_decode,
                    fp8_ctx_lens,
                    fp8_block_table,
                    fp8_schedule_meta,
                    indexer_max_seq_len,
                    dsl_split_kwargs.get("num_next_n_atoms", 1),
                    dsl_split_kwargs.get("atom"),
                    logits_out,
                )
            else:
                logits_decode = self._call_paged_mqa_logits(
                    q_decode,
                    k_cache,
                    weights_decode,
                    context_lens,
                    block_table,
                    scheduler_metadata_buffer,
                    indexer_max_seq_len,
                )

            # Native uses logical lengths; radix uses score-column lengths.
            gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                num_contexts : num_contexts + num_generations
            ]
            assert metadata.gen_indexer_kv_lens_cuda_runtime is not None
            assert metadata.radix_aux_indices is not None
            assert metadata.radix_aux_logits is not None
            if (
                self.gvr_topk(logits_decode.device)
                and logits_decode.dtype == torch.float32
                and (logits_decode.shape[0] > 1 or logits_decode.shape[1] % 4 == 0)
            ):
                # Deterministic GVR Top-K for uniform next_n rows (the radix's set on tie-free
                # rows, the smaller columns on ties). With the generation pool remap the caller
                # armed (gen-only call, no shared Top-K), the rows come out as ordered pool rows.
                gen_rows = topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :]
                pool_remap = self._gen_pool_remap
                if (
                    pool_remap is not None
                    and is_generation
                    and metadata.shared_topk_indices is None
                ):
                    indexer_topk_decode_gvr_paged(
                        logits_decode,
                        gen_kv_lens_cuda,
                        gen_rows,
                        next_n,
                        indexer_max_seq_len,
                        *pool_remap,
                    )
                    self._gen_rows_paged = True
                else:
                    indexer_topk_decode_gvr(
                        logits_decode, gen_kv_lens_cuda, gen_rows, next_n, indexer_max_seq_len
                    )
            else:
                # The CUDA radix decode Top-K (architectures other than SM 10.3), with the
                # split-work workspace the radix kernel takes for fp32 scores.
                radix_aux_indices = radix_aux_logits = None
                if logits_decode.dtype == torch.float32:
                    rows = logits_decode.shape[0]
                    radix_aux_indices = metadata.radix_aux_indices[
                        :rows, :_MAX_RADIX_BLOCKS_PER_ROW, : self.index_topk
                    ]
                    radix_aux_logits = metadata.radix_aux_logits[
                        :rows, :_MAX_RADIX_BLOCKS_PER_ROW, : self.index_topk
                    ]
                indexer_topk_decode(
                    logits_decode,
                    gen_kv_lens_cuda,
                    topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                    next_n,
                    self.index_topk,
                    compress_ratio=1,
                    radix_aux_indices=radix_aux_indices,
                    radix_aux_logits=radix_aux_logits,
                    row_kv_lens=None,
                )

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
            )

        return topk_indices_buffer

    def _project_q_norm_k(self, qr: torch.Tensor, indexer_k: torch.Tensor):
        """Project Q, normalize K, and split each head in two halves (position-free: nothing rotates)."""
        # wq_b: the cuBLAS GEMM.
        q = self.wq_b(qr)
        # k_norm with one fixed reduction order (the same bits on every rank).
        k = indexer_k_norm(self, indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        # The fused cat+quantize kernels take two non-empty halves of each head.
        half = self.head_dim // 2
        q_pe, q_nope = q.split([half, self.head_dim - half], dim=-1)
        k_pe, k_nope = k.split([half, self.head_dim - half], dim=-1)
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate and quantize for Q or K: fused cat + FP8 quantize via CUDA kernel."""
        fp8_out, scale = fused_cat_fp8(qk_pe, qk_nope)  # UE8M0 scales
        return fp8_out, scale

    def pre_indexer_proj(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs the projection GEMMs, the key norm, FP8 quantize, and
        weight scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp8, k_fp8, k_scale, weights); the query scales are folded into ``weights``.
        """
        indexer_k, weights = self._project_k_and_weights(hidden_states)
        return self._finish_pre_indexer_proj(qr, indexer_k, weights)

    def _project_k_and_weights(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused wk / weights_proj projection; returns (indexer_k, weights)."""
        assert self._fused_wk_wp_weight is not None, (
            "cache_derived_state() must be called before pre_indexer_proj()"
        )
        # hidden_states and _fused_wk_wp_weight are both the model dtype: one
        # native GEMM (bf16_linear without cuBLAS) with no fp32 cast around it.
        fused_out = bf16_linear(hidden_states, self._fused_wk_wp_weight, False)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads], dim=-1)
        return indexer_k, weights

    def _finish_pre_indexer_proj(
        self,
        qr: torch.Tensor,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K norm, quantization and weight scaling after the wk / weights_proj projection."""
        q_pe, q_nope, k_pe, k_nope = self._project_q_norm_k(qr, indexer_k)
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)
        weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights

    def _weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def forward_from_projected(
        self,
        metadata: K31DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run sparse prediction from graph-captured indexer projections.

        Projected inputs cover the full batch. Phase-specific prediction slices
        Q-side inputs here while retaining the full-batch K inputs needed by
        context prediction.
        """
        if is_generation is None:
            phase_start = 0
            phase_end = metadata.num_tokens
        elif is_generation:
            phase_start = metadata.num_ctx_tokens
            phase_end = metadata.num_tokens
        else:
            phase_start = 0
            phase_end = metadata.num_ctx_tokens

        q_fp8, k_fp8, k_scale, weights = indexer_intermediates
        q_fp8_phase = q_fp8[phase_start:phase_end]
        weights_phase = weights[phase_start:phase_end]

        return self.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8_phase,
            k_fp8,
            k_scale,
            weights_phase,
            is_generation=is_generation,
        )
