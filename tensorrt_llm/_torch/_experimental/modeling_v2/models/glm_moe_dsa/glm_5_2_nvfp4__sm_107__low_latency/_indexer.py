# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-owned FP8 indexer projections, prefill scoring, and TopK selection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAMetadata,
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_elem_budget,
    use_self_sampling_gvr,
)
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, maybe_compile
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig


def _dsa_state(metadata: DSAtrtllmAttentionMetadata) -> DSAMetadata:
    """The DSA extension of ``metadata``, which holds the indexer and pool-view state."""
    return metadata.ext(DSAMetadata)


_INDEXER_MQA_LOGITS_ELEM_BUDGET = get_indexer_mqa_logits_elem_budget()


def use_cute_dsl_paged_mqa_logits(requested: bool) -> bool:
    """Whether the CuTe DSL paged MQA logits kernel runs instead of DeepGEMM.

    The indexer and the metadata must both use this: the metadata builds
    DSL-only buffers (atom-split expansion, padded logits row stride) and
    skips the sm90 MTP expansion off the same decision. is_sm_100f() is the
    custom op's own precondition.
    """
    return requested and IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses the pool-view values the DSA metadata caches (DSAMetadata._ensure_pool_view_cached)
    to avoid redundant Python/CUDA overhead across layers.
    """
    dsa = _dsa_state(attn_metadata)
    assert topk_indices.dtype == torch.int32

    dsa._ensure_pool_view_cached()
    page_index_scale, layer_offset = (
        attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(layer_idx)
    )

    if is_generation:
        block_table = dsa._cached_block_table_gen
        req_idx = dsa._cached_req_idx_gen
    else:
        block_table = dsa._cached_block_table_ctx
        req_idx = dsa._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        dsa._cached_tokens_per_block,
        topk_indices.shape[1],
        page_index_scale * dsa._cached_tokens_per_block,
        layer_offset,
    )

    return global_indices, dsa._cached_pool_view


def transform_local_topk_and_prepare_pool_view_grouped(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_ids: torch.Tensor,
    page_index_scale: int,
    is_generation: bool = False,
) -> torch.Tensor:
    """Grouped (cross-layer fan-out) variant of the local-topk → global-index
    remap (``TRTLLM_DISABLE_DSA_GROUP_REMAP``).

    Computes the global-index output for a whole full+shared indexer group in a
    single launch. All members of a group share the same top-k selection,
    ``req_idx`` and ``block_table`` (constant across layers within a forward);
    only the additive ``layer_offset * tokens_per_block`` term differs per
    member. ``layer_ids`` (int32 CUDA tensor of length ``group_size``) carries
    each member's layer offset; ``page_index_scale`` is the (uniform) primary-
    pool page scale for the group.

    Returns a tensor of shape ``[group_size, num_tokens, topk]``; slice ``[z]``
    is bit-identical to ``transform_local_topk_and_prepare_pool_view`` for the
    member with offset ``layer_ids[z]``. MLA-only path.
    """
    dsa = _dsa_state(attn_metadata)
    assert topk_indices.dtype == torch.int32

    dsa._ensure_pool_view_cached()

    if is_generation:
        block_table = dsa._cached_block_table_gen
        req_idx = dsa._cached_req_idx_gen
    else:
        block_table = dsa._cached_block_table_ctx
        req_idx = dsa._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global_grouped(
        req_idx,
        block_table,
        topk_indices,
        dsa._cached_tokens_per_block,
        topk_indices.shape[1],
        page_index_scale * dsa._cached_tokens_per_block,
        layer_ids,
    )

    return global_indices


def _split_prefill_queries(
    count: int, mapping: Optional[Mapping], threshold: int
) -> Tuple[int, int, Optional[List[int]]]:
    """Return local query bounds and optional sizes for a TP prefill gather."""
    if (
        threshold < 0
        or count < threshold
        or mapping is None
        or mapping.enable_attention_dp
        or mapping.tp_size <= 1
    ):
        return 0, count, None
    size, rank = mapping.tp_size, mapping.tp_rank
    sizes = [(r + 1) * count // size - r * count // size for r in range(size)]
    return count * rank // size, count * (rank + 1) // size, sizes


def _prefill_q_split_threshold(metadata: DSAtrtllmAttentionMetadata) -> int:
    """Query-token count from which prefill indexer queries are split across TP ranks."""
    sparse_metadata_params = metadata.sparse_metadata_params
    return sparse_metadata_params.q_split_threshold if sparse_metadata_params is not None else 8192


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor, s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    """Cast hidden states to float32 for TF32 GEMM computation."""
    return hidden_states.float()


@contextmanager
def _tf32_matmul_enabled():
    """Temporarily enable TF32 tensor cores for FP32 matmul in this scope.

    Forces PyTorch/cuBLASLt to use CUBLAS_COMPUTE_32F_FAST_TF32, which
    guarantees TF32 tensor cores. Plain CUBLAS_COMPUTE_32F (used by
    torch.ops.trtllm.cublas_mm) falls back to SIMT SGEMM on CUDA cores
    based on cuBLASLt heuristics for small M.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class Glm52Indexer(nn.Module):
    """GLM FP8 sparse-attention projections and top-k selection."""

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DSAParams,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        sm_version = get_sm_version() if torch.cuda.is_available() else 0
        if mla_params is None or pos_embd_params is None:
            raise ValueError("Projecting indexers require MLA and positional parameters")
        self.hidden_size = mla_params.hidden_size if mla_params is not None else None
        self.q_lora_rank = mla_params.q_lora_rank if mla_params is not None else None
        self.rope_dim = mla_params.qk_rope_head_dim if mla_params is not None else 0
        self.n_heads = sparse_params.index_n_heads  # 64
        self.head_dim = sparse_params.index_head_dim  # 128
        self.index_topk = sparse_params.index_topk  # 2048
        self.layer_idx = layer_idx
        if compress_ratio != 1:
            raise ValueError("GLM indexer keys are not compressed")
        self.compress_ratio = compress_ratio

        self.wq_b = self.wk = self.k_norm = self.weights_proj = self.rotary_emb = None
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None
        self._indexer_bf16 = False
        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        # When TRTLLM_DSA_INDEXER_BF16=1, run the fused wk + weights_proj
        # projection in the model dtype instead of fp32. The fp32 path has no
        # native GEMM on recent architectures (it falls back to a TF32 tensor-
        # core kernel) and requires bf16<->fp32 casts around it; the model-dtype
        # path runs a single native GEMM with no casts. The default stays fp32
        # because the indexer top-k selection is precision sensitive.
        self._indexer_bf16 = os.environ.get("TRTLLM_DSA_INDEXER_BF16", "0") == "1"
        _wk_wp_dtype = dtype if self._indexer_bf16 else torch.float32
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=_wk_wp_dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=_wk_wp_dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )

        # Fused wk + weights_proj weight for a single F.linear GEMM. fp32 by
        # default (TF32 tensor cores on Ampere+); model dtype when
        # TRTLLM_DSA_INDEXER_BF16 is enabled.
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

        indexer_rope_interleave = sparse_params.indexer_rope_interleave
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=not indexer_rope_interleave,
        )

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        # Q and K use the same FP8 representation as the indexer cache.
        if sparse_params.indexer_k_dtype != "fp8":
            raise ValueError("The GLM Rubin target requires an FP8 indexer cache")
        self.use_fp4 = False
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        # Fork/join events for the aux-stream prev_topk write-back.
        self.prev_topk_copy_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._prev_topk_copy_pending = False
        self.use_cute_dsl_topk = sparse_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        self.use_cute_dsl_paged_mqa_logits = use_cute_dsl_paged_mqa_logits(
            sparse_params.use_cute_dsl_paged_mqa_logits
        )
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = sparse_params.enable_heuristic_topk and sm_version >= 100
        # Two-level GVR dispatch: enable_heuristic_topk selects the GVR
        # family over the exact radix path; use_self_sampling_topk (default
        # True) selects the hint-free self-sampling engine over the
        # temporal-hint engines. Tuning is frozen from indexer_max_seq_len
        # at capture time, so the launch is CUDA-graph-replay safe. The
        # TopK module's hardware-format gate falls back to the exact
        # insertion/radix path with a one-time warning; contract violations
        # inside the engine raise.
        self._use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self._enable_heuristic_topk,
            use_self_sampling_topk=sparse_params.use_self_sampling_topk,
            index_topk=sparse_params.index_topk,
            compress_ratio=compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=sm_version,
        )
        if os.environ.get("TRTLLM_GVR_SELF_SAMPLING") is not None:
            logger.warning_once(
                "TRTLLM_GVR_SELF_SAMPLING is retired and ignored: the "
                "self-sampling GVR engine is selected by the "
                "use_self_sampling_topk sparse-attention config field "
                "(default True) when enable_heuristic_topk is set.",
                key="gvr_self_sampling_env_retired",
            )
        if (
            self._enable_heuristic_topk
            and sparse_params.use_self_sampling_topk
            and not self._use_self_sampling_topk
        ):
            logger.warning_once(
                "use_self_sampling_topk=True but the self-sampling GVR "
                "prerequisites are not met "
                f"(cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                f"sm={sm_version}, "
                f"index_topk={sparse_params.index_topk}, "
                f"compress_ratio={compress_ratio}); falling back to the "
                "temporal GVR path (exact radix when the DSL engine is "
                "unavailable).",
                key="gvr_self_sampling_prereq_fallback",
            )
        self.mtp_index_share = sparse_params.mtp_index_share

        if self._enable_heuristic_topk:
            logger.warning_once(
                "enable_heuristic_topk=True but the DSL GVR engine is "
                f"unavailable (cutlass_dsl={IS_CUTLASS_DSL_AVAILABLE}, "
                f"sm={get_sm_version()}); using the exact radix decode "
                "top-K instead.",
                key="gvr_prereq_radix_fallback",
            )
        decode_top_k_implementation = (
            TopKImplementation.CUTE_DSL_RADIX
            if self.use_cute_dsl_topk
            else TopKImplementation.CUDA_RADIX
        )
        # Prefill uses the self-sampling engine on exactly the layers where the
        # decode dispatch picks it; the temporal-hint engine has no prefill form.
        prefill_top_k_implementation = (
            TopKImplementation.CUTE_DSL_GVR
            if (
                decode_top_k_implementation == TopKImplementation.CUTE_DSL_GVR
                and self._use_self_sampling_topk
            )
            else TopKImplementation.CUDA_RADIX
        )
        self.top_k = TopK(
            self.index_topk,
            prefill_implementation=prefill_top_k_implementation,
            decode_implementation=decode_top_k_implementation,
            compress_ratio=self.compress_ratio,
            gvr_self_sampling=self._use_self_sampling_topk,
            gvr_deterministic=sparse_params.gvr_deterministic_topk,
        )
        self.use_gvr_emission = False

        # Fused wk + weights_proj weight for single FP32 cuBLAS GEMM
        # (populated in cache_derived_state; maps to TF32 tensor cores on Ampere+)
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

    def cache_derived_state(self) -> None:
        """Fuse wk and weights_proj for F.linear with TF32 tensor cores on Ampere+."""
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
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
        torch.ops.trtllm.indexer_k_cache_scatter_op(
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
        q_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run shared dense MQA and prefill Top-K on prequantized inputs.

        Row bounds index the shared K matrix. Output preserves DSA's row-local
        unsorted offsets. Only those row windows are read by Top-K, so the
        existing no-clean logits fast path remains valid.
        """
        if output_indices.shape != (q_data.shape[0], self.index_topk):
            raise ValueError("Prepared output must match query rows and configured Top-K")
        if q_data.shape[0] == 0 or k_data.shape[0] == 0:
            return output_indices.fill_(-1)
        logits = self._call_mqa_logits(
            q_data,
            k_data,
            k_scale,
            weights,
            row_starts,
            row_ends,
            q_scale,
            clean_logits=False,
        )
        return self.top_k(
            logits,
            output_indices,
            is_prefill=True,
            row_starts=row_starts,
            row_ends=row_ends,
        )

    def _call_mqa_logits(
        self,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        clean_logits: bool = True,
    ) -> torch.Tensor:
        """Score FP8 index queries against the gathered FP8 keys.

        clean_logits=False skips DeepGEMM's smxx_clean_logits pass that fills
        everything outside each row's [ks, ke) window with -inf. Safe only
        when the consumer never reads outside that window (the custom
        indexer_topk_prefill kernel); the torch topk fallback scans the full
        padded row and needs the fill.
        """
        return fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scale.reshape(-1)),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
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
        q_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Score FP8 index queries against the paged FP8 keys."""
        return fp8_paged_mqa_logits(
            q_decode,
            k_cache,
            weights_decode,
            context_lens,
            block_table,
            scheduler_metadata_buffer,
            max_seq_len,
        )

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for one phase or the full batch.

        When ``is_generation`` is ``None``, the inputs contain the full mixed
        batch. Otherwise they contain only the selected context or generation
        phase. Query scales are already folded into ``weights``.

        """
        assert (
            metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size == 128
        )
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        num_gen_tokens = num_tokens - num_ctx_tokens
        gvr_prior_indices = None
        if self.top_k.needs_gvr_prior:
            assert metadata.gvr_prior_indices is not None
            local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
            gvr_prior_indices = metadata.gvr_prior_indices[local_layer]
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

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                q_split_threshold = _prefill_q_split_threshold(metadata)
                # Chunks planned for this rank's share of the queries only are
                # never split again; their top-k rows are gathered after the loop.
                batch_q_split = metadata.indexer_prefill_q_split
                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
                gather_head_dim = self.head_dim

                for chunk in metadata.indexer_prefill_chunks:
                    # Skip chunks with no compressed KV tokens (e.g., warmup
                    # sequences shorter than compress_ratio produce zero KV).
                    if chunk.k_token_start >= chunk.k_token_end:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :].fill_(-1)
                        continue
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                        k_cache_4d,
                        metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        num_k_tokens,
                        gather_head_dim,
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
                    budget = _INDEXER_MQA_LOGITS_ELEM_BUDGET
                    q_tile = max(1, min(local_q_len, budget // max(1, num_k_tokens)))
                    for tile_off in range(0, local_q_len, q_tile):
                        c0 = chunk_q_start + tile_off
                        c1 = min(c0 + q_tile, chunk_q_end)
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        tile_q_scale = None
                        self.forward_prepared(
                            q_fp8[g0:g1, ...],
                            chunk_k_fp8,
                            chunk_k_scale,
                            weights[g0:g1, ...],
                            chunk.cu_seqlen_ks[c0:c1],
                            chunk.cu_seqlen_ke[c0:c1],
                            topk_indices_buffer[g0:g1, :],
                            tile_q_scale,
                        )

                    if q_sizes is not None:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :] = allgather(
                            topk_indices_buffer[global_q_start:global_q_end, :],
                            metadata.mapping,
                            dim=0,
                            sizes=q_sizes,
                        )

                if batch_q_split is not None:
                    local_q_begin, local_q_end, q_sizes = batch_q_split
                    topk_indices_buffer[: sum(q_sizes), :] = allgather(
                        topk_indices_buffer[local_q_begin:local_q_end, :],
                        metadata.mapping,
                        dim=0,
                        sizes=q_sizes,
                    )
            elif metadata.num_ctx_kv_tokens == 0:
                # No compressed KV tokens — fill with -1 (no valid indices)
                topk_indices_buffer[:num_ctx_tokens, :].fill_(-1)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                num_ctx_kv_tokens = metadata.num_ctx_kv_tokens
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = None
                self.forward_prepared(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_kv_tokens, ...],
                    k_scale[:num_ctx_kv_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices_buffer[:num_ctx_tokens, :],
                    ctx_q_scale,
                )
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = metadata.topk_indices_buffer[
                :num_ctx_tokens, :
            ]

        # Dense skip outputs remain valid priors if a later step enters GVR.
        if has_prefill:
            # Device lengths: gathering the CUDA selections with the host
            # seq_lens would force a synchronous H2D copy every layer.
            self.top_k.update_gvr_prior_from_prefill(
                topk_indices_buffer[:num_ctx_tokens],
                metadata.seq_lens_cuda[:num_contexts],
                gvr_prior_indices,
                request_offset=num_generations,
            )

        reuse_topk = (
            self.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and metadata.shared_topk_indices is not None
        )

        if has_decode and not metadata.skip_indexer_for_gen_reqs and reuse_topk:
            self._copy_reused_topk(
                metadata,
                topk_indices_buffer,
                token_offset,
                num_gen_tokens,
                num_generations,
                has_prefill,
                gvr_prior_indices,
            )
        elif (
            has_decode
            and not metadata.skip_indexer_for_gen_reqs
            and self._fill_decode_topk(
                metadata, q_fp8, weights, topk_indices_buffer, token_offset, num_gen_tokens
            )
        ):
            # A subclass filled the generation rows.
            pass
        elif has_decode and not metadata.skip_indexer_for_gen_reqs:
            # SimpleNamespace doubles in test_indexer_gvr_prior reach this path without the field.
            if getattr(metadata, "indexer_schedule_stale", False):
                raise RuntimeError(
                    "DSA indexer: the MQA-logits schedule is stale (its rebuild was skipped) "
                    "but this pass computes TopK "
                    f"(in_mtp_draft_loop={metadata.in_mtp_draft_loop}, "
                    f"indexer_skip_topk={metadata.indexer_skip_topk}, "
                    f"mtp_index_share={self.mtp_index_share})"
                )
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
            if not metadata.is_ragged_verify:
                max_decode_len = gen_seq_lens.max().item()
                min_decode_len = gen_seq_lens.min().item()
                assert max_decode_len == min_decode_len, (
                    "max_decode_len != min_decode_len, we need padding"
                )

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[token_offset : token_offset + num_gen_tokens, ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if metadata.is_ragged_verify:
                take_strided_path = False
            else:
                take_strided_path = not metadata.use_expanded_buffers_for_mtp or next_n == 1
            if take_strided_path:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts : num_contexts + num_generations
                ]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n. The DSL path
                # uses its own schedule buffer (built with num_next_n_atoms=1
                # via a (num_gen, 1) input shape) and overrides this below.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            if metadata.is_ragged_verify:
                assert num_gen_tokens == sum(metadata.ragged_verify_lens), (
                    f"ragged gen tokens {num_gen_tokens} != "
                    f"sum(verify_lens) {sum(metadata.ragged_verify_lens)}"
                )
            else:
                assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[token_offset : token_offset + num_gen_tokens, ...]

            # Score the paged FP8 indexer cache using the prepared decode metadata.
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            indexer_max_seq_len = metadata.get_indexer_max_seq_len()

            if self.use_cute_dsl_paged_mqa_logits and not metadata.is_ragged_verify:
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `Indexer.prepare()`
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
                # next_n). MTPEagle reuses the same metadata for its multi-step
                # draft loop; after i=0 it mutates seq_lens to 1, so i≥1
                # iterations run with next_n=1. The reshape
                # `(num_gen, next_n, ...) -> (num_gen*factor, atom, ...)` is only
                # valid when the caller actually supplies next_n == factor * atom
                # tokens; gate here so i≥1 draft calls fall back to the
                # kernel-native next_n=1 path.
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
                # Atom-split benefits small-batch / low-ntask configs by
                # raising SM utilization at the cost of factorx KV HBM
                # re-reads.
                dsl_q = q_decode
                fp8_ctx_lens = dsl_context_lens
                fp8_block_table = block_table
                fp8_schedule_meta = metadata.scheduler_metadata_buffer
                if dsl_atom_split:
                    factor = metadata.dsl_expand_factor
                    atom = metadata.dsl_atom
                    exp_B = num_generations * factor
                    dsl_q = q_decode.reshape(exp_B, atom, self.n_heads, self.head_dim)
                    fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                    fp8_block_table = metadata.block_table_expanded[:exp_B]
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                elif dsl_kernel_split:
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                logits_decode = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                    dsl_q,
                    k_cache,
                    weights_decode,
                    fp8_ctx_lens,
                    fp8_block_table,
                    fp8_schedule_meta,
                    indexer_max_seq_len,
                    **dsl_split_kwargs,
                )
            else:
                decode_q_scale = None
                logits_decode = self._call_paged_mqa_logits(
                    q_decode,
                    k_cache,
                    weights_decode,
                    context_lens,
                    block_table,
                    scheduler_metadata_buffer,
                    indexer_max_seq_len,
                    decode_q_scale,
                )

            # Native/GVR use logical lengths; radix uses score-column lengths.
            gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                num_contexts : num_contexts + num_generations
            ]
            scan_lengths = metadata.gen_indexer_kv_lens_cuda_runtime
            assert scan_lengths is not None

            gvr_ext_kwargs = (
                {
                    "gvr_prior_indices": gvr_prior_indices[:num_generations],
                    "gvr_row_order": metadata.kv_lens_row_reorder,
                }
                if gvr_prior_indices is not None
                else None
            )
            assert metadata.radix_aux_indices is not None
            assert metadata.radix_aux_logits is not None
            self.top_k(
                logits_decode,
                topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                is_prefill=False,
                sequence_lengths=gen_kv_lens_cuda,
                scan_lengths=scan_lengths,
                row_kv_lens=metadata.ragged_row_kv_lens(num_gen_tokens),
                next_n=1 if metadata.is_ragged_verify else next_n,
                max_seq_len=indexer_max_seq_len,
                gvr_ext_kwargs=gvr_ext_kwargs,
                radix_aux_indices=metadata.radix_aux_indices,
                radix_aux_logits=metadata.radix_aux_logits,
            )

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
            )

        # Keep the GVR prior current for computed, reused, and dense-skip TopK.
        if gvr_prior_indices is not None and has_decode:
            decode_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
            if metadata.is_ragged_verify:
                last_rows = (
                    torch.cumsum(metadata.gen_token_repeats_cuda[:num_generations], dim=0).to(
                        torch.long
                    )
                    - 1
                )
                last_mtp_topk = decode_topk[last_rows]
            else:
                next_n = num_gen_tokens // num_generations
                last_mtp_topk = decode_topk[next_n - 1 :: next_n]
            prev_topk_dst = gvr_prior_indices[:num_generations]
            if do_multi_stream() and self.aux_stream is not None:
                # Overlap the GVR prior write-back with this layer's sparse attention.
                self.prev_topk_copy_events[0].record()
                with torch.cuda.stream(self.aux_stream):
                    self.prev_topk_copy_events[0].wait()
                    prev_topk_dst.copy_(last_mtp_topk)
                    self.prev_topk_copy_events[1].record()
                self._prev_topk_copy_pending = True
            else:
                prev_topk_dst.copy_(last_mtp_topk)

        if self.mtp_index_share and metadata.in_mtp_draft_loop and not reuse_topk:
            self._store_mtp_shared_rows(
                topk_indices_buffer,
                metadata,
                num_contexts,
                num_generations,
                num_gen_tokens,
                token_offset,
                has_prefill,
                has_decode,
                is_generation,
            )
        return topk_indices_buffer

    def _copy_reused_topk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        topk_indices_buffer: torch.Tensor,
        token_offset: int,
        num_gen_tokens: int,
        num_generations: int,
        has_prefill: bool,
        gvr_prior_indices: Optional[torch.Tensor],
    ) -> None:
        """Copy the shared TopK rows an MTP draft step reuses into its generation rows."""
        topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
            metadata.shared_topk_indices[:num_generations, :]
        )

    def _fill_decode_topk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
        token_offset: int,
        num_gen_tokens: int,
    ) -> bool:
        """Fill the generation TopK rows ahead of the decode branch; True if it filled them.

        This class leaves them to the decode branch of ``sparse_attn_indexer``.
        """
        return False

    def _store_mtp_shared_rows(
        self,
        topk_indices_buffer: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        num_contexts: int,
        num_generations: int,
        num_gen_tokens: int,
        token_offset: int,
        has_prefill: bool,
        has_decode: bool,
        is_generation: Optional[bool],
    ) -> None:
        """Store each request's last accepted TopK row for the next MTP draft step to reuse."""
        rows = None
        if has_decode:
            next_n = num_gen_tokens // num_generations
            gen_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
            rows = self._mtp_last_accepted_rows(
                gen_topk, metadata, num_contexts, num_generations, next_n
            )
        if has_prefill:
            ctx_last = torch.cumsum(metadata.seq_lens_cuda[:num_contexts].to(torch.long), dim=0) - 1
            ctx_rows = topk_indices_buffer[ctx_last]
            rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
        if rows is not None:
            index_share = _dsa_state(metadata).index_share
            shared_topk_indices = index_share.shared_topk_indices
            if shared_topk_indices is None:
                index_share.bind_buffers(shared_topk_indices=rows.contiguous())
            else:
                row_start = num_contexts if is_generation else 0
                shared_topk_indices[row_start : row_start + rows.shape[0], : rows.shape[1]].copy_(
                    rows
                )

    def _mtp_last_accepted_rows(
        self,
        gen_topk: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        num_contexts: int,
        num_generations: int,
        next_n: int,
    ) -> torch.Tensor:
        """Select each generation request's last accepted TopK row."""
        num_accepted = metadata.mtp_num_accepted
        if num_accepted is None:
            return gen_topk[next_n - 1 :: next_n]
        gen_num_accepted = num_accepted[num_contexts : num_contexts + num_generations]
        base = torch.arange(num_generations, device=gen_topk.device, dtype=torch.long) * next_n
        offset = (gen_num_accepted - 1).clamp(0, next_n - 1)
        return gen_topk[base + offset]

    def maybe_join_prev_topk_copy(self) -> None:
        """Join the aux-stream GVR prior write-back, if forked."""
        if self._prev_topk_copy_pending:
            self.prev_topk_copy_events[1].wait()
            self._prev_topk_copy_pending = False

    def _weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(
        self, qr: torch.Tensor, indexer_k: torch.Tensor, position_ids: torch.Tensor
    ):
        """Project Q/K and apply RoPE"""
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        k_pe = k_pe[:, 0, :]
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate the rotary and non-rotary columns and quantize to FP8."""
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The query
        scales are folded into weights and retained in the projection outputs
        for the graph partition's tensor contract.
        """
        assert self._fused_wk_wp_weight is not None, (
            "cache_derived_state() must be called before forward()"
        )
        # When the boundary fusion pre-quantized the next layer's kv_a_proj
        # NVFP4 input, hidden_states is an Fp4QuantizedTensor that also carries
        # the BF16 post-RMSNorm value. Use that BF16 view here -- the indexer
        # weight is BF16 and the matmul needs a float input.
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "pre_indexer_proj received Fp4QuantizedTensor without bf16 view; "
                "the producer fusion must request return_norm_out=True"
            )
            hidden_states_bf = hidden_states.unquantized_hidden_states
        else:
            hidden_states_bf = hidden_states
        if self._indexer_bf16:
            # hidden_states_bf and _fused_wk_wp_weight are both the model dtype,
            # so this runs a single native GEMM with no fp32 cast around it.
            fused_out = F.linear(hidden_states_bf, self._fused_wk_wp_weight)
        else:
            with _tf32_matmul_enabled():
                # F.linear computes input @ weight.T internally; no explicit .t() needed.
                # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
                # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
                # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
                # which uses its own handle and always falls back to CUDA-core SGEMM.
                fused_out = F.linear(_to_float(hidden_states_bf), self._fused_wk_wp_weight)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads], dim=-1)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE,
        # FP8 quantize); a no-op when the projection already ran in the model dtype.
        indexer_k = indexer_k.to(hidden_states_bf.dtype)

        q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(qr, indexer_k, position_ids)
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

        return q_fp8, k_fp8, k_scale, weights, q_scale

    def forward_from_projected(
        self,
        metadata: DSAtrtllmAttentionMetadata,
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

        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        q_fp8_phase = q_fp8[phase_start:phase_end]
        weights_phase = weights[phase_start:phase_end]
        q_scale_phase = q_scale[phase_start:phase_end] if q_scale is not None else None

        return self.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8_phase,
            k_fp8,
            k_scale,
            weights_phase,
            q_scale=q_scale_phase,
            is_generation=is_generation,
        )

    @torch.inference_mode()
    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ):
        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids
        )
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        self._update_k_cache(k_fp8, k_scale, metadata)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.forward_from_projected(metadata, hidden_states, indexer_intermediates)
