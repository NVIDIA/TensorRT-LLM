# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    _INDEXER_MQA_LOGITS_ELEM_BUDGET,
    F,
    Fp4QuantizedTensor,
    Indexer,
    _prefill_q_split_threshold,
    _split_prefill_queries,
    _tf32_matmul_enabled,
    _to_float,
    allgather,
    do_multi_stream,
    get_sm_version,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modules.top_k import TopKImplementation
from tensorrt_llm._torch.utils import is_piecewise_running
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...kernels.attention import indexer_prologue
from .full_query_policy import fp8_full_query_weight_cache_limit
from .topk import RubinTopK


class RubinIndexer(Indexer):
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
        *,
        projection_free: bool = False,
    ) -> None:
        use_rubin_gvr = (
            not projection_free
            and sparse_params.enable_heuristic_topk
            and sparse_params.use_self_sampling_topk
            and IS_CUTLASS_DSL_AVAILABLE
            and torch.cuda.is_available()
            and get_sm_version() == 107
            and sparse_params.index_topk == 2048
            and compress_ratio == 1
        )
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            replace(sparse_params, enable_heuristic_topk=False) if use_rubin_gvr else sparse_params,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
            projection_free=projection_free,
        )
        self._k_cache_prewritten_for: (
            tuple[torch.Tensor, torch.Tensor, DSAtrtllmAttentionMetadata, int] | None
        ) = None
        self._projection_outputs: tuple[torch.Tensor, ...] | None = None
        self._projection_metadata: DSAtrtllmAttentionMetadata | None = None
        if use_rubin_gvr:
            self._enable_heuristic_topk = True
            self._use_self_sampling_topk = True
            # Each CUDA graph runs the same eager forward before capture,
            # warming the GVR launcher's exact rows/stride/envelope key.
            self.top_k = RubinTopK(
                self.index_topk,
                prefill_implementation=TopKImplementation.CUDA_RADIX,
                decode_implementation=TopKImplementation.CUTE_DSL_GVR,
                compress_ratio=self.compress_ratio,
                gvr_self_sampling=True,
                gvr_deterministic=sparse_params.gvr_deterministic_topk,
            )

    def can_skip_projection(
        self, metadata: DSAtrtllmAttentionMetadata, rows: int | None = None
    ) -> bool:
        """Reuse draft-step-zero intermediates only while its TopK rows remain live."""
        if rows is None:
            rows = metadata.num_tokens
        return (
            self.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and metadata.shared_topk_indices is not None
            and metadata.num_contexts == 0
            and self._projection_metadata is metadata
            and self._projection_outputs is not None
            and all(output.shape[0] >= rows for output in self._projection_outputs)
        )

    def reused_projection_outputs(
        self, metadata: DSAtrtllmAttentionMetadata, rows: int
    ) -> list[torch.Tensor]:
        """Keep the five-tensor projection interface for a TopK-only draft step."""
        if not self.can_skip_projection(metadata, rows):
            raise RuntimeError("Indexer projection reuse requires initialized draft TopK")
        return [output[:rows] for output in self._projection_outputs]

    @staticmethod
    def _is_prefix_view(full: torch.Tensor, view: torch.Tensor, rows: int) -> bool:
        return (
            view.device == full.device
            and view.dtype == full.dtype
            and view.shape == (rows, *full.shape[1:])
            and view.stride() == full.stride()
            and view.data_ptr() == full.data_ptr()
        )

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
    ) -> None:
        prewritten = self._k_cache_prewritten_for
        self._k_cache_prewritten_for = None
        outputs = self._projection_outputs
        if (
            self.can_skip_projection(metadata)
            and self._is_prefix_view(outputs[1], k_fp8, metadata.num_tokens)
            and self._is_prefix_view(outputs[2], k_scale, metadata.num_tokens)
        ):
            return
        # The outputs stay alive until consumed, so an allocator cannot recycle
        # their addresses into an unrelated cache update.
        if outputs is not None and self._is_prefix_view(outputs[1], k_fp8, metadata.num_tokens):
            self._projection_metadata = metadata
        if prewritten is not None:
            full_k, full_scale, written_metadata, rows = prewritten
            if (
                written_metadata is metadata
                and rows == metadata.num_tokens
                and self._is_prefix_view(full_k, k_fp8, rows)
                and self._is_prefix_view(full_scale, k_scale, rows)
            ):
                return
        super()._update_k_cache(k_fp8, k_scale, metadata)

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
        phase. ``q_scale`` is only consumed by the FP4 dispatch.

        """
        # DSACacheManager / DeepseekV4CacheManager force quant_block_size to
        # 128 (FP8 path) or 32 (MXFP4 path); both round-trip to the same
        # 4-byte scale word per token at index_head_dim=128, so the slot
        # mapping arithmetic is unchanged in either mode.
        assert metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size in (
            32,
            128,
        ), (
            "Unexpected quant_block_size "
            f"{metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
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
                # FP4 packs two codes per byte so the gathered row holds half
                # as many bytes as in the FP8 path. The scale (4 bytes) is the
                # same in both modes because FP4 packs four UE8M0 exponents
                # into one int32 to match FP8's float32 scale width.
                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim

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
                    if self.use_fp4 and (
                        self.head_dim != 128 or not q_fp8.is_cuda or get_sm_version() < 100
                    ):
                        # Decoded MQA also materializes per-head dot products.
                        budget = max(1, budget // self.n_heads)
                    q_tile = max(1, min(local_q_len, budget // max(1, num_k_tokens)))
                    for tile_off in range(0, local_q_len, q_tile):
                        c0 = chunk_q_start + tile_off
                        c1 = min(c0 + q_tile, chunk_q_end)
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        tile_q_scale = q_scale[g0:g1, ...] if self.use_fp4 else None
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

                ctx_q_scale = q_scale[:num_ctx_tokens, ...] if self.use_fp4 else None
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
            if self.use_gvr_emission:
                # reused slots cold-start the emission closed loop; stale
                # lines only mis-place cuts - counts are re-measured
                # in-kernel, so exactness never rides on this reset
                self.top_k.reset_gvr_emission_rows(
                    slice(num_generations, num_generations + num_contexts)
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

            # Get k cache and call fp8_paged_mqa_logits / fp8_fp4_paged_mqa_logits
            # with prepared decode metadata.
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
                gvr_emit_kwargs: dict = {}
                # emitting for a step the Top-K cannot consume only churns
                # the closed-loop state, so gate on the consumable shape
                if (
                    self.use_gvr_emission
                    and gvr_prior_indices is not None
                    and next_n == 1
                    and not dsl_atom_split
                    and not dsl_kernel_split
                    and num_gen_tokens <= 256
                    # ext tiers are single-CTA/sort-path only; row reordering
                    # routes the Top-K through order_row, which excludes them
                    and metadata.kv_lens_row_reorder is None
                ):
                    gvr_emit_kwargs = self.top_k.prepare_gvr_emission(
                        num_generations,
                        indexer_max_seq_len,
                        torch.cuda.get_device_properties(q_decode.device).multi_processor_count,
                        gvr_prior_indices,
                    )
                if self.use_fp4:
                    # FP4 DSL signature splits DG's (q, sf_q) tuple into two
                    # separate args and requires q.dtype == uint8 (q_decode
                    # came in via the FP8 plumbing as int8; reinterpret with
                    # no copy). sf_q is the q_scale slice reshaped to
                    # (B, next_n, H) int32 -- mirrors the non-DSL FP4 branch.
                    decode_q_scale = q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                    dsl_q = q_decode.view(torch.uint8)
                    dsl_block_table = block_table
                    dsl_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        eff_next_n = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = dsl_q.reshape(exp_B, eff_next_n, self.n_heads, self.head_dim // 2)
                        decode_q_scale = decode_q_scale.reshape(exp_B, eff_next_n, self.n_heads)
                        dsl_context_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        dsl_block_table = metadata.block_table_expanded[:exp_B]
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    elif dsl_kernel_split:
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded

                    logits_decode = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q,
                        decode_q_scale,
                        k_cache,
                        weights_decode,
                        dsl_context_lens,
                        dsl_block_table,
                        dsl_schedule_meta,
                        indexer_max_seq_len,
                        **gvr_emit_kwargs,
                        **dsl_split_kwargs,
                    )
                else:
                    full_query = (
                        fp8_full_query_weight_cache_limit(
                            next_n, self.n_heads, self.head_dim, k_cache.shape[1], q_decode.dtype
                        )
                        is not None
                    )
                    if full_query:
                        from .full_query_logits import cute_dsl_fp8_paged_mqa_logits

                        logits_decode = cute_dsl_fp8_paged_mqa_logits(
                            q_decode,
                            k_cache,
                            weights_decode,
                            dsl_context_lens,
                            block_table,
                            metadata.scheduler_metadata_buffer,
                            indexer_max_seq_len,
                        )
                    else:
                        # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                        # Atom-split benefits small-batch / low-ntask configs by
                        # raising SM utilization at the cost of factorx KV HBM
                        # re-reads. Context lengths are indexer KV lengths, which
                        # may be compressed for DeepSeek-V4.
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
                            **gvr_emit_kwargs,
                            **dsl_split_kwargs,
                        )
            else:
                decode_q_scale = (
                    q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    if self.use_fp4
                    else None
                )
                if self.use_fp4:
                    # q_decode shape is either (num_generations, next_n, n_heads,
                    # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                    # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
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

    def pre_indexer_proj(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        q_in: torch.Tensor | None = None,
        wk_wp_out: torch.Tensor | None = None,
        k_cache_metadata: DSAtrtllmAttentionMetadata | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token-wise projections with an optional eager/whole-graph cache write.

        Runs cublas_mm, qk_projection_and_rope, FP8/FP4 quantize, and weight
        scaling. Piecewise callers leave ``k_cache_metadata`` unset, keeping
        the projection partition free of cache mutations and batch metadata.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in dsa.module.forward_dsa_proj sees a stable signature.
        """
        self._k_cache_prewritten_for = None
        self._projection_outputs = None
        self._projection_metadata = None
        if self.projection_free:
            raise RuntimeError("Projection-free Indexer requires forward_prepared()")
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
        if wk_wp_out is not None:
            fused_out = wk_wp_out
        elif self._indexer_bf16:
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
        if q_in is None and type(self)._qk_projection_and_rope is Indexer._qk_projection_and_rope:
            q_in = self.wq_b(qr)
        if self._fused_prologue_eligible(q_in, indexer_k, weights, hidden_states_bf.dtype):
            outputs = self._prologue_fused(indexer_k, weights, q_in, position_ids, k_cache_metadata)
            self._projection_outputs = outputs
            return outputs
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE,
        # FP8 quantize); a no-op when the projection already ran in the model dtype.
        indexer_k = indexer_k.to(hidden_states_bf.dtype)

        if q_in is None:
            q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(qr, indexer_k, position_ids)
        else:
            q = q_in.view(-1, self.n_heads, self.head_dim)
            k = self.k_norm(indexer_k)
            q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
            k_pe = k_pe[:, 0, :]
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        if self.use_fp4:
            # FP4 packs two codes per byte, so the trailing dim is head_dim // 2.
            # fused_cat_fp4 flattens the leading dims to M=N*n_heads; restore
            # the (N, n_heads, ...) shape so downstream slicing in
            # sparse_attn_indexer (which indexes by token) lines up with
            # q_fp8. The DeepGEMM FP4 kernel applies the per-block q_scale
            # internally, so weights carry only softmax_scale * n_heads^-0.5.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            # DeepGEMM's fp8_fp4_(paged_)mqa_logits asserts
            # `weights.scalar_type() == kFloat`. Unlike the FP8 branch (whose
            # `_weight_scale` multiplies by the fp32 `q_scale` tensor and so
            # implicitly upcasts), this branch's only multiplier is a Python
            # float — `bf16 * float` stays bf16 and trips the kernel assert.
            # Cast explicitly so this path doesn't silently rely on an
            # upstream `_to_float`.
            weights = weights.float() * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        outputs = (q_fp8, k_fp8, k_scale, weights, q_scale)
        self._projection_outputs = outputs
        return outputs

    def _fused_prologue_eligible(
        self,
        q_in: torch.Tensor | None,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        model_dtype: torch.dtype,
    ) -> bool:
        if (
            not indexer_prologue.FUSED_PROLOGUE_ENABLED
            or self.use_fp4
            or self.head_dim != 128
            or q_in is None
            or type(self)._qk_projection_and_rope is not Indexer._qk_projection_and_rope
            or type(self)._prep_q_or_k is not Indexer._prep_q_or_k
        ):
            return False
        cos_sin = getattr(self.rotary_emb, "rotary_cos_sin", None)
        return (
            IS_FLASHINFER_AVAILABLE
            and not getattr(self.rotary_emb, "inverse", False)
            and isinstance(cos_sin, torch.Tensor)
            and cos_sin.is_cuda
            and cos_sin.dtype == torch.float32
            and 0 < self.rope_dim <= self.head_dim
            and self.rope_dim % 2 == 0
            and model_dtype == torch.bfloat16
            and q_in.dtype == torch.bfloat16
            and q_in.ndim == 2
            and q_in.shape == (indexer_k.shape[0], self.n_heads * self.head_dim)
            and q_in.stride(-1) == 1
            and indexer_k.dtype in (torch.bfloat16, torch.float32)
            and indexer_k.stride(-1) == 1
            and weights.stride(-1) == 1
        )

    def _prologue_fused(
        self,
        indexer_k: torch.Tensor,
        weights: torch.Tensor,
        q_in: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rows = indexer_k.shape[0]
        k_cache = slot_fp8 = slot_scale = None
        n_write = 0
        if (
            metadata is not None
            and not is_piecewise_running()
            and not torch.compiler.is_compiling()
            and metadata.kv_cache_manager is not None
            and metadata.slot_mapping_fp8 is not None
            and metadata.slot_mapping_scale is not None
            and 0 < metadata.num_tokens <= n_rows
        ):
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            slot_fp8 = metadata.slot_mapping_fp8
            slot_scale = metadata.slot_mapping_scale
            n_write = metadata.num_tokens
        cos_sin = self.rotary_emb.rotary_cos_sin
        outputs = indexer_prologue.dsa_indexer_prologue_fused(
            indexer_k,
            weights,
            q_in.view(n_rows, self.n_heads, self.head_dim),
            position_ids.view(-1),
            cos_sin.view(cos_sin.shape[0], -1),
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.variance_epsilon,
            self.weight_scale_factor,
            self.rope_dim,
            interleave=not self.rotary_emb.is_neox,
            use_ue8m0=self.scale_fmt == "ue8m0",
            k_cache=k_cache,
            slot_mapping_fp8=slot_fp8,
            slot_mapping_scale=slot_scale,
            num_cache_tokens=n_write,
        )
        if n_write:
            self._k_cache_prewritten_for = (outputs[1], outputs[2], metadata, n_write)
        return outputs
