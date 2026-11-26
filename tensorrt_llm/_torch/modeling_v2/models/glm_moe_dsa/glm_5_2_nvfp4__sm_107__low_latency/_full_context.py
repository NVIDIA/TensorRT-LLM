# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-profile storage and generation indexer for the unsharded Rubin path."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

from ._small_batch_draft import DRAFT_SHAPES
from .ops.dsa._full_context_topk import MAX_ROWS, TOPK, full_context_topk, workspace_bytes
from .ops.dsa.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from .ops.dsa.indexer_k_tail import indexer_k_tail
from .ops.dsa.stash_last_accepted_rows import stash_last_accepted_rows
from .ops.gemm._low_m_splitk_fused import q_b_plan
from .ops.gemm.indexer_q_tail import indexer_q_tail


class FullContextIndexer:
    """One fixed arena reused in stream order by verify and draft attention.

    No cache-manager tensor is retained here: profiling may replace the manager.
    ``capacity`` includes the model's conservative speculative-token headroom and
    is allocated before memory profiling so the final KV quota accounts for it.
    """

    def __init__(self, decode, capacity: int, device: torch.device) -> None:
        self.decode = decode
        self.capacity = capacity
        self.max_batch = 3 if decode.mapping.tp_size == 4 else 2
        columns = (capacity + 255) // 256 * 256
        self.logits = torch.empty((MAX_ROWS, columns), dtype=torch.float32, device=device)
        self.topk = torch.empty((MAX_ROWS, TOPK), dtype=torch.int32, device=device)
        self.k_fp8 = torch.empty((MAX_ROWS, 128), dtype=torch.float8_e4m3fn, device=device)
        self.k_scale = torch.empty((MAX_ROWS, 1), dtype=torch.float32, device=device)
        self.workspace = torch.zeros(workspace_bytes() // 4, dtype=torch.int32, device=device)

    def supports(self, metadata, rows: int) -> bool:
        if torch.compiler.is_compiling():
            return False
        manager = metadata.kv_cache_manager
        batch = metadata.num_seqs
        lengths = metadata.gen_indexer_kv_lens_cuda_runtime
        table = metadata.indexer_k_cache_block_offsets
        shared = metadata.shared_topk_indices
        return (
            1 <= batch <= self.max_batch
            and 1 <= rows <= MAX_ROWS
            and rows % batch == 0
            and rows == metadata.num_tokens
            and (
                rows == batch
                or batch <= DRAFT_SHAPES.get((self.decode.mapping.tp_size, rows // batch - 1), 0)
            )
            and not metadata.is_ragged_verify
            and not metadata.use_expanded_buffers_for_mtp
            and not metadata.expand_for_dsl
            and manager.tokens_per_block == 64
            and not manager.use_fp4
            and 1 <= manager.max_seq_len <= self.capacity
            and 1 <= metadata.get_indexer_max_seq_len() <= self.capacity
            and lengths is not None
            and lengths.shape == (batch,)
            and lengths.dtype == torch.int32
            and lengths.is_contiguous()
            and lengths.device == self.logits.device
            and table is not None
            and table.ndim == 2
            and table.shape[0] >= batch
            and table.dtype == torch.int32
            and table.stride(1) == 1
            and table.device == self.logits.device
            and shared is not None
            and shared.shape[0] >= rows
            and shared.shape[1] == TOPK
            and metadata.scheduler_metadata_buffer is not None
        )

    def reuse_draft_selection(self, indexer, metadata, draft_state) -> bool:
        return (
            draft_state is not None
            and indexer is not None
            and indexer.mtp_index_share
            and metadata.in_mtp_draft_loop
            and metadata.indexer_skip_topk
            and not metadata.skip_indexer_for_gen_reqs
            and metadata.shared_topk_indices is not None
        )

    def select(self, layer, kv_a, q_a, position_ids, metadata, draft_state=None, *, beside_topk):
        """Write keys, score the full context, and preserve the current layer's local TopK."""
        decode = self.decode
        mla = layer.self_attn
        indexer = mla.mqa.indexer
        rows, batch = metadata.num_tokens, metadata.num_seqs
        next_n = rows // batch
        manager = metadata.kv_cache_manager
        rope = indexer.rotary_emb
        cos_sin = rope.rotary_cos_sin.view(rope.max_positions, -1)
        positions = position_ids.reshape(-1)[:rows]
        latent_end = mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim
        index_k = kv_a[:, latent_end : latent_end + indexer.head_dim]
        index_weights = kv_a[:, latent_end + indexer.head_dim :]
        k_cache = manager.get_indexer_k_cache_buffers(indexer.layer_idx)
        decode._k_tail_fork.record()
        with torch.cuda.stream(decode._k_tail_stream):
            decode._k_tail_stream.wait_event(decode._k_tail_fork)
            indexer_k_tail(
                index_k,
                positions,
                cos_sin,
                indexer.k_norm.weight,
                indexer.k_norm.bias,
                indexer.k_norm.variance_epsilon,
                indexer.rope_dim,
                not rope.is_neox,
                indexer.scale_fmt == "ue8m0",
                self.k_fp8[:rows],
                self.k_scale[:rows],
                k_cache,
                metadata.slot_mapping_fp8[:rows],
                metadata.slot_mapping_scale[:rows],
                rows,
                True,
                True,
                False,
            )
            decode._k_tail_done.record()
        topk = self.topk[:rows]
        if metadata.skip_indexer_for_gen_reqs:
            # Short-context bypass still appends K for a later step that runs the indexer.
            torch.cuda.current_stream().wait_event(decode._k_tail_done)
            topk.copy_(metadata.topk_indices_buffer[:rows])
            beside_topk()
        else:
            q = q_a.new_empty((rows, indexer.n_heads, indexer.head_dim), dtype=torch.float8_e4m3fn)
            q_scale = q_a.new_empty((rows, indexer.n_heads, 1), dtype=torch.float32)
            weights = q_a.new_empty((rows, indexer.n_heads), dtype=torch.float32)
            indexer_q_tail(
                q_a,
                decode._wq_b_folded[layer.layer_idx],
                positions,
                cos_sin,
                index_weights,
                q,
                q_scale,
                weights,
                mla.q_a_layernorm.variance_epsilon,
                indexer.n_heads,
                indexer.head_dim,
                indexer.rope_dim,
                not rope.is_neox,
                indexer.scale_fmt == "ue8m0",
                indexer.weight_scale_factor,
            )
            torch.cuda.current_stream().wait_event(decode._k_tail_done)
            capacity = metadata.get_indexer_max_seq_len()
            logits = fp8_paged_mqa_logits(
                q.view(batch, next_n, indexer.n_heads, indexer.head_dim),
                k_cache,
                weights,
                metadata.gen_indexer_kv_lens_cuda_runtime,
                metadata.indexer_k_cache_block_offsets[:batch],
                metadata.scheduler_metadata_buffer,
                self.logits[:rows],
                capacity,
                next_n % 2 == 0,
            )
            tactic = q_b_plan(
                "beside_topk", rows, mla.num_heads_tp * 576, mla.q_lora_rank, q_a.device.index
            )
            reserve = (
                (mla.num_heads_tp * 576 + tactic.mma_m - 1)
                // tactic.mma_m
                * ((rows + tactic.mma_n - 1) // tactic.mma_n)
                * tactic.split_k
            )

            def topk_select():
                full_context_topk(
                    logits,
                    metadata.gen_indexer_kv_lens_cuda_runtime,
                    topk,
                    self.workspace,
                    next_n=next_n,
                    max_seq_len=capacity,
                    sm_reserve=reserve,
                    deterministic=indexer.top_k.gvr_deterministic,
                )

            maybe_execute_in_parallel(
                topk_select, beside_topk, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
            )

        if metadata.in_mtp_draft_loop and indexer.mtp_index_share:
            if draft_state is None:
                raise RuntimeError("owned MTP attention needs its accepted-row stash")
            stash = draft_state.topk_stash[:batch]
            accepted = metadata.mtp_num_accepted
            if accepted is None:
                stash.copy_(topk[next_n - 1 :: next_n])
            else:
                stash_last_accepted_rows(topk, accepted[:batch], stash, next_n)
            # Only B accepted rows belong to the MTP shared buffer. Never overwrite
            # them with all B*next_n rows after compaction.
            metadata.shared_topk_indices[:batch].copy_(stash)
        else:
            metadata.shared_topk_indices[:rows].copy_(topk)
        return topk
