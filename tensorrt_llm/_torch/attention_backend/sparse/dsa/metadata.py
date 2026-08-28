# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
from tensorrt_llm.logger import logger

from .cache_manager import is_dsa_cache_manager
from .indexer import (
    _DG_SCHEDULE_BLOCK_KV,
    Indexer,
    IndexerPrefillChunkMetadata,
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _pick_dsl_expand,
    _select_indexer_compress_ratio,
)
from .params import DSAMetadataParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

# Indexer MQA-logits are currently always fp32. The dtype is part of the
# CuTe DSL Top-K compile key, so warmup must use the runtime dtype.
_INDEXER_LOGITS_DTYPE = torch.float32

if TYPE_CHECKING:
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


def build_req_idx_per_token(seq_lens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Map each flattened-batch token to its request index."""
    cu_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    token_idx = torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int32)
    return torch.searchsorted(cu_seq_lens, token_idx, right=True)


@dataclass(init=False)
class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Attention metadata for DSA (Dense Sparse Attention) with indexer state."""

    sparse_metadata_params: Optional[DSAMetadataParams] = None
    use_fp8_ds_mla: bool = field(default=False, init=False)
    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # TopK for static token sparse attention
    num_sparse_topk: int
    # TopK for dynamic sparse MLA
    sparse_mla_topk: int
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Indexer head dimension
    indexer_head_dim: int = 128
    # Indexer quant block size
    indexer_quant_block_size: int = 128
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    shared_topk_indices: Optional[torch.Tensor] = None
    indexer_skip_topk: bool = False
    in_mtp_draft_loop: bool = False
    mtp_num_accepted: Optional[torch.Tensor] = None
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False
    # Whether to reshape the DSL paged MQA logits Q tensor into a kernel-
    # supported `effective_next_n` via caller-side atom-split (FP4: {1,2,3};
    # FP8: {1,2,3,4}; see `_pick_dsl_expand`). Reuses
    # `kv_lens_expanded_cuda` / `block_table_expanded` /
    # `scheduler_metadata_buffer_expanded`; runtime mutually exclusive with
    # `use_expanded_buffers_for_mtp` (the latter requires `not _use_dsl`).
    expand_for_dsl: bool = False
    # Cached (expand_factor, atom) decision from the wave-aware picker. Set at
    # `prepare()` time and read by forward call sites — avoids re-running the
    # picker per call and guarantees prepare/forward use the SAME decision
    # (otherwise the populated buffers would mismatch the kernel reshape).
    dsl_expand_factor: int = 1
    dsl_atom: int = 1
    # Compression ratio for KV tokens
    compress_ratios: List[int] = field(default_factory=lambda: [1])
    # Number of compressed KV tokens for context requests
    num_ctx_kv_tokens: int = 0
    gen_indexer_kv_lens_cuda_runtime: Optional[torch.Tensor] = None

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if (
            kwargs.get("sparse_metadata_params") is None
            and sparse_attention_config is not None
            and hasattr(sparse_attention_config, "to_sparse_metadata_params")
        ):
            kwargs["sparse_metadata_params"] = sparse_attention_config.to_sparse_metadata_params()
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        # Cross-layer fan-out of the DSA index remap (TRTLLM_DSA_GROUP_REMAP).
        # `_group_remap_struct` is the (static) full+shared indexer
        # group layout; `_group_remap_batched` holds the current forward's
        # per-group batched remap output (leader writes, shared members read).
        self._group_remap_struct = None
        self._group_remap_struct_kv_id = 0
        self._group_remap_batched = {}
        super().__init__(*args, **kwargs)
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.indexer_max_chunk_size = sparse_metadata_params.indexer_max_chunk_size

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        super().__post_init__()
        if not is_dsa_cache_manager(self.kv_cache_manager):
            has_deepseek_v4_cache_interface = all(
                hasattr(self.kv_cache_manager, attr)
                for attr in ("compressed_block_sizes", "get_cache_indices")
            )
            assert has_deepseek_v4_cache_interface, (
                "DSAtrtllmAttentionMetadata requires DSACacheManager-compatible "
                f"cache manager, got {type(self.kv_cache_manager)}"
            )
        self.use_fp8_ds_mla = getattr(self.kv_cache_manager, "use_fp8_ds_mla", False)

        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.num_sparse_topk = sparse_metadata_params.max_sparse_topk
        self.sparse_mla_topk = self.num_sparse_topk
        self.indexer_head_dim = sparse_metadata_params.index_head_dim
        self.indexer_quant_block_size = 128
        self.enable_indexer_skip = sparse_metadata_params.enable_indexer_skip
        self.use_cute_dsl_topk = (
            sparse_metadata_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        )
        self.enable_gvr_topk = (
            sparse_metadata_params.enable_heuristic_topk and get_sm_version() >= 100
        )
        self.kv_lens_row_reorder = None
        capture_graph = self.is_cuda_graph
        # Plain DSA has no compression and uses the default [1]. DeepSeek-V4's
        # metadata params carry the model-specific compression ratios.
        self.compress_ratios = getattr(sparse_metadata_params, "compress_ratios", [1])

        # Effective tokens-per-block for the indexer k-cache slot mapping.
        # DeepSeek-V4's indexer cache uses layer-dependent compressed block sizes
        # (tokens_per_block // compress_ratio), so slot mappings must be built
        # against that stride — not kv_cache_manager.tokens_per_block directly.
        tpb = self.kv_cache_manager.tokens_per_block
        self._indexer_compress_ratio = _select_indexer_compress_ratio(self.compress_ratios)
        if hasattr(self.kv_cache_manager, "compressed_block_sizes"):
            tpb = tpb // _effective_compress_ratio_divisor(self._indexer_compress_ratio)
        self._tokens_per_block = tpb

        self.create_buffers_for_mla_rope_append(capture_graph=capture_graph)
        self.create_buffers_for_indexer(capture_graph=capture_graph)

    def prepare(self):
        super().prepare()
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            # For Helix CP, inactive ranks only attend to previously cached
            # tokens (no new token appended), while active ranks add new tokens.
            # This mirrors the kv_lens logic in TrtllmAttentionMetadata.prepare().
            active_rank = ~self.helix_is_inactive_rank_cpu[: self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        # For mla_rope_append_paged_kv_assign_q
        self.prepare_for_mla_rope_append(cached_token_lens, kv_lens)

        # Prepare to support skip indexer
        self.prepare_for_skip_indexer(kv_lens)

        # For indices conversion
        self.prepare_for_indices_conversion()

        # For indexer k cache
        self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

    def prepare_for_draft_forward(self) -> dict | None:
        """Select native DSA indexer metadata for a draft forward."""
        # DeepSeek-V4 metadata inherits DSA metadata, but its cache manager uses a
        # different dual-pool layout. Only native DSA cache managers use the DSA
        # draft-replay buffers below.
        if not is_dsa_cache_manager(self.kv_cache_manager):
            return None

        saved_state = {
            "host_indexer_k_cache_block_offsets": self.host_indexer_k_cache_block_offsets,
            "indexer_k_cache_block_offsets": self.indexer_k_cache_block_offsets,
            "host_slot_mapping_fp8": self.host_slot_mapping_fp8,
            "host_slot_mapping_scale": self.host_slot_mapping_scale,
            "slot_mapping_fp8": self.slot_mapping_fp8,
            "slot_mapping_scale": self.slot_mapping_scale,
            "block_table": self.block_table,
            "block_table_expanded": self.block_table_expanded,
            "host_block_table_expanded": self.host_block_table_expanded,
        }
        # The cached-KV feature owns these references even when an optimized
        # path aliases them to slot_mapping_*. With the feature disabled, the
        # aliases are lazy and may not exist on the first generation replay.
        if self.enable_context_mla_with_cached_kv:
            saved_state.update(
                {
                    "slot_mapping_fp8_fullkv": self.slot_mapping_fp8_fullkv,
                    "slot_mapping_scale_fullkv": self.slot_mapping_scale_fullkv,
                }
            )

        # Rebind to the draft manager's dedicated buffers instead of
        # overwriting the target tensors in place. Rebinding is invisible to
        # CUDA graph capture, so the target and draft segments of the graph
        # bake distinct addresses (like draft_kv_cache_block_offsets) and no
        # graph-recorded copy from a transient host buffer is needed.
        self.host_indexer_k_cache_block_offsets = self.host_draft_indexer_k_cache_block_offsets
        self.indexer_k_cache_block_offsets = self.draft_indexer_k_cache_block_offsets
        self.host_slot_mapping_fp8 = self.host_draft_slot_mapping_fp8
        self.slot_mapping_fp8 = self.draft_slot_mapping_fp8
        self.host_slot_mapping_scale = self.host_draft_slot_mapping_scale
        self.slot_mapping_scale = self.draft_slot_mapping_scale
        self.block_table = self.draft_block_table
        self.block_table_expanded = self.draft_block_table_expanded
        self.host_block_table_expanded = self.host_draft_block_table_expanded
        self._invalidate_pool_view_cache()

        # Recording a capture executes no kernels, so the draft mappings only
        # need refreshing when the transfers actually run: eager forwards
        # (warmup) and the pre-replay call from model_engine. The per-step
        # advance inside the captured graph re-derives slot mappings on
        # device from the rebound block-offset buffer.
        # kv_cache_manager was already swapped to the draft manager above.
        if not torch.cuda.is_current_stream_capturing():
            self.prepare_for_indexer_k_cache()
            self._refresh_expanded_block_table()
            Indexer.recompute_slot_mappings(self)
        Indexer.recompute_context_kv_gather_mappings(self)

        return saved_state

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        """Restore native DSA indexer metadata after a draft forward."""
        if saved_state is None:
            return

        self.host_indexer_k_cache_block_offsets = saved_state["host_indexer_k_cache_block_offsets"]
        self.indexer_k_cache_block_offsets = saved_state["indexer_k_cache_block_offsets"]
        self.host_slot_mapping_fp8 = saved_state["host_slot_mapping_fp8"]
        self.host_slot_mapping_scale = saved_state["host_slot_mapping_scale"]
        self.slot_mapping_fp8 = saved_state["slot_mapping_fp8"]
        self.slot_mapping_scale = saved_state["slot_mapping_scale"]
        self.block_table = saved_state["block_table"]
        self.block_table_expanded = saved_state["block_table_expanded"]
        self.host_block_table_expanded = saved_state["host_block_table_expanded"]
        self._invalidate_pool_view_cache()
        if "slot_mapping_fp8_fullkv" in saved_state:
            self.slot_mapping_fp8_fullkv = saved_state["slot_mapping_fp8_fullkv"]
            self.slot_mapping_scale_fullkv = saved_state["slot_mapping_scale_fullkv"]
        else:
            # The draft recomputation rebound the aliases to the draft tensors;
            # point them back at the restored target tensors.
            self.slot_mapping_fp8_fullkv = self.slot_mapping_fp8
            self.slot_mapping_scale_fullkv = self.slot_mapping_scale

    def get_indexer_kv_lens(self, kv_lens: torch.Tensor) -> torch.Tensor:
        if self._indexer_compress_ratio <= 1:
            return kv_lens
        return kv_lens // self._indexer_compress_ratio

    def get_indexer_max_seq_len(self) -> int:
        if self._indexer_compress_ratio <= 1:
            return self.kv_cache_manager.max_seq_len
        return max(1, self.kv_cache_manager.max_seq_len // self._indexer_compress_ratio)

    def warmup_cute_dsl_radix_topk(self, next_n: int) -> None:
        """Pre-compile CuTe DSL radix variants not covered by engine warmup."""
        sparse_params = self.sparse_metadata_params
        if not self.use_cute_dsl_topk or (
            sparse_params.enable_heuristic_topk and get_sm_version() >= 100
        ):
            return
        if self.kv_cache_manager is None:
            return
        top_k = self.sparse_mla_topk
        if not top_k:
            return
        if self._indexer_compress_ratio > 1 and next_n > 1:
            return
        try:
            from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
                warmup_cute_dsl_radix_topk_decode,
            )
        except ImportError:
            return

        warmup_cute_dsl_radix_topk_decode(
            top_k=int(top_k),
            num_cols=int(self.get_indexer_max_seq_len()),
            next_n=next_n,
            dtype=_INDEXER_LOGITS_DTYPE,
            num_sms=self.num_sms,
        )

    def warmup_selfsampling_topk(
        self, next_n: int, batch_sizes: Optional[List[int]] = None
    ) -> None:
        """Pre-compile the self-sampling GVR varlen engine during warmup.

        Mirrors ``warmup_cute_dsl_radix_topk``. The varlen launcher is keyed
        by the exact row count AND the logits row stride: rows cover the
        small eager batches plus every configured CUDA-graph batch size, and
        the stride mirrors what the active paged-MQA producer emits, so the
        warmed keys are the ones dispatch actually looks up. Batches outside
        this set still compile lazily on first touch. The helper enumerates
        one representative row per distinct engine compile key, so large
        batch lists warm in bounded time and memory. No-op unless the opt-in
        gate (TRTLLM_GVR_SELF_SAMPLING=1) selects the engine.
        """
        if os.environ.get("TRTLLM_GVR_SELF_SAMPLING", "0") != "1":
            return
        # same hardware gates as the dispatch flag (indexer __init__): never
        # compile these kernels on unsupported stacks during warmup
        if not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103):
            return
        if not self.enable_gvr_topk or self.kv_cache_manager is None:
            return
        top_k = getattr(self.sparse_metadata_params, "index_topk", None)
        if not top_k or int(top_k) not in (512, 1024, 2048):
            return
        cr = int(self._indexer_compress_ratio) if self._indexer_compress_ratio else 1
        if cr not in (1, 4):
            return
        try:
            from ....cute_dsl_kernels.blackwell.top_k import (
                gvr_topk_decode_self_sampling_host as _ss_host,
            )
        except ImportError:
            return
        nn = int(next_n)
        # warm the small row counts eager mixed batches commonly produce;
        # larger eager row counts lazy-JIT on first touch, and CUDA-graph
        # geometries are covered through ``batch_sizes`` below
        eager_warm_rows = 32
        rows = set(range(nn, eager_warm_rows + 1, nn)) or {nn}
        for bs in batch_sizes or ():
            rows.add(int(bs) * nn)
        msl_c = int(self.get_indexer_max_seq_len())
        if self.sparse_metadata_params.use_cute_dsl_paged_mqa_logits:
            # mirror the DSL paged-MQA arena stride (rows round up to 256
            # elements). A drift here only degrades warmup to unused keys —
            # dispatch still lazy-JITs the true key outside capture.
            row_stride = (msl_c + 255) // 256 * 256
        else:
            # DeepGEMM emits exact-width rows; a non-float4 width falls
            # through at the dispatch format gate, so there is nothing to warm
            row_stride = msl_c
            if row_stride % 4:
                return
        # helper takes max_seq_len in kv-token space (get_indexer_max_seq_len
        # is compressed — same multiply-back as the dispatch seam)
        try:
            _ss_host.warmup_varlen(
                int(top_k),
                msl_c * cr,
                compress_ratio=cr,
                next_n=int(next_n),
                num_rows_list=tuple(sorted(rows)),
                row_stride=row_stride,
            )
        except torch.cuda.OutOfMemoryError:
            # warmup is best-effort: the dispatch works without it (engines
            # JIT lazily outside capture), so do not fail engine init
            logger.warning(
                "self-sampling GVR warmup ran out of memory; varlen engines "
                "will JIT-compile lazily on first touch instead."
            )

    def on_update_kv_lens(self) -> None:
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside _preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        super().on_update_kv_lens()

        # _preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        # The draft loop can rewrite seq_lens after prepare() built this map.
        if self.num_tokens > 0:
            self.req_idx_per_token[: self.num_tokens] = build_req_idx_per_token(
                self.seq_lens_cuda[: self.num_seqs], self.num_tokens
            ).to(self.req_idx_per_token.dtype)

        if self.kv_cache_manager is not None and self.num_tokens > 0:
            seq_lens = self.seq_lens_cuda[: self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[: self.num_seqs] - seq_lens

            req_indices = self.req_idx_per_token[: self.num_tokens].to(dtype=torch.int64)
            seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = (
                torch.arange(self.num_tokens, device=seq_lens.device, dtype=torch.int64)
                - seq_starts[req_indices]
            )

            global_positions = start_positions[req_indices] + token_offsets
            if self.use_fp8_ds_mla:
                self.token_positions_cuda[: self.num_tokens] = global_positions.to(torch.int32)
            # Honor MXFP4 indexer K cache layout (½ byte per value vs FP8's
            # 1 byte) when the cache manager exposes a use_fp4 flag.
            index_head_dim = self.kv_cache_manager.index_head_dim
            use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
            data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self._tokens_per_block,
                self.kv_cache_manager.quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[: self.num_tokens] = fp8_indices
            self.slot_mapping_scale[: self.num_tokens] = scale_indices

        if self.num_generations > 0:
            torch.cumsum(
                self.kv_lens_cuda[self.num_contexts : self.num_seqs],  # num_contexts should be 0
                dim=0,
                dtype=torch.int64,
                out=self.gen_kv_indptr[1 : self.num_generations + 1],
            )
            torch.cumsum(
                (
                    self.kv_lens_cuda[self.num_contexts : self.num_seqs]
                    - self.seq_lens_cuda[self.num_contexts : self.num_seqs]
                ),
                dim=0,
                dtype=torch.int64,
                out=self.gen_cached_token_indptr[1 : self.num_generations + 1],
            )
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
            gen_indexer_kv_lens = self.get_indexer_kv_lens(gen_kv_lens)
            self.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
            next_n_cap = self.kv_lens_cuda_2d.shape[1]
            self.kv_lens_cuda_2d[: self.num_generations, :next_n_cap].copy_(
                gen_indexer_kv_lens.unsqueeze(-1).expand(-1, next_n_cap)
            )
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                gen_indexer_kv_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, self.num_sms
            )
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if self.max_draft_tokens > 0 and not self.use_expanded_buffers_for_mtp:
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    self.kv_lens_cuda_2d[: self.num_generations, :next_n_cap],
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True
                )
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                kv_lens_expanded = torch.stack([gen_indexer_kv_lens] * num_draft_tokens, dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = (
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                )
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
            if self.expand_for_dsl and self.dsl_expand_factor > 1:
                expand_factor = self.dsl_expand_factor
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_indexer_kv_lens.repeat_interleave(expand_factor)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(gen_kv_lens_expanded)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def _compute_kv_lens_row_reorder(self) -> None:
        """Prepare the longest-job-first GVR row order once per forward step."""
        next_n = 1 + self.max_draft_tokens
        if (
            self.enable_gvr_topk
            and self.use_cute_dsl_topk
            and self.num_generations * next_n >= 2 * self.num_sms
        ):
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
            order = torch.argsort(gen_kv_lens, descending=True).to(torch.int32)
            self.kv_lens_row_reorder_buffer[: self.num_generations].copy_(order)
            self.kv_lens_row_reorder = self.kv_lens_row_reorder_buffer[: self.num_generations]
        else:
            self.kv_lens_row_reorder = None

    def update_for_spec_dec(self):
        super().update_for_spec_dec()
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        self.on_update_kv_lens()

    # Create buffers for mla_rope_append_paged_kv_assign_q
    def create_buffers_for_mla_rope_append(self, capture_graph=False):
        # New context buffers for dsa
        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

    def create_buffers_for_indexer(self, capture_graph=False):
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # Local TopK reused across layers or MTP draft steps.
        if (
            sparse_metadata_params.has_shared_indexer_layers
            or sparse_metadata_params.mtp_index_share
        ):
            self.shared_topk_indices = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="shared_topk_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        else:
            self.shared_topk_indices = None

        self.indexer_skip_topk = False
        self.in_mtp_draft_loop = False
        self.mtp_num_accepted = None

        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.token_positions_cuda = None
        if self.use_fp8_ds_mla:
            self.token_positions_cuda = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="token_positions_cuda",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        # Allocate separate indexer block-offset and slot-mapping buffers for
        # the draft KV cache manager, mirroring draft_kv_cache_block_offsets:
        # the draft-replay context swaps these in by rebinding, so CUDA graph
        # capture bakes distinct addresses for the target and draft segments
        # and both sides can be refreshed eagerly outside the graph.
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_indexer_k_cache_block_offsets = self.get_empty(
                self.cuda_graph_buffers,
                [self.max_num_sequences, self.draft_kv_cache_manager.max_blocks_per_seq],
                cache_name="draft_indexer_k_cache_block_offsets",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_draft_indexer_k_cache_block_offsets = torch.zeros_like(
                self.draft_indexer_k_cache_block_offsets,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.draft_slot_mapping_fp8 = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="draft_slot_mapping_fp8",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_draft_slot_mapping_fp8 = torch.zeros_like(
                self.draft_slot_mapping_fp8,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.draft_slot_mapping_scale = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="draft_slot_mapping_scale",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_draft_slot_mapping_scale = torch.zeros_like(
                self.draft_slot_mapping_scale,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # Allocate these buffers dynamically in Indexer.prepare()
        # based on the actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_req_idx_per_token = torch.empty_like(
            self.req_idx_per_token, device="cpu", pin_memory=prefer_pinned()
        )
        # Block table for topk_indices conversion (shared for context and generation)
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_block_table = self.get_empty(
                self.cuda_graph_buffers,
                [
                    self.max_num_sequences,
                    self.draft_kv_cache_manager.max_blocks_per_seq,
                ],
                cache_name="draft_block_table",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # When MTP runs without the expanded-tokens path, the same forward step
        # alternates between full-window calls (next_n == 1 + max_draft_tokens)
        # and per-token draft calls (next_n == 1). The 2D DeepGEMM metadata
        # API encodes next_n into the schedule, so the precomputed schedule
        # for one shape cannot be reused for the other. Maintain a second
        # buffer holding the schedule for the full next_n window; the draft
        # path keeps using `scheduler_metadata_buffer`. Always allocate (a
        # few KB) so transitions in `update_spec_dec_param` don't have to
        # special-case its existence.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated 2D kv_lens buffer for the new DeepGEMM 2D context_lens
        # API. Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
        # broadcasts the same kv_len across next_n positions; kernel reads a
        # slice per forward. Avoids per-forward .expand().contiguous()
        # allocations that would break CUDA graphs.
        self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        if self.enable_gvr_topk:
            self.gvr_prior_indices = self.get_empty(
                self.cuda_graph_buffers,
                (
                    self.kv_cache_manager.num_local_layers,
                    self.max_num_sequences,
                    self.num_sparse_topk,
                ),
                cache_name="gvr_prior_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.gvr_prior_indices.zero_()
            if self.use_cute_dsl_topk:
                self.kv_lens_row_reorder_buffer = self.get_empty(
                    self.cuda_graph_buffers,
                    (self.max_num_sequences,),
                    cache_name="kv_lens_row_reorder_buffer",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    def _create_kv_lens_2d_buffer(self, capture_graph=False):
        """Pre-allocated buffer for the DeepGEMM 2D context_lens API.

        Avoids per-forward .expand().contiguous() allocations that break CUDA
        graphs. The buffer is written in-place via .copy_() inside
        on_update_kv_lens so its address stays stable across replays.
        """
        self.kv_lens_cuda_2d = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, 1 + self.max_draft_tokens),
            cache_name="kv_lens_cuda_2d",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self.max_draft_tokens),),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self.max_draft_tokens),
                self.kv_cache_manager.max_blocks_per_seq,
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_block_table_expanded = self.get_empty(
                self.cuda_graph_buffers,
                [
                    self.max_num_sequences * (1 + self.max_draft_tokens),
                    self.draft_kv_cache_manager.max_blocks_per_seq,
                ],
                cache_name="draft_block_table_expanded",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_draft_block_table_expanded = torch.zeros_like(
                self.draft_block_table_expanded,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def update_spec_dec_param(
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
    ):
        """Update speculative decoding parameters and create expanded buffers."""
        super().update_spec_dec_param(
            batch_size,
            is_spec_decoding_enabled,
            is_spec_dec_tree,
            is_spec_dec_dynamic_tree,
            max_draft_len,
            max_total_draft_tokens,
            spec_metadata,
            spec_tree_manager,
            num_contexts=num_contexts,
        )
        self.max_draft_tokens = max_draft_len
        capture_graph = self.is_cuda_graph
        if self.kv_lens_cuda_2d.shape[1] != 1 + self.max_draft_tokens:
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            self.create_expanded_buffers(capture_graph=capture_graph)

    def _update_indexer_k_cache_block_offsets(self) -> torch.Tensor:
        """Refresh INDEX_KEY offsets and return their physical pool slots."""
        cache_manager = self.kv_cache_manager
        # Raw block IDs can exceed the primary pool after host-cache offload;
        # the manager resolves their current physical slots.
        pool_indices = cache_manager.get_pool_block_indices(
            self.num_seqs,
            request_ids=self.request_ids,
            num_contexts=self.num_contexts,
            beam_width=self.beam_width,
        )
        page_indices = pool_indices * cache_manager.indexer_k_cache_page_scale
        num_blocks = page_indices.shape[1]
        self.host_indexer_k_cache_block_offsets[: self.num_seqs, :num_blocks].copy_(page_indices)
        self.indexer_k_cache_block_offsets[: self.num_seqs].copy_(
            self.host_indexer_k_cache_block_offsets[: self.num_seqs], non_blocking=True
        )
        # Sanitize graph-padding entries that may be stale after cache
        # eviction or host-cache onboarding.
        self.indexer_k_cache_block_offsets.clamp_(min=0)
        return pool_indices

    def set_skip_topk(self, skip: bool) -> None:
        self.indexer_skip_topk = skip

    def set_in_mtp_draft_loop(self, active: bool) -> None:
        self.in_mtp_draft_loop = active

    def set_mtp_num_accepted(self, num_accepted: Optional[torch.Tensor]) -> None:
        self.mtp_num_accepted = num_accepted

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False

    def _ensure_pool_view_cached(self):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, block table slices, and request index slices)
        are constant across all layers sharing the same KV pool and batch
        dimensions within a forward pass. Caching them avoids redundant
        Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(self.kv_cache_manager):
            return

        pool = self.kv_cache_manager.get_unique_primary_pool()
        kv_cache_manager = self.kv_cache_manager
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_block_table_ctx = self.block_table[: self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts : self.num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[: self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens : self.num_tokens] - self.num_contexts
        )
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._pool_cache_valid = True

    def _ensure_group_remap_struct(self):
        """Build (once) the static full+shared indexer group layout used by the
        cross-layer fan-out remap (``TRTLLM_DSA_GROUP_REMAP``).

        Groups are runs of consecutive local layers starting at a full-indexer
        layer (``indexer_k_cache_local_layer_mask[L] == True``, i.e.
        ``is_full_indexer_layer``); the shared layers that follow reuse the
        leader's top-k. Within a group, the remap output for each member differs
        from the leader's only by the additive ``layer_offset * tokens_per_block``
        term, so one launch (grid.z = group_size) covers the whole group.

        A group is marked active only when it has >= 2 members (a genuine
        fan-out) and a uniform primary-pool page scale (required for a single
        stride_factor). Singleton / non-uniform groups fall back to the
        per-layer single-op path (identical to flag-off behavior).
        """
        kvm = self.kv_cache_manager
        if self._group_remap_struct is not None and self._group_remap_struct_kv_id == id(kvm):
            return self._group_remap_struct

        # Building the small per-group layer-offset device tensors must happen
        # eagerly (host->device copy), never inside a CUDA-graph capture. TRT-LLM
        # runs an eager warmup before capture, so return an uncached empty struct
        # if somehow reached first during capture; it is (re)built on the next
        # eager step and grouping simply falls back to per-layer until then.
        if torch.cuda.is_current_stream_capturing():
            return {}

        mask = getattr(kvm, "indexer_k_cache_local_layer_mask", None)
        if mask is None:
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(kvm)
            return self._group_remap_struct

        n = len(mask)
        leader_of = [-1] * n
        slot_of = [0] * n
        members = {}  # leader local idx -> [member local idx, ...]
        cur_leader = None
        for local_idx in range(n):
            if mask[local_idx]:
                cur_leader = local_idx
                members[cur_leader] = [local_idx]
            else:
                if cur_leader is None:
                    # Shouldn't happen (layer 0 forced full); be safe.
                    cur_leader = local_idx
                    members[cur_leader] = [local_idx]
                else:
                    members[cur_leader].append(local_idx)
            leader_of[local_idx] = cur_leader
            slot_of[local_idx] = len(members[cur_leader]) - 1

        # Device for the small per-group layer-offset tensors.
        device = None
        if self.shared_topk_indices is not None:
            device = self.shared_topk_indices.device
        elif self.block_table is not None:
            device = self.block_table.device

        group_active = {}
        group_scale = {}
        group_size = {}
        group_layer_ids = {}
        for leader, member_list in members.items():
            group_size[leader] = len(member_list)
            params = [kvm.get_primary_pool_page_index_params(m) for m in member_list]
            scale0 = params[0][0]
            uniform_scale = all(p[0] == scale0 for p in params)
            offsets = [int(p[1]) for p in params]
            active = len(member_list) >= 2 and uniform_scale and device is not None
            group_active[leader] = active
            group_scale[leader] = int(scale0)
            if active:
                group_layer_ids[leader] = torch.tensor(offsets, dtype=torch.int32, device=device)

        struct = {
            "leader_of": leader_of,
            "slot_of": slot_of,
            "group_active": group_active,
            "group_scale": group_scale,
            "group_size": group_size,
            "group_layer_ids": group_layer_ids,
        }
        self._group_remap_struct = struct
        self._group_remap_struct_kv_id = id(kvm)
        return struct

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends, global_indices, side="right")
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self, kv_lens, device=False):  # device=False means use CPU
        """Prepare dense TopK indices for short sequences that skip the indexer."""

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[: self.num_contexts],
                        kv_lens[: self.num_contexts],
                        self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[ctx_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[: self.num_contexts],
                    kv_lens[: self.num_contexts],
                    self.num_ctx_tokens,
                )
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :], non_blocking=True
                )

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts : self.num_seqs],
                        kv_lens[self.num_contexts : self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[gen_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[self.num_contexts : self.num_seqs],
                    kv_lens[self.num_contexts : self.num_seqs],
                    self.num_tokens - self.num_ctx_tokens,
                )
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :], non_blocking=True
                )

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        # fp8_paged_mqa_logits supports seq_len 1/2/4 on sm100 and 1/2 on
        # sm90. Flatten Q and expand kv_lens and block_table for other MTP
        # configurations.
        # TODO:
        # - No distinction between sm90 and sm100 is needed once MTP3 is supported on sm90.
        # - Remove this once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
        use_dsl = self.sparse_metadata_params.use_cute_dsl_paged_mqa_logits
        self.use_expanded_buffers_for_mtp = not use_dsl and (
            (self.max_draft_tokens > 1 and get_sm_version() == 90)
            or (
                (self.max_draft_tokens == 2 or self.max_draft_tokens > 3)
                and get_sm_version() >= 100
            )
        )
        if self.use_expanded_buffers_for_mtp:
            # Expand kv_lens_cuda (only generation)
            num_tokens = self.num_generations * (1 + self.max_draft_tokens)
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            gen_kv_lens_expanded = torch.stack([gen_kv_lens] * (1 + self.max_draft_tokens), dim=0)
            gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(0, 1).contiguous().flatten()
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True
            )

            self._refresh_expanded_block_table(1 + self.max_draft_tokens)

        self.expand_for_dsl = (
            use_dsl and self.kv_cache_manager is not None and self.max_draft_tokens >= 1
        )
        if self.expand_for_dsl and self.num_generations > 0:
            next_n = 1 + self.max_draft_tokens
            kernel_atoms = (1, 2, 3) if self.kv_cache_manager.use_fp4 else (1, 2, 3, 4)
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            max_ctx = int(gen_kv_lens.max().item()) if gen_kv_lens.numel() else 0
            expand_factor, atom = _pick_dsl_expand(
                next_n,
                batch_size=self.num_generations,
                max_ctx=max_ctx,
                num_sms=self.num_sms,
                kernel_atoms=kernel_atoms,
            )
            self.dsl_expand_factor = expand_factor
            self.dsl_atom = atom
            if expand_factor > 1:
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_kv_lens.repeat_interleave(expand_factor)
                self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(
                    self.kv_lens_expanded_host[:num_tokens], non_blocking=True
                )
                self._refresh_expanded_block_table(expand_factor)
        else:
            self.dsl_expand_factor = 1
            self.dsl_atom = 1 + self.max_draft_tokens

    def _refresh_expanded_block_table(self, repeat_factor: Optional[int] = None):
        """Refresh the active cache's expanded INDEX_KEY page table."""
        if self.kv_cache_manager is None or self.num_generations == 0:
            return
        if repeat_factor is None:
            if self.use_expanded_buffers_for_mtp:
                repeat_factor = 1 + self.max_draft_tokens
            elif self.expand_for_dsl and self.dsl_expand_factor > 1:
                repeat_factor = self.dsl_expand_factor
            else:
                return

        num_tokens = self.num_generations * repeat_factor
        max_len = self.host_indexer_k_cache_block_offsets.shape[1]
        gen_block_tensor = self.host_indexer_k_cache_block_offsets[
            self.num_contexts : self.num_seqs, :max_len
        ]
        expanded_blocks = gen_block_tensor.repeat_interleave(repeat_factor, dim=0)
        self.host_block_table_expanded[:num_tokens, :max_len].copy_(
            expanded_blocks, non_blocking=True
        )
        self.block_table_expanded[:num_tokens].copy_(
            self.host_block_table_expanded[:num_tokens], non_blocking=True
        )
        self.block_table_expanded.clamp_(min=0)

    def prepare_for_indexer_k_cache(self):
        if self.kv_cache_manager is None:
            return
        # Keep physical slots for the primary-pool TopK conversion below.
        pool_indices = self._update_indexer_k_cache_block_offsets()

        # Build block_table for topk_indices conversion (actual block allocation)
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            active_rank = ~self.helix_is_inactive_rank_cpu[: self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        num_blocks_per_seq = (kv_lens[: self.num_seqs] + tokens_per_block - 1) // tokens_per_block
        max_blocks_used = num_blocks_per_seq.max().item() if self.num_seqs > 0 else 1
        # pool_indices already has correct values; set padding to -1.
        # Stage through a fresh pinned buffer: an async H2D from pageable
        # memory would block the host behind the busy execution stream.
        host_block_table = torch.empty(
            (pool_indices.shape[0], max_blocks_used),
            dtype=pool_indices.dtype,
            pin_memory=prefer_pinned(),
        )
        host_block_table.copy_(pool_indices[:, :max_blocks_used])
        pad_cols = torch.arange(max_blocks_used, dtype=num_blocks_per_seq.dtype)
        host_block_table.masked_fill_(
            pad_cols.unsqueeze(0) >= num_blocks_per_seq[: self.num_seqs].unsqueeze(1), -1
        )
        # Copy to GPU
        self.block_table[: self.num_seqs, :max_blocks_used].copy_(
            host_block_table, non_blocking=True
        )

    def prepare_for_skip_indexer(self, kv_lens: torch.Tensor):
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = (
                kv_lens[: self.num_contexts].max().item()
                <= self.num_sparse_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = (
                kv_lens[self.num_contexts : self.num_seqs].max().item()
                <= self.num_sparse_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

    def prepare_for_mla_rope_append(self, cached_token_lens: torch.Tensor, kv_lens: torch.Tensor):
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[: self.num_contexts].sum().item()
            self.max_ctx_kv_len = kv_lens[: self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[: self.num_contexts].max().item()
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_cached_token_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[: self.num_contexts + 1], non_blocking=True
            )
            # context kv indptr
            torch.cumsum(
                kv_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_kv_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_kv_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[: self.num_contexts + 1], non_blocking=True
            )
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            self.max_gen_seq_len = self.seq_lens[self.num_contexts : self.num_seqs].max().item()
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1 : self.num_generations + 1],
            )
            self.gen_cached_token_indptr[: self.num_generations + 1].copy_(
                self.host_gen_cached_token_indptr[: self.num_generations + 1], non_blocking=True
            )
            # generation kv indptr
            torch.cumsum(
                kv_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_kv_indptr[1 : self.num_generations + 1],
            )
            self.gen_kv_indptr[: self.num_generations + 1].copy_(
                self.host_gen_kv_indptr[: self.num_generations + 1], non_blocking=True
            )
        else:
            self.max_gen_seq_len = 0

    def prepare_for_indices_conversion(self):
        # Build req_idx_per_token for topk_indices conversion
        # Use pinned staging buffer to avoid pageable H2D memcpy
        self.host_req_idx_per_token[: self.num_tokens] = torch.repeat_interleave(
            torch.arange(self.num_seqs, dtype=torch.int32),
            self.seq_lens,
            dim=0,
        )
        self.req_idx_per_token[: self.num_tokens].copy_(
            self.host_req_idx_per_token[: self.num_tokens],
            non_blocking=True,
        )
