# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 target-owned metadata extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    _DG_SCHEDULE_BLOCK_KV,
    _atom_ctx_offsets,
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _pick_dsl_split,
    _select_indexer_compress_ratio,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata as _BaseDSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    TokenMajorGenView,
    _fused_dsa_meta_enabled,
    build_req_idx_per_token,
)
from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
from tensorrt_llm.logger import logger

from ....ops.attention.dsa_params import DSAMetadataParams
from .cache_utils import is_dsa_cache_manager
from .indexer import Indexer


@dataclass(init=False)
class DSAtrtllmAttentionMetadata(_BaseDSAtrtllmAttentionMetadata):
    """Attention behavior owned by the NVIDIA K3.1 target."""

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        self.enable_ragged_verification = bool(kwargs.pop("enable_ragged_verification", False))
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
        self._cached_pool_ptr = 0
        self._cached_pool_view = None
        self._cached_num_pool_tokens = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        # Cross-layer fan-out of the DSA index remap (TRTLLM_DISABLE_DSA_GROUP_REMAP).
        # `_group_remap_struct` is the (static) full+shared indexer
        # group layout; `_group_remap_batched` holds the current forward's
        # per-group batched remap output (leader writes, shared members read).
        self._group_remap_struct = None
        self._group_remap_struct_kv_id = 0
        self._group_remap_batched = {}
        self.num_ctx_mla_kv_tokens = 0
        self.nvfp4_mla_context_fp8_scratch = None
        self._radix_rows_per_sequence = 1
        TrtllmAttentionMetadata.__init__(self, *args, **kwargs)
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.indexer_max_chunk_size = sparse_metadata_params.indexer_max_chunk_size

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        TrtllmAttentionMetadata.__post_init__(self)
        if self.is_cross:
            # Ragged verification is a target self-attention capability. A
            # cross-attention metadata object may be a shallow copy of one, but
            # must not inherit its verifier-only buffers or per-step layout.
            self.enable_ragged_verification = False
            self.ragged_verify_lens = None
        self._validate_ragged_verification()
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
        # Mirror the indexer's two-level GVR decision. The representative
        # compress ratio matches every live indexer: the DeepSeek-V4 backend
        # only builds indexers on cr=4 layers and plain DSA uses cr=1.
        self.use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self.enable_gvr_topk,
            use_self_sampling_topk=sparse_metadata_params.use_self_sampling_topk,
            index_topk=self.num_sparse_topk,
            compress_ratio=self._indexer_compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=get_sm_version(),
        )
        self.needs_gvr_prior = (
            self.enable_gvr_topk
            and IS_CUTLASS_DSL_AVAILABLE
            and get_sm_version() in (100, 103)
            and not self.use_self_sampling_topk
        )

        self.create_buffers_for_mla_rope_append(capture_graph=capture_graph)
        self.create_buffers_for_indexer(capture_graph=capture_graph)
        self._create_nvfp4_mla_generation_buffers(capture_graph=capture_graph)
        self._create_helix_sparse_row_buffers(capture_graph=capture_graph)

    def _helix_local_kv_lens(self, cached_token_lens: torch.Tensor) -> torch.Tensor:
        """Rank-local KV lengths under Helix CP, as TrtllmAttentionMetadata computes them.

        Context rows append all their tokens. A generation row appends only the
        new tokens this rank owns: one token on the active rank without
        speculation, and a per-sequence count when a verify group can straddle a
        page boundary. The Helix per-sequence buffers are generation-relative.
        Under the overlap scheduler these host values are provisional;
        on_update_kv_lens() rederives the device state.
        """
        kv_lens = cached_token_lens + self.seq_lens_kv
        gen = slice(self.num_contexts, self.num_seqs)
        num_gen = self.num_generations
        if self._helix_spec_tokens_valid:
            kv_lens[gen] = cached_token_lens[gen] + self.helix_owned_new_tokens_cpu[:num_gen]
        else:
            inactive = self.helix_is_inactive_rank_cpu[:num_gen]
            kv_lens[gen] = torch.where(inactive, cached_token_lens[gen], kv_lens[gen])
        return kv_lens

    def _create_helix_sparse_row_buffers(self, capture_graph=False):
        """Allocate the token-major rows token-sparse MLA reads under Helix.

        After the Helix Top-K merge a rank keeps only its share of each row's
        global Top-K: the kept entries come first and the rest of the row is
        -1. The sparse-MLA kernel attends to the first min(top_k, kv_len)
        entries of a row and does not mask -1 inside that range, so every
        generation token is presented as its own row whose length is its kept
        count (``set_helix_sparse_row_lens``).
        """
        names = (
            "helix_row_kv_lens_cuda",
            "helix_row_kv_lens_host",
            "helix_row_req_idx_cuda",
            "helix_row_req_idx_host",
            "helix_row_prompt_lens_cuda",
            "helix_row_prompt_lens_cpu",
            "helix_row_request_types_host",
            "helix_row_block_offsets",
        )
        for name in names:
            setattr(self, name, None)
        self._helix_num_rows = 0
        self._helix_rows_set = False
        if not self.enable_helix or self.kv_cache_manager is None:
            return
        rows_cap = self.max_num_sequences * (2 + self._draft_sizing_cap)
        for name, dtype in (
            ("helix_row_kv_lens_cuda", torch.int32),
            ("helix_row_req_idx_cuda", torch.long),
            ("helix_row_prompt_lens_cuda", torch.int32),
        ):
            setattr(
                self,
                name,
                self.get_empty(
                    self.cuda_graph_buffers,
                    (rows_cap,),
                    cache_name=name,
                    dtype=dtype,
                    capture_graph=capture_graph,
                ),
            )
        pinned = prefer_pinned()
        self.helix_row_kv_lens_host = torch.zeros((rows_cap,), dtype=torch.int32, pin_memory=pinned)
        self.helix_row_req_idx_host = torch.zeros((rows_cap,), dtype=torch.long, pin_memory=pinned)
        self.helix_row_prompt_lens_cpu = torch.zeros(
            (rows_cap,), dtype=torch.int32, pin_memory=pinned
        )
        self.helix_row_request_types_host = torch.ones(
            (rows_cap,), dtype=torch.int32, pin_memory=pinned
        )
        num_attention_op_pools = getattr(
            self.kv_cache_manager, "num_attention_op_pools", self.kv_cache_manager.num_pools
        )
        self.helix_row_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            (num_attention_op_pools, rows_cap, 2, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="helix_row_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def _prepare_helix_sparse_rows(self, kv_lens: torch.Tensor) -> None:
        """Lay out one attention row per context request and generation token.

        Host lengths are upper bounds for the op's host-side planning; the
        device lengths start there and the indexer narrows each generation
        row to its kept count layer by layer.
        """
        self._helix_num_rows = 0
        if self.helix_row_kv_lens_cuda is None or self.num_generations <= 0:
            return
        num_contexts = self.num_contexts
        gen_seq_lens = self.seq_lens[num_contexts : self.num_seqs].to(torch.long)
        num_rows = num_contexts + int(gen_seq_lens.sum())
        assert num_rows <= self.helix_row_kv_lens_cuda.shape[0], (
            f"Helix sparse rows need {num_rows} rows, capacity "
            f"{self.helix_row_kv_lens_cuda.shape[0]}"
        )
        req_idx = torch.cat(
            [
                torch.arange(num_contexts, dtype=torch.long),
                torch.repeat_interleave(
                    torch.arange(num_contexts, self.num_seqs, dtype=torch.long), gen_seq_lens
                ),
            ]
        )
        self.helix_row_req_idx_host[:num_rows].copy_(req_idx)
        # A generation row reads at most top_k entries and never more than its
        # sequence holds locally after this step's appends (an upper bound
        # under the overlap scheduler's provisional lengths).
        upper = (kv_lens[: self.num_seqs].to(torch.long) + self.seq_lens_kv.to(torch.long)).clamp(
            max=self.num_sparse_topk
        )
        upper[:num_contexts] = kv_lens[:num_contexts]
        self.helix_row_kv_lens_host[:num_rows].copy_(upper[req_idx].to(torch.int32))
        self.helix_row_prompt_lens_cpu[:num_contexts].copy_(
            self.prompt_lens_cpu[:num_contexts].to(torch.int32)
        )
        self.helix_row_prompt_lens_cpu[num_contexts:num_rows].fill_(1)
        self.helix_row_request_types_host[:num_contexts].zero_()
        self.helix_row_request_types_host[num_contexts:num_rows].fill_(1)
        for dst, src in (
            (self.helix_row_kv_lens_cuda, self.helix_row_kv_lens_host),
            (self.helix_row_req_idx_cuda, self.helix_row_req_idx_host),
            (self.helix_row_prompt_lens_cuda, self.helix_row_prompt_lens_cpu),
        ):
            dst[:num_rows].copy_(src[:num_rows], non_blocking=True)
        source = self.kv_cache_block_offsets
        num_pools = min(source.shape[0], self.helix_row_block_offsets.shape[0])
        width = min(source.shape[-1], self.helix_row_block_offsets.shape[-1])
        torch.index_select(
            source[:num_pools, :, :, :width],
            1,
            self.helix_row_req_idx_cuda[:num_rows],
            out=self.helix_row_block_offsets[:num_pools, :num_rows, :, :width],
        )
        self._helix_num_rows = num_rows
        self._helix_rows_set = False

    def set_helix_sparse_row_lens(self, gen_rows: torch.Tensor) -> None:
        """Present each generation row with exactly its kept Top-K entries.

        Called by every indexer layer after its Helix merge; a layer that
        shares the previous layer's Top-K reads the same lengths.
        """
        if self._helix_num_rows <= 0:
            return
        num_contexts = self.num_contexts
        torch.sum(
            gen_rows >= 0,
            dim=1,
            dtype=torch.int32,
            out=self.helix_row_kv_lens_cuda[num_contexts : num_contexts + gen_rows.shape[0]],
        )
        self._helix_rows_set = True

    def helix_sparse_empty_rows(self, num_gen_tokens: int) -> Optional[torch.Tensor]:
        """Generation rows whose current Top-K kept nothing on this rank, or None.

        Such a row attends to no local KV, so its partial output must not
        enter the Helix combine.
        """
        if self._helix_num_rows <= 0 or not self._helix_rows_set:
            return None
        num_contexts = self.num_contexts
        return self.helix_row_kv_lens_cuda[num_contexts : num_contexts + num_gen_tokens] == 0

    def helix_spec_row_kv_lens(self, num_gen_tokens: int) -> Optional[torch.Tensor]:
        """Per-token rank-local KV bounds of Helix verify groups, or None."""
        if not (self.enable_helix and self._helix_spec_tokens_valid):
            return None
        return self.helix_kv_bounds[:num_gen_tokens]

    def prepare(self):
        self._validate_ragged_verification()
        TrtllmAttentionMetadata.prepare(self)
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            kv_lens = self._helix_local_kv_lens(cached_token_lens)
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        kv_lens = self._prepare_device_window_kv_lens(kv_lens)

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

        if self.enable_helix:
            self._prepare_helix_sparse_rows(kv_lens)

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
        self.swap_for_draft(
            "host_indexer_k_cache_block_offsets", self.host_draft_indexer_k_cache_block_offsets
        )
        self.swap_for_draft(
            "indexer_k_cache_block_offsets", self.draft_indexer_k_cache_block_offsets
        )
        self.swap_for_draft("host_slot_mapping_fp8", self.host_draft_slot_mapping_fp8)
        self.swap_for_draft("slot_mapping_fp8", self.draft_slot_mapping_fp8)
        self.swap_for_draft("host_slot_mapping_scale", self.host_draft_slot_mapping_scale)
        self.swap_for_draft("slot_mapping_scale", self.draft_slot_mapping_scale)
        self.swap_for_draft("block_table", self.draft_block_table)
        self.swap_for_draft("block_table_expanded", self.draft_block_table_expanded)
        self.swap_for_draft("host_block_table_expanded", self.host_draft_block_table_expanded)
        # recompute_context_kv_gather_mappings() rebinds the aliases below.
        self.record_draft_swap("slot_mapping_fp8_fullkv")
        self.record_draft_swap("slot_mapping_scale_fullkv")
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

    def _apply_helix_gen_ownership(self) -> None:
        """Restrict generation indexer-K writes and scoring to this CP rank's tokens.

        Under Helix each generation token's KV lives on the rank that owns its
        page. The generic slot map appends every generation token at the local
        tail, so here each generation row gets its rank-local slot, or -1 (the
        scatter skips it) when another rank owns the token. Verify groups also
        score one row per token against helix_kv_bounds, its rank-local causal
        extent. Device-only and allocation-free, so safe under graph capture.
        """
        num_ctx_tokens = self.num_ctx_tokens
        num_gen_tokens = self.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return
        if self._helix_spec_tokens_valid:
            local = self.helix_local_slots[:num_gen_tokens].to(torch.int64)
            owned = local >= 0
            req_indices = self.req_idx_per_token[num_ctx_tokens : self.num_tokens].to(torch.int64)
        else:
            # One token per generation row; the active rank appends it locally.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs].to(torch.int64)
            local = gen_kv_lens - 1
            owned = ~self.helix_is_inactive_rank[: self.num_generations]
            req_indices = torch.arange(
                self.num_contexts, self.num_seqs, dtype=torch.int64, device=local.device
            )
        index_head_dim = self.kv_cache_manager.index_head_dim
        use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
        data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
        fp8_indices, scale_indices = _compute_slot_mappings(
            local.clamp(min=0),
            self.indexer_k_cache_block_offsets,
            req_indices,
            index_head_dim,
            self._tokens_per_block,
            self.indexer_quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )
        gen_rows = slice(num_ctx_tokens, self.num_tokens)
        self.slot_mapping_fp8[gen_rows] = torch.where(owned, fp8_indices, -1)
        self.slot_mapping_scale[gen_rows] = torch.where(owned, scale_indices, -1)

        if self._helix_spec_tokens_valid and self.use_expanded_buffers_for_mtp:
            # A rank may see no KV for a token (bound 0); scoring keeps at least
            # one row slot so the schedule never sees an all-empty batch, and
            # the indexer drops those rows' candidates (helix_spec_row_kv_lens).
            bounds = self.helix_kv_bounds[:num_gen_tokens]
            self.kv_lens_expanded_cuda[:num_gen_tokens].copy_(bounds.clamp(min=1))
            scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                self.kv_lens_expanded_cuda[:num_gen_tokens].view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                self.num_sms,
            )
            self.scheduler_metadata_buffer_expanded.copy_(
                scheduler_metadata_buffer_expanded, non_blocking=True
            )

    def on_update_kv_lens(self) -> None:
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in DecoderInputPreparer.preprocess_inputs().
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        TrtllmAttentionMetadata.on_update_kv_lens(self)

        # preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        # Default fused path: collapse the eager DSA decode-metadata chain
        # (req_idx_per_token + slot mappings + the two gen indptr cumsums) into
        # one fused Triton launch. Only for the pure-decode/generation step
        # (num_contexts == 0); see kernels.py and
        # _run_fused_dsa_decode_metadata().
        #
        # The fused kernel produces the five shared decode-metadata outputs but
        # NOT token_positions_cuda, which the eager slot block writes only when
        # use_fp8_ds_mla is set (consumed by the FlashInfer sparse-attention
        # paths for RoPE). Exclude that cache mode so the fused path never leaves
        # it stale; those configs keep the eager chain.
        fused_eligible = (
            _fused_dsa_meta_enabled()
            and self.kv_cache_manager is not None
            and self.num_tokens > 0
            and self.num_generations > 0
            and self.num_contexts == 0
            and not self.use_fp8_ds_mla
            # Helix rewrites the generation slots per rank below.
            and not self.enable_helix
        )

        # The draft loop can rewrite seq_lens after prepare() built this map.
        if self.num_tokens > 0 and not fused_eligible:
            self.req_idx_per_token[: self.num_tokens] = build_req_idx_per_token(
                self.seq_lens_cuda[: self.num_seqs], self.num_tokens
            ).to(self.req_idx_per_token.dtype)

        # The overlap correction may have changed request KV lengths after
        # prepare(); rebuild each ragged row's causal extent on device.
        if self.enable_ragged_verification:
            self.refresh_ragged_row_kv_lens()
            self.refresh_token_major_gen_rows()

        # Uncached DSA prefill aliases these scatter maps as gather maps.
        if self.kv_cache_manager is not None and self.num_tokens > 0 and not fused_eligible:
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
                self.indexer_quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[: self.num_tokens] = fp8_indices
            self.slot_mapping_scale[: self.num_tokens] = scale_indices

        if self.num_generations > 0:
            if not fused_eligible:
                torch.cumsum(
                    self.kv_lens_cuda[
                        self.num_contexts : self.num_seqs
                    ],  # num_contexts should be 0
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
            scheduler_context_lens = self.update_indexer_kv_lens_2d(gen_indexer_kv_lens)
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                gen_indexer_kv_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, self.num_sms
            )
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if self.max_draft_tokens > 0 and not self.use_expanded_buffers_for_mtp:
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    scheduler_context_lens.contiguous(),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True
                )
            if self.use_expanded_buffers_for_mtp:
                kv_lens_expanded, num_tokens = self.expand_per_gen_token(gen_indexer_kv_lens)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(kv_lens_expanded)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
            if self.expand_for_dsl and self.dsl_kernel_atoms > 1:
                num_rows = self.num_generations * self.dsl_kernel_atoms
                atom_lens = gen_indexer_kv_lens.unsqueeze(1) - self.dsl_atom_ctx_offsets
                self.kv_lens_expanded_cuda[:num_rows].copy_(atom_lens.reshape(-1))
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_rows].view(-1, 1),
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

        if fused_eligible:
            if not getattr(self, "_fused_dsa_meta_armed", False):
                logger.info(
                    "Fused DSA decode-metadata "
                    f"kernel armed (num_seqs={self.num_seqs}, "
                    f"num_tokens={self.num_tokens})."
                )
                self._fused_dsa_meta_armed = True
            self._run_fused_dsa_decode_metadata()

        if self.enable_helix and self.kv_cache_manager is not None and self.num_generations > 0:
            self._apply_helix_gen_ownership()

        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

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
        if self.needs_gvr_prior:
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
            self.kv_lens_row_reorder_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                cache_name="kv_lens_row_reorder_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        self._create_radix_aux_buffers(capture_graph)
        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self._draft_sizing_cap),),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.gen_token_repeats_cuda = None
        self.host_gen_token_repeats = None
        self._ragged_num_rows = 0
        self._attn_num_rows = 0
        self.row_kv_lens_cuda = None
        self.row_kv_lens_host = None
        self.row_kv_correction_cuda = None
        self.row_kv_correction_host = None
        self.row_req_idx_cuda = None
        self.row_req_idx_host = None
        self.attn_row_kv_lens_cuda = None
        self.attn_row_kv_correction_cuda = None
        self.attn_row_req_idx_cuda = None
        self.attn_row_kv_lens_host = None
        self.attn_row_kv_correction_host = None
        self.attn_row_req_idx_host = None
        self.attn_row_request_types_host = None
        self.attn_row_prompt_lens_cuda = None
        self.attn_row_prompt_lens_cpu = None
        self.attn_row_block_offsets = None
        self.ragged_mla_cu_q_rows = None
        self.ragged_mla_cu_kv_seqlens = None
        if self.enable_ragged_verification:
            self._create_ragged_verification_buffers(capture_graph)
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self._draft_sizing_cap),
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
                    self.max_num_sequences * (1 + self._draft_sizing_cap),
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
        TrtllmAttentionMetadata.update_spec_dec_param(
            self,
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
        rows_per_sequence = max(1 + max_draft_len, 1 + (max_total_draft_tokens or 0))
        if is_spec_dec_dynamic_tree and spec_tree_manager is not None:
            rows_per_sequence = max(rows_per_sequence, spec_tree_manager._internal_buf_dim)
        resize_radix_workspace = rows_per_sequence != self._radix_rows_per_sequence
        self._radix_rows_per_sequence = rows_per_sequence
        self._draft_alloc_cap = max(
            int(max_total_draft_tokens),
            int(getattr(self, "_draft_alloc_cap", 0) or 0),
        )
        capture_graph = self.is_cuda_graph
        max_gen_tokens = self.max_num_sequences * (1 + self.max_draft_tokens)
        if (
            self.nvfp4_mla_fp8_scratch is not None
            and self.nvfp4_mla_fp8_scratch.shape[0] != max_gen_tokens
        ):
            self._create_nvfp4_mla_generation_buffers(capture_graph=capture_graph)
        previous_cap = int(getattr(self, "_expanded_alloc_draft_cap", -1))
        allocation_cap = max(self._draft_sizing_cap, previous_cap)
        if allocation_cap > previous_cap:
            self._expanded_alloc_draft_cap = allocation_cap
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
            self.create_expanded_buffers(capture_graph=capture_graph)
            # One Helix sparse row per verify token.
            self._create_helix_sparse_row_buffers(capture_graph=capture_graph)
        if resize_radix_workspace:
            self._create_radix_aux_buffers(capture_graph)

    def _ensure_pool_view_cached(self, local_layer_idx: int = 0):
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
        kv_cache_manager = self.kv_cache_manager
        if hasattr(kv_cache_manager, "get_primary_pool"):
            pool = kv_cache_manager.get_primary_pool(local_layer_idx)
        else:
            pool = kv_cache_manager.get_unique_primary_pool()
        if (
            self._pool_cache_valid
            and self._cached_kv_mgr_id == id(kv_cache_manager)
            and self._cached_pool_ptr == pool.data_ptr()
        ):
            return
        self._cached_pool_ptr = pool.data_ptr()
        num_blocks, num_layers = pool.shape[:2]
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_num_pool_tokens = num_blocks * num_layers * self._cached_tokens_per_block
        if pool.dtype in (torch.int8, torch.uint8):
            # FP4 packs two values per byte and therefore cannot use the
            # ordinary [token, head_dim] view. Its gather op reads the raw
            # data/scale pools through their stable host pointers instead.
            self._cached_pool_view = None
        else:
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
        cross-layer fan-out remap (``TRTLLM_DISABLE_DSA_GROUP_REMAP``).

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

        # Without shared-indexer layers every group is a singleton. Skipping
        # the scan also keeps hybrid managers, whose local layers include
        # recurrent layers that own no indexer, from forming groups.
        sparse_metadata_params = getattr(self, "sparse_metadata_params", None)
        if sparse_metadata_params is not None and not getattr(
            sparse_metadata_params, "has_shared_indexer_layers", True
        ):
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(kvm)
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

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        # The DeepGEMM paged-MQA kernel (fp8_paged_mqa_logits) runs a native
        # next_n >= 1 on sm100+: its scheduler tiles the query tokens into
        # BLOCK_Q-sized blocks (num_q_blocks = ceil_div(num_q_tokens, BLOCK_Q)),
        # so any MTP depth is handled without a per-draft-token Q flatten /
        # kv_lens / block_table expansion on Blackwell. sm90 still lacks native
        # MTP support (seq_len 1/2 only) and must expand for max_draft_tokens > 1.
        # TODO:
        # - Drop this sm90 branch (and the expanded buffers) once
        #   fp8_paged_mqa_logits supports an arbitrary next_n on sm90 too.
        use_dsl = self.use_cute_dsl_paged_mqa_logits
        if not self.is_ragged_verify:
            self._ragged_num_rows = 0
            self._attn_num_rows = 0
        else:
            # The DSL paged-MQA path reconstructs rows from one scalar window
            # and cannot express a per-request split.
            use_dsl = False
            assert not self.is_spec_decoding_enabled, (
                "ragged verification requires dense spec-decoding metadata to be disabled"
            )

        helix_spec = self.enable_helix and self._helix_spec_tokens_valid
        if helix_spec:
            # Each Helix verify token has its own rank-local KV bound
            # (helix_kv_bounds), which the in-kernel next_n stagger cannot
            # express, so score one row per token with explicit lengths.
            use_dsl = False
        self.use_expanded_buffers_for_mtp = not use_dsl and (
            helix_spec
            or self.is_ragged_verify
            or (self.max_draft_tokens > 1 and get_sm_version() == 90)
        )
        if self.use_expanded_buffers_for_mtp:
            if self.is_ragged_verify:
                n_gen = self.num_generations
                self.host_gen_token_repeats[:n_gen].copy_(
                    torch.tensor(self.gen_token_repeat_list(), dtype=torch.int64)
                )
                self.gen_token_repeats_cuda[:n_gen].copy_(
                    self.host_gen_token_repeats[:n_gen], non_blocking=True
                )

            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            gen_kv_lens_expanded, num_tokens = self.expand_per_gen_token(gen_kv_lens)
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True
            )

            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_blocks = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts : self.num_seqs, :max_len
                ]
                expanded_blocks, _ = self.expand_per_gen_token(gen_blocks, dim=0)
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(expanded_blocks)
                self.block_table_expanded[:num_tokens].copy_(
                    self.host_block_table_expanded[:num_tokens], non_blocking=True
                )
                self.block_table_expanded.clamp_(min=0)

            if self.is_ragged_verify:
                self._prepare_ragged_row_kv_lens(kv_lens)
                self._prepare_token_major_gen_rows(kv_lens)

        self.expand_for_dsl = (
            use_dsl and self.kv_cache_manager is not None and self.max_draft_tokens >= 1
        )
        if self.expand_for_dsl and self.num_generations > 0:
            next_n = self.gen_token_stride
            kernel_atoms = (1, 2, 3) if self.kv_cache_manager.use_fp4 else (1, 2, 3, 4)
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            max_ctx = int(gen_kv_lens.max().item()) if gen_kv_lens.numel() else 0
            expand_factor, atom, kernel_split_atoms = _pick_dsl_split(
                next_n,
                batch_size=self.num_generations,
                max_ctx=max_ctx,
                num_sms=self.num_sms,
                kernel_atoms=kernel_atoms,
            )
            self.dsl_expand_factor = expand_factor
            self.dsl_atom = atom
            self.dsl_kernel_atoms = kernel_split_atoms
            self.dsl_kernel_next_n = next_n
            if kernel_split_atoms > 1:
                offsets = _atom_ctx_offsets(next_n, kernel_split_atoms)
                if self.dsl_atom_ctx_offsets_key != (next_n, kernel_split_atoms):
                    # Built outside graph capture; on_update_kv_lens only subtracts.
                    self.dsl_atom_ctx_offsets = torch.tensor(
                        offsets, dtype=torch.int32, device="cuda"
                    )
                    self.dsl_atom_ctx_offsets_key = (next_n, kernel_split_atoms)
                num_rows = self.num_generations * kernel_split_atoms
                atom_lens = gen_kv_lens.unsqueeze(1) - torch.tensor(
                    offsets, dtype=gen_kv_lens.dtype, device=gen_kv_lens.device
                )
                atom_lens = atom_lens.reshape(-1)
                self.kv_lens_expanded_host[:num_rows].copy_(atom_lens)
                self.kv_lens_expanded_cuda[:num_rows].copy_(
                    self.kv_lens_expanded_host[:num_rows], non_blocking=True
                )
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
            self.dsl_atom = self.gen_token_stride
            self.dsl_kernel_atoms = 1

    def token_major_gen_view(
        self, token_sparse: Optional[bool] = None
    ) -> Optional[TokenMajorGenView]:
        """Return the ragged generation view expected by ``trtllm.py``.

        Under Helix, token-sparse MLA is presented one row per generation token
        instead, each row as long as its kept Top-K entries; see
        ``_create_helix_sparse_row_buffers``.
        """
        if token_sparse is None:
            token_sparse = getattr(self, "_present_token_sparse", False)
        if token_sparse and not self.is_ragged_verify and self._helix_num_rows > 0:
            num_rows = self._helix_num_rows
            return TokenMajorGenView(
                num_rows=num_rows,
                sequence_length=self.helix_row_kv_lens_cuda[:num_rows],
                host_past_key_value_lengths=self.helix_row_kv_lens_host[:num_rows],
                host_context_lengths=self.helix_row_prompt_lens_cpu[:num_rows],
                prompt_lens_cuda=self.helix_row_prompt_lens_cuda[:num_rows],
                host_request_types=self.helix_row_request_types_host[:num_rows],
                kv_cache_block_offsets=self.helix_row_block_offsets[:, :num_rows],
                max_num_rows=self.helix_row_kv_lens_cuda.shape[0],
            )
        num_rows = self._attn_num_rows
        if not self.is_ragged_verify or num_rows <= 0:
            return None
        return TokenMajorGenView(
            num_rows=num_rows,
            sequence_length=self.attn_row_kv_lens_cuda[:num_rows],
            host_past_key_value_lengths=self.attn_row_kv_lens_host[:num_rows],
            host_context_lengths=self.attn_row_prompt_lens_cpu[:num_rows],
            prompt_lens_cuda=self.attn_row_prompt_lens_cuda[:num_rows],
            host_request_types=self.attn_row_request_types_host[:num_rows],
            kv_cache_block_offsets=self.attn_row_block_offsets[:, :num_rows],
            max_num_rows=self.attn_row_kv_lens_cuda.shape[0],
        )

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
            kv_lens = self._helix_local_kv_lens(cached_token_lens)
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        num_blocks_per_seq = (kv_lens[: self.num_seqs] + tokens_per_block - 1) // tokens_per_block
        # Overlap can overestimate the first decode step's host length until
        # preprocessing applies the accepted-token counts on device. Only list
        # pages the request holds; the corrected device lengths bound reads.
        get_block_counts = getattr(self.kv_cache_manager, "get_allocated_block_counts", None)
        if get_block_counts is not None and self.num_seqs > 0:
            allocated_blocks = get_block_counts(self.request_ids)
            num_blocks_per_seq = torch.minimum(
                num_blocks_per_seq, allocated_blocks.to(num_blocks_per_seq.dtype)
            )
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

        # Under Helix kv_lens are this rank's share of each sequence, which does
        # not bound the global Top-K; the indexer runs and merge_helix_topk
        # reduces the per-rank selections instead.
        if self.num_generations > 0 and self.enable_indexer_skip and not self.enable_helix:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = (
                kv_lens[self.num_contexts : self.num_seqs].max().item()
                <= self.num_sparse_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)


if TYPE_CHECKING:
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
