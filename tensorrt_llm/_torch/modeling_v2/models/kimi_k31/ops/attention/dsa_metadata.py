# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention metadata of the Kimi K3.1 DSA backend: the engine's DSA metadata for the family's indexer.

``K31DSAtrtllmAttentionMetadata`` inherits the engine's (``attention/backends/sparse/dsa/metadata.py``) fields and every
method that is the engine's; it overrides what differs for Kimi K3.1: the family ``K31Indexer`` and its K-cache manager,
FP8 index keys, no Helix context parallelism, no NVFP4 MLA KV cache and no cross-layer grouped remap, and the step's
host-to-device uploads through the input staging. The indexer's Top-K (GVR on SM 10.3, the CUDA radix Top-K elsewhere)
keeps no state in the metadata. Overrides call the base ``TrtllmAttentionMetadata`` directly where the engine's DSA
metadata would add behaviour this target does not run.

``HelixDSAMetadataMixin`` adds Helix context parallelism to the engine's DSA metadata, for the family metadata classes
that build on that class unchanged: the general target's under Helix (``ops/attention/helix.py``) and the Rubin
targets' (``rubin_dsa_metadata.py``).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata as _EngineDSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    TokenMajorGenView,
    build_req_idx_per_token,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import dsa_prefill_share
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_cache import (
    is_dsa_cache_manager,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import K31Indexer
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    _DG_SCHEDULE_BLOCK_KV,
    _atom_ctx_offsets,
    _compute_slot_mappings,
    _pick_dsl_split,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_key_shard import (
    KeyShardDecode,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    DSAMetadataParams,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fused_dsa_decode_metadata import (
    fused_dsa_decode_metadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (
    get_paged_mqa_logits_metadata as _get_paged_mqa_logits_metadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.paged_mqa_schedule import (
    paged_mqa_schedule,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


@dataclass(init=False)
class K31DSAtrtllmAttentionMetadata(_EngineDSAtrtllmAttentionMetadata):
    """Attention metadata for DSA (DeepSeek Sparse Attention) with indexer state."""

    #: The SM 10.3 perf targets' metadata: the CuTe DSL paged-logits schedule and native logits
    #: tiles up to next_n 8.
    _k31_perf = False
    #: The step's shared indexer prefill plan (dsa_prefill_share, on the DEP perf target), else None.
    indexer_share_plan = None

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
        self.num_ctx_mla_kv_tokens = 0
        self._radix_rows_per_sequence = 1
        TrtllmAttentionMetadata.__init__(self, *args, **kwargs)
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.indexer_max_chunk_size = sparse_metadata_params.indexer_max_chunk_size

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        TrtllmAttentionMetadata.__post_init__(self)
        assert is_dsa_cache_manager(self.kv_cache_manager), (
            "K31DSAtrtllmAttentionMetadata requires DSACacheManager-compatible "
            f"cache manager, got {type(self.kv_cache_manager)}"
        )

        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.num_sparse_topk = sparse_metadata_params.max_sparse_topk
        self.sparse_mla_topk = self.num_sparse_topk
        self.indexer_head_dim = sparse_metadata_params.index_head_dim
        self.indexer_quant_block_size = 128
        self.enable_indexer_skip = sparse_metadata_params.enable_indexer_skip
        capture_graph = self.is_cuda_graph
        # DSA index keys are not compressed: one key per token. The engine's metadata helpers
        # (indexer KV lengths) read the ratio.
        self.compress_ratios = [1]
        self._indexer_compress_ratio = 1
        self._tokens_per_block = self.kv_cache_manager.tokens_per_block

        self.create_buffers_for_mla_rope_append(capture_graph=capture_graph)
        self.create_buffers_for_indexer(capture_graph=capture_graph)
        # The attention-TP perf target's key-sharded decode Top-K (None elsewhere).
        self._k31_key_shard = KeyShardDecode.create(self) if self._k31_perf else None

    def get_adp_token_counts(self) -> List[int]:
        """The input count, plus this rank's indexer prefill-share offers on the DEP perf target."""
        return list(super().get_adp_token_counts()) + dsa_prefill_share.offer_counts(self)

    def set_adp_token_counts(self, counts: List[List[int]]) -> List[int]:
        return super().set_adp_token_counts(dsa_prefill_share.take_plan(self, counts))

    def prepare(self):
        self.indexer_share_plan = None
        TrtllmAttentionMetadata.prepare(self)
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
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
        K31Indexer.prepare(metadata=self)

    def prepare_for_draft_forward(self) -> dict | None:
        """Select native DSA indexer metadata for a draft forward."""
        # Only DSA cache managers have the indexer draft-replay buffers below.
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
            "indexer_share_plan": self.indexer_share_plan,
        }
        # A draft forward never shares the target's indexer prefill.
        self.indexer_share_plan = None
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
            K31Indexer.recompute_slot_mappings(self)
        K31Indexer.recompute_context_kv_gather_mappings(self)

        return saved_state

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        _EngineDSAtrtllmAttentionMetadata.restore_after_draft_forward(self, saved_state)
        if saved_state is not None:
            self.indexer_share_plan = saved_state.get("indexer_share_plan")

    def warmup_cute_dsl_radix_topk(self, next_n: int) -> None:
        """Nothing to compile: the K3.1 indexer runs no CuTe DSL decode Top-K (routing refuses
        use_cute_dsl_topk). The engine's warmup calls this on every DSA metadata."""

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

        # A pure-generation step (num_contexts == 0) builds req_idx_per_token, the slot mappings and the two gen
        # indptr cumsums in one fused Triton launch (_run_fused_dsa_decode_metadata) instead of the eager chain.
        fused_eligible = (
            self.kv_cache_manager is not None
            and self.num_tokens > 0
            and self.num_generations > 0
            and self.num_contexts == 0
        )

        # The draft loop can rewrite seq_lens after prepare() built this map.
        if self.num_tokens > 0 and not fused_eligible:
            self.req_idx_per_token[: self.num_tokens] = build_req_idx_per_token(
                self.seq_lens_cuda[: self.num_seqs], self.num_tokens
            ).to(self.req_idx_per_token.dtype)

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
            index_head_dim = self.kv_cache_manager.index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self._tokens_per_block,
                self.indexer_quant_block_size,
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
            # The perf targets build the schedule with the CuTe DSL kernel (same output, in place).
            get_paged_mqa_logits_metadata = (
                paged_mqa_schedule if self._k31_perf else _get_paged_mqa_logits_metadata
            )
            get_paged_mqa_logits_metadata(
                gen_indexer_kv_lens.view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                self.num_sms,
                self.scheduler_metadata_buffer,
            )
            if self.max_draft_tokens > 0:
                get_paged_mqa_logits_metadata(
                    scheduler_context_lens.contiguous(),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                    self.scheduler_metadata_buffer_full_next_n,
                )
            if self.expand_for_dsl and self.dsl_kernel_atoms > 1:
                num_rows = self.num_generations * self.dsl_kernel_atoms
                atom_lens = gen_indexer_kv_lens.unsqueeze(1) - self.dsl_atom_ctx_offsets
                self.kv_lens_expanded_cuda[:num_rows].copy_(atom_lens.reshape(-1))
                get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_rows].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                    self.scheduler_metadata_buffer_expanded,
                )
            if self.expand_for_dsl and self.dsl_expand_factor > 1:
                expand_factor = self.dsl_expand_factor
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_indexer_kv_lens.repeat_interleave(expand_factor)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(gen_kv_lens_expanded)
                get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                    self.scheduler_metadata_buffer_expanded,
                )

        key_shard = getattr(self, "_k31_key_shard", None)
        if key_shard is not None:
            key_shard.prepare(self, gen_indexer_kv_lens if self.num_generations > 0 else None)

        if fused_eligible:
            if not getattr(self, "_fused_dsa_meta_armed", False):
                logger.info(
                    "Fused DSA decode-metadata "
                    f"kernel armed (num_seqs={self.num_seqs}, "
                    f"num_tokens={self.num_tokens})."
                )
                self._fused_dsa_meta_armed = True
            self._run_fused_dsa_decode_metadata()

        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def _run_fused_dsa_decode_metadata(self):
        """Fill req_idx_per_token + slot mappings + gen indptrs via one Triton
        launch (``fused_dsa_decode_metadata``), replacing the eager chain for the
        pure-decode step. Capture-safe: the launch is pre-compiled at warmup and
        replays inside the decode CUDA graph."""
        num_tokens = self.num_tokens
        num_seqs = self.num_seqs
        # The fused header writes gen_*_indptr[1 : num_seqs + 1] while the eager
        # chain writes [1 : num_generations + 1]; they coincide only because the
        # eligibility gate requires num_contexts == 0 (=> num_seqs ==
        # num_generations). Pin that coupling down explicitly.
        assert num_seqs == self.num_generations, (
            f"fused DSA metadata expects num_seqs ({num_seqs}) == "
            f"num_generations ({self.num_generations})"
        )
        index_head_dim = self.kv_cache_manager.index_head_dim

        fused_dsa_decode_metadata(
            self.seq_lens_cuda[:num_seqs],
            self.kv_lens_cuda[:num_seqs],
            self.indexer_k_cache_block_offsets[:num_seqs],
            self.req_idx_per_token[:num_tokens],
            self.slot_mapping_fp8[:num_tokens],
            self.slot_mapping_scale[:num_tokens],
            self.gen_kv_indptr[: num_seqs + 1],
            self.gen_cached_token_indptr[: num_seqs + 1],
            num_tokens=num_tokens,
            max_query_len=1 + self.max_draft_tokens,
            tokens_per_block=self._tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=self.indexer_quant_block_size,
            data_bytes_per_token=index_head_dim,
        )

    def update_for_spec_dec(self, refresh_device_metadata: bool = True):
        """``refresh_device_metadata=False`` leaves the device-side indexer and
        sparse-MLA metadata as the target forward used them (the cached-state
        invalidations still run), for a draft forward that reads none of it;
        the next step's ``on_update_kv_lens`` rebuilds all of it."""
        TrtllmAttentionMetadata.update_for_spec_dec(self)
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        if refresh_device_metadata:
            self.on_update_kv_lens()
        else:
            TrtllmAttentionMetadata.on_update_kv_lens(self)
            self._invalidate_pool_view_cache()

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
        # Local TopK reused across layers.
        if sparse_metadata_params.has_shared_indexer_layers:
            self.shared_topk_indices = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="shared_topk_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        else:
            self.shared_topk_indices = None

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
        # Allocate these buffers dynamically in K31Indexer.prepare()
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
        # The paged-logits schedule encodes next_n: this buffer holds the one for a verify step
        # (next_n == 1 + max_draft_tokens), `scheduler_metadata_buffer` the one for next_n == 1. Always
        # allocated (a few KB), so `update_spec_dec_param` need not special-case it.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated 2D kv_lens buffer for the paged-logits schedule's 2D context_lens.
        # Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
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
        self._create_radix_aux_buffers(capture_graph)
        # The per-row buffers of the CuTe DSL atom split / in-kernel split of a verify step.
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

    # Resizes the 2D KV-length, expanded and radix buffers when the draft length changes.
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
        resize_radix_workspace = rows_per_sequence != self._radix_rows_per_sequence
        self._radix_rows_per_sequence = rows_per_sequence
        self._draft_alloc_cap = max(
            int(max_total_draft_tokens),
            int(getattr(self, "_draft_alloc_cap", 0) or 0),
        )
        capture_graph = self.is_cuda_graph
        previous_cap = int(getattr(self, "_expanded_alloc_draft_cap", -1))
        allocation_cap = max(self._draft_sizing_cap, previous_cap)
        if allocation_cap > previous_cap:
            self._expanded_alloc_draft_cap = allocation_cap
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
            self.create_expanded_buffers(capture_graph=capture_graph)
        if resize_radix_workspace:
            self._create_radix_aux_buffers(capture_graph)

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
        num_blocks, num_layers = pool.shape[:2]
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

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        # The DeepGEMM paged-MQA kernel (fp8_paged_mqa_logits) runs a native
        # next_n >= 1 on Blackwell: its scheduler tiles the query tokens into
        # BLOCK_Q-sized blocks (num_q_blocks = ceil_div(num_q_tokens, BLOCK_Q)),
        # so any draft depth runs without a per-draft-token Q flatten /
        # kv_lens / block_table expansion. Only the CuTe DSL logits split a verify step.
        use_dsl = self.use_cute_dsl_paged_mqa_logits
        self.expand_for_dsl = (
            use_dsl and self.kv_cache_manager is not None and self.max_draft_tokens >= 1
        )
        if self.expand_for_dsl and self.num_generations > 0:
            next_n = self.gen_token_stride
            # FP8 index keys; the perf targets' native tiles go up to N = 32 x 8 = 256 (one UMMA
            # stage) and read each request's keys once.
            kernel_atoms = (1, 2, 3, 4, 5, 6, 7, 8) if self._k31_perf else (1, 2, 3, 4)
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            # A CUDA graph keeps the split it was captured with, while this method rewrites the
            # expanded block table every step: a graph picks its split from its shape alone.
            max_ctx = (
                0
                if self.is_cuda_graph or not gen_kv_lens.numel()
                else int(gen_kv_lens.max().item())
            )
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

    def _refresh_expanded_block_table(self, repeat_factor: Optional[int] = None):
        """Refresh the active cache's expanded INDEX_KEY page table."""
        if self.kv_cache_manager is None or self.num_generations == 0:
            return
        if repeat_factor is None:
            if not (self.expand_for_dsl and self.dsl_expand_factor > 1):
                return
            repeat_factor = self.dsl_expand_factor

        max_len = self.host_indexer_k_cache_block_offsets.shape[1]
        gen_block_tensor = self.host_indexer_k_cache_block_offsets[
            self.num_contexts : self.num_seqs, :max_len
        ]
        num_tokens = self.num_generations * repeat_factor
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
        kv_lens = cached_token_lens + self.seq_lens_kv
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        num_blocks_per_seq = (kv_lens[: self.num_seqs] + tokens_per_block - 1) // tokens_per_block
        # The host length of an overlap-scheduled generation row assumes that its in-flight step
        # added 1 + max_draft_len tokens. On a request's first decode step that step was its
        # context step, which added one, so the host length runs one token past the capacity
        # reserved for it (prompt + num_extra_kv_tokens + 1 + max_draft_len) and can need one
        # more page than the request holds. preprocess_inputs corrects the device lengths by the
        # accepted counts before any kernel runs, and no kernel reads past those, so the table
        # lists only the pages a row holds and leaves later columns at -1.
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
        dst = self.block_table[: self.num_seqs, :max_blocks_used]
        if dst.is_contiguous():
            dst.copy_(host_block_table, non_blocking=True)
        else:
            dst.copy_(host_block_table, non_blocking=True)

    def prepare_for_mla_rope_append(self, cached_token_lens: torch.Tensor, kv_lens: torch.Tensor):
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[: self.num_contexts].sum().item()
            self.num_ctx_mla_kv_tokens = kv_lens[: self.num_contexts].sum().item()
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
            self.num_ctx_mla_kv_tokens = 0
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
            self.host_req_idx_per_token[: self.num_tokens], non_blocking=True
        )

    def warmup_selfsampling_topk(
        self, next_n: int, batch_sizes: Optional[List[int]] = None
    ) -> None:
        """Nothing to pre-compile here: the indexer's Top-K (GVR on SM 10.3, the CUDA radix Top-K elsewhere) needs
        no metadata-side warmup."""
        return None

    def token_major_gen_view(self, token_sparse: bool = False) -> None:
        """No token-major generation view: every request verifies the same number of tokens."""
        return None


class HelixDSAMetadataMixin:
    """Helix context parallelism on the engine's DSA metadata (``DSAtrtllmAttentionMetadata``), mixed in ahead of it.

    Under Helix each CP rank holds a slice of every sequence's KV cache. The mixin's methods, by lifecycle point:

    * init: ``_create_helix_sparse_row_buffers`` allocates the token-sparse MLA rows, one per generation token; it runs
      again when the speculative update grows the per-token (expanded) buffers.
    * prepare input: ``_helix_local_kv_lens`` gives the rank-local KV lengths, which count only the new generation
      tokens this rank owns, per token when a verify group straddles a page boundary. From them
      ``_helix_indexer_block_table`` builds the indexer K-cache table and ``_prepare_helix_sparse_rows`` lays out the
      sparse rows; ``_helix_skip_indexer`` keeps the indexer on for generation rows, since a rank-local length does
      not bound the global Top-K.
    * attention call: inside ``token_sparse_call`` (or with ``token_sparse``), ``token_major_gen_view`` presents the
      sparse rows, each as long as its kept Top-K entries (``set_helix_sparse_row_lens``, ``helix_sparse_empty_rows``).

    The engine methods it overrides only adapt to those methods, and are the engine's without Helix:
    ``__post_init__`` and ``update_spec_dec_param`` (init), ``prepare_for_indexer_k_cache`` and
    ``prepare_for_skip_indexer`` (prepare input), ``token_major_gen_view`` (attention call). With or without Helix,
    ``_ensure_group_remap_struct`` forms no cross-layer indexer groups for a layout without shared indexer layers
    (every K3.1 one).
    """

    #: Whether the attention call under way is a token-sparse MLA call (``token_sparse_call``).
    _token_sparse_call = False

    # --- init ---

    def _create_helix_sparse_row_buffers(self, capture_graph=False):
        """Allocate the token-major rows token-sparse MLA reads under Helix.

        After the Helix Top-K merge a rank keeps only its share of each row's global Top-K: the kept entries come
        first and the rest of the row is -1. The sparse-MLA kernel attends to the first min(top_k, kv_len) entries of
        a row and does not mask -1 inside that range, so every generation token is presented as its own row whose
        length is its kept count (``set_helix_sparse_row_lens``).
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

    def __post_init__(self):
        super().__post_init__()
        self._create_helix_sparse_row_buffers(capture_graph=self.is_cuda_graph)

    def update_spec_dec_param(self, *args, **kwargs):
        """The engine's update, then the Helix sparse rows again when it grew the per-token (expanded) buffers: one
        Helix sparse row per verify token."""
        previous_cap = int(getattr(self, "_expanded_alloc_draft_cap", -1))
        super().update_spec_dec_param(*args, **kwargs)
        if int(getattr(self, "_expanded_alloc_draft_cap", -1)) > previous_cap:
            self._create_helix_sparse_row_buffers(capture_graph=self.is_cuda_graph)

    # --- prepare input ---

    def _helix_local_kv_lens(self, cached_token_lens: torch.Tensor) -> torch.Tensor:
        """Rank-local KV lengths under Helix CP, as TrtllmAttentionMetadata computes them.

        Context rows append all their tokens. A generation row appends only the new tokens this rank owns: one token
        on the active rank without speculation, and a per-sequence count when a verify group can straddle a page
        boundary. The Helix per-sequence buffers are generation-relative. Under the overlap scheduler these host
        values are provisional; on_update_kv_lens() rederives the device state.
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

    def _helix_indexer_block_table(self) -> None:
        """The engine's indexer K-cache table (``DSAtrtllmAttentionMetadata.prepare_for_indexer_k_cache``) built
        from the rank-local KV lengths of ``_helix_local_kv_lens`` instead of the engine's Helix lengths."""
        # Keep physical slots for the primary-pool TopK conversion below.
        pool_indices = self._update_indexer_k_cache_block_offsets()

        # Build block_table for topk_indices conversion (actual block allocation)
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        kv_lens = self._helix_local_kv_lens(cached_token_lens)
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

    def prepare_for_indexer_k_cache(self):
        if not self.enable_helix or self.kv_cache_manager is None:
            super().prepare_for_indexer_k_cache()
            return
        self._helix_indexer_block_table()

    def _helix_skip_indexer(self, kv_lens: torch.Tensor) -> None:
        """The engine's indexer skip (``DSAtrtllmAttentionMetadata.prepare_for_skip_indexer``) with no generation
        skip: ``kv_lens`` are this rank's share of each sequence, which does not bound the global Top-K, so the
        indexer runs and the Helix Top-K merge reduces the per-rank selections."""
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
        self.skip_indexer_for_gen_reqs = False
        if self.dense_topk_on_device:
            self.prepare_dense_topk_indices(
                kv_lens.to(self.seq_lens_cuda.device, non_blocking=True), device=True
            )
        else:
            self.prepare_dense_topk_indices(kv_lens)

    def prepare_for_skip_indexer(self, kv_lens: torch.Tensor):
        if not self.enable_helix:
            super().prepare_for_skip_indexer(kv_lens)
            return
        self._helix_skip_indexer(kv_lens)

    def _prepare_helix_sparse_rows(self, kv_lens: torch.Tensor) -> None:
        """Lay out one attention row per context request and generation token.

        Host lengths are upper bounds for the op's host-side planning; the device lengths start there and the indexer
        narrows each generation row to its kept count layer by layer.
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

    # --- attention call ---

    @contextlib.contextmanager
    def token_sparse_call(self) -> Iterator[None]:
        """Mark the attention calls inside as token-sparse MLA calls, for ``token_major_gen_view``."""
        self._token_sparse_call = True
        try:
            yield
        finally:
            self._token_sparse_call = False

    def token_major_gen_view(self, token_sparse: bool = False) -> Optional[TokenMajorGenView]:
        """The engine's ragged generation view; under Helix, for a token-sparse MLA call (``token_sparse`` or inside
        ``token_sparse_call``), one row per generation token instead, each row as long as its kept Top-K entries (see
        ``_create_helix_sparse_row_buffers``)."""
        token_sparse = token_sparse or self._token_sparse_call
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
        return super().token_major_gen_view()

    def set_helix_sparse_row_lens(self, gen_rows: torch.Tensor) -> None:
        """Present each generation row with exactly its kept Top-K entries.

        Called by every indexer layer after its Helix merge; a layer that shares the previous layer's Top-K reads the
        same lengths.
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

        Such a row attends to no local KV, so its partial output must not enter the Helix combine.
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

    # --- cross-layer indexer groups ---

    def _ensure_group_remap_struct(self):
        """No groups without shared indexer layers, else the engine's group scan.

        Without shared indexer layers every group is a singleton. Skipping the scan also keeps hybrid managers, whose
        local layers include recurrent layers that own no indexer, from forming groups.
        """
        sparse_metadata_params = getattr(self, "sparse_metadata_params", None)
        if sparse_metadata_params is not None and not getattr(
            sparse_metadata_params, "has_shared_indexer_layers", True
        ):
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(self.kv_cache_manager)
            return self._group_remap_struct
        return super()._ensure_group_remap_struct()
