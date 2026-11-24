# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA attention metadata of the Kimi K3.1 ULL generation target (Helix CP16, MoE TP16 / EP1).

``RubinUllDSAMetadata`` is the engine's DSA metadata (``attention/backends/sparse/dsa/metadata.py``)
with what this target runs differently, so the engine keeps its behaviour for every other caller:

* ``stages_input_commit``: under the decode-step shell's ``step_prep`` the step's uploads are
  staged into one batched input commit (``pyexecutor/input_commit.py``); the decoder runner opens
  that commit only for a metadata class that declares it.
* ``kv_lens_refresh_once``: ``preprocess_inputs`` refreshes the KV lengths once, after its last
  correction. The engine ``on_update_kv_lens`` rebuilds every output it writes from the step's
  inputs, so the state after the last call does not depend on the calls before it.
* the key-sharded decode Top-K under Helix CP (switchboard ``dsa_helix_key_shard``): the CP ranks
  score their own pages and merge their local Top-K. Its state lives here, not in the engine
  metadata: the merge workspace (``_create_key_shard_buffers``, the last step of ``__post_init__``,
  grown by ``update_spec_dec_param`` for a larger draft ceiling) and each step's decision
  (``_prepare_key_shard_decode``, at the end of ``on_update_kv_lens``). It needs the self-sampling
  GVR, which this target runs on sm_107 too (the engine's capability gate stops at sm_103).
* ``update_for_spec_dec``: ``refresh_device_metadata=False`` leaves the device-side indexer and
  sparse-MLA metadata as the target forward used them, for a draft forward that reads none of it.
* ``prepare_for_indexer_k_cache``: under Helix the Top-K remap table covers the cached and every
  new token of the step, and its columns stop at the batch's KV capacity.
* ``helix_host_gen_slot_rows``: under Helix the indexer's host slot mappings give every
  generation row this rank's slot (the engine keeps the generic mapping for other metadata).
* ``_prepare_helix_sparse_rows``: first the per-step statistics (diagnostics, off on the served
  path): the key-shard merge's counters and the callbacks ``register_step_stats`` adds.

``RubinUllDSABackend`` carries the class: the target returns it from ``attention_backend_for``.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.helix_key_shard import (
    helix_key_shard_enabled,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import (
    KEY_SHARD_TOKENS_PER_BLOCK,
    get_key_shard_merge,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard_rows import key_shard_max_rows
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata,
    _indexer_k_dtype,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger

# Host callbacks run once per prepared step, beside the key-shard merge's statistics
# (``register_step_stats``); e.g. the target's own device counters logged at the same cadence.
_STEP_STATS_CALLBACKS: List = []


# ``_held_launch`` after the held launch ran.
_LAUNCHED = object()


def register_step_stats(callback) -> None:
    """Run ``callback()`` (no arguments) at every ``prepare`` of a ``RubinUllDSAMetadata``, where
    the key-shard merge logs its statistics. Registering the same callback again is a no-op."""
    if callback not in _STEP_STATS_CALLBACKS:
        _STEP_STATS_CALLBACKS.append(callback)


def self_sampling_gvr(metadata: DSAtrtllmAttentionMetadata) -> bool:
    """The engine's self-sampling GVR gate (``sparse/params.py``), with sm_107."""
    return bool(
        metadata.enable_gvr_topk
        and metadata.sparse_metadata_params.use_self_sampling_topk
        and IS_CUTLASS_DSL_AVAILABLE
        and get_sm_version() in (100, 103, 107)
        and metadata.num_sparse_topk in (512, 1024, 2048)
        and metadata._indexer_compress_ratio in (1, 4)
    )


class RubinUllDSAMetadata(DSAtrtllmAttentionMetadata):
    """The DSA metadata of the ULL generation target (see the module docstring)."""

    stages_input_commit: ClassVar[bool] = True
    kv_lens_refresh_once: ClassVar[bool] = True

    # Key-sharded decode Top-K: ``key_shard_world`` is the CP size when active (1 = off);
    # ``key_shard_next_n`` is the verify width of this step's key-sharded rows (0 = this step takes
    # merge_helix_topk); ``key_shard_merge`` is the CP group's merge workspace, or None.
    key_shard_world: int = 1
    key_shard_merge: Optional[object] = None
    key_shard_next_n: int = 0

    def __post_init__(self):
        super().__post_init__()
        self._create_key_shard_buffers(capture_graph=self.is_cuda_graph)

    def update_spec_dec_param(self, *args, **kwargs):
        super().update_spec_dec_param(*args, **kwargs)
        if (
            self.key_shard_merge is not None
            and self.key_shard_merge.rows_max < self._key_shard_rows_cap()
        ):
            # A draft ceiling above the KV cache manager's: grow the key-shard workspace before a step needs
            # it. Every rank of the group makes this call in the same step with the same ceiling, outside
            # graph capture, so the collective allocation lines up.
            self._create_key_shard_buffers(capture_graph=self.is_cuda_graph)

    # A spec-head prep held for the step begin (``defer_spec_head_prep``) and, while
    # ``run_held_step_begin`` runs, the launch that takes it (``_LAUNCHED`` once it ran).
    _held_prep: Optional[dict] = None
    _held_launch: Optional[object] = None

    def defer_spec_head_prep(self, prep: dict) -> bool:
        """Hold the engine's spec-head prep (``fused_spec_head_prep``'s keyword arguments) for
        the ULL step begin when the KV-length refresh uses fused Helix metadata.

        While a prep is held, ``on_update_kv_lens`` defers to ``run_held_step_begin``: every
        device read of the corrected KV lengths and Helix buffers waits for the prep.
        """
        if not self._helix_fused_meta_applies():
            return False
        if self._held_prep is not None:
            raise RuntimeError("the previous step's held spec-head prep was never launched")
        self._held_prep = prep
        return True

    def run_held_step_begin(self, launch) -> bool:
        """Run the KV-length refresh the held prep deferred, its fused Helix metadata launch
        replaced by ``launch(prep, meta_args, meta_kwargs)``, which must launch both (in that
        order, or as one launch). False, having done nothing, when no prep is held."""
        prep = self._held_prep
        if prep is None:
            return False
        self._held_launch = launch
        try:
            self.on_update_kv_lens()
            if self._held_launch is not _LAUNCHED:
                raise RuntimeError("the held spec-head prep's refresh launched no Helix metadata")
        finally:
            self._held_prep = self._held_launch = None
        return True

    def _launch_fused_dsa_helix_decode_metadata(self, *args, **kwargs) -> None:
        launch = self._held_launch
        if launch is None:
            super()._launch_fused_dsa_helix_decode_metadata(*args, **kwargs)
            return
        if launch is _LAUNCHED:
            raise RuntimeError("a second Helix metadata launch in one held refresh")
        self._held_launch = _LAUNCHED
        launch(self._held_prep, args, kwargs)

    def on_update_kv_lens(self) -> None:
        if self._held_prep is not None and self._held_launch is None:
            return  # run_held_step_begin runs it after the held prep
        super().on_update_kv_lens()
        self.key_shard_next_n = 0
        if self.key_shard_world > 1 and self.num_generations > 0:
            self._prepare_key_shard_decode()

    def _key_shard_eligible(self) -> bool:
        """Whether this metadata runs the key-sharded decode Top-K (static configuration checks)."""
        mapping = self.mapping
        if mapping is None or self.is_cross or not self.enable_helix:
            return False
        # Helix: the CP ranks already hold disjoint pages (page g on rank g % cp).
        if not helix_key_shard_enabled():
            return False
        world = mapping.cp_size
        return (
            world > 1
            and world & (world - 1) == 0
            and mapping.tp_size == 1
            and not mapping.enable_attention_dp
            and self.kv_cache_manager is not None
            and _indexer_k_dtype(self.kv_cache_manager) in ("fp8", "fp4")
            and self.use_cute_dsl_paged_mqa_logits
            and self.use_self_sampling_topk
            and self._indexer_compress_ratio == 1
            and self._tokens_per_block == KEY_SHARD_TOKENS_PER_BLOCK
        )

    def _create_key_shard_buffers(self, capture_graph=False) -> None:
        """This rank's local Top-K rows and the CP group's merge workspace."""
        # The first reader of the self-sampling GVR gate, which this target extends to sm_107.
        self.use_self_sampling_topk = self_sampling_gvr(self)
        if not self._key_shard_eligible():
            if self.enable_helix and helix_key_shard_enabled() and self.mapping is not None:
                mapping = self.mapping
                logger.warning_once(
                    "Helix key-sharded decode Top-K on (dsa_helix_key_shard), but the Helix decode indexer "
                    "is not key-shard eligible, so "
                    "merge_helix_topk serves every step: it needs tp 1, a power-of-two cp, no attention DP, FP8 / "
                    "FP4 index keys, the CuTe DSL paged logits, the self-sampling GVR (enable_heuristic_topk + "
                    "use_self_sampling_topk), indexer compress ratio 1 and tokens_per_block "
                    f"{KEY_SHARD_TOKENS_PER_BLOCK}; here tp {mapping.tp_size}, cp {mapping.cp_size}, attention DP "
                    f"{mapping.enable_attention_dp}, DSL logits {self.use_cute_dsl_paged_mqa_logits}, "
                    f"self-sampling GVR {self.use_self_sampling_topk}, compress ratio "
                    f"{self._indexer_compress_ratio}, tokens_per_block {self._tokens_per_block}",
                    key="dsa_helix_key_shard_ineligible",
                )
            return
        world = self.mapping.cp_size
        # The workspace keeps every row's local scores (fp32, max_seq_len / world wide), so it is sized for
        # the largest step this server runs and no more.
        rows_max = self._key_shard_rows_cap()
        # max_blocks_per_seq counts this rank's pages.
        local_blocks = self.kv_cache_manager.max_blocks_per_seq
        self.key_shard_max_local_len = local_blocks * KEY_SHARD_TOKENS_PER_BLOCK
        # The merge workspace spans the TP group of the repurposed mapping, which is the CP group, with
        # tp_rank = cp_rank.
        merge_mapping = self.mapping.repurpose_helix_cp_to_tp()
        self.key_shard_merge = get_key_shard_merge(
            merge_mapping, rows_max, self.num_sparse_topk, self.key_shard_max_local_len
        )
        # One row of the merge's local Top-L per decode row (L from the merge, by world size).
        self.key_shard_local_topk = self.get_empty(
            self.cuda_graph_buffers,
            (rows_max, self.key_shard_merge.local_k),
            cache_name="key_shard_local_topk",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.key_shard_world = world
        logger.info_once(
            f"DSA key-sharded decode indexer on: helix cp {world}, "
            f"rank {merge_mapping.tp_rank}, "
            f"local top-{self.key_shard_merge.local_k}, <= {rows_max} rows.",
            key=f"dsa_key_shard_on_{rows_max}",
        )

    def _key_shard_rows_cap(self) -> int:
        """Rows of the largest decode step this server runs, ``max_batch_size x (max_draft_len + 1)``.

        The engine builds the eager metadata before its first ``update_spec_dec_param``, so its own draft
        length is still 0 there; the draft ceiling the KV cache manager was built with counts as well. The
        eager and the CUDA graph metadata then size the same workspace once.
        """
        manager = self.kv_cache_manager
        server_draft = max(
            int(getattr(manager, "max_draft_len", 0) or 0),
            int(getattr(manager, "max_total_draft_tokens", 0) or 0),
        )
        return key_shard_max_rows(self.max_num_sequences, max(self._draft_sizing_cap, server_draft))

    def _prepare_key_shard_decode(self) -> None:
        """Set ``key_shard_next_n``: the verify width of this Helix decode step's key-sharded Top-K (0 = none)."""
        self.key_shard_next_n = 0
        num_gen = self.num_generations
        num_gen_tokens = self.num_tokens - self.num_ctx_tokens
        if self.is_ragged_verify or num_gen_tokens % num_gen:
            self._warn_helix_key_shard_skip(
                "ragged",
                f"{num_gen} sequences, ragged verify {self.is_ragged_verify}, {num_gen_tokens} rows",
            )
            return
        next_n = num_gen_tokens // num_gen
        if num_gen_tokens > self.key_shard_merge.rows_max:
            raise RuntimeError(
                f"DSA key-sharded decode Top-K: {num_gen_tokens} decode rows exceed the merge workspace's "
                f"{self.key_shard_merge.rows_max} (max_batch_size x (max_draft_len + 1))"
            )
        # The rank's own page table, local lengths and schedule are the Helix decode inputs
        # themselves (see helix_key_shard_inputs); only the per-row bounds must be known.
        if self.helix_spec_row_kv_lens(num_gen_tokens) is None and next_n != 1:
            self._warn_helix_key_shard_skip(
                "bounds", f"{next_n}-row verify windows without Helix verify-group bounds"
            )
            return
        self.key_shard_next_n = next_n

    @staticmethod
    def _warn_helix_key_shard_skip(kind: str, detail: str) -> None:
        """Helix key-sharded merge on: warn once per reason when a Helix decode step takes merge_helix_topk."""
        logger.warning_once(
            f"Helix key-sharded decode Top-K skipped for a step ({detail}); merge_helix_topk serves it",
            key=f"dsa_helix_key_shard_skip_{kind}",
        )

    def helix_key_shard_inputs(self, num_gen_tokens: int):
        """``(page table, local lengths, schedule, per-row local bounds)`` of the Helix key-sharded decode."""
        num_gen = self.num_generations
        ctx_lens = self.gen_indexer_kv_lens_cuda_runtime
        row_lens = self.helix_spec_row_kv_lens(num_gen_tokens)
        return (
            self.indexer_k_cache_block_offsets[self.num_contexts : self.num_seqs],
            ctx_lens,
            self.scheduler_metadata_buffer,
            ctx_lens[:num_gen] if row_lens is None else row_lens,
        )

    def update_for_spec_dec(self, refresh_device_metadata: bool = True):
        """``refresh_device_metadata=False`` leaves the device-side indexer and sparse-MLA
        metadata as the target forward used them (the base metadata's refresh and the cached
        state invalidations still run), for a draft forward that reads none of it; the next
        step's ``on_update_kv_lens`` rebuilds all of it from the step's inputs."""
        if refresh_device_metadata:
            super().update_for_spec_dec(refresh_device_metadata)
            return
        TrtllmAttentionMetadata.update_for_spec_dec(self)
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1
        # device
        TrtllmAttentionMetadata.on_update_kv_lens(self)
        self._invalidate_pool_view_cache()

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
        # Columns past a row's KV length are padding (-1). Under Helix the columns cover
        # cached + every new token of the step, not the provisional local length
        # (_helix_local_kv_lens): with an in-flight predecessor that length counts none of the
        # step's new tokens as this rank's, while the device recompute after the overlap
        # correction can place up to seq_lens_kv of them here, past it and possibly in a new
        # page. Their local Top-K positions must remap to that page, not to -1 or a stale entry.
        # The host cached length assumes full acceptance, so it bounds the device one.
        kv_lens = cached_token_lens + self.seq_lens_kv
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        num_blocks_per_seq = (kv_lens[: self.num_seqs] + tokens_per_block - 1) // tokens_per_block
        # Columns past the batch's KV capacity hold no page (and are never selected).
        num_blocks_per_seq.clamp_(max=pool_indices.shape[1])
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
        self._stage_or_copy(self.block_table[: self.num_seqs, :max_blocks_used], host_block_table)

    def helix_host_gen_slot_rows(self, indexer_params) -> slice | None:
        """Give every generation row its own Helix-aware indexer-K slot in the host slot mappings;
        returns the rows written (the indexer uploads them with its own), or None.

        The scatter pairs K row i with slot i, but under Helix a rank appends only the generation
        tokens it owns, so the counts from new_kv_tokens would leave rows paired with other
        tokens' (or stale) slots. Here the active rank writes its single token at the local tail
        and other ranks get -1, which the scatter skips. Verify groups get placeholders that
        on_update_kv_lens() replaces from the per-token Helix buffers.
        """
        from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _compute_slot_mappings

        num_ctx_tokens = self.num_ctx_tokens
        num_gen_tokens = self.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return None
        if self._helix_spec_tokens_valid:
            fp8_flat_indices = torch.full((num_gen_tokens,), -1, dtype=torch.int64)
            scale_flat_indices = fp8_flat_indices
        else:
            assert num_gen_tokens == self.num_generations, (
                "Helix generation without verify groups appends one token per request"
            )
            gen = slice(self.num_contexts, self.num_seqs)
            inactive = self.helix_is_inactive_rank_cpu[: self.num_generations]
            positions = indexer_params.cached_kv_tokens[gen].to(torch.int64)
            positions = torch.where(inactive, torch.zeros_like(positions), positions)
            req_indices = torch.arange(
                self.num_contexts, self.num_seqs, dtype=torch.int64, device="cpu"
            )
            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                positions,
                self.host_indexer_k_cache_block_offsets,
                req_indices,
                indexer_params.head_dim,
                indexer_params.tokens_per_block,
                indexer_params.quant_block_size,
                data_bytes_per_token=indexer_params.data_bytes_per_token,
            )
            fp8_flat_indices = fp8_flat_indices.masked_fill(inactive, -1)
            scale_flat_indices = scale_flat_indices.masked_fill(inactive, -1)
        gen_rows = slice(num_ctx_tokens, num_ctx_tokens + num_gen_tokens)
        self.host_slot_mapping_fp8[gen_rows] = fp8_flat_indices
        self.host_slot_mapping_scale[gen_rows] = scale_flat_indices
        return gen_rows

    def _prepare_helix_sparse_rows(self, kv_lens: torch.Tensor):
        # The engine prepare runs this right after the indexer plan (Indexer.prepare): the
        # step's statistics come first, as they did there.
        if self.key_shard_merge is not None:
            self.key_shard_merge.maybe_log_stats()
        for callback in _STEP_STATS_CALLBACKS:
            callback()
        super()._prepare_helix_sparse_rows(kv_lens)


class RubinUllDSABackend(DSATrtllmAttention):
    """Carrier of ``RubinUllDSAMetadata``: the engine reads only ``Metadata``."""

    Metadata = RubinUllDSAMetadata


def attention_backend_for(backend_name: str, sparse_params) -> type:
    """``RubinUllDSABackend`` when the engine asks for the TRTLLM DSA backend; raises otherwise."""
    if backend_name.upper() != "TRTLLM" or getattr(sparse_params, "algorithm", None) != "dsa":
        raise ValueError(
            "the Kimi K3.1 ULL generation target runs the TRTLLM DSA attention backend only; got "
            f"{backend_name} with {getattr(sparse_params, 'algorithm', None)!r} sparse parameters"
        )
    return RubinUllDSABackend
