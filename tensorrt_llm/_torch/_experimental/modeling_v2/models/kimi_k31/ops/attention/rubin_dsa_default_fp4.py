# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The index-key part of the default Rubin target's DSA metadata: FP8 or FP4 index keys.

``RubinDsaDefaultFp4Mixin`` gives the default target's metadata classes
(``rubin_dsa_default_metadata``) the step metadata of an index-key cache of either dtype the cache
manager stores: FP8 keys, as the specialized targets run them, or FP4 keys (two E2M1 codes per
byte, four UE8M0 scale bytes per key). The mixin defines hook methods only (``HOOKS``); the
metadata classes call them, so it overrides no engine or specialized method.

What the index-key dtype changes:

* the generation rows' index-key slots. A key's codes take ``index_head_dim`` bytes of a page
  for FP8 and half as many for FP4; its four scale bytes follow either way
  (``index_k_data_bytes``). The prefill and draft-replay slots come from the engine's indexer
  helpers, which apply the same rule;
* the bound on the verify window. The DSL paged logits score a request's ``next_n`` rows as one
  atom of ``next_n`` x 32 index heads. The FP4 kernel's MMA N tile on sm_107 is 256, as the FP8
  kernel's, so a window of up to 8 rows (k7) is one atom: the metadata plans no split, and a
  captured graph and its replays agree;
* helix8's key-sharded Top-K: the CP ranks score their own pages with the paged logits of either
  dtype, so FP4 index keys keep it.

The rest is the same for both dtypes and stays the specialized metadata's: the prefill plan
(query tiles under the per-call budget, one MQA-logits schedule per tile), the request-major
decode schedule (256-key units, the FP4 kernel's split being two 128-key blocks as the FP8
kernel's), the key-shard step inputs and the Top-K warmup.

Each function below is the specialized metadata's function of the same name at the cache's
index-key dtype (``data_bytes_per_token`` and the FP8-only key-shard rule are the deltas).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Mapping, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _compute_slot_mappings
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import build_req_idx_per_token
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.logger import logger

from .fused_dsa_decode_metadata import fused_dsa_decode_metadata
from .get_paged_mqa_logits_metadata import SCHEDULE_BLOCK_KV, get_paged_mqa_logits_metadata
from .rubin_dsa_metadata import (
    _DSL_MAX_ATOM_N,
    _INDEX_N_HEADS,
    _KEY_SHARD_TOKENS_PER_BLOCK,
    _decode_schedules,
    _helix_fused_applies,
    _key_shard_next_n,
)
from .rubin_fused_dsa_helix_decode_metadata import rubin_fused_dsa_helix_decode_metadata

#: The index-key dtypes the default target's metadata serves.
INDEX_K_DTYPES = ("fp8", "fp4")
#: The FP4 DSL paged logits kernel's MMA N bound on sm_107 (next_n x index heads of one atom).
FP4_DSL_MAX_ATOM_N = 256


def index_k_data_bytes(manager) -> int:
    """Bytes of one index key's codes in a page of ``manager``'s index-key cache: one per value
    for FP8 keys, half as many for FP4 keys. The four scale bytes follow either way. The engine's
    indexer helpers apply the same rule (``use_fp4``) to the prefill and draft-replay slots."""
    head_dim = manager.index_head_dim
    return head_dim // 2 if getattr(manager, "use_fp4", False) else head_dim


def _fused_decode_metadata(md) -> bool:
    num_tokens, num_seqs = md.num_tokens, md.num_seqs
    if not (
        md.kv_cache_manager is not None
        and num_tokens > 0
        and md.num_generations > 0
        and md.num_contexts == 0
        and not md.use_fp8_ds_mla
        and not md.enable_helix
    ):
        return False
    index_head_dim = md.kv_cache_manager.index_head_dim
    fused_dsa_decode_metadata(
        md.seq_lens_cuda[:num_seqs],
        md.kv_lens_cuda[:num_seqs],
        md.indexer_k_cache_block_offsets[:num_seqs],
        md.req_idx_per_token[:num_tokens],
        md.slot_mapping_fp8[:num_tokens],
        md.slot_mapping_scale[:num_tokens],
        md.gen_kv_indptr[: num_seqs + 1],
        md.gen_cached_token_indptr[: num_seqs + 1],
        num_tokens=num_tokens,
        max_query_len=1 + md.max_draft_tokens,
        tokens_per_block=md._tokens_per_block,
        index_head_dim=index_head_dim,
        quant_block_size=md.indexer_quant_block_size,
        data_bytes_per_token=index_k_data_bytes(md.kv_cache_manager),
    )
    return True


def _eager_decode_metadata(md) -> None:
    num_tokens, num_seqs = md.num_tokens, md.num_seqs
    seq_lens = md.seq_lens_cuda[:num_seqs]
    md.req_idx_per_token[:num_tokens] = build_req_idx_per_token(seq_lens, num_tokens).to(
        md.req_idx_per_token.dtype
    )
    if md.kv_cache_manager is None:
        return
    # Runtime cached lengths after the overlap / speculative-decode correction.
    start_positions = md.kv_lens_cuda[:num_seqs] - seq_lens
    req_indices = md.req_idx_per_token[:num_tokens].to(dtype=torch.int64)
    seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
    token_offsets = (
        torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int64)
        - seq_starts[req_indices]
    )
    global_positions = start_positions[req_indices] + token_offsets
    if md.use_fp8_ds_mla:
        md.token_positions_cuda[:num_tokens] = global_positions.to(torch.int32)
    index_head_dim = md.kv_cache_manager.index_head_dim
    fp8_indices, scale_indices = _compute_slot_mappings(
        global_positions,
        md.indexer_k_cache_block_offsets,
        req_indices,
        index_head_dim,
        md._tokens_per_block,
        md.indexer_quant_block_size,
        data_bytes_per_token=index_k_data_bytes(md.kv_cache_manager),
    )
    md.slot_mapping_fp8[:num_tokens] = fp8_indices
    md.slot_mapping_scale[:num_tokens] = scale_indices


def _key_shard_config(md) -> Tuple[int, int, int]:
    mapping = md.mapping
    cp = mapping.cp_size
    eligible = (
        cp > 1
        and cp & (cp - 1) == 0
        and mapping.tp_size == 1
        and not mapping.enable_attention_dp
        and getattr(md.kv_cache_manager, "indexer_k_dtype", "fp8") in INDEX_K_DTYPES
        and md.use_cute_dsl_paged_mqa_logits
        and md.use_self_sampling_topk
        and md._indexer_compress_ratio == 1
        and md._tokens_per_block == _KEY_SHARD_TOKENS_PER_BLOCK
    )
    if not eligible:
        logger.warning_once(
            "Helix DSA indexer: no key-sharded Top-K (it needs tp 1, a power-of-two cp, no "
            "attention DP, FP8 or FP4 index keys, the DSL logits, the self-sampling GVR, "
            f"uncompressed keys and {_KEY_SHARD_TOKENS_PER_BLOCK}-token pages); every step "
            "merges over CP",
            key="rubin_dsa_default_key_shard_ineligible",
        )
        return 1, 0, 0
    manager = md.kv_cache_manager
    server_draft = max(
        int(getattr(manager, "max_draft_len", 0) or 0),
        int(getattr(manager, "max_total_draft_tokens", 0) or 0),
    )
    rows = md.max_num_sequences * (1 + max(md._draft_sizing_cap, server_draft))
    return cp, rows, manager.max_blocks_per_seq * _KEY_SHARD_TOKENS_PER_BLOCK


def _helix_fused_decode_metadata(md) -> None:
    num_seqs, num_tokens = md.num_seqs, md.num_tokens
    index_head_dim = md.kv_cache_manager.index_head_dim
    gen_kv_lens = md.kv_lens_cuda[:num_seqs]
    rubin_fused_dsa_helix_decode_metadata(
        md.seq_lens_cuda[:num_seqs],
        gen_kv_lens,
        md.indexer_k_cache_block_offsets,
        md.helix_local_slots,
        md.helix_kv_bounds,
        md.req_idx_per_token,
        md.slot_mapping_fp8,
        md.slot_mapping_scale,
        md.gen_kv_indptr,
        md.gen_cached_token_indptr,
        md.kv_lens_cuda_2d,
        md.kv_lens_expanded_cuda,
        num_tokens=num_tokens,
        width=min(md.gen_token_stride, md.kv_lens_cuda_2d.shape[1]),
        tokens_per_block=md._tokens_per_block,
        index_head_dim=index_head_dim,
        quant_block_size=md.indexer_quant_block_size,
        data_bytes_per_token=index_k_data_bytes(md.kv_cache_manager),
    )
    gen_indexer_kv_lens = md.get_indexer_kv_lens(gen_kv_lens)
    md.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
    get_paged_mqa_logits_metadata(
        gen_indexer_kv_lens.view(-1, 1), SCHEDULE_BLOCK_KV, md.num_sms, md.scheduler_metadata_buffer
    )
    get_paged_mqa_logits_metadata(
        md.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
        SCHEDULE_BLOCK_KV,
        md.num_sms,
        md.scheduler_metadata_buffer_expanded,
    )


def _helix_gen_ownership(md) -> None:
    num_ctx_tokens = md.num_ctx_tokens
    num_gen_tokens = md.num_tokens - num_ctx_tokens
    if num_gen_tokens <= 0:
        return
    if md._helix_spec_tokens_valid:
        local = md.helix_local_slots[:num_gen_tokens].to(torch.int64)
        owned = local >= 0
        req_indices = md.req_idx_per_token[num_ctx_tokens : md.num_tokens].to(torch.int64)
    else:
        # One token per generation row; the active rank appends it locally.
        local = md.kv_lens_cuda[md.num_contexts : md.num_seqs].to(torch.int64) - 1
        owned = ~md.helix_is_inactive_rank[: md.num_generations]
        req_indices = torch.arange(
            md.num_contexts, md.num_seqs, dtype=torch.int64, device=local.device
        )
    index_head_dim = md.kv_cache_manager.index_head_dim
    fp8_indices, scale_indices = _compute_slot_mappings(
        local.clamp(min=0),
        md.indexer_k_cache_block_offsets,
        req_indices,
        index_head_dim,
        md._tokens_per_block,
        md.indexer_quant_block_size,
        data_bytes_per_token=index_k_data_bytes(md.kv_cache_manager),
    )
    rows = slice(num_ctx_tokens, md.num_tokens)
    md.slot_mapping_fp8[rows] = torch.where(owned, fp8_indices, -1)
    md.slot_mapping_scale[rows] = torch.where(owned, scale_indices, -1)
    if md._helix_spec_tokens_valid and md.use_expanded_buffers_for_mtp:
        md.kv_lens_expanded_cuda[:num_gen_tokens].copy_(
            md.helix_kv_bounds[:num_gen_tokens].clamp(min=1)
        )
        get_paged_mqa_logits_metadata(
            md.kv_lens_expanded_cuda[:num_gen_tokens].view(-1, 1),
            SCHEDULE_BLOCK_KV,
            md.num_sms,
            md.scheduler_metadata_buffer_expanded,
        )


class RubinDsaDefaultFp4Mixin:
    """The index-key hooks of the default target's DSA metadata (FP8 or FP4 index keys)."""

    # The methods this mixin defines: hooks only, none an engine or specialized method name.
    HOOKS: ClassVar[Tuple[str, ...]] = (
        "_fp4_post_init",
        "_fp4_check_verify_window",
        "_fp4_decode_metadata",
        "_fp4_helix_decode_metadata",
        "_fp4_key_shard_config",
    )
    # The DSL one-atom bound of each index-key dtype's paged logits kernel (FP8: the specialized
    # metadata's).
    DSL_MAX_ATOM_N: ClassVar[Mapping[str, int]] = MappingProxyType(
        {"fp8": _DSL_MAX_ATOM_N, "fp4": FP4_DSL_MAX_ATOM_N}
    )

    def _fp4_post_init(self) -> None:
        """Bind the cache's index-key dtype (``index_k_dtype``) and its one-atom bound; refuse
        any dtype but FP8 and FP4, and a cache whose two statements of it disagree."""
        manager = self.kv_cache_manager
        dtype = getattr(manager, "indexer_k_dtype", "fp8") if manager is not None else "fp8"
        if dtype not in INDEX_K_DTYPES:
            raise ValueError(
                f"the default Kimi K3.1 Rubin DSA metadata serves {INDEX_K_DTYPES} index keys; "
                f"the cache stores {dtype!r}"
            )
        if manager is not None and bool(getattr(manager, "use_fp4", False)) != (dtype == "fp4"):
            raise ValueError(
                f"the cache states index keys {dtype!r} and use_fp4="
                f"{getattr(manager, 'use_fp4', False)}"
            )
        self.index_k_dtype = dtype
        self.index_k_dsl_max_atom_n = self.DSL_MAX_ATOM_N[dtype]

    def _fp4_check_verify_window(self) -> None:
        """Refuse a verify window the DSL paged logits of the cache's index-key dtype cannot
        score as one atom; the specialized ``prepare_for_spec_decode`` then plans that atom."""
        use_dsl = (
            self.use_cute_dsl_paged_mqa_logits
            and not self.is_ragged_verify
            and not (self.enable_helix and self._helix_spec_tokens_valid)
        )
        if use_dsl and self.max_draft_tokens >= 1 and self.num_generations > 0:
            next_n = self.gen_token_stride
            if next_n * _INDEX_N_HEADS > self.index_k_dsl_max_atom_n:
                raise ValueError(
                    f"a {next_n}-row verify window does not fit one DSL logits atom of the "
                    f"{self.index_k_dtype} index keys ({next_n} x {_INDEX_N_HEADS} index heads > "
                    f"{self.index_k_dsl_max_atom_n}); the default Kimi K3.1 Rubin target "
                    "verifies at most 7 draft tokens"
                )

    def _fp4_decode_metadata(self) -> None:
        """``RubinDSAMetadata.on_update_kv_lens`` at the cache's index-key dtype: the generation
        step's decode metadata from the current KV lengths (graph-safe: device work only)."""
        TrtllmAttentionMetadata.on_update_kv_lens(self)
        self._invalidate_pool_view_cache()
        fused = _fused_decode_metadata(self)
        if not fused and self.num_tokens > 0:
            _eager_decode_metadata(self)
        if self.enable_ragged_verification:
            self.refresh_ragged_row_kv_lens()
            self.refresh_token_major_gen_rows()
        if self.num_generations > 0:
            nc, ns = self.num_contexts, self.num_seqs
            gen_kv_lens = self.kv_lens_cuda[nc:ns]
            if not fused:
                torch.cumsum(
                    gen_kv_lens, 0, dtype=torch.int64, out=self.gen_kv_indptr[1 : ns - nc + 1]
                )
                torch.cumsum(
                    gen_kv_lens - self.seq_lens_cuda[nc:ns],
                    0,
                    dtype=torch.int64,
                    out=self.gen_cached_token_indptr[1 : ns - nc + 1],
                )
            gen_indexer_kv_lens = self.get_indexer_kv_lens(gen_kv_lens)
            if self.use_expanded_buffers_for_mtp:
                expanded, rows = self.expand_per_gen_token(gen_indexer_kv_lens)
                self.kv_lens_expanded_cuda[:rows].copy_(expanded)
            _decode_schedules(self, gen_indexer_kv_lens)
        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def _fp4_helix_decode_metadata(self) -> None:
        """``RubinDSAHelixMetadata.on_update_kv_lens`` at the cache's index-key dtype: the base
        rebuild plus the Helix ownership, or one launch on a verify step with bounds; then the
        step's key-sharded Top-K width."""
        if _helix_fused_applies(self):
            TrtllmAttentionMetadata.on_update_kv_lens(self)
            self._invalidate_pool_view_cache()
            _helix_fused_decode_metadata(self)
            self._compute_kv_lens_row_reorder()
            self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)
        else:
            self._fp4_decode_metadata()
            if self.kv_cache_manager is not None and self.num_generations > 0:
                _helix_gen_ownership(self)
        self.key_shard_next_n = _key_shard_next_n(self)

    def _fp4_key_shard_config(self) -> Tuple[int, int, int]:
        """helix8's key-sharded Top-K configuration, FP8 or FP4 index keys (``_key_shard_config``)."""
        return _key_shard_config(self)


__all__ = [
    "FP4_DSL_MAX_ATOM_N",
    "INDEX_K_DTYPES",
    "RubinDsaDefaultFp4Mixin",
    "index_k_data_bytes",
]
