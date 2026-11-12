# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation-phase attention through FlashAttention-4 (CuTe DSL) on SM90.

Serves the generation rows of a batch, plain decode (one query token per row) and
linear speculative-decoding verification (``q_len = 1 + K`` tokens per row), for dense
GQA/MHA layers with an unquantized bf16/fp16 paged KV cache (KV cache manager V2).
Per layer it

1. applies the op's NeoX RoPE to the packed Q heads in place and writes the step's rotated K
   and its V at ``kv_len - q_len + j`` of every row into the paged pool, in one launch
   (``rope_write_kv_paged``);
2. runs FA4's paged forward over the whole cache (``seqused_k`` = device KV lengths,
   causal, bottom-right aligned), so the step's tokens see each other causally.

The page table is derived on the device from the attention op's block offsets once per step and
shared by the layers of a forward (``fmha_step_caches``), and every
launch reads only fixed-address tensors, so the layer can be captured in a CUDA graph.
Draft layers (they run on a draft KV-cache manager) are served too unless ``TLLM_FLASH_ATTN4_GENERATION_DRAFT=0``.
In a mixed context+generation batch the library serves the generation rows while the regular op serves
the context phase (``CombinedFmha``), so continuous request arrival keeps the generation rows on FA4.
The kernels are in ``flash_attn4_kernels``, shared with the shared-prefix libraries.

Opt in with ``TLLM_FMHA_LIBS=+flash_attn4_generation``.
"""

from __future__ import annotations

import functools
import os
import weakref
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .combined import generation_rows_uniform
from .flash_attn4_kernels import (
    SPLITS_ENV,
    allocate_split_workspace,
    fa4_forward,
    fa4_paged_generation,
    fa4_paged_generation_split,
    fused_rope_supported,
    generation_splits,
    num_sms,
    page_layout,
    rope_write_kv_paged,
    rotary_dim,
    split_tables,
    write_kv_paged,
)
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LIB_NAME = "flash_attn4_generation"

CHECK_ENV = "TLLM_FLASH_ATTN4_GENERATION_CHECK"
"""Set to 1 to compare the device page table with the host block-offset table on eager steps.
Debugging aid: it synchronizes the device on every eager step."""

DRAFT_ENV = "TLLM_FLASH_ATTN4_GENERATION_DRAFT"
"""Serve the EAGLE-3 draft layer too (default 1). The draft forward swaps the draft KV-cache manager
and its block offsets into the metadata (``draft_kv_cache_context``), so the page table and pool view
below follow it. Set to 0 to keep draft layers on the regular kernel."""


PAGE_TABLE_CACHE_KEY = f"{LIB_NAME}.page_tables"
SPLIT_TABLE_CACHE_KEY = f"{LIB_NAME}.split_tables"
"""Key of the per-step page tables in ``TrtllmAttentionMetadata.fmha_step_caches``."""


MIXED_MIN_KV_ENV = "TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV"
"""Mixed context+generation steps run eagerly, where each FA4 layer call issues several launches
(RoPE and KV write, page table, FA4) against a single attention-op call. The generation rows of a
mixed step go to FA4 only when their mean KV length reaches this many tokens (default 2048), so that
the kernel time FA4 saves outweighs that host cost; 0 always uses FA4."""


@functools.lru_cache(maxsize=4)
def _mixed_min_kv(value: Optional[str]) -> int:
    return int(value) if value else 2048


def _draft_enabled() -> bool:
    return os.environ.get(DRAFT_ENV, "1") != "0"


_split_workspaces: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()
"""Split-KV workspaces of each KV cache manager by call shape: the layers of an executor (and of its
draft model, which has its own manager) share them, concurrent executors in one process never do."""


def _split_workspace(
    kv_cache_manager, rows: int, splits: int, q_len: int, num_heads: int, head_dim: int, q
) -> tuple:
    workspaces = _split_workspaces.setdefault(kv_cache_manager, {})
    key = (q.device, q.dtype, rows, splits, q_len, num_heads, head_dim)
    workspace = workspaces.get(key)
    if workspace is None:
        workspace = workspaces[key] = allocate_split_workspace(
            rows, splits, q_len, num_heads, head_dim, q.dtype, q.device
        )
    return workspace


class FlashAttn4GenerationFmha(PhasedFmha):
    """Generation-phase library on FlashAttention-4; see the module docstring."""

    # Mixed context+generation batches: the thop op serves the context requests through a
    # context-only call and this library the generation rows (``CombinedFmha``).
    pairs_with_whole_batch_context = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # A malformed setting fails when the model is built rather than inside a forward.
        for name in (SPLITS_ENV, MIXED_MIN_KV_ENV):
            value = os.environ.get(name, "").strip()
            if value and not value.isdigit():
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
        self._pool_views: dict[int, torch.Tensor] = {}
        self._pool_manager: Optional[object] = None
        self._logged = False

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if get_sm_version() != 90 or fa4_forward() is None:
            return False
        if (
            getattr(attn, "is_mla_enable", False)
            or getattr(attn, "sparse_params", None) is not None
        ):
            return False
        if getattr(attn, "predicted_tokens_per_seq", 1) != 1 or getattr(
            attn, "attention_chunk_size", None
        ):
            return False
        head_dim = getattr(attn, "head_dim", 0)
        num_heads = getattr(attn, "num_heads", 0)
        num_kv_heads = getattr(attn, "num_kv_heads", 0) or 0
        if head_dim not in (64, 128) or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return False
        if QuantMode(getattr(attn, "quant_mode", 0)).has_kv_cache_quant():
            return False
        position_type = PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
        if position_type.is_rope():
            return fused_rope_supported(attn)
        # ALiBi, relative and deferred embeddings are applied inside the attention op.
        return position_type == PositionEmbeddingType.learned_absolute

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT:
            return False
        if forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        if metadata.num_generations == 0:
            return False
        if phase is None and metadata.num_contexts > 0:
            # Mixed batches are served phase by phase (CombinedFmha).
            return False
        num_gen_tokens = q.shape[0]
        if forward_args.attention_input_type == AttentionInputType.mixed:
            num_gen_tokens -= metadata.num_ctx_tokens
        if not generation_rows_uniform(q, metadata, num_gen_tokens):
            # run_generation gives every row tokens // rows query tokens; a row without drafts
            # next to 1 + K rows needs the regular kernel (FmhaManager keys the selection on this).
            return False
        manager = metadata.kv_cache_manager
        if (
            manager is None
            or not hasattr(manager, "index_mapper")
            or (getattr(manager, "is_draft", False) and not _draft_enabled())
            or metadata.kv_cache_block_offsets is None
            # Per-layer page tables are indexed by local layer, and two layers of one lifecycle
            # group can map different pages; ``_page_table`` reads (and shares) the table of the
            # lifecycle group.
            or getattr(manager, "_use_per_layer_page_tables", False)
        ):
            return False
        if (
            metadata.beam_width != 1
            or metadata.is_cross
            or metadata.enable_helix
            or metadata.is_spec_dec_tree
            or metadata.is_spec_dec_dynamic_tree
        ):
            return False
        if not (
            forward_args.is_fused_qkv and k is None and v is None and forward_args.update_kv_cache
        ):
            return False
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False
        if (
            forward_args.attention_mask_data is not None
            or forward_args.attention_sinks is not None
            or forward_args.relative_attention_bias is not None
            or forward_args.out_scale is not None
            or forward_args.output_sf is not None
        ):
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        window = forward_args.attention_window_size
        if (
            window is not None
            and metadata.max_seq_len is not None
            and window < metadata.max_seq_len
        ):
            return False
        if PositionEmbeddingType(getattr(self.attn, "position_embedding_type", 0)).is_rope():
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None or max_seq_len > (
                self.attn.rotary_cos_sin.numel() // rotary_dim(self.attn)
            ):
                return False
        return True

    def serves_mixed_generation_rows(self, metadata: "TrtllmAttentionMetadata") -> bool:
        min_kv = _mixed_min_kv(os.environ.get(MIXED_MIN_KV_ENV))
        if min_kv <= 0:
            return True
        total_kv = getattr(metadata, "host_total_kv_lens", None)
        if total_kv is None or metadata.num_generations <= 0:
            return False
        return int(total_kv[1]) >= min_kv * metadata.num_generations

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        manager = metadata.kv_cache_manager
        if self._pool_manager is not manager:
            # The executor replaces the manager after memory estimation.
            self._pool_views.clear()
            self._pool_manager = manager
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{LIB_NAME}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def _page_table(self, params: FmhaParams) -> torch.Tensor:
        """[rows, max_pages] int32 pool pages of the phase's rows, derived on the device.

        The attention op's K block offset of a page is ``page * kv_factor`` in the
        layer's pool view, which folds K and V into one page. The layers of a forward share
        one table per (KV manager, pool, rows): the first layer to ask for it after the
        metadata's ``prepare`` derives it in every forward of that step (so a captured graph
        records the derivation), and the later layers reuse it.
        """
        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index = int(
            manager.layer_to_pool_mapping_dict[manager.layer_offsets[self.attn.layer_idx]]
        )
        rows = slice(params.seq_offset, params.seq_offset + params.batch_size)
        offsets = metadata.kv_cache_block_offsets[pool_index, rows, 0, :]
        step_caches = getattr(metadata, "fmha_step_caches", None)
        tables = None if step_caches is None else step_caches.setdefault(PAGE_TABLE_CACHE_KEY, {})
        key = (
            id(manager),
            pool_index,
            params.seq_offset,
            params.batch_size,
            metadata.kv_cache_block_offsets.data_ptr(),
        )
        entry = None if tables is None else tables.get(key)
        if entry is not None and entry[0] != self.attn.layer_idx:
            return entry[1]
        table = torch.div(offsets, int(manager.kv_factor), rounding_mode="floor")
        if tables is not None:
            tables[key] = (self.attn.layer_idx, table)
        if os.environ.get(CHECK_ENV, "0") == "1" and not torch.cuda.is_current_stream_capturing():
            self._check_page_table(params, table)
        return table

    def _split_tables(
        self,
        params: FmhaParams,
        page_table: torch.Tensor,
        kv_lens: torch.Tensor,
        q_len: int,
        splits: int,
        tokens_per_block: int,
    ) -> tuple:
        """``split_tables`` of the phase's rows, shared per step like the page table."""
        metadata = params.meta
        step_caches = getattr(metadata, "fmha_step_caches", None)
        cache = None if step_caches is None else step_caches.setdefault(SPLIT_TABLE_CACHE_KEY, {})
        key = (page_table.data_ptr(), kv_lens.data_ptr(), params.batch_size, q_len, splits)
        entry = None if cache is None else cache.get(key)
        if entry is not None and entry[0] != self.attn.layer_idx:
            return entry[1]
        tables = split_tables(page_table, kv_lens, q_len, splits, tokens_per_block)
        if cache is not None:
            cache[key] = (self.attn.layer_idx, tables)
        return tables

    def _check_page_table(self, params: FmhaParams, table: torch.Tensor) -> None:
        """Compare with the host table read the way the n-sample library reads it."""
        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index, scale, divisor = page_layout(metadata, self.attn.layer_idx)
        request_ids = metadata.request_ids
        beam_stride = int(manager.max_beam_width)
        lens = metadata.kv_lens_runtime[params.seq_offset : params.seq_offset + params.batch_size]
        host = metadata.host_kv_cache_block_offsets
        device = table.cpu()
        tpb = int(metadata.tokens_per_block)
        for i in range(params.batch_size):
            slot = manager.index_mapper.get_index(request_ids[params.seq_offset + i]) * beam_stride
            pages = (int(lens[i]) + tpb - 1) // tpb
            expect = host[pool_index, slot, 0, :pages].long() * scale // divisor
            if not torch.equal(device[i, :pages].long(), expect):
                raise RuntimeError(
                    f"{LIB_NAME} check: device page table of row {i} disagrees with the host table "
                    f"(layer {self.attn.layer_idx}): {device[i, :pages].tolist()[:8]} vs {expect.tolist()[:8]}"
                )

    def run_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        qkv = params.qkv_input
        output = params.output
        if qkv is None or output is None:
            raise RuntimeError(f"{LIB_NAME} requires packed QKV input and an output tensor.")
        rows = params.batch_size
        tokens = params.num_tokens
        if rows <= 0 or tokens % rows:
            raise RuntimeError(
                f"{LIB_NAME}: {tokens} generation tokens are not uniform over {rows} rows."
            )
        q_len = tokens // rows
        if not self._logged:
            self._logged = True
            logger.info_once(
                f"{LIB_NAME}: first generation call ({rows} rows x {q_len} tokens)",
                key=f"{LIB_NAME}_first_call",
            )
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        kv_lens = params.meta.kv_lens_cuda_runtime[params.seq_offset : params.seq_offset + rows]
        rope = PositionEmbeddingType(attn.position_embedding_type).is_rope()
        pool = self._pool_view(params.meta)
        page_table = self._page_table(params)
        # RoPE on Q (in place) and K, and the step's K/V into the pool, in one launch.
        rope_write_kv_paged(
            qkv,
            attn.rotary_cos_sin if rope else None,
            pool,
            page_table,
            kv_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim(attn) if rope else head_dim,
            q_len=q_len,
        )
        q_size = num_heads * head_dim
        q = qkv[:, :q_size].reshape(tokens, num_heads, head_dim)
        sm_scale = 1.0 / (head_dim**0.5 * attn.q_scaling)
        splits = generation_splits(
            rows, num_heads, num_kv_heads, q_len, num_sms(qkv.device.index or 0)
        )
        if splits > 1:
            fa4_paged_generation_split(
                pool,
                q,
                output.view(tokens, num_heads, head_dim),
                self._split_tables(params, page_table, kv_lens, q_len, splits, pool.shape[3]),
                q_len,
                splits,
                sm_scale,
                _split_workspace(
                    params.meta.kv_cache_manager, rows, splits, q_len, num_heads, head_dim, q
                ),
            )
            return
        fa4_paged_generation(
            pool,
            q,
            None,
            None,
            output.view(tokens, num_heads, head_dim),
            page_table,
            kv_lens,
            q_len,
            sm_scale,
        )


__all__ = [
    "FlashAttn4GenerationFmha",
    "LIB_NAME",
    "fa4_paged_generation",
    "fa4_paged_generation_split",
    "generation_splits",
    "rope_write_kv_paged",
    "split_tables",
    "write_kv_paged",
]
