# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One chunk of the context requests' cached latent prefix, read out of the paged MLA cache (chunked prefill)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class LoadChunkedKvCacheForMla(OpWrapper):
    """``(compressed_kv [N, lora_size], k_pe [N, rope_size])``: for each context request ``b``, the cached rows
    ``[chunked_ld_global_offset[b], chunked_ld_global_offset[b] + len_b)`` of its paged latent cache, packed in request
    order at ``cu_ctx_chunked_kv_lens[b]``, where ``len_b`` is ``cu_ctx_chunked_kv_lens[b + 1] -
    cu_ctx_chunked_kv_lens[b]`` and ``N = num_ctx_cached_tokens`` is the chunk's total.

    The stock MLA chunked prefill reads its cached prefix one chunk per loop this way (the attention metadata's
    ``cu_chunked_seq_len[loop]``, ``chunked_global_offset[loop]`` and ``max_chunk_len_per_loop[loop]``), attends each
    chunk with the softmax statistics kept, and merges the chunks with ``merge_chunked_attention_for_mla``.
    """

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Kimi K3: bf16 latent pool (quant_mode 0), page 64, three requests whose chunks start mid-page "
            "and cross pages",
            spec=dict(tokens_per_block=64, cached=(100, 37, 128), chunk=64, quant_mode=0),
        ),
    )

    note = """
    Only the bf16 latent pool is certified (quant_mode 0, kv_scale_quant_orig None): the rows come back as stored.
    The op reads only the context requests (the first `num_contexts` rows of `kv_cache_block_offsets`); `layer_idx`
    is the layer's row of `host_kv_cache_pool_mapping` (the KV cache manager's `layer_offsets[layer]`).
    `chunked_max_seq_len` must cover the longest request's chunk: a smaller value drops rows silently.
    """

    def __call__(
        self,
        out_dtype: torch.dtype,
        num_contexts: int,
        num_ctx_cached_tokens: int,
        cu_ctx_chunked_kv_lens: torch.Tensor,
        chunked_ld_global_offset: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_quant_orig: Optional[torch.Tensor],
        layer_idx: int,
        lora_size: int,
        rope_size: int,
        tokens_per_block: int,
        chunked_max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv, k_pe = torch.ops.trtllm.load_chunked_kv_cache_for_mla(
            out_dtype,
            num_contexts,
            num_ctx_cached_tokens,
            cu_ctx_chunked_kv_lens,
            chunked_ld_global_offset,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_quant_orig,
            layer_idx,
            lora_size,
            rope_size,
            tokens_per_block,
            chunked_max_seq_len,
            attention_window_size,
            beam_width,
            quant_mode,
        )
        return kv, k_pe

    def reference(
        self,
        rows: torch.Tensor,
        cu_ctx_chunked_kv_lens: torch.Tensor,
        chunked_ld_global_offset: torch.Tensor,
        lora_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The chunk from each request's stored rows ``rows[b]`` (``[len, lora_size + rope_size]``, in position
        order, as the test reads them back from the pool). The pool addressing arguments are the test's business."""
        cu = cu_ctx_chunked_kv_lens.tolist()
        start = chunked_ld_global_offset.tolist()
        parts = [rows[b][start[b] : start[b] + cu[b + 1] - cu[b]] for b in range(len(rows))]
        packed = torch.cat(parts) if parts else rows[0][:0]
        return packed[:, :lora_size], packed[:, lora_size:]

    def is_valid(
        self,
        out_dtype: torch.dtype,
        num_contexts: int,
        num_ctx_cached_tokens: int,
        cu_ctx_chunked_kv_lens: torch.Tensor,
        chunked_ld_global_offset: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_quant_orig: Optional[torch.Tensor],
        layer_idx: int,
        lora_size: int,
        rope_size: int,
        tokens_per_block: int,
        chunked_max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
    ) -> None:
        lens = cu_ctx_chunked_kv_lens.diff()
        assert int(lens.max()) <= chunked_max_seq_len if lens.numel() else True, (
            "chunked_max_seq_len is shorter than a request's chunk; the op drops the rows past it"
        )
        assert int(cu_ctx_chunked_kv_lens[num_contexts]) == num_ctx_cached_tokens, (
            "num_ctx_cached_tokens must be the chunk's total; the op sizes its outputs from it"
        )


load_chunked_kv_cache_for_mla = LoadChunkedKvCacheForMla()
