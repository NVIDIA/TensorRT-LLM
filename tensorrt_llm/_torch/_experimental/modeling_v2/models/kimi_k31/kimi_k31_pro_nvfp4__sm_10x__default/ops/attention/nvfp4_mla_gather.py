# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather selected NVFP4 latent-cache rows into FP8 sparse-attention scratch."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Nvfp4MlaKvCacheGather(OpWrapper):
    """Dequantize selected latent rows; write FP8 output and compact int32 indices."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"page {page}, {rows} query rows, two interleaved layers",
            spec=dict(page=page, rows=rows),
        )
        for page in (32, 64, 128)
        for rows in (1, 4, 17)
    )

    def reference(
        self,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        global_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Selected rows before compaction, decoded in FP32 then rounded to E4M3.

        The pools are passed as tensors here: the operation receives their raw
        addresses, which are not torch values. Negative/out-of-pool indices
        have zero placeholders here; the operation marks their compact indices
        as -1 and leaves their output bytes unspecified. Kimi stores all
        576 features in NVFP4 (no HP tail).
        """
        valid = (global_indices >= 0) & (global_indices < data_pool.shape[0])
        indices = global_indices.clamp(0, data_pool.shape[0] - 1).long()
        packed = data_pool.view(torch.uint8)[indices]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
        magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
        values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
        scales = scale_pool.float()[indices].repeat_interleave(16, dim=-1)
        values = values * scales * global_dequant_scale
        return torch.where(valid[..., None], values, 0.0).clamp(-448, 448).to(torch.float8_e4m3fn)

    def __call__(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        global_indices: torch.Tensor,
        output: torch.Tensor,
        compact_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> None:
        torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
            host_pool_pointers,
            host_pool_mapping,
            global_indices,
            output,
            compact_indices,
            global_dequant_scale,
            layer_idx,
            residual_dim,
            num_pool_tokens,
        )


nvfp4_mla_kv_cache_gather = Nvfp4MlaKvCacheGather()


class Nvfp4MlaContextKvCacheGather(OpWrapper):
    """Dequantize selected latent rows; write FP8 output and compact int32 indices."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"page {page}, {rows} query rows, two interleaved layers",
            spec=dict(page=page, rows=rows),
        )
        for page in (32, 64, 128)
        for rows in (1, 4, 17)
    )

    def reference(
        self,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        global_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Selected rows before compaction, decoded in FP32 then rounded to E4M3.

        The pools are passed as tensors here: the operation receives their raw
        addresses, which are not torch values. Negative/out-of-pool indices
        have zero placeholders here; the operation marks their compact indices
        as -1 and leaves their output bytes unspecified. Kimi stores all
        576 features in NVFP4 (no HP tail).
        """
        valid = (global_indices >= 0) & (global_indices < data_pool.shape[0])
        indices = global_indices.clamp(0, data_pool.shape[0] - 1).long()
        packed = data_pool.view(torch.uint8)[indices]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
        magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
        values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
        scales = scale_pool.float()[indices].repeat_interleave(16, dim=-1)
        values = values * scales * global_dequant_scale
        return torch.where(valid[..., None], values, 0.0).clamp(-448, 448).to(torch.float8_e4m3fn)

    def __call__(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        local_topk_indices: torch.Tensor,
        query_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        cu_kv_lengths: torch.Tensor,
        output: torch.Tensor,
        compact_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        total_kv_tokens: int,
        tokens_per_block: int,
        page_stride: int,
        layer_id: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> None:
        torch.ops.trtllm.nvfp4_mla_context_kv_cache_gather(
            host_pool_pointers,
            host_pool_mapping,
            local_topk_indices,
            query_req_indices,
            block_table,
            cu_kv_lengths,
            output,
            compact_indices,
            global_dequant_scale,
            layer_idx,
            total_kv_tokens,
            tokens_per_block,
            page_stride,
            layer_id,
            residual_dim,
            num_pool_tokens,
        )


nvfp4_mla_context_kv_cache_gather = Nvfp4MlaContextKvCacheGather()
