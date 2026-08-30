# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Footer-scale KV packing for DeepSeek-V4 sparse MLA.

Each token occupies 584 bytes: 448 FP8 values, 64 BF16 RoPE values, and
seven UE8M0 scales plus one padding byte in the page footer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType

from .kernels import deepseek_v4_local_to_global_indices
from .params import DeepseekV4AttentionType

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

    from .metadata import DeepseekV4TrtllmAttentionMetadata

DIM_NOPE = 448
DIM_ROPE = 64
QUANT_TILE = 64
NUM_NOPE_TILES = DIM_NOPE // QUANT_TILE
DATA_ROW_BYTES = DIM_NOPE + DIM_ROPE * 2
FOOTER_ROW_BYTES = NUM_NOPE_TILES + 1
TOKEN_BYTES = DATA_ROW_BYTES + FOOTER_ROW_BYTES
PAGE_SIZE = 64  # the page size the FlashInfer SM120 kernels are built for
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = torch.finfo(_FP8_DTYPE).min


@triton.jit
def _quant_scatter_kernel(
    rows_ptr,  # bf16 [num_tokens, 512]
    loc_ptr,  # int32/int64 [num_tokens] global slot ids, < 0 skips
    pool_fp8_ptr,  # fp8 view of the pool buffer
    pool_bf16_ptr,  # bf16 view of the pool buffer
    pool_u8_ptr,  # uint8 view of the pool buffer
    rows_stride0,
    PAGE_SIZE_C: tl.constexpr,
    PAGE_BYTES_C: tl.constexpr,
    DATA_ROW_BYTES_C: tl.constexpr,
    FOOTER_OFFSET: tl.constexpr,
    FOOTER_ROW_BYTES_C: tl.constexpr,
    DIM_NOPE_C: tl.constexpr,
    DIM_ROPE_C: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    loc = tl.load(loc_ptr + token_id)
    if loc >= 0:
        # Pool byte offsets can exceed int32 even when slot ids do not.
        loc64 = loc.to(tl.int64)
        loc_page = loc64 // PAGE_SIZE_C
        loc_off = loc64 % PAGE_SIZE_C

        if tile_id == NUM_TILES:
            rope_range = tl.arange(0, DIM_ROPE_C)
            in_offsets = token_id * rows_stride0 + DIM_NOPE_C + rope_range
            rope = tl.load(rows_ptr + in_offsets)
            out_offsets = (
                loc_page * (PAGE_BYTES_C // 2)
                + (loc_off * DATA_ROW_BYTES_C + DIM_NOPE_C) // 2
                + rope_range
            )
            tl.store(pool_bf16_ptr + out_offsets, rope)
        else:
            tile_range = tl.arange(0, TILE)
            in_offsets = token_id * rows_stride0 + tile_id * TILE + tile_range
            x = tl.load(rows_ptr + in_offsets).to(tl.float32)

            max_abs = tl.maximum(tl.max(tl.abs(x)), EPS)
            ceil_log2 = tl.math.ceil(tl.log2(max_abs / FP8_MAX))
            scale_inv = tl.exp2(-ceil_log2)
            x_fp8 = tl.clamp(x * scale_inv, FP8_MIN, FP8_MAX).to(pool_fp8_ptr.dtype.element_ty)

            out_offsets = (
                loc_page * PAGE_BYTES_C + loc_off * DATA_ROW_BYTES_C + tile_id * TILE + tile_range
            )
            tl.store(pool_fp8_ptr + out_offsets, x_fp8)

            scale_u8 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
            scale_offset = (
                loc_page * PAGE_BYTES_C + FOOTER_OFFSET + loc_off * FOOTER_ROW_BYTES_C + tile_id
            )
            tl.store(pool_u8_ptr + scale_offset, scale_u8)


def quant_scatter(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    rows_bf16: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> None:
    """Pack and scatter BF16 latent rows into footer-scale pages."""
    page_bytes = page_size * TOKEN_BYTES
    assert pool_u8.dtype == torch.uint8 and pool_u8.is_contiguous()
    assert pool_u8.shape[-1] == page_bytes, (
        f"pool page bytes {pool_u8.shape[-1]} != {page_bytes} (page_size {page_size})"
    )
    assert rows_bf16.dtype == torch.bfloat16
    assert rows_bf16.shape[-1] == DIM_NOPE + DIM_ROPE
    assert loc.dtype in (torch.int32, torch.int64) and loc.is_contiguous()
    num_tokens = rows_bf16.shape[0]
    assert loc.shape[0] == num_tokens
    if num_tokens == 0:
        return

    rows = rows_bf16.contiguous()
    pool_fp8 = pool_u8.view(_FP8_DTYPE)
    pool_bf16 = pool_u8.view(torch.bfloat16)

    _quant_scatter_kernel[(num_tokens, NUM_NOPE_TILES + 1)](
        rows,
        loc,
        pool_fp8,
        pool_bf16,
        pool_u8,
        rows.stride(0),
        PAGE_SIZE_C=page_size,
        PAGE_BYTES_C=page_bytes,
        DATA_ROW_BYTES_C=DATA_ROW_BYTES,
        FOOTER_OFFSET=page_size * DATA_ROW_BYTES,
        FOOTER_ROW_BYTES_C=FOOTER_ROW_BYTES,
        DIM_NOPE_C=DIM_NOPE,
        DIM_ROPE_C=DIM_ROPE,
        TILE=QUANT_TILE,
        NUM_TILES=NUM_NOPE_TILES,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        EPS=1e-8,
    )


def get_pool_2d(
    metadata: DeepseekV4TrtllmAttentionMetadata,
    attn_type: DeepseekV4AttentionType,
    compress_ratio: int,
    page_size: int | None = None,
) -> torch.Tensor:
    """Return a pool-base-relative uint8 view, cached per cache manager."""
    manager = metadata.kv_cache_manager
    cache = getattr(metadata, "_footer_scale_pools", None)
    if cache is None:
        cache = {}
        metadata._footer_scale_pools = cache
    if attn_type == DeepseekV4AttentionType.SWA:
        base_ptr = manager.swa_pool_ptr
        block_tokens = manager.tokens_per_block
        layers = list(manager.pp_layers)
    else:
        base_ptr = manager.compress_pool_ptrs[compress_ratio]
        layers = [
            layer
            for layer in manager.pp_layers
            if manager._compress_ratios[layer] == compress_ratio
        ]
        block_tokens = manager.compressed_block_sizes[layers[0]]

    page_size = min(block_tokens, PAGE_SIZE) if page_size is None else page_size
    assert block_tokens % page_size == 0, (
        f"{attn_type.name} block size {block_tokens} must be divisible by "
        f"footer-scale page size {page_size}"
    )
    key = (id(manager), attn_type, compress_ratio, page_size)
    pool = cache.get(key)
    if pool is not None:
        return pool

    max_end_bytes = 0
    for layer in layers:
        buffer = manager.get_buffers(layer, attn_type)
        offset = buffer.data_ptr() - base_ptr
        assert offset >= 0 and offset % TOKEN_BYTES == 0, (
            f"{attn_type.name} buffer for layer {layer} is not slot-aligned "
            f"to its pool base (offset {offset} bytes)"
        )
        max_end_bytes = max(max_end_bytes, offset + buffer.numel() * buffer.element_size())

    page_bytes = page_size * TOKEN_BYTES
    assert max_end_bytes % page_bytes == 0, (
        f"{attn_type.name} pool extent {max_end_bytes} is not a whole number "
        f"of {page_size}-token footer-scale pages"
    )
    pool = convert_to_torch_tensor(
        TensorWrapper(base_ptr, DataType.UINT8, (max_end_bytes // page_bytes, page_bytes))
    )
    cache[key] = pool
    return pool


def append_swa(
    attn: TrtllmAttention,
    metadata: DeepseekV4TrtllmAttentionMetadata,
    latent_rows: torch.Tensor,
    start_idx: int,
    end_idx: int,
    page_size: int = PAGE_SIZE,
) -> None:
    """Quantize new latent rows and scatter them into the canonical SWA pool."""
    positions = metadata.token_positions_cuda[start_idx:end_idx]
    req_id = metadata.req_idx_per_token[start_idx:end_idx]
    local_layer_idx = metadata.kv_cache_manager.layer_offsets[attn.layer_idx]
    block_table_swa = metadata.sliding_block_tables[
        local_layer_idx, DeepseekV4AttentionType.SWA.value
    ]
    loc = deepseek_v4_local_to_global_indices(
        req_id=req_id,
        block_table_swa=block_table_swa,
        swa_local_indices=positions.unsqueeze(1).contiguous(),
        swa_pool_base_ptr=metadata.sparse_mla_base_ptrs[1],
        swa_buffer_ptr=metadata.swa_buffer_ptrs[attn.layer_idx],
        tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
        token_stride=TOKEN_BYTES,
    ).view(-1)
    swa_pool = get_pool_2d(metadata, DeepseekV4AttentionType.SWA, 1, page_size=page_size)
    quant_scatter(swa_pool, loc, latent_rows, page_size=page_size)


def apply_rope_and_append_swa(
    attn: TrtllmAttention,
    metadata: DeepseekV4TrtllmAttentionMetadata,
    q: torch.Tensor,
    latent_rows: torch.Tensor | None,
    start_idx: int,
    end_idx: int,
    rotary_cos_sin: torch.Tensor,
    is_neox: bool,
    *,
    apply_rope: bool = True,
    page_size: int = PAGE_SIZE,
) -> None:
    """Apply MLA RoPE and append latent rows to the canonical SWA cache."""
    num_tokens = end_idx - start_idx
    if num_tokens == 0:
        return
    if attn.mla_params is None:
        raise ValueError("Footer-scale MLA cache requires MLA parameters")

    positions = metadata.token_positions_cuda[start_idx:end_idx]
    if positions.numel() != num_tokens:
        raise ValueError(
            "Expected one cached token position per footer-scale append, got "
            f"{positions.numel()} positions for {num_tokens=}"
        )
    head_dim = attn.mla_params.kv_lora_rank + attn.mla_params.qk_rope_head_dim
    q_view = q.view(num_tokens, attn.num_heads, head_dim)
    if apply_rope:
        torch.ops.trtllm.mla_rope_inplace(
            q_view,
            positions,
            rotary_cos_sin,
            attn.num_heads,
            attn.mla_params.kv_lora_rank,
            attn.mla_params.qk_rope_head_dim,
            False,
            is_neox,
        )
        if latent_rows is not None:
            torch.ops.trtllm.mla_rope_inplace(
                latent_rows.view(num_tokens, 1, head_dim),
                positions,
                rotary_cos_sin,
                1,
                attn.mla_params.kv_lora_rank,
                attn.mla_params.qk_rope_head_dim,
                False,
                is_neox,
            )

    if latent_rows is not None:
        append_swa(
            attn,
            metadata,
            latent_rows,
            start_idx,
            end_idx,
            page_size=page_size,
        )


@maybe_compile(dynamic=True, options={"max-autotune": True})
def dequant_gather(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Gather footer-scale slots and dequantize them to BF16 latent rows."""
    assert pool_u8.dtype == torch.uint8 and pool_u8.is_contiguous()
    page_bytes = page_size * TOKEN_BYTES
    footer_offset = page_size * DATA_ROW_BYTES
    pages = pool_u8.reshape(-1, page_bytes)
    loc_flat = loc.reshape(-1).to(torch.long)
    page = loc_flat // page_size
    offset = loc_flat % page_size

    data_offsets = offset.unsqueeze(1) * DATA_ROW_BYTES + torch.arange(
        DATA_ROW_BYTES, dtype=torch.long, device=pool_u8.device
    )
    encoded_rows = pages[page.unsqueeze(1), data_offsets]
    scale_offsets = (
        footer_offset
        + offset.unsqueeze(1) * FOOTER_ROW_BYTES
        + torch.arange(NUM_NOPE_TILES, dtype=torch.long, device=pool_u8.device)
    )
    scales = pages[page.unsqueeze(1), scale_offsets]

    nope = encoded_rows[:, :DIM_NOPE].contiguous().view(_FP8_DTYPE).to(torch.float32)
    tile_scales = torch.exp2(scales.to(torch.float32) - 127.0).repeat_interleave(QUANT_TILE, dim=-1)
    rope = encoded_rows[:, DIM_NOPE:].contiguous().view(torch.bfloat16).reshape(-1, DIM_ROPE)
    rows = torch.cat(((nope * tile_scales).to(torch.bfloat16), rope), dim=-1)
    return rows.reshape(*loc.shape, DIM_NOPE + DIM_ROPE)
