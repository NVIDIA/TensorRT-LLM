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

"""DeepSeek V4 FP8 NoPE cache helpers.

The public DSV4 sparse attention op keeps the BF16 paged cache as its reference
path. These helpers provide the narrower FP8 NoPE storage contract used by
kernel experiments: split normalized KV rows into 448 NoPE dims and 64 RoPE
dims, quantize NoPE values to E4M3 with one E8M0 scale per 128 values, and
store RoPE values as BF16.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...utils.e8m0 import fp8_block_dequant_ref, fp8_block_quant_ref, fp8_block_scale_shape

if TYPE_CHECKING:
    from ..attention_interface import DeepSeekV4PagedResourceHandler

DSV4_FP8_BLOCK_SIZE = 128
DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_NOPE_SCALE_BLOCKS = 4
DSV4_ROPE_DIM = 64
FP8_E4M3_DTYPE = torch.float8_e4m3fn
DSV4_SWA_PAGED_RESOURCE_NAME = "swa"
DSV4_BF16_SWA_CACHE_ARG = "swa_cache"
DSV4_FP8_NOPE_CACHE_ARG = "nope_cache"
DSV4_BF16_ROPE_CACHE_ARG = "rope_cache"
DSV4_E8M0_SCALE_CACHE_ARG = "scale_cache"
DSV4_FP8_NOPE_CACHE_SUFFIX = "swa_nope_cache"
DSV4_BF16_ROPE_CACHE_SUFFIX = "swa_rope_cache"
DSV4_E8M0_SCALE_CACHE_SUFFIX = "swa_scale_cache"
DSV4_FP8_NOPE_CACHE_MUTATED_ARGS = (
    DSV4_FP8_NOPE_CACHE_ARG,
    DSV4_BF16_ROPE_CACHE_ARG,
    DSV4_E8M0_SCALE_CACHE_ARG,
)
DSV4_FP8_NOPE_PAGED_WRITE_OP_NAME = "auto_deploy::torch_deepseek_v4_fp8_nope_paged_cache_write"
DSV4_FP8_NOPE_PAGED_GATHER_OP_NAME = "auto_deploy::torch_deepseek_v4_fp8_nope_paged_cache_gather"
DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON = (
    "FP8 NoPE split cache writes are not consumed by the current DeepSeek V4 cached "
    "attention contract because the registered op still accepts one full-width BF16 "
    "swa_cache and has no NoPE, RoPE, and E8M0 scale cache resources."
)

__all__ = [
    "DSV4_BF16_ROPE_CACHE_ARG",
    "DSV4_BF16_ROPE_CACHE_SUFFIX",
    "DSV4_BF16_SWA_CACHE_ARG",
    "DSV4_E8M0_SCALE_CACHE_ARG",
    "DSV4_E8M0_SCALE_CACHE_SUFFIX",
    "DSV4_FP8_BLOCK_SIZE",
    "DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON",
    "DSV4_FP8_NOPE_CACHE_ARG",
    "DSV4_FP8_NOPE_CACHE_MUTATED_ARGS",
    "DSV4_FP8_NOPE_CACHE_SUFFIX",
    "DSV4_FP8_NOPE_PAGED_GATHER_OP_NAME",
    "DSV4_FP8_NOPE_PAGED_WRITE_OP_NAME",
    "DSV4_HEAD_DIM",
    "DSV4_NOPE_DIM",
    "DSV4_NOPE_SCALE_BLOCKS",
    "DSV4_ROPE_DIM",
    "DSV4_SWA_PAGED_RESOURCE_NAME",
    "DeepSeekV4FP8NopeCacheResourceSpec",
    "DeepSeekV4FP8NopeCacheRows",
    "deepseek_v4_fp8_nope_cache_resource_handlers",
    "deepseek_v4_fp8_nope_cache_resource_specs",
    "gather_deepseek_v4_fp8_nope_paged_cache_rows",
    "quantize_deepseek_v4_fp8_nope_cache_rows",
    "reconstruct_deepseek_v4_fp8_nope_cache_rows",
    "split_deepseek_v4_kv_nope_rope",
    "torch_deepseek_v4_fp8_nope_paged_cache_gather",
    "torch_deepseek_v4_fp8_nope_paged_cache_write",
    "validate_deepseek_v4_fp8_nope_paged_cache_resources",
    "write_deepseek_v4_attention_cache_rows",
    "write_deepseek_v4_bf16_flat_cache_rows",
    "write_deepseek_v4_fp8_nope_flat_cache_rows",
    "write_deepseek_v4_fp8_nope_paged_cache_rows",
]


@dataclass(frozen=True)
class DeepSeekV4FP8NopeCacheRows:
    """Quantized DSV4 KV rows ready for split NoPE/RoPE cache storage."""

    nope: torch.Tensor
    rope: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4FP8NopeCacheResourceSpec:
    """Graph-visible resource contract for one split DSV4 SWA cache tensor."""

    schema_arg: str
    cache_suffix: str
    resource_name: str
    token_shape: tuple[int, ...]
    dtype: torch.dtype


def _get_e8m0_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e8m0fnu", None)


def _default_scale_cache_dtype() -> torch.dtype:
    return _get_e8m0_dtype() or torch.float32


def _requires_raw_byte_copy(dtype: torch.dtype) -> bool:
    e8m0_dtype = _get_e8m0_dtype()
    return dtype == FP8_E4M3_DTYPE or (e8m0_dtype is not None and dtype == e8m0_dtype)


def deepseek_v4_fp8_nope_cache_resource_specs(
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> tuple[DeepSeekV4FP8NopeCacheResourceSpec, ...]:
    """Return the split SWA resource specs used by the FP8 NoPE cache path.

    All three tensors share the DSV4 ``swa`` page table but use distinct graph
    cache arguments and cache suffixes so cached attention can tell them apart
    from the full-width BF16 ``swa_cache`` resource.
    """
    expected_scale_blocks = fp8_block_scale_shape((DSV4_NOPE_DIM,), block_size)[-1]
    return (
        DeepSeekV4FP8NopeCacheResourceSpec(
            schema_arg=DSV4_FP8_NOPE_CACHE_ARG,
            cache_suffix=DSV4_FP8_NOPE_CACHE_SUFFIX,
            resource_name=DSV4_SWA_PAGED_RESOURCE_NAME,
            token_shape=(1, 1, DSV4_NOPE_DIM),
            dtype=FP8_E4M3_DTYPE,
        ),
        DeepSeekV4FP8NopeCacheResourceSpec(
            schema_arg=DSV4_BF16_ROPE_CACHE_ARG,
            cache_suffix=DSV4_BF16_ROPE_CACHE_SUFFIX,
            resource_name=DSV4_SWA_PAGED_RESOURCE_NAME,
            token_shape=(1, 1, DSV4_ROPE_DIM),
            dtype=torch.bfloat16,
        ),
        DeepSeekV4FP8NopeCacheResourceSpec(
            schema_arg=DSV4_E8M0_SCALE_CACHE_ARG,
            cache_suffix=DSV4_E8M0_SCALE_CACHE_SUFFIX,
            resource_name=DSV4_SWA_PAGED_RESOURCE_NAME,
            token_shape=(1, 1, expected_scale_blocks),
            dtype=_default_scale_cache_dtype(),
        ),
    )


def deepseek_v4_fp8_nope_cache_resource_handlers(
    block_size: int = DSV4_FP8_BLOCK_SIZE,
    tokens_per_block: int | None = None,
    max_logical_entries_per_seq: int | None = None,
) -> dict[str, "DeepSeekV4PagedResourceHandler"]:
    """Build resource handlers for the split FP8 NoPE SWA cache tensors."""
    from ..attention_interface import DeepSeekV4PagedResourceHandler

    return {
        spec.cache_suffix: DeepSeekV4PagedResourceHandler(
            resource_name=spec.resource_name,
            token_shape=spec.token_shape,
            dtype=spec.dtype,
            logical_length_divisor=1,
            tokens_per_block=tokens_per_block,
            max_logical_entries_per_seq=max_logical_entries_per_seq,
        )
        for spec in deepseek_v4_fp8_nope_cache_resource_specs(block_size)
    }


def _validate_floating_rows(name: str, rows: torch.Tensor, expected_dim: int) -> None:
    if rows.dim() == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if not rows.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {rows.dtype}")
    if rows.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {rows.shape[-1]}")


def _validate_cache_rows(name: str, cache: torch.Tensor, row_dim: int) -> None:
    if cache.dim() != 2:
        raise ValueError(f"{name} must have rank 2, got rank {cache.dim()}")
    if cache.shape[-1] != row_dim:
        raise ValueError(f"{name} last dimension must be {row_dim}, got {cache.shape[-1]}")


def _validate_fp8_cache_shape(nope_cache: torch.Tensor, rope_cache: torch.Tensor) -> None:
    _validate_cache_rows("nope_cache", nope_cache, DSV4_NOPE_DIM)
    _validate_cache_rows("rope_cache", rope_cache, DSV4_ROPE_DIM)
    if nope_cache.dtype != FP8_E4M3_DTYPE:
        raise TypeError(f"nope_cache must have dtype {FP8_E4M3_DTYPE}, got {nope_cache.dtype}")
    if rope_cache.dtype != torch.bfloat16:
        raise TypeError(f"rope_cache must have dtype torch.bfloat16, got {rope_cache.dtype}")


def _validate_scale_cache_dtype(scale_cache: torch.Tensor) -> None:
    e8m0_dtype = _get_e8m0_dtype()
    allowed_dtypes = [torch.float32, torch.uint8]
    if e8m0_dtype is not None:
        allowed_dtypes.append(e8m0_dtype)
    if scale_cache.dtype not in allowed_dtypes:
        raise TypeError(
            f"scale_cache must have dtype torch.float32, torch.uint8"
            f"{', or ' + str(e8m0_dtype) if e8m0_dtype is not None else ''}, "
            f"got {scale_cache.dtype}"
        )


def _validate_scale_cache(scale_cache: torch.Tensor, block_size: int) -> None:
    expected_scale_blocks = fp8_block_scale_shape((DSV4_NOPE_DIM,), block_size)[-1]
    _validate_cache_rows("scale_cache", scale_cache, expected_scale_blocks)
    _validate_scale_cache_dtype(scale_cache)


def _validate_paged_cache_rank(name: str, cache: torch.Tensor) -> None:
    if cache.dim() not in (3, 4, 5):
        raise ValueError(f"{name} must have rank 3, 4, or 5, got rank {cache.dim()}")


def _validate_fp8_paged_cache_shape(
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int,
) -> int:
    expected_scale_blocks = fp8_block_scale_shape((DSV4_NOPE_DIM,), block_size)[-1]
    _validate_paged_cache_rank("nope_cache", nope_cache)
    _validate_paged_cache_rank("rope_cache", rope_cache)
    _validate_paged_cache_rank("scale_cache", scale_cache)
    if nope_cache.shape[:-1] != rope_cache.shape[:-1]:
        raise ValueError(
            "nope_cache and rope_cache must have identical page geometry before the "
            f"last dimension, got {tuple(nope_cache.shape[:-1])} and "
            f"{tuple(rope_cache.shape[:-1])}"
        )
    if nope_cache.shape[:-1] != scale_cache.shape[:-1]:
        raise ValueError(
            "nope_cache and scale_cache must have identical page geometry before the "
            f"last dimension, got {tuple(nope_cache.shape[:-1])} and "
            f"{tuple(scale_cache.shape[:-1])}"
        )
    if nope_cache.shape[-1] != DSV4_NOPE_DIM:
        raise ValueError(f"nope_cache last dimension must be {DSV4_NOPE_DIM}")
    if rope_cache.shape[-1] != DSV4_ROPE_DIM:
        raise ValueError(f"rope_cache last dimension must be {DSV4_ROPE_DIM}")
    if scale_cache.shape[-1] != expected_scale_blocks:
        raise ValueError(f"scale_cache last dimension must be {expected_scale_blocks}")
    if nope_cache.dtype != FP8_E4M3_DTYPE:
        raise TypeError(f"nope_cache must have dtype {FP8_E4M3_DTYPE}, got {nope_cache.dtype}")
    if rope_cache.dtype != torch.bfloat16:
        raise TypeError(f"rope_cache must have dtype torch.bfloat16, got {rope_cache.dtype}")
    _validate_scale_cache_dtype(scale_cache)
    if rope_cache.device != nope_cache.device or scale_cache.device != nope_cache.device:
        raise ValueError("nope_cache, rope_cache, and scale_cache must be on the same device")
    return expected_scale_blocks


def validate_deepseek_v4_fp8_nope_paged_cache_resources(
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> int:
    """Validate the split FP8 NoPE paged-cache resources.

    Args:
        nope_cache: FP8 E4M3 NoPE paged cache with last dimension ``448``.
        rope_cache: BF16 RoPE paged cache with last dimension ``64``.
        scale_cache: E8M0/raw-byte/FP32 scale paged cache with last dimension
            ``ceil(448 / block_size)``.
        block_size: Number of contiguous NoPE values covered by one scale.

    Returns:
        The expected number of scale blocks per cache row.
    """
    return _validate_fp8_paged_cache_shape(nope_cache, rope_cache, scale_cache, block_size)


def _validate_int_vector(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must have rank 1, got rank {tensor.dim()}")
    if tensor.dtype not in (torch.int32, torch.int64, torch.int):
        raise TypeError(f"{name} must be an int32/int64 tensor, got {tensor.dtype}")


def _validate_scalar_int_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() != 1:
        raise ValueError(f"{name} tensor must have one element, got {tensor.numel()}")
    if tensor.dtype not in (torch.int32, torch.int64, torch.int):
        raise TypeError(f"{name} must be an int32/int64 tensor, got {tensor.dtype}")


def _validate_fp8_nope_paged_cache_write_contract(
    kv_rows: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    input_pos: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int,
) -> None:
    _validate_floating_rows("kv_rows", kv_rows, DSV4_HEAD_DIM)
    _validate_int_vector("cache_loc_host", cache_loc_host)
    _validate_int_vector("cu_num_pages_host", cu_num_pages_host)
    _validate_scalar_int_tensor("input_pos", input_pos)
    validate_deepseek_v4_fp8_nope_paged_cache_resources(
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )


def _flatten_rows(rows: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    if rows.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {rows.shape[-1]}")
    return rows.reshape(-1, expected_dim)


def _normalize_indices(
    cache_indices: torch.Tensor, expected_rows: int, device: torch.device
) -> torch.Tensor:
    if cache_indices.numel() != expected_rows:
        raise ValueError(
            f"cache_indices must have {expected_rows} entries, got {cache_indices.numel()}"
        )
    return cache_indices.reshape(-1).to(device=device, dtype=torch.long)


def _copy_raw_or_numeric(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.shape != src.shape:
        raise ValueError(
            f"dst shape {tuple(dst.shape)} does not match src shape {tuple(src.shape)}"
        )

    if _requires_raw_byte_copy(dst.dtype):
        if not dst.is_contiguous():
            raise ValueError("FP8/E8M0 destinations must be contiguous for raw-byte copy")
        values = src.contiguous() if src.dtype == dst.dtype else src.to(dst.dtype).contiguous()
        dst.view(torch.uint8).copy_(values.view(torch.uint8))
    else:
        values = src.to(dst.dtype).contiguous()
        dst.copy_(values)


def _scale_rows_for_cache(scale_rows: torch.Tensor, scale_cache: torch.Tensor) -> torch.Tensor:
    if scale_cache.dtype != torch.uint8:
        return scale_rows

    if scale_rows.dtype == torch.uint8:
        return scale_rows.contiguous()

    e8m0_dtype = _get_e8m0_dtype()
    if e8m0_dtype is not None and scale_rows.dtype == e8m0_dtype:
        return scale_rows.contiguous().view(torch.uint8)

    e8m0_dtype_name = str(e8m0_dtype) if e8m0_dtype is not None else "E8M0 scale rows"
    raise TypeError(
        "torch.uint8 scale_cache requires raw uint8 scale rows or "
        f"{e8m0_dtype_name}; got {scale_rows.dtype}"
    )


def _index_copy_rows(cache: torch.Tensor, row_indices: torch.Tensor, values: torch.Tensor) -> None:
    if cache.shape[1:] != values.shape[1:]:
        raise ValueError(
            f"cache row shape {tuple(cache.shape[1:])} does not match values row shape "
            f"{tuple(values.shape[1:])}"
        )

    if _requires_raw_byte_copy(cache.dtype):
        if not cache.is_contiguous():
            raise ValueError("FP8/E8M0 cache tensors must be contiguous for raw-byte row copy")
        values = (
            values.contiguous()
            if values.dtype == cache.dtype
            else values.to(cache.dtype).contiguous()
        )
        cache_bytes = cache.view(torch.uint8).reshape(cache.shape[0], -1)
        value_bytes = values.view(torch.uint8).reshape(values.shape[0], -1)
        cache_bytes.index_copy_(0, row_indices, value_bytes)
    else:
        values = values.to(cache.dtype).contiguous()
        cache.index_copy_(0, row_indices, values)


def _index_copy_scale_rows(
    scale_cache: torch.Tensor, row_indices: torch.Tensor, scale_rows: torch.Tensor
) -> None:
    _index_copy_rows(
        scale_cache,
        row_indices,
        _scale_rows_for_cache(scale_rows, scale_cache),
    )


def _cache_token_rows(cache: torch.Tensor) -> torch.Tensor:
    tokens_per_block = _cache_tokens_per_block(cache)
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            if int(cache.shape[2]) >= int(cache.shape[3]):
                token_rows = cache[:, 0, :, 0]
            else:
                token_rows = cache[:, 0, 0, :]
        else:
            token_rows = cache[:, :, 0, 0]
    elif cache.dim() == 4:
        token_rows = cache[:, :, 0]
    elif cache.dim() == 3:
        token_rows = cache
    else:
        raise ValueError(f"cache must have rank 3, 4, or 5, got rank {cache.dim()}")
    return token_rows.reshape(cache.shape[0] * tokens_per_block, cache.shape[-1])


def _input_pos_tensor(input_pos: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(input_pos, torch.Tensor):
        if input_pos.numel() != 1:
            raise ValueError(f"input_pos tensor must have one element, got {input_pos.numel()}")
        return input_pos.reshape(()).to(device=device, dtype=torch.long)
    return torch.tensor(input_pos, dtype=torch.long, device=device)


def _paged_cache_row_indices(
    cache_loc: torch.Tensor,
    cu_num_pages: torch.Tensor,
    seq_idx: int,
    input_pos: int | torch.Tensor,
    row_count: int,
    tokens_per_block: int,
    device: torch.device,
) -> torch.Tensor:
    position_offsets = torch.arange(row_count, dtype=torch.long, device=device)
    input_pos_base = (
        _input_pos_tensor(input_pos, device)
        if isinstance(input_pos, torch.Tensor)
        else int(input_pos)
    )
    positions = input_pos_base + position_offsets
    page_ordinal = torch.div(positions, tokens_per_block, rounding_mode="floor")
    token_offset = positions - page_ordinal * tokens_per_block
    page_table_start = cu_num_pages.to(device=device, dtype=torch.long).reshape(-1)[seq_idx]
    page_table_indices = page_table_start + page_ordinal
    page_indices = (
        cache_loc.to(device=device, dtype=torch.long)
        .reshape(-1)
        .index_select(0, page_table_indices)
    )
    return page_indices * tokens_per_block + token_offset


def split_deepseek_v4_kv_nope_rope(kv_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split DSV4 KV rows into 448 NoPE dims and 64 RoPE dims.

    Args:
        kv_rows: Floating-point tensor with shape ``[..., 512]``.

    Returns:
        ``(nope, rope)`` where ``nope`` has shape ``[..., 448]`` and ``rope``
        has shape ``[..., 64]``.
    """
    _validate_floating_rows("kv_rows", kv_rows, DSV4_HEAD_DIM)
    nope = kv_rows[..., :DSV4_NOPE_DIM].contiguous()
    rope = kv_rows[..., DSV4_NOPE_DIM:].contiguous()
    return nope, rope


def quantize_deepseek_v4_fp8_nope_cache_rows(
    kv_rows: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> DeepSeekV4FP8NopeCacheRows:
    """Quantize DSV4 NoPE dims to FP8 and keep RoPE dims in BF16."""
    nope, rope = split_deepseek_v4_kv_nope_rope(kv_rows)
    nope_fp8, scale = fp8_block_quant_ref(nope, block_size)
    return DeepSeekV4FP8NopeCacheRows(
        nope=nope_fp8,
        rope=rope.to(torch.bfloat16).contiguous(),
        scale=scale,
    )


def reconstruct_deepseek_v4_fp8_nope_cache_rows(
    nope: torch.Tensor,
    rope: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct BF16/FP32 DSV4 KV rows from split FP8-NoPE/BF16-RoPE cache rows."""
    if nope.shape[-1] != DSV4_NOPE_DIM:
        raise ValueError(f"nope last dimension must be {DSV4_NOPE_DIM}, got {nope.shape[-1]}")
    if rope.shape[-1] != DSV4_ROPE_DIM:
        raise ValueError(f"rope last dimension must be {DSV4_ROPE_DIM}, got {rope.shape[-1]}")
    expected_scale_shape = fp8_block_scale_shape(tuple(nope.shape), block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} must match expected shape {expected_scale_shape}"
        )

    nope_dequant = fp8_block_dequant_ref(nope, scale, block_size, dtype=dtype)
    return torch.cat([nope_dequant, rope.to(dtype)], dim=-1).contiguous()


def write_deepseek_v4_fp8_nope_flat_cache_rows(
    kv_rows: torch.Tensor,
    cache_indices: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> DeepSeekV4FP8NopeCacheRows:
    """Write quantized DSV4 rows into split flat NoPE/RoPE/scale caches."""
    _validate_fp8_cache_shape(nope_cache, rope_cache)
    _validate_scale_cache(scale_cache, block_size)

    rows = quantize_deepseek_v4_fp8_nope_cache_rows(kv_rows, block_size)
    row_count = rows.nope.reshape(-1, DSV4_NOPE_DIM).shape[0]
    row_indices = _normalize_indices(cache_indices, row_count, nope_cache.device)

    _index_copy_rows(nope_cache, row_indices, _flatten_rows(rows.nope, DSV4_NOPE_DIM, "nope"))
    _index_copy_rows(rope_cache, row_indices, _flatten_rows(rows.rope, DSV4_ROPE_DIM, "rope"))
    _index_copy_scale_rows(
        scale_cache,
        row_indices.to(device=scale_cache.device),
        rows.scale.reshape(row_count, -1),
    )
    return rows


def write_deepseek_v4_bf16_flat_cache_rows(
    kv_rows: torch.Tensor,
    cache_indices: torch.Tensor,
    bf16_cache: torch.Tensor,
) -> torch.Tensor:
    """Reference fallback: write full DSV4 KV rows into a flat BF16 cache."""
    _validate_floating_rows("kv_rows", kv_rows, DSV4_HEAD_DIM)
    _validate_cache_rows("bf16_cache", bf16_cache, DSV4_HEAD_DIM)
    if bf16_cache.dtype != torch.bfloat16:
        raise TypeError(f"bf16_cache must have dtype torch.bfloat16, got {bf16_cache.dtype}")

    flat_rows = kv_rows.reshape(-1, DSV4_HEAD_DIM).to(torch.bfloat16).contiguous()
    row_indices = _normalize_indices(cache_indices, flat_rows.shape[0], bf16_cache.device)
    _index_copy_rows(bf16_cache, row_indices, flat_rows)
    return flat_rows.reshape(*kv_rows.shape[:-1], DSV4_HEAD_DIM)


def write_deepseek_v4_attention_cache_rows(
    kv_rows: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    bf16_cache: torch.Tensor | None = None,
    nope_cache: torch.Tensor | None = None,
    rope_cache: torch.Tensor | None = None,
    scale_cache: torch.Tensor | None = None,
    use_fp8_nope_cache: bool = False,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> torch.Tensor | DeepSeekV4FP8NopeCacheRows:
    """Write DSV4 rows through either the BF16 fallback or split FP8 NoPE path.

    ``use_fp8_nope_cache=False`` is the default so callers can keep the BF16
    cache as a local debug/reference path while opt-in FP8 cache experiments
    share the same row preparation helper.
    """
    if not use_fp8_nope_cache:
        if bf16_cache is None:
            raise ValueError("bf16_cache must be provided when use_fp8_nope_cache=False")
        return write_deepseek_v4_bf16_flat_cache_rows(kv_rows, cache_indices, bf16_cache)

    if nope_cache is None or rope_cache is None or scale_cache is None:
        missing = [
            name
            for name, cache in (
                ("nope_cache", nope_cache),
                ("rope_cache", rope_cache),
                ("scale_cache", scale_cache),
            )
            if cache is None
        ]
        raise ValueError(f"{', '.join(missing)} must be provided when use_fp8_nope_cache=True")
    return write_deepseek_v4_fp8_nope_flat_cache_rows(
        kv_rows,
        cache_indices,
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )


def _cache_tokens_per_block(cache: torch.Tensor) -> int:
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            token_dim = 2 if int(cache.shape[2]) >= int(cache.shape[3]) else 3
        else:
            token_dim = 1
    elif cache.dim() in (3, 4):
        token_dim = 1
    else:
        raise ValueError(f"cache must have rank 3, 4, or 5, got rank {cache.dim()}")
    return int(cache.shape[token_dim])


def _cache_token_view(cache: torch.Tensor, page_idx: int, token_offset: int) -> torch.Tensor:
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            if int(cache.shape[2]) >= int(cache.shape[3]):
                return cache[page_idx, 0, token_offset, 0]
            return cache[page_idx, 0, 0, token_offset]
        return cache[page_idx, token_offset, 0, 0]
    if cache.dim() == 4:
        return cache[page_idx, token_offset, 0]
    if cache.dim() == 3:
        return cache[page_idx, token_offset]
    raise ValueError(f"cache must have rank 3, 4, or 5, got rank {cache.dim()}")


def _page_for_position(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    position: int,
    tokens_per_block: int,
) -> tuple[int, int]:
    page_ordinal = position // tokens_per_block
    token_offset = position - page_ordinal * tokens_per_block
    page_table_start = int(cu_num_pages_host[seq_idx].item())
    page_table_end = int(cu_num_pages_host[seq_idx + 1].item())
    page_table_idx = page_table_start + page_ordinal
    if page_table_idx >= page_table_end:
        raise ValueError(
            f"Sequence {seq_idx} position {position} needs page ordinal {page_ordinal}, "
            f"but only {page_table_end - page_table_start} page(s) are available"
        )
    return int(cache_loc_host[page_table_idx].item()), token_offset


def write_deepseek_v4_fp8_nope_paged_cache_rows(
    kv_rows: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int | torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> DeepSeekV4FP8NopeCacheRows:
    """Write one sequence of DSV4 rows into split paged NoPE/RoPE/scale caches."""
    expected_scale_blocks = _validate_fp8_paged_cache_shape(
        nope_cache, rope_cache, scale_cache, block_size
    )

    tokens_per_block = _cache_tokens_per_block(nope_cache)
    if _cache_tokens_per_block(rope_cache) != tokens_per_block:
        raise ValueError("rope_cache tokens_per_block must match nope_cache")
    if _cache_tokens_per_block(scale_cache) != tokens_per_block:
        raise ValueError("scale_cache tokens_per_block must match nope_cache")

    rows = quantize_deepseek_v4_fp8_nope_cache_rows(kv_rows, block_size)
    nope_rows = _flatten_rows(rows.nope, DSV4_NOPE_DIM, "nope")
    rope_rows = _flatten_rows(rows.rope, DSV4_ROPE_DIM, "rope")
    scale_rows = rows.scale.reshape(nope_rows.shape[0], expected_scale_blocks)
    row_indices = _paged_cache_row_indices(
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        input_pos,
        nope_rows.shape[0],
        tokens_per_block,
        nope_cache.device,
    )

    _index_copy_rows(_cache_token_rows(nope_cache), row_indices, nope_rows)
    _index_copy_rows(_cache_token_rows(rope_cache), row_indices, rope_rows)
    _index_copy_scale_rows(_cache_token_rows(scale_cache), row_indices, scale_rows)

    return rows


def gather_deepseek_v4_fp8_nope_paged_cache_rows(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather and dequantize DSV4 rows from split paged NoPE/RoPE/scale caches.

    Args:
        cache_loc_host: Page-table entries for the sequence batch.
        cu_num_pages_host: Cumulative page-table offsets per sequence.
        seq_idx: Sequence index into ``cu_num_pages_host``.
        start_pos: First logical token position to gather.
        end_pos: Exclusive logical token position.
        nope_cache: FP8 E4M3 NoPE paged cache with last dimension ``448``.
        rope_cache: BF16 RoPE paged cache with last dimension ``64``.
        scale_cache: E8M0/raw-byte/FP32 scale paged cache with last dimension ``4``.
        block_size: NoPE FP8 block size. DeepSeek V4 uses ``128``.
        dtype: Reconstructed output dtype.

    Returns:
        Reconstructed rows with shape ``[end_pos - start_pos, 512]``.
    """
    _validate_fp8_paged_cache_shape(nope_cache, rope_cache, scale_cache, block_size)
    if end_pos < start_pos:
        raise ValueError(f"end_pos must be >= start_pos, got {end_pos} < {start_pos}")

    tokens_per_block = _cache_tokens_per_block(nope_cache)
    if _cache_tokens_per_block(rope_cache) != tokens_per_block:
        raise ValueError("rope_cache tokens_per_block must match nope_cache")
    if _cache_tokens_per_block(scale_cache) != tokens_per_block:
        raise ValueError("scale_cache tokens_per_block must match nope_cache")

    row_count = end_pos - start_pos
    if row_count == 0:
        return torch.empty(0, DSV4_HEAD_DIM, dtype=dtype, device=nope_cache.device)

    row_indices = _paged_cache_row_indices(
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        start_pos,
        row_count,
        tokens_per_block,
        nope_cache.device,
    )
    nope_rows = _cache_token_rows(nope_cache).index_select(0, row_indices)
    rope_rows = _cache_token_rows(rope_cache).index_select(0, row_indices)
    scale_rows = _cache_token_rows(scale_cache).index_select(0, row_indices)
    return reconstruct_deepseek_v4_fp8_nope_cache_rows(
        nope_rows,
        rope_rows,
        scale_rows,
        block_size=block_size,
        dtype=dtype,
    )


@torch.library.custom_op(
    DSV4_FP8_NOPE_PAGED_WRITE_OP_NAME,
    mutates_args=DSV4_FP8_NOPE_CACHE_MUTATED_ARGS,
)
def torch_deepseek_v4_fp8_nope_paged_cache_write(
    kv_rows: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Graph-visible split FP8 NoPE paged-cache write contract."""
    _validate_fp8_nope_paged_cache_write_contract(
        kv_rows,
        cache_loc_host,
        cu_num_pages_host,
        input_pos,
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )
    write_deepseek_v4_fp8_nope_paged_cache_rows(
        kv_rows,
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        input_pos,
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )
    return kv_rows.new_empty(0)


@torch_deepseek_v4_fp8_nope_paged_cache_write.register_fake
def torch_deepseek_v4_fp8_nope_paged_cache_write_fake(
    kv_rows: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> torch.Tensor:
    del seq_idx
    _validate_fp8_nope_paged_cache_write_contract(
        kv_rows,
        cache_loc_host,
        cu_num_pages_host,
        input_pos,
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )
    return kv_rows.new_empty(0)


@torch.library.custom_op(DSV4_FP8_NOPE_PAGED_GATHER_OP_NAME, mutates_args=())
def torch_deepseek_v4_fp8_nope_paged_cache_gather(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Graph-visible BF16 gather from split FP8 NoPE paged-cache resources."""
    _validate_int_vector("cache_loc_host", cache_loc_host)
    _validate_int_vector("cu_num_pages_host", cu_num_pages_host)
    return gather_deepseek_v4_fp8_nope_paged_cache_rows(
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        start_pos,
        end_pos,
        nope_cache,
        rope_cache,
        scale_cache,
        block_size=block_size,
        dtype=torch.bfloat16,
    )


@torch_deepseek_v4_fp8_nope_paged_cache_gather.register_fake
def torch_deepseek_v4_fp8_nope_paged_cache_gather_fake(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_size: int = DSV4_FP8_BLOCK_SIZE,
) -> torch.Tensor:
    del seq_idx
    _validate_int_vector("cache_loc_host", cache_loc_host)
    _validate_int_vector("cu_num_pages_host", cu_num_pages_host)
    validate_deepseek_v4_fp8_nope_paged_cache_resources(
        nope_cache,
        rope_cache,
        scale_cache,
        block_size,
    )
    if end_pos < start_pos:
        raise ValueError(f"end_pos must be >= start_pos, got {end_pos} < {start_pos}")
    return rope_cache.new_empty((end_pos - start_pos, DSV4_HEAD_DIM)).contiguous()
