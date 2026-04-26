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

"""Standalone DeepSeek V4 attention kernel building blocks.

The functions in this module are intentionally small and reference-first. They
mirror the math used by the prefill-only DeepSeek V4 scaffold while exposing
semantic custom-op wrappers that later Triton/CUDA kernels can replace.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ...utils.e8m0 import maybe_e8m0_to_fp32
from ..attention_interface import BatchInfo
from .deepseek_v4_attention import torch_deepseek_v4_sparse_attention

FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = torch.finfo(FP8_E4M3_DTYPE).max
FP8_E4M3_MIN = torch.finfo(FP8_E4M3_DTYPE).min
_DSV4_TRITON_LOCAL_NUM_HEADS = 8
_DSV4_TRITON_HEAD_DIM = 512
_DSV4_TRITON_ROPE_DIM = 64
_DSV4_RATIO4_COMPRESS_RATIO = 4
_DSV4_RATIO4_WINDOW_SIZE = 128
_DSV4_RATIO4_TOPK = 512
_DSV4_RATIO4_TOPK_WIDTH = _DSV4_RATIO4_WINDOW_SIZE + _DSV4_RATIO4_TOPK
_DSV4_RATIO4_MAX_COMPRESSED_LEN = 2048
_DSV4_RATIO4_INDEXER_NUM_HEADS = 8
_DSV4_RATIO4_INDEXER_HEAD_DIM = 128
_DSV4_RATIO4_INDEXER_STATE_DIM = 2 * _DSV4_RATIO4_INDEXER_HEAD_DIM
_DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE = 32
_DSV4_TRITON_BLOCK_DIM = triton.next_power_of_2(_DSV4_TRITON_HEAD_DIM)

__all__ = [
    "deepseek_v4_compressor_pool_norm_rope_ref",
    "deepseek_v4_fp8_block_dequant_ref",
    "deepseek_v4_fp8_block_quant_ref",
    "deepseek_v4_indexer_q_rope_quant_ref",
    "deepseek_v4_indexer_fp4_quant_dequant_ref",
    "deepseek_v4_inverse_rope_fp8_output_quant_ref",
    "deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref",
    "deepseek_v4_kv_rmsnorm_rope_cache_insert_ref",
    "deepseek_v4_kv_rmsnorm_rope_ref",
    "deepseek_v4_local_window_topk_idxs",
    "deepseek_v4_q_rmsnorm_rope_ref",
    "deepseek_v4_ratio4_indexer_build_topk_ref",
    "deepseek_v4_ratio4_indexer_compressed_kv_ref",
    "deepseek_v4_ratio4_indexer_q_ref",
    "deepseek_v4_ratio4_indexer_scores_ref",
    "deepseek_v4_ratio4_indexer_topk_ref",
    "deepseek_v4_ratio4_overlap_compress_ref",
    "deepseek_v4_sparse_attention_microkernel_ref",
    "torch_deepseek_v4_compressor_pool_norm_rope",
    "torch_deepseek_v4_indexer_q_rope_quant",
    "torch_deepseek_v4_inverse_rope_fp8_output_quant",
    "torch_deepseek_v4_kv_rmsnorm_rope_cache_insert",
    "torch_deepseek_v4_q_rmsnorm_rope",
    "torch_deepseek_v4_ratio4_indexer_scores",
    "torch_deepseek_v4_ratio4_indexer_topk",
    "torch_deepseek_v4_sparse_attention_microkernel",
    "triton_deepseek_v4_kv_norm_rope_cache_insert",
    "triton_deepseek_v4_q_rmsnorm_rope",
    "triton_deepseek_v4_ratio4_indexer_scores",
    "triton_deepseek_v4_ratio4_indexer_topk",
]


def _get_e8m0_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e8m0fnu", None)


def _scale_dtype() -> torch.dtype:
    return _get_e8m0_dtype() or torch.float32


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _validate_floating(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")


def _validate_rope_dim(rope_dim: int, head_dim: int) -> None:
    if rope_dim < 0:
        raise ValueError(f"rope_dim must be non-negative, got {rope_dim}")
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim must be <= head_dim ({head_dim}), got {rope_dim}")
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


def _rms_norm_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    _validate_floating("x", x)
    compute = x.to(torch.float32)
    output = compute * torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        if weight.shape != (x.shape[-1],):
            raise ValueError(f"weight must have shape ({x.shape[-1]},), got {tuple(weight.shape)}")
        output = output * weight.to(device=x.device, dtype=torch.float32)
    return output.to(x.dtype)


def _apply_rope_ref(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    _validate_floating("x", x)
    _validate_rope_dim(rope_dim, x.shape[-1])
    if rope_dim == 0:
        return x.contiguous()

    if freqs_cis.shape[-1] != rope_dim // 2:
        raise ValueError(
            f"freqs_cis last dim must be rope_dim // 2 ({rope_dim // 2}), got {freqs_cis.shape[-1]}"
        )

    nope = x[..., : x.shape[-1] - rope_dim]
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().reshape(*rope.shape[:-1], -1, 2))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if freqs.dim() == rope_complex.dim() - 1:
        freqs = freqs.unsqueeze(-2)
    rope_out = torch.view_as_real(rope_complex * freqs).flatten(-2).to(x.dtype)
    return torch.cat([nope, rope_out], dim=-1).contiguous()


def _quantize_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    safe_scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(safe_scale))
    exp_bits = torch.clamp(exponent + 127, min=0, max=255).to(torch.uint8)

    e8m0_dtype = _get_e8m0_dtype()
    if e8m0_dtype is not None:
        return exp_bits.view(e8m0_dtype)

    # Fallback for older PyTorch builds: keep the exponent-only contract by
    # rounding positive scales up to powers of two and storing them as FP32.
    return torch.pow(2.0, exp_bits.to(torch.float32) - 127).to(torch.float32)


def _fake_scale(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=_scale_dtype(), device=device)


def _requires_byte_index_copy(dtype: torch.dtype) -> bool:
    e8m0_dtype = _get_e8m0_dtype()
    return dtype == FP8_E4M3_DTYPE or (e8m0_dtype is not None and dtype == e8m0_dtype)


def _index_copy_rows(cache: torch.Tensor, row_indices: torch.Tensor, values: torch.Tensor) -> None:
    if cache.shape[1:] != values.shape[1:]:
        raise ValueError(
            f"cache row shape {tuple(cache.shape[1:])} does not match values row shape "
            f"{tuple(values.shape[1:])}"
        )
    values = values.to(cache.dtype).contiguous()
    if _requires_byte_index_copy(cache.dtype):
        if not cache.is_contiguous():
            raise ValueError("FP8/E8M0 cache tensors must be contiguous for raw-byte row copy.")
        cache_u8 = cache.view(torch.uint8).reshape(cache.shape[0], -1)
        values_u8 = values.view(torch.uint8).reshape(values.shape[0], -1)
        cache_u8.index_copy_(0, row_indices, values_u8)
    else:
        cache.index_copy_(0, row_indices, values)


def _insert_flat_cache(
    cache: torch.Tensor, cache_indices: torch.Tensor, values: torch.Tensor
) -> None:
    flat_values = values.reshape(-1, values.shape[-1])
    if cache_indices.numel() != flat_values.shape[0]:
        raise ValueError(
            f"cache_indices must have {flat_values.shape[0]} entries, got {cache_indices.numel()}"
        )
    if flat_values.numel() == 0:
        return
    flat_indices = cache_indices.reshape(-1).to(device=cache.device, dtype=torch.long)
    _index_copy_rows(cache, flat_indices, flat_values)


def _validate_int_metadata(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must have rank 1, got rank {tensor.dim()}")
    if tensor.dtype not in (torch.int32, torch.int64, torch.int):
        raise TypeError(f"{name} must be an int32/int64 tensor, got {tensor.dtype}")


def _validate_norm_weight_for_dim(
    norm_weight: Optional[torch.Tensor],
    dim: int,
) -> None:
    if norm_weight is None:
        return
    _validate_floating("norm_weight", norm_weight)
    if norm_weight.shape != (dim,):
        raise ValueError(f"norm_weight must have shape ({dim},), got {tuple(norm_weight.shape)}")


def _validate_dsv4_freqs(
    freqs_cis: torch.Tensor,
    batch_size: int,
    seq_len: int,
    rope_dim: int,
) -> None:
    _validate_rank("freqs_cis", freqs_cis, 3)
    if not freqs_cis.is_complex():
        raise TypeError(f"freqs_cis must be complex, got {freqs_cis.dtype}")
    expected_shape = (batch_size, seq_len, rope_dim // 2)
    if freqs_cis.shape != expected_shape:
        raise ValueError(
            f"freqs_cis must have shape {expected_shape}, got {tuple(freqs_cis.shape)}"
        )


def _validate_triton_q_rmsnorm_rope_contract(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor],
) -> None:
    _validate_rank("q", q, 4)
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bfloat16 for the DSV4 Triton contract, got {q.dtype}")
    _validate_rope_dim(rope_dim, q.shape[-1])
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        raise ValueError(f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    batch_size, seq_len, num_heads, head_dim = q.shape
    if num_heads != _DSV4_TRITON_LOCAL_NUM_HEADS:
        raise ValueError(
            f"q local head count must be {_DSV4_TRITON_LOCAL_NUM_HEADS}, got {num_heads}"
        )
    if head_dim != _DSV4_TRITON_HEAD_DIM:
        raise ValueError(f"q head dimension must be {_DSV4_TRITON_HEAD_DIM}, got {head_dim}")
    _validate_norm_weight_for_dim(norm_weight, head_dim)
    _validate_dsv4_freqs(freqs_cis, batch_size, seq_len, rope_dim)

    if out is not None:
        if out.shape != q.shape:
            raise ValueError(f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}")
        if out.dtype != q.dtype:
            raise TypeError(f"out dtype must be {q.dtype}, got {out.dtype}")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}, got {out.device}")


def _cache_token_view(
    cache: torch.Tensor,
    page_idx: int,
    token_offset: int,
) -> torch.Tensor:
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
    raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")


def _cache_tokens_per_block(cache: torch.Tensor) -> int:
    if cache.dim() not in (3, 4, 5):
        raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")
    if cache.dim() == 5 and int(cache.shape[1]) == 1:
        return int(cache.shape[2] if int(cache.shape[2]) >= int(cache.shape[3]) else cache.shape[3])
    return int(cache.shape[1])


def _page_for_position(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    position: int,
    tokens_per_block: int,
) -> tuple[int, int]:
    page_ordinal = position // tokens_per_block
    page_table_start = int(cu_num_pages_host[seq_idx].item())
    page_table_end = int(cu_num_pages_host[seq_idx + 1].item())
    page_table_idx = page_table_start + page_ordinal
    if page_table_idx >= page_table_end:
        raise ValueError(
            f"Sequence {seq_idx} position {position} needs page ordinal {page_ordinal}, "
            f"but only {page_table_end - page_table_start} page(s) are available."
        )
    page_idx = int(cache_loc_host[page_table_idx].item())
    return page_idx, position % tokens_per_block


def _write_bf16_swa_cache(
    kv_seq: torch.Tensor,
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
) -> None:
    tokens_per_block = _cache_tokens_per_block(cache)
    for token_offset in range(kv_seq.shape[0]):
        page_idx, page_offset = _page_for_position(
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos + token_offset,
            tokens_per_block,
        )
        _cache_token_view(cache, page_idx, page_offset).copy_(kv_seq[token_offset].to(cache.dtype))


def _to_host_long(name: str, tensor: torch.Tensor, min_numel: int) -> torch.Tensor:
    _validate_int_metadata(name, tensor)
    if tensor.numel() < min_numel:
        raise ValueError(f"{name} needs at least {min_numel} entries, got {tensor.numel()}")
    return tensor.detach().cpu().to(torch.long).flatten()


def _validate_triton_kv_norm_rope_cache_insert_contract(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor],
    *,
    validate_metadata_values: bool,
) -> None:
    _validate_rank("kv", kv, 3)
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bfloat16 for the DSV4 Triton contract, got {kv.dtype}")
    _validate_rope_dim(rope_dim, kv.shape[-1])
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        raise ValueError(f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    batch_size, seq_len, head_dim = kv.shape
    if head_dim != _DSV4_TRITON_HEAD_DIM:
        raise ValueError(f"kv head dimension must be {_DSV4_TRITON_HEAD_DIM}, got {head_dim}")
    _validate_norm_weight_for_dim(norm_weight, head_dim)
    _validate_dsv4_freqs(freqs_cis, batch_size, seq_len, rope_dim)
    _validate_int_metadata("batch_info_host", batch_info_host)
    _validate_int_metadata("seq_len_host", seq_len_host)
    _validate_int_metadata("input_pos_host", input_pos_host)
    _validate_int_metadata("cu_seqlen_host", cu_seqlen_host)
    _validate_int_metadata("cache_loc_host", cache_loc_host)
    _validate_int_metadata("cu_num_pages_host", cu_num_pages_host)

    if swa_cache.dim() not in (3, 4, 5):
        raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {swa_cache.dim()}")
    if swa_cache.dtype != torch.bfloat16:
        raise TypeError(f"swa_cache must be bfloat16, got {swa_cache.dtype}")
    if swa_cache.shape[-1] != _DSV4_TRITON_HEAD_DIM:
        raise ValueError(
            f"swa_cache last dimension must be {_DSV4_TRITON_HEAD_DIM}, got {swa_cache.shape[-1]}"
        )

    if out is not None:
        if out.shape != kv.shape:
            raise ValueError(f"out shape must be {tuple(kv.shape)}, got {tuple(out.shape)}")
        if out.dtype != kv.dtype:
            raise TypeError(f"out dtype must be {kv.dtype}, got {out.dtype}")
        if out.device != kv.device:
            raise ValueError(f"out must be on {kv.device}, got {out.device}")

    if not validate_metadata_values:
        return

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    max_graph_tokens = int(batch_size * seq_len)
    if active_tokens > max_graph_tokens:
        raise ValueError(
            f"active tokens ({active_tokens}) exceed kv capacity ({max_graph_tokens}); "
            "padded graph slots may be present, but active tokens must fit."
        )
    _to_host_long("seq_len_host", seq_len_host, num_seq)
    _to_host_long("input_pos_host", input_pos_host, num_seq)
    _to_host_long("cu_seqlen_host", cu_seqlen_host, num_seq + 1)
    _to_host_long("cu_num_pages_host", cu_num_pages_host, num_seq + 1)

    if cache_loc_host.numel() == 0 and active_tokens:
        raise ValueError("cache_loc_host must contain at least one page for active tokens")


def _any_cuda_tensor(*tensors: Optional[torch.Tensor]) -> bool:
    return any(tensor is not None and tensor.is_cuda for tensor in tensors)


def _device_skip_reason(name: str, tensor: torch.Tensor, device: torch.device) -> str | None:
    if not tensor.is_cuda:
        return f"{name} must be a CUDA tensor for the DSV4 Triton path"
    if tensor.device != device:
        return f"{name} must be on {device}, got {tensor.device}"
    return None


def _triton_q_rmsnorm_rope_skip_reason(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    out: Optional[torch.Tensor],
) -> str | None:
    for name, tensor in (("q", q), ("freqs_cis", freqs_cis)):
        reason = _device_skip_reason(name, tensor, q.device)
        if reason is not None:
            return reason
    if norm_weight is not None:
        reason = _device_skip_reason("norm_weight", norm_weight, q.device)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, q.device)
        if reason is not None:
            return reason
    if freqs_cis.dtype != torch.complex64:
        return f"freqs_cis must be complex64 for the DSV4 Triton path, got {freqs_cis.dtype}"
    return None


def _cache_layout_for_triton(cache: torch.Tensor) -> tuple[int, int, int, int]:
    """Return ``(tokens_per_block, page_stride, token_stride, dim_stride)``."""
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            token_dim = 2 if int(cache.shape[2]) >= int(cache.shape[3]) else 3
        else:
            token_dim = 1
    elif cache.dim() in (3, 4):
        token_dim = 1
    else:
        raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")

    return (
        int(cache.shape[token_dim]),
        int(cache.stride(0)),
        int(cache.stride(token_dim)),
        int(cache.stride(-1)),
    )


def _triton_kv_norm_rope_cache_insert_skip_reason(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    out: Optional[torch.Tensor],
) -> str | None:
    for name, tensor in (
        ("kv", kv),
        ("freqs_cis", freqs_cis),
        ("batch_info_host", batch_info_host),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("swa_cache", swa_cache),
    ):
        reason = _device_skip_reason(name, tensor, kv.device)
        if reason is not None:
            return reason
    if norm_weight is not None:
        reason = _device_skip_reason("norm_weight", norm_weight, kv.device)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, kv.device)
        if reason is not None:
            return reason
    if freqs_cis.dtype != torch.complex64:
        return f"freqs_cis must be complex64 for the DSV4 Triton path, got {freqs_cis.dtype}"

    batch_size = int(kv.shape[0])
    if batch_info_host.numel() < 5:
        return f"batch_info_host needs at least 5 entries, got {batch_info_host.numel()}"
    if seq_len_host.numel() < batch_size:
        return f"seq_len_host needs at least {batch_size} entries, got {seq_len_host.numel()}"
    if input_pos_host.numel() < batch_size:
        return f"input_pos_host needs at least {batch_size} entries, got {input_pos_host.numel()}"
    if cu_seqlen_host.numel() < batch_size + 1:
        return (
            f"cu_seqlen_host needs at least {batch_size + 1} entries, got {cu_seqlen_host.numel()}"
        )
    if cu_num_pages_host.numel() < batch_size + 1:
        return (
            f"cu_num_pages_host needs at least {batch_size + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )
    if cache_loc_host.numel() == 0:
        return "cache_loc_host must contain at least one page for the DSV4 Triton path"

    try:
        tokens_per_block, _, _, _ = _cache_layout_for_triton(swa_cache)
    except ValueError as exc:
        return str(exc)
    if tokens_per_block <= 0:
        return f"swa_cache tokens_per_block must be positive, got {tokens_per_block}"
    return None


@triton.jit
def _dsv4_q_rmsnorm_kernel(
    q_ptr,
    norm_weight_ptr,
    out_ptr,
    q_stride_batch: tl.constexpr,
    q_stride_seq: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    weight_stride: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
):
    token_head_idx = tl.program_id(0)
    head_idx = token_head_idx % num_heads
    token_idx = token_head_idx // num_heads
    seq_idx = token_idx % seq_len
    batch_idx = token_idx // seq_len
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim

    q_offsets = (
        batch_idx * q_stride_batch
        + seq_idx * q_stride_seq
        + head_idx * q_stride_head
        + dim_offsets * q_stride_dim
    )
    values = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(tl.where(dim_mask, values * values, 0.0), axis=0)
    rms_scale = tl.rsqrt(square_sum / head_dim + eps)
    output = values * rms_scale
    if has_weight:
        weight = tl.load(norm_weight_ptr + dim_offsets * weight_stride, mask=dim_mask, other=0.0)
        output = output * weight.to(tl.float32)

    out_offsets = (
        batch_idx * out_stride_batch
        + seq_idx * out_stride_seq
        + head_idx * out_stride_head
        + dim_offsets * out_stride_dim
    )
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


@triton.jit
def _dsv4_q_rope_kernel(
    freqs_real_ptr,
    out_ptr,
    freq_stride_batch: tl.constexpr,
    freq_stride_seq: tl.constexpr,
    freq_stride_pair: tl.constexpr,
    freq_stride_component: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    token_head_idx = tl.program_id(0)
    head_idx = token_head_idx % num_heads
    token_idx = token_head_idx // num_heads
    seq_idx = token_idx % seq_len
    batch_idx = token_idx // seq_len

    rope_offsets = tl.arange(0, rope_dim)
    dim_offsets = nope_dim + rope_offsets
    pair_offsets = rope_offsets // 2
    is_second = rope_offsets != pair_offsets * 2
    partner_offsets = tl.where(is_second, dim_offsets - 1, dim_offsets + 1)

    out_base = batch_idx * out_stride_batch + seq_idx * out_stride_seq + head_idx * out_stride_head
    values = tl.load(out_ptr + out_base + dim_offsets * out_stride_dim).to(tl.float32)
    partner_values = tl.load(out_ptr + out_base + partner_offsets * out_stride_dim).to(tl.float32)
    first = tl.where(is_second, partner_values, values)
    second = tl.where(is_second, values, partner_values)

    freq_base = batch_idx * freq_stride_batch + seq_idx * freq_stride_seq
    cos = tl.load(freqs_real_ptr + freq_base + pair_offsets * freq_stride_pair).to(tl.float32)
    sin = tl.load(
        freqs_real_ptr + freq_base + pair_offsets * freq_stride_pair + freq_stride_component
    ).to(tl.float32)
    rotated = tl.where(is_second, first * sin + second * cos, first * cos - second * sin)
    tl.store(out_ptr + out_base + dim_offsets * out_stride_dim, rotated)


@triton.jit
def _dsv4_kv_rmsnorm_kernel(
    kv_ptr,
    norm_weight_ptr,
    out_ptr,
    kv_stride_batch: tl.constexpr,
    kv_stride_seq: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    weight_stride: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_dim: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_idx = token_idx // seq_len
    seq_idx = token_idx - batch_idx * seq_len
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim

    kv_offsets = batch_idx * kv_stride_batch + seq_idx * kv_stride_seq + dim_offsets * kv_stride_dim
    values = tl.load(kv_ptr + kv_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(tl.where(dim_mask, values * values, 0.0), axis=0)
    rms_scale = tl.rsqrt(square_sum / head_dim + eps)
    output = values * rms_scale
    if has_weight:
        weight = tl.load(norm_weight_ptr + dim_offsets * weight_stride, mask=dim_mask, other=0.0)
        output = output * weight.to(tl.float32)

    out_offsets = (
        batch_idx * out_stride_batch + seq_idx * out_stride_seq + dim_offsets * out_stride_dim
    )
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


@triton.jit
def _dsv4_kv_rope_cache_insert_kernel(
    freqs_real_ptr,
    batch_info_ptr,
    input_pos_ptr,
    cu_seqlen_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    out_ptr,
    freq_stride_batch: tl.constexpr,
    freq_stride_seq: tl.constexpr,
    freq_stride_pair: tl.constexpr,
    freq_stride_component: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    cache_dim_stride: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_dim: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    tokens_per_block: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_idx = token_idx // seq_len
    seq_idx_in_tensor = token_idx - batch_idx * seq_len
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim

    num_prefill = tl.load(batch_info_ptr) + tl.load(batch_info_ptr + 2)
    num_prefill_tokens = tl.load(batch_info_ptr + 1) + tl.load(batch_info_ptr + 3)
    num_decode = tl.load(batch_info_ptr + 4)
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    found = token_idx < 0
    selected_seq = tl.full((), 0, dtype=tl.int64)
    selected_seq_start = tl.full((), 0, dtype=tl.int64)
    for metadata_seq in range(batch_size):
        seq_start = tl.load(cu_seqlen_ptr + metadata_seq, mask=metadata_seq < num_seq, other=0).to(
            tl.int64
        )
        seq_end = tl.load(
            cu_seqlen_ptr + metadata_seq + 1,
            mask=metadata_seq < num_seq,
            other=0,
        ).to(tl.int64)
        in_seq = (metadata_seq < num_seq) & (token_idx >= seq_start) & (token_idx < seq_end)
        selected_seq = tl.where(in_seq, metadata_seq, selected_seq)
        selected_seq_start = tl.where(in_seq, seq_start, selected_seq_start)
        found = found | in_seq

    token_is_active = (token_idx < active_tokens) & found
    seq_token_offset = token_idx - selected_seq_start
    input_pos = tl.load(input_pos_ptr + selected_seq, mask=token_is_active, other=0).to(tl.int64)
    cache_position = input_pos + seq_token_offset
    page_ordinal = cache_position // tokens_per_block
    page_offset = cache_position - page_ordinal * tokens_per_block
    page_table_start = tl.load(cu_num_pages_ptr + selected_seq, mask=token_is_active, other=0).to(
        tl.int64
    )
    page_idx = tl.load(
        cache_loc_ptr + page_table_start + page_ordinal,
        mask=token_is_active,
        other=0,
    ).to(tl.int64)

    out_offsets = (
        batch_idx * out_stride_batch
        + seq_idx_in_tensor * out_stride_seq
        + dim_offsets * out_stride_dim
    )
    values = tl.load(out_ptr + out_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    partner_offsets = tl.where((dim_offsets - nope_dim) % 2 == 1, dim_offsets - 1, dim_offsets + 1)
    partner_values = tl.load(
        out_ptr + out_offsets - dim_offsets * out_stride_dim + partner_offsets * out_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    rope_idx = dim_offsets - nope_dim
    rope_mask = (rope_idx >= 0) & (rope_idx < rope_dim)
    pair_offsets = rope_idx // 2
    is_second = rope_idx != pair_offsets * 2
    first = tl.where(is_second, partner_values, values)
    second = tl.where(is_second, values, partner_values)
    freq_base = batch_idx * freq_stride_batch + seq_idx_in_tensor * freq_stride_seq
    cos = tl.load(
        freqs_real_ptr + freq_base + pair_offsets * freq_stride_pair,
        mask=rope_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        freqs_real_ptr + freq_base + pair_offsets * freq_stride_pair + freq_stride_component,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = tl.where(is_second, first * sin + second * cos, first * cos - second * sin)
    output = tl.where(rope_mask, rotated, values)
    output = tl.where(token_is_active, output, 0.0)

    tl.store(out_ptr + out_offsets, output, mask=dim_mask)

    cache_offsets = (
        page_idx * cache_page_stride
        + page_offset * cache_token_stride
        + dim_offsets * cache_dim_stride
    )
    tl.store(swa_cache_ptr + cache_offsets, output, mask=dim_mask & token_is_active)


def _run_triton_q_rmsnorm_rope(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(q) if out is None else out
    freqs_real = torch.view_as_real(freqs_cis)
    weight_arg = q if norm_weight is None else norm_weight
    weight_stride = 1 if norm_weight is None else int(norm_weight.stride(0))

    grid = (int(q.shape[0] * q.shape[1] * q.shape[2]),)
    _dsv4_q_rmsnorm_kernel[grid](
        q,
        weight_arg,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        weight_stride,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        int(q.shape[1]),
        _DSV4_TRITON_LOCAL_NUM_HEADS,
        _DSV4_TRITON_HEAD_DIM,
        _DSV4_TRITON_BLOCK_DIM,
        eps,
        norm_weight is not None,
        num_warps=8,
    )
    _dsv4_q_rope_kernel[grid](
        freqs_real,
        output,
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_real.stride(2),
        freqs_real.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        int(q.shape[1]),
        _DSV4_TRITON_LOCAL_NUM_HEADS,
        _DSV4_TRITON_HEAD_DIM - _DSV4_TRITON_ROPE_DIM,
        _DSV4_TRITON_ROPE_DIM,
        num_warps=2,
    )
    if out is not None:
        return out.new_empty(0)
    return output


def _run_triton_kv_norm_rope_cache_insert(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(kv) if out is None else out
    freqs_real = torch.view_as_real(freqs_cis)
    weight_arg = kv if norm_weight is None else norm_weight
    weight_stride = 1 if norm_weight is None else int(norm_weight.stride(0))
    tokens_per_block, cache_page_stride, cache_token_stride, cache_dim_stride = (
        _cache_layout_for_triton(swa_cache)
    )
    grid = (int(kv.shape[0] * kv.shape[1]),)

    _dsv4_kv_rmsnorm_kernel[grid](
        kv,
        weight_arg,
        output,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        weight_stride,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        int(kv.shape[1]),
        _DSV4_TRITON_HEAD_DIM,
        _DSV4_TRITON_BLOCK_DIM,
        eps,
        norm_weight is not None,
        num_warps=8,
    )
    _dsv4_kv_rope_cache_insert_kernel[grid](
        freqs_real,
        batch_info_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        output,
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_real.stride(2),
        freqs_real.stride(3),
        cache_page_stride,
        cache_token_stride,
        cache_dim_stride,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        int(kv.shape[0]),
        int(kv.shape[1]),
        tokens_per_block,
        _DSV4_TRITON_HEAD_DIM,
        _DSV4_TRITON_HEAD_DIM - _DSV4_TRITON_ROPE_DIM,
        _DSV4_TRITON_ROPE_DIM,
        _DSV4_TRITON_BLOCK_DIM,
        num_warps=8,
    )
    if out is not None:
        return out.new_empty(0)
    return output


def deepseek_v4_fp8_block_quant_ref(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to E4M3 with one E8M0 scale per last-dim block.

    Args:
        x: Tensor of shape ``[..., dim]``.
        block_size: Number of contiguous last-dimension elements covered by one
            scale. Partial tail blocks are zero-padded for amax calculation.

    Returns:
        ``(x_fp8, scale)`` where ``x_fp8`` has shape ``[..., dim]`` and dtype
        ``torch.float8_e4m3fn``. ``scale`` has shape
        ``[..., ceil(dim / block_size)]`` and uses ``torch.float8_e8m0fnu`` when
        available, otherwise FP32 powers of two.
    """
    _validate_floating("x", x)
    _validate_block_size(block_size)

    last_dim = x.shape[-1]
    num_blocks = (last_dim + block_size - 1) // block_size
    scale_shape = (*x.shape[:-1], num_blocks)
    if last_dim == 0:
        return x.to(FP8_E4M3_DTYPE).contiguous(), _fake_scale(scale_shape, x.device)

    pad = num_blocks * block_size - last_dim
    x_float = x.to(torch.float32)
    if pad:
        x_float = F.pad(x_float, (0, pad))
    blocked = x_float.reshape(*x.shape[:-1], num_blocks, block_size)
    amax = blocked.abs().amax(dim=-1)
    scale_fp32 = torch.where(
        amax > 0,
        amax / FP8_E4M3_MAX,
        torch.ones((), device=x.device, dtype=torch.float32),
    )
    scale = _quantize_scale_to_e8m0(scale_fp32)
    scale_decoded = maybe_e8m0_to_fp32(scale).to(device=x.device, dtype=torch.float32)
    quant = (blocked / scale_decoded.unsqueeze(-1)).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    quant = quant.to(FP8_E4M3_DTYPE).reshape(*x.shape[:-1], num_blocks * block_size)
    return quant[..., :last_dim].contiguous(), scale.contiguous()


def deepseek_v4_fp8_block_dequant_ref(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensors produced by :func:`deepseek_v4_fp8_block_quant_ref`."""
    _validate_block_size(block_size)
    if x_fp8.dtype != FP8_E4M3_DTYPE:
        raise TypeError(f"x_fp8 must have dtype {FP8_E4M3_DTYPE}, got {x_fp8.dtype}")

    scale_expanded = maybe_e8m0_to_fp32(scale).repeat_interleave(block_size, dim=-1)
    scale_expanded = scale_expanded[..., : x_fp8.shape[-1]]
    return (x_fp8.to(torch.float32) * scale_expanded.to(torch.float32)).to(dtype).contiguous()


def deepseek_v4_q_rmsnorm_rope_ref(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> torch.Tensor:
    """Apply per-head RMSNorm and complex RoPE to query states ``[B, S, H, D]``."""
    _validate_rank("q", q, 4)
    q_norm = _rms_norm_ref(q, norm_weight, eps)
    return _apply_rope_ref(q_norm, freqs_cis, rope_dim)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_q_rmsnorm_rope", mutates_args=())
def torch_deepseek_v4_q_rmsnorm_rope(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> torch.Tensor:
    return deepseek_v4_q_rmsnorm_rope_ref(q, norm_weight, freqs_cis, eps, rope_dim)


@torch_deepseek_v4_q_rmsnorm_rope.register_fake
def torch_deepseek_v4_q_rmsnorm_rope_fake(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> torch.Tensor:
    _validate_rank("q", q, 4)
    _validate_rope_dim(rope_dim, q.shape[-1])
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_q_rmsnorm_rope", mutates_args=())
def triton_deepseek_v4_q_rmsnorm_rope(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Guarded Triton-facing DSV4 Q RMSNorm+RoPE contract.

    The device path accepts the observed local BF16 DSV4 query shape
    ``[batch, seq_len, 8, 512]`` and applies RoPE to the trailing 64 dims while
    preserving the caller-owned ``out=`` replay convention.
    """
    _validate_triton_q_rmsnorm_rope_contract(q, norm_weight, freqs_cis, eps, rope_dim, out)
    skip_reason = _triton_q_rmsnorm_rope_skip_reason(q, norm_weight, freqs_cis, out)
    if skip_reason is None:
        return _run_triton_q_rmsnorm_rope(q, norm_weight, freqs_cis, eps, out)
    if _any_cuda_tensor(q, norm_weight, freqs_cis, out):
        raise RuntimeError(skip_reason)

    output = deepseek_v4_q_rmsnorm_rope_ref(q, norm_weight, freqs_cis, eps, rope_dim)
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@triton_deepseek_v4_q_rmsnorm_rope.register_fake
def triton_deepseek_v4_q_rmsnorm_rope_fake(
    q: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_triton_q_rmsnorm_rope_contract(q, norm_weight, freqs_cis, eps, rope_dim, out)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


def deepseek_v4_kv_rmsnorm_rope_ref(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> torch.Tensor:
    """Apply RMSNorm and complex RoPE to shared KV states ``[B, S, D]``."""
    _validate_rank("kv", kv, 3)
    kv_norm = _rms_norm_ref(kv, norm_weight, eps)
    return _apply_rope_ref(kv_norm, freqs_cis, rope_dim)


def deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> torch.Tensor:
    """Normalize/RoPE KV rows and insert active tokens into BF16 SWA cache pages.

    Args:
        kv: BF16 KV states with shape ``[batch, seq_len, 512]``.
        norm_weight: Optional RMSNorm weight with shape ``[512]``.
        freqs_cis: Per-token RoPE frequencies with shape
            ``[batch, seq_len, rope_dim // 2]``.
        batch_info_host: Serialized :class:`BatchInfo` metadata.
        seq_len_host: Active per-sequence token counts.
        input_pos_host: Absolute starting positions for each active sequence.
        cu_seqlen_host: Prefix sums mapping active sequences into flattened
            graph tokens.
        cache_loc_host: Paged-cache page table.
        cu_num_pages_host: Prefix sums mapping active sequences into
            ``cache_loc_host``.
        swa_cache: Caller-owned BF16 SWA cache with rank 3, 4, or 5 and last
            dimension 512.
        eps: RMSNorm epsilon.
        rope_dim: Number of trailing dimensions using RoPE; must be 64 for the
            guarded Triton contract.

    Returns:
        Normalized/RoPE KV tensor with shape ``[batch, seq_len, 512]``. Padded
        graph slots beyond active tokens are zeroed.
    """
    _validate_triton_kv_norm_rope_cache_insert_contract(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        eps,
        rope_dim,
        out=None,
        validate_metadata_values=True,
    )

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    seq_len_host = _to_host_long("seq_len_host", seq_len_host, num_seq)
    input_pos_host = _to_host_long("input_pos_host", input_pos_host, num_seq)
    cu_seqlen_host = _to_host_long("cu_seqlen_host", cu_seqlen_host, num_seq + 1)
    cache_loc_host = _to_host_long("cache_loc_host", cache_loc_host, 1 if active_tokens else 0)
    cu_num_pages_host = _to_host_long("cu_num_pages_host", cu_num_pages_host, num_seq + 1)

    kv_out = deepseek_v4_kv_rmsnorm_rope_ref(kv, norm_weight, freqs_cis, eps, rope_dim)
    kv_out_flat = kv_out.reshape(-1, kv_out.shape[-1])
    output_flat = torch.zeros_like(kv_out_flat)

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        flat_end = flat_start + seq_len_i
        if flat_end > kv_out_flat.shape[0]:
            raise ValueError(
                f"Sequence {seq_idx} flat range [{flat_start}, {flat_end}) exceeds "
                f"kv capacity {kv_out_flat.shape[0]}"
            )
        input_pos_i = int(input_pos_host[seq_idx].item())
        kv_seq = kv_out_flat[flat_start:flat_end]
        output_flat[flat_start:flat_end].copy_(kv_seq)
        _write_bf16_swa_cache(
            kv_seq,
            swa_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos_i,
        )

    if active_tokens < output_flat.shape[0]:
        output_flat[active_tokens:].zero_()
    return output_flat.view_as(kv)


@torch.library.custom_op(
    "auto_deploy::triton_deepseek_v4_kv_norm_rope_cache_insert",
    mutates_args=("swa_cache",),
)
def triton_deepseek_v4_kv_norm_rope_cache_insert(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Guarded Triton-facing DSV4 KV RMSNorm+RoPE+BF16 SWA cache insert."""
    _validate_triton_kv_norm_rope_cache_insert_contract(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        eps,
        rope_dim,
        out,
        validate_metadata_values=False,
    )
    skip_reason = _triton_kv_norm_rope_cache_insert_skip_reason(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        out,
    )
    if skip_reason is None:
        return _run_triton_kv_norm_rope_cache_insert(
            kv,
            norm_weight,
            freqs_cis,
            batch_info_host,
            input_pos_host,
            cu_seqlen_host,
            cache_loc_host,
            cu_num_pages_host,
            swa_cache,
            eps,
            out,
        )
    if _any_cuda_tensor(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        out,
    ):
        raise RuntimeError(skip_reason)

    output = deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        eps,
        rope_dim,
    )
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@triton_deepseek_v4_kv_norm_rope_cache_insert.register_fake
def triton_deepseek_v4_kv_norm_rope_cache_insert_fake(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_triton_kv_norm_rope_cache_insert_contract(
        kv,
        norm_weight,
        freqs_cis,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        eps,
        rope_dim,
        out,
        validate_metadata_values=False,
    )
    if out is not None:
        return out.new_empty(0)
    return kv.new_empty(kv.shape).contiguous()


def deepseek_v4_kv_rmsnorm_rope_cache_insert_ref(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    cache_indices: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize/RoPE KV, FP8-quantize NoPE dims, and insert flat cache rows.

    ``cache_indices`` is a flat gather/scatter contract supplied by the caller.
    It deliberately does not encode V4 paged-cache metadata.
    """
    kv_out = deepseek_v4_kv_rmsnorm_rope_ref(kv, norm_weight, freqs_cis, eps, rope_dim)
    nope = kv_out[..., : kv_out.shape[-1] - rope_dim]
    rope = kv_out[..., -rope_dim:] if rope_dim else kv_out.new_empty(*kv_out.shape[:-1], 0)
    nope_fp8, nope_scale = deepseek_v4_fp8_block_quant_ref(nope, fp8_block_size)

    _insert_flat_cache(nope_cache, cache_indices, nope_fp8)
    _insert_flat_cache(rope_cache, cache_indices, rope)
    flat_scale = nope_scale.reshape(-1, nope_scale.shape[-1])
    if cache_indices.numel() != flat_scale.shape[0]:
        raise ValueError(
            f"cache_indices must have {flat_scale.shape[0]} entries, got {cache_indices.numel()}"
        )
    if flat_scale.numel() != 0:
        _index_copy_rows(
            scale_cache,
            cache_indices.reshape(-1).to(device=scale_cache.device, dtype=torch.long),
            flat_scale,
        )
    return kv_out, nope_fp8, nope_scale


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_kv_rmsnorm_rope_cache_insert",
    mutates_args=("nope_cache", "rope_cache", "scale_cache"),
)
def torch_deepseek_v4_kv_rmsnorm_rope_cache_insert(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    cache_indices: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return deepseek_v4_kv_rmsnorm_rope_cache_insert_ref(
        kv,
        norm_weight,
        freqs_cis,
        cache_indices,
        nope_cache,
        rope_cache,
        scale_cache,
        eps,
        rope_dim,
        fp8_block_size,
    )


@torch_deepseek_v4_kv_rmsnorm_rope_cache_insert.register_fake
def torch_deepseek_v4_kv_rmsnorm_rope_cache_insert_fake(
    kv: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    cache_indices: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_rank("kv", kv, 3)
    _validate_rope_dim(rope_dim, kv.shape[-1])
    _validate_block_size(fp8_block_size)
    nope_dim = kv.shape[-1] - rope_dim
    num_blocks = (nope_dim + fp8_block_size - 1) // fp8_block_size
    return (
        kv.new_empty(kv.shape).contiguous(),
        torch.empty((*kv.shape[:-1], nope_dim), dtype=FP8_E4M3_DTYPE, device=kv.device),
        scale_cache.new_empty((*kv.shape[:-1], num_blocks)).contiguous(),
    )


def _overlap_transform_ref(
    tensor: torch.Tensor, ratio: int, head_dim: int, value: float
) -> torch.Tensor:
    batch_size, compressed_len, _, _ = tensor.shape
    out = tensor.new_full((batch_size, compressed_len, 2 * ratio, head_dim), value)
    out[:, :, ratio:] = tensor[:, :, :, head_dim:]
    out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return out


def deepseek_v4_compressor_pool_norm_rope_ref(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> torch.Tensor:
    """Apply DeepSeek V4 compressor pooling, RMSNorm, and RoPE.

    Args:
        kv: Compressor value projection with shape ``[B, S, channels * D]``.
        gate: Compressor gate projection with shape ``[B, S, channels * D]``.
        ape: Additive positional embedding with shape ``[ratio, channels * D]``.
        norm_weight: Optional RMSNorm weight with shape ``[D]``.
        freqs_cis: Full-token complex RoPE frequencies ``[B, S, rope_dim // 2]``.
        eps: RMSNorm epsilon.
        rope_dim: Number of trailing dimensions using RoPE.
        compress_ratio: Compression ratio, e.g. 4 or 128.
        overlap: Whether to use the ratio-4 overlapping compressor layout.
    """
    _validate_rank("kv", kv, 3)
    _validate_rank("gate", gate, 3)
    _validate_rank("ape", ape, 2)
    if kv.shape != gate.shape:
        raise ValueError(
            f"kv and gate shapes must match, got {tuple(kv.shape)} and {tuple(gate.shape)}"
        )
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")

    channels = 2 if overlap else 1
    if kv.shape[-1] % channels != 0:
        raise ValueError(f"kv last dim {kv.shape[-1]} is not divisible by channels={channels}")
    head_dim = kv.shape[-1] // channels
    _validate_rope_dim(rope_dim, head_dim)
    if ape.shape != (compress_ratio, channels * head_dim):
        raise ValueError(
            f"ape must have shape ({compress_ratio}, {channels * head_dim}), got {tuple(ape.shape)}"
        )

    batch_size, seq_len, _ = kv.shape
    cutoff = (seq_len // compress_ratio) * compress_ratio
    compressed_len = cutoff // compress_ratio
    if cutoff == 0:
        return kv.new_empty(batch_size, 0, head_dim)

    kv_blocks = kv[:, :cutoff].view(batch_size, compressed_len, compress_ratio, -1)
    gate_blocks = gate[:, :cutoff].view(batch_size, compressed_len, compress_ratio, -1)
    gate_blocks = gate_blocks + ape.to(device=gate.device, dtype=gate.dtype)
    if overlap:
        kv_blocks = _overlap_transform_ref(kv_blocks, compress_ratio, head_dim, 0.0)
        gate_blocks = _overlap_transform_ref(gate_blocks, compress_ratio, head_dim, float("-inf"))

    pooled = (kv_blocks * gate_blocks.softmax(dim=2)).sum(dim=2)
    pooled = _rms_norm_ref(pooled, norm_weight, eps)
    compressed_freqs = freqs_cis[:, :cutoff:compress_ratio]
    return _apply_rope_ref(pooled, compressed_freqs, rope_dim)


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_compressor_pool_norm_rope", mutates_args=()
)
def torch_deepseek_v4_compressor_pool_norm_rope(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> torch.Tensor:
    return deepseek_v4_compressor_pool_norm_rope_ref(
        kv, gate, ape, norm_weight, freqs_cis, eps, rope_dim, compress_ratio, overlap
    )


@torch_deepseek_v4_compressor_pool_norm_rope.register_fake
def torch_deepseek_v4_compressor_pool_norm_rope_fake(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> torch.Tensor:
    _validate_rank("kv", kv, 3)
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    channels = 2 if overlap else 1
    head_dim = kv.shape[-1] // channels
    _validate_rope_dim(rope_dim, head_dim)
    return kv.new_empty(kv.shape[0], kv.shape[1] // compress_ratio, head_dim).contiguous()


def _validate_ratio4_indexer_q(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp4_block_size: int,
) -> None:
    _validate_rank("q", q, 4)
    _validate_rope_dim(rope_dim, q.shape[-1])
    _validate_block_size(fp4_block_size)
    if q.shape[2:] != (_DSV4_RATIO4_INDEXER_NUM_HEADS, _DSV4_RATIO4_INDEXER_HEAD_DIM):
        raise ValueError(
            "q must have observed ratio-4 indexer tail shape "
            f"({_DSV4_RATIO4_INDEXER_NUM_HEADS}, {_DSV4_RATIO4_INDEXER_HEAD_DIM}), "
            f"got {tuple(q.shape[2:])}"
        )
    if q.shape[-1] % fp4_block_size != 0:
        raise ValueError(
            f"q head dimension {q.shape[-1]} must be divisible by fp4_block_size={fp4_block_size}"
        )
    _validate_dsv4_freqs(freqs_cis, int(q.shape[0]), int(q.shape[1]), rope_dim)


def _validate_ratio4_compressor_inputs(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int,
) -> int:
    _validate_rank("kv", kv, 3)
    _validate_rank("gate", gate, 3)
    _validate_rank("ape", ape, 2)
    if kv.shape != gate.shape:
        raise ValueError(
            f"kv and gate shapes must match, got {tuple(kv.shape)} and {tuple(gate.shape)}"
        )
    if kv.shape[1] == 0:
        raise ValueError("kv sequence length must be positive for ratio-4 compression")
    if kv.shape[-1] % 2 != 0:
        raise ValueError(f"kv last dim {kv.shape[-1]} must be even for two-channel overlap")
    if max_compressed_len <= 0:
        raise ValueError(f"max_compressed_len must be positive, got {max_compressed_len}")
    if kv.shape[1] > max_compressed_len * _DSV4_RATIO4_COMPRESS_RATIO:
        raise ValueError(
            f"kv sequence length {kv.shape[1]} exceeds ratio-4 capacity "
            f"{max_compressed_len * _DSV4_RATIO4_COMPRESS_RATIO}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    head_dim = kv.shape[-1] // 2
    if ape.shape != (_DSV4_RATIO4_COMPRESS_RATIO, kv.shape[-1]):
        raise ValueError(
            f"ape must have shape {(_DSV4_RATIO4_COMPRESS_RATIO, kv.shape[-1])}, "
            f"got {tuple(ape.shape)}"
        )
    _validate_norm_weight_for_dim(norm_weight, head_dim)
    _validate_rope_dim(rope_dim, head_dim)
    _validate_dsv4_freqs(freqs_cis, int(kv.shape[0]), int(kv.shape[1]), rope_dim)
    return head_dim


def _hadamard_transform_ref(x: torch.Tensor) -> torch.Tensor:
    last_dim = int(x.shape[-1])
    if last_dim <= 0 or (last_dim & (last_dim - 1)) != 0:
        raise ValueError(f"x last dimension must be a positive power of two, got {last_dim}")

    out = x.to(torch.float32).reshape(-1, last_dim)
    h = 1
    while h < last_dim:
        out = out.reshape(-1, 2, h)
        left, right = out.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2).reshape(-1, last_dim)
        h *= 2
    return (out * (last_dim**-0.5)).to(x.dtype).reshape_as(x)


def deepseek_v4_indexer_fp4_quant_dequant_ref(
    x: torch.Tensor,
    block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Reference DSV4 indexer activation fake-quantization for 32-value FP4 groups."""
    _validate_floating("x", x)
    _validate_block_size(block_size)
    last_dim = int(x.shape[-1])
    if last_dim % block_size != 0:
        raise ValueError(f"x last dim {last_dim} must be divisible by block_size={block_size}")

    x_blocks = x.contiguous().to(torch.float32).reshape(-1, last_dim)
    x_blocks = x_blocks.reshape(-1, last_dim // block_size, block_size)
    min_scale = torch.full((), 6.0 * (2.0**-126), dtype=torch.float32, device=x.device)
    scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(min_scale) / 6.0)
        ),
    )
    scaled = (x_blocks / scale).clamp(-6.0, 6.0)

    abs_scaled = scaled.abs()
    quantized = torch.zeros_like(abs_scaled)
    quantized = torch.where(abs_scaled > 0.25, torch.full_like(quantized, 0.5), quantized)
    quantized = torch.where(abs_scaled >= 0.75, torch.full_like(quantized, 1.0), quantized)
    quantized = torch.where(abs_scaled > 1.25, torch.full_like(quantized, 1.5), quantized)
    quantized = torch.where(abs_scaled >= 1.75, torch.full_like(quantized, 2.0), quantized)
    quantized = torch.where(abs_scaled > 2.5, torch.full_like(quantized, 3.0), quantized)
    quantized = torch.where(abs_scaled >= 3.5, torch.full_like(quantized, 4.0), quantized)
    quantized = torch.where(abs_scaled > 5.0, torch.full_like(quantized, 6.0), quantized)
    quantized = torch.where(scaled < 0, -quantized, quantized)
    return (quantized * scale).to(x.dtype).reshape_as(x)


def deepseek_v4_ratio4_overlap_compress_ref(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
) -> torch.Tensor:
    """Build padded ratio-4 overlap-compressed rows for attention or indexer projections.

    ``kv`` and ``gate`` use the observed two-channel layout ``[..., 2 * head_dim]``.
    The returned rows are padded to ``max_compressed_len`` so the indexer score
    boundary has a stable graph shape across prefill chunks and decode.
    """
    head_dim = _validate_ratio4_compressor_inputs(
        kv, gate, ape, norm_weight, freqs_cis, eps, rope_dim, max_compressed_len
    )

    batch_size, seq_len, state_dim = kv.shape
    max_tokens = max_compressed_len * _DSV4_RATIO4_COMPRESS_RATIO
    pad_len = max_tokens - seq_len
    kv_padded = F.pad(kv, (0, 0, 0, pad_len))
    gate_padded = F.pad(gate, (0, 0, 0, pad_len), value=float("-inf"))
    kv_blocks = kv_padded.view(
        batch_size, max_compressed_len, _DSV4_RATIO4_COMPRESS_RATIO, state_dim
    )
    gate_blocks = gate_padded.view(
        batch_size, max_compressed_len, _DSV4_RATIO4_COMPRESS_RATIO, state_dim
    )
    gate_blocks = gate_blocks + ape.to(device=gate.device, dtype=gate.dtype)
    kv_overlap = _overlap_transform_ref(kv_blocks, _DSV4_RATIO4_COMPRESS_RATIO, head_dim, 0.0)
    gate_overlap = _overlap_transform_ref(
        gate_blocks, _DSV4_RATIO4_COMPRESS_RATIO, head_dim, float("-inf")
    )

    weights = torch.nan_to_num(gate_overlap.softmax(dim=2), nan=0.0)
    pooled = (kv_overlap * weights).sum(dim=2)
    pooled = _rms_norm_ref(pooled, norm_weight, eps)

    row_idx = torch.arange(max_compressed_len, device=kv.device) * _DSV4_RATIO4_COMPRESS_RATIO
    row_idx = torch.minimum(row_idx, torch.full_like(row_idx, seq_len - 1))
    row_idx = row_idx.view(1, max_compressed_len, 1).expand(batch_size, -1, rope_dim // 2)
    compressed_freqs = torch.gather(freqs_cis, 1, row_idx)
    return _apply_rope_ref(pooled, compressed_freqs, rope_dim)


def deepseek_v4_ratio4_indexer_q_ref(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Apply RoPE, Hadamard, and 4x32 FP4 fake quantization to indexer Q rows."""
    _validate_ratio4_indexer_q(q, freqs_cis, rope_dim, fp4_block_size)
    q = _apply_rope_ref(q, freqs_cis, rope_dim)
    q = _hadamard_transform_ref(q)
    return deepseek_v4_indexer_fp4_quant_dequant_ref(q, fp4_block_size)


def deepseek_v4_ratio4_indexer_compressed_kv_ref(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Build ratio-4 indexer compressed rows ``[B, max_compressed_len, 128]``."""
    if compressor_kv.shape[-1] != _DSV4_RATIO4_INDEXER_STATE_DIM:
        raise ValueError(
            f"compressor_kv last dim must be {_DSV4_RATIO4_INDEXER_STATE_DIM}, "
            f"got {compressor_kv.shape[-1]}"
        )
    rows = deepseek_v4_ratio4_overlap_compress_ref(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
    )
    rows = _hadamard_transform_ref(rows)
    return deepseek_v4_indexer_fp4_quant_dequant_ref(rows, fp4_block_size)


def _validate_ratio4_indexer_scores_contract(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int,
    fp4_block_size: int,
) -> None:
    _validate_ratio4_indexer_q(q, freqs_cis, rope_dim, fp4_block_size)
    if weights.shape != (*q.shape[:2], _DSV4_RATIO4_INDEXER_NUM_HEADS):
        raise ValueError(
            f"weights must have shape {(*q.shape[:2], _DSV4_RATIO4_INDEXER_NUM_HEADS)}, "
            f"got {tuple(weights.shape)}"
        )
    if not weights.is_floating_point():
        raise TypeError(f"weights must be floating point, got {weights.dtype}")
    _validate_ratio4_compressor_inputs(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
    )
    if compressor_kv.shape[-1] != _DSV4_RATIO4_INDEXER_STATE_DIM:
        raise ValueError(
            f"compressor_kv last dim must be {_DSV4_RATIO4_INDEXER_STATE_DIM}, "
            f"got {compressor_kv.shape[-1]}"
        )


def deepseek_v4_ratio4_indexer_scores_ref(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
    weights_are_scaled: bool = False,
) -> torch.Tensor:
    """Compute local ratio-4 indexer scores before the required all-reduce."""
    _validate_ratio4_indexer_scores_contract(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    q = deepseek_v4_ratio4_indexer_q_ref(q, freqs_cis, rope_dim, fp4_block_size)
    compressed_kv = deepseek_v4_ratio4_indexer_compressed_kv_ref(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    score = torch.einsum("bshd,btd->bsht", q.float(), compressed_kv.float())
    scaled_weights = weights.float()
    if not weights_are_scaled:
        scaled_weights = scaled_weights * (
            _DSV4_RATIO4_INDEXER_HEAD_DIM**-0.5 * _DSV4_RATIO4_INDEXER_NUM_HEADS**-0.5
        )
    return (score.relu() * scaled_weights.unsqueeze(-1)).sum(dim=2).contiguous()


def _validate_ratio4_indexer_topk_inputs(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    topk_count: int,
    compress_ratio: int,
) -> None:
    _validate_rank("index_score", index_score, 3)
    if not index_score.is_floating_point():
        raise TypeError(f"index_score must be floating point, got {index_score.dtype}")
    _validate_int_metadata("source_seq_lens", source_seq_lens)
    _validate_int_metadata("input_pos", input_pos)
    if source_seq_lens.numel() != index_score.shape[0]:
        raise ValueError(
            f"source_seq_lens must have {index_score.shape[0]} entries, got {source_seq_lens.numel()}"
        )
    if input_pos.numel() != index_score.shape[0]:
        raise ValueError(
            f"input_pos must have {index_score.shape[0]} entries, got {input_pos.numel()}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if topk_count <= 0:
        raise ValueError(f"topk_count must be positive, got {topk_count}")
    if compress_ratio != _DSV4_RATIO4_COMPRESS_RATIO:
        raise ValueError(
            f"compress_ratio must be {_DSV4_RATIO4_COMPRESS_RATIO}, got {compress_ratio}"
        )


def deepseek_v4_ratio4_indexer_topk_ref(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
) -> torch.Tensor:
    """Mask invisible compressed rows and select offset ratio-4 indexer rows."""
    _validate_ratio4_indexer_topk_inputs(
        index_score,
        source_seq_lens,
        input_pos,
        window_size=1,
        topk_count=topk_count,
        compress_ratio=compress_ratio,
    )
    batch_size, seq_len, max_compressed_len = index_score.shape
    device = index_score.device
    source_seq_lens = source_seq_lens.to(device=device, dtype=torch.long).flatten()
    input_pos = input_pos.to(device=device, dtype=torch.long).flatten()
    query_positions = input_pos[:, None] + torch.arange(1, seq_len + 1, device=device)
    valid_lengths = query_positions // compress_ratio
    compressed_positions = torch.arange(max_compressed_len, device=device)
    invalid = compressed_positions.view(1, 1, -1) >= valid_lengths.unsqueeze(-1)
    masked_score = torch.where(
        invalid,
        torch.full_like(index_score, float("-inf")),
        index_score,
    )

    selected = masked_score.topk(min(topk_count, max_compressed_len), dim=-1).indices
    valid_selected = selected < valid_lengths.unsqueeze(-1)
    offset = source_seq_lens.view(batch_size, 1, 1)
    selected = torch.where(valid_selected, selected + offset, torch.full_like(selected, -1))
    if selected.shape[-1] < topk_count:
        pad = selected.new_full((*selected.shape[:-1], topk_count - selected.shape[-1]), -1)
        selected = torch.cat([selected, pad], dim=-1)
    return selected.to(torch.int32).contiguous()


def _ratio4_local_window_topk_idxs(
    window_size: int,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    source_seq_lens = source_seq_lens.to(device=device, dtype=torch.long).flatten()
    input_pos = input_pos.to(device=device, dtype=torch.long).flatten()
    positions = input_pos[:, None] + torch.arange(seq_len, device=device)
    start = (positions - window_size + 1).clamp(min=0)
    offsets = torch.arange(window_size, device=device)
    local = start.unsqueeze(-1) + offsets.view(1, 1, window_size)
    valid = (local <= positions.unsqueeze(-1)) & (local < source_seq_lens.view(-1, 1, 1))
    return torch.where(valid, local, torch.full_like(local, -1)).to(torch.int32)


def deepseek_v4_ratio4_indexer_build_topk_ref(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int = _DSV4_RATIO4_WINDOW_SIZE,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
) -> torch.Tensor:
    """Build final ratio-4 sparse-attention indices with 128 local and 512 compressed rows."""
    _validate_ratio4_indexer_topk_inputs(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    local = _ratio4_local_window_topk_idxs(
        window_size,
        source_seq_lens,
        input_pos,
        int(index_score.shape[1]),
        index_score.device,
    )
    compressed = deepseek_v4_ratio4_indexer_topk_ref(
        index_score, source_seq_lens, input_pos, topk_count, compress_ratio
    )
    return torch.cat([local, compressed], dim=-1).contiguous()


def _apply_rope_device(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0:
        return x.contiguous()

    nope = x[..., : x.shape[-1] - rope_dim]
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().reshape(*rope.shape[:-1], -1, 2))
    freqs = freqs_cis
    if freqs.dim() == rope_complex.dim() - 1:
        freqs = freqs.unsqueeze(-2)
    rope_out = torch.view_as_real(rope_complex * freqs).flatten(-2).to(x.dtype)
    return torch.cat([nope, rope_out], dim=-1).contiguous()


def _hadamard_transform_device(x: torch.Tensor) -> torch.Tensor:
    last_dim = int(x.shape[-1])
    out = x.to(torch.float32).reshape(-1, last_dim)
    h = 1
    while h < last_dim:
        out = out.reshape(-1, 2, h)
        left, right = out.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2).reshape(-1, last_dim)
        h *= 2
    return (out * (last_dim**-0.5)).to(x.dtype).reshape_as(x)


def _indexer_fp4_quant_dequant_device(
    x: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    last_dim = int(x.shape[-1])
    x_blocks = x.contiguous().to(torch.float32).reshape(-1, last_dim)
    x_blocks = x_blocks.reshape(-1, last_dim // block_size, block_size)
    min_scale = torch.full((), 6.0 * (2.0**-126), dtype=torch.float32, device=x.device)
    scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(min_scale) / 6.0)
        ),
    )
    scaled = (x_blocks / scale).clamp(-6.0, 6.0)

    abs_scaled = scaled.abs()
    quantized = torch.zeros_like(abs_scaled)
    quantized = torch.where(abs_scaled > 0.25, torch.full_like(quantized, 0.5), quantized)
    quantized = torch.where(abs_scaled >= 0.75, torch.full_like(quantized, 1.0), quantized)
    quantized = torch.where(abs_scaled > 1.25, torch.full_like(quantized, 1.5), quantized)
    quantized = torch.where(abs_scaled >= 1.75, torch.full_like(quantized, 2.0), quantized)
    quantized = torch.where(abs_scaled > 2.5, torch.full_like(quantized, 3.0), quantized)
    quantized = torch.where(abs_scaled >= 3.5, torch.full_like(quantized, 4.0), quantized)
    quantized = torch.where(abs_scaled > 5.0, torch.full_like(quantized, 6.0), quantized)
    quantized = torch.where(scaled < 0, -quantized, quantized)
    return (quantized * scale).to(x.dtype).reshape_as(x)


def _overlap_transform_device(
    tensor: torch.Tensor,
    value: float,
) -> torch.Tensor:
    batch_size, compressed_len, ratio, _ = tensor.shape
    out = tensor.new_full(
        (batch_size, compressed_len, 2 * ratio, _DSV4_RATIO4_INDEXER_HEAD_DIM), value
    )
    out[:, :, ratio:] = tensor[:, :, :, _DSV4_RATIO4_INDEXER_HEAD_DIM:]
    out[:, 1:, :ratio] = tensor[:, :-1, :, :_DSV4_RATIO4_INDEXER_HEAD_DIM]
    return out


def _ratio4_overlap_compress_device(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int,
) -> torch.Tensor:
    batch_size, seq_len, state_dim = kv.shape
    max_tokens = max_compressed_len * _DSV4_RATIO4_COMPRESS_RATIO
    pad_len = max_tokens - seq_len
    kv_padded = F.pad(kv, (0, 0, 0, pad_len))
    gate_padded = F.pad(gate, (0, 0, 0, pad_len), value=float("-inf"))
    kv_blocks = kv_padded.view(
        batch_size, max_compressed_len, _DSV4_RATIO4_COMPRESS_RATIO, state_dim
    )
    gate_blocks = gate_padded.view(
        batch_size, max_compressed_len, _DSV4_RATIO4_COMPRESS_RATIO, state_dim
    )
    gate_blocks = gate_blocks + ape.to(device=gate.device, dtype=gate.dtype)
    kv_overlap = _overlap_transform_device(kv_blocks, 0.0)
    gate_overlap = _overlap_transform_device(gate_blocks, float("-inf"))

    weights = torch.nan_to_num(gate_overlap.softmax(dim=2), nan=0.0)
    pooled = (kv_overlap * weights).sum(dim=2)
    pooled_float = pooled.to(torch.float32)
    pooled = pooled_float * torch.rsqrt(pooled_float.square().mean(dim=-1, keepdim=True) + eps)
    pooled = pooled * norm_weight.to(device=pooled.device, dtype=torch.float32)
    pooled = pooled.to(kv.dtype)

    row_idx = torch.arange(max_compressed_len, device=kv.device) * _DSV4_RATIO4_COMPRESS_RATIO
    row_idx = torch.minimum(row_idx, torch.full_like(row_idx, seq_len - 1))
    row_idx = row_idx.view(1, max_compressed_len, 1).expand(batch_size, -1, rope_dim // 2)
    compressed_freqs = torch.gather(freqs_cis, 1, row_idx)
    return _apply_rope_device(pooled, compressed_freqs, rope_dim)


def _ratio4_indexer_q_device(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp4_block_size: int,
) -> torch.Tensor:
    q = _apply_rope_device(q, freqs_cis, rope_dim)
    q = _hadamard_transform_device(q)
    return _indexer_fp4_quant_dequant_device(q, fp4_block_size)


def _ratio4_indexer_compressed_kv_device(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int,
    fp4_block_size: int,
) -> torch.Tensor:
    rows = _ratio4_overlap_compress_device(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
    )
    rows = _hadamard_transform_device(rows)
    return _indexer_fp4_quant_dequant_device(rows, fp4_block_size)


def _validate_ratio4_indexer_scores_out(
    q: torch.Tensor,
    max_compressed_len: int,
    out: Optional[torch.Tensor],
) -> None:
    if out is None:
        return
    expected_shape = (q.shape[0], q.shape[1], max_compressed_len)
    if out.shape != expected_shape:
        raise ValueError(f"out shape must be {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != torch.float32:
        raise TypeError(f"out dtype must be torch.float32, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")


def _triton_ratio4_indexer_scores_skip_reason(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    out: Optional[torch.Tensor],
) -> str | None:
    for name, tensor in (
        ("q", q),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("compressor_ape", compressor_ape),
        ("compressor_norm_weight", compressor_norm_weight),
        ("weights", weights),
        ("freqs_cis", freqs_cis),
    ):
        reason = _device_skip_reason(name, tensor, q.device)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, q.device)
        if reason is not None:
            return reason
    if freqs_cis.dtype != torch.complex64:
        return f"freqs_cis must be complex64 for the DSV4 Triton path, got {freqs_cis.dtype}"
    return None


def _run_ratio4_indexer_scores_device(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int,
    max_compressed_len: int,
    fp4_block_size: int,
    weights_are_scaled: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    q_indexer = _ratio4_indexer_q_device(q, freqs_cis, rope_dim, fp4_block_size)
    compressed_kv = _ratio4_indexer_compressed_kv_device(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    score = torch.einsum("bshd,btd->bsht", q_indexer.float(), compressed_kv.float())
    scaled_weights = weights.float()
    if not weights_are_scaled:
        scaled_weights = scaled_weights * (
            _DSV4_RATIO4_INDEXER_HEAD_DIM**-0.5 * _DSV4_RATIO4_INDEXER_NUM_HEADS**-0.5
        )
    output = (score.relu() * scaled_weights.unsqueeze(-1)).sum(dim=2).contiguous()
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


def _validate_ratio4_indexer_topk_out(
    index_score: torch.Tensor,
    window_size: int,
    topk_count: int,
    out: Optional[torch.Tensor],
) -> None:
    if out is None:
        return
    expected_shape = (index_score.shape[0], index_score.shape[1], window_size + topk_count)
    if out.shape != expected_shape:
        raise ValueError(f"out shape must be {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != torch.int32:
        raise TypeError(f"out dtype must be torch.int32, got {out.dtype}")
    if out.device != index_score.device:
        raise ValueError(f"out must be on {index_score.device}, got {out.device}")


def _triton_ratio4_indexer_topk_skip_reason(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    out: Optional[torch.Tensor],
) -> str | None:
    for name, tensor in (
        ("index_score", index_score),
        ("source_seq_lens", source_seq_lens),
        ("input_pos", input_pos),
    ):
        reason = _device_skip_reason(name, tensor, index_score.device)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, index_score.device)
        if reason is not None:
            return reason
    return None


def _ratio4_indexer_topk_device(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    topk_count: int,
    compress_ratio: int,
) -> torch.Tensor:
    batch_size, seq_len, max_compressed_len = index_score.shape
    device = index_score.device
    source_seq_lens = source_seq_lens.to(device=device, dtype=torch.long).flatten()
    input_pos = input_pos.to(device=device, dtype=torch.long).flatten()

    positions = input_pos[:, None] + torch.arange(seq_len, device=device)
    start = (positions - window_size + 1).clamp(min=0)
    local_offsets = torch.arange(window_size, device=device)
    local = start.unsqueeze(-1) + local_offsets.view(1, 1, window_size)
    local_valid = (local <= positions.unsqueeze(-1)) & (
        local < source_seq_lens.view(batch_size, 1, 1)
    )
    local = torch.where(local_valid, local, torch.full_like(local, -1)).to(torch.int32)

    query_positions = input_pos[:, None] + torch.arange(1, seq_len + 1, device=device)
    valid_lengths = query_positions // compress_ratio
    compressed_positions = torch.arange(max_compressed_len, device=device)
    invalid = compressed_positions.view(1, 1, -1) >= valid_lengths.unsqueeze(-1)
    masked_score = torch.where(invalid, torch.full_like(index_score, float("-inf")), index_score)

    selected = masked_score.topk(min(topk_count, max_compressed_len), dim=-1).indices
    valid_selected = selected < valid_lengths.unsqueeze(-1)
    selected = torch.where(
        valid_selected,
        selected + source_seq_lens.view(batch_size, 1, 1),
        torch.full_like(selected, -1),
    )
    if selected.shape[-1] < topk_count:
        pad = selected.new_full((*selected.shape[:-1], topk_count - selected.shape[-1]), -1)
        selected = torch.cat([selected, pad], dim=-1)
    return torch.cat([local, selected.to(torch.int32)], dim=-1).contiguous()


def _run_ratio4_indexer_topk_device(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    topk_count: int,
    compress_ratio: int,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    output = _ratio4_indexer_topk_device(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_ratio4_indexer_scores", mutates_args=())
def torch_deepseek_v4_ratio4_indexer_scores(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
    weights_are_scaled: bool = False,
) -> torch.Tensor:
    return deepseek_v4_ratio4_indexer_scores_ref(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
        weights_are_scaled,
    )


@torch_deepseek_v4_ratio4_indexer_scores.register_fake
def torch_deepseek_v4_ratio4_indexer_scores_fake(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
    weights_are_scaled: bool = False,
) -> torch.Tensor:
    del weights_are_scaled
    _validate_ratio4_indexer_scores_contract(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    return q.new_empty(q.shape[0], q.shape[1], max_compressed_len, dtype=torch.float32)


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_ratio4_indexer_scores", mutates_args=())
def triton_deepseek_v4_ratio4_indexer_scores(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
    weights_are_scaled: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_ratio4_indexer_scores_contract(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    if max_compressed_len != _DSV4_RATIO4_MAX_COMPRESSED_LEN:
        raise ValueError(
            f"max_compressed_len must be {_DSV4_RATIO4_MAX_COMPRESSED_LEN}, got {max_compressed_len}"
        )
    _validate_ratio4_indexer_scores_out(q, max_compressed_len, out)
    skip_reason = _triton_ratio4_indexer_scores_skip_reason(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        out,
    )
    if skip_reason is None:
        return _run_ratio4_indexer_scores_device(
            q,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            weights,
            freqs_cis,
            eps,
            rope_dim,
            max_compressed_len,
            fp4_block_size,
            weights_are_scaled,
            out,
        )
    if _any_cuda_tensor(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        out,
    ):
        raise RuntimeError(skip_reason)

    output = deepseek_v4_ratio4_indexer_scores_ref(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
        weights_are_scaled,
    )
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@triton_deepseek_v4_ratio4_indexer_scores.register_fake
def triton_deepseek_v4_ratio4_indexer_scores_fake(
    q: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float,
    rope_dim: int = _DSV4_TRITON_ROPE_DIM,
    max_compressed_len: int = _DSV4_RATIO4_MAX_COMPRESSED_LEN,
    fp4_block_size: int = _DSV4_RATIO4_INDEXER_FP4_BLOCK_SIZE,
    weights_are_scaled: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del weights_are_scaled
    _validate_ratio4_indexer_scores_contract(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs_cis,
        eps,
        rope_dim,
        max_compressed_len,
        fp4_block_size,
    )
    if max_compressed_len != _DSV4_RATIO4_MAX_COMPRESSED_LEN:
        raise ValueError(
            f"max_compressed_len must be {_DSV4_RATIO4_MAX_COMPRESSED_LEN}, got {max_compressed_len}"
        )
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape[0], q.shape[1], max_compressed_len, dtype=torch.float32)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_ratio4_indexer_topk", mutates_args=())
def torch_deepseek_v4_ratio4_indexer_topk(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int = _DSV4_RATIO4_WINDOW_SIZE,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
) -> torch.Tensor:
    return deepseek_v4_ratio4_indexer_build_topk_ref(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )


@torch_deepseek_v4_ratio4_indexer_topk.register_fake
def torch_deepseek_v4_ratio4_indexer_topk_fake(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int = _DSV4_RATIO4_WINDOW_SIZE,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
) -> torch.Tensor:
    _validate_ratio4_indexer_topk_inputs(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    return torch.empty(
        index_score.shape[0],
        index_score.shape[1],
        window_size + topk_count,
        dtype=torch.int32,
        device=index_score.device,
    )


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_ratio4_indexer_topk", mutates_args=())
def triton_deepseek_v4_ratio4_indexer_topk(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int = _DSV4_RATIO4_WINDOW_SIZE,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_ratio4_indexer_topk_inputs(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    if index_score.shape[-1] != _DSV4_RATIO4_MAX_COMPRESSED_LEN:
        raise ValueError(
            f"index_score width must be {_DSV4_RATIO4_MAX_COMPRESSED_LEN}, "
            f"got {index_score.shape[-1]}"
        )
    if window_size != _DSV4_RATIO4_WINDOW_SIZE:
        raise ValueError(f"window_size must be {_DSV4_RATIO4_WINDOW_SIZE}, got {window_size}")
    if topk_count != _DSV4_RATIO4_TOPK:
        raise ValueError(f"topk_count must be {_DSV4_RATIO4_TOPK}, got {topk_count}")
    _validate_ratio4_indexer_topk_out(index_score, window_size, topk_count, out)
    skip_reason = _triton_ratio4_indexer_topk_skip_reason(
        index_score, source_seq_lens, input_pos, out
    )
    if skip_reason is None:
        return _run_ratio4_indexer_topk_device(
            index_score,
            source_seq_lens,
            input_pos,
            window_size,
            topk_count,
            compress_ratio,
            out,
        )
    if _any_cuda_tensor(index_score, source_seq_lens, input_pos, out):
        raise RuntimeError(skip_reason)

    output = deepseek_v4_ratio4_indexer_build_topk_ref(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@triton_deepseek_v4_ratio4_indexer_topk.register_fake
def triton_deepseek_v4_ratio4_indexer_topk_fake(
    index_score: torch.Tensor,
    source_seq_lens: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int = _DSV4_RATIO4_WINDOW_SIZE,
    topk_count: int = _DSV4_RATIO4_TOPK,
    compress_ratio: int = _DSV4_RATIO4_COMPRESS_RATIO,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_ratio4_indexer_topk_inputs(
        index_score, source_seq_lens, input_pos, window_size, topk_count, compress_ratio
    )
    if index_score.shape[-1] != _DSV4_RATIO4_MAX_COMPRESSED_LEN:
        raise ValueError(
            f"index_score width must be {_DSV4_RATIO4_MAX_COMPRESSED_LEN}, "
            f"got {index_score.shape[-1]}"
        )
    if window_size != _DSV4_RATIO4_WINDOW_SIZE:
        raise ValueError(f"window_size must be {_DSV4_RATIO4_WINDOW_SIZE}, got {window_size}")
    if topk_count != _DSV4_RATIO4_TOPK:
        raise ValueError(f"topk_count must be {_DSV4_RATIO4_TOPK}, got {topk_count}")
    if out is not None:
        return out.new_empty(0)
    return torch.empty(
        index_score.shape[0],
        index_score.shape[1],
        window_size + topk_count,
        dtype=torch.int32,
        device=index_score.device,
    )


def deepseek_v4_indexer_q_rope_quant_ref(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
    quant_format: str = "fp8",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to indexer Q and quantize it for the initial FP8 path."""
    if quant_format != "fp8":
        raise ValueError(f"Only quant_format='fp8' is supported, got {quant_format!r}")
    q_rope = _apply_rope_ref(q, freqs_cis, rope_dim)
    return deepseek_v4_fp8_block_quant_ref(q_rope, fp8_block_size)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_indexer_q_rope_quant", mutates_args=())
def torch_deepseek_v4_indexer_q_rope_quant(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
    quant_format: str = "fp8",
) -> Tuple[torch.Tensor, torch.Tensor]:
    return deepseek_v4_indexer_q_rope_quant_ref(
        q, freqs_cis, rope_dim, fp8_block_size, quant_format
    )


@torch_deepseek_v4_indexer_q_rope_quant.register_fake
def torch_deepseek_v4_indexer_q_rope_quant_fake(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
    quant_format: str = "fp8",
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_rope_dim(rope_dim, q.shape[-1])
    _validate_block_size(fp8_block_size)
    if quant_format != "fp8":
        raise ValueError(f"Only quant_format='fp8' is supported, got {quant_format!r}")
    num_blocks = (q.shape[-1] + fp8_block_size - 1) // fp8_block_size
    return (
        torch.empty(q.shape, dtype=FP8_E4M3_DTYPE, device=q.device),
        _fake_scale((*q.shape[:-1], num_blocks), q.device),
    )


def deepseek_v4_inverse_rope_fp8_output_quant_ref(
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply inverse RoPE to attention output and quantize for FP8 output projection."""
    output = _apply_rope_ref(output, freqs_cis, rope_dim, inverse=True)
    return deepseek_v4_fp8_block_quant_ref(output, fp8_block_size)


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_inverse_rope_fp8_output_quant", mutates_args=()
)
def torch_deepseek_v4_inverse_rope_fp8_output_quant(
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return deepseek_v4_inverse_rope_fp8_output_quant_ref(
        output, freqs_cis, rope_dim, fp8_block_size
    )


@torch_deepseek_v4_inverse_rope_fp8_output_quant.register_fake
def torch_deepseek_v4_inverse_rope_fp8_output_quant_fake(
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    fp8_block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_rope_dim(rope_dim, output.shape[-1])
    _validate_block_size(fp8_block_size)
    num_blocks = (output.shape[-1] + fp8_block_size - 1) // fp8_block_size
    return (
        torch.empty(output.shape, dtype=FP8_E4M3_DTYPE, device=output.device),
        _fake_scale((*output.shape[:-1], num_blocks), output.device),
    )


def deepseek_v4_local_window_topk_idxs(
    window_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build source-op indices for causal local-window attention."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    positions = torch.arange(seq_len, device=device)
    offsets = torch.arange(window_size, device=device)
    matrix = (positions[:, None] - window_size + 1).clamp(min=0) + offsets[None, :]
    matrix = torch.where(matrix <= positions[:, None], matrix, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def deepseek_v4_sparse_attention_microkernel_ref(
    q: torch.Tensor,
    local_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    compressed_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention over local-window rows plus selected compressed rows.

    ``compressed_idxs`` is relative to ``compressed_kv``. Non-negative entries
    are offset internally before calling the Plan 07 source op.
    """
    _validate_rank("q", q, 4)
    _validate_rank("local_kv", local_kv, 3)
    _validate_rank("compressed_kv", compressed_kv, 3)
    _validate_rank("compressed_idxs", compressed_idxs, 3)

    batch_size, seq_len, _, _ = q.shape
    local_idxs = deepseek_v4_local_window_topk_idxs(window_size, batch_size, seq_len, q.device)
    if compressed_idxs.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            f"compressed_idxs prefix must be {(batch_size, seq_len)}, "
            f"got {tuple(compressed_idxs.shape[:2])}"
        )

    compressed_offset = torch.where(
        compressed_idxs >= 0,
        compressed_idxs + local_kv.shape[1],
        compressed_idxs,
    )
    topk_idxs = torch.cat([local_idxs.to(compressed_idxs.dtype), compressed_offset], dim=-1)
    kv = torch.cat([local_kv, compressed_kv], dim=1)
    return torch_deepseek_v4_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_microkernel", mutates_args=()
)
def torch_deepseek_v4_sparse_attention_microkernel(
    q: torch.Tensor,
    local_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    compressed_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    return deepseek_v4_sparse_attention_microkernel_ref(
        q, local_kv, compressed_kv, compressed_idxs, attn_sink, window_size, softmax_scale
    )


@torch_deepseek_v4_sparse_attention_microkernel.register_fake
def torch_deepseek_v4_sparse_attention_microkernel_fake(
    q: torch.Tensor,
    local_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    compressed_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    _validate_rank("q", q, 4)
    return q.new_empty(q.shape).contiguous()
