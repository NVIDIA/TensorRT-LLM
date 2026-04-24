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

from ...utils.e8m0 import maybe_e8m0_to_fp32
from .deepseek_v4_attention import torch_deepseek_v4_sparse_attention

FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = torch.finfo(FP8_E4M3_DTYPE).max
FP8_E4M3_MIN = torch.finfo(FP8_E4M3_DTYPE).min

__all__ = [
    "deepseek_v4_compressor_pool_norm_rope_ref",
    "deepseek_v4_fp8_block_dequant_ref",
    "deepseek_v4_fp8_block_quant_ref",
    "deepseek_v4_indexer_q_rope_quant_ref",
    "deepseek_v4_inverse_rope_fp8_output_quant_ref",
    "deepseek_v4_kv_rmsnorm_rope_cache_insert_ref",
    "deepseek_v4_kv_rmsnorm_rope_ref",
    "deepseek_v4_local_window_topk_idxs",
    "deepseek_v4_q_rmsnorm_rope_ref",
    "deepseek_v4_sparse_attention_microkernel_ref",
    "torch_deepseek_v4_compressor_pool_norm_rope",
    "torch_deepseek_v4_indexer_q_rope_quant",
    "torch_deepseek_v4_inverse_rope_fp8_output_quant",
    "torch_deepseek_v4_kv_rmsnorm_rope_cache_insert",
    "torch_deepseek_v4_q_rmsnorm_rope",
    "torch_deepseek_v4_sparse_attention_microkernel",
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
