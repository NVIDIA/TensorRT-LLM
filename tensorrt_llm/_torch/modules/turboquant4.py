# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TurboQuant4 KV cache compression utilities.

Implements the turbo4-resurrection variant: 4-bit PolarQuant with Walsh-Hadamard
Transform rotation and Lloyd-Max optimal centroids. No QJL correction.

Reference: https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md
"""

import os

import torch
import torch.nn.functional as F

_REFERENCE_HEAD_DIM = 128

# 16 Lloyd-Max optimal centroids for N(0, 1/sqrt(d)), d=128. Symmetric around
# zero. Each maps to a 4-bit index (0-15). Scale by sqrt(128 / head_dim) before
# using them with other head dimensions.
TURBOQUANT4_CENTROIDS = torch.tensor(
    [
        -0.1739,
        -0.1172,
        -0.0895,
        -0.0688,
        -0.0513,
        -0.0356,
        -0.0210,
        -0.0069,
        0.0069,
        0.0210,
        0.0356,
        0.0513,
        0.0688,
        0.0895,
        0.1172,
        0.1739,
    ],
    dtype=torch.float32,
)

# Decision boundaries: midpoints between adjacent centroids.
# Used by torch.bucketize for fast nearest-centroid assignment.
TURBOQUANT4_BOUNDARIES = torch.tensor(
    [
        (-0.1739 + -0.1172) / 2,
        (-0.1172 + -0.0895) / 2,
        (-0.0895 + -0.0688) / 2,
        (-0.0688 + -0.0513) / 2,
        (-0.0513 + -0.0356) / 2,
        (-0.0356 + -0.0210) / 2,
        (-0.0210 + -0.0069) / 2,
        (-0.0069 + 0.0069) / 2,
        (0.0069 + 0.0210) / 2,
        (0.0210 + 0.0356) / 2,
        (0.0356 + 0.0513) / 2,
        (0.0513 + 0.0688) / 2,
        (0.0688 + 0.0895) / 2,
        (0.0895 + 0.1172) / 2,
        (0.1172 + 0.1739) / 2,
    ],
    dtype=torch.float32,
)

_SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _use_native_ops() -> bool:
    return os.environ.get("TLLM_TURBOQUANT4_DISABLE_NATIVE", "0") != "1"


def _validate_activation_dtype(dtype: torch.dtype, op_name: str) -> None:
    if dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise NotImplementedError(f"TurboQuant4 {op_name} supports FP16, BF16, and FP32 tensors.")


def _validate_output_dtype(dtype: torch.dtype) -> None:
    if dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise NotImplementedError(
            "TurboQuant4 dequantize supports FP16, BF16, and FP32 output tensors."
        )


def _validate_int32_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.int32:
        raise ValueError(f"TurboQuant4 {name} must be int32.")


def _validate_same_device(reference: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if tensor.device != reference.device:
        raise ValueError(
            f"TurboQuant4 {name} must be on the same device as {reference.device}, "
            f"got {tensor.device}."
        )


def _validate_attention_head_counts(num_heads: int, num_kv_heads: int) -> None:
    if num_heads <= 0:
        raise ValueError("TurboQuant4 attention requires at least one query head.")
    if num_kv_heads <= 0:
        raise ValueError("TurboQuant4 attention requires at least one KV head.")
    if num_heads % num_kv_heads != 0:
        raise ValueError("TurboQuant4 attention heads must be divisible by KV heads.")


def _scaled_quantization_grid(
    head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (_REFERENCE_HEAD_DIM / head_dim) ** 0.5
    centroids = TURBOQUANT4_CENTROIDS.to(device=device, dtype=torch.float32) * scale
    boundaries = TURBOQUANT4_BOUNDARIES.to(device=device, dtype=torch.float32) * scale
    return centroids, boundaries


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform via butterfly operations.

    Applies the normalized WHT along the last dimension of x.
    The normalized WHT is its own inverse: fwht(fwht(x)) = x.

    Args:
        x: Input tensor with last dimension being a power of 2.

    Returns:
        WHT-transformed tensor, same shape as input.
    """
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"TurboQuant4 requires power-of-2 head_dim, got {n}")

    orig_dtype = x.dtype
    # Work on a float32 copy for numerical stability and to avoid mutating the
    # caller's tensor when the input is already float32.
    x = x.to(dtype=torch.float32, copy=True).contiguous()

    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*x.shape[:-3], n)
        h *= 2

    return (x * (n**-0.5)).to(orig_dtype)


def turboquant4_quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    """Lossy round-trip: quantize to 4-bit centroids then dequantize.

    Applies WHT rotation, scalar quantizes each element to the nearest
    of 16 Lloyd-Max centroids, then inverse-WHT back. The result is a
    BF16/FP16 tensor with the same shape as input, but with the information
    loss of 4-bit quantization applied.

    Args:
        x: KV tensor of shape [..., head_dim] where head_dim is power of 2.

    Returns:
        Quantize-dequantized tensor, same shape and dtype as input.
    """
    nibbles, scale = turboquant4_quantize(x)
    return turboquant4_dequantize(nibbles, scale, dtype=x.dtype)


def turboquant4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to packed 4-bit nibbles plus one scale per KV vector.

    Args:
        x: KV tensor of shape [..., head_dim] where head_dim is an even power of 2.

    Returns:
        nibbles: uint8 tensor of shape [..., head_dim // 2].
        scale: float32 tensor of shape [..., 1].
    """
    if x.ndim < 1:
        raise ValueError("TurboQuant4 input must have at least one dimension.")
    _validate_activation_dtype(x.dtype, "quantize")
    if x.is_cuda and _use_native_ops():
        try:
            return torch.ops.trtllm.turboquant4_quantize(x.contiguous())
        except AttributeError:
            pass

    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"TurboQuant4 requires even head_dim, got {head_dim}")

    # 1. Rotate into WHT basis.
    rotated_float = fwht(x.float())

    # 2. Compute an initial per-vector scale.
    #    For a unit-norm vector after WHT, each coordinate ~ N(0, 1/sqrt(d)).
    #    Scale captures the vector magnitude so centroids calibrated for the
    #    unit-norm distribution can be applied to the normalized values.
    scale = rotated_float.norm(dim=-1, keepdim=True) + 1e-10

    # 3. Normalize to the unit-norm distribution used by the centroids.
    normalized = rotated_float / scale

    # 4. Map each element to nearest centroid via bucket boundaries.
    centroids, boundaries = _scaled_quantization_grid(head_dim, x.device)
    indices = torch.bucketize(normalized, boundaries)

    # 5. Refit the scale to the selected centroids. This preserves the
    #    original Lloyd-Max bins while reducing value reconstruction error.
    quantized = centroids[indices]
    refined_scale = (rotated_float * quantized).sum(
        dim=-1, keepdim=True) / quantized.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-10)
    scale = refined_scale.clamp_min(1e-10)

    # 6. Pack adjacent 4-bit indices into one uint8 container.
    low = indices[..., 0::2].to(torch.uint8)
    high = indices[..., 1::2].to(torch.uint8)
    nibbles = low | (high << 4)

    return nibbles, scale


def turboquant4_dequantize(
    nibbles: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize packed 4-bit nibbles plus scales back to a dense tensor."""
    if nibbles.dtype != torch.uint8:
        raise ValueError("TurboQuant4 codes must be uint8.")
    if scale.dtype != torch.float32:
        raise ValueError("TurboQuant4 scales must be float32.")
    if nibbles.device != scale.device:
        raise ValueError("TurboQuant4 scales must be on the same device as codes.")
    if nibbles.ndim < 1:
        raise ValueError("TurboQuant4 codes must have at least one dimension.")
    if nibbles.ndim != scale.ndim:
        raise ValueError("TurboQuant4 codes/scales rank mismatch.")
    if scale.shape[-1] != 1:
        raise ValueError("TurboQuant4 scales last dimension must be 1.")
    if nibbles.shape[:-1] != scale.shape[:-1]:
        raise ValueError("TurboQuant4 codes/scales shape mismatch.")
    _validate_output_dtype(dtype)
    if nibbles.is_cuda and _use_native_ops():
        try:
            return torch.ops.trtllm.turboquant4_dequantize(
                nibbles.contiguous(), scale.contiguous(), dtype
            )
        except AttributeError:
            pass

    low = (nibbles & 0x0F).to(torch.int64)
    high = ((nibbles >> 4) & 0x0F).to(torch.int64)
    head_dim_half = nibbles.shape[-1]
    indices = torch.stack([low, high], dim=-1).view(*nibbles.shape[:-1], head_dim_half * 2)

    centroids, _ = _scaled_quantization_grid(head_dim_half * 2, nibbles.device)
    quantized = centroids[indices]

    return fwht(quantized * scale.float()).to(dtype)


def _block_ids_to_tensor(block_ids: list[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(block_ids, torch.Tensor):
        _validate_int32_tensor(block_ids, "block ids")
        return block_ids.to(device=device, non_blocking=True)
    return torch.tensor(block_ids, dtype=torch.int32, device=device)


def _batch_block_ids_to_tensor(
    block_ids: list[list[int]] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    if isinstance(block_ids, torch.Tensor):
        if block_ids.ndim != 2:
            raise ValueError("TurboQuant4 batched block ids must be a 2D tensor.")
        _validate_int32_tensor(block_ids, "block ids")
        return block_ids.to(device=device, non_blocking=True)
    if not block_ids:
        raise ValueError("TurboQuant4 batched block ids must not be empty.")
    max_blocks = max(len(ids) for ids in block_ids)
    if max_blocks == 0:
        raise ValueError("TurboQuant4 batched block ids must include at least one block.")
    result = torch.zeros((len(block_ids), max_blocks), dtype=torch.int32, device=device)
    for batch_idx, ids in enumerate(block_ids):
        result[batch_idx, : len(ids)] = torch.tensor(ids, dtype=torch.int32, device=device)
    return result


def _validate_block_ids(
    block_ids: list[int] | torch.Tensor,
    required_blocks: int,
    max_blocks: int,
) -> None:
    if isinstance(block_ids, torch.Tensor):
        _validate_int32_tensor(block_ids, "block ids")
    if isinstance(block_ids, torch.Tensor) and block_ids.ndim != 1:
        raise ValueError("TurboQuant4 block ids must be a 1D tensor.")
    block_count = len(block_ids)
    if block_count < required_blocks:
        raise RuntimeError(
            "TurboQuant4 KV cache block list is shorter than the requested "
            f"sequence; need {required_blocks}, got {block_count}."
        )
    if required_blocks == 0:
        return
    if isinstance(block_ids, torch.Tensor) and block_ids.is_cuda:
        required_block_ids = block_ids[:required_blocks]
        invalid = (required_block_ids < 0) | (required_block_ids >= max_blocks)
        if bool(invalid.any().item()):
            block_id = int(required_block_ids[invalid][0].item())
            raise RuntimeError(
                f"TurboQuant4 KV cache block id {block_id} is out of range "
                f"for {max_blocks} cache blocks."
            )
        return
    values = block_ids.tolist() if isinstance(block_ids, torch.Tensor) else block_ids
    for block_id in values[:required_blocks]:
        if block_id < 0 or block_id >= max_blocks:
            raise RuntimeError(
                f"TurboQuant4 KV cache block id {block_id} is out of range "
                f"for {max_blocks} cache blocks."
            )


def _validate_batch_block_ids(
    block_ids: list[list[int]] | torch.Tensor,
    seq_lens: torch.Tensor,
    tokens_per_block: int,
    max_blocks: int,
) -> None:
    if isinstance(block_ids, torch.Tensor):
        if block_ids.ndim != 2:
            raise ValueError("TurboQuant4 batched block ids must be a 2D tensor.")
        _validate_int32_tensor(block_ids, "block ids")
    seq_len_values = seq_lens.detach().cpu().tolist()
    if isinstance(block_ids, torch.Tensor):
        if block_ids.shape[0] != len(seq_len_values):
            raise ValueError("TurboQuant4 batch attention block-id batch size mismatch.")
        for batch_idx, seq_len in enumerate(seq_len_values):
            required_blocks = (
                (int(seq_len) + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
            )
            if block_ids.shape[1] < required_blocks:
                raise RuntimeError(
                    "TurboQuant4 KV cache block list is shorter than the "
                    f"requested sequence for batch {batch_idx}; need "
                    f"{required_blocks}, got {block_ids.shape[1]}."
                )
            if required_blocks == 0:
                continue
            required_block_ids = block_ids[batch_idx, :required_blocks]
            invalid = (required_block_ids < 0) | (required_block_ids >= max_blocks)
            if bool(invalid.any().item()):
                block_id = int(required_block_ids[invalid][0].item())
                raise RuntimeError(
                    f"TurboQuant4 KV cache block id {block_id} is out of range "
                    f"for {max_blocks} cache blocks in batch {batch_idx}."
                )
        return

    block_id_rows = block_ids
    if len(block_id_rows) != len(seq_len_values):
        raise ValueError("TurboQuant4 batch attention block-id batch size mismatch.")

    for batch_idx, (row, seq_len) in enumerate(zip(block_id_rows, seq_len_values)):
        required_blocks = (
            (int(seq_len) + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
        )
        if len(row) < required_blocks:
            raise RuntimeError(
                "TurboQuant4 KV cache block list is shorter than the "
                f"requested sequence for batch {batch_idx}; need "
                f"{required_blocks}, got {len(row)}."
            )
        for block_id in row[:required_blocks]:
            if block_id < 0 or block_id >= max_blocks:
                raise RuntimeError(
                    f"TurboQuant4 KV cache block id {block_id} is out of range "
                    f"for {max_blocks} cache blocks in batch {batch_idx}."
                )


def _validate_batch_query_metadata(
    q_batch_indices: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> None:
    batch_size = seq_lens_cpu.numel()
    invalid_batch = (q_batch_indices < 0) | (q_batch_indices >= batch_size)
    if bool(invalid_batch.any().item()):
        batch_idx = int(q_batch_indices[invalid_batch][0].item())
        raise RuntimeError(
            f"TurboQuant4 batch index {batch_idx} is out of range "
            f"for batch size {batch_size}."
        )

    invalid_query_position = query_positions < 0
    if bool(invalid_query_position.any().item()):
        query_index = int(
            torch.nonzero(invalid_query_position, as_tuple=False)[0].item())
        query_position = int(query_positions[query_index].item())
        raise RuntimeError(
            f"TurboQuant4 query position {query_position} is "
            f"negative at query index {query_index}."
        )

    metadata_device = q_batch_indices.device
    batch_indices = q_batch_indices.to(device=metadata_device, dtype=torch.long)
    seq_lens_for_queries = seq_lens_cpu.to(
        device=metadata_device)[batch_indices]
    query_positions_for_check = query_positions.to(device=metadata_device)
    invalid_query_position = query_positions_for_check >= seq_lens_for_queries
    if bool(invalid_query_position.any().item()):
        query_index = int(
            torch.nonzero(invalid_query_position, as_tuple=False)[0].item())
        query_position = int(query_positions_for_check[query_index].item())
        seq_len = int(seq_lens_for_queries[query_index].item())
        raise RuntimeError(
            "TurboQuant4 query position must be within the KV sequence "
            f"length for query index {query_index}; got {query_position} "
            f"for seq_len {seq_len}."
        )


def _validate_cache_tensors(
    kv_cache_tensor: torch.Tensor,
    kv_cache_scales: torch.Tensor,
    kv_index: int,
    tokens_per_block: int,
) -> None:
    if tokens_per_block <= 0:
        raise ValueError("TurboQuant4 tokens_per_block must be positive.")
    if kv_cache_tensor.dtype != torch.uint8:
        raise ValueError("TurboQuant4 cache must be uint8.")
    if kv_cache_scales.dtype != torch.float32:
        raise ValueError("TurboQuant4 scale cache must be float32.")
    if kv_cache_tensor.device != kv_cache_scales.device:
        raise ValueError("TurboQuant4 scale cache must be on the same device as cache.")
    if kv_cache_tensor.ndim != 5:
        raise ValueError(
            "TurboQuant4 cache must have shape [blocks, kv, tokens, heads, head_dim / 2]."
        )
    if kv_cache_scales.ndim != 5:
        raise ValueError("TurboQuant4 scale cache must have shape [blocks, kv, tokens, heads, 1].")
    if kv_cache_tensor.shape[:-1] != kv_cache_scales.shape[:-1]:
        raise ValueError("TurboQuant4 cache/scales leading shapes mismatch.")
    if kv_cache_scales.shape[-1] != 1:
        raise ValueError("TurboQuant4 scale cache last dimension must be 1.")
    if kv_cache_tensor.shape[2] != tokens_per_block:
        raise ValueError("TurboQuant4 cache tokens_per_block mismatch.")
    if kv_index < 0 or kv_index >= kv_cache_tensor.shape[1]:
        raise ValueError("TurboQuant4 kv_index is out of range.")
    head_dim = kv_cache_tensor.shape[-1] * 2
    if head_dim <= 0 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(f"TurboQuant4 requires power-of-2 head_dim, got {head_dim}")


def turboquant4_update_cache(
    x: torch.Tensor,
    kv_cache_tensor: torch.Tensor,
    kv_cache_scales: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    kv_index: int,
    start_pos: int,
    tokens_per_block: int,
) -> None:
    """Quantize ``x`` and write it into paged TurboQuant4 KV cache buffers.

    Args:
        x: Dense KV tensor with shape [seq_len, num_kv_heads, head_dim].
        kv_cache_tensor: Packed cache, shape [num_blocks, 2, tokens, heads, head_dim // 2].
        kv_cache_scales: Scale cache, shape [num_blocks, 2, tokens, heads, 1].
        block_ids: Cache block ids for this request.
        kv_index: 0 for K, 1 for V.
        start_pos: First absolute token position to update.
        tokens_per_block: Number of tokens in each cache block.
    """
    if start_pos < 0:
        raise ValueError("TurboQuant4 cache update start_pos must be non-negative.")
    if x.ndim != 3:
        raise ValueError(
            f"TurboQuant4 cache update expects [seq, heads, dim], got {tuple(x.shape)}"
        )
    _validate_activation_dtype(x.dtype, "cache update")
    _validate_cache_tensors(kv_cache_tensor, kv_cache_scales, kv_index, tokens_per_block)
    _validate_same_device(kv_cache_tensor, x, "cache update input")
    if x.shape[1] != kv_cache_tensor.shape[3]:
        raise ValueError("TurboQuant4 cache head count mismatch.")
    if x.shape[2] != kv_cache_tensor.shape[4] * 2:
        raise ValueError("TurboQuant4 cache head_dim mismatch.")
    required_blocks = (
        (start_pos + x.shape[0] + tokens_per_block - 1) // tokens_per_block if x.shape[0] > 0 else 0
    )
    _validate_block_ids(block_ids, required_blocks, kv_cache_tensor.shape[0])

    block_ids_tensor = _block_ids_to_tensor(block_ids, x.device)
    if x.is_cuda and _use_native_ops():
        try:
            torch.ops.trtllm.turboquant4_update_cache(
                x.contiguous(),
                kv_cache_tensor,
                kv_cache_scales,
                block_ids_tensor.contiguous(),
                kv_index,
                start_pos,
                tokens_per_block,
            )
            return
        except AttributeError:
            pass

    codes, scales = turboquant4_quantize(x.unsqueeze(0))
    token_offset = 0
    seq_len = x.shape[0]
    while token_offset < seq_len:
        token_pos = start_pos + token_offset
        block_list_pos = token_pos // tokens_per_block
        if block_list_pos >= block_ids_tensor.numel():
            raise RuntimeError(
                "TurboQuant4 KV cache block list is shorter than the "
                f"requested cache position {token_pos}."
            )
        block_id = int(block_ids_tensor[block_list_pos].item())
        block_offset = token_pos % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        tensor_slice = slice(token_offset, token_offset + num_tokens)
        kv_cache_tensor[block_id, kv_index, cache_slice].copy_(codes[0, tensor_slice])
        kv_cache_scales[block_id, kv_index, cache_slice].copy_(scales[0, tensor_slice])
        token_offset += num_tokens


def turboquant4_dequantize_cache(
    kv_cache_tensor: torch.Tensor,
    kv_cache_scales: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    kv_index: int,
    seq_len: int,
    tokens_per_block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read and dequantize a request's paged TurboQuant4 KV cache."""
    if seq_len < 0:
        raise ValueError("TurboQuant4 cache dequantize seq_len must be non-negative.")
    _validate_cache_tensors(kv_cache_tensor, kv_cache_scales, kv_index, tokens_per_block)
    required_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
    _validate_block_ids(block_ids, required_blocks, kv_cache_tensor.shape[0])
    block_ids_tensor = _block_ids_to_tensor(block_ids, kv_cache_tensor.device)
    if kv_cache_tensor.is_cuda and _use_native_ops():
        try:
            return torch.ops.trtllm.turboquant4_dequantize_cache(
                kv_cache_tensor,
                kv_cache_scales,
                block_ids_tensor.contiguous(),
                kv_index,
                seq_len,
                tokens_per_block,
                dtype,
            )
        except AttributeError:
            pass

    code_chunks = []
    scale_chunks = []
    token_offset = 0
    while token_offset < seq_len:
        block_list_pos = token_offset // tokens_per_block
        if block_list_pos >= block_ids_tensor.numel():
            raise RuntimeError(
                "TurboQuant4 KV cache block list is shorter than the "
                f"requested sequence length {seq_len}."
            )
        block_id = int(block_ids_tensor[block_list_pos].item())
        block_offset = token_offset % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        code_chunks.append(kv_cache_tensor[block_id, kv_index, cache_slice])
        scale_chunks.append(kv_cache_scales[block_id, kv_index, cache_slice])
        token_offset += num_tokens

    if not code_chunks:
        num_kv_heads = kv_cache_tensor.shape[-2]
        head_dim = kv_cache_tensor.shape[-1] * 2
        return kv_cache_tensor.new_empty((1, 0, num_kv_heads, head_dim), dtype=dtype)

    codes = torch.cat(code_chunks, dim=0).unsqueeze(0)
    scales = torch.cat(scale_chunks, dim=0).unsqueeze(0)
    return turboquant4_dequantize(codes, scales, dtype=dtype)


def _validate_dense_key_cache_tensors(
    key_cache_tensor: torch.Tensor,
    tokens_per_block: int,
) -> None:
    if key_cache_tensor.ndim != 4:
        raise ValueError(
            "TurboQuant4 dense key cache must have shape "
            "[blocks, tokens, heads, head_dim]."
        )
    if tokens_per_block <= 0:
        raise ValueError("TurboQuant4 tokens_per_block must be positive.")
    if key_cache_tensor.shape[1] != tokens_per_block:
        raise ValueError("TurboQuant4 dense key cache tokens_per_block mismatch.")
    _validate_activation_dtype(key_cache_tensor.dtype, "dense key cache")


def _validate_value_cache_tensors(
    value_cache_tensor: torch.Tensor,
    value_cache_scales: torch.Tensor,
    tokens_per_block: int,
) -> None:
    if value_cache_tensor.dtype != torch.uint8:
        raise ValueError("TurboQuant4 value cache codes must be uint8.")
    if value_cache_scales.dtype != torch.float32:
        raise ValueError("TurboQuant4 value cache scales must be float32.")
    if value_cache_tensor.ndim != 4 or value_cache_scales.ndim != 4:
        raise ValueError(
            "TurboQuant4 value cache must have shape "
            "[blocks, tokens, heads, packed_head_dim] plus scale shape "
            "[blocks, tokens, heads, 1]."
        )
    if tokens_per_block <= 0:
        raise ValueError("TurboQuant4 tokens_per_block must be positive.")
    if value_cache_tensor.shape[1] != tokens_per_block:
        raise ValueError("TurboQuant4 value cache tokens_per_block mismatch.")
    if value_cache_scales.shape[1] != tokens_per_block:
        raise ValueError("TurboQuant4 value scale cache tokens_per_block mismatch.")
    if value_cache_tensor.shape[:3] != value_cache_scales.shape[:3]:
        raise ValueError("TurboQuant4 value cache scale shape mismatch.")
    if value_cache_scales.shape[-1] != 1:
        raise ValueError("TurboQuant4 value scales last dimension must be 1.")


def update_turboquant4_dense_key_cache(
    x: torch.Tensor,
    key_cache_tensor: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    start_pos: int,
    tokens_per_block: int,
) -> None:
    """Write dense K states into the asymmetric TurboQuant4 cache layout."""
    if start_pos < 0:
        raise ValueError("TurboQuant4 key cache update start_pos must be non-negative.")
    if x.ndim != 3:
        raise ValueError(
            f"TurboQuant4 key cache update expects [seq, heads, dim], got {tuple(x.shape)}"
        )
    _validate_activation_dtype(x.dtype, "key cache update")
    _validate_dense_key_cache_tensors(key_cache_tensor, tokens_per_block)
    _validate_same_device(key_cache_tensor, x, "key cache update input")
    if x.shape[1:] != key_cache_tensor.shape[2:]:
        raise ValueError("TurboQuant4 dense key cache shape mismatch.")
    required_blocks = (
        (start_pos + x.shape[0] + tokens_per_block - 1) // tokens_per_block
        if x.shape[0] > 0
        else 0
    )
    _validate_block_ids(block_ids, required_blocks, key_cache_tensor.shape[0])

    block_ids_tensor = _block_ids_to_tensor(block_ids, x.device)
    token_offset = 0
    seq_len = x.shape[0]
    while token_offset < seq_len:
        token_pos = start_pos + token_offset
        block_list_pos = token_pos // tokens_per_block
        if block_list_pos >= block_ids_tensor.numel():
            raise RuntimeError(
                "TurboQuant4 key cache block list is shorter than the "
                f"requested cache position {token_pos}."
            )
        block_id = int(block_ids_tensor[block_list_pos].item())
        block_offset = token_pos % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        tensor_slice = slice(token_offset, token_offset + num_tokens)
        key_cache_tensor[block_id, cache_slice].copy_(x[tensor_slice])
        token_offset += num_tokens


def read_turboquant4_dense_key_cache(
    key_cache_tensor: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    seq_len: int,
    tokens_per_block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read dense K states from the asymmetric TurboQuant4 cache layout."""
    if seq_len < 0:
        raise ValueError("TurboQuant4 key cache read seq_len must be non-negative.")
    _validate_output_dtype(dtype)
    _validate_dense_key_cache_tensors(key_cache_tensor, tokens_per_block)
    required_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
    _validate_block_ids(block_ids, required_blocks, key_cache_tensor.shape[0])
    block_ids_tensor = _block_ids_to_tensor(block_ids, key_cache_tensor.device)

    chunks = []
    token_offset = 0
    while token_offset < seq_len:
        block_list_pos = token_offset // tokens_per_block
        if block_list_pos >= block_ids_tensor.numel():
            raise RuntimeError(
                "TurboQuant4 key cache block list is shorter than the "
                f"requested sequence length {seq_len}."
            )
        block_id = int(block_ids_tensor[block_list_pos].item())
        block_offset = token_offset % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        chunks.append(key_cache_tensor[block_id, cache_slice])
        token_offset += num_tokens

    if not chunks:
        num_kv_heads = key_cache_tensor.shape[-2]
        head_dim = key_cache_tensor.shape[-1]
        return key_cache_tensor.new_empty((1, 0, num_kv_heads, head_dim), dtype=dtype)

    return torch.cat(chunks, dim=0).unsqueeze(0).to(dtype=dtype)


def turboquant4_update_value_cache(
    x: torch.Tensor,
    value_cache_tensor: torch.Tensor,
    value_cache_scales: torch.Tensor,
    value_block_ids: list[int] | torch.Tensor,
    scale_block_ids: list[int] | torch.Tensor,
    start_pos: int,
    tokens_per_block: int,
) -> None:
    """Quantize V states and write them into the asymmetric TurboQuant4 cache."""
    if start_pos < 0:
        raise ValueError("TurboQuant4 value cache update start_pos must be non-negative.")
    if x.ndim != 3:
        raise ValueError(
            f"TurboQuant4 value cache update expects [seq, heads, dim], got {tuple(x.shape)}"
        )
    _validate_activation_dtype(x.dtype, "value cache update")
    _validate_value_cache_tensors(value_cache_tensor, value_cache_scales, tokens_per_block)
    _validate_same_device(value_cache_tensor, x, "value cache update input")
    _validate_same_device(value_cache_tensor, value_cache_scales, "value scale cache")
    if x.shape[1] != value_cache_tensor.shape[2]:
        raise ValueError("TurboQuant4 value cache head count mismatch.")
    if x.shape[2] != value_cache_tensor.shape[3] * 2:
        raise ValueError("TurboQuant4 value cache head_dim mismatch.")
    required_blocks = (
        (start_pos + x.shape[0] + tokens_per_block - 1) // tokens_per_block
        if x.shape[0] > 0
        else 0
    )
    _validate_block_ids(value_block_ids, required_blocks, value_cache_tensor.shape[0])
    _validate_block_ids(scale_block_ids, required_blocks, value_cache_scales.shape[0])

    value_block_ids_tensor = _block_ids_to_tensor(value_block_ids, x.device)
    scale_block_ids_tensor = _block_ids_to_tensor(scale_block_ids, x.device)
    codes, scales = turboquant4_quantize(x.unsqueeze(0))
    token_offset = 0
    seq_len = x.shape[0]
    while token_offset < seq_len:
        token_pos = start_pos + token_offset
        block_list_pos = token_pos // tokens_per_block
        if (block_list_pos >= value_block_ids_tensor.numel()
                or block_list_pos >= scale_block_ids_tensor.numel()):
            raise RuntimeError(
                "TurboQuant4 value cache block list is shorter than the "
                f"requested cache position {token_pos}."
            )
        value_block_id = int(value_block_ids_tensor[block_list_pos].item())
        scale_block_id = int(scale_block_ids_tensor[block_list_pos].item())
        block_offset = token_pos % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        tensor_slice = slice(token_offset, token_offset + num_tokens)
        value_cache_tensor[value_block_id, cache_slice].copy_(codes[0, tensor_slice])
        value_cache_scales[scale_block_id, cache_slice].copy_(scales[0, tensor_slice])
        token_offset += num_tokens


def turboquant4_dequantize_value_cache(
    value_cache_tensor: torch.Tensor,
    value_cache_scales: torch.Tensor,
    value_block_ids: list[int] | torch.Tensor,
    scale_block_ids: list[int] | torch.Tensor,
    seq_len: int,
    tokens_per_block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read and dequantize V states from the asymmetric TurboQuant4 cache."""
    if seq_len < 0:
        raise ValueError("TurboQuant4 value cache read seq_len must be non-negative.")
    _validate_value_cache_tensors(value_cache_tensor, value_cache_scales, tokens_per_block)
    _validate_output_dtype(dtype)
    required_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
    _validate_block_ids(value_block_ids, required_blocks, value_cache_tensor.shape[0])
    _validate_block_ids(scale_block_ids, required_blocks, value_cache_scales.shape[0])
    value_block_ids_tensor = _block_ids_to_tensor(value_block_ids, value_cache_tensor.device)
    scale_block_ids_tensor = _block_ids_to_tensor(scale_block_ids, value_cache_tensor.device)

    code_chunks = []
    scale_chunks = []
    token_offset = 0
    while token_offset < seq_len:
        block_list_pos = token_offset // tokens_per_block
        if (block_list_pos >= value_block_ids_tensor.numel()
                or block_list_pos >= scale_block_ids_tensor.numel()):
            raise RuntimeError(
                "TurboQuant4 value cache block list is shorter than the "
                f"requested sequence length {seq_len}."
            )
        value_block_id = int(value_block_ids_tensor[block_list_pos].item())
        scale_block_id = int(scale_block_ids_tensor[block_list_pos].item())
        block_offset = token_offset % tokens_per_block
        num_tokens = min(seq_len - token_offset, tokens_per_block - block_offset)
        cache_slice = slice(block_offset, block_offset + num_tokens)
        code_chunks.append(value_cache_tensor[value_block_id, cache_slice])
        scale_chunks.append(value_cache_scales[scale_block_id, cache_slice])
        token_offset += num_tokens

    if not code_chunks:
        num_kv_heads = value_cache_tensor.shape[-2]
        head_dim = value_cache_tensor.shape[-1] * 2
        return value_cache_tensor.new_empty((1, 0, num_kv_heads, head_dim), dtype=dtype)

    codes = torch.cat(code_chunks, dim=0).unsqueeze(0)
    scales = torch.cat(scale_chunks, dim=0).unsqueeze(0)
    return turboquant4_dequantize(codes, scales, dtype=dtype)


def _turboquant4_sdpa(
    q: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_kv_heads: int,
    seq_len: int,
    q_start_pos: int,
    q_scaling: float,
    is_causal: bool,
    attention_window_size: int | None,
) -> torch.Tensor:
    q_states = q.unsqueeze(0).transpose(1, 2)
    key_states = key_states.transpose(1, 2).to(q.dtype)
    value_states = value_states.transpose(1, 2).to(q.dtype)
    kv_repeats = q.shape[1] // num_kv_heads
    if kv_repeats != 1:
        key_states = key_states[:, :, None, :, :].expand(
            1, num_kv_heads, kv_repeats, seq_len, q.shape[-1]
        )
        key_states = key_states.reshape(1, q.shape[1], seq_len, q.shape[-1])
        value_states = value_states[:, :, None, :, :].expand(
            1, num_kv_heads, kv_repeats, seq_len, q.shape[-1]
        )
        value_states = value_states.reshape(1, q.shape[1], seq_len, q.shape[-1])

    window_size = attention_window_size or 0
    attn_mask = None
    if is_causal or window_size > 0:
        cache_position = torch.arange(q_start_pos, q_start_pos + q.shape[0], device=q.device)
        kv_positions = torch.arange(seq_len, device=q.device).unsqueeze(0)
        allowed = kv_positions <= cache_position.unsqueeze(-1)
        if window_size > 0:
            allowed = allowed & (kv_positions > cache_position.unsqueeze(-1) - window_size)
        attn_mask = allowed[None, None, :, :]

    scale = 1 / ((q.shape[-1] ** 0.5) * q_scaling)
    return (
        F.scaled_dot_product_attention(
            q_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=False,
            scale=scale,
        )
        .transpose(1, 2)
        .squeeze(0)
        .contiguous()
    )


def turboquant4_value_only_attention(
    q: torch.Tensor,
    key_cache_tensor: torch.Tensor,
    value_cache_tensor: torch.Tensor,
    value_cache_scales: torch.Tensor,
    key_block_ids: list[int] | torch.Tensor,
    value_block_ids: list[int] | torch.Tensor,
    scale_block_ids: list[int] | torch.Tensor,
    seq_len: int,
    q_start_pos: int,
    tokens_per_block: int,
    q_scaling: float,
    is_causal: bool,
    attention_window_size: int | None,
    dense_value_states: torch.Tensor | None = None,
    dense_value_start_pos: int | None = None,
) -> torch.Tensor:
    """Attention over dense K and TurboQuant4-packed V cache for one request."""
    if seq_len < 0:
        raise ValueError("TurboQuant4 attention seq_len must be non-negative.")
    if q_start_pos < 0:
        raise ValueError("TurboQuant4 attention q_start_pos must be non-negative.")
    if q_scaling <= 0:
        raise ValueError("TurboQuant4 attention q_scaling must be positive.")
    if attention_window_size is not None and attention_window_size < 0:
        raise ValueError("TurboQuant4 attention_window_size must be non-negative.")
    if q.ndim != 3:
        raise ValueError(f"TurboQuant4 attention expects [seq, heads, dim], got {tuple(q.shape)}")
    _validate_activation_dtype(q.dtype, "attention")
    _validate_dense_key_cache_tensors(key_cache_tensor, tokens_per_block)
    _validate_value_cache_tensors(value_cache_tensor, value_cache_scales, tokens_per_block)
    _validate_same_device(q, key_cache_tensor, "key cache")
    _validate_same_device(q, value_cache_tensor, "value cache")
    _validate_same_device(q, value_cache_scales, "value scale cache")
    if q.shape[-1] != key_cache_tensor.shape[-1]:
        raise ValueError("TurboQuant4 key cache head_dim mismatch.")
    if q.shape[-1] != value_cache_tensor.shape[-1] * 2:
        raise ValueError("TurboQuant4 value cache head_dim mismatch.")
    if key_cache_tensor.shape[2] != value_cache_tensor.shape[2]:
        raise ValueError("TurboQuant4 K/V cache head count mismatch.")
    _validate_attention_head_counts(q.shape[1], key_cache_tensor.shape[2])
    if q.shape[0] > 0 and q_start_pos + q.shape[0] > seq_len:
        raise ValueError(
            "TurboQuant4 attention query positions must be within the KV sequence length."
        )
    if dense_value_states is not None:
        if dense_value_start_pos is None:
            raise ValueError(
                "TurboQuant4 dense value suffix requires dense_value_start_pos."
            )
        if dense_value_start_pos < 0:
            raise ValueError(
                "TurboQuant4 dense value suffix start must be non-negative."
            )
        if dense_value_states.ndim != 3:
            raise ValueError(
                "TurboQuant4 dense value suffix expects [seq, heads, dim], "
                f"got {tuple(dense_value_states.shape)}."
            )
        _validate_activation_dtype(dense_value_states.dtype, "dense value suffix")
        _validate_same_device(q, dense_value_states, "dense value suffix")
        if dense_value_states.shape[1] != value_cache_tensor.shape[2]:
            raise ValueError("TurboQuant4 dense value suffix head count mismatch.")
        if dense_value_states.shape[2] != q.shape[-1]:
            raise ValueError("TurboQuant4 dense value suffix head_dim mismatch.")
        dense_value_end_pos = dense_value_start_pos + dense_value_states.shape[0]
        if dense_value_end_pos > seq_len:
            raise ValueError(
                "TurboQuant4 dense value suffix must fit within the KV sequence length."
            )
    required_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
    _validate_block_ids(key_block_ids, required_blocks, key_cache_tensor.shape[0])
    _validate_block_ids(value_block_ids, required_blocks, value_cache_tensor.shape[0])
    _validate_block_ids(scale_block_ids, required_blocks, value_cache_scales.shape[0])

    key_states = read_turboquant4_dense_key_cache(
        key_cache_tensor, key_block_ids, seq_len, tokens_per_block, q.dtype
    )
    value_states = turboquant4_dequantize_value_cache(
        value_cache_tensor,
        value_cache_scales,
        value_block_ids,
        scale_block_ids,
        seq_len,
        tokens_per_block,
        q.dtype,
    )
    if dense_value_states is not None and dense_value_states.shape[0] > 0:
        value_states[:, dense_value_start_pos:dense_value_end_pos].copy_(
            dense_value_states.unsqueeze(0).to(dtype=q.dtype)
        )
    return _turboquant4_sdpa(
        q,
        key_states,
        value_states,
        key_cache_tensor.shape[2],
        seq_len,
        q_start_pos,
        q_scaling,
        is_causal,
        attention_window_size,
    )


def turboquant4_value_only_batch_attention(
    q: torch.Tensor,
    key_cache_tensor: torch.Tensor,
    value_cache_tensor: torch.Tensor,
    value_cache_scales: torch.Tensor,
    key_block_ids: list[list[int]] | torch.Tensor,
    value_block_ids: list[list[int]] | torch.Tensor,
    scale_block_ids: list[list[int]] | torch.Tensor,
    q_batch_indices: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    tokens_per_block: int,
    q_scaling: float,
    is_causal: bool,
    attention_window_size: int | None,
    max_seq_len: int | None = None,
    dense_value_states_by_seq: list[torch.Tensor | None] | None = None,
    dense_value_start_positions: list[int | None] | None = None,
) -> torch.Tensor:
    """Batched attention over dense K and TurboQuant4-packed V cache."""
    if q.ndim != 3:
        raise ValueError(
            f"TurboQuant4 batch attention expects [seq, heads, dim], got {tuple(q.shape)}"
        )
    _validate_activation_dtype(q.dtype, "batch attention")
    if q_scaling <= 0:
        raise ValueError("TurboQuant4 attention q_scaling must be positive.")
    if attention_window_size is not None and attention_window_size < 0:
        raise ValueError("TurboQuant4 attention_window_size must be non-negative.")
    _validate_dense_key_cache_tensors(key_cache_tensor, tokens_per_block)
    _validate_value_cache_tensors(value_cache_tensor, value_cache_scales, tokens_per_block)
    _validate_same_device(q, key_cache_tensor, "key cache")
    _validate_same_device(q, value_cache_tensor, "value cache")
    _validate_same_device(q, value_cache_scales, "value scale cache")
    if q.shape[-1] != key_cache_tensor.shape[-1]:
        raise ValueError("TurboQuant4 key cache head_dim mismatch.")
    if q.shape[-1] != value_cache_tensor.shape[-1] * 2:
        raise ValueError("TurboQuant4 value cache head_dim mismatch.")
    if key_cache_tensor.shape[2] != value_cache_tensor.shape[2]:
        raise ValueError("TurboQuant4 K/V cache head count mismatch.")
    _validate_attention_head_counts(q.shape[1], key_cache_tensor.shape[2])
    if q_batch_indices.ndim != 1 or query_positions.ndim != 1:
        raise ValueError("TurboQuant4 batch attention query metadata must be 1D.")
    if q_batch_indices.numel() != q.shape[0] or query_positions.numel() != q.shape[0]:
        raise ValueError("TurboQuant4 batch attention query metadata length mismatch.")
    if seq_lens.ndim != 1:
        raise ValueError("TurboQuant4 batch attention seq_lens must be 1D.")
    if dense_value_states_by_seq is not None:
        if dense_value_start_positions is None:
            raise ValueError(
                "TurboQuant4 dense value suffixes require start positions."
            )
        if len(dense_value_states_by_seq) != seq_lens.numel():
            raise ValueError(
                "TurboQuant4 dense value suffix batch size mismatch."
            )
        if len(dense_value_start_positions) != seq_lens.numel():
            raise ValueError(
                "TurboQuant4 dense value suffix start batch size mismatch."
            )
    elif dense_value_start_positions is not None:
        raise ValueError(
            "TurboQuant4 dense value suffix starts require suffix tensors."
        )
    _validate_int32_tensor(q_batch_indices, "batch attention q_batch_indices")
    _validate_int32_tensor(query_positions, "batch attention query_positions")
    _validate_int32_tensor(seq_lens, "batch attention seq_lens")

    seq_lens_cpu = seq_lens.detach().cpu()
    if bool((seq_lens_cpu < 0).any().item()):
        raise ValueError("TurboQuant4 batch attention seq_lens must be non-negative.")
    actual_max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
    if max_seq_len is None:
        max_seq_len = actual_max_seq_len
    elif max_seq_len < 0:
        raise ValueError("TurboQuant4 batch attention max_seq_len must be non-negative.")
    elif max_seq_len < actual_max_seq_len:
        raise ValueError(
            "TurboQuant4 batch attention max_seq_len must be at least "
            f"the maximum sequence length; got {max_seq_len}, "
            f"max seq_len is {actual_max_seq_len}."
        )
    _validate_batch_block_ids(key_block_ids, seq_lens, tokens_per_block, key_cache_tensor.shape[0])
    _validate_batch_block_ids(value_block_ids, seq_lens, tokens_per_block, value_cache_tensor.shape[0])
    _validate_batch_block_ids(scale_block_ids, seq_lens, tokens_per_block, value_cache_scales.shape[0])
    key_block_ids_tensor = _batch_block_ids_to_tensor(key_block_ids, q.device)
    value_block_ids_tensor = _batch_block_ids_to_tensor(value_block_ids, q.device)
    scale_block_ids_tensor = _batch_block_ids_to_tensor(scale_block_ids, q.device)
    required_blocks = (
        (max_seq_len + tokens_per_block - 1) // tokens_per_block if max_seq_len > 0 else 0
    )
    for name, tensor in (
        ("key", key_block_ids_tensor),
        ("value", value_block_ids_tensor),
        ("scale", scale_block_ids_tensor),
    ):
        if tensor.shape[0] != seq_lens.numel():
            raise ValueError(f"TurboQuant4 {name} block-id batch size mismatch.")
        if tensor.shape[1] < required_blocks:
            raise RuntimeError(
                "TurboQuant4 KV cache block list is shorter than the requested "
                f"batch sequence for {name}; need {required_blocks}, got {tensor.shape[1]}."
            )
    _validate_batch_query_metadata(q_batch_indices, query_positions,
                                   seq_lens_cpu)

    output = torch.zeros_like(q)
    q_batch_indices_cpu = q_batch_indices.cpu()
    query_positions_cpu = query_positions.cpu()
    seq_lens_cpu = seq_lens.cpu()
    for batch_idx in range(seq_lens_cpu.numel()):
        query_mask = q_batch_indices_cpu == batch_idx
        if not torch.any(query_mask):
            continue
        query_indices_cpu = torch.nonzero(query_mask, as_tuple=False).flatten()
        for query_index_cpu in query_indices_cpu:
            query_index = int(query_index_cpu.item())
            query_position = int(query_positions_cpu[query_index].item())
            dense_value_states = None
            dense_value_start_pos = None
            if dense_value_states_by_seq is not None:
                dense_value_states = dense_value_states_by_seq[batch_idx]
                dense_value_start_pos = dense_value_start_positions[batch_idx]
            single_output = turboquant4_value_only_attention(
                q[query_index : query_index + 1],
                key_cache_tensor,
                value_cache_tensor,
                value_cache_scales,
                key_block_ids_tensor[batch_idx],
                value_block_ids_tensor[batch_idx],
                scale_block_ids_tensor[batch_idx],
                int(seq_lens_cpu[batch_idx].item()),
                query_position,
                tokens_per_block,
                q_scaling,
                is_causal,
                attention_window_size,
                dense_value_states,
                dense_value_start_pos,
            )
            output[query_index : query_index + 1].copy_(single_output)
    return output


def turboquant4_attention(
    q: torch.Tensor,
    kv_cache_tensor: torch.Tensor,
    kv_cache_scales: torch.Tensor,
    block_ids: list[int] | torch.Tensor,
    seq_len: int,
    q_start_pos: int,
    tokens_per_block: int,
    q_scaling: float,
    is_causal: bool,
    attention_window_size: int | None,
) -> torch.Tensor:
    """Attention over a packed TurboQuant4 paged KV cache for one request.

    Args:
        q: Query tensor with shape [q_len, num_heads, head_dim].
        kv_cache_tensor: Packed cache, shape [num_blocks, 2, tokens, heads, head_dim // 2].
        kv_cache_scales: Scale cache, shape [num_blocks, 2, tokens, heads, 1].
        block_ids: Cache block ids for this request.
        seq_len: Total number of valid KV tokens in the cache.
        q_start_pos: Absolute token position of q[0].
        tokens_per_block: Number of tokens in each cache block.
        q_scaling: TRTLLM attention q scaling.
        is_causal: Whether to apply causal masking.
        attention_window_size: Optional sliding-window size.

    Returns:
        Dense attention output with shape [q_len, num_heads, head_dim].
    """
    if seq_len < 0:
        raise ValueError("TurboQuant4 attention seq_len must be non-negative.")
    if q_start_pos < 0:
        raise ValueError("TurboQuant4 attention q_start_pos must be non-negative.")
    if q_scaling <= 0:
        raise ValueError("TurboQuant4 attention q_scaling must be positive.")
    if attention_window_size is not None and attention_window_size < 0:
        raise ValueError("TurboQuant4 attention_window_size must be non-negative.")
    if q.ndim != 3:
        raise ValueError(f"TurboQuant4 attention expects [seq, heads, dim], got {tuple(q.shape)}")
    _validate_activation_dtype(q.dtype, "attention")
    _validate_cache_tensors(kv_cache_tensor, kv_cache_scales, 0, tokens_per_block)
    _validate_same_device(q, kv_cache_tensor, "cache")
    _validate_same_device(q, kv_cache_scales, "scale cache")
    if kv_cache_tensor.shape[1] < 2:
        raise ValueError("TurboQuant4 attention requires K and V cache entries.")
    if q.shape[-1] != kv_cache_tensor.shape[-1] * 2:
        raise ValueError("TurboQuant4 cache head_dim mismatch.")
    _validate_attention_head_counts(q.shape[1], kv_cache_tensor.shape[3])
    if q.shape[0] > 0 and q_start_pos + q.shape[0] > seq_len:
        raise ValueError(
            "TurboQuant4 attention query positions must be within the KV sequence length."
        )
    required_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block if seq_len > 0 else 0
    _validate_block_ids(block_ids, required_blocks, kv_cache_tensor.shape[0])

    block_ids_tensor = _block_ids_to_tensor(block_ids, q.device)
    window_size = attention_window_size or 0
    if q.is_cuda and _use_native_ops():
        try:
            return torch.ops.trtllm.turboquant4_attention(
                q.contiguous(),
                kv_cache_tensor,
                kv_cache_scales,
                block_ids_tensor.contiguous(),
                seq_len,
                q_start_pos,
                tokens_per_block,
                float(q_scaling),
                is_causal,
                window_size,
            )
        except AttributeError:
            pass

    key_states = turboquant4_dequantize_cache(
        kv_cache_tensor, kv_cache_scales, block_ids, 0, seq_len, tokens_per_block, q.dtype
    )
    value_states = turboquant4_dequantize_cache(
        kv_cache_tensor, kv_cache_scales, block_ids, 1, seq_len, tokens_per_block, q.dtype
    )
    q_states = q.unsqueeze(0).transpose(1, 2)
    key_states = key_states.transpose(1, 2).to(q.dtype)
    value_states = value_states.transpose(1, 2).to(q.dtype)
    kv_repeats = q.shape[1] // kv_cache_tensor.shape[3]
    if kv_repeats != 1:
        key_states = key_states[:, :, None, :, :].expand(
            1, kv_cache_tensor.shape[3], kv_repeats, seq_len, q.shape[-1]
        )
        key_states = key_states.reshape(1, q.shape[1], seq_len, q.shape[-1])
        value_states = value_states[:, :, None, :, :].expand(
            1, kv_cache_tensor.shape[3], kv_repeats, seq_len, q.shape[-1]
        )
        value_states = value_states.reshape(1, q.shape[1], seq_len, q.shape[-1])

    attn_mask = None
    if is_causal or window_size > 0:
        cache_position = torch.arange(q_start_pos, q_start_pos + q.shape[0], device=q.device)
        kv_positions = torch.arange(seq_len, device=q.device).unsqueeze(0)
        allowed = kv_positions <= cache_position.unsqueeze(-1)
        if window_size > 0:
            allowed = allowed & (kv_positions > cache_position.unsqueeze(-1) - window_size)
        attn_mask = allowed[None, None, :, :]

    scale = 1 / ((q.shape[-1] ** 0.5) * q_scaling)
    return (
        F.scaled_dot_product_attention(
            q_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=False,
            scale=scale,
        )
        .transpose(1, 2)
        .squeeze(0)
        .contiguous()
    )


def turboquant4_batch_attention(
    q: torch.Tensor,
    kv_cache_tensor: torch.Tensor,
    kv_cache_scales: torch.Tensor,
    block_ids: list[list[int]] | torch.Tensor,
    q_batch_indices: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    tokens_per_block: int,
    q_scaling: float,
    is_causal: bool,
    attention_window_size: int | None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Batched attention over packed TurboQuant4 paged KV cache.

    Args:
        q: Flattened query tensor with shape [total_q, num_heads, head_dim].
        block_ids: Padded block ids with shape [batch, max_blocks] or a list of
            per-request block-id lists.
        q_batch_indices: int tensor with shape [total_q] mapping each query
            token to its request index.
        query_positions: int tensor with shape [total_q] containing each query
            token's absolute cache position.
        seq_lens: int tensor with shape [batch] containing total KV lengths.
    """
    if q.ndim != 3:
        raise ValueError(
            f"TurboQuant4 batch attention expects [seq, heads, dim], got {tuple(q.shape)}"
        )
    _validate_activation_dtype(q.dtype, "batch attention")
    if q_scaling <= 0:
        raise ValueError("TurboQuant4 attention q_scaling must be positive.")
    if attention_window_size is not None and attention_window_size < 0:
        raise ValueError("TurboQuant4 attention_window_size must be non-negative.")
    _validate_cache_tensors(kv_cache_tensor, kv_cache_scales, 0, tokens_per_block)
    _validate_same_device(q, kv_cache_tensor, "cache")
    _validate_same_device(q, kv_cache_scales, "scale cache")
    if kv_cache_tensor.shape[1] < 2:
        raise ValueError("TurboQuant4 attention requires K and V cache entries.")
    if q.shape[-1] != kv_cache_tensor.shape[-1] * 2:
        raise ValueError("TurboQuant4 cache head_dim mismatch.")
    _validate_attention_head_counts(q.shape[1], kv_cache_tensor.shape[3])
    if q_batch_indices.ndim != 1 or query_positions.ndim != 1:
        raise ValueError("TurboQuant4 batch attention query metadata must be 1D.")
    if q_batch_indices.numel() != q.shape[0] or query_positions.numel() != q.shape[0]:
        raise ValueError("TurboQuant4 batch attention query metadata length mismatch.")
    if seq_lens.ndim != 1:
        raise ValueError("TurboQuant4 batch attention seq_lens must be 1D.")
    _validate_int32_tensor(q_batch_indices, "batch attention q_batch_indices")
    _validate_int32_tensor(query_positions, "batch attention query_positions")
    _validate_int32_tensor(seq_lens, "batch attention seq_lens")

    seq_lens_cpu = seq_lens.detach().cpu()
    if bool((seq_lens_cpu < 0).any().item()):
        raise ValueError("TurboQuant4 batch attention seq_lens must be non-negative.")
    actual_max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
    if max_seq_len is None:
        max_seq_len = actual_max_seq_len
    elif max_seq_len < 0:
        raise ValueError("TurboQuant4 batch attention max_seq_len must be non-negative.")
    elif max_seq_len < actual_max_seq_len:
        raise ValueError(
            "TurboQuant4 batch attention max_seq_len must be at least "
            f"the maximum sequence length; got {max_seq_len}, "
            f"max seq_len is {actual_max_seq_len}."
        )
    _validate_batch_block_ids(block_ids, seq_lens, tokens_per_block, kv_cache_tensor.shape[0])
    block_ids_tensor = _batch_block_ids_to_tensor(block_ids, q.device)
    if block_ids_tensor.shape[0] != seq_lens.numel():
        raise ValueError("TurboQuant4 batch attention block-id batch size mismatch.")
    required_blocks = (
        (max_seq_len + tokens_per_block - 1) // tokens_per_block if max_seq_len > 0 else 0
    )
    if block_ids_tensor.shape[1] < required_blocks:
        raise RuntimeError(
            "TurboQuant4 KV cache block list is shorter than the requested "
            f"batch sequence; need {required_blocks}, got {block_ids_tensor.shape[1]}."
        )
    _validate_batch_query_metadata(q_batch_indices, query_positions,
                                   seq_lens_cpu)

    q_batch_indices = q_batch_indices.to(device=q.device, non_blocking=True)
    query_positions = query_positions.to(device=q.device, non_blocking=True)
    seq_lens = seq_lens.to(device=q.device, non_blocking=True)
    window_size = attention_window_size or 0
    if q.is_cuda and _use_native_ops():
        try:
            return torch.ops.trtllm.turboquant4_batch_attention(
                q.contiguous(),
                kv_cache_tensor,
                kv_cache_scales,
                block_ids_tensor.contiguous(),
                q_batch_indices.contiguous(),
                query_positions.contiguous(),
                seq_lens.contiguous(),
                max_seq_len,
                tokens_per_block,
                float(q_scaling),
                is_causal,
                window_size,
            )
        except AttributeError:
            pass

    output = torch.zeros_like(q)
    q_batch_indices_cpu = q_batch_indices.cpu()
    query_positions_cpu = query_positions.cpu()
    seq_lens_cpu = seq_lens.cpu()
    for batch_idx in range(seq_lens_cpu.numel()):
        query_mask = q_batch_indices_cpu == batch_idx
        if not torch.any(query_mask):
            continue
        query_indices_cpu = torch.nonzero(query_mask, as_tuple=False).flatten()
        for query_index_cpu in query_indices_cpu:
            query_index = int(query_index_cpu.item())
            query_position = int(query_positions_cpu[query_index].item())
            single_output = turboquant4_attention(
                q[query_index : query_index + 1],
                kv_cache_tensor,
                kv_cache_scales,
                block_ids_tensor[batch_idx],
                int(seq_lens_cpu[batch_idx].item()),
                query_position,
                tokens_per_block,
                q_scaling,
                is_causal,
                attention_window_size,
            )
            output[query_index : query_index + 1].copy_(single_output)
    return output
