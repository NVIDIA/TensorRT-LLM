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

import torch

E8M0_DTYPE_NAME = "torch.float8_e8m0fnu"
E8M0_EXPONENT_BIAS = 127
E8M0_MAX_FINITE_BYTE = 254
E8M0_NAN_BYTE = 255
FP8_E4M3_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = torch.finfo(FP8_E4M3_DTYPE).max
FP8_E4M3_MIN = torch.finfo(FP8_E4M3_DTYPE).min

__all__ = [
    "e8m0_to_fp32",
    "e8m0_to_uint8",
    "e8m0_uint8_to_fp32",
    "fp32_to_e8m0",
    "fp8_block_dequant_ref",
    "fp8_block_quant_ref",
    "fp8_block_scale_shape",
    "maybe_e8m0_to_fp32",
]


def _get_e8m0_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e8m0fnu", None)


def _require_e8m0_dtype() -> torch.dtype:
    dtype = _get_e8m0_dtype()
    if dtype is None:
        raise RuntimeError(
            f"{E8M0_DTYPE_NAME} is not available in this PyTorch build; "
            "E8M0 scale tensors cannot be reinterpreted or decoded."
        )
    return dtype


def _scale_dtype() -> torch.dtype:
    return _get_e8m0_dtype() or torch.float32


def _check_e8m0_tensor(scale: torch.Tensor) -> None:
    dtype = _require_e8m0_dtype()
    if scale.dtype != dtype:
        raise TypeError(f"expected an {E8M0_DTYPE_NAME} tensor, got {scale.dtype}")


def _check_block_quant_input(x: torch.Tensor) -> None:
    if x.dim() == 0:
        raise ValueError("x must have at least one dimension")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")


def _check_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


def _decode_e8m0_like_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return e8m0_uint8_to_fp32(scale)
    return maybe_e8m0_to_fp32(scale)


def _validate_fp32_scale_values(scale_fp32: torch.Tensor) -> None:
    if scale_fp32.device.type != "cpu":
        return
    if not bool(torch.all(torch.isfinite(scale_fp32))):
        raise ValueError("scale must contain only finite values")
    if bool(torch.any(scale_fp32 < 0)):
        raise ValueError("scale must be non-negative")


def e8m0_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Return raw E8M0 exponent bytes without numeric conversion.

    Args:
        scale: E8M0 scale tensor of any shape.

    Returns:
        A ``torch.uint8`` view over the same raw bytes as ``scale``.

    Raises:
        RuntimeError: If this PyTorch build does not expose E8M0 tensors.
        TypeError: If ``scale`` is not an E8M0 tensor.
    """
    _check_e8m0_tensor(scale)
    return scale.view(torch.uint8)


def e8m0_uint8_to_fp32(exp_bits: torch.Tensor) -> torch.Tensor:
    """Decode raw E8M0 exponent bytes to FP32 scale values.

    Args:
        exp_bits: ``torch.uint8`` tensor with raw E8M0 exponent bytes.

    Returns:
        A ``torch.float32`` tensor where bytes ``0`` through ``254`` decode to
        ``2 ** (byte - 127)`` and byte ``255`` decodes to ``NaN``.

    Raises:
        TypeError: If ``exp_bits`` is not a ``torch.uint8`` tensor.
    """
    if exp_bits.dtype != torch.uint8:
        raise TypeError(f"exp_bits must have dtype torch.uint8, got {exp_bits.dtype}")

    fp32_bits = exp_bits.to(torch.int32) << 23
    tiny_bits = torch.full_like(fp32_bits, 1 << 22)
    nan_bits = torch.full_like(fp32_bits, 0x7FC00000)
    fp32_bits = torch.where(exp_bits == 0, tiny_bits, fp32_bits)
    fp32_bits = torch.where(exp_bits == E8M0_NAN_BYTE, nan_bits, fp32_bits)
    return fp32_bits.contiguous().view(torch.float32)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent-only scale values to FP32 powers of two.

    Args:
        scale: E8M0 scale tensor of any shape.

    Returns:
        A ``torch.float32`` tensor with each E8M0 byte decoded as an IEEE-754
        FP32 exponent field.

    Raises:
        RuntimeError: If this PyTorch build does not expose E8M0 tensors.
        TypeError: If ``scale`` is not an E8M0 tensor.
    """
    return e8m0_uint8_to_fp32(e8m0_to_uint8(scale))


def maybe_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scales; return non-E8M0 scales as FP32.

    Args:
        scale: E8M0, BF16, FP16, FP32, or other numeric scale tensor of any shape.

    Returns:
        ``scale`` decoded from E8M0 when applicable, otherwise ``scale`` converted
        numerically to ``torch.float32``.
    """
    dtype = _get_e8m0_dtype()
    if dtype is not None and scale.dtype == dtype:
        return e8m0_to_fp32(scale)
    return scale.to(torch.float32)


def fp32_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Quantize non-negative FP32 scales to E8M0 powers of two.

    The quantization rounds each positive scale up to the next power of two so
    the decoded scale is never smaller than the input scale. Byte ``255`` is
    reserved for ``NaN`` and is not emitted.

    Args:
        scale: Floating-point tensor of non-negative finite scale values.

    Returns:
        An E8M0 tensor when this PyTorch build exposes ``torch.float8_e8m0fnu``;
        otherwise a ``torch.float32`` tensor containing the same decoded
        powers-of-two values.

    Raises:
        TypeError: If ``scale`` is not floating point.
        ValueError: If a CPU ``scale`` tensor contains negative or non-finite
            values. CUDA graph paths avoid data-dependent host checks.
    """
    if not scale.is_floating_point():
        raise TypeError(f"scale must be floating point, got {scale.dtype}")

    scale_fp32 = scale.to(torch.float32)
    _validate_fp32_scale_values(scale_fp32)

    min_scale = torch.full((), 2.0**-E8M0_EXPONENT_BIAS, dtype=torch.float32, device=scale.device)
    safe_scale = torch.maximum(scale_fp32, min_scale)
    exponent = torch.ceil(torch.log2(safe_scale))
    exp_bits = torch.clamp(
        exponent + E8M0_EXPONENT_BIAS,
        min=0,
        max=E8M0_MAX_FINITE_BYTE,
    ).to(torch.uint8)

    dtype = _get_e8m0_dtype()
    if dtype is not None:
        return exp_bits.contiguous().view(dtype)
    return e8m0_uint8_to_fp32(exp_bits)


def fp8_block_scale_shape(input_shape: tuple[int, ...], block_size: int = 128) -> tuple[int, ...]:
    """Return the per-last-dimension block scale shape for an input tensor shape."""
    _check_block_size(block_size)
    if len(input_shape) == 0:
        raise ValueError("input_shape must have at least one dimension")

    last_dim = input_shape[-1]
    if last_dim < 0:
        raise ValueError(f"last dimension must be non-negative, got {last_dim}")
    num_blocks = (last_dim + block_size - 1) // block_size
    return (*input_shape[:-1], num_blocks)


def fp8_block_quant_ref(
    x: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference E4M3 quantization with one E8M0 scale per last-dim block.

    Args:
        x: Floating-point tensor of shape ``[..., dim]``.
        block_size: Number of contiguous last-dimension elements covered by one
            scale. Partial tail blocks are padded with zeros only for amax
            calculation.

    Returns:
        ``(x_fp8, scale)`` where ``x_fp8`` has shape ``[..., dim]`` and dtype
        ``torch.float8_e4m3fn``. ``scale`` has shape
        ``[..., ceil(dim / block_size)]`` and uses ``torch.float8_e8m0fnu`` when
        available, otherwise decoded ``torch.float32`` powers of two.
    """
    _check_block_quant_input(x)
    _check_block_size(block_size)

    last_dim = x.shape[-1]
    scale_shape = fp8_block_scale_shape(tuple(x.shape), block_size)
    if last_dim == 0:
        scale = torch.empty(scale_shape, dtype=_scale_dtype(), device=x.device)
        return x.to(FP8_E4M3_DTYPE).contiguous(), scale

    num_blocks = scale_shape[-1]
    pad = num_blocks * block_size - last_dim
    x_fp32 = x.to(torch.float32)
    if pad:
        x_fp32 = torch.nn.functional.pad(x_fp32, (0, pad))

    blocked = x_fp32.reshape(*x.shape[:-1], num_blocks, block_size)
    amax = blocked.abs().amax(dim=-1)
    scale_fp32 = torch.where(
        amax > 0,
        amax / FP8_E4M3_MAX,
        torch.ones((), dtype=torch.float32, device=x.device),
    )
    scale = fp32_to_e8m0(scale_fp32)
    scale_decoded = _decode_e8m0_like_scale(scale).to(device=x.device, dtype=torch.float32)
    quant = (blocked / scale_decoded.unsqueeze(-1)).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    quant = quant.to(FP8_E4M3_DTYPE).reshape(*x.shape[:-1], num_blocks * block_size)
    return quant[..., :last_dim].contiguous(), scale.contiguous()


def fp8_block_dequant_ref(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensors produced by :func:`fp8_block_quant_ref`."""
    _check_block_size(block_size)
    if x_fp8.dim() == 0:
        raise ValueError("x_fp8 must have at least one dimension")
    if x_fp8.dtype != FP8_E4M3_DTYPE:
        raise TypeError(f"x_fp8 must have dtype {FP8_E4M3_DTYPE}, got {x_fp8.dtype}")

    expected_scale_shape = fp8_block_scale_shape(tuple(x_fp8.shape), block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} must match expected shape {expected_scale_shape} "
            f"for x_fp8 shape {tuple(x_fp8.shape)} and block_size {block_size}"
        )

    scale_expanded = _decode_e8m0_like_scale(scale).repeat_interleave(block_size, dim=-1)
    scale_expanded = scale_expanded[..., : x_fp8.shape[-1]]
    return (x_fp8.to(torch.float32) * scale_expanded.to(torch.float32)).to(dtype).contiguous()
