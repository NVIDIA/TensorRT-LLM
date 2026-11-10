# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sage attention recipe and per-run scale tensors of the PrimTS attention plans.

Sage attention runs ``QK^T`` on 8-bit Q/K with one dequantization scale per
token block and ``PV`` on E4M3 P/V with one scale per V channel. Q/K scales use
the trtllm-gen flat layout of TensorRT-LLM's ``sageQuant``: per head, sequence
``b`` starts at slot ``b * S // blk + b`` and token ``t`` uses slot
``t // blk`` inside it. This module holds the recipe, the scale tensors, their
flat layout and validation, and the INT8 Q/K device gate; the plans of both
the decode and the context kernels use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from flashinfer.utils import ceil_div

# Only SM100 has the INT8 tcgen05 MMA kind; later GPUs use E4M3 Q/K.
_INT8_QK_COMPUTE_CAPABILITIES = ((10, 0),)


def _validate_int8_qk_device(q_dtype: torch.dtype, device_index: int) -> None:
    """Reject Int8 Q/K on GPUs without the INT8 tcgen05 MMA."""

    if q_dtype != torch.int8:
        return
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in _INT8_QK_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "Int8 Q/K Sage attention requires an SM100a/B200 GPU, whose tcgen05 "
            "MMA supports INT8 operands; use the torch.float8_e4m3fn Q/K recipe "
            f"on device cuda:{device_index} with compute capability {capability}"
        )


SAGE_K_BLOCK_SIZES = (1, 4, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class SageAttentionConfig:
    """Compile-time Sage attention recipe of one plan.

    ``q_block_size`` tokens share one Q scale and must be a power of two; a
    block may span several Q tiles, since every Q row reads its own prepared
    scale. ``k_block_size`` tokens share one K scale;
    ``k_summary_block_size`` is the K block size of a proxy plan's summary
    scales; construction sets it to ``k_block_size`` when it is omitted.
    Both must be one of ``SAGE_K_BLOCK_SIZES``. ``v_mean`` says whether every
    run supplies a per-channel V mean that is added back to the output. The
    defaults are TensorRT-LLM's production recipe.
    """

    q_block_size: int = 1
    k_block_size: int = 16
    v_mean: bool = False
    k_summary_block_size: int | None = None

    def __post_init__(self) -> None:
        if self.k_summary_block_size is None:
            object.__setattr__(self, "k_summary_block_size", self.k_block_size)


@dataclass(frozen=True)
class SageAttentionParams:
    """Per-block Q/K scales and per-channel V scales of one run.

    All tensors are FP32 with the block sizes of the plan's
    :class:`SageAttentionConfig`: ``q_scale`` is
    ``[Hq, sage_flat_scale_numel(B, Sq, q_block_size)]``, ``k_scale`` is
    ``[Hkv, sage_flat_scale_numel(B, Skv, k_block_size)]``, and ``v_scale`` and
    ``v_mean`` are ``[B, Hkv, D]``, one per-channel row per batch entry and
    KV head. ``k_summary_scale`` holds the scales of the
    K summaries of a proxy plan,
    ``[Hkv, sage_flat_scale_numel(B, num_kv_blocks, k_summary_block_size)]``.
    ``v_mean`` and ``k_summary_scale`` are present exactly when the plan uses
    them. Scales must be positive and finite; the kernel does not check them.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_summary_scale: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None


def sage_flat_scale_numel(batch_size: int, seq_len: int, block_size: int) -> int:
    """Return the per-head slot count of the flat scale layout."""
    return ceil_div(batch_size * seq_len, block_size) + batch_size - 1


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def sage_log2_block_size(block_size: int) -> int:
    """Return ``log2`` of a power-of-two scale block size."""
    if not is_power_of_two(block_size):
        raise ValueError(f"block size must be a power of two, got {block_size}")
    return block_size.bit_length() - 1


def sage_flat_scale_slot(batch_idx, token_idx, seq_len, log2_block: int):
    """Return the per-head slot of one token in the flat scale layout.

    Sequence ``b`` starts at ``b * S // blk + b``; the extra ``b`` keeps its
    last block apart from the next sequence's first block when ``blk`` does
    not divide ``S``. The arithmetic serves host and device ``Int32`` values.
    """
    return ((batch_idx * seq_len) >> log2_block) + batch_idx + (token_idx >> log2_block)


def _validate_scale_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32, got {tensor.dtype}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


# Q/K and output dtypes of the Sage recipe.
_SAGE_QK_DTYPES = (torch.float8_e4m3fn, torch.int8)
_SAGE_OUTPUT_DTYPES = (torch.float16, torch.bfloat16)


def validate_sage_recipe(
    config: object,
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Reject a Sage recipe at plan time, the same way for every wrapper.

    The recipe quantizes Q and K to one 8-bit dtype, float8_e4m3fn or int8,
    and V to float8_e4m3fn, and returns a float16 or bfloat16 output: an
    8-bit output would discard the precision the scales keep. The block
    sizes follow ``SageAttentionConfig``. Int8 Q/K also need an SM100 GPU
    (``_validate_int8_qk_device``), and each wrapper rejects the plan
    geometries its kernel does not run.
    """
    if not isinstance(config, SageAttentionConfig):
        raise TypeError("sage must be a SageAttentionConfig")
    if q_dtype != k_dtype:
        raise ValueError(
            "Sage attention requires Q and K in one dtype; "
            f"got q {q_dtype} and k {k_dtype}"
        )
    if q_dtype not in _SAGE_QK_DTYPES:
        raise NotImplementedError(
            f"Sage attention requires float8_e4m3fn or int8 Q/K; got {q_dtype}"
        )
    if v_dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            f"Sage attention requires float8_e4m3fn V; got {v_dtype}"
        )
    if output_dtype not in _SAGE_OUTPUT_DTYPES:
        raise NotImplementedError(
            f"Sage attention requires a float16 or bfloat16 output; got {output_dtype}"
        )
    for name in ("k_block_size", "k_summary_block_size"):
        block_size = getattr(config, name)
        if block_size not in SAGE_K_BLOCK_SIZES:
            raise ValueError(
                f"sage.{name} must be one of {SAGE_K_BLOCK_SIZES}, got {block_size}"
            )
    if not is_power_of_two(config.q_block_size):
        raise ValueError(
            f"sage.q_block_size must be a power of two, got {config.q_block_size}"
        )


def sage_scale_shapes(
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    summary_seq_len: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of every scale tensor a plan consumes, by field name.

    ``summary_seq_len`` is the KV block count of a proxy plan and ``None``
    otherwise.
    """
    shapes = {
        "q_scale": (
            num_qo_heads,
            sage_flat_scale_numel(batch_size, seq_len_q, config.q_block_size),
        ),
        "k_scale": (
            num_kv_heads,
            sage_flat_scale_numel(batch_size, seq_len_kv, config.k_block_size),
        ),
        "v_scale": (batch_size, num_kv_heads, head_dim),
    }
    if summary_seq_len is not None:
        shapes["k_summary_scale"] = (
            num_kv_heads,
            sage_flat_scale_numel(
                batch_size, summary_seq_len, config.k_summary_block_size
            ),
        )
    if config.v_mean:
        shapes["v_mean"] = (batch_size, num_kv_heads, head_dim)
    return shapes


def validate_sage_params(
    params: SageAttentionParams,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    device: torch.device,
) -> None:
    """Validate the scale tensors of one run against the plan's expected shapes.

    ``expected_shapes`` is the plan's ``sage_scale_shapes`` result; an
    optional tensor is required exactly when its name is present.
    """
    if not isinstance(params, SageAttentionParams):
        raise TypeError("sage must be a SageAttentionParams instance")
    for name, consumer in (
        ("v_mean", "a plan configured with v_mean=True"),
        ("k_summary_scale", "block-sparse proxy routes"),
    ):
        if (getattr(params, name) is None) == (name in expected_shapes):
            raise ValueError(f"{name} is required by {consumer} and rejected otherwise")
    for name, expected_shape in expected_shapes.items():
        _validate_scale_tensor(
            getattr(params, name), name, expected_shape=expected_shape, device=device
        )
