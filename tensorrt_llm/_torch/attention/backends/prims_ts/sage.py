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

"""Sage attention inputs for the PrimTS decode kernels.

Sage attention runs ``QK^T`` on 8-bit Q/K with one dequantization scale per
token block and ``PV`` on E4M3 P/V with one scale per V channel. Q/K scales use
the trtllm-gen flat layout of TensorRT-LLM's ``sageQuant``: per head, sequence
``b`` starts at slot ``b * S // blk + b`` and token ``t`` uses slot
``t // blk`` inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch

from flashinfer.utils import ceil_div

SAGE_K_BLOCK_SIZES = (1, 4, 16, 32, 64, 128, 256)
# Sage scale slots of the contiguous attention adapter, in ABI order.
SAGE_ADAPTER_SLOTS = ("q_scale", "k_scale", "k_summary_scale", "v_scale", "v_mean")

_T = TypeVar("_T")


@dataclass(frozen=True)
class SageAttentionConfig:
    """Compile-time Sage attention recipe of one plan.

    ``q_block_size`` tokens share one Q scale and must be a power of two no
    larger than the planned Q tile. ``k_block_size`` tokens share one K scale;
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
    ``[Hq, flat_scale_numel(B, Sq, q_block_size)]``, ``k_scale`` is
    ``[Hkv, flat_scale_numel(B, Skv, k_block_size)]``, and ``v_scale`` and
    ``v_mean`` are ``[Hkv, D]``. ``k_summary_scale`` holds the scales of the
    K summaries of a proxy plan,
    ``[Hkv, flat_scale_numel(B, num_kv_blocks, k_summary_block_size)]``.
    ``v_mean`` and ``k_summary_scale`` are present exactly when the plan uses
    them. Scales must be positive and finite; the kernel does not check them.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_summary_scale: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None


def flat_scale_numel(batch_size: int, seq_len: int, block_size: int) -> int:
    """Return the per-head slot count of the flat scale layout."""
    return ceil_div(batch_size * seq_len, block_size) + batch_size - 1


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def log2_block_size(block_size: int) -> int:
    """Return ``log2`` of a power-of-two scale block size."""
    if not is_power_of_two(block_size):
        raise ValueError(f"block size must be a power of two, got {block_size}")
    return block_size.bit_length() - 1


def sage_adapter_slots(values: Mapping[str, _T | None]) -> tuple[_T | None, ...]:
    """Arrange Sage values named by scale into the adapter slots.

    An absent name leaves its slot ``None``, which the adapter binds as a null
    pointer.
    """
    return tuple(values.get(name) for name in SAGE_ADAPTER_SLOTS)


def flat_scale_slot(batch_idx, token_idx, seq_len, log2_block: int):
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
    expected_shape: tuple[int, int],
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
) -> dict[str, tuple[int, int]]:
    """Return the shape of every scale tensor a plan consumes, by field name.

    ``summary_seq_len`` is the KV block count of a proxy plan and ``None``
    otherwise.
    """
    shapes = {
        "q_scale": (
            num_qo_heads,
            flat_scale_numel(batch_size, seq_len_q, config.q_block_size),
        ),
        "k_scale": (
            num_kv_heads,
            flat_scale_numel(batch_size, seq_len_kv, config.k_block_size),
        ),
        "v_scale": (num_kv_heads, head_dim),
    }
    if summary_seq_len is not None:
        shapes["k_summary_scale"] = (
            num_kv_heads,
            flat_scale_numel(batch_size, summary_seq_len, config.k_summary_block_size),
        )
    if config.v_mean:
        shapes["v_mean"] = (num_kv_heads, head_dim)
    return shapes


def validate_sage_params(
    params: SageAttentionParams,
    expected_shapes: Mapping[str, tuple[int, int]],
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


__all__ = [
    "SAGE_ADAPTER_SLOTS",
    "SAGE_K_BLOCK_SIZES",
    "SageAttentionConfig",
    "SageAttentionParams",
    "flat_scale_numel",
    "flat_scale_slot",
    "is_power_of_two",
    "log2_block_size",
    "sage_adapter_slots",
    "sage_scale_shapes",
    "validate_sage_params",
]
