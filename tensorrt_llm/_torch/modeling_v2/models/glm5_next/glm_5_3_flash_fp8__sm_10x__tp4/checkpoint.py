# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reads this rank's share of checkpoint tensors and records what was read.

The engine hands ``load_weights`` a mapping of checkpoint keys to tensors or lazy
safetensors slices. ``Checkpoint.read`` cuts a lazy entry before it is materialized, so a
rank reads only its rows or columns of a split weight, and records every key it touches:
after loading, every key must have been read except the ones the target predicts it does
not load (the vision tower, and the MTP layer without speculative decoding).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch

#: An index into a checkpoint tensor: one slice per leading axis.
Cut = tuple[slice, ...]

#: The side of the checkpoint's square FP8 scale blocks (``weight_scale_inv``).
BLOCK = 128


class Checkpoint:
    def __init__(self, weights: Mapping[str, Any], device: torch.device) -> None:
        self.weights = weights
        self.device = device
        self.read_keys: set[str] = set()

    def read(
        self, key: str, cut: Optional[Cut] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """``key``'s tensor (``cut`` of it, read from the lazy entry), on the device.

        ``dtype`` converts after the read; ``None`` keeps the checkpoint's dtype (FP8
        payloads and FP32 block scales stay as stored).
        """
        if key not in self.weights:
            raise KeyError(f"checkpoint has no tensor {key!r}")
        entry = self.weights[key]
        if cut is not None:
            tensor = entry[cut]
        else:
            tensor = entry if torch.is_tensor(entry) else entry[...]
        self.read_keys.add(key)
        tensor = tensor.to(device=self.device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.contiguous()

    def check_coverage(self, skipped: Iterable[str]) -> None:
        """Every key was read or is predicted to be skipped; nothing predicted was read."""
        skipped = set(skipped)
        keys = set(self.weights.keys())
        unread = sorted(keys - self.read_keys - skipped)
        if unread:
            raise ValueError(f"{len(unread)} checkpoint tensors were not loaded, e.g. {unread[:5]}")
        both = sorted(self.read_keys & skipped)
        if both:
            raise ValueError(f"tensors predicted to be skipped were loaded: {both[:5]}")


def rows(start: int, end: int) -> Cut:
    """The cut of rows ``[start, end)``."""
    return (slice(start, end),)


def cols(start: int, end: int) -> Cut:
    """The cut of columns ``[start, end)`` of a 2-D tensor."""
    return (slice(None), slice(start, end))


def share(total: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    """This rank's contiguous ``[start, end)`` of ``total`` split evenly over ``tp_size``."""
    if total % tp_size:
        raise ValueError(f"{total} does not split evenly over {tp_size} ranks")
    size = total // tp_size
    return tp_rank * size, (tp_rank + 1) * size


def dequantize_block_fp8(
    weight: torch.Tensor, scale: torch.Tensor, block: int = BLOCK
) -> torch.Tensor:
    """BF16 ``weight`` from a 128 x 128 block-FP8 weight and its FP32 block scales."""
    n, k = weight.shape
    assert scale.shape == (-(-n // block), -(-k // block)), (
        f"block scales {tuple(scale.shape)} for a {n} x {k} weight"
    )
    full = scale.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (weight.float() * full).to(torch.bfloat16)
