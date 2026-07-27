# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blocked-scale swizzle helpers for the MegaMoE CuteDSL NVFP4 backend.

Ported byte-for-byte from the upstream runner_fc12.py helpers used by the
external mega_runner. The MegaMoE kernel ABI reads each per-slot weight
scale tensor as a 1-D atom-swizzled FP8 buffer. This module exposes:

* :func:`to_blocked`: pad a 2-D ``(rows, cols)`` raw FP8 scale tensor to
  ``(round_up(rows, SfPaddingBlock=128), round_up(cols, 4))`` and apply
  the 32x4x4 atom layout permutation, returning a flat 1-D byte view.
* :func:`from_blocked`: inverse of :func:`to_blocked`, used by tests to
  read kernel-written ``fc1_output_sf`` workspaces back into a raw 2-D
  view for byte-equivalence comparisons against the host reference.
* :func:`stack_byte_reinterpretable_tensors`: stack helper that works for
  FP8 dtypes (``torch.stack`` does not support FP8 on older torch
  releases). Used by the quantization method when stacking per-slot
  blocked scales into a single ``(num_local_slots, fc?_sf_flat_size)``
  parameter.

Constants are reused from :mod:`.megamoe_constants` so a future change to
``SfPaddingBlock`` propagates to host and kernel together.

NOTE: Validate ``to_blocked`` / ``from_blocked`` byte-equivalence against
the upstream runner_fc12.py helpers in a single-rank backend test
before relying on these helpers in production. The current copy is a
verbatim port; see test_moe_backend.py for the regression check.
"""

from __future__ import annotations

from typing import List

import torch

from .megamoe_constants import SfPaddingBlock

__all__ = [
    "SfPaddingBlock",
    "to_blocked",
    "from_blocked",
    "stack_byte_reinterpretable_tensors",
    "cat_byte_reinterpretable_tensors",
    "ceil_div",
]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """Pad and apply the 32x4x4 scale swizzle to one raw scale tensor.

    Input  : ``(rows, cols)`` FP8 (``torch.float8_e4m3fn``) tensor.
    Output : 1-D FP8 tensor of length
             ``round_up(rows, SfPaddingBlock) * round_up(cols, 4)``.

    Empty tensors are allowed and short-circuit to a length-0 view of
    the same dtype/device.
    """
    if scale_2d.dim() != 2:
        raise ValueError(f"Expected 2D scale tensor, got {scale_2d.dim()}D.")
    rows, cols = scale_2d.shape
    if rows == 0 or cols == 0:
        return scale_2d.new_empty((0, ))

    row_blocks = ceil_div(rows, SfPaddingBlock)
    col_blocks = ceil_div(cols, 4)
    padded_rows = row_blocks * SfPaddingBlock
    padded_cols = col_blocks * 4

    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols),
                             dtype=scale_2d.dtype,
                             device=scale_2d.device)
        padded[:rows, :cols] = scale_2d

    blocks = padded.view(row_blocks, SfPaddingBlock, col_blocks,
                         4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1,
                                                        2).reshape(-1, 32, 16)
    return rearranged.flatten()


def from_blocked(flat: torch.Tensor, raw_rows: int,
                 raw_cols: int) -> torch.Tensor:
    """Inverse of :func:`to_blocked`: un-swizzle a flat 1-D FP8 atom buffer.

    Used by tests to read back the kernel-written ``fc1_output_sf``
    workspace bytes (atom-swizzled by the kernel via the same 32x4x4
    layout convention as :func:`to_blocked` produces on the host) into a
    raw ``(raw_rows, raw_cols)`` FP8 view comparable to the reference's
    per-expert raw SF.

    The trailing pad rows / cols (forward-padded by :func:`to_blocked` to
    multiples of ``SfPaddingBlock`` / 4) are dropped before return.
    """
    if flat.dim() != 1:
        raise ValueError(f"Expected 1D flat tensor, got {flat.dim()}D.")
    if raw_rows == 0 or raw_cols == 0:
        return flat.new_empty((raw_rows, raw_cols))

    row_blocks = ceil_div(raw_rows, SfPaddingBlock)
    col_blocks = ceil_div(raw_cols, 4)
    padded_rows = row_blocks * SfPaddingBlock
    padded_cols = col_blocks * 4
    expected = padded_rows * padded_cols
    if flat.numel() != expected:
        raise ValueError(
            f"from_blocked: flat size {flat.numel()} != expected "
            f"row_blocks*col_blocks*128*4 = {expected} for raw "
            f"({raw_rows}, {raw_cols}) padded to ({padded_rows}, {padded_cols})."
        )

    # Reverse to_blocked's atom-pack chain. Each atom (32, 16) was built
    # from a (128, 4) block via:
    #   (128, 4) -> view(4, 32, 4) -> transpose(0, 1) -> reshape(32, 16)
    # Reverse: reshape(32, 4, 4) -> transpose(0, 1) -> reshape(128, 4)
    rearranged = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4)
    blocks = rearranged.transpose(1, 2).reshape(-1, SfPaddingBlock, 4)
    blocks = blocks.reshape(row_blocks, col_blocks, SfPaddingBlock, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return padded[:raw_rows, :raw_cols].contiguous()


def cat_byte_reinterpretable_tensors(tensors: List[torch.Tensor],
                                     dim: int = 0) -> torch.Tensor:
    """Concatenate byte-backed float tensors via uint8 view.

    Works around the lack of ``torch.cat`` support for FP8 dtypes on
    some torch releases by reinterpreting as uint8 first.
    """
    if not tensors:
        raise ValueError("Expected at least one tensor to concatenate.")
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        concatenated = torch.cat([t.view(torch.uint8) for t in tensors],
                                 dim=dim)
        return concatenated.view(first.dtype)
    return torch.cat(tensors, dim=dim)


def stack_byte_reinterpretable_tensors(tensors: List[torch.Tensor],
                                       dim: int = 0) -> torch.Tensor:
    """Stack byte-backed float tensors via uint8 view.

    Works around the lack of ``torch.stack`` support for FP8 dtypes on
    some torch releases by reinterpreting as uint8 first.
    """
    if not tensors:
        raise ValueError("Expected at least one tensor to stack.")
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        stacked = torch.stack([t.view(torch.uint8) for t in tensors], dim=dim)
        return stacked.view(first.dtype)
    return torch.stack(tensors, dim=dim)
