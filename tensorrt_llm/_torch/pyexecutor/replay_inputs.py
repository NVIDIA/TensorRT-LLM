# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch staging of a CUDA graph replay's input ids and positions."""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK = 512


@triton.jit
def _replay_inputs_kernel(
    ids_src,
    ids_dst,
    positions_src,
    positions_dst,
    offsets,
    hook_positions,
    num_tokens,
    HOOK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = token < num_tokens
    tl.store(ids_dst + token, tl.load(ids_src + token, mask=mask), mask=mask)
    position = tl.load(positions_src + token, mask=mask)
    tl.store(positions_dst + token, position, mask=mask)
    if HOOK:
        offset = tl.load(offsets + token, mask=mask)
        tl.store(hook_positions + token, position + offset, mask=mask)


def stage_replay_inputs(
    ids_dst: torch.Tensor,
    ids_src: torch.Tensor,
    positions_dst: torch.Tensor,
    positions_src: torch.Tensor,
    hook_positions: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> bool:
    """Copy the ids and positions, and ``positions_src + offsets`` into ``hook_positions``.

    Returns False, writing nothing, unless all operands are contiguous same-device CUDA
    integer tensors of one length, the ids sharing one dtype and the positions another.
    """
    positions = [positions_dst, positions_src]
    if hook_positions is not None:
        if offsets is None:
            return False
        positions += [hook_positions, offsets]
    num_tokens = ids_dst.numel()
    if not all(
        tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.dtype in (torch.int32, torch.int64)
        and tensor.device == ids_dst.device
        and tensor.numel() == num_tokens
        for tensor in (ids_dst, ids_src, *positions)
    ):
        return False
    if ids_src.dtype != ids_dst.dtype or any(
        tensor.dtype != positions_dst.dtype for tensor in positions
    ):
        return False
    if num_tokens:
        _replay_inputs_kernel[(triton.cdiv(num_tokens, _BLOCK),)](
            ids_src,
            ids_dst,
            positions_src,
            positions_dst,
            offsets if hook_positions is not None else positions_src,
            hook_positions if hook_positions is not None else positions_dst,
            num_tokens,
            HOOK=hook_positions is not None,
            BLOCK=_BLOCK,
        )
    return True
