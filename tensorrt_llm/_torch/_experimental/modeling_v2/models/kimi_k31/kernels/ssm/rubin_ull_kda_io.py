# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy-only checkpoint IO between MR8 verify pages and its native q1 layout.

Each active request owns a distinct physical checkpoint page. Slot indices are
256-byte units, not dense state rows: writing reserved index zero through an
indexed PyTorch copy could overwrite an overlapping live state. Both kernels
mask slot<=0 before any cache load/store. State and conv use their own strides.
No arithmetic, quantization or dtype conversion is applied to cache values.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_copy(
    state,
    conv,
    state_tmp,
    conv_tmp,
    slots,
    accepted,
    STATE_ELEMS: tl.constexpr,
    CONV_CHANNELS: tl.constexpr,
    state_stride: tl.constexpr,
    conv_stride: tl.constexpr,
    conv_channel_stride: tl.constexpr,
    conv_column_stride: tl.constexpr,
    slots_stride: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    destination = tl.load(slots + row * slots_stride).to(tl.int64)
    active = destination > 0
    position = tl.maximum(tl.load(accepted + row).to(tl.int64) - 1, 0)
    source = tl.load(slots + row * slots_stride + position, mask=active, other=0).to(tl.int64)
    offset = block * BLOCK + tl.arange(0, BLOCK)
    state_mask = (offset < STATE_ELEMS) & active
    if SCATTER:
        value = tl.load(state_tmp + (row + 1) * STATE_ELEMS + offset, mask=state_mask, other=0)
        tl.store(state + destination * state_stride + offset, value, mask=state_mask)
    else:
        value = tl.load(state + source * state_stride + offset, mask=state_mask, other=0)
        tl.store(state_tmp + (row + 1) * STATE_ELEMS + offset, value, mask=offset < STATE_ELEMS)
    conv_mask = (offset < CONV_CHANNELS * 3) & active
    channel, column = offset // 3, offset % 3
    if SCATTER:
        value = tl.load(conv_tmp + (row + 1) * CONV_CHANNELS * 3 + offset, mask=conv_mask, other=0)
        target = (
            destination * conv_stride + channel * conv_channel_stride + column * conv_column_stride
        )
        tl.store(conv + target, value, mask=conv_mask)
    else:
        source_offset = (
            destination * conv_stride
            + channel * conv_channel_stride
            + (position + column) * conv_column_stride
        )
        value = tl.load(conv + source_offset, mask=conv_mask, other=0)
        tl.store(
            conv_tmp + (row + 1) * CONV_CHANNELS * 3 + offset,
            value,
            mask=offset < CONV_CHANNELS * 3,
        )


def copy_decode_checkpoint(
    caches: tuple[torch.Tensor, torch.Tensor],
    scratch: tuple[torch.Tensor, torch.Tensor],
    slots: torch.Tensor,
    accepted: torch.Tensor,
    *,
    scatter: bool,
) -> None:
    """Gather selected history or publish q1's result at verify checkpoint zero.

    ``slots`` is int32 [requests,qcap], ``accepted`` int32 [>=requests] including
    the target token. Scratch row zero is padding; request r owns row r+1.
    After a q1 step, the cache manager must record zero accepted drafts, as for
    its normal plain-decode path. All layer copies complete before that reset.
    """
    conv, state = caches
    conv_tmp, state_tmp = scratch
    rows = slots.shape[0]
    if slots.dtype != torch.int32 or slots.ndim != 2 or accepted.dtype != torch.int32:
        raise TypeError("Checkpoint copy requires int32 position indices and accepted counts")
    if state.dtype != torch.float32 or state_tmp.dtype != torch.float32:
        raise TypeError("KDA state copy must preserve FP32")
    if conv.dtype != torch.bfloat16 or conv_tmp.dtype != torch.bfloat16:
        raise TypeError("KDA raw convolution copy must preserve BF16")
    if state.stride()[1:] != (128 * 128, 128, 1) or state_tmp.shape[1:] != state.shape[1:]:
        raise ValueError("KDA state views require contiguous [heads,128,128] payloads")
    if conv_tmp.shape != (rows + 1, conv.shape[1], 3) or not conv_tmp.is_contiguous():
        raise ValueError("q1 convolution scratch requires [requests+1,channels,3]")
    if state_tmp.shape[0] != rows + 1 or not state_tmp.is_contiguous():
        raise ValueError("q1 state scratch requires one dense row per request plus padding")
    if (
        slots.stride(1) != 1
        or accepted.ndim != 1
        or accepted.numel() < rows
        or accepted.stride(0) != 1
    ):
        raise ValueError("KDA copy metadata must contain every request and position")
    if any(t.device != state.device for t in (conv, conv_tmp, state_tmp, slots, accepted)):
        raise ValueError("KDA copy tensors must share one device")
    if not scatter:
        valid = (accepted[:rows] >= 1) & (accepted[:rows] <= slots.shape[1])
        torch._assert_async(
            torch.all((slots[:, 0] <= 0) | valid), "KDA accepted position outside checkpoint bank"
        )
    elements = state.shape[1] * 128 * 128
    _decode_copy[(rows, triton.cdiv(max(elements, conv.shape[1] * 3), 256))](
        state,
        conv,
        state_tmp,
        conv_tmp,
        slots,
        accepted,
        elements,
        conv.shape[1],
        state.stride(0),
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        slots.stride(0),
        scatter,
        256,
    )
