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
"""MXFP8 fake quantization with an optional slot scatter: ``trtllm::mxfp8_fake_quantize``.

Per row of ``x`` and per 32-element block along its last dim::

    amax  = max(max|x_block|, 1e-4)                      # fp32
    e     = ceil(log2(fp32(amax * fp32(1/448))))          # from the IEEE bits
    scale = 2^e                                          # UE8M0 byte = e + 127
    q     = e4m3_rne(clamp(x / scale, -448, 448))

and the row is written to slot ``slot_mapping[row]`` of ``out`` (row ``row``
when no mapping is given; a negative or out-of-range slot is skipped), either
as the dequantized values ``bf16(q * scale)`` (a bf16 ``out``) or as the raw
e4m3 codes plus one UE8M0 byte per block (an fp8/uint8 ``out`` together with
``out_scale``). The dequantized values are exact in bf16, so both forms hold
the same value set.

``out`` is ``[slots, W]`` or paged ``[pages, page_size, W]`` with arbitrary
leading strides; slot ``s`` of a paged ``out`` is ``(s // page_size, s % page_size)``.
One Triton program per row of ``x``; no autotuning.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

MXFP8_BLOCK = 32
_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4

_NATIVE_DATA_DTYPES = (torch.float8_e4m3fn, torch.uint8)
_NATIVE_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
_SLOT_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _mxfp8_fake_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    slot_ptr,
    x_row_stride,
    slot_stride,
    out_page_stride,
    out_slot_stride,
    scale_page_stride,
    scale_slot_stride,
    page_size,
    num_slots,
    W: tl.constexpr,
    NUM_BLOCKS_P2: tl.constexpr,
    HAS_SLOT: tl.constexpr,
    NATIVE: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    E4M3_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_SLOT:
        slot = tl.load(slot_ptr + row * slot_stride).to(tl.int64)
    else:
        slot = row
    if (slot >= 0) & (slot < num_slots):
        page = slot // page_size
        in_page = slot % page_size

        blk = tl.arange(0, NUM_BLOCKS_P2)
        col = blk[:, None] * 32 + tl.arange(0, 32)[None, :]
        mask = col < W
        x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(x), axis=1), AMAX_FLOOR)
        # 2^ceil(log2(amax / 448)) from the IEEE bits of the fp32 product with the
        # fp32 reciprocal: the exponent, plus one when any mantissa bit is set.
        bits = (amax * (1.0 / E4M3_MAX)).to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
        # The scale is a power of two, so multiplying by its exact reciprocal is
        # the IEEE quotient x / scale.
        inv_scale = ((127 - e) << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(x * inv_scale[:, None], -E4M3_MAX, E4M3_MAX).to(tl.float8e4nv)

        out_off = page * out_page_stride + in_page * out_slot_stride + col
        if NATIVE:
            tl.store(out_ptr + out_off, q, mask=mask)
            scale_off = page * scale_page_stride + in_page * scale_slot_stride + blk
            tl.store(scale_ptr + scale_off, (e + 127).to(tl.uint8), mask=blk * 32 < W)
        else:
            y = (q.to(tl.float32) * scale[:, None]).to(tl.bfloat16)
            tl.store(out_ptr + out_off, y, mask=mask)


def _slot_geometry(t: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    """(page_size, num_slots, page_stride, slot_stride) of a [slots, C] or [pages, page_size, C] tensor."""
    if t.dim() == 2:
        return 1, t.shape[0], t.stride(0), 0
    if t.dim() == 3:
        return t.shape[1], t.shape[0] * t.shape[1], t.stride(0), t.stride(1)
    raise ValueError(
        f"{name} must be [slots, C] or [pages, page_size, C], got shape {tuple(t.shape)}"
    )


@torch.library.custom_op("trtllm::mxfp8_fake_quantize", mutates_args=("out", "out_scale"))
def mxfp8_fake_quantize(
    x: torch.Tensor,
    out: torch.Tensor,
    slot_mapping: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> None:
    """Fake-quantize each row of ``x`` to MXFP8 (E4M3, UE8M0 per 32, amax floor 1e-4) into ``out``.

    ``x``: bf16 ``[..., W]``, ``W % 32 == 0``, unit last-dim stride, leading dims
    viewable as one row dim. ``out``: bf16 (dequantized values) or
    float8_e4m3fn / uint8 (codes, with ``out_scale`` uint8 / float8_e8m0fnu
    ``[..., W // 32]`` of the same leading shape), ``[slots, W]`` or
    ``[pages, page_size, W]``. ``slot_mapping``: int32/int64 ``[rows]``; row
    ``r`` goes to slot ``slot_mapping[r]``, skipped when negative or
    ``>= slots``. Without it row ``r`` goes to slot ``r``.
    """
    if not x.is_cuda:
        raise ValueError(f"x must be a CUDA tensor, got device {x.device}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if x.dim() < 2:
        raise ValueError(f"x must have rank >= 2, got shape {tuple(x.shape)}")
    width = x.shape[-1]
    if width <= 0 or width % MXFP8_BLOCK != 0:
        raise ValueError(f"x's last dim must be a positive multiple of {MXFP8_BLOCK}, got {width}")
    if x.stride(-1) != 1:
        raise ValueError(f"x must have a unit stride along its last dim, got strides {x.stride()}")
    try:
        x2 = x.view(-1, width)
    except RuntimeError as err:
        raise ValueError(
            f"x's leading dims must be viewable as one row dim, got strides {x.stride()}"
        ) from err
    num_rows = x2.shape[0]

    if out.device != x.device:
        raise ValueError(f"out must be on {x.device}, got {out.device}")
    if out.shape[-1] != width or out.stride(-1) != 1:
        raise ValueError(
            f"out's last dim must be {width} with unit stride, got shape {tuple(out.shape)} strides {out.stride()}"
        )
    page_size, num_slots, out_page_stride, out_slot_stride = _slot_geometry(out, "out")

    native = out.dtype != torch.bfloat16
    if native:
        if out.dtype not in _NATIVE_DATA_DTYPES:
            raise ValueError(f"out must be bfloat16, float8_e4m3fn or uint8, got {out.dtype}")
        if out_scale is None:
            raise ValueError(
                "a float8_e4m3fn/uint8 out (native codes) needs out_scale for the UE8M0 bytes"
            )
        if out_scale.device != x.device:
            raise ValueError(f"out_scale must be on {x.device}, got {out_scale.device}")
        if out_scale.dtype not in _NATIVE_SCALE_DTYPES:
            raise ValueError(f"out_scale must be uint8 or float8_e8m0fnu, got {out_scale.dtype}")
        if (
            tuple(out_scale.shape) != tuple(out.shape[:-1]) + (width // MXFP8_BLOCK,)
            or out_scale.stride(-1) != 1
        ):
            raise ValueError(
                f"out_scale must be {tuple(out.shape[:-1]) + (width // MXFP8_BLOCK,)} with unit last-dim stride, "
                f"got shape {tuple(out_scale.shape)} strides {out_scale.stride()}"
            )
        _, _, scale_page_stride, scale_slot_stride = _slot_geometry(out_scale, "out_scale")
        out_arg = out.view(torch.float8_e4m3fn) if out.dtype == torch.uint8 else out
        scale_arg = out_scale.view(torch.uint8) if out_scale.dtype != torch.uint8 else out_scale
    else:
        if out_scale is not None:
            raise ValueError(
                "out_scale is only written with a float8_e4m3fn/uint8 out; a bfloat16 out takes None"
            )
        scale_page_stride = scale_slot_stride = 0
        out_arg = out
        scale_arg = (
            out  # never dereferenced: NATIVE is a constexpr that compiles the scale store out
        )

    if slot_mapping is None:
        if num_slots < num_rows:
            raise ValueError(
                f"without slot_mapping out needs >= {num_rows} slots (one per row), got {num_slots}"
            )
        slot_arg, slot_stride = x2, 0  # never dereferenced: HAS_SLOT is a constexpr
    else:
        if slot_mapping.device != x.device:
            raise ValueError(f"slot_mapping must be on {x.device}, got {slot_mapping.device}")
        if slot_mapping.dtype not in _SLOT_DTYPES:
            raise ValueError(f"slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
        if slot_mapping.dim() != 1 or slot_mapping.shape[0] != num_rows:
            raise ValueError(
                f"slot_mapping must be [{num_rows}] (one slot per row), got {tuple(slot_mapping.shape)}"
            )
        slot_arg, slot_stride = slot_mapping, slot_mapping.stride(0)

    if num_rows == 0:
        return
    num_blocks = width // MXFP8_BLOCK
    _mxfp8_fake_quantize_kernel[(num_rows,)](
        x2,
        out_arg,
        scale_arg,
        slot_arg,
        x2.stride(0),
        slot_stride,
        out_page_stride,
        out_slot_stride,
        scale_page_stride,
        scale_slot_stride,
        page_size,
        num_slots,
        W=width,
        NUM_BLOCKS_P2=triton.next_power_of_2(num_blocks),
        HAS_SLOT=slot_mapping is not None,
        NATIVE=native,
        AMAX_FLOOR=_AMAX_FLOOR,
        E4M3_MAX=_E4M3_MAX,
        num_warps=4,
    )


@mxfp8_fake_quantize.register_fake
def _mxfp8_fake_quantize_fake(
    x: torch.Tensor,
    out: torch.Tensor,
    slot_mapping: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> None:
    return None
