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
"""MXFP8 row gather + transcode to per-tensor-scaled E4M3: ``trtllm::mxfp8_to_e4m3``.

The source is a pool of native MXFP8 rows: ``codes`` holds E4M3 codes
``[slots, W]`` (or paged ``[pages, page_size, W]``) and ``scales`` one UE8M0
byte per 32-element block ``[..., W // 32]`` with the same slot geometry.
For output row ``r``::

    s          = src_slots[r]            (r when src_slots is None)
    d          = dst_slots[r]            (r when dst_slots is None)
    v          = e4m3(codes[s, i]) * 2^(scales[s, i // 32] - 127)      (exact in fp32)
    out[d, i]  = e4m3_rne(clamp(v * inv_scale, -448, 448))

``inv_scale`` is one fp32 factor for the whole output (a per-tensor FP8
scale ``1 / inv_scale``). With ``inv_scale`` a power of two, ``v * inv_scale``
is exact, and the transcode is exact for every value whose magnitude lands in
E4M3's normal range ``[2^-6, 448]``: an MXFP8 value is an E4M3 code times a
power of two, so it keeps its 3-bit mantissa. Values below ``2^-6`` round to
E4M3 subnormals; values above 448 saturate.

A source slot that is negative or ``>= slots`` yields a row of zeros; a
destination slot that is negative or ``>= out slots`` is skipped. Slot ``s``
of a paged tensor is ``(s // page_size, s % page_size)``; leading strides are
free, so interleaved ``[.., 528]`` byte rows are passed as two strided views.
One Triton program per output row; no autotuning.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

MXFP8_BLOCK = 32
_E4M3_MAX = 448.0

_CODE_DTYPES = (torch.float8_e4m3fn, torch.uint8)
_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
_OUT_DTYPES = (torch.float8_e4m3fn,)
_SLOT_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _mxfp8_to_e4m3_kernel(
    codes_ptr,
    scales_ptr,
    out_ptr,
    src_ptr,
    dst_ptr,
    inv_scale,
    src_stride,
    dst_stride,
    codes_page_stride,
    codes_slot_stride,
    scales_page_stride,
    scales_slot_stride,
    src_page_size,
    src_num_slots,
    out_page_stride,
    out_slot_stride,
    out_page_size,
    out_num_slots,
    W: tl.constexpr,
    NUM_BLOCKS_P2: tl.constexpr,
    HAS_SRC: tl.constexpr,
    HAS_DST: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_DST:
        dst = tl.load(dst_ptr + row * dst_stride).to(tl.int64)
    else:
        dst = row
    if (dst >= 0) & (dst < out_num_slots):
        if HAS_SRC:
            src = tl.load(src_ptr + row * src_stride).to(tl.int64)
        else:
            src = row
        valid = (src >= 0) & (src < src_num_slots)
        src = tl.where(valid, src, 0)

        blk = tl.arange(0, NUM_BLOCKS_P2)
        col = blk[:, None] * 32 + tl.arange(0, 32)[None, :]
        mask = (col < W) & valid
        s_page = src // src_page_size
        s_in = src % src_page_size

        c = tl.load(
            codes_ptr + s_page * codes_page_stride + s_in * codes_slot_stride + col,
            mask=mask,
            other=0,
        )
        c = c.to(tl.float8e4nv, bitcast=True)
        b = tl.load(
            scales_ptr + s_page * scales_page_stride + s_in * scales_slot_stride + blk,
            mask=(blk * 32 < W) & valid,
            other=127,
        ).to(tl.int32)
        # UE8M0 byte b is 2^(b - 127): the fp32 exponent field itself for b in
        # 1..254, the fp32 subnormal 2^-127 for b = 0, NaN for b = 255.
        bits = tl.where(b == 0, 0x00400000, tl.where(b == 255, 0x7FC00000, b << 23))
        scale = bits.to(tl.float32, bitcast=True)
        v = c.to(tl.float32) * scale[:, None] * inv_scale
        v = tl.where(v != v, v, tl.minimum(tl.maximum(v, -448.0), 448.0))  # NaN stays NaN
        y = v.to(tl.float8e4nv)

        d_page = dst // out_page_size
        d_in = dst % out_page_size
        tl.store(
            out_ptr + d_page * out_page_stride + d_in * out_slot_stride + col,
            y.to(tl.uint8, bitcast=True),
            mask=col < W,
        )


def _slot_geometry(t: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    """(page_size, num_slots, page_stride, slot_stride) of a [slots, C] or [pages, page_size, C] tensor."""
    if t.dim() == 2:
        return 1, t.shape[0], t.stride(0), 0
    if t.dim() == 3:
        return t.shape[1], t.shape[0] * t.shape[1], t.stride(0), t.stride(1)
    raise ValueError(
        f"{name} must be [slots, C] or [pages, page_size, C], got shape {tuple(t.shape)}"
    )


def _check_slots(slots: torch.Tensor, name: str, device: torch.device) -> None:
    if slots.device != device:
        raise ValueError(f"{name} must be on {device}, got {slots.device}")
    if slots.dtype not in _SLOT_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {slots.dtype}")
    if slots.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(slots.shape)}")


@torch.library.custom_op("trtllm::mxfp8_to_e4m3", mutates_args=("out",))
def mxfp8_to_e4m3(
    codes: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    inv_scale: float,
    src_slots: Optional[torch.Tensor] = None,
    dst_slots: Optional[torch.Tensor] = None,
) -> None:
    """Gather MXFP8 rows (E4M3 codes x UE8M0 per 32) from a pool and write them to ``out`` as E4M3
    with one per-tensor scale: ``out = e4m3_rne(clamp(value * inv_scale, +-448))``.

    ``codes``: float8_e4m3fn / uint8 ``[slots, W]`` or ``[pages, page_size, W]``,
    ``W % 32 == 0``, unit last-dim stride. ``scales``: uint8 / float8_e8m0fnu
    ``codes.shape[:-1] + (W // 32,)``, unit last-dim stride. ``out``: float8_e4m3fn
    ``[out_slots, W]`` or ``[pages, page_size, W]``, unit last-dim stride.
    ``src_slots`` / ``dst_slots``: int32/int64 ``[rows]``. Row ``r`` reads slot
    ``src_slots[r]`` (``r`` when None; zeros when negative or ``>= slots``) and
    writes slot ``dst_slots[r]`` of ``out`` (``r`` when None; skipped when
    negative or ``>= out_slots``).
    """
    if not codes.is_cuda:
        raise ValueError(f"codes must be a CUDA tensor, got device {codes.device}")
    device = codes.device
    if codes.dtype not in _CODE_DTYPES:
        raise ValueError(f"codes must be float8_e4m3fn or uint8, got {codes.dtype}")
    width = codes.shape[-1] if codes.dim() > 0 else 0
    if width <= 0 or width % MXFP8_BLOCK != 0:
        raise ValueError(
            f"codes' last dim must be a positive multiple of {MXFP8_BLOCK}, got {width}"
        )
    if codes.stride(-1) != 1:
        raise ValueError(
            f"codes must have a unit stride along its last dim, got strides {codes.stride()}"
        )
    src_page_size, src_num_slots, codes_page_stride, codes_slot_stride = _slot_geometry(
        codes, "codes"
    )

    if scales.device != device:
        raise ValueError(f"scales must be on {device}, got {scales.device}")
    if scales.dtype not in _SCALE_DTYPES:
        raise ValueError(f"scales must be uint8 or float8_e8m0fnu, got {scales.dtype}")
    expected_scales = tuple(codes.shape[:-1]) + (width // MXFP8_BLOCK,)
    if tuple(scales.shape) != expected_scales or scales.stride(-1) != 1:
        raise ValueError(
            f"scales must be {expected_scales} with unit last-dim stride, "
            f"got shape {tuple(scales.shape)} strides {scales.stride()}"
        )
    _, _, scales_page_stride, scales_slot_stride = _slot_geometry(scales, "scales")

    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if out.dtype not in _OUT_DTYPES:
        raise ValueError(f"out must be float8_e4m3fn, got {out.dtype}")
    if out.dim() == 0 or out.shape[-1] != width or out.stride(-1) != 1:
        raise ValueError(
            f"out's last dim must be {width} with unit stride, got shape {tuple(out.shape)} strides {out.stride()}"
        )
    out_page_size, out_num_slots, out_page_stride, out_slot_stride = _slot_geometry(out, "out")
    if not (inv_scale > 0.0):
        raise ValueError(f"inv_scale must be a positive float, got {inv_scale}")

    if src_slots is not None:
        _check_slots(src_slots, "src_slots", device)
    if dst_slots is not None:
        _check_slots(dst_slots, "dst_slots", device)
    if src_slots is not None and dst_slots is not None and src_slots.shape[0] != dst_slots.shape[0]:
        raise ValueError(
            f"src_slots and dst_slots must have the same length, got {src_slots.shape[0]} and {dst_slots.shape[0]}"
        )
    if src_slots is not None:
        num_rows = src_slots.shape[0]
    elif dst_slots is not None:
        num_rows = dst_slots.shape[0]
    else:
        num_rows = src_num_slots
    if dst_slots is None and out_num_slots < num_rows:
        raise ValueError(
            f"without dst_slots out needs >= {num_rows} slots (one per row), got {out_num_slots}"
        )

    if num_rows == 0:
        return
    codes_arg = codes.view(torch.uint8) if codes.dtype != torch.uint8 else codes
    scales_arg = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
    out_arg = out.view(torch.uint8)
    # Never dereferenced when the matching HAS_* constexpr is False.
    src_arg, src_stride = (
        (src_slots, src_slots.stride(0)) if src_slots is not None else (scales_arg, 0)
    )
    dst_arg, dst_stride = (
        (dst_slots, dst_slots.stride(0)) if dst_slots is not None else (scales_arg, 0)
    )
    _mxfp8_to_e4m3_kernel[(num_rows,)](
        codes_arg,
        scales_arg,
        out_arg,
        src_arg,
        dst_arg,
        float(inv_scale),
        src_stride,
        dst_stride,
        codes_page_stride,
        codes_slot_stride,
        scales_page_stride,
        scales_slot_stride,
        src_page_size,
        src_num_slots,
        out_page_stride,
        out_slot_stride,
        out_page_size,
        out_num_slots,
        W=width,
        NUM_BLOCKS_P2=triton.next_power_of_2(width // MXFP8_BLOCK),
        HAS_SRC=src_slots is not None,
        HAS_DST=dst_slots is not None,
        num_warps=4,
    )


@mxfp8_to_e4m3.register_fake
def _mxfp8_to_e4m3_fake(
    codes: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    inv_scale: float,
    src_slots: Optional[torch.Tensor] = None,
    dst_slots: Optional[torch.Tensor] = None,
) -> None:
    return None
