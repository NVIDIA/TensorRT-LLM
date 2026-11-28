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
"""Host entry of the sm_120 attention: the launch plan of a call, its scratch and the compile cache.

* A call is one launch on the current stream: the attention kernel (kernel.py), one CTA per (query tile, head,
  batch) unit, on the per-tile images of K', V' and their scales that the preparation wrote (layout.py).
* The last units % nsm units of the grid (nsm: the SM count of the device) run as two key halves each (split_plan).
* The kernels take the shape and the plan at run time: one compile per device and rescale threshold, and one more
  for calls with lengths.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import Int32

from ...runtime import Layout, compiled, current_stream, gmem_ptr, num_sms
from .kernel import NTHREADS, RECORD_BYTES, SMEM_BYTES, attn_kernel
from .layout import key_tiles


@cute.jit
def attn_launch(
    q_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    score_scale_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    img_ptr: cute.Pointer,
    ws_ptr: cute.Pointer,
    lens_ptr,
    S: Int32,
    H: Int32,
    nfull: Int32,
    rsplit: Int32,
    kmid: Int32,
    stream: cuda.CUstream,
    rescale_threshold: cutlass.Constexpr,
):
    # lens_ptr: None, or the valid prefix lengths, which the kernel reads in place; nfull, rsplit, kmid: split_plan;
    # rescale_threshold: the rescale threshold of the softmax
    attn_kernel(
        q_ptr,
        qsf_ptr,
        score_scale_ptr,
        vs_ptr,
        img_ptr,
        o_ptr,
        ws_ptr,
        S,
        H,
        nfull,
        rsplit,
        kmid,
        lens_ptr,
        rescale_threshold=rescale_threshold,
    ).launch(
        grid=[nfull + 2 * rsplit, 1, 1], block=[NTHREADS, 1, 1], smem=SMEM_BYTES, stream=stream
    )


def split_plan(B: int, S: int, H: int, nsm: int) -> tuple[int, int, int]:
    """Return (full units, split units, split tile). The units (query tile, head, batch) are equal; the last
    R = units mod nsm of them (nsm: the SM count of the device) run as two key halves each, split at a multiple of 4
    tiles and merged by the half that finishes second through records in global memory."""
    nT = key_tiles(S)
    n = nT * H * B
    r = n % nsm
    if r == 0 or nT < 24:
        return n, 0, 0
    return n - r, r, 4 * ((nT + 4) // 8)


def scratch(B: int, S: int, H: int, nsm: int) -> Layout:
    """Return the scratch of the attention at this shape on a device of nsm SMs, name -> (dtype, shape).

    * split: the split units' records, then their arrival counters.
    * Only the counters must be zero when the launch starts; the half of a unit that merges returns its counter to zero.
    """
    _, rsplit, _ = split_plan(B, S, H, nsm)
    return {"split": (torch.uint8, (rsplit * (RECORD_BYTES + 4),))}


def split_counters(split: torch.Tensor) -> torch.Tensor:
    """Return the arrival counters in the split scratch: its bytes after the records of all units."""
    return split[split.numel() // (RECORD_BYTES + 4) * RECORD_BYTES :]


_compiled = {}


def run(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    lens: torch.Tensor | None,
    rescale_threshold: float,
) -> None:
    """Run the attention on the current stream: BF16 o [B, S, H, 128] from the operands in the buffers of the call.
    Nothing here reads a device value or allocates. lens: None, or the int32 valid prefix lengths [B], read in place
    by the kernel variant with lengths; the plan follows the padded S. rescale_threshold: one of
    constants.RESCALE_THRESHOLDS["nvfp4"]."""
    q4 = buffers["q4"]
    B, S, H, _ = q4.shape
    dev = q4.device
    stream = current_stream(dev)
    ptrs = [
        gmem_ptr(dt, x, 16)
        for dt, x in (
            (cutlass.Uint8, q4),
            (cutlass.Uint8, buffers["q_sf"]),
            (cutlass.Float32, buffers["score_scale"]),
            (cutlass.Float32, buffers["v4_scale"]),
            (cutlass.BFloat16, o),
            (cutlass.Uint8, buffers["img"]),
            (cutlass.Uint8, buffers["split"]),
        )
    ]
    lens_ptr = gmem_ptr(cutlass.Int32, lens, 4) if lens is not None else None
    plan = split_plan(B, S, H, num_sms(dev))
    args = (*ptrs, lens_ptr, *(Int32(x) for x in (S, H, *plan)), stream)
    key = (dev.index, lens is not None, rescale_threshold)
    compiled(_compiled, key, lambda: (attn_launch, *args, rescale_threshold))(*args)
