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

* A call is two launches on the current stream: the re-layout of K', V' and their scales into per-tile images
  (layout.py), then the attention kernel (kernel.py), one CTA per (query tile, head, batch) unit.
* The last units % nsm units of the grid (nsm: the SM count of the device) run as two key halves each (split_plan).
* The kernels are specialized on the shape, so each (device, shape, SM count) compiles once per process.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from ...runtime import compiled, current_stream, gmem_ptr, num_sms
from .kernel import NTHREADS, RECORD_BYTES, SMEM_BYTES, attn_kernel
from .layout import STAGE, relayout_kernel


@cute.jit
def attn_launch(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    ksf_ptr: cute.Pointer,
    score_scale_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    vsf_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    img_ptr: cute.Pointer,
    ws_ptr: cute.Pointer,
    lens_ptr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    B: cutlass.Constexpr,
    NSM: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    nT = (S + 127) // 128
    # lens_ptr: None, or the valid prefix lengths, which both kernels read in place
    relayout_kernel(k_ptr, ksf_ptr, v_ptr, vsf_ptr, vm_ptr, img_ptr, S, H, lens_ptr).launch(
        grid=[nT * H * B, 1, 1], block=[256, 1, 1], stream=stream
    )
    nfull, rsplit, kmid = split_plan(B, S, H, NSM)
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
    ).launch(
        grid=[nfull + 2 * rsplit, 1, 1], block=[NTHREADS, 1, 1], smem=SMEM_BYTES, stream=stream
    )


def split_plan(B, S, H, nsm):
    """Return (full units, split units, split tile). The units (query tile, head, batch) are equal; the last
    R = units mod nsm of them (nsm: the SM count of the device) run as two key halves each, split at a multiple of 4
    tiles and merged by the half that finishes second through records in global memory."""
    nT = (S + 127) // 128
    n = nT * H * B
    r = n % nsm
    if r == 0 or nT < 24:
        return n, 0, 0
    return n - r, r, 4 * ((nT + 4) // 8)


def scratch(B, S, H, nsm):
    """Return the scratch of the attention at this shape on a device of nsm SMs, name -> (dtype, shape).

    * img: the tile images of the operand re-layout; split: the split units' records, then their arrival counters.
    * Only the counters must be zero when the launch starts; the half of a unit that merges returns its counter to zero.
    """
    nT = (S + 127) // 128
    _, rsplit, _ = split_plan(B, S, H, nsm)
    return {
        "img": (torch.uint8, (B * H * nT * STAGE,)),
        "split": (torch.uint8, (rsplit * (RECORD_BYTES + 4),)),
    }


def split_counters(split):
    """Return the arrival counters in the split scratch: its bytes after the records of all units."""
    return split[split.numel() // (RECORD_BYTES + 4) * RECORD_BYTES :]


_compiled = {}


def run(buffers, o, lens=None):
    """Run the attention on the current stream (the re-layout launch, then the attention launch): BF16 o [B, S, H, 128]
    from the operands in the buffers of the call. Nothing here reads a device value or allocates. lens: None, or the
    int32 valid prefix lengths [B], read in place by the kernel variant with lengths; the plan follows the padded S."""
    q4 = buffers["q4"]
    B, S, H, _ = q4.shape
    dev = q4.device
    nsm = num_sms(dev)
    stream = current_stream(dev)
    ptrs = [
        gmem_ptr(dt, x, 16)
        for dt, x in (
            (cutlass.Uint8, q4),
            (cutlass.Uint8, buffers["k4"]),
            (cutlass.Uint8, buffers["q_sf"]),
            (cutlass.Uint8, buffers["k_sf"]),
            (cutlass.Float32, buffers["score_scale"]),
            (cutlass.Uint8, buffers["v4"]),
            (cutlass.Uint8, buffers["v4_sf"]),
            (cutlass.Float32, buffers["v4_scale"]),
            (cutlass.BFloat16, buffers["v_mean"]),
            (cutlass.BFloat16, o),
            (cutlass.Uint8, buffers["img"]),
            (cutlass.Uint8, buffers["split"]),
        )
    ]
    lens_ptr = gmem_ptr(cutlass.Int32, lens, 4) if lens is not None else None
    fn = compiled(
        _compiled,
        (dev.index, B, S, H, nsm, lens is not None),
        lambda: attn_launch,
        *ptrs,
        lens_ptr,
        S,
        H,
        B,
        nsm,
        stream,
    )
    fn(*ptrs, lens_ptr, stream)
