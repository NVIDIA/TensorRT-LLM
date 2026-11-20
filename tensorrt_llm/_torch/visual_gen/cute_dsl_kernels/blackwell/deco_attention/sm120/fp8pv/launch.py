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
"""Host entry of the sm_120 FP8-PV attention: the launch plan of a call, its scratch and the compile cache.

With at least one full key tile (S >= 128) a call is up to three launches on the current stream: the scan of V and
v_mean for non-finite values (scan.py), the attention kernel (kernel.py) over the whole units (query tile, head,
batch) followed by the key-range pieces of the split tail, and the merge of the pieces (epilogue.py). The host
function builds the TMA atoms of K', V, the K' scales and v_mean. A shorter sequence is one launch of the single-tile
kernel (small.py). The kernels are specialized on the shape, so each (device, shape, SM count) compiles once per
process."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync

from ...runtime import compiled, current_stream, gmem_ptr, num_sms
from .epilogue import PART_N, merge_partial_kernel
from .kernel import attn_kernel
from .layout import EP, K_STEPS, KA_BYTES, KT, LEN_BYTES, NTHR, QT, SF_BYTES, SMEM_BYTES
from .scan import SCAN_THREADS, v_tile_nan_kernel
from .small import SMALL_SMEM_BYTES, small_kernel

MAX_SPLIT = 6  # key-range pieces per unit of the split tail, at most


def split_plan(units, nT, nsm):
    """(whole units, split units, pieces per split unit): the last units % nsm + nsm // 2 units of the grid (nsm: the
    SM count of the device) run as key-range pieces after all whole units."""
    parts = min(MAX_SPLIT, nT // 2)
    tail = min(units, units % nsm + nsm // 2)
    if parts < 2:
        return units, 0, 1
    return units - tail, tail, parts


@cute.jit
def attn_launch(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    ksf_ptr: cute.Pointer,
    score_scale_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    part_ptr: cute.Pointer,
    flags_ptr: cute.Pointer,
    lens_ptr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    B: cutlass.Constexpr,
    NSM: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    nT = (S + KT - 1) // KT
    # K' as (32 bytes, keys, K_STEPS atom columns) of [S, 128 bytes, H, B]: one box per tile of K_STEPS x 128 keys x
    # 32 bytes, atom-major in shared memory (32-byte swizzle)
    lay_k = cute.make_layout((32, S, K_STEPS, H, B), stride=(1, H * 128, 32, 128, S * H * 128))
    tma_atom_k, tma_k = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(k_ptr, lay_k),
        cute.make_composed_layout(
            cute.make_swizzle(1, 4, 3),
            0,
            cute.make_layout((32, KT, K_STEPS), stride=(1, 32, KA_BYTES)),
        ),
        (32, KT, K_STEPS),
    )
    # V as [S, 128 bytes, H, B] in boxes of 128 keys x 128 bytes (128-byte swizzle)
    lay = cute.make_layout((S, 128, H, B), stride=(H * 128, 1, 128, S * H * 128))
    tma_atom_v, tma_v = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(v_ptr, lay),
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((KT, 128), stride=(128, 1))
        ),
        (KT, 128),
    )
    # the K' scale image of a (batch, head) as rows of 16 bytes (an atom is 32 rows)
    srows = 32 * K_STEPS
    lay_s = cute.make_layout(
        (nT * srows, 16, H, B), stride=(16, 1, nT * SF_BYTES, H * nT * SF_BYTES)
    )
    tma_atom_s, tma_s = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(ksf_ptr, lay_s),
        cute.make_layout((srows, 16), stride=(16, 1)),
        (srows, 16),
    )
    # v_mean [B, H, 8 nT, 128] BF16 as rows of 256 bytes: a tile's 8 rows in two boxes of 8 rows x 128 bytes
    lay_m = cute.make_layout(
        (128, 8 * nT, 2, H, B), stride=(1, 256, 128, 8 * nT * 256, H * 8 * nT * 256)
    )
    tma_atom_m, tma_m = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(vm_ptr, lay_m),
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 8, 2), stride=(1, 128, 1024))
        ),
        (128, 8, 2),
    )
    n_full, n_tail, parts = split_plan(nT * H * B, nT, NSM)
    # lens_ptr: None, or the valid prefix lengths, which the three kernels read in place
    v_tile_nan_kernel(v_ptr, vm_ptr, flags_ptr, S, H, lens_ptr).launch(
        grid=[nT, H, B], block=[SCAN_THREADS, 1, 1], smem=16, stream=stream
    )
    attn_kernel(
        q_ptr,
        qsf_ptr,
        score_scale_ptr,
        vs_ptr,
        o_ptr,
        part_ptr,
        flags_ptr,
        tma_atom_k,
        tma_k,
        tma_atom_v,
        tma_v,
        tma_atom_s,
        tma_s,
        tma_atom_m,
        tma_m,
        S,
        H,
        n_full,
        n_tail,
        parts,
        lens_ptr,
    ).launch(
        grid=[n_full + n_tail * parts, 1, 1],
        block=[NTHR, 1, 1],
        smem=SMEM_BYTES + (LEN_BYTES if lens_ptr is not None else 0),
        stream=stream,
    )
    if cutlass.const_expr(n_tail > 0):
        merge_partial_kernel(part_ptr, o_ptr, vs_ptr, S, H, nT, parts, n_full, lens_ptr).launch(
            grid=[n_tail, 1, 1], block=[NTHR, 1, 1], smem=QT * EP, stream=stream
        )


@cute.jit
def small_launch(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    qsf_ptr: cute.Pointer,
    ksf_ptr: cute.Pointer,
    score_scale_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    vm_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lens_ptr,
    S: cutlass.Constexpr,
    H: cutlass.Constexpr,
    B: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    small_kernel(
        q_ptr,
        k_ptr,
        qsf_ptr,
        ksf_ptr,
        score_scale_ptr,
        v_ptr,
        vs_ptr,
        vm_ptr,
        o_ptr,
        S,
        H,
        lens_ptr,
    ).launch(grid=[1, H, B], block=[NTHR, 1, 1], smem=SMALL_SMEM_BYTES, stream=stream)


def scratch(B, S, H, nsm):
    """The scratch of the attention at this shape on a device of nsm SMs, name -> (dtype, shape), which every call
    writes before it reads it: the FP32 partials of the split tail's pieces and the per-tile flags of the scan launch
    (none for fewer than 128 keys, whose kernel needs none)."""
    if S < KT:
        return {}
    nT = (S + KT - 1) // KT
    _, n_tail, parts = split_plan(nT * H * B, nT, nsm)
    return {
        "partial": (torch.float32, (max(1, n_tail * parts * PART_N * NTHR),)),
        "scan_flags": (torch.int32, (B, H, nT)),
    }


_compiled = {}


def run(buffers, o, lens=None):
    """BF16 o [B, S, H, 128] = (P V' + S M) * v8_scale / l from the operands of the preparation (buffers: the buffers
    of the call, the layout of this package) on the current stream. Nothing here reads a device value or allocates.
    lens: None, or the int32 valid prefix lengths [B] of the preparation, read in place by the kernel variants with
    lengths; the launch plan follows the padded sequence either way."""
    q4 = buffers["q4"]
    B, S, H, _ = q4.shape
    dev = q4.device
    stream = current_stream(dev)
    ops = (
        (cutlass.Uint8, q4),
        (cutlass.Uint8, buffers["k4"]),
        (cutlass.Uint8, buffers["q_sf"]),
        (cutlass.Uint8, buffers["k_sf"]),
        (cutlass.Float32, buffers["score_scale"]),
        (cutlass.Uint8, buffers["v8"]),
        (cutlass.Float32, buffers["v8_scale"]),
        (cutlass.Uint8, buffers["v_mean"]),
        (cutlass.BFloat16, o),
    )
    lens_ptr = gmem_ptr(cutlass.Int32, lens, 4) if lens is not None else None
    if S < KT:
        args = [gmem_ptr(dt, x, 16) for dt, x in ops] + [lens_ptr]
        fn = compiled(
            _compiled,
            (dev.index, "small", B, S, H, lens is not None),
            lambda: small_launch,
            *args,
            S,
            H,
            B,
            stream,
        )
        fn(*args, stream)
        return
    nsm = num_sms(dev)
    args = [
        gmem_ptr(dt, x, 16)
        for dt, x in ops
        + ((cutlass.Float32, buffers["partial"]), (cutlass.Int32, buffers["scan_flags"]))
    ]
    args.append(lens_ptr)
    fn = compiled(
        _compiled,
        (dev.index, B, S, H, nsm, lens is not None),
        lambda: attn_launch,
        *args,
        S,
        H,
        B,
        nsm,
        stream,
    )
    fn(*args, stream)
