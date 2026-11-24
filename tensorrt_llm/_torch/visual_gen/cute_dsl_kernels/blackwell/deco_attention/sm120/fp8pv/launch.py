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

Launches of a call, on the current stream; each compiles once per process, device and presence of valid lengths:
    * The attention kernel (kernel.py) over the whole units (query tile, head, batch), then the key-range pieces of
      the split tail; its host function builds the TMA atoms of K', V, the K' scales and v_mean.
    * The merge of the pieces (epilogue.py).
    * For fewer than 128 keys instead: one launch of the single-tile kernel (small.py).
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Int32, Int64

from ...runtime import compiled, current_stream, gmem_ptr, num_sms
from .epilogue import PART_N, merge_partial_kernel
from .kernel import attn_kernel
from .layout import EP, K_STEPS, KA_BYTES, KT, LEN_BYTES, NTHR, QT, SF_BYTES, SMEM_BYTES
from .small import SMALL_SMEM_BYTES, small_kernel

MAX_SPLIT = 6  # key-range pieces per unit of the split tail, at most


def split_plan(units: int, nT: int, nsm: int) -> tuple[int, int, int]:
    """Return (whole units, split units, pieces per unit); the last units % nsm + nsm // 2 units run as pieces."""
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
    S: Int32,
    H: Int32,
    B: Int32,
    n_full: Int32,
    n_tail: Int32,
    parts: Int32,
    stream: cuda.CUstream,
):
    nT = (S + KT - 1) // KT
    batch = Int64(S) * H * 128
    # K' as (32 bytes, keys, K_STEPS atom columns) of [S, 128 bytes, H, B]: one box per tile of K_STEPS x 128 keys x
    # 32 bytes, atom-major in shared memory (32-byte swizzle)
    lay_k = cute.make_layout((32, S, K_STEPS, H, B), stride=(1, H * 128, 32, 128, batch))
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
    lay = cute.make_layout((S, 128, H, B), stride=(H * 128, 1, 128, batch))
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
        (nT * srows, 16, H, B), stride=(16, 1, nT * SF_BYTES, Int64(H) * nT * SF_BYTES)
    )
    tma_atom_s, tma_s = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(ksf_ptr, lay_s),
        cute.make_layout((srows, 16), stride=(16, 1)),
        (srows, 16),
    )
    # v_mean [B, H, 8 nT, 128] BF16 as rows of 256 bytes: a tile's 8 rows in two boxes of 8 rows x 128 bytes
    lay_m = cute.make_layout(
        (128, 8 * nT, 2, H, B), stride=(1, 256, 128, 8 * nT * 256, Int64(H) * nT * (8 * 256))
    )
    tma_atom_m, tma_m = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        cute.make_tensor(vm_ptr, lay_m),
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 8, 2), stride=(1, 128, 1024))
        ),
        (128, 8, 2),
    )
    # lens_ptr: None, or the valid prefix lengths, which the kernels read in place
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
        parts,
        lens_ptr,
    ).launch(
        grid=[n_full + n_tail * parts, 1, 1],
        block=[NTHR, 1, 1],
        smem=SMEM_BYTES + (LEN_BYTES if lens_ptr is not None else 0),
        stream=stream,
    )


@cute.jit
def merge_launch(
    part_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    vs_ptr: cute.Pointer,
    lens_ptr,
    S: Int32,
    H: Int32,
    n_full: Int32,
    n_tail: Int32,
    parts: Int32,
    stream: cuda.CUstream,
):
    merge_partial_kernel(part_ptr, o_ptr, vs_ptr, S, H, parts, n_full, lens_ptr).launch(
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
    S: Int32,
    H: Int32,
    B: Int32,
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


def scratch(B: int, S: int, H: int, nsm: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return the scratch of the attention, name -> (dtype, shape), which every call writes before it reads it."""
    if S < KT:
        return {}
    nT = (S + KT - 1) // KT
    _, n_tail, parts = split_plan(nT * H * B, nT, nsm)
    return {"partial": (torch.float32, (max(1, n_tail * parts * PART_N * NTHR),))}


_compiled = {}


def run(
    buffers: dict[str, torch.Tensor], o: torch.Tensor, lens: torch.Tensor | None = None
) -> None:
    """Write BF16 o [B, S, H, 128] = (P V' + S M) * v8_scale / l from the buffers of the call, on the current stream.
    Nothing here reads a device value or allocates. lens: None, or the int32 valid prefix lengths [B] of the
    preparation, read in place by the kernel variants with lengths; the launch plan follows the padded sequence."""
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
    ptrs = [gmem_ptr(dt, x, 16) for dt, x in ops]

    def launch(fn: Callable, ptrs: list, sizes: tuple[int, ...]) -> None:
        key = (dev.index, fn, lens is not None)
        args = (*ptrs, lens_ptr, *(Int32(n) for n in sizes), stream)
        compiled(_compiled, key, lambda: fn, *args)(*ptrs, lens_ptr, *sizes, stream)

    if S < KT:
        launch(small_launch, ptrs, (S, H, B))
        return
    nT = (S + KT - 1) // KT
    plan = split_plan(nT * H * B, nT, num_sms(dev))
    partial = gmem_ptr(cutlass.Float32, buffers["partial"], 16)
    flags = gmem_ptr(cutlass.Int32, buffers["v_flags"], 16)
    launch(attn_launch, [*ptrs, partial, flags], (S, H, B, *plan))
    if plan[1]:
        launch(merge_launch, [partial, ptrs[8], ptrs[6]], (S, H, *plan))
