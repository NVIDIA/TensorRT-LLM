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
"""Host entry of the attention kernel: the launch plan of a call, the scratch of its split launch, the compile cache.

* Work units: the (256-row Q block pair, head, batch) triples of a call, one per persistent CTA at a time.
* Main launch: the whole waves of units (a wave: one unit per SM).
* Split launch: the units of a last partial wave, each split along KV into pieces so that every SM stays busy; the
  pieces are merged in the epilogue through scratch in global memory (epilogue.py). It is a programmatic dependent
  of the main launch: it reads only the inputs and writes other rows of O.
* A kernel compiles once per process and key (_get_compiled).
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute.typing import Int32, Int64

from ...constants import key_tiles
from ...runtime import Layout, compiled, current_stream, gmem_ptr, num_sms, pdl_enabled
from .kernel import PingPongAttention


def split_plan(s: int, h: int, b: int, sm_count: int) -> tuple[int, int, int]:
    """Return (work units, units of the main launch, KV pieces per unit of the split launch) at this shape."""
    n_units = ((s + 255) // 256) * h * b
    n_kv = key_tiles(s)
    n_full = (n_units // sm_count) * sm_count
    rem = n_units - n_full
    n_parts = 1
    # the units of a last partial wave are split into as many KV pieces as keep every SM busy, of at least four KV
    # tiles each; a wave that cannot be split goes to the main launch
    if rem > 0:
        n_parts = max(1, min(sm_count // rem, n_kv // 4))
    if n_parts == 1:
        n_full = n_units
    return n_units, n_full, n_parts


def scratch(b: int, s: int, h: int, sm_count: int) -> Layout:
    """Return the merge scratch of the split launch at this shape, name -> (dtype, shape); empty without one.

    A merge slot is a (tail unit, KV piece, O tile) triple. The pieces write its unnormalized O rows and (anchor, row
    sum) pairs before they are read. The word of a unit's first slot of a tile counts the pieces that published the
    tile; it must be zero when the launch starts (the piece that merges zeroes it again).
    """
    n_units, n_full, n_parts = split_plan(s, h, b, sm_count)
    n_slots = (n_units - n_full) * n_parts * 2
    if not n_slots:
        return {}
    return {
        "split_o": (torch.float32, (n_slots * 128 * 128,)),
        "split_ml": (torch.float32, (n_slots * 128 * 2,)),
        "split_flags": (torch.int32, (n_slots,)),
    }


_compiled = {}


def _get_compiled(
    args: tuple,
    device: torch.device,
    sm_count: int,
    qk_channels: int,
    cluster_m: int,
    *,
    split: bool,
    pdl: bool,
    lens: bool,
    rescale_threshold: float,
) -> Callable:
    """Return the kernel compiled for args, kept per key: the device and its SM count, the channels of the QK build,
    the cluster shape (CTA pairs or single CTAs), the launch kind (main or split launch, as a programmatic dependent
    or not), with or without lengths, and the rescale threshold."""
    return compiled(
        _compiled,
        (device.index, sm_count, qk_channels, cluster_m, split, pdl, lens, rescale_threshold),
        lambda: (
            PingPongAttention(
                sm_count,
                qk_channels,
                cluster_m=cluster_m,
                split=split,
                pdl=pdl,
                lens=lens,
                rescale_threshold=rescale_threshold,
            ),
            *args,
        ),
    )


def _make_args(
    qk_ptrs: tuple,
    v: torch.Tensor,
    v_mean: torch.Tensor,
    o: torch.Tensor,
    s: int,
    h: int,
    b: int,
    score_scale: torch.Tensor,
    v_scale: torch.Tensor,
    lens: torch.Tensor | None,
    stream: cuda.CUstream,
    n_full: int,
    n_parts: int = 1,
    ws: tuple[torch.Tensor, ...] | None = None,
) -> tuple:
    return (
        qk_ptrs,
        gmem_ptr(cutlass.Float8E4M3FN, v),
        gmem_ptr(cutlass.BFloat16, v_mean),
        gmem_ptr(cutlass.BFloat16, o),
        Int32(s),
        Int32(h),
        Int32(b),
        Int64(score_scale.data_ptr()),
        Int64(v_scale.data_ptr()),
        Int32(n_full),
        Int32(n_parts),
        # the split launch's merge scratch (null addresses without one)
        *((Int64(t.data_ptr()) for t in ws) if ws is not None else (Int64(0),) * 3),
        # the valid prefix lengths, read in place (a null address without them)
        Int64(lens.data_ptr() if lens is not None else 0),
        stream,
    )


@torch.no_grad()
def run(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    qk_channels: int,
    lens: torch.Tensor | None,
    rescale_threshold: float,
) -> None:
    """Run the NVFP4 Q'K' + FP8 PV attention on the operands of the preparation into BF16 o [B, S, H, 128].

    * qk_channels: the channels of the QK build of the operands (constants.qk_channels). rescale_threshold: one of
      constants.RESCALE_THRESHOLDS["fp8"].
    * buffers: the buffers of the call, the layout of this package. score_scale is the multiplier of the raw scores
      in log2 units, a power of two of at least 2^softmax.MIN_SCORE_LOG2, and stays on the device like v8_scale.
    * Every launch goes to the current stream; nothing here reads a device value or allocates (graph-capturable).
    * lens: None, or the int32 valid prefix lengths [B] of the preparation, read in place by the kernel variant with
      lengths; the launch plan follows the padded sequence either way.
    """
    q4, k4, q_sf, k_sf, v8, v_mean, score_scale, v8_scale = (
        buffers[n] for n in ("q4", "k4", "q_sf", "k_sf", "v8", "v_mean", "score_scale", "v8_scale")
    )
    qk_ptrs = (
        gmem_ptr(cutlass.Float4E2M1FN, q4),
        gmem_ptr(cutlass.Float4E2M1FN, k4),
        gmem_ptr(cutlass.Float8E4M3FN, q_sf),
        gmem_ptr(cutlass.Float8E4M3FN, k_sf),
    )
    b, s, h, _ = q4.shape
    dev = q4.device
    stream = current_stream(dev)
    sm_count = num_sms(dev)
    _, n_full, n_parts = split_plan(s, h, b, sm_count)

    def make_args(n_parts=1, ws=None):
        return _make_args(
            qk_ptrs,
            v8,
            v_mean,
            o,
            s,
            h,
            b,
            score_scale,
            v8_scale,
            lens,
            stream,
            n_full,
            n_parts,
            ws,
        )

    pdl = pdl_enabled()
    if n_full > 0:
        args = make_args()
        # CTA pairs share K/V when both always work on the same head: an even number of 256-row blocks per head,
        # an even unit count and an even persistent stride keep units (2c, 2c+1) in one head in every round.
        cluster_m = (
            2
            if (((s + 255) // 256) % 2 == 0 and n_full % 2 == 0 and min(sm_count, n_full) % 2 == 0)
            else 1
        )
        _get_compiled(
            args,
            dev,
            sm_count,
            qk_channels,
            cluster_m,
            split=False,
            pdl=pdl,
            lens=lens is not None,
            rescale_threshold=rescale_threshold,
        )(*args)
    if "split_flags" in buffers:
        # last partial wave: every remaining unit split into n_parts KV pieces (at most one piece per SM)
        ws = (buffers["split_o"], buffers["split_ml"], buffers["split_flags"])
        args = make_args(n_parts, ws)
        _get_compiled(
            args,
            dev,
            sm_count,
            qk_channels,
            1,
            split=True,
            pdl=pdl and n_full > 0,
            lens=lens is not None,
            rescale_threshold=rescale_threshold,
        )(*args)
