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
"""Host entry of the attention kernel: the launch plan of a call, the scratch of its split launch and the compile
cache.

The work units of a call are the (256-row Q block pair, head, batch) triples, one per persistent CTA at a time. The
main launch covers the whole waves of units (a wave: one unit per SM). The units of a last partial wave run as a
second, split launch: each is split along KV into pieces so that every SM stays busy, and the pieces are merged in
the epilogue through scratch in global memory (epilogue.py). The split launch is a programmatic dependent of the
main launch: it reads only the inputs and writes other rows of O. Each (device, compensation rank, cluster shape,
launch kind) compiles once per process."""

import cutlass
import torch
from cutlass.cute.typing import Int32, Int64

from ...runtime import compiled, current_stream, gmem_ptr, num_sms
from .kernel import PingPongAttention


def split_plan(s, h, b, sm_count):
    """(work units, units of the main launch, KV pieces per unit of the split launch) at this shape. The main launch
    takes the whole waves of sm_count units; the units of a last partial wave are split into as many KV pieces as
    keep every SM busy, of at least four KV tiles each. A wave that cannot be split goes to the main launch."""
    n_units = ((s + 255) // 256) * h * b
    n_kv = (s + 127) // 128
    n_full = (n_units // sm_count) * sm_count
    rem = n_units - n_full
    n_parts = 1
    if rem > 0:
        n_parts = max(1, min(sm_count // rem, n_kv // 4))
    if n_parts == 1:
        n_full = n_units
    return n_units, n_full, n_parts


def split_flag_count(s, h, b, sm_count):
    """Ready flags of the split launch at this shape, one per (tail unit, KV piece, O tile) merge slot; 0 without a
    split launch."""
    n_units, n_full, n_parts = split_plan(s, h, b, sm_count)
    return (n_units - n_full) * n_parts * 2


def scratch(b, s, h, sm_count):
    """The merge scratch of the split launch at this shape on a device of sm_count SMs, name -> (dtype, shape): the
    unnormalized O rows and the (anchor, row sum) pairs of every merge slot, which the pieces write before they are
    read, and the slots' ready flags, which must be zero when the launch starts (the merging CTAs return them to
    zero). Empty when the shape has no split launch."""
    n_slots = split_flag_count(s, h, b, sm_count)
    if not n_slots:
        return {}
    return {
        "split_o": (torch.float32, (n_slots * 128 * 128,)),
        "split_ml": (torch.float32, (n_slots * 128 * 2,)),
        "split_flags": (torch.int32, (n_slots,)),
    }


_compiled = {}


def _get_compiled(args, device, sm_count, rank, cluster_m, split, pdl):
    return compiled(
        _compiled,
        (device.index, sm_count, rank, cluster_m, split, pdl),
        lambda: PingPongAttention(sm_count, rank, cluster_m=cluster_m, split=split, pdl=pdl),
        *args,
    )


def _make_args(
    qk_ptrs, v, v_mean, o, s, h, b, score_scale, v_scale, stream, n_valid, n_parts=1, ws=None
):
    return (
        qk_ptrs,
        gmem_ptr(cutlass.Float8E4M3FN, v),
        gmem_ptr(cutlass.BFloat16, v_mean),
        gmem_ptr(cutlass.BFloat16, o),
        Int32(s),
        Int32(h),
        Int32(b),
        # the scales stay on the device: the log2-domain softmax multiplier and v_scale are read in place
        Int64(score_scale.data_ptr()),
        Int64(v_scale.data_ptr()),
        Int32(n_valid),
        Int32(n_parts),
        # the split launch's merge scratch (null addresses without one)
        *((Int64(t.data_ptr()) for t in ws) if ws is not None else (Int64(0),) * 3),
        stream,
    )


@torch.no_grad()
def run(buffers, o, rank):
    """NVFP4 Q'K' + FP8 PV attention on the operands of the preparation at compensation rank `rank` (buffers: the
    buffers of the call, the layout of this package) into BF16 o [B, S, H, 128]. score_scale is the multiplier of the
    raw scores in log2 units, a power of two of at least 2^softmax.MIN_SCORE_LOG2, and stays on the device like
    v8_scale. The main launch covers the whole waves and the split launch the last partial wave. Nothing here reads a
    device value or allocates (graph-capturable, no host sync); every launch goes to the current stream."""
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
    if n_full > 0:
        args = _make_args(qk_ptrs, v8, v_mean, o, s, h, b, score_scale, v8_scale, stream, n_full)
        # CTA pairs share K/V when both always work on the same head: an even number of 256-row blocks per head,
        # an even unit count and an even persistent stride keep units (2c, 2c+1) in one head in every round.
        cluster_m = (
            2
            if (((s + 255) // 256) % 2 == 0 and n_full % 2 == 0 and min(sm_count, n_full) % 2 == 0)
            else 1
        )
        _get_compiled(args, dev, sm_count, rank, cluster_m, False, True)(*args)
    if "split_flags" in buffers:
        # last partial wave: every remaining unit split into n_parts KV pieces (at most one piece per SM), merged in
        # the epilogue
        ws = (buffers["split_o"], buffers["split_ml"], buffers["split_flags"])
        args = _make_args(
            qk_ptrs, v8, v_mean, o, s, h, b, score_scale, v8_scale, stream, n_full, n_parts, ws
        )
        _get_compiled(args, dev, sm_count, rank, 1, True, n_full > 0)(*args)
