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
"""Host side of the preparation chains of the sm_120 kernels. The chain of a kernel package is its host function
(<kernel package>/prep.py: the basis launch and the V pass with the V operands of its PV format, the Q'/K' pass and
the encode launch); what the chains share is here: the compile of a host function per device and chain form, the
sizes of the launches, the scratch and the call."""

import inspect

import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ...constants import DEFAULT_RANK, HEAD_DIM, QK_CHANNELS
from ...inputs import hadamard
from ...runtime import current_stream, num_sms
from .encode import ENCODE_THREADS


def _compile(host, cache, device, fused):
    """The host function of a chain compiled for a device and a chain form (fused: the V pass in the basis launch),
    kept in the cache of its kernel package."""
    fn = cache.get((device.index, fused))
    if fn is None:
        # placeholders of the host function's run-time parameters (Int64 addresses, Int32 sizes), then the stream and
        # the switch
        args = [
            Int64(0) if p.annotation is Int64 else Int32(1)
            for p in inspect.signature(host).parameters.values()
            if p.annotation in (Int64, Int32)
        ]
        args += [cute.runtime.make_fake_stream(), fused]
        fn = cache[(device.index, fused)] = cute.compile(host, *args)
    return fn


def _grids(nsm, B, S, H):
    """The SM-count dependent sizes of the launches: spare CTAs of the basis launch, and the item and CTA counts of
    the Q/K pass, the V pass and the encode launch."""
    BH = B * H
    T = (S + 127) // 128
    nspare = max(nsm - BH, 16)
    nqk_items = 2 * BH * T
    nqk_cta = min(nqk_items, nsm)
    nv_items = BH * T
    nv_cta = min(nv_items, 2 * nsm)
    nenc_cta = min(
        max((2 * BH * T * QK_CHANNELS + ENCODE_THREADS - 1) // ENCODE_THREADS, 1), 8 * nsm
    )
    return nspare, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta


def scratch(B, S, H, nsm):
    """The scratch of a chain at this shape on a device of nsm SMs, name -> (dtype, shape): the bases U^T, the V-scan
    slots (one float32 each), the group maxima and the provisional block scales, which every call writes before it
    reads them, and the two counters of the basis launch (the V-scan barrier and the V-pass tile counter), which
    must be zero when it starts."""
    nspare, _, nqk_cta, _, _, _ = _grids(nsm, B, S, H)
    BH, T = B * H, (S + 127) // 128
    return {
        "ut": (torch.bfloat16, (BH * DEFAULT_RANK * HEAD_DIM,)),
        "vmax": (torch.float32, (nspare,)),
        "gmax": (torch.float32, (nqk_cta * 8,)),
        "prov": (torch.bfloat16, (2 * BH * T * 1536,)),
        "done": (torch.int32, (2,)),
    }


def run(host, cache, host_buffers, min_score_log2, q, k, v, qk_rows, buffers):
    """The launches of a chain on the current stream, from raw BF16 q, k, v [B, S, H, 128] into the buffers of the
    call. host: the host function of the kernel package, cache: its compile cache, host_buffers: the buffers whose
    addresses the host function takes after q, k, v, the rows and the rotation, in its order, min_score_log2: log2
    of the score-scale floor of the attention. qk_rows: the sampled tokens of the Q/K basis (inputs.sample_rows)."""
    B, S, H, _ = q.shape
    T = (S + 127) // 128
    dev = q.device
    nsm = num_sms(dev)
    nspare, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta = _grids(nsm, B, S, H)
    # the V pass shares the basis launch when the eigenproblems take at most a quarter of the SMs and the grid of the
    # launch has at most as many CTAs as the device has SMs
    fn = _compile(host, cache, dev, 4 * B * H <= nsm and B * H + nspare <= nsm)
    fn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        qk_rows.data_ptr(),
        hadamard(dev).data_ptr(),
        *(buffers[n].data_ptr() for n in host_buffers),
        B,
        S,
        H,
        qk_rows.shape[0],
        T,
        min_score_log2,
        nspare,
        nqk_items,
        nqk_cta,
        nv_items,
        nv_cta,
        nenc_cta,
        current_stream(dev),
    )
