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
"""Host side that the preparation chains of the sm_120 kernels share: compile, launch sizes, scratch and the call.

The chain of a kernel package is its host function (<kernel package>/prep.py): basis launch, V pass, Q'/K' pass, encode.
"""

import inspect

import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ...constants import DEFAULT_RANK, HEAD_DIM, QK_CHANNELS
from ...inputs import hadamard
from ...runtime import current_stream, num_sms, token_pitches
from .encode import ENCODE_THREADS


def _compile(host, cache, device, fused, with_lens):
    """Return the host function of a chain compiled for a device and a chain form (fused: the V pass in the basis
    launch; with_lens: the kernels read valid prefix lengths), kept in the cache of its kernel package."""
    fn = cache.get((device.index, fused, with_lens))
    if fn is None:
        # placeholders of the run-time parameters (Int64 addresses, Int32 sizes), then the stream and the switches
        args = [
            Int64(0) if p.annotation is Int64 else Int32(1)
            for p in inspect.signature(host).parameters.values()
            if p.annotation in (Int64, Int32)
        ]
        args += [cute.runtime.make_fake_stream(), fused, with_lens]
        fn = cache[(device.index, fused, with_lens)] = cute.compile(host, *args)
    return fn


def _grids(B, S, H, nsm):
    """Return the SM-count dependent sizes of the launches: spare CTAs of the basis launch, and the item and CTA counts
    of the Q/K pass, the V pass and the encode launch."""
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
    """Return the scratch of a chain at this shape on a device of nsm SMs, name -> (dtype, shape).

    * ut, vmax, gmax, prov: the bases U^T, the V-scan slots, the group maxima, the provisional block scales.
    * Only done, the four words of the basis launch (V-scan tiles published, V-pass tile counter, V-scan tile claim
      counter, V-scan maximum), must be zero at its start.
    """
    nspare, _, nqk_cta, _, _, _ = _grids(B, S, H, nsm)
    BH, T = B * H, (S + 127) // 128
    return {
        "ut": (torch.bfloat16, (BH * DEFAULT_RANK * HEAD_DIM,)),
        "vmax": (torch.float32, (nspare,)),
        "gmax": (torch.float32, (nqk_cta * 8,)),
        "prov": (torch.bfloat16, (2 * BH * T * 1536,)),
        "done": (torch.int32, (4,)),
    }


def run(
    host, cache, host_buffers, min_score_log2, q, k, v, rows, buffers, lens=None, map_rows=False
):
    """Run the launches of a chain on the current stream: raw BF16 q, k, v [B, S, H, 128] into the buffers of the call.

    * q, k, v: contiguous, or views of a fused projection; the kernels read their tokens at the pitch of each tensor.

    * host, cache, host_buffers: the host function of the kernel package, its compile cache and the buffers whose
      addresses it takes after q, k, v, the rows and the rotation, in its order.
    * min_score_log2: log2 of the score-scale floor; rows: the sampled tokens of the Q/K basis (inputs.sample_rows).
    * lens: None, or the int32 valid prefix lengths [B] on the device, which the kernels read in place: every pass
      then takes the tokens of a batch element below its length only.
    * rows with lens: with map_rows the default draw over the padded sequence, which the eigensolver maps into each
      valid prefix; else the caller's rows [n] or [B, n] as given (a row at or past a length: the last valid token).
    """
    B, S, H, _ = q.shape
    T = (S + 127) // 128
    dev = q.device
    nsm = num_sms(dev)
    nspare, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta = _grids(B, S, H, nsm)
    # the V pass shares the basis launch when the eigenproblems take at most a quarter of the SMs and the grid of the
    # launch has at most as many CTAs as the device has SMs (its scan claims tiles from a counter and waits for
    # tiles, not for CTAs, so the form is correct for any residency; the limit keeps the pass on free SMs)
    fn = _compile(host, cache, dev, 4 * B * H <= nsm and B * H + nspare <= nsm, lens is not None)
    fn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        rows.data_ptr(),
        hadamard(dev).data_ptr(),
        *(buffers[n].data_ptr() for n in host_buffers),
        lens.data_ptr() if lens is not None else 0,
        B,
        S,
        H,
        *token_pitches(q, k, v),
        rows.shape[-1],
        T,
        min_score_log2,
        nspare,
        nqk_items,
        nqk_cta,
        nv_items,
        nv_cta,
        nenc_cta,
        # rows mode of the eigensolver: 0 the mapped default draw, 1 the caller's shared rows, 2 its rows per element
        0 if map_rows else rows.dim(),
        current_stream(dev),
    )
