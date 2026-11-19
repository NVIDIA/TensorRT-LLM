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
"""Host entry of the preparation of the sm_120 FP8-PV kernel (the passes of ../prep and this kernel's V pass): raw
BF16 q, k, v into the operands of the attention kernel (kernel.py), launched through one host function per device and
chain form.

Launches, on the current stream: the basis launch (prep/basis.py with the V scan of v_pass.py: per (batch, head) the
Gram matrices of the sampled q and k tokens and their top-32 eigenvectors; the other CTAs scan V for max|v - vbar|
and, when the launch has room for it, run the V pass), the V pass otherwise (v_pass.v_kernel), the Q'/K' pass
(qk_pass.py: rotation, body and compensation channels in NVFP4, provisional block scales, group maxima) and the
encode launch (encode.py: group scales, the score-scale floor, score_scale, UE4M3 block scales). Every launch takes
the shape at run time, so each of the two forms of the chain (the V pass in the basis launch or on its own) compiles
once.
"""

import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ...constants import DEFAULT_RANK, HEAD_DIM, QK_CHANNELS
from ...inputs import hadamard
from ...runtime import current_stream, num_sms
from ..prep.basis import EIG_THREADS
from ..prep.encode import ENCODE_THREADS, encode_kernel
from ..prep.qk_pass import QK_THREADS, qk_kernel
from .softmax import MIN_SCORE_LOG2
from .v_pass import V_THREADS, basis_kernel, v_kernel


@cute.jit
def _host(
    q_p: Int64,
    k_p: Int64,
    v_p: Int64,
    rows_p: Int64,
    had_p: Int64,
    q4_p: Int64,
    k4_p: Int64,
    qsf_p: Int64,
    ksf_p: Int64,
    score_scale_p: Int64,
    v8_p: Int64,
    vmean_p: Int64,
    v8s_p: Int64,
    ut_p: Int64,
    vmax_p: Int64,
    gmax_p: Int64,
    prov_p: Int64,
    done_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    NQK: Int32,
    T: Int32,
    msl: Int32,
    nspare: Int32,
    eigwait: Int32,
    nqk_items: Int32,
    nqk_cta: Int32,
    nv_items: Int32,
    nv_cta: Int32,
    nenc_cta: Int32,
    stream: cuda.CUstream,
    FUSED: cutlass.Constexpr,
):
    BH = B * H
    # the placeholder parameters _unused0.._unused2 of the V kernels, which read none of them
    null = Int64(0)
    basis_kernel(
        q_p,
        k_p,
        v_p,
        rows_p,
        had_p,
        ut_p,
        vmax_p,
        done_p,
        v8_p,
        null,
        null,
        vmean_p,
        v8s_p,
        null,
        B,
        S,
        H,
        T,
        NQK,
        BH,
        nspare,
        eigwait,
        FUSED,
    ).launch(grid=(BH + nspare, 1, 1), block=(EIG_THREADS, 1, 1), stream=stream)
    if cutlass.const_expr(not FUSED):
        v_kernel(
            v_p, vmax_p, v8_p, null, null, vmean_p, v8s_p, null, S, H, T, nspare, nv_items, nv_cta
        ).launch(grid=(nv_cta, 1, 1), block=(V_THREADS, 1, 1), stream=stream)
    qk_kernel(
        q_p, k_p, had_p, ut_p, q4_p, k4_p, prov_p, gmax_p, B, S, H, T, nqk_items, nqk_cta
    ).launch(grid=(nqk_cta, 1, 1), block=(QK_THREADS, 1, 1), stream=stream)
    encode_kernel(
        prov_p, gmax_p, qsf_p, ksf_p, score_scale_p, done_p, B, H, T, nqk_cta, msl, nenc_cta
    ).launch(grid=(nenc_cta, 1, 1), block=(ENCODE_THREADS, 1, 1), stream=stream)


_compiled = {}


def _compile(device, fused):
    fn = _compiled.get((device.index, fused))
    if fn is None:
        # placeholders of _host's run-time parameters (Int64 addresses, Int32 sizes), then the stream and the switch
        args = [
            Int64(0) if p.annotation is Int64 else Int32(1)
            for p in inspect.signature(_host).parameters.values()
            if p.annotation in (Int64, Int32)
        ]
        args += [cute.runtime.make_fake_stream(), fused]
        fn = _compiled[(device.index, fused)] = cute.compile(_host, *args)
    return fn


def _grids(nsm, B, S, H):
    """The SM-count dependent sizes of the launches: spare CTAs of the basis launch, whether the eigensolver CTAs
    wait for the V scan, and the item and CTA counts of the Q/K pass, the V pass and the encode launch."""
    BH = B * H
    T = (S + 127) // 128
    nspare = max(nsm - BH, 16)
    eigwait = 1 if BH + nspare <= nsm else 0
    nqk_items = 2 * BH * T
    nqk_cta = min(nqk_items, nsm)
    nv_items = BH * T
    nv_cta = min(nv_items, 2 * nsm)
    nenc_cta = min(
        max((2 * BH * T * QK_CHANNELS + ENCODE_THREADS - 1) // ENCODE_THREADS, 1), 8 * nsm
    )
    return nspare, eigwait, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta


# the buffers of a call whose addresses the host function takes after q, k, v, the rows and the rotation, in its order
_HOST_BUFFERS = (
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "score_scale",
    "v8",
    "v_mean",
    "v8_scale",
    "ut",
    "vmax",
    "gmax",
    "prov",
    "done",
)


def scratch(B, S, H, nsm):
    """The scratch of the chain at this shape on a device of nsm SMs, name -> (dtype, shape): the bases U^T, the
    V-scan slots, the group maxima and the provisional block scales, which every call writes before it reads them,
    and the two counters of the basis launch (the V-scan barrier and the V-pass tile counter), which must be zero when
    it starts."""
    nspare, _, _, nqk_cta, _, _, _ = _grids(nsm, B, S, H)
    BH, T = B * H, (S + 127) // 128
    return {
        "ut": (torch.bfloat16, (BH * DEFAULT_RANK * HEAD_DIM,)),
        "vmax": (torch.float32, (nspare * 2,)),
        "gmax": (torch.float32, (nqk_cta * 8,)),
        "prov": (torch.bfloat16, (2 * BH * T * 1536,)),
        "done": (torch.int32, (2,)),
    }


def run(q, k, v, qk_rows, buffers):
    """The operands of the attention kernel (buffers: the buffers of the call, the layout of this package) from raw
    BF16 q, k, v [B, S, H, 128]: Q'/K' (q4, k4, q_sf, k_sf), score_scale (floored at 2^MIN_SCORE_LOG2) and the
    smoothed E4M3 V (v8, v_mean, v8_scale). qk_rows: the sampled tokens of the Q/K basis (inputs.sample_rows)."""
    B, S, H, _ = q.shape
    T = (S + 127) // 128
    dev = q.device
    nsm = num_sms(dev)
    nspare, eigwait, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta = _grids(nsm, B, S, H)
    # the V pass shares the basis launch when the grid has room for it
    fn = _compile(dev, 4 * B * H <= nsm)
    fn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        qk_rows.data_ptr(),
        hadamard(dev).data_ptr(),
        *(buffers[n].data_ptr() for n in _HOST_BUFFERS),
        B,
        S,
        H,
        qk_rows.shape[0],
        T,
        MIN_SCORE_LOG2,
        nspare,
        eigwait,
        nqk_items,
        nqk_cta,
        nv_items,
        nv_cta,
        nenc_cta,
        current_stream(dev),
    )
