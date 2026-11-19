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
chain form (the host side that both kernel packages share is ../prep/chain.py).

Launches, on the current stream: the basis launch (prep/basis.py with the V scan of v_pass.py: per (batch, head) the
Gram matrices of the sampled q and k tokens and their top-32 eigenvectors; the other CTAs scan V for max|v - vbar|
and, when the launch has room for it, run the V pass), the V pass otherwise (v_pass.v_kernel), the Q'/K' pass
(qk_pass.py: rotation, body and compensation channels in NVFP4, provisional block scales, group maxima) and the
encode launch (encode.py: group scales, the score-scale floor, score_scale, UE4M3 block scales). Every launch takes
the shape at run time, so each of the two forms of the chain (the V pass in the basis launch or on its own) compiles
once.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from ..prep import chain
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
    """The scratch of the chain at this shape on a device of nsm SMs, name -> (dtype, shape) (chain.scratch), with
    V-scan slots of two words."""
    return chain.scratch(B, S, H, nsm, 2)


def run(q, k, v, qk_rows, buffers):
    """The operands of the attention kernel (buffers: the buffers of the call, the layout of this package) from raw
    BF16 q, k, v [B, S, H, 128]: Q'/K' (q4, k4, q_sf, k_sf), score_scale (floored at 2^MIN_SCORE_LOG2) and the
    smoothed E4M3 V (v8, v_mean, v8_scale). qk_rows: the sampled tokens of the Q/K basis (inputs.sample_rows)."""
    chain.run(_host, _compiled, _HOST_BUFFERS, MIN_SCORE_LOG2, q, k, v, qk_rows, buffers)
