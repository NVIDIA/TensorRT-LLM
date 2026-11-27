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
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ...constants import DEFAULT_RANK, HEAD_DIM, QK_CHANNELS, key_tiles
from ...inputs import hadamard
from ...runtime import current_stream, num_sms, token_pitches
from .basis import EIG_THREADS
from .encode import ENCODE_THREADS, encode_kernel
from .qk_pass import QK_THREADS, TILE_SCALES, qk_kernel


def _compile(
    host: Callable, cache: dict, device: torch.device, fused: bool, with_lens: bool
) -> Callable:
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


def make_host(
    basis_kernel, v_kernel, v_threads: int, n_vout: int, k_image=(0, 0, None)
) -> Callable:
    """Return the host function of the chain of a kernel package; every launch takes the shape at run time, so each of
    its two forms (FUSED: the V pass in the basis launch) compiles once per device and presence of lengths.

    * Basis launch (basis_kernel): per (batch, head) the Gram matrices of the sampled q and k tokens and their top-32
      eigenvectors; the other CTAs scan V for max|v - vbar| and, when the launch has room for it, run the V pass.
    * V pass (v_kernel, v_threads threads), otherwise.
    * Q'/K' pass: rotation, body and compensation channels in NVFP4, provisional block scales, group maxima.
    * Encode launch: group scales, the score-scale floor, score_scale, UE4M3 block scales.
    * The host takes four output addresses of the V pass after score_scale_p; the first n_vout go to the kernels.
      k_image = (codes offset, scales offset, pitch): K' goes to tile images, at these offsets from k4_p and ksf_p
      and that many bytes apart; without a pitch to k4 and k_sf.
    """

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
        vo0_p: Int64,
        vo1_p: Int64,
        vo2_p: Int64,
        vo3_p: Int64,
        ut_p: Int64,
        vmax_p: Int64,
        gmax_p: Int64,
        prov_p: Int64,
        done_p: Int64,
        lens_p: Int64,
        B: Int32,
        S: Int32,
        H: Int32,
        QP: Int32,
        KP: Int32,
        VP: Int32,
        NQK: Int32,
        T: Int32,
        msl: Int32,
        nspare: Int32,
        nqk_items: Int32,
        nqk_cta: Int32,
        nv_items: Int32,
        nv_cta: Int32,
        nenc_cta: Int32,
        rows_mode: Int32,
        stream: cuda.CUstream,
        FUSED: cutlass.Constexpr,
        LENS: cutlass.Constexpr,
    ):
        # LENS: the kernels read the lengths at lens_p in place and take the rows mode; otherwise both are None
        lens_p = lens_p if cutlass.const_expr(LENS) else None
        rows_mode = rows_mode if cutlass.const_expr(LENS) else None
        BH = B * H
        vout = (vo0_p, vo1_p, vo2_p, vo3_p)[:n_vout]
        k_codes, k_scales, k_pitch = k_image
        basis_kernel(
            q_p,
            k_p,
            v_p,
            rows_p,
            had_p,
            ut_p,
            vmax_p,
            done_p,
            *vout,
            B,
            S,
            H,
            QP,
            KP,
            VP,
            T,
            NQK,
            BH,
            nspare,
            FUSED,
            lens_p,
            rows_mode,
        ).launch(grid=(BH + nspare, 1, 1), block=(EIG_THREADS, 1, 1), stream=stream)
        if cutlass.const_expr(not FUSED):
            v_kernel(v_p, vmax_p, *vout, S, H, VP, T, nspare, nv_items, nv_cta, lens_p).launch(
                grid=(nv_cta, 1, 1), block=(v_threads, 1, 1), stream=stream
            )
        qk_kernel(
            q_p,
            k_p,
            had_p,
            ut_p,
            q4_p,
            k4_p + k_codes,
            prov_p,
            gmax_p,
            B,
            S,
            H,
            QP,
            KP,
            T,
            nqk_items,
            nqk_cta,
            lens_p,
            tile_pitch=k_pitch,
        ).launch(grid=(nqk_cta, 1, 1), block=(QK_THREADS, 1, 1), stream=stream)
        encode_kernel(
            prov_p,
            gmax_p,
            qsf_p,
            ksf_p + k_scales,
            score_scale_p,
            B,
            H,
            T,
            nqk_cta,
            msl,
            nenc_cta,
            tile_pitch=k_pitch,
        ).launch(grid=(nenc_cta, 1, 1), block=(ENCODE_THREADS, 1, 1), stream=stream)

    return _host


def _grids(B: int, S: int, H: int, nsm: int) -> tuple[int, int, int, int, int, int]:
    """Return the SM-count dependent sizes of the launches: spare CTAs of the basis launch, and the item and CTA counts
    of the Q/K pass, the V pass and the encode launch."""
    BH = B * H
    T = key_tiles(S)
    nspare = max(nsm - BH, 16)
    nqk_items = 2 * BH * T
    nqk_cta = min(nqk_items, nsm)
    nv_items = BH * T
    nv_cta = min(nv_items, 2 * nsm)
    nenc_cta = min(
        max((2 * BH * T * QK_CHANNELS + ENCODE_THREADS - 1) // ENCODE_THREADS, 1), 8 * nsm
    )
    return nspare, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta


def scratch(B: int, S: int, H: int, nsm: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Return the scratch of a chain at this shape on a device of nsm SMs, name -> (dtype, shape).

    * ut, vmax, gmax, prov: the bases U^T, the V-scan slots, the group maxima, the provisional block scales.
    * Only done, the four words of the basis launch (V-scan tiles published, V-pass tile counter, V-scan tile claim
      counter, V-scan maximum), must be zero at its start.
    """
    nspare, _, nqk_cta, _, _, _ = _grids(B, S, H, nsm)
    BH, T = B * H, key_tiles(S)
    return {
        "ut": (torch.bfloat16, (BH * DEFAULT_RANK * HEAD_DIM,)),
        "vmax": (torch.float32, (nspare,)),
        "gmax": (torch.float32, (nqk_cta * 8,)),
        "prov": (torch.bfloat16, (2 * BH * T * TILE_SCALES,)),
        "done": (torch.int32, (4,)),
    }


def run(
    host: Callable,
    cache: dict,
    host_buffers: tuple[str, ...],
    min_score_log2: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None = None,
    map_rows: bool = False,
) -> None:
    """Run the launches of a chain on the current stream: raw BF16 q, k, v [B, S, H, 128] into the buffers of the call.

    * q, k, v: contiguous, or views of a fused projection; the kernels read their tokens at the pitch of each tensor.

    * host, cache, host_buffers: the host function of the kernel package (make_host), its compile cache and the
      buffers whose addresses it takes after q, k, v, the rows and the rotation, in its order (None: no buffer).
    * min_score_log2: log2 of the score-scale floor; rows: the sampled tokens of the Q/K basis (inputs.sample_rows).
    * lens: None, or the int32 valid prefix lengths [B] on the device, which the kernels read in place: every pass
      then takes the tokens of a batch element below its length only.
    * rows with lens: with map_rows the default draw over the padded sequence, which the eigensolver maps into each
      valid prefix; else the caller's rows [n] or [B, n] as given (a row at or past a length: the last valid token).
    """
    B, S, H, _ = q.shape
    T = key_tiles(S)
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
        *(buffers[n].data_ptr() if n else 0 for n in host_buffers),
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
