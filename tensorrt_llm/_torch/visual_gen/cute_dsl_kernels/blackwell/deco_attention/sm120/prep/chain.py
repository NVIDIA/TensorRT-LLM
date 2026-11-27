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
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ...constants import HEAD_DIM, QK_CHANNELS, key_tiles
from ...inputs import Compensation, Transforms, hadamard
from ...runtime import Basis, Layout, compiled, current_stream, num_sms, token_pitches
from .basis import EIG_THREADS, TRANSFORM_BYTES, basis_plan
from .encode import ENCODE_THREADS, encode_kernel
from .qk_pass import QK_THREADS, TILE_SCALES, qk_kernel

# the buffers of a stored basis (scratch): that of the Q-residual term, which is the one basis of bases "shared", and
# that of the K-residual term
_UT = ("ut_q", "ut_k")


def basis_rows(rank: tuple[int, int], separate: bool) -> tuple[int, ...]:
    """Return the rows of the tensors of a stored basis (runtime.Basis): one row per kept direction, the leading
    direction last.

    * Bases "shared": one tensor of the max(r_q, r_k) leading directions, which serves both terms with its leading
      r_q and r_k.
    * Bases "separate": two tensors, of r_q and of r_k directions.
    """
    return rank if separate else (max(rank),)


def basis(buffers: dict[str, torch.Tensor], bases: str) -> Basis:
    """Return the basis that the eigenproblems of a call left in its buffers (scratch): the one tensor of bases
    "shared", or the two of "separate" (basis of the Q-residual term, basis of the K-residual term)."""
    return tuple(buffers[name] for name in _UT[: 2 if bases == "separate" else 1])


def _compile(
    host: Callable,
    cache: dict,
    device: torch.device,
    fused: bool,
    with_lens: bool,
    rank: tuple[int, int],
    separate: bool,
    scale_rule: str,
    transform: tuple[bool, bool],
) -> Callable:
    """Return the host function of a chain compiled for one key, kept in the cache of its kernel package.

    * Key: the device; the chain form (fused: the V pass in the basis launch), which _grids takes from the
      eigenproblems of the call against the SM count, so it follows the shape; with_lens (the kernels read valid
      prefix lengths); rank (the channels of the two terms); separate (each term on a basis of its own);
      scale_rule (the rule for the scale of a head block of Q' and K'); transform (equalize, center_k: the
      transformations of Q and K).
    * A first use of a key compiles (runtime.compiled); the rule of a capture is that of api.py.
    """
    key = (device.index, fused, with_lens, rank, separate, scale_rule, transform)

    def make() -> tuple:
        # the host function, placeholders of its run-time parameters (Int64 addresses, Int32 sizes, Float32
        # weights), then the stream and the switches
        args = [
            Int64(0) if p.annotation is Int64 else p.annotation(1)
            for p in inspect.signature(host).parameters.values()
            if p.annotation in (Int64, Int32, Float32)
        ]
        return (
            host,
            *args,
            cute.runtime.make_fake_stream(),
            fused,
            with_lens,
            rank,
            separate,
            scale_rule,
            transform,
        )

    return compiled(cache, key, make)


def make_host(
    basis_kernel, v_kernel, v_threads: int, n_vout: int, k_image=(0, 0, None)
) -> Callable:
    """Return the host function of the chain of a kernel package.

    * Every launch takes the shape at run time; FUSED, LENS, RANK, SEPARATE, SCALE_RULE and TRANSFORM are the
      trace-time switches of the function, the key of _compile.
    * ut_q_p, ut_k_p: the addresses of the basis of the Q-residual term and of that of the K-residual term
      (basis_rows); with bases "shared" the one address, twice.
    * Basis launch (basis_kernel): the eigenproblems of the compensation basis, neig of them (none when the caller
      gave the basis), on the Gram matrices of the sampled q and k tokens; the other CTAs scan V for max|v - vbar|
      and, when the launch has room for it, run the V pass.
    * V pass (v_kernel, v_threads threads), otherwise.
    * Q'/K' pass: rotation, head and compensation channels in NVFP4 (the scale of a head block by SCALE_RULE),
      provisional block scales, group maxima.
    * Encode launch: group scales, the score-scale floor, score_scale, UE4M3 block scales.
    * TRANSFORM = (equalize, center_k): with a switch the basis launch takes the statistics of the sampled rows,
      solves for the basis of the transformed operands and writes the record of every (batch, head) at tr_p,
      which the Q'/K' pass applies; video_p, video_mode: the video rows of k (basis._eig).
    * The host takes four output addresses of the V pass after score_scale_p; the first n_vout go to the kernels.
      k_image = (codes offset, scales offset, pitch): K' goes to tile images, at these offsets from k4_p and ksf_p
      and that many bytes apart; without a pitch to k4 and k_sf.
    """

    @cute.jit
    def _host(
        q_p: Int64,
        k_p: Int64,
        v_p: Int64,
        rows_q_p: Int64,
        rows_k_p: Int64,
        ut_q_p: Int64,
        ut_k_p: Int64,
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
        vmax_p: Int64,
        gmax_p: Int64,
        prov_p: Int64,
        done_p: Int64,
        tr_p: Int64,
        lens_p: Int64,
        video_p: Int64,
        B: Int32,
        S: Int32,
        H: Int32,
        QP: Int32,
        KP: Int32,
        VP: Int32,
        NQ: Int32,
        NK: Int32,
        T: Int32,
        msl: Int32,
        neig: Int32,
        nspare: Int32,
        nqk_items: Int32,
        nqk_cta: Int32,
        nv_items: Int32,
        nv_cta: Int32,
        nenc_cta: Int32,
        rows_mode: Int32,
        video_mode: Int32,
        weight_q: Float32,
        weight_k: Float32,
        stream: cuda.CUstream,
        FUSED: cutlass.Constexpr,
        LENS: cutlass.Constexpr,
        RANK: cutlass.Constexpr,
        SEPARATE: cutlass.Constexpr,
        SCALE_RULE: cutlass.Constexpr,
        TRANSFORM: cutlass.Constexpr,
    ):
        # LENS: the kernels read the lengths at lens_p in place and take the rows mode; otherwise both are None
        lens_p = lens_p if cutlass.const_expr(LENS) else None
        rows_mode = rows_mode if cutlass.const_expr(LENS) else None
        # TRANSFORM: the kernels take the records and the video rows; otherwise None
        tr_p = tr_p if cutlass.const_expr(any(TRANSFORM)) else None
        video_p = video_p if cutlass.const_expr(any(TRANSFORM)) else None
        video_mode = video_mode if cutlass.const_expr(any(TRANSFORM)) else None
        BH = B * H
        vout = (vo0_p, vo1_p, vo2_p, vo3_p)[:n_vout]
        k_codes, k_scales, k_pitch = k_image
        basis_kernel(
            q_p,
            k_p,
            v_p,
            had_p,
            ut_q_p,
            ut_k_p,
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
            BH,
            nspare,
            rows_q_p,
            rows_k_p,
            NQ,
            NK,
            neig,
            weight_q,
            weight_k,
            FUSED,
            lens_p,
            rows_mode,
            rank=RANK,
            separate=SEPARATE,
            transform=TRANSFORM,
            tr_p=tr_p,
            video_p=video_p,
            video_mode=video_mode,
        ).launch(grid=(neig + nspare, 1, 1), block=(EIG_THREADS, 1, 1), stream=stream)
        if cutlass.const_expr(not FUSED):
            v_kernel(v_p, vmax_p, *vout, S, H, VP, T, nspare, nv_items, nv_cta, lens_p).launch(
                grid=(nv_cta, 1, 1), block=(v_threads, 1, 1), stream=stream
            )
        qk_kernel(
            q_p,
            k_p,
            had_p,
            ut_q_p,
            ut_k_p,
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
            rank=RANK,
            separate=SEPARATE,
            scale_rule=SCALE_RULE,
            transform=TRANSFORM,
            tr_p=tr_p,
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
            rank=RANK,
        ).launch(grid=(nenc_cta, 1, 1), block=(ENCODE_THREADS, 1, 1), stream=stream)

    return _host


class Grids(NamedTuple):
    """The form of a chain and the SM-count dependent sizes of its launches (_grids).

    neig                 the eigenproblems of the call
    fused                the V pass shares the basis launch (the form of the chain)
    nspare               the CTAs of the basis launch beside its eigenproblems
    nqk_items, nqk_cta   the items and the CTAs of the Q/K pass
    nv_items, nv_cta     the items and the CTAs of the V pass
    nenc_cta             the CTAs of the encode launch
    """

    neig: int
    fused: bool
    nspare: int
    nqk_items: int
    nqk_cta: int
    nv_items: int
    nv_cta: int
    nenc_cta: int


def _grids(B: int, S: int, H: int, nsm: int, rank: tuple[int, int], separate: bool) -> Grids:
    """Return the form and the launch sizes of a chain at a shape on a device of nsm SMs.

    * The eigenproblems: those of basis_plan per (batch, head).
    * The V pass shares the basis launch when the eigenproblems take at most a quarter of the SMs and leave at
      least 16 of them. The spare CTAs are then the SMs that the problems leave: the grid of the launch is the
      SM count. Its scan claims tiles from a counter and waits for tiles, not for CTAs.
    * Otherwise the spare CTAs, which scan V, are the SMs that one problem per (batch, head) leaves, at least 16: a
      second problem per (batch, head) takes none of them.
    """
    BH = B * H
    T = key_tiles(S)
    neig = basis_plan(rank, separate)[0] * BH
    fused = 4 * neig <= nsm and nsm - neig >= 16
    nspare = nsm - neig if fused else max(nsm - min(neig, BH), 16)
    nqk_items = 2 * BH * T
    nqk_cta = min(nqk_items, nsm)
    nv_items = BH * T
    nv_cta = min(nv_items, 2 * nsm)
    nenc_cta = min(
        max((2 * BH * T * QK_CHANNELS + ENCODE_THREADS - 1) // ENCODE_THREADS, 1), 8 * nsm
    )
    return Grids(neig, fused, nspare, nqk_items, nqk_cta, nv_items, nv_cta, nenc_cta)


def scratch(B: int, S: int, H: int, nsm: int, rank: tuple[int, int], bases: str) -> Layout:
    """Return the scratch of a chain at this shape on a device of nsm SMs, name -> (dtype, shape), for the rank
    (r_q, r_k) of the compensation terms and a bases kind.

    * ut_q, ut_k, vmax, gmax, prov: the tensors of the basis (basis_rows; ut_k with bases "separate" only),
      the V-scan slots, the group maxima, the provisional block scales.
    * Only done, the four words of the basis launch (V-scan tiles published, V-pass tile counter, V-scan tile claim
      counter, V-scan maximum), must be zero at its start.
    """
    separate = bases == "separate"
    grids = _grids(B, S, H, nsm, rank, separate)
    BH, T = B * H, key_tiles(S)
    return {
        **{
            name: (torch.float32, (B, H, rows, HEAD_DIM))
            for name, rows in zip(_UT, basis_rows(rank, separate))
        },
        "vmax": (torch.float32, (grids.nspare,)),
        "gmax": (torch.float32, (grids.nqk_cta * 8,)),
        "prov": (torch.bfloat16, (2 * BH * T * TILE_SCALES,)),
        "done": (torch.int32, (4,)),
    }


def transform_record(B: int, H: int, rank: tuple[int, int]) -> Layout:
    """Return the buffer of the records of the transformations of Q and K (basis.TRANSFORM_*), the last of the
    layout of a kernel package: one record per (batch, head) for a rank with compensation channels, which the
    basis launch writes and the Q'/K' pass reads when a transformation is on."""
    return {"transform": (torch.float32, (B, H, TRANSFORM_BYTES // 4 if max(rank) else 0))}


def run(
    host: Callable,
    cache: dict,
    host_buffers: tuple[str, ...],
    min_score_log2: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None,
    comp: Compensation,
    scale_rule: str,
    transforms: Transforms,
) -> None:
    """Run the launches of a chain on the current stream: raw BF16 q, k, v [B, S, H, 128] into the buffers of the call.

    * q, k, v: contiguous, or views of a fused projection; the kernels read their tokens at the pitch of each tensor.

    * host, cache, host_buffers: the host function of the kernel package (make_host), its compile cache and the
      buffers whose addresses it takes after q, k, v, the rows, the basis addresses and the rotation, in its order
      (None: no buffer).
    * min_score_log2: log2 of the score-scale floor.
    * lens: None, or the int32 valid prefix lengths [B] on the device, which the kernels read in place: every pass
      then takes the tokens of a batch element below its length only.
    * comp: the compensation of the call (inputs.Compensation).
    * scale_rule: the rule for the scale of an NVFP4 block of the head channels of Q' and K', one of
      constants.SCALE_RULES; a block of the compensation channels takes that of "max" under every rule.
    * transforms: the transformations of Q and K (inputs.Transforms); its video rows are read in place.
    """
    B, S, H, _ = q.shape
    T = key_tiles(S)
    dev = q.device
    nsm = num_sms(dev)
    separate = comp.bases == "separate"
    grids = _grids(B, S, H, nsm, comp.rank, separate)
    transform = (transforms.equalize, transforms.center_k)
    video = transforms.video_rows
    fn = _compile(
        host, cache, dev, grids.fused, lens is not None, comp.rank, separate, scale_rule, transform
    )
    neig, rows, ut = grids.neig, (comp.rows_q, comp.rows_k), basis(buffers, comp.bases)
    if comp.basis is not None:
        neig, rows, ut = 0, (None, None), comp.basis
    fn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        *(r.data_ptr() if r is not None else 0 for r in rows),
        # the one tensor of a basis of bases "shared" serves both terms
        ut[0].data_ptr(),
        ut[-1].data_ptr(),
        hadamard(dev).data_ptr(),
        *(buffers[n].data_ptr() if n else 0 for n in host_buffers),
        lens.data_ptr() if lens is not None else 0,
        video.data_ptr() if video is not None else 0,
        B,
        S,
        H,
        *token_pitches(q, k, v),
        *(r.shape[-1] if r is not None else 0 for r in rows),
        T,
        min_score_log2,
        neig,
        grids.nspare,
        grids.nqk_items,
        grids.nqk_cta,
        grids.nv_items,
        grids.nv_cta,
        grids.nenc_cta,
        # the rows mode of the eigensolver, once per operand (q in bits 0..1, k in bits 2..3): 0 the mapped
        # default draw, 1 the caller's shared rows, 2 its rows per element
        0 if comp.map_rows or neig == 0 else rows[0].dim() | rows[1].dim() << 2,
        # the video rows of k: 0 every row, 1 one count for the batch, 2 one per batch element
        0 if video is None else 1 if video.numel() == 1 else 2,
        # weight_q and weight_k of one problem per (batch, head): 1 and 1 at gram_weight 0.5 (the two problems of
        # bases "separate" each take one Gram matrix alone)
        2.0 * comp.gram_weight,
        2.0 - 2.0 * comp.gram_weight,
        current_stream(dev),
    )
