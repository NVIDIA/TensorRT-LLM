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
"""Host entry of the preparation of the tcgen05 FP8-PV kernel: run turns raw BF16 q, k, v into its operands.

One TVM-FFI host function per device and configuration launches the kernel chain on the current stream (_make_host,
which lists what a configuration is); its scratch is part of the buffers of the call (scratch). Nothing reads a
device value back, so the chain captures into a CUDA graph; the first call of a configuration compiles its host
function (the rule of a capture: api.py).

    clear launch        zeroes the per-call words of gmax                                    basis.clear_kernel
    eigensolver launch  one CTA per basis of a (batch, head): the directions of the          basis.eig_kernel
                        compensation terms; not launched with a caller's basis or rank (0, 0)
    V scan              128-key means of V (v_mean) and their largest deviation, beside the   v_pass.v_scan_kernel
                        eigensolver
    V encode            E4M3 V minus its means (v8) and the scaled means (v_mean)            v_pass.v_encode_kernel
    Q/K pass            Hadamard rotation, NVFP4 codes and compensation channels (q4, k4)    qk_pass.qk_kernel
    encode launch       UE4M3 scale factors (q_sf, k_sf), score_scale and v8_scale           encode.encode_kernel

The first three are chained by programmatic dependent launch: the eigensolver waits for the clear launch, and the V
scan starts beside the eigensolver CTAs and ends with a wait for them, so the plain launches after it see the basis.
Without an eigensolver launch the V scan is a plain launch after the clear launch.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ...constants import HEAD_DIM, KEY_TILE, key_tiles, qk_channels, qk_scale_atoms
from ...inputs import Compensation, hadamard
from ...runtime import Basis, Layout, compiled, num_sms, pdl_enabled, token_pitches
from ..prep.basis import EIG_THREADS, GMAX_WORDS, clear_kernel, eig_directions, eig_kernel
from ..prep.encode import encode_kernel, encode_threads
from ..prep.qk_pass import QK_THREADS, X_STAGES, qk_kernel
from .softmax import MIN_SCORE_LOG2
from .v_pass import V_BLOCK, V_CTAS_PER_SM, V_THREADS, v_encode_kernel, v_scan_kernel


def _gemm_view(t: cute.Tensor) -> cute.Tensor:
    """Return the view (S, D, (H, B)) of a tensor [B, S, H, D]."""
    return cute.make_tensor(
        t.iterator, cute.group_modes(cute.select(t.layout, mode=[1, 3, 2, 0]), 2, 4)
    )


def _launch_chain(
    inputs: tuple,
    operands: tuple,
    scratch: tuple,
    pitches: tuple,
    grids: tuple,
    gram_weight: Float32,
    lens: Int64,
    stream: cuda.CUstream,
    config: tuple,
) -> None:
    """Issue the launches of the chain of a configuration (_make_host), on the tensors and sizes of a call."""
    q, k, v, rows_q, rows_k, NK, hadamard = inputs
    q4, k4, q_sf, k_sf, score_scale, v8, v_mean, v8_scale = operands
    gmax, ut_q, ut_k, q_sfv, k_sfv = scratch
    QP, KP, VP = pitches
    v_grid, qgrid, egrid, v_counted = grids
    rank, separate, eig, two_row_sets, rows_mode, pdl = config
    # the shape of the call, and the sizes of its launches that follow from it
    B, S, H = (Int32(n) for n in q.shape[:3])
    BH = B * H
    NB = (S + (KEY_TILE - 1)) // KEY_TILE
    v_units = B * ((S + (V_BLOCK - 1)) // V_BLOCK)
    ntiles = BH * NB * 2
    min_prod = HEAD_DIM * 2.0**MIN_SCORE_LOG2
    with_lens = rows_mode is not None
    clear_kernel(gmax).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    if eig:
        NQ = Int32(rows_q.shape[-1])
        eig_args = (q, k, rows_q, rows_k, hadamard, ut_q, S, H, QP, KP, NQ, NK, gram_weight, lens)
        eig_kernel(*eig_args, eig_directions(rank), rows_mode, two_row_sets, separate).launch(
            grid=(BH * (2 if separate else 1), 1, 1),
            block=(EIG_THREADS, 1, 1),
            use_pdl=pdl,
            stream=stream,
        )
    vs = v_scan_kernel(v, v_mean, gmax, S, H, VP, NB, v_units, v_counted, lens, with_lens)
    ve = v_encode_kernel(v, v8, v_mean, gmax, S, H, VP, NB, v_units, lens, with_lens)
    vs.launch(
        grid=(v_grid, 1, 1),
        block=(V_THREADS, 1, 1),
        min_blocks_per_mp=V_CTAS_PER_SM,
        use_pdl=pdl and eig,
        stream=stream,
    )
    ve.launch(
        grid=(v_grid, 1, 1), block=(V_THREADS, 1, 1), min_blocks_per_mp=V_CTAS_PER_SM, stream=stream
    )
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        Float32,
        tcgen05.CtaGroup.ONE,
        (128, 128),
        tcgen05.OperandSource.SMEM,
    )
    a_layout = sm100_utils.make_smem_layout_a(tiled_mma, (128, 128, 128), BFloat16, X_STAGES)
    op = cpasync.CopyBulkTensorTileG2SOp()
    tma_q = cute.nvgpu.make_tiled_tma_atom_A(
        op, _gemm_view(q), a_layout, (128, 128, 128), tiled_mma
    )
    tma_k = cute.nvgpu.make_tiled_tma_atom_A(
        op, _gemm_view(k), a_layout, (128, 128, 128), tiled_mma
    )
    qk_args = (tiled_mma, tma_q, tma_k, hadamard, ut_q, ut_k, q4, k4, q_sfv, k_sfv, gmax)
    qkp = qk_kernel(*qk_args, S, H, NB, ntiles, lens, rank, with_lens, separate)
    qkp.launch(grid=(qgrid, 1, 1), block=(QK_THREADS, 1, 1), stream=stream)
    encode_kernel(
        q_sfv, k_sfv, q_sf, k_sf, gmax, score_scale, BH * NB, v8_scale, min_prod, rank
    ).launch(grid=(egrid, 1, 1), block=(encode_threads(qk_channels(rank)), 1, 1), stream=stream)


def _make_host(config: tuple) -> Callable:
    """Return the host function that launches the chain of a configuration on its `stream` argument.

    * config: (rank, separate, eig, two_row_sets, rows_mode, pdl).
        - rank: (r_q, r_k), the channels of the compensation terms. separate: each term has a basis of its own, ut_q
          (the Q-residual term's) and the one at the address ut_k (else both read ut_q); the eigensolver then writes
          both from the address of ut_q.
        - eig: the eigensolver launch is part of the chain; without it ut_q and ut_k hold the caller's basis or
          are not read (rank (0, 0)), and the rows and the Gram weight are not read.
        - two_row_sets: the Gram matrix of K takes the NK rows at the address rows_k (else both take rows_q).
        - rows_mode: None for a call whose batch elements are all full (the host function's lens is not read);
          otherwise lens is the address of the valid prefix lengths [B], and rows_mode is (rows per sample, mapped
          rows): the row arguments are [B, rows] or [rows], and the eigensolver maps them into each valid prefix
          (the default draw) or takes them as given.
        - pdl: the launches of the chain follow each other as programmatic dependents.
    * The score prescale log2(e) / sqrt(128) is folded into Q' (QPRE), so the multiplier of the raw scores in log2
      units is the score scale (group scale product / 128) itself, a power of two; the encode launch raises it to at
      least 2^MIN_SCORE_LOG2, the floor of the attention's landing (softmax.py), by shrinking the Q' / K' block scales.
    * gram_weight: the Gram weight of the shared basis. QP, KP, VP: the bytes between consecutive tokens of q, k
      and v, which the eigensolver and the V scan and encode read in place (the Q/K pass reads them through TMA
      descriptors built on the tensors' layouts). v_grid, qgrid, egrid: the CTAs of the V passes, the Q/K pass and
      the encode launch. v_counted: nonzero when the V scan shares its SMs with the eigensolver CTAs.
    """

    @cute.jit
    def deco_host(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        rows_q: cute.Tensor,
        hadamard: cute.Tensor,
        q4: cute.Tensor,
        k4: cute.Tensor,
        q_sf: cute.Tensor,
        k_sf: cute.Tensor,
        score_scale: cute.Tensor,
        v8: cute.Tensor,
        v_mean: cute.Tensor,
        v8_scale: cute.Tensor,
        gmax: cute.Tensor,
        ut_q: cute.Tensor,
        q_sfv: cute.Tensor,
        k_sfv: cute.Tensor,
        QP: Int32,
        KP: Int32,
        VP: Int32,
        v_grid: Int32,
        qgrid: Int32,
        egrid: Int32,
        v_counted: Int32,
        NK: Int32,
        gram_weight: Float32,
        rows_k: Int64,
        ut_k: Int64,
        lens: Int64,
        stream: cuda.CUstream,
    ):
        _launch_chain(
            (q, k, v, rows_q, rows_k, NK, hadamard),
            (q4, k4, q_sf, k_sf, score_scale, v8, v_mean, v8_scale),
            (gmax, ut_q, ut_k, q_sfv, k_sfv),
            (QP, KP, VP),
            (v_grid, qgrid, egrid, v_counted),
            gram_weight,
            lens,
            stream,
            config,
        )

    return deco_host


_hosts = {}

# Size of the V pass per free SM up to which it is launched on the SMs that the eigensolver launch leaves free.
V_SMALL_PASS_BYTES = 6.76e6
V_SMALL_PASS_MIN_SMS = 16
ENCODE_CTAS_PER_SM = 4


def _ffi_tensor(t: torch.Tensor) -> cute.Tensor:
    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


# the operands of a call that the host function takes after q, k, v, the rows and the rotation, in its order
_HOST_OPERANDS = ("q4", "k4", "q_sf", "k_sf", "score_scale", "v8", "v_mean", "v8_scale")


def scratch(B: int, S: int, H: int, rank: tuple[int, int], bases: str) -> Layout:
    """Return the scratch of the chain at this shape, rank and bases, name -> (dtype, shape); every call rewrites it
    first, except uT under a caller's basis."""
    NB = key_tiles(S)
    atoms = qk_scale_atoms(qk_channels(rank))
    directions = eig_directions(rank)
    return {
        # the per-call words (maxima and the V scan's unit count), U^T of the basis (one per term with bases
        # "separate"; none without compensation channels) and the provisional Q' / K' block scales
        "gmax": (torch.float32, (GMAX_WORDS,)),
        **(
            {
                "uT": (
                    torch.float32,
                    (*((2,) if bases == "separate" else ()), B, H, directions, HEAD_DIM),
                )
            }
            if directions
            else {}
        ),
        "q_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
        "k_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
    }


def basis(buffers: dict[str, torch.Tensor], bases: str) -> Basis:
    """Return the basis that the eigensolver left in the buffers of a call: U^T [B, H, directions, 128], the leading
    direction last; with bases "separate" the two (basis of the Q-residual term, of the K-residual term)."""
    u = buffers["uT"]
    return (u[0], u[1]) if bases == "separate" else (u,)


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    lens: torch.Tensor | None,
    comp: Compensation,
) -> None:
    """Fill the operands of a call (buffers: the buffers of the call, the layout of this package) from raw BF16 q, k, v.

    * Operands: see layout. score_scale is the multiplier of the raw scores in log2 units, a power of two of at
      least 2^MIN_SCORE_LOG2 (_make_host); v8 and v_mean (row j: the mean of block j) are over v8_scale (v_pass.py).
    * lens: None, or the int32 valid prefix lengths [B] on the device, which the kernels read in place.
    * comp: the compensation of the call (inputs.Compensation). With a caller's basis the eigensolver is not
      launched and the rows are None, as they are with rank (0, 0).
    """
    B, S, H, _ = q.shape
    NB = key_tiles(S)
    dev = q.device
    nsm = num_sms(dev)
    BH = B * H
    rank, bases, gram_weight, rows_q, rows_k, map_rows, caller_basis = comp
    separate = bases == "separate"
    compensated = rank != (0, 0)
    eig = compensated and caller_basis is None
    gmax = buffers["gmax"]
    # the basis of the Q-residual term and the rows of Gq (gmax where a configuration reads neither); the basis of the
    # K-residual term and the rows of Gk go by address
    ut_q, ut_k_addr = gmax, 0
    if compensated:
        u = basis(buffers, bases) if eig else caller_basis
        ut_q, ut_k_addr = u[0], u[-1].data_ptr()
    if not eig:
        rows_q = rows_k = gmax
    two_row_sets = rows_k is not rows_q
    nk, rows_k_addr = (rows_k.shape[-1], rows_k.data_ptr()) if two_row_sets else (0, 0)
    rows_mode = None if lens is None else (eig and rows_q.dim() == 2, eig and map_rows)
    # The V scan runs beside the eigensolver CTAs, V_CTAS_PER_SM per SM. A pass of at most V_SMALL_PASS_BYTES per
    # free SM is launched on the SMs that the eigensolver launch leaves free (at least V_SMALL_PASS_MIN_SMS, at most
    # the device's); a larger one on every SM.
    eig_ctas = BH * (2 if separate else 1) if eig else 0
    free = min(nsm, max(nsm - eig_ctas, V_SMALL_PASS_MIN_SMS))
    v_sms = free if eig_ctas < nsm and B * S * H * 16 * 40 <= V_SMALL_PASS_BYTES * free else nsm
    v_grid = min(V_CTAS_PER_SM * v_sms, B * ((S + V_BLOCK - 1) // V_BLOCK))
    # a V scan on SMs that also hold eigensolver CTAs takes its work from the counter of the call (v_pass.py)
    v_counted = int(eig_ctas > 0 and v_sms + min(eig_ctas, nsm) > nsm)
    qgrid = min(nsm, BH * NB * 2)
    egrid = min(ENCODE_CTAS_PER_SM * nsm, BH * NB)
    tensors = (q, k, v, rows_q, hadamard(dev), *(buffers[n] for n in _HOST_OPERANDS))
    tensors += (gmax, ut_q, buffers["q_sfv"], buffers["k_sfv"])
    sizes = (*token_pitches(q, k, v), v_grid, qgrid, egrid, v_counted, nk)
    lens_addr = 0 if lens is None else lens.data_ptr()
    pdl = pdl_enabled()
    config = (rank, separate, eig, two_row_sets, rows_mode, pdl)
    values = (gram_weight, rows_k_addr, ut_k_addr, lens_addr)

    # the chain compiles per device and configuration (_make_host)
    def make():
        cargs = [_ffi_tensor(t) for t in tensors] + [Int32(x) for x in sizes]
        cargs += [Float32(values[0]), *(Int64(x) for x in values[1:])]
        cargs.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
        return _make_host(config), *cargs

    fn = compiled(_hosts, (dev.index, *config), make, options="--enable-tvm-ffi")
    fn(*tensors, *sizes, *values)
