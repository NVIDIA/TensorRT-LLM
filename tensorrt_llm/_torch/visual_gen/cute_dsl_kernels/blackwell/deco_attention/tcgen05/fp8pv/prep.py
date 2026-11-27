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

One TVM-FFI host function per device launches the kernel chain on the current stream (_make_host); its scratch is
part of the buffers of the call (scratch). Nothing reads a device value back, so the chain captures into a CUDA graph.

    clear launch        zeroes the per-call words of gmax                                    basis.clear_kernel
    eigensolver launch  one CTA per (batch, head): the compensation basis U of Q' and K'     basis.eig_kernel
    V scan              128-key means of V (v_mean) and their largest deviation, beside the   v_pass.v_scan_kernel
                        eigensolver
    V encode            E4M3 V minus its means (v8) and the scaled means (v_mean)            v_pass.v_encode_kernel
    Q/K pass            Hadamard rotation, NVFP4 codes and compensation channels (q4, k4)    qk_pass.qk_kernel
    encode launch       UE4M3 scale factors (q_sf, k_sf), score_scale and v8_scale           encode.encode_kernel

The first three are chained by programmatic dependent launch: the eigensolver waits for the clear launch, and the V
scan starts beside the eigensolver CTAs and ends with a wait for them, so the plain launches after it see the basis.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ...constants import HEAD_DIM, key_tiles, qk_scale_atoms
from ...inputs import hadamard
from ...runtime import Layout, num_sms, pdl_enabled, token_pitches
from ..prep.basis import EIG_THREADS, GMAX_WORDS, clear_kernel, eig_kernel
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
    sizes: tuple,
    grids: tuple,
    lens: Int64,
    stream: cuda.CUstream,
    rank: int,
    rows_mode: tuple | None,
    pdl: bool,
) -> None:
    """Issue the launches of the chain; rows_mode: None, or (rows per sample, mapped rows) of a call with valid prefix
    lengths [B] at the address lens; pdl: the V scan and the eigensolver launch are programmatic dependents."""
    q, k, v, rows, hadamard = inputs
    q4, k4, q_sf, k_sf, score_scale, v8, v_mean, v8_scale = operands
    gmax, uT, q_sfv, k_sfv = scratch
    S, H, QP, KP, VP, NQK, NCQ, BH, NB = sizes
    v_units, v_grid, qgrid, egrid = grids
    ntiles = BH * NB * 2
    min_prod = HEAD_DIM * 2.0**MIN_SCORE_LOG2
    clear_kernel(gmax).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    with_lens = rows_mode is not None
    eig = eig_kernel(q, k, rows, hadamard, uT, S, H, QP, KP, NQK, NCQ, lens, rank, rows_mode)
    vs = v_scan_kernel(v, v_mean, gmax, S, H, VP, NB, v_units, lens, with_lens)
    ve = v_encode_kernel(v, v8, v_mean, gmax, S, H, VP, NB, v_units, lens, with_lens)
    eig.launch(grid=(BH, 1, 1), block=(EIG_THREADS, 1, 1), use_pdl=pdl, stream=stream)
    vs.launch(
        grid=(v_grid, 1, 1),
        block=(V_THREADS, 1, 1),
        min_blocks_per_mp=V_CTAS_PER_SM,
        use_pdl=pdl,
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
    qk_args = (tiled_mma, tma_q, tma_k, hadamard, uT, q4, k4, q_sfv, k_sfv, gmax)
    qkp = qk_kernel(*qk_args, S, H, NB, ntiles, lens, rank, with_lens)
    qkp.launch(grid=(qgrid, 1, 1), block=(QK_THREADS, 1, 1), stream=stream)
    encode_kernel(
        q_sfv, k_sfv, q_sf, k_sf, gmax, score_scale, BH * NB, v8_scale, min_prod, rank
    ).launch(grid=(egrid, 1, 1), block=(encode_threads(rank), 1, 1), stream=stream)


def _make_host(
    rank: int, rows_per_sample: bool | None = None, map_rows: bool = False, pdl: bool = True
) -> Callable:
    """Return the host function that launches the chain of compensation rank `rank` on its `stream` argument.

    * The score prescale log2(e) / sqrt(128) is folded into Q' (QPRE), so the multiplier of the raw scores in log2
      units is the score scale (group scale product / 128) itself, a power of two; the encode launch raises it to at
      least 2^MIN_SCORE_LOG2, the floor of the attention's landing (softmax.py), by shrinking the Q' / K' block scales.
    * rows_per_sample: None for a call whose batch elements are all full (the host function's lens is not read);
      otherwise lens is the address of the valid prefix lengths [B], with rows [B, NQK] (True) or [NQK] (False).
      map_rows: the eigensolver maps the rows into each valid prefix (the default draw).
    * pdl: the launches of the chain follow each other as programmatic dependents.
    * QP, KP, VP: the bytes between consecutive tokens of q, k and v, which the eigensolver and the V scan and encode
      read in place (the Q/K pass reads them through TMA descriptors built on the tensors' layouts).
    """
    rows_mode = None if rows_per_sample is None else (rows_per_sample, map_rows)

    @cute.jit
    def deco_host(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        rows: cute.Tensor,
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
        uT: cute.Tensor,
        q_sfv: cute.Tensor,
        k_sfv: cute.Tensor,
        S: Int32,
        H: Int32,
        QP: Int32,
        KP: Int32,
        VP: Int32,
        NQK: Int32,
        NCQ: Int32,
        BH: Int32,
        NB: Int32,
        v_units: Int32,
        v_grid: Int32,
        qgrid: Int32,
        egrid: Int32,
        lens: Int64,
        stream: cuda.CUstream,
    ):
        _launch_chain(
            (q, k, v, rows, hadamard),
            (q4, k4, q_sf, k_sf, score_scale, v8, v_mean, v8_scale),
            (gmax, uT, q_sfv, k_sfv),
            (S, H, QP, KP, VP, NQK, NCQ, BH, NB),
            (v_units, v_grid, qgrid, egrid),
            lens,
            stream,
            rank,
            rows_mode,
            pdl,
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


# the buffers of a call that the host function takes after q, k, v, the rows and the rotation, in its order
_HOST_BUFFERS = (
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "score_scale",
    "v8",
    "v_mean",
    "v8_scale",
    "gmax",
    "uT",
    "q_sfv",
    "k_sfv",
)


def scratch(B: int, S: int, H: int, rank: int) -> Layout:
    """Return the scratch of the chain at this shape and rank, name -> (dtype, shape); every call rewrites it first."""
    NB = key_tiles(S)
    atoms = qk_scale_atoms(rank)
    return {
        # the per-call words (maxima and the V pass's CTA count), U^T and the provisional Q' / K' block scales
        "gmax": (torch.float32, (GMAX_WORDS,)),
        "uT": (torch.float32, (B, H, rank, HEAD_DIM)),
        "q_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
        "k_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
    }


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    rank: int,
    lens: torch.Tensor | None = None,
    map_rows: bool = False,
) -> None:
    """Fill the operands of a call (buffers: the buffers of the call, the layout of this package) from raw BF16 q, k, v.

    * Operands: see layout. score_scale is the multiplier of the raw scores in log2 units, a power of two of at
      least 2^MIN_SCORE_LOG2 (_make_host); v8 and v_mean (row j: the mean of block j) are over v8_scale (v_pass.py).
    * rows: int32 indices of the tokens whose q and k rows estimate the compensation basis (inputs.sample_rows).
    * lens: None, or the int32 valid prefix lengths [B] on the device, which the kernels read in place. rows is
      then [NQK] for every batch element or [B, NQK] per batch element: positions in the padded sequence that the
      eigensolver maps into each valid prefix with map_rows, tokens otherwise.
    """
    B, S, H, _ = q.shape
    NB = key_tiles(S)
    NQK = rows.shape[-1]
    rows_per_sample = None if lens is None else rows.dim() == 2
    dev = q.device
    nsm = num_sms(dev)
    BH = B * H
    # The V scan runs beside the eigensolver CTAs, V_CTAS_PER_SM per SM. A pass of at most V_SMALL_PASS_BYTES per
    # free SM is launched on the SMs that the eigensolver launch leaves free (at least V_SMALL_PASS_MIN_SMS, at most
    # the device's); a larger one on every SM.
    free = min(nsm, max(nsm - BH, V_SMALL_PASS_MIN_SMS))
    v_sms = free if BH < nsm and B * S * H * 16 * 40 <= V_SMALL_PASS_BYTES * free else nsm
    v_units = B * ((S + V_BLOCK - 1) // V_BLOCK)
    v_grid = min(V_CTAS_PER_SM * v_sms, v_units)
    ntiles = BH * NB * 2
    qgrid = min(nsm, ntiles)
    egrid = min(ENCODE_CTAS_PER_SM * nsm, BH * NB)
    tensors = (q, k, v, rows, hadamard(dev), *(buffers[n] for n in _HOST_BUFFERS))
    sizes = (S, H, *token_pitches(q, k, v), NQK, (NQK + 255) // 256, BH, NB)
    sizes += (v_units, v_grid, qgrid, egrid)
    lens_addr = 0 if lens is None else lens.data_ptr()
    map_rows = lens is not None and map_rows
    pdl = pdl_enabled()
    key = (dev.index, rank, rows_per_sample, map_rows, pdl)
    fn = _hosts.get(key)
    if fn is None:
        cargs = [_ffi_tensor(t) for t in tensors] + [Int32(x) for x in sizes] + [Int64(lens_addr)]
        cargs.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
        fn = _hosts[key] = cute.compile(
            _make_host(rank, rows_per_sample, map_rows, pdl), *cargs, options="--enable-tvm-ffi"
        )
    fn(*tensors, *sizes, lens_addr)
