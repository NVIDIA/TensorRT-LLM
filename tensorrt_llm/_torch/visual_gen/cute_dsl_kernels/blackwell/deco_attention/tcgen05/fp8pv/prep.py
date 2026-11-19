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
"""Host entry of the preparation of the tcgen05 FP8-PV kernel: run turns raw BF16 q, k, v into its operands. One
TVM-FFI host function per device launches the kernel chain on the current stream (_make_host); its scratch is
part of the buffers of the call (scratch). Nothing reads a device value back, so the chain captures into a CUDA
graph.

  clear launch        zeroes the per-call words of gmax                                   basis.clear_kernel
  eigensolver launch  one CTA per (batch, head): the compensation basis U of Q' and K'    basis.eig_kernel
  V pass              E4M3 V minus its 128-key means (v8, v_mean), beside the eigensolver  v_pass.v_kernel
  Q/K pass            Hadamard rotation, NVFP4 codes and compensation channels (q4, k4)   qk_pass.qk_kernel
  encode launch       UE4M3 scale factors (q_sf, k_sf), score_scale and v8_scale          encode.encode_kernel

The first three are chained by programmatic dependent launch: the eigensolver waits for the clear launch, and the V
pass starts beside the eigensolver CTAs and ends with a wait for them, so the plain launches after it see the basis."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ...constants import HEAD_DIM, qk_scale_atoms
from ...inputs import hadamard
from ...runtime import num_sms
from ..prep.basis import EIG_THREADS, GMAX_WORDS, clear_kernel, eig_kernel
from ..prep.encode import encode_kernel, encode_threads
from ..prep.qk_pass import QK_THREADS, X_STAGES, qk_kernel
from .softmax import MIN_SCORE_LOG2
from .v_pass import V_BLOCK, V_CTAS_PER_SM, V_THREADS, v_kernel


def gemm_view(t):
    """[B, S, H, D] -> (S, D, (H, B))."""
    return cute.make_tensor(
        t.iterator, cute.group_modes(cute.select(t.layout, mode=[1, 3, 2, 0]), 2, 4)
    )


def _make_host(rank):
    """The host function launching the chain of compensation rank `rank` on `stream`. The score prescale log2(e) /
    sqrt(128) is folded into Q' (QPRE), so the multiplier of the raw scores in log2 units is the score scale (group
    scale product / 128) itself, a power of two, which the encode launch raises to at least 2^MIN_SCORE_LOG2, the
    floor of the attention's landing (softmax.py), by shrinking the Q' and K' block scales."""
    min_prod = HEAD_DIM * 2.0**MIN_SCORE_LOG2

    @cute.jit
    def deco_host(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        qk_rows: cute.Tensor,
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
        NQK: Int32,
        NCQ: Int32,
        BH: Int32,
        NB: Int32,
        v_units: Int32,
        v_grid: Int32,
        ntiles: Int32,
        qgrid: Int32,
        egrid: Int32,
        stream: cuda.CUstream,
    ):
        clear_kernel(gmax).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        eig_kernel(q, k, qk_rows, hadamard, uT, S, H, NQK, NCQ, rank).launch(
            grid=(BH, 1, 1), block=(EIG_THREADS, 1, 1), use_pdl=True, stream=stream
        )
        v_kernel(v, v8, v_mean, gmax, S, H, NB, v_units).launch(
            grid=(v_grid, 1, 1),
            block=(V_THREADS, 1, 1),
            min_blocks_per_mp=V_CTAS_PER_SM,
            use_pdl=True,
            stream=stream,
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
            op, gemm_view(q), a_layout, (128, 128, 128), tiled_mma
        )
        tma_k = cute.nvgpu.make_tiled_tma_atom_A(
            op, gemm_view(k), a_layout, (128, 128, 128), tiled_mma
        )
        qk_kernel(
            tiled_mma,
            tma_q,
            tma_k,
            hadamard,
            uT,
            q4,
            k4,
            q_sfv,
            k_sfv,
            gmax,
            S,
            H,
            NB,
            ntiles,
            rank,
        ).launch(grid=(qgrid, 1, 1), block=(QK_THREADS, 1, 1), stream=stream)
        encode_kernel(
            q_sfv, k_sfv, q_sf, k_sf, gmax, score_scale, BH * NB, v8_scale, min_prod, rank
        ).launch(grid=(egrid, 1, 1), block=(encode_threads(rank), 1, 1), stream=stream)

    return deco_host


_hosts = {}

# Size of the V pass per free SM up to which it is launched on the SMs that the eigensolver launch leaves free.
V_SMALL_PASS_BYTES = 6.76e6


def _ctf(t):
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


def scratch(B, S, H, rank):
    """The scratch of the chain at this shape and compensation rank, name -> (dtype, shape), which every call
    rewrites before it reads it: the per-call words (maxima and the V pass's CTA count), U^T and the provisional Q' /
    K' block scales."""
    NB = (S + 127) // 128
    atoms = qk_scale_atoms(rank)
    return {
        "gmax": (torch.float32, (GMAX_WORDS,)),
        "uT": (torch.float32, (B, H, rank, HEAD_DIM)),
        "q_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
        "k_sfv": (torch.bfloat16, (B, H, NB, atoms, 512)),
    }


def run(q, k, v, qk_rows, buffers, rank):
    """Fill the operands (buffers: the buffers of the call, the layout of this package) from raw BF16 q, k, v [B, S,
    H, 128]: q4, k4 rows of packed E2M1 (the rotated head channels and the 2 x rank compensation channels), q_sf, k_sf
    their UE4M3 scale factors, v8 E4M3 (V minus the BF16 mean of its block of 128 keys, over v8_scale), v_mean BF16
    (row j: the mean of block j over v8_scale) and the one-element float32 score_scale and v8_scale. qk_rows: int32
    indices of the tokens whose q and k rows estimate the compensation basis (inputs.sample_rows). The score prescale
    log2(e) / sqrt(128) is folded into Q', and score_scale is the multiplier of the raw scores in log2 units, a power
    of two of at least 2^MIN_SCORE_LOG2."""
    B, S, H, _ = q.shape
    NB = (S + 127) // 128
    NQK = qk_rows.shape[0]
    dev = q.device
    nsm = num_sms(dev)
    BH = B * H
    # The V pass runs beside the eigensolver CTAs, V_CTAS_PER_SM per SM. A pass of at most V_SMALL_PASS_BYTES per
    # free SM is launched on the SMs that the eigensolver launch leaves free; a larger one on every SM.
    free = max(nsm - BH, 16)
    v_sms = free if BH < nsm and B * S * H * 16 * 40 <= V_SMALL_PASS_BYTES * free else nsm
    v_units = B * ((S + V_BLOCK - 1) // V_BLOCK)
    v_grid = min(V_CTAS_PER_SM * v_sms, v_units)
    ntiles = BH * NB * 2
    qgrid = min(nsm, ntiles)
    egrid = min(4 * nsm, BH * NB)
    tensors = (q, k, v, qk_rows, hadamard(dev), *(buffers[n] for n in _HOST_BUFFERS))
    sizes = (S, H, NQK, (NQK + 255) // 256, BH, NB, v_units, v_grid, ntiles, qgrid, egrid)
    fn = _hosts.get((dev.index, rank))
    if fn is None:
        cargs = [_ctf(t) for t in tensors] + [Int32(x) for x in sizes]
        cargs.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
        fn = _hosts[dev.index, rank] = cute.compile(
            _make_host(rank), *cargs, options="--enable-tvm-ffi"
        )
    fn(*tensors, *sizes)
