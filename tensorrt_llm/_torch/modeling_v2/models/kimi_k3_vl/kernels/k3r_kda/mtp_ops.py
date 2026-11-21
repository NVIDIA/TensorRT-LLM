# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL host launcher of the Kimi K3 KDA multi-token speculative verify.

``kda_mtp_decode_impl`` launches the source-integrated ``kda_decode_mtp_kernel`` (see
``kernels/k3r_kda/kda_mtp_decode.py``) and its v2 / two-CTA forms; the model calls it through
the ``ops/ssm/kda_mtp_decode`` entry. One launch fuses, per generation
request: replay of previously-accepted draft tokens from the ``kg/v/beta``
caches, causal conv + SiLU, Q/K L2 norm, beta sigmoid, lower-bound gate, and
the KDA delta-rule recurrence over the ``1 + num_spec`` new tokens. The
recurrent state and base conv windows are committed **in place** after the
first new (golden) token; the spec tokens are cached for the next round.

State-management contract (differs from the legacy intermediate-buffer +
``update_mamba_states`` promotion flow): after this op returns, the pools
hold the state as of the last golden token, with the new spec tokens pending
in the replay caches. The next round passes ``num_accepted_tokens`` (how
many of those pending drafts the sampler accepted) and the kernel replays
them before the new tokens. No host-side promotion of KDA SSM/conv state is
required or allowed.

Host-side contract:

* ``zero_accepted_hint`` / ``regular_metadata_hint`` are explicit caller
  arguments, not derived by *reading the device tensors*
  (``torch.count_nonzero(...).item()`` / ``torch.equal``): that is a
  host-device sync per call, and caching it by ``id()`` a stale-cache
  hazard on id reuse. Callers that statically know the pattern
  (benchmarks, the first verify round after prefill) may pass the hints;
  the runtime default (``False``/``False``) is always correct and never
  syncs.
* The compile cache is keyed purely by dtype/shape/stride layouts and
  constexpr flags — no ``id()`` or ``data_ptr`` keys.
* There are no bias / ``pad_slot_id`` arguments (the specialized kernel
  supports neither). Gated output norm and MXFP8 quantization are optional
  packed-layout epilogues.

Kernel shape contract: ``K == V == 128``, conv width ``W == 4``,
``HV == H``, one V pass (TILE_V=128, 512 threads). ``num_spec`` is a
compile-time constant per cache allocation; every shape compiles the same
kernel structure (no setmaxreg / register-weight specializations).

Kernel selection: calls on the runtime contract (packed token rows, fused gated
RMSNorm with a bf16 output, 3xTF32 recurrence, vector-aligned operands) run
``kda_decode_mtp_v2_kernel`` (``kda_mtp_decode_v2.py``: the same arithmetic in
the same order, bitwise identical results, shorter critical path) unless
``_MTP_V2`` is off (tests). On SM107 the same contract with the MXFP8 output runs
``kda_decode_mtp_2cta_kernel`` with its MXFP8 epilogue (one CTA per (head, request)
tile, two CTAs per SM, bitwise identical results). Every other variant runs
``kda_decode_mtp_kernel``, which on SM107 uses its compact two-CTAs-per-SM layout on
multi-wave grids with at most three drafts.

The launcher runs the private kernels in ``kernels/k3r_kda`` and is not a registered torch op
(main's ``custom_ops`` registers its own ``trtllm::kda_mtp_decode`` / ``kda_mtp_decode_fp8`` over
main's kernel, with the replay-only commit the cache manager uses). It reads and writes the replay
caches and the slot-indexed accepted counts main's KDA cache manager allocates, in main's layout.
The switches above are read per call. ``kernels/k3r_kda/_kda_kernels.py`` imports the entry (and
through it this module) on first use; importing it raises ``ImportError`` without CuTe DSL.
"""

import contextlib
from math import gcd
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.logger import logger

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack

    from .kda_mtp_decode import NUM_THREADS, SPLIT_MAX_TILES, T_PAD, TILE_K, kda_decode_mtp_kernel
    from .kda_mtp_decode_2cta import kda_decode_mtp_2cta_kernel
    from .kda_mtp_decode_v2 import kda_decode_mtp_v2_kernel
else:
    raise ImportError("Kimi K3 KDA MTP decode requires NVIDIA CUTLASS DSL")

# CTAs per SM the two-CTA verify kernel is compiled for (64 registers per thread).
_TWO_CTA_MIN_BLOCKS = 2
# CTAs per SM the compact layout of ``kda_decode_mtp_kernel`` is launched for (64 registers per
# thread; see ``COMPACT_TMEM_COLS`` in kda_mtp_decode.py).
_COMPACT_MIN_BLOCKS = 2
# One V pass: the kernel runs 16 warps x 8 rows over the full 128-row state.
_TILE_V = 128


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _run_kda_decode_mtp(
        h0: cute.Tensor,
        x_q: cute.Tensor,
        x_k: cute.Tensor,
        x_v: cute.Tensor,
        w_q: cute.Tensor,
        w_k: cute.Tensor,
        w_v: cute.Tensor,
        cs_q: cute.Tensor,
        cs_k: cute.Tensor,
        cs_v: cute.Tensor,
        A_log: cute.Tensor,
        g: cute.Tensor,
        dt_bias: cute.Tensor,
        beta: cute.Tensor,
        onorm_g: cute.Tensor,
        onorm_weight: cute.Tensor,
        o: cute.Tensor,
        output_scale: cute.Tensor,
        ht: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        stage_timing: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        precompute_control: cute.Tensor,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        N: cutlass.Int32,
        NUM_SPEC: cutlass.Constexpr[int],
        TILE_V: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        onorm_eps: cutlass.Constexpr[float],
        scale_leading_dim: cutlass.Constexpr[int],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        USE_SETMAXREG: cutlass.Constexpr[bool],
        USE_REGULAR_METADATA: cutlass.Constexpr[bool],
        USE_PACKED_TOKEN_LAYOUT: cutlass.Constexpr[bool],
        USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
        USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
        FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
        RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
        FUSE_OUTPUT_NORM: cutlass.Constexpr[bool],
        QUANTIZE_OUTPUT: cutlass.Constexpr[bool],
        PROFILE_STAGES: cutlass.Constexpr[bool],
        SPLIT_V: cutlass.Constexpr[int],
        BF16_MMA: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
        COMPACT: cutlass.Constexpr[bool] = False,
    ):
        if cutlass.const_expr(USE_ZERO_ACCEPTED):
            t_max = 1 + NUM_SPEC
        else:
            t_max = 2 * NUM_SPEC + 1
        smem_qk_layout = cute.make_layout((t_max, K), stride=(K, 1))
        # BF16_MMA: bf16 tensor-core operands ([K~; Q~] only, K = 16 per instruction).
        OP_DTYPE = cutlass.BFloat16 if BF16_MMA else cutlass.Float32
        N1 = 2 * T_PAD if BF16_MMA else 3 * T_PAD
        # GEMM1: M = V rows of the state (TMEM), N = operand rows ([K~; Q~; K~lo] for tf32).
        tiled_mma1 = sm100_utils.make_trivial_tiled_mma(
            OP_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, N1),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1 = sm100_utils.make_smem_layout_b(tiled_mma1, (V, N1, K), OP_DTYPE, 1)
        # fp32 A-fragment layout of the (V x K) state in TMEM: the GEMM2 accumulator seed / state
        # store views use it in every mode.
        tiled_mma_f32a = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        # GEMM1b: the S0_lo sweep only needs K~hi (N = T_PAD) -> a third of the N=48 work.
        tiled_mma1k = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1k = sm100_utils.make_smem_layout_b(
            tiled_mma1k, (V, T_PAD, K), cutlass.Float32, 1
        )
        # TMA bulk store of the committed state: (V, K, tiles) view of ht, one 128B-swizzled
        # (V/SPLIT_V x 32 fp32 | 64 bf16) box per 128 B-wide column slab. The view takes ht's own strides (the
        # per-layer state cache is a strided slice of a larger buffer); the tile mode is the pool
        # index (flat layout) or the (head, slot) pair.
        if cutlass.const_expr(cute.rank(ht.shape) == 3):
            mT_vkl = cute.make_tensor(
                ht.iterator,
                cute.make_layout(
                    (V, K, ht.shape[0]),
                    stride=(ht.layout.stride[1], ht.layout.stride[2], ht.layout.stride[0]),
                ),
            )
        else:
            mT_vkl = cute.make_tensor(
                ht.iterator,
                cute.make_layout(
                    (V, K, (ht.shape[1], ht.shape[0])),
                    stride=(
                        ht.layout.stride[2],
                        ht.layout.stride[3],
                        (ht.layout.stride[1], ht.layout.stride[0]),
                    ),
                ),
            )
        # 128 B swizzled box rows: 32 fp32 or 64 bf16 state columns per box
        box_cols = 128 // (h0.element_type.width // 8)
        box_atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, h0.element_type
        )
        box_layout = cute.tile_to_shape(box_atom, (V, box_cols), order=(0, 1))
        # committed-state store: each CTA writes its V/SPLIT_V rows through (V/SPLIT_V x box_cols) boxes
        box_layout_t = cute.tile_to_shape(box_atom, (V // SPLIT_V, box_cols), order=(0, 1))
        tma_atom_t, mT_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mT_vkl, box_layout_t, (V // SPLIT_V, box_cols)
        )
        # TMA bulk load of the initial state S0 into the same box layout (issued at kernel start).
        if cutlass.const_expr(cute.rank(h0.shape) == 3):
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, h0.shape[0]),
                    stride=(h0.layout.stride[1], h0.layout.stride[2], h0.layout.stride[0]),
                ),
            )
        else:
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, (h0.shape[1], h0.shape[0])),
                    stride=(
                        h0.layout.stride[2],
                        h0.layout.stride[3],
                        (h0.layout.stride[1], h0.layout.stride[0]),
                    ),
                ),
            )
        tma_atom_l, mS_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mS_vkl, box_layout, (V, box_cols)
        )
        # GEMM2: (V x T_PAD) x (T_PAD x K), both operands in smem, tf32; issued as two N = K/2
        # column halves so the state store can start on half 0.
        tiled_mma2h = sm100_utils.make_trivial_tiled_mma(
            OP_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2h = sm100_utils.make_smem_layout_b(
            tiled_mma2h, (V, K // 2, T_PAD), OP_DTYPE, 1
        )
        smem_layout_a2 = sm100_utils.make_smem_layout_a(
            tiled_mma2h, (V, K // 2, T_PAD), OP_DTYPE, 1
        )
        # GEMM2 half 0 also carries the output-combination GEMM: N = T_PAD (Bcoef) + K/2 (Kbar rows 0..63).
        tiled_mma2q = sm100_utils.make_trivial_tiled_mma(
            OP_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD + K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2q = sm100_utils.make_smem_layout_b(
            tiled_mma2q, (V, T_PAD + K // 2, T_PAD), OP_DTYPE, 1
        )
        # GEMM3: T x T coupling matrices, A3 = [Kf; Qf] padded to 64 rows, B3 = Kb (16 rows).
        tiled_mma3 = sm100_utils.make_trivial_tiled_mma(
            OP_DTYPE,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (4 * T_PAD, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_a3 = sm100_utils.make_smem_layout_a(
            tiled_mma3, (4 * T_PAD, T_PAD, K), OP_DTYPE, 1
        )
        smem_layout_b3 = sm100_utils.make_smem_layout_b(
            tiled_mma3, (4 * T_PAD, T_PAD, K), OP_DTYPE, 1
        )
        # Output combination D4[v, t] = sum_s V'[s, v] Bcoef[t, s]: computed as the first 16 columns
        # of GEMM2 half 0; this tiled MMA only provides the accumulator layout of those columns.
        tiled_mma4 = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        launched = kda_decode_mtp_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            output_scale,
            ht,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            smem_qk_layout,
            ssm_state_indices,
            cu_seqlens,
            num_accepted_tokens,
            precompute_control,
            TILE_V,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            onorm_eps,
            scale_leading_dim,
            USE_FLAT_LAYOUT,
            USE_SETMAXREG,
            USE_REGULAR_METADATA,
            USE_PACKED_TOKEN_LAYOUT,
            USE_REG_Q_WEIGHTS,
            USE_ZERO_ACCEPTED,
            FUSE_PRECOMPUTE,
            RUNTIME_PRECOMPUTE_FLAG,
            FUSE_OUTPUT_NORM,
            QUANTIZE_OUTPUT,
            stage_timing,
            PROFILE_STAGES,
            tiled_mma1,
            tiled_mma3,
            tiled_mma1k,
            smem_layout_b1,
            smem_layout_a2,
            smem_layout_a3,
            smem_layout_b3,
            smem_layout_b1k,
            tiled_mma4,
            tma_atom_t,
            mT_tma,
            box_layout,
            tma_atom_l,
            mS_tma,
            tiled_mma2h,
            smem_layout_b2h,
            box_layout_t,
            tiled_mma2q,
            smem_layout_b2q,
            SPLIT_V,
            BF16_MMA,
            tiled_mma_f32a,
            COMPACT,
        )
        if cutlass.const_expr(SPLIT_V == 2):
            # cluster pair: the two CTAs of a (head, request) must be co-scheduled (in-place state)
            launched.launch(
                grid=(HV, N, 2), block=[NUM_THREADS, 1, 1], cluster=(1, 1, 2), stream=stream
            )
        elif cutlass.const_expr(COMPACT):
            # Two co-resident 512-thread CTAs need <= 64 registers per thread (the sm_107a build
            # allocates 75 without the launch bound, which leaves room for one CTA per SM).
            launched.launch(
                grid=(HV, N, 1),
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                min_blocks_per_mp=_COMPACT_MIN_BLOCKS,
            )
        else:
            launched.launch(grid=(HV, N, 1), block=[NUM_THREADS, 1, 1], stream=stream)


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _run_kda_decode_mtp_v2(
        h0: cute.Tensor,
        x_q: cute.Tensor,
        x_k: cute.Tensor,
        x_v: cute.Tensor,
        w_q: cute.Tensor,
        w_k: cute.Tensor,
        w_v: cute.Tensor,
        cs_q: cute.Tensor,
        cs_k: cute.Tensor,
        cs_v: cute.Tensor,
        A_log: cute.Tensor,
        g: cute.Tensor,
        dt_bias: cute.Tensor,
        beta: cute.Tensor,
        onorm_g: cute.Tensor,
        onorm_weight: cute.Tensor,
        o: cute.Tensor,
        ht: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        fa: cute.Tensor,
        w_fb: cute.Tensor,
        N: cutlass.Int32,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        NUM_SPEC: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        onorm_eps: cutlass.Constexpr[float],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        SPLIT_V: cutlass.Constexpr[int],
        PDL: cutlass.Constexpr[bool],
        FUSED_FB: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        """Host side of ``kda_decode_mtp_v2_kernel`` (packed rows, fused gated RMSNorm, bf16
        output, 3xTF32): the tf32 MMA atoms, operand smem layouts and state TMA atoms of
        ``_run_kda_decode_mtp``, then the launch. With ``FUSED_FB``, ``fa`` is the BF16
        ``[1, T, 128]`` f_a view and ``w_fb`` the BF16 ``[HV * 128, 128]`` f_b weight (else both
        are unused placeholders)."""
        f32 = cutlass.Float32
        smem_qk_layout = cute.make_layout((2 * NUM_SPEC + 1, K), stride=(K, 1))
        tiled_mma1 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1 = sm100_utils.make_smem_layout_b(tiled_mma1, (V, 3 * T_PAD, K), f32, 1)
        tiled_mma_f32a = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        tiled_mma1k = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1k = sm100_utils.make_smem_layout_b(tiled_mma1k, (V, T_PAD, K), f32, 1)
        if cutlass.const_expr(cute.rank(ht.shape) == 3):
            mT_vkl = cute.make_tensor(
                ht.iterator,
                cute.make_layout(
                    (V, K, ht.shape[0]),
                    stride=(ht.layout.stride[1], ht.layout.stride[2], ht.layout.stride[0]),
                ),
            )
        else:
            mT_vkl = cute.make_tensor(
                ht.iterator,
                cute.make_layout(
                    (V, K, (ht.shape[1], ht.shape[0])),
                    stride=(
                        ht.layout.stride[2],
                        ht.layout.stride[3],
                        (ht.layout.stride[1], ht.layout.stride[0]),
                    ),
                ),
            )
        box_cols = 128 // (h0.element_type.width // 8)
        box_atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, h0.element_type
        )
        box_layout = cute.tile_to_shape(box_atom, (V, box_cols), order=(0, 1))
        box_layout_t = cute.tile_to_shape(box_atom, (V // SPLIT_V, box_cols), order=(0, 1))
        tma_atom_t, mT_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mT_vkl, box_layout_t, (V // SPLIT_V, box_cols)
        )
        if cutlass.const_expr(cute.rank(h0.shape) == 3):
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, h0.shape[0]),
                    stride=(h0.layout.stride[1], h0.layout.stride[2], h0.layout.stride[0]),
                ),
            )
        else:
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, (h0.shape[1], h0.shape[0])),
                    stride=(
                        h0.layout.stride[2],
                        h0.layout.stride[3],
                        (h0.layout.stride[1], h0.layout.stride[0]),
                    ),
                ),
            )
        tma_atom_l, mS_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mS_vkl, box_layout, (V, box_cols)
        )
        tiled_mma2h = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2h = sm100_utils.make_smem_layout_b(tiled_mma2h, (V, K // 2, T_PAD), f32, 1)
        smem_layout_a2 = sm100_utils.make_smem_layout_a(tiled_mma2h, (V, K // 2, T_PAD), f32, 1)
        tiled_mma2q = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD + K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2q = sm100_utils.make_smem_layout_b(
            tiled_mma2q, (V, T_PAD + K // 2, T_PAD), f32, 1
        )
        tiled_mma3 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (4 * T_PAD, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_a3 = sm100_utils.make_smem_layout_a(tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1)
        smem_layout_b3 = sm100_utils.make_smem_layout_b(tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1)
        tiled_mma4 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        if cutlass.const_expr(FUSED_FB):
            # W_fb rows of one head: two 128 x 64 BF16 boxes with 128B swizzle.
            fb_atom = sm100_utils.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
            )
            mFb = cute.make_tensor(
                w_fb.iterator, cute.make_layout((w_fb.shape[0], K), stride=(K, 1))
            )
            tma_atom_fb, mFb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mFb,
                cute.tile_to_shape(fb_atom, (K, 64), order=(0, 1)),
                (K, 64),
            )
            fb_layout = cute.tile_to_shape(fb_atom, (K, 64, K // 64), (0, 1, 2))
        else:
            tma_atom_fb, mFb_tma, fb_layout = tma_atom_l, mS_tma, box_layout
        launched = kda_decode_mtp_v2_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            smem_qk_layout,
            tiled_mma1,
            tiled_mma3,
            tiled_mma1k,
            smem_layout_b1,
            smem_layout_a2,
            smem_layout_a3,
            smem_layout_b3,
            smem_layout_b1k,
            tiled_mma4,
            tma_atom_t,
            mT_tma,
            box_layout,
            tma_atom_l,
            mS_tma,
            tiled_mma2h,
            smem_layout_b2h,
            box_layout_t,
            tiled_mma2q,
            smem_layout_b2q,
            tiled_mma_f32a,
            fa,
            tma_atom_fb,
            mFb_tma,
            fb_layout,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            onorm_eps,
            USE_FLAT_LAYOUT,
            SPLIT_V,
            PDL,
            FUSED_FB,
        )
        if cutlass.const_expr(SPLIT_V == 2):
            # cluster pair: the two CTAs of a (head, request) must be co-scheduled (in-place state)
            launched.launch(
                grid=(HV, N, 2),
                block=[NUM_THREADS, 1, 1],
                cluster=(1, 1, 2),
                stream=stream,
                use_pdl=PDL,
            )
        else:
            launched.launch(
                grid=(HV, N, 1),
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=PDL,
            )


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _run_kda_decode_mtp_2cta(
        h0: cute.Tensor,
        x_q: cute.Tensor,
        x_k: cute.Tensor,
        x_v: cute.Tensor,
        w_q: cute.Tensor,
        w_k: cute.Tensor,
        w_v: cute.Tensor,
        cs_q: cute.Tensor,
        cs_k: cute.Tensor,
        cs_v: cute.Tensor,
        A_log: cute.Tensor,
        g: cute.Tensor,
        dt_bias: cute.Tensor,
        beta: cute.Tensor,
        onorm_g: cute.Tensor,
        onorm_weight: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        o8: cute.Tensor,
        osf: cute.Tensor,
        N: cutlass.Int32,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        NUM_SPEC: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        onorm_eps: cutlass.Constexpr[float],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        PDL: cutlass.Constexpr[bool],
        MIN_BLOCKS: cutlass.Constexpr[int],
        stream: cuda.CUstream,
        EARLY_TRIGGER: cutlass.Constexpr[bool] = False,
    ):
        """Host side of ``kda_decode_mtp_2cta_kernel``: v2's tf32 MMA atoms, operand smem
        layouts and state TMA atoms (fp32 or bf16 state), launched for ``MIN_BLOCKS`` CTAs per SM
        (64 registers per thread at 2). The kernel writes the E4M3 output ``o8`` and its R128c4
        UE8M0 scales ``osf``."""
        f32 = cutlass.Float32
        tiled_mma1 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1 = sm100_utils.make_smem_layout_b(tiled_mma1, (V, 3 * T_PAD, K), f32, 1)
        tiled_mma_f32a = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        tiled_mma1k = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1k = sm100_utils.make_smem_layout_b(tiled_mma1k, (V, T_PAD, K), f32, 1)
        tiled_mma2h = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2h = sm100_utils.make_smem_layout_b(tiled_mma2h, (V, K // 2, T_PAD), f32, 1)
        smem_layout_a2 = sm100_utils.make_smem_layout_a(tiled_mma2h, (V, K // 2, T_PAD), f32, 1)
        tiled_mma2q = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD + K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2q = sm100_utils.make_smem_layout_b(
            tiled_mma2q, (V, T_PAD + K // 2, T_PAD), f32, 1
        )
        tiled_mma3 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (4 * T_PAD, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_a3 = sm100_utils.make_smem_layout_a(tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1)
        smem_layout_b3 = sm100_utils.make_smem_layout_b(tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1)
        tiled_mma4 = sm100_utils.make_trivial_tiled_mma(
            f32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            f32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        if cutlass.const_expr(cute.rank(h0.shape) == 3):
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, h0.shape[0]),
                    stride=(h0.layout.stride[1], h0.layout.stride[2], h0.layout.stride[0]),
                ),
            )
        else:
            mS_vkl = cute.make_tensor(
                h0.iterator,
                cute.make_layout(
                    (V, K, (h0.shape[1], h0.shape[0])),
                    stride=(
                        h0.layout.stride[2],
                        h0.layout.stride[3],
                        (h0.layout.stride[1], h0.layout.stride[0]),
                    ),
                ),
            )
        # 128 B swizzled box rows: 32 fp32 or 64 bf16 state columns per box
        box_cols = 128 // (h0.element_type.width // 8)
        box_atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, h0.element_type
        )
        box_layout = cute.tile_to_shape(box_atom, (V, box_cols), order=(0, 1))
        tma_atom_l, mS_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mS_vkl, box_layout, (V, box_cols)
        )
        tma_atom_t, mT_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mS_vkl, box_layout, (V, box_cols)
        )
        launched = kda_decode_mtp_2cta_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            o8,
            osf,
            tiled_mma1,
            tiled_mma3,
            tiled_mma1k,
            smem_layout_b1,
            smem_layout_a2,
            smem_layout_a3,
            smem_layout_b3,
            smem_layout_b1k,
            tiled_mma4,
            tiled_mma2h,
            smem_layout_b2h,
            tiled_mma2q,
            smem_layout_b2q,
            tiled_mma_f32a,
            tma_atom_l,
            mS_tma,
            box_layout,
            tma_atom_t,
            mT_tma,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            onorm_eps,
            USE_FLAT_LAYOUT,
            PDL,
            EARLY_TRIGGER,
        )
        launched.launch(
            grid=(HV, N, 1),
            block=[NUM_THREADS, 1, 1],
            stream=stream,
            use_pdl=PDL,
            min_blocks_per_mp=MIN_BLOCKS,
        )


def _require_stride_layout(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    k_cache,
    g_cache,
    v_cache,
    beta_cache,
    ssm_state_indices,
    cu_seqlens,
    num_accepted_tokens,
    out,
    onorm_g,
    onorm_weight,
    H,
    HV,
    K,
    V,
    W,
    num_spec,
    T_total,
    packed_token_layout,
    fuse_output_norm,
    quantize_output,
    fa=None,
    w_fb=None,
):
    if x_q.ndim != 4 or x_k.ndim != 4 or x_v.ndim != 4:
        raise ValueError("Expected x_q/x_k/x_v to have shape [1, T, H, D].")
    if x_q.shape != (1, T_total, H, K) or x_k.shape != (1, T_total, H, K):
        raise ValueError(f"Expected x_q/x_k shape [1, {T_total}, {H}, {K}].")
    if x_v.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected x_v shape [1, {T_total}, {HV}, {V}].")
    if fa is not None:
        if fa.ndim != 3 or fa.shape != (1, T_total, K) or fa.dtype != torch.bfloat16:
            raise ValueError(f"Expected bf16 fa shape [1, {T_total}, {K}].")
        if w_fb.shape != (HV * K, K) or w_fb.dtype != torch.bfloat16 or not w_fb.is_contiguous():
            raise ValueError(f"Expected a contiguous bf16 w_fb of shape [{HV * K}, {K}].")
        if w_fb.data_ptr() % 16:
            raise ValueError("Expected w_fb to be 16-byte aligned (TMA source).")
    elif g.ndim != 4 or g.shape != (1, T_total, HV, K):
        raise ValueError(f"Expected g shape [1, {T_total}, {HV}, {K}].")
    if beta.ndim != 3 or beta.shape != (1, T_total, HV):
        raise ValueError(f"Expected beta shape [1, {T_total}, {HV}].")
    if quantize_output:
        if out.ndim != 2 or out.shape != (T_total, HV * V):
            raise ValueError(f"Expected FP8 out shape [{T_total}, {HV * V}].")
        if out.dtype is not torch.float8_e4m3fn:
            raise TypeError("Expected FP8 out dtype torch.float8_e4m3fn.")
    elif out.ndim != 4 or out.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected BF16 out shape [1, {T_total}, {HV}, {V}].")
    if fuse_output_norm:
        if not packed_token_layout:
            raise ValueError("fuse_output_norm requires packed_token_layout=True.")
        if onorm_g is None or onorm_weight is None:
            raise ValueError("fuse_output_norm requires onorm_g and onorm_weight.")
        if onorm_g.ndim != 4 or onorm_g.shape != (1, T_total, HV, V):
            raise ValueError(f"Expected onorm_g shape [1, {T_total}, {HV}, {V}].")
        if onorm_g.stride(-1) != 1 or onorm_g.stride(-2) != V:
            raise ValueError("Expected onorm_g to have contiguous head/value blocks.")
        if onorm_weight.ndim != 1 or onorm_weight.shape[0] != V:
            raise ValueError(f"Expected onorm_weight shape [{V}].")
        if onorm_weight.stride(0) != 1:
            raise ValueError("Expected onorm_weight to be contiguous.")
    if quantize_output and not fuse_output_norm:
        raise ValueError("quantize_output requires fuse_output_norm=True.")

    last_dim_tensors = {
        "x_q": x_q,
        "x_k": x_k,
        "x_v": x_v,
        "g": g if fa is None else fa,
        "beta": beta,
        "recurrent_state": recurrent_state,
        "k_cache": k_cache,
        "g_cache": g_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
    }
    if not quantize_output:
        last_dim_tensors["out"] = out
    for name, tensor in last_dim_tensors.items():
        if tensor.stride(-1) != 1:
            raise ValueError(f"Expected {name} to be contiguous in its last dimension.")

    if w_q.shape != (H * K, W) or w_k.shape != (H * K, W) or w_v.shape != (HV * V, W):
        raise ValueError(f"Expected w_q/w_k shape [{H * K}, {W}] and w_v shape [{HV * V}, {W}].")
    if w_q.stride(1) != 1 or w_k.stride(1) != 1 or w_v.stride(1) != 1:
        raise ValueError("Expected w_q/w_k/w_v to be contiguous in the kernel-width axis.")

    if A_log.ndim != 1 or A_log.shape[0] != H:
        raise ValueError(f"Expected A_log shape [{H}].")
    if dt_bias.ndim != 1 or dt_bias.shape[0] != H * K:
        raise ValueError(f"Expected dt_bias shape [{H * K}].")

    state_s = W - 1 + num_spec
    if cs_q.ndim != 3 or cs_k.ndim != 3 or cs_v.ndim != 3:
        raise ValueError("Expected cs_q/cs_k/cs_v to have shape [pool, dim, S].")
    if cs_q.shape[1] != H * K or cs_k.shape[1] != H * K:
        raise ValueError(f"Expected cs_q/cs_k shape [pool, {H * K}, S].")
    if cs_v.shape[1] != HV * V:
        raise ValueError(f"Expected cs_v shape [pool, {HV * V}, S].")
    if cs_q.shape[2] < state_s or cs_k.shape[2] < state_s or cs_v.shape[2] < state_s:
        raise ValueError(f"Expected conv-state S dimension to be at least {state_s}.")
    if cs_q.stride(1) != 1 or cs_k.stride(1) != 1 or cs_v.stride(1) != 1:
        raise ValueError(
            "Expected cs_q/cs_k/cs_v to use dim-contiguous layout "
            "(allocate as [pool, S, dim] and transpose(1, 2))."
        )

    pool_size = recurrent_state.shape[0]
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (HV, V, K):
        raise ValueError(
            f"Expected recurrent_state shape [pool, {HV}, {V}, {K}] (V-first pool layout)."
        )
    if recurrent_state.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"recurrent_state must be fp32 or bf16, got {recurrent_state.dtype}")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (num_spec, H * K):
        raise ValueError(f"Expected k_cache shape [pool, {num_spec}, {H * K}].")
    if g_cache.ndim != 3 or g_cache.shape[1:] != (num_spec, H * K):
        raise ValueError(f"Expected g_cache shape [pool, {num_spec}, {H * K}].")
    if g_cache.dtype != torch.float32:
        raise ValueError("g_cache (the replayed gates) must be fp32.")
    if v_cache.ndim != 3 or v_cache.shape[1:] != (num_spec, HV * V):
        raise ValueError(f"Expected v_cache shape [pool, {num_spec}, {HV * V}].")
    if beta_cache.ndim != 3 or beta_cache.shape[1:] != (num_spec, HV):
        raise ValueError(f"Expected beta_cache shape [pool, {num_spec}, {HV}].")
    if (
        k_cache.shape[0] < pool_size
        or g_cache.shape[0] < pool_size
        or v_cache.shape[0] < pool_size
        or beta_cache.shape[0] < pool_size
    ):
        raise ValueError("Expected cache pool dimensions to cover recurrent_state rows.")

    if ssm_state_indices.ndim != 1 or num_accepted_tokens.ndim != 1:
        raise ValueError("Expected ssm_state_indices and num_accepted_tokens to be 1D.")
    num_requests = ssm_state_indices.shape[0]
    if packed_token_layout:
        if cu_seqlens is not None:
            raise ValueError("packed_token_layout derives row offsets; cu_seqlens must be None.")
        if T_total != num_requests * (num_spec + 1):
            raise ValueError(
                "packed_token_layout expects exactly num_spec + 1 new-token rows per request."
            )
        if num_accepted_tokens.shape[0] < pool_size:
            raise ValueError("Expected slot-indexed num_accepted_tokens to cover the state pool.")
    else:
        if cu_seqlens is None or cu_seqlens.ndim != 1:
            raise ValueError("Expected cu_seqlens to be a 1D tensor.")
        if cu_seqlens.shape[0] != num_requests + 1:
            raise ValueError("Expected cu_seqlens length to be N + 1.")
        if num_accepted_tokens.shape[0] != num_requests:
            raise ValueError("Expected num_accepted_tokens length to match N.")


def _fits_32bit_stride(tensor: torch.Tensor) -> bool:
    int32_max = 2**31 - 1
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        stride = abs(int(stride))
        if stride > int32_max:
            return False
        if size:
            max_offset += (int(size) - 1) * stride
            if max_offset > int32_max:
                return False
    return True


def _from_dlpack_arg(tensor: torch.Tensor, *, assumed_align: int = 16):
    return from_dlpack(
        tensor,
        assumed_align=assumed_align,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )


def _beta_cache_assumed_align(beta_cache: torch.Tensor) -> int:
    """Return the alignment shared by KDA per-layer beta-cache views.

    The producer allocates ``[layers, slots, num_spec, local_heads]``. After
    selecting a layer, ``slots * stride(0)`` is the physical layer span even
    when the head rows are padded. Combine that span with the current pointer
    and dtype, capped at the CuTe bridge's useful 16-byte guarantee.
    """
    layer_span_bytes = beta_cache.shape[0] * beta_cache.stride(0) * beta_cache.element_size()
    return gcd(16, beta_cache.data_ptr(), layer_span_bytes)


def _dlpack_arg(tensor: torch.Tensor, *, assumed_align: int):
    # Alignment is deliberately mandatory: layout dynamism does not imply
    # arbitrary pointer alignment. Each call site must state the guarantee
    # provided by that tensor's producer and view pattern.
    arg = _from_dlpack_arg(tensor, assumed_align=assumed_align)
    for dim, stride in enumerate(tensor.stride()):
        if stride == 1:
            return arg.mark_layout_dynamic(dim)
    return arg.mark_layout_dynamic()


def _layout_key(tensor: torch.Tensor, dynamic_layout: bool = False, *, assumed_align: int = 16):
    arg = (
        _dlpack_arg(tensor, assumed_align=assumed_align)
        if dynamic_layout
        else _from_dlpack_arg(tensor, assumed_align=assumed_align)
    )
    shape_mask = arg.dynamic_shapes_mask
    stride_mask = arg.dynamic_strides_mask
    shape = tuple(None if dynamic else size for size, dynamic in zip(tensor.shape, shape_mask))
    stride = tuple(
        None if dynamic else value for value, dynamic in zip(tensor.stride(), stride_mask)
    )
    return (tensor.dtype, shape, stride, _fits_32bit_stride(tensor), assumed_align)


# (device_index, enabled) -> persistent int32 [1] control tensor. Keys are
# plain values (not tensor identities), so entries never go stale.
_precompute_control_cache = {}


# Recurrence precision of the verify kernel. False: 3xTF32 on every product that feeds the state.
# True (tests): bf16 tensor-core operands (bf16 TMEM / smem tiles, K = 16 per instruction, fp32
# accumulation -- the prefill kernel's precision).
_BF16_MATH = False


def _bf16_math_enabled() -> bool:
    """Whether the verify kernel runs the bf16 tensor-core recurrence (``_BF16_MATH``)."""
    return _BF16_MATH


# Verify-kernel schedule. True: calls on the runtime contract (packed rows, fused gated RMSNorm,
# bf16 output, 3xTF32 recurrence, vector-aligned operands) run ``kda_decode_mtp_v2_kernel``, which
# produces bitwise the same results with a shorter critical path. False (tests): every call runs
# ``kda_decode_mtp_kernel``.
_MTP_V2 = True


def _mtp_v2_enabled() -> bool:
    """Whether calls on the runtime contract run ``kda_decode_mtp_v2_kernel`` (``_MTP_V2``)."""
    return _MTP_V2


_num_sms_cache = {}


def _num_sms(device: torch.device) -> int:
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _num_sms_cache:
        _num_sms_cache[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _num_sms_cache[idx]


def _compact_layout_enabled(device: torch.device) -> bool:
    """Whether ``kda_decode_mtp_kernel`` may use its compact (two CTAs per SM) layout: on SM107.
    The layout produces bitwise the results of the default layout."""
    return torch.cuda.get_device_capability(device) == (10, 7)


def _mxfp8_two_cta_enabled(device: torch.device) -> bool:
    """Whether MXFP8-output verify calls run the two-CTAs-per-SM kernel (``kda_mtp_decode_2cta``
    with its MXFP8 epilogue) instead of ``kda_decode_mtp_kernel``: on SM107. Both kernels produce
    bitwise the same results."""
    return torch.cuda.get_device_capability(device) == (10, 7)


def verify_ctas(num_heads: int, num_requests: int, device: torch.device) -> int:
    """CTAs of the v2 verify kernel's grid for these tiles (cluster pairs up to
    ``SPLIT_MAX_TILES`` tiles, or one CTA per tile)."""
    tiles = num_heads * num_requests
    return tiles * (2 if tiles <= SPLIT_MAX_TILES else 1)


def _vec4_rows(tensor: torch.Tensor, dim: int = -1) -> bool:
    """True when every group of four consecutive elements along the unit-stride dimension `dim`
    that starts at a multiple of four is one aligned 8 B (bf16) or 16 B (fp32) access: the v2
    kernel moves its per-token rows with such vector loads and stores."""
    if tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.stride(dim) != 1:
        return False
    if tensor.data_ptr() % (4 * tensor.element_size()):
        return False
    return all(
        stride % 4 == 0 for d, stride in enumerate(tensor.stride()) if d != dim % tensor.ndim
    )


def _v2_operands_ok(*, w_q, w_k, w_v, rows, conv_caches) -> bool:
    """Operand layouts / dtypes the v2 kernel's vector accesses support (see `_vec4_rows`):
    token rows and replay caches vectorized along their last dimension, the dim-contiguous conv
    caches along the channel dimension, conv weights as contiguous [channel][4] rows."""
    weights_ok = all(
        w.dtype == w_q.dtype and w.is_contiguous() and w.shape[1] == 4 and _vec4_rows(w)
        for w in (w_q, w_k, w_v)
    )
    return (
        weights_ok
        and all(_vec4_rows(t) for t in rows)
        and all(_vec4_rows(t, dim=1) for t in conv_caches)
    )


def _precompute_control_tensor(device: torch.device, enabled: bool) -> torch.Tensor:
    dev = torch.device(device)
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), bool(enabled))
    if key not in _precompute_control_cache:
        _precompute_control_cache[key] = torch.tensor(
            [1 if enabled else 0], dtype=torch.int32, device=dev
        )
    return _precompute_control_cache[key]


def _try_flatten_args(
    *,
    recurrent_state: torch.Tensor,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    T_total: int,
    H: int,
    HV: int,
    K: int,
    V: int,
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        h0 = recurrent_state.view(-1, V, K)
        x_q_flat = x_q.view(1, T_total, H * K)
        x_k_flat = x_k.view(1, T_total, H * K)
        x_v_flat = x_v.view(1, T_total, HV * V)
    except RuntimeError:
        return False, recurrent_state, x_q, x_k, x_v
    return True, h0, x_q_flat, x_k_flat, x_v_flat


# Layout-and-constexpr-keyed compile cache. Request count and packed-token
# length are dynamic; batches sharing the same kernel variant reuse one
# artifact even when their launch grid and token-buffer extents differ.
_compiled_cache = {}

# Set around one verify call by :func:`mxfp8_pdl_early_trigger`: an MXFP8-output verify launched
# with ``use_pdl`` runs the two-CTA kernel as a PDL launch that releases its dependent right after
# griddepcontrol.wait.
MXFP8_PDL_EARLY_TRIGGER = False


@contextlib.contextmanager
def mxfp8_pdl_early_trigger(enabled: bool):
    """Within the block, MXFP8-output verify calls with ``use_pdl`` take the two-CTA kernel's PDL
    launch with the early trigger (only for a consumer that waits before reading its output)."""
    global MXFP8_PDL_EARLY_TRIGGER
    saved, MXFP8_PDL_EARLY_TRIGGER = MXFP8_PDL_EARLY_TRIGGER, bool(enabled)
    try:
        yield
    finally:
        MXFP8_PDL_EARLY_TRIGGER = saved


def kda_mtp_decode_impl(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    zero_accepted_hint: bool = False,
    regular_metadata_hint: bool = False,
    packed_token_layout: bool = False,
    onorm_g: Optional[torch.Tensor] = None,
    onorm_weight: Optional[torch.Tensor] = None,
    onorm_eps: float = 1e-5,
    fuse_output_norm: bool = False,
    quantize_output: bool = False,
    fa: Optional[torch.Tensor] = None,
    w_fb: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused KDA MTP verify kernel. See the module docstring.

    Args (device tensors unless noted):
        x_q/x_k/x_v: post-projection, pre-conv token states
            ``[1, T_total, H, 128]`` bf16 — new tokens only, ``1 +
            num_spec`` per request. Rows may be described by ``cu_seqlens``
            or packed uniformly by request.
        w_q/w_k/w_v: conv weights ``[H*128, W]`` fp32, width-contiguous.
        cs_q/cs_k/cs_v: extended conv caches ``[pool, H*128, >= W-1+M]``
            fp32, dim-contiguous. Columns ``[0, W-1)`` are the committed
            window; tail columns hold raw pending-draft inputs. Mutated.
        g, beta: raw gate ``[1, T, H, 128]`` and beta ``[1, T, H]`` bf16.
        A_log, dt_bias: fp32 ``[H]`` / ``[H*128]``.
        recurrent_state: pool ``[pool, H, V, K]`` fp32 or bf16, **V-first** layout
            (matches the executor ssm pool and the single-token decode
            kernel). Committed in place.
        k_cache/g_cache/v_cache/beta_cache: replay caches ``[pool, M, H*K]`` (k,
            fp32 or bf16) / ``[pool, M, H*K]`` (gate, fp32) / ``[pool, M, H*V]`` /
            ``[pool, M, H]`` (fp32 or bf16). Mutated.
        ssm_state_indices: per-request state-pool slots. On the legacy
            metadata path, ``cu_seqlens`` supplies token offsets ``[N+1]``
            and accepted counts are request-indexed ``[N]``. On the packed
            path, ``cu_seqlens`` is absent and accepted counts are pool-level.
        zero_accepted_hint: caller asserts every ``num_accepted_tokens`` is
            zero (compiles the smaller-smem no-replay variant). Wrong hints
            produce wrong results — pass True only when statically known.
        regular_metadata_hint: caller asserts ``cu_seqlens`` is the uniform
            ``arange * (2*num_spec+1)`` pattern and ``ssm_state_indices``
            is ``arange(N)`` (benchmark identity layout).
        packed_token_layout: input projections contain exactly ``1 +
            num_spec`` rows per request. ``cu_seqlens`` must be ``None`` and
            ``num_accepted_tokens`` is indexed by state-pool slot. The
            kernel derives every new-token row from the request index.
        fuse_output_norm: on the packed path, apply gated RMSNorm in the CuTe
            epilogue using section-strided ``onorm_g`` and ``onorm_weight``.
        quantize_output: return the normalized activation directly as MXFP8
            E4M3 plus packed UE8M0 1x128 scales for a prequantized projection.
        fa, w_fb: fused forget gate (v2 contract only): ``fa`` is the BF16
            ``[1, T_total, 128]`` f_a projection (rows may be strided, e.g. the
            first 128 columns of the ``[f_a | b]`` output) and ``w_fb`` the BF16
            ``[H * 128, 128]`` f_b weight; the kernel computes the forget gate
            ``bf16(f_a @ w_fb^T)`` itself and ``g`` must be ``None``.
        use_pdl: launch as a programmatic dependent of the previous grid (v2
            contract only); the kernel loads its state, replay entries and
            weights before ``griddepcontrol.wait``.

    Returns either BF16 ``[1, T_total, H, V]`` or, with
    ``quantize_output=True``, E4M3 ``[T_total, H*V]`` and packed UE8M0 scales
    in the flat uint8 R128c4 layout consumed by the CuTe MXFP8 GEMM.
    """
    _, T_total, _, D = x_q.shape
    H = A_log.shape[0]
    fused_fb = fa is not None
    if fused_fb != (w_fb is not None) or fused_fb == (g is not None):
        raise ValueError("pass either g, or fa and w_fb (fused forget gate)")
    HV = H if fused_fb else (g.shape[2] if g.ndim == 4 else H)
    K = D
    V_dim = x_v.shape[-1]
    W = w_q.shape[1]
    if K != TILE_K or V_dim != 128 or W != 4:
        raise ValueError("specialized kernel expects K=128, V=128, W=4")
    if HV != H:
        raise ValueError("specialized kernel expects HV == H")
    if scale is None:
        scale = K**-0.5

    if packed_token_layout and regular_metadata_hint:
        raise ValueError("packed_token_layout and regular_metadata_hint are mutually exclusive.")
    if packed_token_layout:
        N = ssm_state_indices.shape[0]
    elif cu_seqlens is None:
        raise ValueError("cu_seqlens may be None only with packed_token_layout=True.")
    else:
        N = cu_seqlens.shape[0] - 1
    if quantize_output:
        if out is not None:
            raise ValueError("quantize_output does not accept a caller-provided out tensor.")
        out = torch.empty(
            (T_total, HV * V_dim),
            dtype=torch.float8_e4m3fn,
            device=x_q.device,
        )
        scale_leading_dim = (T_total + 127) // 128 * 128
        output_scale = torch.empty(
            (scale_leading_dim * HV * 4,),
            dtype=torch.uint8,
            device=x_q.device,
        )
    elif out is None:
        output_shape = (1, T_total, HV, V_dim)
        if packed_token_layout:
            # Packed outputs contain new-token rows only, and the kernel
            # writes every head/value element. Avoid a redundant zero-fill.
            out = torch.empty(output_shape, dtype=x_q.dtype, device=x_q.device)
        else:
            # Legacy layouts include replay-position holes that remain zero.
            out = torch.zeros(output_shape, dtype=x_q.dtype, device=x_q.device)
    if num_accepted_tokens.dtype != torch.int32:
        num_accepted_tokens = num_accepted_tokens.to(torch.int32)
    if not quantize_output:
        output_scale = num_accepted_tokens
        scale_leading_dim = 1
    _require_stride_layout(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        onorm_g=onorm_g,
        onorm_weight=onorm_weight,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
        W=W,
        num_spec=num_spec,
        T_total=T_total,
        packed_token_layout=packed_token_layout,
        fuse_output_norm=fuse_output_norm,
        quantize_output=quantize_output,
        fa=fa,
        w_fb=w_fb,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    precompute_control = _precompute_control_tensor(x_q.device, True)
    # The packed variant proves at compile time that this tensor is never
    # read. Reuse an existing metadata tensor as the CuTe placeholder.
    cu_seqlens_arg = ssm_state_indices if cu_seqlens is None else cu_seqlens
    # These tensors are compile-time dead in the raw-output variant. Reuse
    # existing aligned arguments instead of allocating placeholders.
    onorm_g_arg = onorm_g if onorm_g is not None else out
    onorm_weight_arg = onorm_weight if onorm_weight is not None else A_log

    use_flat_layout, h0_arg, x_q_arg, x_k_arg, x_v_arg = _try_flatten_args(
        recurrent_state=recurrent_state,
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        T_total=T_total,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
    )
    pool_size = h0_arg.shape[0]
    # The kernel keeps the conv weights in smem and prefetches the
    # recurrent state before precompute; the setmaxreg and register-weight
    # specializations apply to no shape.
    use_setmaxreg = False
    use_reg_q_weights = False
    use_regular_metadata = bool(regular_metadata_hint)
    # The kernel's USE_ZERO_ACCEPTED fast path unrolls exactly
    # 1 + NUM_SPEC == 3 new tokens, so it is only valid for num_spec == 2.
    # For other num_spec fall back to the generic loop (the hint implies
    # num_accepted_tokens is all zeros, so the generic path computes the
    # same result).
    use_zero_accepted = bool(zero_accepted_hint) and num_spec == 2
    # stage_timing is unused (PROFILE_STAGES=False); pass `out` as the
    # placeholder tensor argument. The alias is
    # only valid while profiling stays off: with PROFILE_STAGES=True the
    # kernel writes int64 stage deltas through this tensor, corrupting the
    # bf16 output. Enabling profiling requires a dedicated int64 buffer of
    # at least HV * N * 4 elements.
    profile_stages = False
    assert not profile_stages, (
        "stage_timing aliases `out`; allocate a dedicated int64 [HV * N * 4] "
        "buffer before enabling PROFILE_STAGES"
    )
    stage_timing_arg = out
    beta_cache_assumed_align = _beta_cache_assumed_align(beta_cache)
    output_scale_assumed_align = 16 if quantize_output else 4
    # Two CTAs per (head, request) while they fit in one wave (see SPLIT_MAX_TILES).
    split_v = 2 if HV * N <= SPLIT_MAX_TILES else 1
    bf16_math = _bf16_math_enabled()

    gate_rows = (fa,) if fused_fb else (g,)

    def v2_operands_ok():
        return _v2_operands_ok(
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            rows=(x_q_arg, x_k_arg, x_v_arg, *gate_rows, onorm_g, k_cache, g_cache, v_cache),
            conv_caches=(cs_q, cs_k, cs_v),
        )

    def two_cta_operands_ok():
        return (
            h0_arg.dtype in (torch.float32, torch.bfloat16)
            and all(t.dtype == torch.bfloat16 for t in (x_q, x_k, x_v, cs_q, cs_k, cs_v))
            and g_cache.dtype == torch.float32
        )

    # MXFP8 output: the two-CTAs-per-SM kernel with kda_decode_mtp_kernel's quantization epilogue
    # (the same arithmetic, see kda_mtp_decode_2cta.py), one CTA per tile at every grid size. Two
    # of its tiles share an SM on multi-wave grids; on SM107 a tile alone on an SM also finishes
    # sooner than in kda_decode_mtp_kernel, with or without its cluster pair.
    # Under mxfp8_pdl_early_trigger: the MXFP8 two-CTA kernel also as a PDL launch that releases
    # its dependent right after its griddepcontrol.wait, for a consumer that only streams weights
    # before its own wait (the mid-M o_proj GEMM with the all-reduce push).
    mx_pdl = bool(quantize_output and use_pdl and MXFP8_PDL_EARLY_TRIGGER)
    use_mxfp8_2cta = (
        quantize_output
        and _mxfp8_two_cta_enabled(x_q.device)
        and packed_token_layout
        and fuse_output_norm
        and not fused_fb
        and (not use_pdl or mx_pdl)
        and not bf16_math
        and num_spec <= 7
        and two_cta_operands_ok()
        and v2_operands_ok()
    )
    mx_pdl = mx_pdl and use_mxfp8_2cta
    use_v2 = (
        (_mtp_v2_enabled() or fused_fb or use_pdl)
        and packed_token_layout
        and fuse_output_norm
        and not quantize_output
        and not bf16_math
        and num_spec <= 7
        and v2_operands_ok()
    )
    if (fused_fb or (use_pdl and not mx_pdl)) and not use_v2:
        raise ValueError(
            "the fused forget gate and PDL launch need the v2 kernel contract (packed rows, "
            "fused output norm, bf16 output, 3xTF32, vector-aligned operands)"
        )
    if use_mxfp8_2cta:
        # E4M3 [T, HV * V] and the R128c4 UE8M0 scales (no BF16 output)
        args_2cta = (
            (h0_arg, 16),
            (x_q_arg, 16),
            (x_k_arg, 16),
            (x_v_arg, 16),
            (w_q, 16),
            (w_k, 16),
            (w_v, 16),
            (cs_q, 16),
            (cs_k, 16),
            (cs_v, 16),
            (A_log, 16),
            (g, 16),
            (dt_bias, 16),
            (beta, 16),
            (onorm_g, 16),
            (onorm_weight, 16),
            (k_cache, 16),
            (g_cache, 16),
            (v_cache, 16),
            (beta_cache, beta_cache_assumed_align),
            (ssm_state_indices, 4),
            (num_accepted_tokens, 4),
            (out, 16),
            (output_scale, 16),
        )
        key_2cta = (
            "2cta",
            tuple(_layout_key(t, dynamic_layout=True, assumed_align=a) for t, a in args_2cta),
            scale,
            HV,
            K,
            V_dim,
            num_spec,
            W,
            float(lower_bound),
            float(onorm_eps),
            use_flat_layout,
            bool(use_pdl),
            _TWO_CTA_MIN_BLOCKS,
            bool(mx_pdl),
        )
        if key_2cta not in _compiled_cache:
            logger.info(
                f"kda_mtp_decode: compiling 2-CTA MXFP8 variant H={HV} num_spec={num_spec}"
                f" pdl={bool(use_pdl)} early_trigger={bool(mx_pdl)}"
            )
            _compiled_cache[key_2cta] = cute.compile(
                _run_kda_decode_mtp_2cta,
                *[_dlpack_arg(t, assumed_align=a) for t, a in args_2cta],
                N,
                scale=scale,
                HV=HV,
                K=K,
                V=V_dim,
                NUM_SPEC=num_spec,
                KERNEL_WIDTH=W,
                lower_bound=float(lower_bound),
                onorm_eps=float(onorm_eps),
                USE_FLAT_LAYOUT=use_flat_layout,
                PDL=bool(use_pdl),
                MIN_BLOCKS=_TWO_CTA_MIN_BLOCKS,
                stream=stream,
                EARLY_TRIGGER=bool(mx_pdl),
            )
        _compiled_cache[key_2cta](
            *[_dlpack_arg(t, assumed_align=a) for t, a in args_2cta], N, stream
        )
        return out, output_scale
    if use_v2:
        # Placeholders for the operands of the other gate mode (compile-time dead).
        g_arg = onorm_g if fused_fb else g
        fa_arg = fa if fused_fb else beta
        w_fb_arg = w_fb if fused_fb else w_q
        v2_args = (
            (h0_arg, 16),
            (x_q_arg, 16),
            (x_k_arg, 16),
            (x_v_arg, 16),
            (w_q, 16),
            (w_k, 16),
            (w_v, 16),
            (cs_q, 16),
            (cs_k, 16),
            (cs_v, 16),
            (A_log, 16),
            (g_arg, 16),
            (dt_bias, 16),
            (beta, 16),
            (onorm_g, 16),
            (onorm_weight, 16),
            (out, 16),
            (h0_arg, 16),
            (k_cache, 16),
            (g_cache, 16),
            (v_cache, 16),
            (beta_cache, beta_cache_assumed_align),
            (ssm_state_indices, 4),
            (num_accepted_tokens, 4),
            (fa_arg, 16 if fused_fb else 2),
            (w_fb_arg, 16),
        )
        v2_key = (
            "v2",
            tuple(_layout_key(t, dynamic_layout=True, assumed_align=a) for t, a in v2_args),
            scale,
            HV,
            K,
            V_dim,
            num_spec,
            W,
            float(lower_bound),
            float(onorm_eps),
            use_flat_layout,
            split_v,
            bool(use_pdl),
            fused_fb,
        )
        if v2_key not in _compiled_cache:
            logger.info(
                f"kda_mtp_decode: compiling v2 variant H={HV} num_spec={num_spec} split_v={split_v}"
                f" pdl={bool(use_pdl)} fused_fb={fused_fb}"
            )
            _compiled_cache[v2_key] = cute.compile(
                _run_kda_decode_mtp_v2,
                *[_dlpack_arg(t, assumed_align=a) for t, a in v2_args],
                N,
                scale=scale,
                HV=HV,
                K=K,
                V=V_dim,
                NUM_SPEC=num_spec,
                KERNEL_WIDTH=W,
                lower_bound=float(lower_bound),
                onorm_eps=float(onorm_eps),
                USE_FLAT_LAYOUT=use_flat_layout,
                SPLIT_V=split_v,
                PDL=bool(use_pdl),
                FUSED_FB=fused_fb,
                stream=stream,
            )
        _compiled_cache[v2_key](*[_dlpack_arg(t, assumed_align=a) for t, a in v2_args], N, stream)
        return out

    # Throughput regime (one CTA per tile, more tiles than SMs): the compact smem/TMEM layout
    # lets two CTAs share an SM. Same arithmetic, so the results are bitwise identical. A
    # single partial wave keeps the default layout (slightly lower latency).
    compact = (
        split_v == 1
        and HV * N > _num_sms(x_q.device)
        and h0_arg.dtype == torch.bfloat16
        and bool(fuse_output_norm)
        and num_spec <= 3
        and not bf16_math
        and _compact_layout_enabled(x_q.device)
    )

    key = (
        x_q.dtype,
        scale,
        HV,
        K,
        V_dim,
        num_spec,
        W,
        pool_size,
        lower_bound,
        use_flat_layout,
        _layout_key(h0_arg),
        _layout_key(x_q_arg, dynamic_layout=True, assumed_align=16),
        _layout_key(x_k_arg, dynamic_layout=True, assumed_align=16),
        _layout_key(x_v_arg, dynamic_layout=True, assumed_align=16),
        _layout_key(w_q),
        _layout_key(w_k),
        _layout_key(w_v),
        _layout_key(cs_q),
        _layout_key(cs_k),
        _layout_key(cs_v),
        _layout_key(A_log),
        _layout_key(g, dynamic_layout=True, assumed_align=16),
        _layout_key(dt_bias),
        _layout_key(beta, dynamic_layout=True, assumed_align=16),
        _layout_key(onorm_g_arg, dynamic_layout=True, assumed_align=16),
        _layout_key(onorm_weight_arg),
        _layout_key(out, dynamic_layout=True, assumed_align=16),
        _layout_key(output_scale, assumed_align=output_scale_assumed_align),
        _layout_key(k_cache),
        _layout_key(g_cache),
        _layout_key(v_cache),
        _layout_key(beta_cache, assumed_align=beta_cache_assumed_align),
        _layout_key(ssm_state_indices, dynamic_layout=True, assumed_align=4),
        _layout_key(cu_seqlens_arg, dynamic_layout=True, assumed_align=4),
        _layout_key(num_accepted_tokens, dynamic_layout=True, assumed_align=4),
        use_setmaxreg,
        use_regular_metadata,
        bool(packed_token_layout),
        use_reg_q_weights,
        use_zero_accepted,
        float(onorm_eps),
        bool(fuse_output_norm),
        bool(quantize_output),
        split_v,
        bf16_math,
        compact,
    )

    if key not in _compiled_cache:
        logger.info(
            f"kda_mtp_decode: compiling variant N={N} H={HV} T={T_total} "
            f"num_spec={num_spec} zero_accepted={use_zero_accepted} "
            f"regular_metadata={use_regular_metadata} "
            f"fuse_output_norm={fuse_output_norm} "
            f"quantize_output={quantize_output} split_v={split_v} bf16_math={bf16_math} "
            f"compact={compact}"
        )
        _compiled_cache[key] = cute.compile(
            _run_kda_decode_mtp,
            _from_dlpack_arg(h0_arg),
            _dlpack_arg(x_q_arg, assumed_align=16),
            _dlpack_arg(x_k_arg, assumed_align=16),
            _dlpack_arg(x_v_arg, assumed_align=16),
            _from_dlpack_arg(w_q),
            _from_dlpack_arg(w_k),
            _from_dlpack_arg(w_v),
            _from_dlpack_arg(cs_q),
            _from_dlpack_arg(cs_k),
            _from_dlpack_arg(cs_v),
            _from_dlpack_arg(A_log),
            _dlpack_arg(g, assumed_align=16),
            _from_dlpack_arg(dt_bias),
            _dlpack_arg(beta, assumed_align=16),
            _dlpack_arg(onorm_g_arg, assumed_align=16),
            _from_dlpack_arg(onorm_weight_arg),
            _dlpack_arg(out, assumed_align=16),
            _from_dlpack_arg(output_scale, assumed_align=output_scale_assumed_align),
            _from_dlpack_arg(h0_arg),
            _from_dlpack_arg(k_cache),
            _from_dlpack_arg(g_cache),
            _from_dlpack_arg(v_cache),
            _from_dlpack_arg(beta_cache, assumed_align=beta_cache_assumed_align),
            _dlpack_arg(stage_timing_arg, assumed_align=16),
            _dlpack_arg(ssm_state_indices, assumed_align=4),
            _dlpack_arg(cu_seqlens_arg, assumed_align=4),
            _dlpack_arg(num_accepted_tokens, assumed_align=4),
            _from_dlpack_arg(precompute_control),
            scale=scale,
            HV=HV,
            K=K,
            V=V_dim,
            N=N,
            NUM_SPEC=num_spec,
            TILE_V=_TILE_V,
            KERNEL_WIDTH=W,
            lower_bound=lower_bound,
            onorm_eps=float(onorm_eps),
            scale_leading_dim=scale_leading_dim,
            USE_FLAT_LAYOUT=use_flat_layout,
            USE_SETMAXREG=use_setmaxreg,
            USE_REGULAR_METADATA=use_regular_metadata,
            USE_PACKED_TOKEN_LAYOUT=bool(packed_token_layout),
            USE_REG_Q_WEIGHTS=use_reg_q_weights,
            USE_ZERO_ACCEPTED=use_zero_accepted,
            FUSE_PRECOMPUTE=True,
            RUNTIME_PRECOMPUTE_FLAG=False,
            FUSE_OUTPUT_NORM=bool(fuse_output_norm),
            QUANTIZE_OUTPUT=bool(quantize_output),
            PROFILE_STAGES=profile_stages,
            SPLIT_V=split_v,
            BF16_MMA=bf16_math,
            stream=stream,
            COMPACT=compact,
        )

    # Runtime alignment policy:
    # - 16 B: model inputs/outputs, parameters, recurrent/conv/kg/v pools,
    #   output-norm tensors, and the persistent control scalar.
    # - beta_cache: derived from its actual per-layer physical span and dtype.
    # - 4 B: int32 scheduler metadata, which may be an element-offset view.
    # - output_scale: 16 B when separately allocated, otherwise it aliases
    #   int32 num_accepted_tokens and inherits its 4-byte guarantee.
    _compiled_cache[key](
        _dlpack_arg(h0_arg, assumed_align=16),
        _dlpack_arg(x_q_arg, assumed_align=16),
        _dlpack_arg(x_k_arg, assumed_align=16),
        _dlpack_arg(x_v_arg, assumed_align=16),
        _dlpack_arg(w_q, assumed_align=16),
        _dlpack_arg(w_k, assumed_align=16),
        _dlpack_arg(w_v, assumed_align=16),
        _dlpack_arg(cs_q, assumed_align=16),
        _dlpack_arg(cs_k, assumed_align=16),
        _dlpack_arg(cs_v, assumed_align=16),
        _dlpack_arg(A_log, assumed_align=16),
        _dlpack_arg(g, assumed_align=16),
        _dlpack_arg(dt_bias, assumed_align=16),
        _dlpack_arg(beta, assumed_align=16),
        _dlpack_arg(onorm_g_arg, assumed_align=16),
        _dlpack_arg(onorm_weight_arg, assumed_align=16),
        _dlpack_arg(out, assumed_align=16),
        _dlpack_arg(output_scale, assumed_align=output_scale_assumed_align),
        _dlpack_arg(h0_arg, assumed_align=16),
        _dlpack_arg(k_cache, assumed_align=16),
        _dlpack_arg(g_cache, assumed_align=16),
        _dlpack_arg(v_cache, assumed_align=16),
        _dlpack_arg(beta_cache, assumed_align=beta_cache_assumed_align),
        _dlpack_arg(stage_timing_arg, assumed_align=16),
        _dlpack_arg(ssm_state_indices, assumed_align=4),
        _dlpack_arg(cu_seqlens_arg, assumed_align=4),
        _dlpack_arg(num_accepted_tokens, assumed_align=4),
        _dlpack_arg(precompute_control, assumed_align=16),
        N,
        stream,
    )

    if quantize_output:
        return out, output_scale
    return out
