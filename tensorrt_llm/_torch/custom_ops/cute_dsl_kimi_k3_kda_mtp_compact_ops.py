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

"""Host launcher of the compact KDA MTP verify kernel.

``_run_kda_decode_mtp_compact`` is the ``_run_kda_decode_mtp`` of ``feat/k3_26.08``
(fp32 / tf32 tiled MMAs; with ``COMPACT_TWO_CTAS`` the ``min_blocks_per_mp=2``
launch) with the split ``k_cache`` / ``g_cache`` replay caches of this tree. It
launches ``kda_decode_mtp_compact_kernel`` (``kda_mtp_decode_compact.py``).

It takes the in-tree launcher's arguments, except that ``COMPACT`` /
``COMPACT_TWO_CTAS`` replace ``BF16_MMA``. ``COMPACT=False`` runs the vendored
kernel in its default layout; the parity tests use that to isolate the
replay-cache port from the layout change.

``cute_dsl_kimi_k3_kda_mtp_ops`` imports this module only when
``cute_dsl_kimi_k3_kda_mtp_compact_policy.select_compact_layout`` first selects
the compact layout, so the in-tree path never loads the vendored kernel.
"""

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.kda_mtp_decode_compact import (
        NUM_THREADS,
        T_PAD,
        kda_decode_mtp_compact_kernel,
    )
else:
    raise ImportError("Kimi K3 KDA MTP decode (compact layout) requires NVIDIA CUTLASS DSL")


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _run_kda_decode_mtp_compact(
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
        stream: cuda.CUstream,
        COMPACT: cutlass.Constexpr[bool] = False,
        COMPACT_TWO_CTAS: cutlass.Constexpr[bool] = False,
        USE_PDL: cutlass.Constexpr[bool] = False,
    ):
        if cutlass.const_expr(USE_ZERO_ACCEPTED):
            t_max = 1 + NUM_SPEC
        else:
            t_max = 2 * NUM_SPEC + 1
        smem_qk_layout = cute.make_layout((t_max, K), stride=(K, 1))
        # GEMM1: M = V rows of the state (TMEM), N = 3 * T_PAD operand rows, tf32.
        tiled_mma1 = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, 3 * T_PAD),
            tcgen05.OperandSource.TMEM,
        )
        smem_layout_b1 = sm100_utils.make_smem_layout_b(
            tiled_mma1, (V, 3 * T_PAD, K), cutlass.Float32, 1
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
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2h = sm100_utils.make_smem_layout_b(
            tiled_mma2h, (V, K // 2, T_PAD), cutlass.Float32, 1
        )
        smem_layout_a2 = sm100_utils.make_smem_layout_a(
            tiled_mma2h, (V, K // 2, T_PAD), cutlass.Float32, 1
        )
        # GEMM2 half 0 also carries the output-combination GEMM: N = T_PAD (Bcoef) + K/2 (Kbar rows 0..63).
        tiled_mma2q = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (V, T_PAD + K // 2),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_b2q = sm100_utils.make_smem_layout_b(
            tiled_mma2q, (V, T_PAD + K // 2, T_PAD), cutlass.Float32, 1
        )
        # GEMM3: T x T coupling matrices, A3 = [Kf; Qf] padded to 64 rows, B3 = Kb (16 rows).
        tiled_mma3 = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (4 * T_PAD, T_PAD),
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_a3 = sm100_utils.make_smem_layout_a(
            tiled_mma3, (4 * T_PAD, T_PAD, K), cutlass.Float32, 1
        )
        smem_layout_b3 = sm100_utils.make_smem_layout_b(
            tiled_mma3, (4 * T_PAD, T_PAD, K), cutlass.Float32, 1
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
        launched = kda_decode_mtp_compact_kernel(
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
            COMPACT,
            COMPACT_TWO_CTAS,
            USE_PDL,
        )
        if cutlass.const_expr(SPLIT_V == 2):
            # cluster pair: the two CTAs of a (head, request) must be co-scheduled (in-place state)
            launched.launch(
                grid=(HV, N, 2),
                block=[NUM_THREADS, 1, 1],
                cluster=(1, 1, 2),
                stream=stream,
                use_pdl=USE_PDL,
            )
        elif cutlass.const_expr(COMPACT_TWO_CTAS):
            # Two co-resident 512-thread CTAs need <= 64 registers per thread. Without the
            # launch bound, the sm_107a build allocates 75 and only one CTA fits per SM.
            launched.launch(
                grid=(HV, N, 1),
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                min_blocks_per_mp=2,
                use_pdl=USE_PDL,
            )
        else:
            launched.launch(
                grid=(HV, N, 1), block=[NUM_THREADS, 1, 1], stream=stream, use_pdl=USE_PDL
            )
