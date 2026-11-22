/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Target-private copy of the base-2 softmax of the SM120 e4m3 head-256 context FMHA.
//
// Additions to the stock cpp/kernels/fmha_v2/src/fmha/softmax.h: the
// Kernel_traits_store_o_from_regs / Kernel_traits_use_exp2_softmax / Softmax_uses_exp2 detectors, ex2_approx_ftz,
// exp2_update_o, exp2_update_sum, and the Softmax_qmma<Ada_qmma_e4m3_fp32_traits> specialization with the base-2
// path (USE_EXP2, s2_, exp_bias_, sum_to_base_units, apply_mask_alibi, and the USE_EXP2 branches of unpack /
// apply_exp_with_mask / pack), plus the unchanged Softmax<Ada_qmma_e4m3_fp32_traits> wrapper over it.
// Namespace TRTLLM_MV2_NS::fmha_sm120 instead of fmha (only the Ada e4m3 FP32 specializations exist
// here), the arch macro is renamed TRTLLM_MV2_FMHA_SM120_EXP2_ARCH, and the stock names are qualified
// (fmha::Softmax_imma, fmha::get_alibi_head_scaling_factor). The base classes (Softmax_imma, Softmax_base) and the
// reductions are the stock fmha ones.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <fmha/fragment.h>
#include <fmha/kernel_traits.h>
#include <fmha/softmax.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

#include <cfloat>
#include <type_traits>

// Base-2 softmax: an SM120 e4m3 tiled no-loop context kernel with the register-direct O epilogue
// (Kernel_traits::STORE_O_FROM_REGS) whose traits set USE_EXP2_SOFTMAX takes the softmax in base 2: the QK^T
// accumulators stay raw, the row max is taken in raw units, P = ex2.approx(acc * s2 - (max * s2 - log2(256))) with
// s2 = scale_bmm1 * log2(e), and the O / sum rescale is ex2.approx((prev - curr) * s2). Every other kernel, including
// every generated one, keeps the expf softmax.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define TRTLLM_MV2_FMHA_SM120_EXP2_ARCH 1
#else
#define TRTLLM_MV2_FMHA_SM120_EXP2_ARCH 0
#endif

TRTLLM_MV2_NAMESPACE_BEGIN

namespace fmha_sm120
{

// The unqualified stock names in the copy below (Max_, Sum_, ...) are the fmha ones.
using namespace ::fmha;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Whether a kernel traits type has the register-direct O epilogue (the SM120 e4m3 head-256 tiled no-loop kernels).
template <typename T, typename = void>
struct Kernel_traits_store_o_from_regs
{
    static constexpr bool VALUE = false;
};

template <typename T>
struct Kernel_traits_store_o_from_regs<T, std::void_t<decltype(T::STORE_O_FROM_REGS)>>
{
    static constexpr bool VALUE = T::STORE_O_FROM_REGS;
};

// Whether a kernel traits type asks for the base-2 softmax.
template <typename T, typename = void>
struct Kernel_traits_use_exp2_softmax
{
    static constexpr bool VALUE = false;
};

template <typename T>
struct Kernel_traits_use_exp2_softmax<T, std::void_t<decltype(T::USE_EXP2_SOFTMAX)>>
{
    static constexpr bool VALUE = T::USE_EXP2_SOFTMAX;
};

// Whether a Softmax type runs the base-2 path (USE_EXP2 is defined only by the e4m3 FP32 QMMA softmax).
template <typename T, typename = void>
struct Softmax_uses_exp2
{
    static constexpr bool VALUE = false;
};

template <typename T>
struct Softmax_uses_exp2<T, std::void_t<decltype(T::USE_EXP2)>>
{
    static constexpr bool VALUE = T::USE_EXP2;
};

static inline __device__ float ex2_approx_ftz(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// The base-2 forms of Tile_o_normalizer_fp32::update / Tile_o_normalizer<Ada_qmma_e4m3_fp32_traits>::update_sum, for
// row max values in raw accumulator units (s2 = scale_bmm1 * log2(e)).
template <typename Fragment_accu, int M, int N, int ROWS>
inline __device__ void exp2_update_o(Fragment_accu (&acc_o)[M][N], float (&curr_max)[ROWS],
    float const (&prev_max)[ROWS], float (&sum)[ROWS], float const s2)
{
    static_assert(ROWS == 2 * M, "");
#pragma unroll
    for (int mi = 0; mi < M; ++mi)
    {
        float alpha[2];
#pragma unroll
        for (int ii = 0; ii < 2; ++ii)
        {
            int jj = 2 * mi + ii;
            curr_max[jj] = fmax(prev_max[jj], curr_max[jj]);
            alpha[ii] = ex2_approx_ftz((prev_max[jj] - curr_max[jj]) * s2);
            sum[jj] *= alpha[ii];
        }
#pragma unroll
        for (int ni = 0; ni < N; ++ni)
        {
#pragma unroll
            for (int ii = 0; ii < Fragment_accu::NUM_ELTS; ++ii)
            {
                acc_o[mi][ni].elt(ii) = alpha[(ii & 2) / 2] * acc_o[mi][ni].elt(ii);
            }
        }
    }
}

// The attention-sink term (once per CTA): the row max is converted back to scaled units (scale = scale_bmm1) and the
// term is the expf of Tile_o_normalizer<Ada_qmma_e4m3_fp32_traits>::update_sum.
template <int ROWS>
inline __device__ void exp2_update_sum(
    float const sink, float const (&max)[ROWS], float (&sum)[ROWS], float const scale, float const quant_scale)
{
#pragma unroll
    for (int i = 0; i < ROWS; ++i)
    {
        sum[i] += expf(sink - max[i] * scale) * quant_scale;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only the Ada QMMA e4m3 FP32 specializations below are defined.
template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax_qmma;

template <typename Traits, typename Cta_tile, typename Kernel_traits>
struct Softmax;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax_qmma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
    : public fmha::Softmax_imma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{

    // The Traits
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = fmha::Softmax_imma<Traits, Cta_tile, Kernel_traits>;

    // The MMAs.
    static constexpr int MMAS_M = Base::MMAS_M;

    static constexpr int MMAS_N = Base::MMAS_N;

    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator<Traits>;

    // The base-2 softmax (see the top of this file): elt_ holds the raw accumulators, and the row max / O rescale are
    // in raw units. s2_ = scale_bmm1 * log2(e). The softmax scale (scale_softmax, which the expf path multiplies into
    // P before the e4m3 cast) is folded into the exponent too:
    //   P_e4m3 = e4m3(2^(raw * s2 - (max * s2 - 8 - log2(scale_softmax)))),
    // so the row sums carry the factor scale_softmax; sum_to_base_units() (1 / scale_softmax) takes them back to the
    // units of the expf path before the attention-sink term and the final normalization.
    static constexpr bool USE_EXP2 = TRTLLM_MV2_FMHA_SM120_EXP2_ARCH
        && Kernel_traits_store_o_from_regs<Kernel_traits>::VALUE
        && Kernel_traits_use_exp2_softmax<Kernel_traits>::VALUE;
    static_assert(Traits::SOFTMAX_FP_QUANT_SCALE == 256.f, "log2(SOFTMAX_FP_QUANT_SCALE) = 8");
    static constexpr float LOG2_SOFTMAX_FP_QUANT_SCALE = 8.f;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_qmma(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
        , params_scale_bmm1_(params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1)
        , params_scale_softmax_(params.scale_softmax)
        , s2_(reinterpret_cast<float const&>(params_scale_bmm1_) * 1.4426950408889634f)
        , exp_bias_(LOG2_SOFTMAX_FP_QUANT_SCALE + log2f(reinterpret_cast<float const&>(params_scale_softmax_)))
    {
    }

    // The factor that takes the base-2 path's row sums to the units of the expf path.
    inline __device__ float sum_to_base_units() const
    {
        return 1.f / reinterpret_cast<float const&>(params_scale_softmax_);
    }

    // ALiBi on raw units: (raw * scale) * scale_after_alibi + m * col, divided by scale (the base-2 path only).
    template <typename Mask, typename AlibiParams>
    inline __device__ void apply_mask_alibi(Mask const& mask, int head_id, AlibiParams const& alibi_params)
    {
        if constexpr (USE_EXP2 && Kernel_traits::CAUSAL_MASK)
        {
            float const inv_scale = 1.f / reinterpret_cast<float const&>(params_scale_bmm1_);
            float m = fmha::get_alibi_head_scaling_factor<AlibiParams>(head_id, alibi_params) * inv_scale;
#pragma unroll
            for (int mi = 0; mi < MMAS_M; ++mi)
            {
#pragma unroll
                for (int ii = 0; ii < 2; ++ii)
                {
#pragma unroll
                    for (int ni = 0; ni < MMAS_N; ++ni)
                    {
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                        {
                            int row, col;
                            mask.get_row_col(row, col, mi, ni, ii, jj);
                            if (mask.is_valid(row, col))
                            {
                                this->elt_[2 * mi + ii][4 * ni + jj]
                                    = this->elt_[2 * mi + ii][4 * ni + jj] * alibi_params.scale_after_alibi
                                    + m * (col + alibi_params.sequence_pos_offset);
                            }
                            else
                            {
                                this->elt_[2 * mi + ii][4 * ni + jj] = -FLT_MAX;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Base::apply_mask_alibi(mask, head_id, alibi_params);
        }
    }

    // Store the tile after softmax.
    template <typename Gmem_tile>
    inline __device__ void store(Gmem_tile& gmem_tile)
    {

        float const scale = reinterpret_cast<float const&>(params_scale_softmax_);
        Accumulator acc[MMAS_M][MMAS_N];
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // scale
                acc[mi][ni].ele(0) = this->elt_[2 * mi + 0][4 * ni + 0] * scale;
                acc[mi][ni].ele(1) = this->elt_[2 * mi + 0][4 * ni + 1] * scale;
                acc[mi][ni].ele(4) = this->elt_[2 * mi + 0][4 * ni + 2] * scale;
                acc[mi][ni].ele(5) = this->elt_[2 * mi + 0][4 * ni + 3] * scale;
                acc[mi][ni].ele(2) = this->elt_[2 * mi + 1][4 * ni + 0] * scale;
                acc[mi][ni].ele(3) = this->elt_[2 * mi + 1][4 * ni + 1] * scale;
                acc[mi][ni].ele(6) = this->elt_[2 * mi + 1][4 * ni + 2] * scale;
                acc[mi][ni].ele(7) = this->elt_[2 * mi + 1][4 * ni + 3] * scale;
            }
        }

        // Delegate to the gmem tile to store.
        // TODO: need fp32 to fp8 conversion (move this to gmem_tile)
        gmem_tile.store(acc);
    }

    // Convert from accumulators to floats.
    inline __device__ void unpack(Accumulator const (&acc)[MMAS_M][MMAS_N])
    {
        // The base-2 path keeps the raw accumulators (the scale is folded into the exponent).
        float const scale = USE_EXP2 ? 1.f : reinterpret_cast<float const&>(params_scale_bmm1_);
#pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi)
        {
#pragma unroll
            for (int ni = 0; ni < MMAS_N; ++ni)
            {

                // Convert to FP32 and scale.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scale;
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scale;
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scale;
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scale;
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scale;
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scale;
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scale;
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scale;
            }
        }
    }

    template <bool APPLY_MASK = false>
    inline __device__ void apply_exp_with_mask(float const (&max)[MMAS_M * 2])
    {
        if constexpr (USE_EXP2)
        {
            // P * 256 * scale_softmax = 2^(raw * s2 - (max * s2 - 8 - log2(scale_softmax))); a fully masked row
            // (max == -FLT_MAX) gives 0, as on the expf path.
#pragma unroll
            for (int mi = 0; mi < MMAS_M * 2; ++mi)
            {
                float const offset = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : fmaf(max[mi], s2_, -exp_bias_);
#pragma unroll
                for (int ni = 0; ni < MMAS_N * 4; ++ni)
                {
                    this->elt_[mi][ni] = ex2_approx_ftz(fmaf(this->elt_[mi][ni], s2_, -offset));
                }
            }
            return;
        }
#pragma unroll
        for (int mi = 0; mi < MMAS_M * 2; ++mi)
        {
            float max_val = APPLY_MASK && max[mi] == -FLT_MAX ? 0.f : (max[mi] - logf(Traits::SOFTMAX_FP_QUANT_SCALE));
#pragma unroll
            for (int ni = 0; ni < MMAS_N * 4; ++ni)
            {
                this->elt_[mi][ni] = expf(this->elt_[mi][ni] - max_val);
            }
        }
    }

    // Pack the data to a fragment for the next GEMM.
    template <typename Fragment_a, int K, int M>
    inline __device__ void pack(Fragment_a (&dst)[K][M]) const
    {
        // The base-2 path has scale_softmax in the exponent already.
        float const scale = USE_EXP2 ? 1.f : reinterpret_cast<float const&>(this->params_scale_softmax_);

// The canonical layout in K should be R0: [0,1,2,3] R2: [16,17,18,19]
// Note below that this is not possible with the register layout of the accumulator.
#pragma unroll
        for (int mi = 0; mi < M; ++mi)
        {
#pragma unroll
            for (int ki = 0; ki < K; ++ki)
            {
                // 1st row - 8 elements per row.
                float tmp_00 = this->elt_[2 * mi + 0][8 * ki + 0] * scale; // + 0
                float tmp_01 = this->elt_[2 * mi + 0][8 * ki + 1] * scale; // + 1
                float tmp_02 = this->elt_[2 * mi + 0][8 * ki + 2] * scale; // + 8
                float tmp_03 = this->elt_[2 * mi + 0][8 * ki + 3] * scale; // + 9
                float tmp_04 = this->elt_[2 * mi + 0][8 * ki + 4] * scale; // +16
                float tmp_05 = this->elt_[2 * mi + 0][8 * ki + 5] * scale; // +17
                float tmp_06 = this->elt_[2 * mi + 0][8 * ki + 6] * scale; // +24
                float tmp_07 = this->elt_[2 * mi + 0][8 * ki + 7] * scale; // +25

                // 2nd row - 4 elements per row.
                float tmp_10 = this->elt_[2 * mi + 1][8 * ki + 0] * scale; // + 0
                float tmp_11 = this->elt_[2 * mi + 1][8 * ki + 1] * scale; // + 1
                float tmp_12 = this->elt_[2 * mi + 1][8 * ki + 2] * scale; // + 8
                float tmp_13 = this->elt_[2 * mi + 1][8 * ki + 3] * scale; // + 9
                float tmp_14 = this->elt_[2 * mi + 1][8 * ki + 4] * scale; // +16
                float tmp_15 = this->elt_[2 * mi + 1][8 * ki + 5] * scale; // +17
                float tmp_16 = this->elt_[2 * mi + 1][8 * ki + 6] * scale; // +24
                float tmp_17 = this->elt_[2 * mi + 1][8 * ki + 7] * scale; // +25

                // Pack to 4 registers.
                dst[ki][mi].reg(0) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_00, tmp_01, tmp_02, tmp_03);
                dst[ki][mi].reg(1) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_10, tmp_11, tmp_12, tmp_13);
                dst[ki][mi].reg(2) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_04, tmp_05, tmp_06, tmp_07);
                dst[ki][mi].reg(3) = fmha::float4_to_fp8x4<Traits::A_type>(tmp_14, tmp_15, tmp_16, tmp_17);
            }
        }
    }

    // The scaling factors.
    uint32_t const params_scale_bmm1_, params_scale_softmax_;
    // scale_bmm1 * log2(e), and log2(SOFTMAX_FP_QUANT_SCALE * scale_softmax) (used by the base-2 softmax only).
    float const s2_, exp_bias_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
    : public Softmax_qmma<fmha::Ada_qmma_e4m3_fp32_traits, Cta_tile, Kernel_traits>
{

    // The traits.
    using Traits = fmha::Ada_qmma_e4m3_fp32_traits;
    // The base class.
    using Base = Softmax_qmma<Traits, Cta_tile, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, void* smem, int bidb, int tidx)
        : Base(params, smem, bidb, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha_sm120

TRTLLM_MV2_NAMESPACE_END
