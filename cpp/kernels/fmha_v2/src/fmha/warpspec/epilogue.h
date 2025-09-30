/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <fmha/softmax.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{
namespace ws
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Special Softmax struct to handle optimization tricks on Hopper Warp-Specialized Kernels.
template <template <int, int, int, bool, bool> class Traits, typename Kernel_traits>
struct Softmax_base
{

    // The instruction traits for BMM1.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Kernel_traits::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // The step size of KV loop.
    enum
    {
        STEP_KV = Kernel_traits::STEP_KV
    };

    // Whether apply causal mask or not.
    enum
    {
        CAUSAL_MASK = Kernel_traits::CAUSAL_MASK
    };

    // Whether do we attend to the specific sliding window or chunk ?
    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = Kernel_traits::SLIDING_OR_CHUNKED_ATTENTION
    };

    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    enum
    {
        APPLY_ALIBI = Kernel_traits::APPLY_ALIBI
    };

    // Are we applying softcapping scale for qk products ?
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Kernel_traits::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Do we use custom mask input ?
    enum
    {
        USE_CUSTOM_MASK = Kernel_traits::USE_CUSTOM_MASK
    };

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    enum
    {
        EXP2F_OPTIMIZATION = Kernel_traits::EXP2F_OPTIMIZATION
    };

    // Whether we need to check if local_max could be -inf or not.
    enum
    {
        CHECK_IF_NEG_INF_EXISTS = SLIDING_OR_CHUNKED_ATTENTION || USE_CUSTOM_MASK
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_base(Params params, int tidx)
        : tidx_(tidx)
        , scale_bmm1_(params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1)
        , softcapping_scale_bmm1_(params.softcapping_scale_bmm1)
        , sliding_window_size_(params.sliding_window_size)
        , log2_chunked_attention_size_(params.log2_chunked_attention_size)
        , packed_mask_ptr_{reinterpret_cast<uint32_t*>(params.packed_mask_ptr)}
        , params_packed_mask_stride_in_bytes_{params.packed_mask_stride_in_bytes}
    {

        int warp = tidx / 32;
        int lane = tidx % 32;
        // The corresponding row/col for each thread after MMA.
        // fixed 4x1 warp layout.
        quad_col_ = lane % 4;
        if (CAUSAL_MASK)
        {
            quad_row_ = warp * 16 + lane / 4;
        }
    }

    // Compute the sliding window or chunk start.
    inline __device__ int compute_sliding_window_or_chunk_start(int row)
    {
        // If the chunked atteniton is used.
        if (log2_chunked_attention_size_ > 0)
        {
            // The attention chunk start.
            return (row >> log2_chunked_attention_size_) << log2_chunked_attention_size_;
        }
        else
        {
            // The sliding window start is the max of 0 and row - sliding_window_size.
            return max(0, row + 1 - sliding_window_size_);
        }
    }

    // Load the packed mask in global memory.
    inline __device__ void load_packed_mask(int row_offset, int col_offset)
    {
        if constexpr (USE_CUSTOM_MASK)
        {
            static_assert(Mma_tile_p::CORES_M == 2, "Not implemented!");
            // Note that row_offset takes sum_s into consideration.
            // row_offset % 64 == 0.
            // 32 bits per thread, so 128 bits (32 bytes) per mma row (4 threads).
            int64_t mask_row_offset_in_bytes = row_offset * params_packed_mask_stride_in_bytes_;
            // offset_in_bytes = (tidx_ * 32 + (col_offset / (16 * 4)) * 128 * 32) / 8.
            // note that col_offset % 64 == 0 here.
            int64_t mask_col_offset_in_bytes = tidx_ * 4 + col_offset * Cta_tile_p::THREADS_PER_CTA / 16;
            // add the two offsets for uint32 packed mask.
            int64_t mask_offset = (mask_row_offset_in_bytes + mask_col_offset_in_bytes) / 4;
            if constexpr (STEP_KV == 64)
            {
                // 32 bits (2 rows, 16 cols) are needed.
                packed_mask_.x = packed_mask_ptr_[mask_offset];
            }
            else if constexpr (STEP_KV == 128)
            {
                // 2 x 32 bits (4 rows, 16 cols) are needed.
                packed_mask_.x = packed_mask_ptr_[mask_offset];
                packed_mask_.y = packed_mask_ptr_[mask_offset + 128];
            }
            else if constexpr (STEP_KV == 256)
            {
                // 4 x 32 bits (4 rows, 16 cols) are needed.
                packed_mask_.x = packed_mask_ptr_[mask_offset];
                packed_mask_.y = packed_mask_ptr_[mask_offset + 128];
                packed_mask_.z = packed_mask_ptr_[mask_offset + 256];
                packed_mask_.w = packed_mask_ptr_[mask_offset + 384];
            }
        }
    }

    // Check if the two positions are valid or not.
    inline __device__ void valid_positions(int mi, int ni, bool& v0, bool& v1)
    {
        // Only need one uint32_t packed mask in this case.
        if constexpr (STEP_KV == 64)
        {
            // Packed mask input.
            v0 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 0));
            v1 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 1));
        }
        else if constexpr (STEP_KV == 128)
        {
            // Packed mask input.
            if (ni < 8)
            {
                v0 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 0));
                v1 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 1));
            }
            else
            {
                v0 = packed_mask_.y & (1 << (ni * 4 + mi * 2 + 0 - 32));
                v1 = packed_mask_.y & (1 << (ni * 4 + mi * 2 + 1 - 32));
            }
        }
        else if constexpr (STEP_KV == 256)
        {
            // KV step size is 256 in this case (i.e CORES_N = 32).
            if (ni < 8)
            {
                v0 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 0));
                v1 = packed_mask_.x & (1 << (ni * 4 + mi * 2 + 1));
            }
            else if (ni < 16)
            {
                v0 = packed_mask_.y & (1 << (ni * 4 + mi * 2 + 0 - 32));
                v1 = packed_mask_.y & (1 << (ni * 4 + mi * 2 + 1 - 32));
            }
            else if (ni < 24)
            {
                v0 = packed_mask_.z & (1 << (ni * 4 + mi * 2 + 0 - 64));
                v1 = packed_mask_.z & (1 << (ni * 4 + mi * 2 + 1 - 64));
            }
            else
            {
                v0 = packed_mask_.w & (1 << (ni * 4 + mi * 2 + 0 - 96));
                v1 = packed_mask_.w & (1 << (ni * 4 + mi * 2 + 1 - 96));
            }
        }
    }

    // Convert from bmm1 output fragments to floats.
    inline __device__ void unpack(Compute_tile_p& ctile_p)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
                // Satfinite in case of overflow (due to fp16 accumulation).
                // When there is no alibi bias, we fuse bmm1_scale with -max by FMAs.
                uint32_t scaled_h2 = EXP2F_OPTIMIZATION
                    ? satfinite_h2(ctile_p.acc_[0][0].reg(ni * Mma_tile_p::CORES_M + mi))
                    : satfinite_h2(hmul2(ctile_p.acc_[0][0].reg(ni * Mma_tile_p::CORES_M + mi), scale_bmm1_));
                // Convert from half2 to float2.
                reinterpret_cast<float2*>(&elt_[mi][2 * ni])[0] = half2_to_float2(scaled_h2);
            }
        }
    }

    // Convert from bmm1 output fragments to floats.
    template <bool APPLY_MASK, typename AlibiParams>
    inline __device__ void apply_alibi_and_mask(Compute_tile_p& ctile_p, AlibiParams const& alibi_params,
        float const alibi_head_scale, int actual_seqlen, int row_offset, int col_offset)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {

                bool v0 = true, v1 = true;
                int col = 0;
                if constexpr (APPLY_MASK)
                {
                    // Custom mask input.
                    if constexpr (USE_CUSTOM_MASK)
                    {
                        // Packed mask input.
                        valid_positions(mi, ni, v0, v1);
                        // Causal mask.
                    }
                    else if constexpr (CAUSAL_MASK)
                    {
                        // Causal Mask: we have to apply mask before getting max.
                        int row = row_offset + quad_row_ + mi * 8;
                        col = col_offset + quad_col_ * 2 + ni * 8;
                        // Mask for the two N elements.
                        v0 = (col <= row);
                        v1 = (col + 1 <= row);

                        // Attend to the specific sliding window or chunk.
                        if constexpr (SLIDING_OR_CHUNKED_ATTENTION)
                        {
                            int sliding_window_or_chunk_start = compute_sliding_window_or_chunk_start(row);
                            v0 &= (col >= sliding_window_or_chunk_start);
                            v1 &= (col + 1 >= sliding_window_or_chunk_start);
                        }
                        // Dense(padding) mask.
                    }
                    else
                    {
                        col = col_offset + quad_col_ * 2 + ni * 8;
                        v0 = (col < actual_seqlen);
                        v1 = (col + 1 < actual_seqlen);
                    }
                }

                // The unpacked floats from the array.
                float2& f2 = reinterpret_cast<float2*>(&elt_[mi][2 * ni])[0];

                // Attention logit softcapping scale.
                // 1.0f / softcapping_scale has been fused into scale_bmm1.
                if constexpr (ENABLE_BMM1_SOFTCAPPING_SCALE)
                {
                    f2.x = softcapping_scale_bmm1_ * fmha::__tanhf(f2.x);
                    f2.y = softcapping_scale_bmm1_ * fmha::__tanhf(f2.y);
                }

                // Use minimum value of float here to avoid generating NANs with expf.
                if constexpr (APPLY_ALIBI)
                {
                    f2.x = v0 ? (f2.x * alibi_params.scale_after_alibi
                               + (col + alibi_params.sequence_pos_offset) * alibi_head_scale)
                              : -FLT_MAX;
                    f2.y = v1 ? (f2.y * alibi_params.scale_after_alibi
                               + (col + 1 + alibi_params.sequence_pos_offset) * alibi_head_scale)
                              : -FLT_MAX;
                }
                else
                {
                    f2.x = v0 ? f2.x : -FLT_MAX;
                    f2.y = v1 ? f2.y : -FLT_MAX;
                }
            }
        }
    }

    // Calculate max/sum, and update flash-attention scales.
    template <bool IS_FIRST_COL>
    inline __device__ void compute_and_update_scale(
        float (&global_max)[Mma_tile_p::CORES_M], float (&global_sum)[Mma_tile_p::CORES_M])
    {
        float const scale = reinterpret_cast<float const&>(scale_bmm1_);

// Row-wise max of current tile.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            if (IS_FIRST_COL)
            {
                local_max_[mi] = elt_[mi][0];
            }
            else
            {
                local_max_[mi] = fmaxf(global_max[mi], elt_[mi][0]);
            }
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                local_max_[mi] = fmaxf(local_max_[mi], elt_[mi][ni]);
            }
            local_max_[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 1), local_max_[mi]);
            local_max_[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 2), local_max_[mi]);
        }

// Softmax Exp.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {

                float& p0 = elt_[mi][2 * ni + 0];
                float& p1 = elt_[mi][2 * ni + 1];

                // When all elts of the tile are -FLT_MAX, we have to make sure
                //  expf generates 0 for all values instead of 1.
                if constexpr (!EXP2F_OPTIMIZATION)
                {
                    float masked_max = (!CHECK_IF_NEG_INF_EXISTS || local_max_[mi] != -FLT_MAX) ? local_max_[mi] : 0.f;
                    p0 = expf(p0 - masked_max);
                    p1 = expf(p1 - masked_max);
                }
                else
                {
                    // Use exp2f optimization for cases without alibi.
                    float masked_max
                        = (!CHECK_IF_NEG_INF_EXISTS || local_max_[mi] != -FLT_MAX) ? local_max_[mi] * scale : 0.f;
                    p0 = custom_exp2f(p0, scale, masked_max);
                    p1 = custom_exp2f(p1, scale, masked_max);
                }
            }
        }

// Row-wise sum of current tile.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            local_sum_[mi] = elt_[mi][0];
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                local_sum_[mi] += elt_[mi][ni];
            }
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 1);
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 2);
        }

        // Initialize or update the global sum and max.
        if (IS_FIRST_COL)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                global_sum[mi] = local_sum_[mi];
                global_max[mi] = local_max_[mi];
            }
        }
        else
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                float max_old = global_max[mi];
                float max_new = local_max_[mi];
                float sum_old = global_sum[mi];
                float sum_new = local_sum_[mi];
                // Remove the old max and replace by the new one.
                if constexpr (!EXP2F_OPTIMIZATION)
                {
                    correction_[mi] = expf(max_old - max_new);
                }
                else
                {
                    // Use exp2f optimization for cases without alibi.
                    correction_[mi] = exp2f((max_old - max_new) * scale);
                }
                global_sum[mi] = sum_old * correction_[mi] + sum_new;

                // New max already takes into account old max.
                global_max[mi] = max_new;
            }
        }
    }

    // Update flash attention scales and pack elements for BMM2.
    template <bool IS_FIRST_COL>
    inline __device__ void pack(Compute_tile_o& ctile_o, Fragment_p (&frag_p)[Mma_tile_o::MMAS_K])
    {

// Pack 4 cols for BMM2 A tile.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_o::MMAS_K; ni++)
        {
            frag_p[ni].reg(0) = float2_to_half2(elt_[0][4 * ni + 0], elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) = float2_to_half2(elt_[1][4 * ni + 0], elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) = float2_to_half2(elt_[0][4 * ni + 2], elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) = float2_to_half2(elt_[1][4 * ni + 2], elt_[1][4 * ni + 3]);
        }

        if (!IS_FIRST_COL)
        {
// Correct accumulators to current max.
#pragma unroll
            for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
            {
                const uint32_t scale = float_to_half2(correction_[mi]);

// Assume only N has multiple MMAs (MMAS_M = 1).
// MMAS_N > 1 when N dimension is split.
#pragma unroll
                for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
                {
#pragma unroll
                    for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                    {
                        uint32_t& reg = ctile_o.acc_[0][mma_ni].reg(ni * Mma_tile_o::CORES_M + mi);
                        reg = hmul2(reg, scale);
                    }
                }
            }
        }
        else
        {
            ctile_o.clear();
        }
    }

    // BMM1 scale.
    const uint32_t scale_bmm1_;
    // BMM1 softcapping scale.
    float const softcapping_scale_bmm1_;

    // The sliding window size.
    int const sliding_window_size_;
    // The log2 attention chunk size.
    int const log2_chunked_attention_size_;

    // The thread idx in the warp group.
    int tidx_;
    // The col index for the mma thread layout.
    int quad_col_;
    // The row index for the mma thread layout.
    int quad_row_;

    // The packed mask ptr.
    uint32_t const* packed_mask_ptr_;
    // The packed mask k-dim stride in bytes;
    const int64_t params_packed_mask_stride_in_bytes_;

    // Unpacked BMM1 output buffer.
    float elt_[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N * 2];
    // Local max.
    float local_max_[Mma_tile_p::CORES_M];
    // Local sum.
    float local_sum_[Mma_tile_p::CORES_M];
    // Correction_ scales for ctil_o.
    float correction_[Mma_tile_p::CORES_M];
    // The packed mask.
    uint4 packed_mask_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <template <int, int, int, bool, bool> class Traits, typename Kernel_traits>
struct Softmax
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp16_traits
template <typename Kernel_traits>
struct Softmax<Hopper_hgmma_fp16_traits, Kernel_traits> : public Softmax_base<Hopper_hgmma_fp16_traits, Kernel_traits>
{

    // The Base class.
    using Base = Softmax_base<Hopper_hgmma_fp16_traits, Kernel_traits>;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, int tidx)
        : Base(params, tidx)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fp32 accumulation traits.
template <template <int, int, int, bool, bool> class Traits, typename Kernel_traits>
struct Softmax_fp32_base : public Softmax_base<Traits, Kernel_traits>
{

    // The Base class.
    using Base = Softmax_base<Traits, Kernel_traits>;

    // The instruction traits for BMM1.
    using Traits_p = typename Base::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Base::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Base::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Base::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Whether apply causal mask or not.
    enum
    {
        CAUSAL_MASK = Base::CAUSAL_MASK
    };

    // Do we use custom mask input ?
    enum
    {
        USE_CUSTOM_MASK = Base::USE_CUSTOM_MASK
    };

    // Whether we attend to the specific sliding window or chunk ?
    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = Base::SLIDING_OR_CHUNKED_ATTENTION
    };

    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    enum
    {
        APPLY_ALIBI = Base::APPLY_ALIBI
    };

    // Are we applying softcapping_scale for qk products ?
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Base::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    enum
    {
        EXP2F_OPTIMIZATION = Base::EXP2F_OPTIMIZATION
    };

    // Whether we need to check if local_max could be -inf or not.
    enum
    {
        CHECK_IF_NEG_INF_EXISTS = Base::CHECK_IF_NEG_INF_EXISTS
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax_fp32_base(Params const& params, int tidx)
        : Base(params, tidx)
    {
    }

    // Convert from bmm1 output fragments to floats.
    inline __device__ void unpack(Compute_tile_p& ctile_p)
    {
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
                float2 f2;
                f2.x = ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi);
                f2.y = ctile_p.acc_[0][0].elt(2 * ni * Mma_tile_p::CORES_M + 2 * mi + 1);

                float const scale = reinterpret_cast<float const&>(this->scale_bmm1_);
                f2.x = EXP2F_OPTIMIZATION ? f2.x : f2.x * scale;
                f2.y = EXP2F_OPTIMIZATION ? f2.y : f2.y * scale;

                // Store to elt array.
                reinterpret_cast<float2*>(&this->elt_[mi][2 * ni])[0] = f2;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp32_traits
template <typename Kernel_traits>
struct Softmax<Hopper_hgmma_fp32_traits, Kernel_traits>
    : public Softmax_fp32_base<Hopper_hgmma_fp32_traits, Kernel_traits>
{

    // The Base class.
    using Base = Softmax_fp32_base<Hopper_hgmma_fp32_traits, Kernel_traits>;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, int tidx)
        : Base(params, tidx)
    {
    }

    // Update flash attention scales and pack elements for BMM2.
    template <bool IS_FIRST_COL>
    inline __device__ void pack(Compute_tile_o& ctile_o, Fragment_p (&frag_p)[Mma_tile_o::MMAS_K])
    {

// Pack 4 cols for BMM2 A tile.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_o::MMAS_K; ni++)
        {
            frag_p[ni].reg(0) = float2_to_half2(this->elt_[0][4 * ni + 0], this->elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) = float2_to_half2(this->elt_[1][4 * ni + 0], this->elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) = float2_to_half2(this->elt_[0][4 * ni + 2], this->elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) = float2_to_half2(this->elt_[1][4 * ni + 2], this->elt_[1][4 * ni + 3]);
        }

        if (!IS_FIRST_COL)
        {
// Correct accumulators to current max.
#pragma unroll
            for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
            {
                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimension is split.
                float correction = this->correction_[mi];
#pragma unroll
                for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
                {
#pragma unroll
                    for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                    {
                        uint32_t& reg0 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                        uint32_t& reg1 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg0) : "f"(correction));
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg1) : "f"(correction));
                    }
                }
            }
        }
        else
        {
            ctile_o.clear();
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_bf16_traits
template <typename Kernel_traits>
struct Softmax<Hopper_hgmma_bf16_traits, Kernel_traits>
    : public Softmax_fp32_base<Hopper_hgmma_bf16_traits, Kernel_traits>
{

    // The Base class.
    using Base = Softmax_fp32_base<Hopper_hgmma_bf16_traits, Kernel_traits>;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, int tidx)
        : Base(params, tidx)
    {
    }

    // Update flash attention scales and pack elements for BMM2.
    template <bool IS_FIRST_COL>
    inline __device__ void pack(Compute_tile_o& ctile_o, Fragment_p (&frag_p)[Mma_tile_o::MMAS_K])
    {

// Pack 4 cols for BMM2 A tile.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_o::MMAS_K; ni++)
        {
            frag_p[ni].reg(0) = float2_to_bf16_x2(this->elt_[0][4 * ni + 0], this->elt_[0][4 * ni + 1]);
            frag_p[ni].reg(1) = float2_to_bf16_x2(this->elt_[1][4 * ni + 0], this->elt_[1][4 * ni + 1]);
            frag_p[ni].reg(2) = float2_to_bf16_x2(this->elt_[0][4 * ni + 2], this->elt_[0][4 * ni + 3]);
            frag_p[ni].reg(3) = float2_to_bf16_x2(this->elt_[1][4 * ni + 2], this->elt_[1][4 * ni + 3]);
        }

        if (!IS_FIRST_COL)
        {
// Correct accumulators to current max.
#pragma unroll
            for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
            {
                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimension is split.
                float correction = this->correction_[mi];
#pragma unroll
                for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
                {
#pragma unroll
                    for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                    {
                        uint32_t& reg0 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                        uint32_t& reg1 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg0) : "f"(correction));
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg1) : "f"(correction));
                    }
                }
            }
        }
        else
        {
            ctile_o.clear();
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_qgmma_e4m3_fp32_traits
template <typename Kernel_traits>
struct Softmax<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>
    : public Softmax_fp32_base<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>
{

    // The Base class.
    using Base = Softmax_fp32_base<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>;

    // The instruction traits for BMM1.
    using Traits_p = typename Base::Traits_p;
    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM1.
    using Cta_tile_p = typename Base::Cta_tile_p;
    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM1.
    using Compute_tile_p = typename Base::Compute_tile_p;
    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM1.
    using Mma_tile_p = typename Base::Mma_tile_p;
    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // The fragment of BMM1 output.
    using Fragment_p = typename Compute_tile_o::Fragment;

    // Whether apply causal mask or not.
    enum
    {
        CAUSAL_MASK = Base::CAUSAL_MASK
    };

    // Whether we attend to the specific sliding window or chunk ?
    enum
    {
        SLIDING_OR_CHUNKED_ATTENTION = Base::SLIDING_OR_CHUNKED_ATTENTION
    };

    // Are we applying alibi bias (drop FMA optimizations for accuracy reasons).
    enum
    {
        APPLY_ALIBI = Base::APPLY_ALIBI
    };

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    enum
    {
        EXP2F_OPTIMIZATION = Base::EXP2F_OPTIMIZATION
    };

    // Are we applying softcapping_scale for qk products ?
    enum
    {
        ENABLE_BMM1_SOFTCAPPING_SCALE = Base::ENABLE_BMM1_SOFTCAPPING_SCALE
    };

    // Whether we need to check if local_max could be -inf or not.
    enum
    {
        CHECK_IF_NEG_INF_EXISTS = Base::CHECK_IF_NEG_INF_EXISTS
    };

    // Ctor.
    template <typename Params>
    inline __device__ Softmax(Params const& params, int tidx)
        : Base(params, tidx)
    {
    }

    // Calculate max/sum, and update flash-attention scales.
    template <bool IS_FIRST_COL>
    inline __device__ void compute_and_update_scale(
        float (&global_max)[Mma_tile_p::CORES_M], float (&global_sum)[Mma_tile_p::CORES_M])
    {
        float const scale = reinterpret_cast<float const&>(this->scale_bmm1_);
        float(&local_max_)[Mma_tile_p::CORES_M] = this->local_max_;
        float(&local_sum_)[Mma_tile_p::CORES_M] = this->local_sum_;
        float(&correction_)[Mma_tile_p::CORES_M] = this->correction_;
        float(&elt_)[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N * 2] = this->elt_;

// Row-wise max of current tile.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            if (IS_FIRST_COL)
            {
                local_max_[mi] = elt_[mi][0];
            }
            else
            {
                local_max_[mi] = fmaxf(global_max[mi], elt_[mi][0]);
            }
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                local_max_[mi] = fmaxf(local_max_[mi], elt_[mi][ni]);
            }
            local_max_[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 1), local_max_[mi]);
            local_max_[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), local_max_[mi], 2), local_max_[mi]);
        }

// Softmax Exp.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                // The equation exp2(scale * x - max) * q_scale_s
                // equals to:   exp2(scale * x - max) * exp2(log2(q_scale_s))
                // equals to:   exp2(scale * x - (max - log2(q_scale_s)))
                //                   ^^^^^   ^   ^^^^^^^^^^^^^^^^^^^^^^^
                // So instead of per-accumulator muls, we can do per-row subs which saves FP cycles.
                // As we scale the softmax output early, before doing the local_sum, we have to unscale
                // the local_sum afterwards.
                float& p0 = elt_[mi][2 * ni + 0];
                float& p1 = elt_[mi][2 * ni + 1];

                // When all elts of the tile are -FLT_MAX, we have to make sure
                //  expf generates 0 for all values instead of 1.
                if constexpr (!EXP2F_OPTIMIZATION)
                {
                    float masked_max = (!CHECK_IF_NEG_INF_EXISTS || local_max_[mi] != -FLT_MAX)
                        ? local_max_[mi] - logf(q_scale_s_)
                        : 0.f;
                    p0 = expf(p0 - masked_max);
                    p1 = expf(p1 - masked_max);
                }
                else
                {
                    // Use exp2f optimization for cases without alibi.
                    float masked_max = (!CHECK_IF_NEG_INF_EXISTS || local_max_[mi] != -FLT_MAX)
                        ? local_max_[mi] * scale - log2f(q_scale_s_)
                        : 0.f;
                    p0 = custom_exp2f(p0, scale, masked_max);
                    p1 = custom_exp2f(p1, scale, masked_max);
                }
            }
        }

// Row-wise sum of current tile.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            local_sum_[mi] = elt_[mi][0];
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                local_sum_[mi] += elt_[mi][ni];
            }
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 1);
            local_sum_[mi] += __shfl_xor_sync(uint32_t(-1), local_sum_[mi], 2);
        }

        // Initialize or update the global sum and max.
        if (IS_FIRST_COL)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                global_sum[mi] = local_sum_[mi];
                global_max[mi] = local_max_[mi];
            }
        }
        else
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                float max_old = global_max[mi];
                float max_new = local_max_[mi];
                float sum_old = global_sum[mi];
                float sum_new = local_sum_[mi];
                // Remove the old max and replace by the new one.
                if constexpr (!EXP2F_OPTIMIZATION)
                {
                    correction_[mi] = expf(max_old - max_new);
                }
                else
                {
                    // Use exp2f optimization for cases without alibi.
                    correction_[mi] = exp2f((max_old - max_new) * scale);
                }
                global_sum[mi] = sum_old * correction_[mi] + sum_new;

                // New max already takes into account old max.
                global_max[mi] = max_new;
            }
        }
    }

    // Update flash attention scales and pack elements for BMM2.
    template <bool IS_FIRST_COL>
    inline __device__ void pack(Compute_tile_o& ctile_o, Fragment_p (&frag_p)[Mma_tile_o::MMAS_K])
    {

// Pack 4 cols for BMM2 A tile.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_o::MMAS_K; ni++)
        {

            // 1st row - 8 elements per row.
            float tmp_00 = this->elt_[0][8 * ni + 0]; // + 0
            float tmp_01 = this->elt_[0][8 * ni + 1]; // + 1
            float tmp_02 = this->elt_[0][8 * ni + 2]; // + 8
            float tmp_03 = this->elt_[0][8 * ni + 3]; // + 9
            float tmp_04 = this->elt_[0][8 * ni + 4]; // +16
            float tmp_05 = this->elt_[0][8 * ni + 5]; // +17
            float tmp_06 = this->elt_[0][8 * ni + 6]; // +24
            float tmp_07 = this->elt_[0][8 * ni + 7]; // +25

            // 2nd row - 8 elements per row.
            float tmp_10 = this->elt_[1][8 * ni + 0]; // + 0
            float tmp_11 = this->elt_[1][8 * ni + 1]; // + 1
            float tmp_12 = this->elt_[1][8 * ni + 2]; // + 8
            float tmp_13 = this->elt_[1][8 * ni + 3]; // + 9
            float tmp_14 = this->elt_[1][8 * ni + 4]; // +16
            float tmp_15 = this->elt_[1][8 * ni + 5]; // +17
            float tmp_16 = this->elt_[1][8 * ni + 6]; // +24
            float tmp_17 = this->elt_[1][8 * ni + 7]; // +25

            // Pack to 4 registers.
            frag_p[ni].reg(0) = fmha::float4_to_fp8x4<Kernel_traits::Element_data_type>(tmp_00, tmp_01, tmp_02, tmp_03);
            frag_p[ni].reg(1) = fmha::float4_to_fp8x4<Kernel_traits::Element_data_type>(tmp_10, tmp_11, tmp_12, tmp_13);
            frag_p[ni].reg(2) = fmha::float4_to_fp8x4<Kernel_traits::Element_data_type>(tmp_04, tmp_05, tmp_06, tmp_07);
            frag_p[ni].reg(3) = fmha::float4_to_fp8x4<Kernel_traits::Element_data_type>(tmp_14, tmp_15, tmp_16, tmp_17);
        }

        if (!IS_FIRST_COL)
        {
// Correct accumulators to current max.
#pragma unroll
            for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
            {
                // Assume only N has multiple MMAs (MMAS_M = 1).
                // MMAS_N > 1 when N dimension is split.
                float correction = this->correction_[mi];
#pragma unroll
                for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
                {
#pragma unroll
                    for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                    {
                        uint32_t& reg0 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                        uint32_t& reg1 = ctile_o.acc_[0][mma_ni].reg(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg0) : "f"(correction));
                        asm volatile("mul.f32 %0, %0, %1;\n" : "+r"(reg1) : "f"(correction));
                    }
                }
            }
        }
        else
        {
            ctile_o.clear();
        }
    }

    static constexpr float q_scale_s_ = Traits_o::SOFTMAX_FP_QUANT_SCALE;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// BMM2 epilogue to apply scales (flash attention).
// FP32 accumulation as default.
template <template <int, int, int, bool, bool> class Traits, typename Kernel_traits>
struct Tile_o_epilogue_base
{

    // The instruction traits for BMM2.
    using Traits_o = typename Kernel_traits::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Kernel_traits::Mma_tile_o;

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    enum
    {
        EXP2F_OPTIMIZATION = Kernel_traits::EXP2F_OPTIMIZATION
    };

    template <typename Params, typename Block_info>
    inline __device__ Tile_o_epilogue_base(Params const& params, Block_info& block_info)
    {
        has_attention_sink_ = params.attention_sinks != nullptr;
        head_idx_ = block_info.bidh;
        attention_sink_ = has_attention_sink_ ? params.attention_sinks[block_info.bidh] : 0.f;
        // It is only need when the exp2f optimization is enabled, so params.scale_bmm1 is always float.
        scale_bmm1_f_ = reinterpret_cast<float const&>(params.scale_bmm1_d ? *params.scale_bmm1_d : params.scale_bmm1);
    };

    // The attention sinks.
    inline __device__ void add_attention_sink(float& sum, float max)
    {
        if (has_attention_sink_)
        {
            // The global max needs to be scaled by the bmm1 scale if exp2f optimization is enabled.
            if constexpr (EXP2F_OPTIMIZATION)
            {
                sum += exp2f(attention_sink_ * M_LOG2E - max * scale_bmm1_f_);
            }
            else
            {
                sum += expf(attention_sink_ - max);
            }
        }
    }

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(
        Compute_tile_o& ctile_o, float (&global_max)[Mma_tile_o::CORES_M], float (&global_sum)[Mma_tile_o::CORES_M])
    {
// Final step's update.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
        {
            // The global sum.
            float global_sum_mi = global_sum[mi];
            // Add the attention sink to the global sum.
            add_attention_sink(global_sum_mi, global_max[mi]);
            // The scale.
            float scale = global_sum_mi == 0.f ? 1.f : 1.0f / global_sum_mi;

// Assume only N has multiple MMAs (MMAS_M = 1).
#pragma unroll
            for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
            {
#pragma unroll
                for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                {
                    float& reg0 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                    float& reg1 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                    reg0 *= scale;
                    reg1 *= scale;
                }
            }
        }
    }

    // Whether the attention sink is enabled.
    bool has_attention_sink_ = false;
    // The attention sink value.
    float attention_sink_ = 0.f;
    // The float scale of bmm1 outputs.
    float scale_bmm1_f_ = 1.f;
    // The head idx.
    int head_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <template <int, int, int, bool, bool> class Traits, typename Kernel_traits>
struct Tile_o_epilogue
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp16_traits
template <typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_fp16_traits, Kernel_traits>
    : public Tile_o_epilogue_base<Hopper_hgmma_fp16_traits, Kernel_traits>
{

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_fp16_traits, Kernel_traits>;

    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Base constructor.
    using Base::Tile_o_epilogue_base;

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(
        Compute_tile_o& ctile_o, float (&global_max)[Mma_tile_o::CORES_M], float (&global_sum)[Mma_tile_o::CORES_M])
    {
// Final step's update.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
        {
            // The global sum.
            float global_sum_mi = global_sum[mi];
            // Add the attention sink to the global sum.
            this->add_attention_sink(global_sum_mi, global_max[mi]);
            // The scale.
            float scale = global_sum_mi == 0.f ? 1.f : 1.0f / global_sum_mi;
            // The scale.
            const uint32_t scale_h = float_to_half2(scale);

// Assume only N has multiple MMAs (MMAS_M = 1).
#pragma unroll
            for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
            {
#pragma unroll
                for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                {
                    uint32_t& reg = ctile_o.acc_[0][mma_ni].reg(ni * Mma_tile_o::CORES_M + mi);
                    reg = hmul2(reg, scale_h);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_fp32_traits
template <typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_fp32_traits, Kernel_traits>
    : public Tile_o_epilogue_base<Hopper_hgmma_fp32_traits, Kernel_traits>
{

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_fp32_traits, Kernel_traits>;

    // Base constructor.
    using Base::Tile_o_epilogue_base;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_hgmma_bf16_traits
template <typename Kernel_traits>
struct Tile_o_epilogue<Hopper_hgmma_bf16_traits, Kernel_traits>
    : public Tile_o_epilogue_base<Hopper_hgmma_bf16_traits, Kernel_traits>
{

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_hgmma_bf16_traits, Kernel_traits>;

    // Base constructor.
    using Base::Tile_o_epilogue_base;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper_qgmma_e4m3_fp32_traits
template <typename Kernel_traits>
struct Tile_o_epilogue<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>
    : public Tile_o_epilogue_base<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>
{

    // The Base class.
    using Base = Tile_o_epilogue_base<Hopper_qgmma_e4m3_fp32_traits, Kernel_traits>;

    // The instruction traits for BMM2.
    using Traits_o = typename Base::Traits_o;

    // The CTA description for BMM2.
    using Cta_tile_o = typename Base::Cta_tile_o;

    // The GMMA compute tile for BMM2.
    using Compute_tile_o = typename Base::Compute_tile_o;

    // The MMA tile for the BMM2.
    using Mma_tile_o = typename Base::Mma_tile_o;

    // Apply the exp2f optimization (fuse bmm1_scale and -max into FMAs).
    enum
    {
        EXP2F_OPTIMIZATION = Base::EXP2F_OPTIMIZATION
    };

    // Ctor.
    template <typename Params, typename Block_info>
    inline __device__ Tile_o_epilogue(Params const& params, Block_info& block_info)
        : Base(params, block_info)
        , scale_bmm2_(*params.scale_bmm2_d)
    {
    }

    // Add the attention sink to the global sum.
    inline __device__ void add_attention_sink(float& sum, float max)
    {
        if (this->has_attention_sink_)
        {
            // The global max needs to be scaled by the bmm1 scale if exp2f optimization is enabled.
            // Take the log2f(Traits_o::SOFTMAX_FP_QUANT_SCALE) into account as the same scale has been applied to sum.
            float quant_scale_in_log2 = log2f(Traits_o::SOFTMAX_FP_QUANT_SCALE);
            if constexpr (EXP2F_OPTIMIZATION)
            {
                sum += exp2f(this->attention_sink_ * M_LOG2E - max * this->scale_bmm1_f_ + quant_scale_in_log2);
            }
            else
            {
                sum += expf(this->attention_sink_ - max + quant_scale_in_log2);
            }
        }
    }

    // Scale ctile_o output by 1/sum
    inline __device__ void scale(
        Compute_tile_o& ctile_o, float (&global_max)[Mma_tile_o::CORES_M], float (&global_sum)[Mma_tile_o::CORES_M])
    {
// Final step's update.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_o::CORES_M; mi++)
        {
            // The global sum.
            float global_sum_mi = global_sum[mi];
            // Add the attention sink to the global sum.
            add_attention_sink(global_sum_mi, global_max[mi]);
#ifdef UNIFIED_EPILOGUE_SCALE
            // Descaling factor
            float const scale_bmm2_f_ = reinterpret_cast<float&>(scale_bmm2_);
            // The scale.
            float scale = global_sum_mi == 0.f ? scale_bmm2_f_ : scale_bmm2_f_ / global_sum_mi;
#else
            float scale = global_sum_mi == 0.f ? 1.0f : 1.0f / global_sum_mi;
#endif
            if constexpr (Kernel_traits::RETURN_SOFTMAX_STATS)
            {
                // Save the dequant exp sum for softmax saver.
                global_sum[mi] *= Traits_o::SOFTMAX_FP_DEQUANT_SCALE;
            }
// Assume only N has multiple MMAs (MMAS_M = 1).
#pragma unroll
            for (int mma_ni = 0; mma_ni < Mma_tile_o::MMAS_N; mma_ni++)
            {
#pragma unroll
                for (int ni = 0; ni < Mma_tile_o::CORES_N; ni++)
                {
                    float& reg0 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi);
                    float& reg1 = ctile_o.acc_[0][mma_ni].elt(2 * ni * Mma_tile_o::CORES_M + 2 * mi + 1);
                    reg0 *= scale;
                    reg1 *= scale;
                }
            }
        }
    }

    // Scale_bmm2.
    uint32_t scale_bmm2_;
};

} // namespace ws
} // namespace fmha
