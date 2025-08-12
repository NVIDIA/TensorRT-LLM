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

#include <fmha/fragment.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// F R A G M E N T  (A)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template <int M, int N, int K, bool A_RF, bool B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_fp16_traits<M, N, K, A_RF, B_RF>, Layout>
    : public Fragment<uint16_t, (M * K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>
{
    // A should be coming from RF.
    static_assert(A_RF, "A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template <int M, int N, int K, bool A_RF, bool B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_bf16_traits<M, N, K, A_RF, B_RF>, Layout>
    : public Fragment<uint16_t, (M * K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>
{
    // A should be coming from RF.
    static_assert(A_RF, "A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout>
    : public Fragment<uint16_t, (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>
{
    // A should be coming from RF.
    static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Only needed if Operand A is coming from RF.
template <int GMMA_M, int GMMA_N, int GMMA_K, bool GMMA_A_RF, bool GMMA_B_RF, typename Layout>
struct Fragment_a<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, GMMA_A_RF, GMMA_B_RF>, Layout>
    : public Fragment<int8_t, (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>
{
    // A should be coming from RF.
    static_assert(GMMA_A_RF == true, "GMMA_A_RF must be true to allocate RF for Operand A.\n");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int GMMA_M, int GMMA_N, int GMMA_K, typename Input_type_A, typename Input_type_B, typename Output_type,
    typename Layout>
struct Fragment_a<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false, Input_type_A, Input_type_B, Output_type>, Layout>
    // TODO: Do we need the * 4 or not?
    : public Fragment<Input_type_A, (GMMA_M * GMMA_K) / (Hopper::WARPS_PER_WARP_GROUP * Hopper::THREADS_PER_WARP)>
{
    static_assert(sizeof(Input_type_A) == 1);
    static_assert(sizeof(Input_type_B) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM

template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, false, false>>
    : public Fragment<uint16_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<uint16_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_REGS; ++ii)
        {
            this->reg(ii) = hadd2(this->reg(ii), other.reg(ii));
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Gmma_single_desc_a const& single_desc_a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_fp16<Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false, GMMA_N,
            INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM

template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, false, false>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_REGS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Gmma_single_desc_a const& single_desc_a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_bf16<Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false, GMMA_N,
            INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>>
    : public Fragment<uint16_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<uint16_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // The Traits
    using Traits = Hopper_hgmma_fp16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_REGS; ++ii)
        {
            this->reg(ii) = hadd2(this->reg(ii), other.reg(ii));
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Fragment_a<Traits, Layout_a> const& a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_rfa_fp16<Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // The Traits
    using Traits = Hopper_hgmma_bf16_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Fragment_a<Traits, Layout_a> const& a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_rfa_bf16<Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H G M M A . F 3 2
//
//////////////////////////////////////////////////////////////////////////////////////////////////
// both operands are coming from SMEM
template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, false, false>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Gmma_single_desc_a const& single_desc_a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_fp32<Gmma_single_desc_a::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false, GMMA_N,
            INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
    }
};

//
////////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM
template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // The Traits
    using Traits = Hopper_hgmma_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Fragment_a<Traits, Layout_a> const& a, Gmma_single_desc_b const& single_desc_b)
    {
        // call hgmma
        fmha::hgmma_rfa_fp32<Gmma_single_desc_b::TRANS_MODE == fmha::Gmma_descriptor_transpose::TRANS ? true : false,
            GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Q G M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I G M M A . I N T 8
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Both operands are coming from SMEM.
template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, false, false>>
    : public Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Gmma_single_desc_a const& single_desc_a, Gmma_single_desc_b const& single_desc_b)
    {
        fmha::igmma_int8_int32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A is coming from RF; B is coming from SMEM

template <int GMMA_M, int GMMA_N, int GMMA_K>
struct Fragment_accumulator<Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>>
    : public Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<int32_t, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // The Traits.
    using Traits = Hopper_igmma_int8_int32_traits<GMMA_M, GMMA_N, GMMA_K, true, false>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Fragment_a<Traits, Layout_a> const& a, Gmma_single_desc_b const& single_desc_b)
    {

        fmha::igmma_rfa_int8_int32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fp32 Accumulator A operand from RF and B operand from SMEM
template <int GMMA_M, int GMMA_N, int GMMA_K, typename Input_type_A, typename Input_type_B, typename Output_type>
struct Fragment_accumulator<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false, Input_type_A, Input_type_B, Output_type>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Add two fragments.
    template <typename Other_fragment_>
    inline __device__ void add(Other_fragment_ const& other)
    {
        for (int ii = 0; ii < Base::NUM_ELTS; ++ii)
        {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // The Traits
    using Traits
        = Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, true, false, Input_type_A, Input_type_B, Output_type>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Layout_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Fragment_a<Traits, Layout_a> const& a, Gmma_single_desc_b const& single_desc_b)
    {

        // call hgmma
        if (std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e4m3_t>)
        {
            qgmma_rfa_e4m3_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e4m3_t>)
        {
            qgmma_rfa_e5m2_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e5m2_t>)
        {
            qgmma_rfa_e4m3_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e5m2_t>)
        {
            qgmma_rfa_e5m2_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(a.regs_, single_desc_b.get(), this->regs_);
        }
        else
        {
            assert(false && "unsupported");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// fp32 accumulator
// Both operands are coming from SMEM.
template <int GMMA_M, int GMMA_N, int GMMA_K, typename Input_type_A, typename Input_type_B, typename Output_type>
struct Fragment_accumulator<
    Hopper_qgmma_fp8_fp32_traits<GMMA_M, GMMA_N, GMMA_K, false, false, Input_type_A, Input_type_B, Output_type>>
    : public Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>
{

    // The base class.
    using Base = Fragment<float, (GMMA_M * GMMA_N) / (Hopper::THREADS_PER_WARP * Hopper::WARPS_PER_WARP_GROUP)>;

    // Do the GMMA.
    template <bool INCREMENT_SCORE_BOARD, typename Gmma_single_desc_a, typename Gmma_single_desc_b>
    inline __device__ void mma(Gmma_single_desc_a const& single_desc_a, Gmma_single_desc_b const& single_desc_b)
    {
        if (std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e4m3_t>)
        {
            qgmma_e4m3_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e4m3_t>)
        {
            qgmma_e5m2_e4m3_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e4m3_t> && std::is_same_v<Input_type_B, e5m2_t>)
        {
            qgmma_e4m3_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        }
        else if (std::is_same_v<Input_type_A, e5m2_t> && std::is_same_v<Input_type_B, e5m2_t>)
        {
            qgmma_e5m2_e5m2_fp32<GMMA_N, INCREMENT_SCORE_BOARD>(single_desc_a.get(), single_desc_b.get(), this->regs_);
        }
        else
        {
            assert(false && "unsupported");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Mma_tile>
struct Softmax_saver_tma
{

    // Warps.
    enum
    {
        WARPS_M = Cta_tile::WARPS_M
    };

    enum
    {
        WARPS_N = Cta_tile::WARPS_N
    };

    enum
    {
        WARPS_K = Cta_tile::WARPS_K
    };

    // Ctor.
    template <typename Params, typename Head_info>
    inline __device__ Softmax_saver_tma(Params const& params, Head_info const& head_info)
        : actual_len_(head_info.actual_seqlen)
        , local_q_tile_offset_(head_info.local_q_tile_offset)
        , softmax_sum_ptr_(reinterpret_cast<char*>(params.softmax_stats_ptr))
        , softmax_stats_stride_in_bytes_(params.softmax_stats_stride_in_bytes)
    {
        softmax_max_ptr_ = reinterpret_cast<char*>(params.softmax_stats_ptr);
        int warp = (threadIdx.x % 128) / Cta_tile::THREADS_PER_WARP;
        int lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;
        // MMA row0 index (8x4 thread layout)
        row0_ = warp * Mma_tile::M_PER_MMA / WARPS_M + (lane / 4);

        int sum_s = params.is_s_padded ? params.s * head_info.bidb : params.cu_q_seqlens[head_info.bidb];
        int token_id = sum_s * params.h + head_info.bidh;
        size_t const bh_offset = token_id * sizeof(float) * 2 + local_q_tile_offset_ * softmax_stats_stride_in_bytes_;
        softmax_max_ptr_ += bh_offset + row0_ * softmax_stats_stride_in_bytes_;
        softmax_sum_ptr_ += bh_offset + row0_ * softmax_stats_stride_in_bytes_ + sizeof(float);
    };

    inline __device__ void store(float* p_sum, float* p_max, float sqrt_d, int row_offset, bool valid_run)
    {
        // Four threads process two rows in mma, each row has one softmax_sum and one softmax_max.
        // Here we use one thread to write one softmax element.
        float values;
        int lane = threadIdx.x % Cta_tile::THREADS_PER_WARP;
        if (lane % 4 < 2)
        {
            values = p_sum[lane % 2];
        }
        else
        {
            values = p_max[lane % 2] / sqrt_d;
        }
        if (!valid_run && (lane % 4) < 2)
        {
            values = 1.0;
        }
        char* dst_ptr = (lane % 4 < 2) ? softmax_sum_ptr_ : softmax_max_ptr_;
        size_t off_inside_mma = (lane % 2 == 0) ? row_offset : row_offset + 8;
        if (local_q_tile_offset_ + row0_ + off_inside_mma < actual_len_)
        {
            fmha::stg(dst_ptr + off_inside_mma * softmax_stats_stride_in_bytes_, values);
        }
    }

    // ptr
    char* softmax_sum_ptr_ = nullptr;
    char* softmax_max_ptr_ = nullptr;

    // the first row's idx
    int row0_;
    // actual seq length
    int const actual_len_;
    int const softmax_stats_stride_in_bytes_;
    int const local_q_tile_offset_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
