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

namespace fmha
{
namespace hopper
{
namespace fprop
{

template <typename Kernel_traits, bool IS_TRAINING>
struct Fprop_4x1
{

    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_o = typename Kernel_traits::Traits_o;

    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;

    using Smem_tile_q_a = typename Kernel_traits::Smem_tile_q_a;
    using Smem_tile_v_b = typename Kernel_traits::Smem_tile_v_b;
    using Smem_tile_kt_b = typename Kernel_traits::Smem_tile_kt_b;
    using Smem_tile_vt_b = typename Kernel_traits::Smem_tile_vt_b;

    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;
    using Compute_tile_o = typename Kernel_traits::Compute_tile_o;

    using Mma_tile_p = typename Traits_p ::template Mma_tile<Cta_tile_p>;
    using Mma_tile_o = typename Traits_o ::template Mma_tile<Cta_tile_o>;
    using Transposer = fmha::Transposer<Traits_o, Cta_tile_o, Cta_tile_o::K>;

    // TODO probably ctile o
    using Fragment_p = typename Compute_tile_o::Fragment;
    using A_type_o = typename Traits_o::A_type;
    using Acc_p = Fragment_accumulator<Traits_p>;
    static_assert(Acc_p::NUM_ELTS == Mma_tile_p::CORES_M * Mma_tile_p::CORES_N * 2);

    static_assert(Compute_tile_p::MMAS_N == 1);
    static_assert(Compute_tile_p::MMAS_K == 2);
    static_assert(Mma_tile_p::CORES_M == 2);
    // static_assert(Mma_tile_p::CORES_N == 8);

    using Shared_storage = typename Kernel_traits::Shared_storage;

    static constexpr bool USE_LDGSTS = true;

    __device__ inline Fprop_4x1(
        Fmha_fprop_params const& params, int const bidb, int const bidh, int const tidx, char* smem)
        : params_(params)
        , binfo_(params, bidb, bidh, tidx)
        , tidx_(tidx)
        , smem_aligned_(fmha::align_1024(smem))
        , smem_(reinterpret_cast<Shared_storage*>(smem_aligned_))
        , m_ptr_(&reinterpret_cast<float*>(params.m_ptr)[binfo_.hidx * params.s])
        , zi_ptr_(&reinterpret_cast<float*>(params.zi_ptr)[binfo_.hidx * params.s])
        , amax_s_(0.f)
        , amax_o_(0.f)
        , q_scale_o_(*params_.ptr_q_scale_o)
    {
        float d_scale_qkv = *params_.ptr_d_scale_qkv;

        float d_scale_s = 1.f / (*params_.ptr_q_scale_s);
        float d_scale_o = 1.f / q_scale_o_;

        if (binfo_.hidx == 0 && tidx == 0)
        {
            *params_.ptr_d_scale_s = d_scale_s;
            *params_.ptr_d_scale_o = d_scale_o;
        }

        d_scale_q_k_ = d_scale_qkv * d_scale_qkv;
        d_scale_s_v_ = d_scale_qkv * d_scale_s_;

        int warp = tidx_ / 32;
        int lane = tidx_ % 32;
        int quad = lane / 4;
        qid_ = lane % 4;
        row_local_ = warp * 16 + quad;
    }

    __device__ inline void operator()(uint64_t seed, uint64_t offset)
    {

        if (binfo_.hidx == 0 && tidx_ == 0)
        {
            params_.ptr_philox_unpacked[0] = seed;
            params_.ptr_philox_unpacked[1] = offset;
        }

        static_assert(Cta_tile_p::M == 64);

        // Input tiles.
        uint32_t scale = reinterpret_cast<uint32_t const&>(q_scale_o_);
        Gmem_tile_o gmem_o(params_.o_ptr, params_.o_stride_in_bytes, binfo_, tidx_, scale, 0, 0);
        Gmem_tile_q gmem_q(params_, 0, binfo_, tidx_);
        Gmem_tile_k gmem_k(params_, 1, binfo_, tidx_);
        Gmem_tile_v gmem_v(params_, 2, binfo_, tidx_);

        Smem_tile_q_a smem_q(&smem_->q_a[0], tidx_);
        Smem_tile_kt_b smem_k(&smem_->kt_b[0], tidx_);
        Smem_tile_vt_b smem_v(&smem_->vt_b[0], tidx_);

        // static_assert(Cta_tile_p::M == Cta_tile_p::N);
        static_assert(Cta_tile_p::M == 64);
        int const steps_m = (binfo_.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M;
        int const steps_n = (binfo_.actual_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N;

        uint32_t smem_nvvm_vt_b = __nvvm_get_smem_pointer(&smem_->vt_b[0]);
        uint32_t smem_nvvm_v_b = __nvvm_get_smem_pointer(&smem_->vt_b[0]);

        // Load K, V. Transpose V.
        gmem_k.load(smem_k);
        gmem_v.load(smem_v);

        for (int ci = 0; ci < steps_n - 1; ++ci)
        {
            gmem_k.move();
            gmem_v.move();
            smem_k.move_to_next_write_buffer();
            smem_v.move_to_next_write_buffer();

            gmem_k.load(smem_k);
            gmem_v.load(smem_v);
        }

        fmha::ldgdepbar<USE_LDGSTS>(); // 1: K, V done.

        gmem_q.load(smem_q);

        fmha::ldgdepbar<USE_LDGSTS>();

        fmha::depbar_<USE_LDGSTS, 1>();
        __syncthreads(); // K done.

        for (int ci = 0; ci < steps_n; ++ci)
        {

            Transposer::template transpose_<true>(tidx_, smem_nvvm_vt_b, smem_nvvm_v_b);
            smem_nvvm_vt_b += Smem_tile_vt_b::BYTES_PER_BUFFER;
            smem_nvvm_v_b += Smem_tile_v_b ::BYTES_PER_BUFFER;
        }

        // Fence syncs STS with GMMA to make transposed tiles visible.
        fmha::fence_view_async_shared();
        // Make sure smem Vt is clear to write for Q.

        for (int ri = 0; ri < steps_m; ++ri)
        {
            if (ri < steps_m - 1)
            {
                gmem_q.move();
                smem_q.move_to_next_write_buffer();
                gmem_q.load(smem_q);
                fmha::ldgdepbar<USE_LDGSTS>();
            }

            uint32_t smem_nvvm_q_a = smem_q.smem_ + (ri % 2) * Smem_tile_q_a::BYTES_PER_BUFFER;

            uint32_t smem_nvvm_kt_b = smem_k.smem_;
            uint32_t smem_nvvm_v_b = __nvvm_get_smem_pointer(&smem_->vt_b[0]);

            // O = S x V
            Compute_tile_o ctile_o(0, smem_nvvm_v_b);
            ctile_o.clear();

            if (ri < steps_m - 1)
            {
                fmha::depbar_<USE_LDGSTS, 1>();
                __syncthreads();
            }
            else
            {
                fmha::depbar_<USE_LDGSTS, 0>();
                __syncthreads();
            }

            float p_max[2], p_sum[2];

            Compute_tile_p ctile_p(smem_nvvm_q_a, smem_nvvm_kt_b);

            constexpr int ELTS_PER_TILE_PER_THREAD = Cta_tile_p::M * Cta_tile_p::N / Kernel_traits::THREADS;
            int ci = 0;
            Philox ph(seed, binfo_.tidx_global + (ri * steps_n + ci) * ELTS_PER_TILE_PER_THREAD, offset);
            compute_single_tile<true>(ctile_p, ctile_o, p_max, p_sum, tidx_, ri, ci, ph);

            for (int ci = 1; ci < steps_n; ++ci)
            {
                Philox ph(seed, binfo_.tidx_global + (ri * steps_n + ci) * ELTS_PER_TILE_PER_THREAD, offset);
                smem_nvvm_kt_b += Smem_tile_kt_b::BYTES_PER_BUFFER;

                ctile_o.increment_gmma_desc_group();

                Compute_tile_p ctile_p(smem_nvvm_q_a, smem_nvvm_kt_b);

                compute_single_tile<false>(ctile_p, ctile_o, p_max, p_sum, tidx_, ri, ci, ph);
            }
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                // TODO: NaN, Inf?
                p_sum[mi] = p_sum[mi] == 0.f ? 0.f : 1.f / p_sum[mi];
                // No need for abs :)
                // We use the fact that for stable softmax, the largest value occurs at the position of the max.
                // That position will have value 1/p_sum.
                amax_s_ = fmaxf(amax_s_, p_sum[mi]);
                // p_sum[mi] *= d_scale_s_v_;
                float const scale = p_sum[mi] * d_scale_s_v_;
#pragma unroll
                for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
                {
                    float& o0 = ctile_o.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                    float& o1 = ctile_o.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                    // o0 *= params_.d_scale_s_v;
                    // o1 *= params_.d_scale_s_v;
                    // o0 *= p_sum[mi];
                    // o1 *= p_sum[mi];
                    o0 *= scale;
                    o1 *= scale;
                    amax_o_ = fmaxf(amax_o_, fmaxf(fabsf(o0), fabsf(o1)));
                    // Conversion scale is applied in gmem_tile.
                }
            }

            gmem_o.store(ctile_o.acc_);
            gmem_o.move();

            m_ptr_[ri * Cta_tile_p::M + row_local_ + 0] = p_max[0];
            m_ptr_[ri * Cta_tile_p::M + row_local_ + 8] = p_max[1];

            zi_ptr_[ri * Cta_tile_p::M + row_local_ + 0] = p_sum[0];
            zi_ptr_[ri * Cta_tile_p::M + row_local_ + 8] = p_sum[1];
        }

        // Assumes proper initialization of amax pointer by the user.
        // We're making use of amax >= 0 to save a branch for atomicMax.
        // Also the compiler generates REDUX for intra-warp reduction and calls atomicMax only once per warp.
        atomicMaxFloatPos_(params_.amax_s, amax_s_);
        atomicMaxFloatPos_(params_.amax_o, amax_o_);
    }

    template <bool IS_FIRST_COL>
    __device__ inline void compute_single_tile(Compute_tile_p& ctile_p, Compute_tile_o& ctile_o, float (&p_max)[2],
        float (&p_sum)[2], int const tidx, int const ri, int const ci, Philox& ph)
    {

        ctile_p.clear();
        fmha::warpgroup_arrive();
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K - 1; ++ki)
        {
            ctile_p.compute(ki);
        }
        ctile_p.compute(Mma_tile_p::MMAS_K - 1, true, false);

        fmha::warpgroup_wait<0>();

        float p_[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N * 2];

        float const scale_q_k = d_scale_q_k_ * params_.scale_bmm_q_k;
// Compute S.
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
                float& p0 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                float& p1 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                //// Unscale acc.
                // p0 *= params_.d_scale_q_k;
                // p1 *= params_.d_scale_q_k;
                //// Apply attention scale.
                // p0 *= params_.scale_bmm_q_k;
                // p1 *= params_.scale_bmm_q_k;
                p0 *= scale_q_k;
                p1 *= scale_q_k;
                p_[mi][2 * ni + 0] = p0;
                p_[mi][2 * ni + 1] = p1;
            }
        }

        // Row-wise max of current tile.
        float p_max_t[2];
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            if (IS_FIRST_COL)
            {
                p_max_t[mi] = p_[mi][0];
            }
            else
            {
                p_max_t[mi] = fmaxf(p_max[mi], p_[mi][0]);
            }
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                p_max_t[mi] = fmaxf(p_max_t[mi], p_[mi][ni]);
            }
            p_max_t[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), p_max_t[mi], 1), p_max_t[mi]);
            p_max_t[mi] = fmaxf(__shfl_xor_sync(uint32_t(-1), p_max_t[mi], 2), p_max_t[mi]);
        }

// Masking.
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                int row = ri * Cta_tile_p::M + row_local_ + mi * 8;
                int col = ci * Cta_tile_p::N + qid_ * 2 + ni * 8;

                float& p0 = p_[mi][2 * ni + 0];
                float& p1 = p_[mi][2 * ni + 1];
                bool v0 = row < binfo_.actual_seqlen && (col + 0) < binfo_.actual_seqlen;
                bool v1 = row < binfo_.actual_seqlen && (col + 1) < binfo_.actual_seqlen;

                p0 = v0 ? expf(p0 - p_max_t[mi]) : 0.f;
                p1 = v1 ? expf(p1 - p_max_t[mi]) : 0.f;
            }
        }

        // Row-wise sum of current tile.
        float p_sum_t[2];
#pragma unroll
        for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
        {
            p_sum_t[mi] = p_[mi][0];
#pragma unroll
            for (int ni = 1; ni < Mma_tile_p::CORES_N * 2; ni++)
            {
                p_sum_t[mi] += p_[mi][ni];
            }
            p_sum_t[mi] += __shfl_xor_sync(uint32_t(-1), p_sum_t[mi], 1);
            p_sum_t[mi] += __shfl_xor_sync(uint32_t(-1), p_sum_t[mi], 2);
        }

        float correction[2];
        // Initialize or update the global sum and max.
        if (IS_FIRST_COL)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                p_sum[mi] = p_sum_t[mi];
                p_max[mi] = p_max_t[mi];
                p_sum_t[mi] = p_sum[mi] == 0.f ? 0.f : 1.f / p_sum[mi];
            }
        }
        else
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                float max_old = p_max[mi];
                float max_new = p_max_t[mi];
                float sum_old = p_sum[mi];
                float sum_new = p_sum_t[mi];
                // Remove the old max and replace by the new one.
                correction[mi] = expf(max_old - max_new);
                p_sum[mi] = sum_old * correction[mi] + sum_new;

                // New max already takes into account old max.
                p_max[mi] = max_new;
            }
        }

        Acc_p s_[1][1];
        if (IS_TRAINING)
        {
            // Dropout.
            float const scale_dropout = params_.rp_keep * q_scale_s_;
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {

                float4 const tmp = uniform4(ph());
                float const rand[4] = {tmp.x, tmp.y, tmp.z, tmp.w};

#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                {
                    // Dropout mask.
                    float const dmask0 = rand[mi * 2 + 0] <= params_.p_keep ? scale_dropout : 0.f;
                    float const dmask1 = rand[mi * 2 + 1] <= params_.p_keep ? scale_dropout : 0.f;

                    // Apply dropout and prepare for conversion.
                    p_[mi][2 * ni + 0] *= dmask0;
                    p_[mi][2 * ni + 1] *= dmask1;

                    // Store for debug (should be optimized if unused).
                    s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0) = dmask0;
                    s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1) = dmask1;
                }
            }
        }
        else
        {
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                {
                    // Store for debug (should be optimized if unused).
                    s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0) = p_[mi][2 * ni + 0];
                    s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1) = p_[mi][2 * ni + 1];

                    // Prepare for conversion.
                    p_[mi][2 * ni + 0] *= q_scale_s_;
                    p_[mi][2 * ni + 1] *= q_scale_s_;
                }
            }
        }

        Fragment_p frag_p[Mma_tile_o::MMAS_K];
        static_assert((Mma_tile_p::CORES_N * 2) % 8 == 0);
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N * 2 / 8; ni++)
        {
            frag_p[ni].reg(0) = fmha::float4_to_fp8x4<A_type_o>(
                p_[0][8 * ni + 0], p_[0][8 * ni + 1], p_[0][8 * ni + 2], p_[0][8 * ni + 3]);
            frag_p[ni].reg(1) = fmha::float4_to_fp8x4<A_type_o>(
                p_[1][8 * ni + 0], p_[1][8 * ni + 1], p_[1][8 * ni + 2], p_[1][8 * ni + 3]);
            frag_p[ni].reg(2) = fmha::float4_to_fp8x4<A_type_o>(
                p_[0][8 * ni + 4], p_[0][8 * ni + 5], p_[0][8 * ni + 6], p_[0][8 * ni + 7]);
            frag_p[ni].reg(3) = fmha::float4_to_fp8x4<A_type_o>(
                p_[1][8 * ni + 4], p_[1][8 * ni + 5], p_[1][8 * ni + 6], p_[1][8 * ni + 7]);
        }

        if (!IS_FIRST_COL)
        {
// Correct accumulators to current max
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                {
                    float& o0 = ctile_o.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                    float& o1 = ctile_o.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                    o0 *= correction[mi];
                    o1 *= correction[mi];
                }
            }
        }

        fmha::warpgroup_arrive();

#pragma unroll
        for (int ki = 0; ki < Mma_tile_o::MMAS_K - 1; ++ki)
        {
            ctile_o.fill_frag_a(frag_p[ki]);
            ctile_o.compute(ki);
        }
        ctile_o.fill_frag_a(frag_p[Mma_tile_o::MMAS_K - 1]);
        ctile_o.compute(Mma_tile_o::MMAS_K - 1, true, true);

        fmha::warpgroup_wait<0>();

#if defined(FMHA_TRAIN_OPS_DEBUG_HOPPER_FPROP)
        enum
        {
            BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8
        };

        using Gmem_tile_p = fmha::Gmem_tile_ps_hopper<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
        constexpr float one_f = 1.f;
        uint32_t scale_ = reinterpret_cast<uint32_t const&>(one_f);
        size_t sbhs_tile_offset_bytes = ri * Cta_tile_p::M * params_.s_stride_in_bytes + ci * Cta_tile_p::N * 4;
        Gmem_tile_p gmem_p(static_cast<char*>(params_.p_ptr) + sbhs_tile_offset_bytes, params_.s_stride_in_bytes,
            params_.s * 4, scale_, tidx_);
        Gmem_tile_p gmem_s(static_cast<char*>(params_.s_ptr) + sbhs_tile_offset_bytes, params_.s_stride_in_bytes,
            params_.s * 4, scale_, tidx_);

        gmem_p.store(ctile_p.acc_);

        gmem_s.store(s_);

#endif
    }

    Fmha_fprop_params params_;
    fused_multihead_attention::Block_info_padded<Kernel_traits::THREADS> const binfo_;
    int const tidx_;

    char* smem_aligned_;
    Shared_storage* smem_;

    float* m_ptr_;
    float* zi_ptr_;

    int row_local_;
    int qid_;

    // Keeps track of max abs of S.
    float amax_s_;
    // Keeps track of max abs of O.
    float amax_o_;

    float d_scale_q_k_;
    float d_scale_s_v_;
    // float q_scale_s_;
    //  2^(floor(log2(E4M3_MAX / amax_exp_p))) = 2^(floor(log2(448 / 1))) = 2 ^ 8
    static constexpr float q_scale_s_ = 256.f;
    static constexpr float d_scale_s_ = 1.f / 256.f;
    float q_scale_o_;

    // Philox ph_;
};

} // namespace fprop
} // namespace hopper
} // namespace fmha
