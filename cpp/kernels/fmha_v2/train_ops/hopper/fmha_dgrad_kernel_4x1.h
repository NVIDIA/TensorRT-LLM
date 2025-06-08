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
namespace dgrad
{

template <typename Kernel_traits>
struct Dgrad_4x1
{

    using Traits_p = typename Kernel_traits::Traits_p;
    using Traits_t = typename Kernel_traits::Traits_t;
    using Traits_ds = typename Kernel_traits::Traits_ds;
    using Traits_dq = typename Kernel_traits::Traits_dq;
    using Traits_dk = typename Kernel_traits::Traits_dk;
    using Traits_dv = typename Kernel_traits::Traits_dv;

    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_t = typename Kernel_traits::Cta_tile_t;
    using Cta_tile_ds = typename Kernel_traits::Cta_tile_ds;
    using Cta_tile_dq = typename Kernel_traits::Cta_tile_dq;
    using Cta_tile_dk = typename Kernel_traits::Cta_tile_dk;
    using Cta_tile_dv = typename Kernel_traits::Cta_tile_dv;

    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    using Gmem_tile_dq = typename Kernel_traits::Gmem_tile_dq;
    using Gmem_tile_dk = typename Kernel_traits::Gmem_tile_dk;
    using Gmem_tile_dv = typename Kernel_traits::Gmem_tile_dv;

    using Smem_tile_q_a = typename Kernel_traits::Smem_tile_q_a;
    using Smem_tile_do_a = typename Kernel_traits::Smem_tile_do_a;
    using Smem_tile_o_b = typename Kernel_traits::Smem_tile_o_b;
    using Smem_tile_k_b = typename Kernel_traits::Smem_tile_k_b;
    using Smem_tile_q_b = typename Kernel_traits::Smem_tile_q_b;
    using Smem_tile_kt_b = typename Kernel_traits::Smem_tile_kt_b;
    using Smem_tile_vt_b = typename Kernel_traits::Smem_tile_vt_b;
    using Smem_tile_do_b = typename Kernel_traits::Smem_tile_do_b;

    using Compute_tile_p = typename Kernel_traits::Compute_tile_p;
    using Compute_tile_t = typename Kernel_traits::Compute_tile_t;
    using Compute_tile_ds = typename Kernel_traits::Compute_tile_ds;
    using Compute_tile_dq = typename Kernel_traits::Compute_tile_dq;
    using Compute_tile_dk = typename Kernel_traits::Compute_tile_dk;
    using Compute_tile_dv = typename Kernel_traits::Compute_tile_dv;

    using Mma_tile_p = typename Traits_p ::template Mma_tile<Cta_tile_p>;
    using Mma_tile_dq = typename Traits_dq::template Mma_tile<Cta_tile_dq>;
    using Mma_tile_dk = typename Traits_dk::template Mma_tile<Cta_tile_dk>;
    using Mma_tile_dv = typename Traits_dv::template Mma_tile<Cta_tile_dv>;

    using Transposer = fmha::Transposer<Traits_p, Cta_tile_p, Cta_tile_p::K>;

    using Fragment_dp = typename Compute_tile_dq::Fragment;
    using Fragment_dpt = typename Compute_tile_dk::Fragment;
    using Fragment_st = typename Compute_tile_dv::Fragment;
    static_assert(Fragment_dp::NUM_REGS == 4);
    static_assert(Fragment_dpt::NUM_REGS == 4);
    static_assert(Fragment_st::NUM_REGS == 4);

    using A_type_dq = typename Traits_dq::A_type;
    using A_type_dk = typename Traits_dk::A_type;
    using A_type_dv = typename Traits_dv::A_type;

    using Acc_p = Fragment_accumulator<Traits_p>;
    static_assert(Acc_p::NUM_ELTS == Mma_tile_dq::CORES_M * Mma_tile_dq::CORES_N * 2);

    using Shared_storage = typename Kernel_traits::Shared_storage;

    static constexpr bool USE_LDGSTS = true;

    static constexpr int STRIDE_DQ_TILE = Acc_p::NUM_ELTS * Kernel_traits::THREADS;
    static_assert(STRIDE_DQ_TILE == 64 * 64);

    static constexpr int BYTES_PER_ROW = 128;
    static constexpr int BYTES_PER_STS = 16;

    template <typename Acc>
    __device__ inline void serialize_acc(typename Acc::Data_type* ptr, Acc const& src)
    {
        uint4* dst = reinterpret_cast<uint4*>(ptr);
        static_assert(Acc::NUM_REGS % 4 == 0);
#pragma unroll
        for (int it = 0; it < Acc::NUM_REGS / 4; it++)
        {
            uint4 data;
            data.x = src.reg(it * 4 + 0);
            data.y = src.reg(it * 4 + 1);
            data.z = src.reg(it * 4 + 2);
            data.w = src.reg(it * 4 + 3);
            fmha::stg(dst, data);
            dst += Kernel_traits::THREADS;
        }
    }

    template <typename Acc>
    __device__ inline void deserialize_acc(Acc& dst, typename Acc::Data_type const* ptr)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(ptr);
        static_assert(Acc::NUM_REGS % 4 == 0);
#pragma unroll
        for (int it = 0; it < Acc::NUM_REGS / 4; it++)
        {
            uint4 data;
            fmha::ldg(data, src);
            dst.reg(it * 4 + 0) = data.x;
            dst.reg(it * 4 + 1) = data.y;
            dst.reg(it * 4 + 2) = data.z;
            dst.reg(it * 4 + 3) = data.w;
            src += Kernel_traits::THREADS;
        }
    }

    __device__ inline Dgrad_4x1(Fmha_dgrad_params const& params, int const bidb, int const bidh, int const tidx,
        char* smem //, std::tuple<uint64_t, uint64_t> & seeds
        )
        : params_(params)
        , binfo_(params, bidb, bidh, tidx)
        , tidx_(tidx)
        , smem_aligned_(fmha::align_1024(smem))
        , smem_(reinterpret_cast<Shared_storage*>(smem_aligned_))
        , m_ptr_(&reinterpret_cast<float*>(params.m_ptr)[binfo_.hidx * params.s])
        , zi_ptr_(&reinterpret_cast<float*>(params.zi_ptr)[binfo_.hidx * params.s])
        , amax_dp_(0.f)
        , amax_dqkv_(0.f)
        , q_scale_dqkv_(*params.ptr_q_scale_dqkv)
        , q_scale_s_(*params.ptr_q_scale_s)
        , q_scale_dp_(*params.ptr_q_scale_dp)
    {

        float d_scale_qkv = *params_.ptr_d_scale_qkv;
        float d_scale_s = *params_.ptr_d_scale_s;
        float d_scale_o = *params_.ptr_d_scale_o;
        float d_scale_do = *params_.ptr_d_scale_do;
        // float d_scale_dp  = *params_.ptr_d_scale_dp;
        float d_scale_dp = 1.f / q_scale_dp_;
        float d_scale_dqkv = 1.f / q_scale_dqkv_;

        if (binfo_.hidx == 0 && tidx == 0)
        {
            *params_.ptr_d_scale_dp = d_scale_dp;
            *params_.ptr_d_scale_dqkv = d_scale_dqkv;
        }

        d_scale_do_o_ = d_scale_do * d_scale_o;
        d_scale_q_k_ = d_scale_qkv * d_scale_qkv;
        d_scale_do_v_ = d_scale_do * d_scale_qkv;
        d_scale_s_do_ = d_scale_s * d_scale_do;
        d_scale_dp_qkv_ = d_scale_dp * d_scale_qkv;

        // TODO  pass in a b*s*h*d workspace?
        int steps = params.s / Cta_tile_p::M;
        dq_tmp_ = reinterpret_cast<float*>(params.dq_tmp_ptr) + binfo_.hidx * params_.s * 64 + tidx * 4;

        // smem_nvvm_s_t = __nvvm_get_smem_pointer(&smem_->s_t[0]);
        // smem_nvvm_dp_t = __nvvm_get_smem_pointer(&smem_->dp_t[0]);

        int warp = tidx_ / 32;
        int lane = tidx_ % 32;
        uint32_t read_row = warp * 16 + lane / 2;
        uint32_t read_col = lane % 2;
        read_col ^= (read_row % 4) * 2;
        read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_STS;

        uint32_t write_row = lane / 2;
        uint32_t write_col = warp * 2 + lane % 2;
        write_col ^= (write_row % 4) * 2;
        write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

        int quad = lane / 4;
        qid_ = lane % 4;
        row_local_ = warp * 16 + quad;
    }

    __device__ inline void operator()(uint64_t seed, uint64_t offset)
    {

        static_assert(Cta_tile_p::M == 64);

        // Output tiles.
        uint32_t q_scale_dqkv = reinterpret_cast<uint32_t const&>(q_scale_dqkv_);
        Gmem_tile_dq gmem_dq8(params_.dqkv_ptr, params_.qkv_stride_in_bytes, binfo_, tidx_, q_scale_dqkv, 0, 0);
        Gmem_tile_dk gmem_dk8(params_.dqkv_ptr, params_.qkv_stride_in_bytes, binfo_, tidx_, q_scale_dqkv, 0, 1);
        Gmem_tile_dv gmem_dv8(params_.dqkv_ptr, params_.qkv_stride_in_bytes, binfo_, tidx_, q_scale_dqkv, 0, 2);

        // Input tiles.
        Gmem_tile_o gmem_do(params_.do_ptr, params_.o_stride_in_bytes, params_.h, 0, binfo_, tidx_);
        Gmem_tile_o gmem_o(params_.o_ptr, params_.o_stride_in_bytes, params_.h, 0, binfo_, tidx_);
        Gmem_tile_q gmem_q(params_, 0, binfo_, tidx_);
        Gmem_tile_k gmem_k(params_, 1, binfo_, tidx_);
        Gmem_tile_v gmem_v(params_, 2, binfo_, tidx_);

        Smem_tile_q_a smem_q(&smem_->q_a[0], tidx_);
        Smem_tile_kt_b smem_k(&smem_->kt_b[0], tidx_);

        Smem_tile_o_b smem_o(&smem_->vt_b[0], tidx_);

        Smem_tile_do_a smem_do(&smem_->do_a[0], tidx_);
        Smem_tile_vt_b smem_v(&smem_->vt_b[0], tidx_);

        static_assert(Cta_tile_p::M == Cta_tile_p::N);
        static_assert(Cta_tile_p::M == 64);
        int const steps = (binfo_.actual_seqlen + Cta_tile_p::M - 1) / Cta_tile_p::M;

        gmem_do.load(smem_do);
        gmem_o.load(smem_o);
        gmem_q.load(smem_q);
        fmha::ldgdepbar<USE_LDGSTS>();

        uint32_t smem_nvvm_do_a = smem_do.smem_;
        uint32_t smem_nvvm_ot_b = smem_o.smem_;
        float* diag_doo = &smem_->diag_doo[0];

        for (int row = 0; row < steps - 1; ++row)
        {
            gmem_do.move();
            gmem_o.move();
            gmem_q.move();

            smem_do.move_to_next_write_buffer();
            smem_o.move_to_next_write_buffer();
            smem_q.move_to_next_write_buffer();

            gmem_do.load(smem_do);
            gmem_o.load(smem_o);
            gmem_q.load(smem_q);
            fmha::ldgdepbar<USE_LDGSTS>();

            fmha::depbar_<USE_LDGSTS, 1>();
            __syncthreads();

            // T  = dO x O'
            Compute_tile_t ctile_t(smem_nvvm_do_a, smem_nvvm_ot_b + (row % 2) * Smem_tile_o_b::BYTES_PER_BUFFER);
            ctile_t.clear();
            compute_single_diag_doo(diag_doo, ctile_t, row_local_, qid_);
            diag_doo += Cta_tile_p::M;
            smem_nvvm_do_a += Smem_tile_q_a::BYTES_PER_BUFFER;
        }
        fmha::depbar_<USE_LDGSTS, 0>();
        __syncthreads();

        int row = steps - 1;
        Compute_tile_t ctile_t(smem_nvvm_do_a, smem_nvvm_ot_b + (row % 2) * Smem_tile_o_b::BYTES_PER_BUFFER);
        ctile_t.clear();
        compute_single_diag_doo(diag_doo, ctile_t, row_local_, qid_);

        // Issue loads: column tiles.
        gmem_k.load(smem_k);
        gmem_v.load(smem_v);
        fmha::ldgdepbar<USE_LDGSTS>(); // 1: K, V done.

        for (int ci = 0; ci < steps; ++ci)
        {

            if (ci < steps - 1)
            {
                gmem_k.move();
                gmem_v.move();
                smem_k.move_to_next_write_buffer();
                smem_v.move_to_next_write_buffer();
                // Issue loads: column tiles.
                gmem_k.load(smem_k);
                gmem_v.load(smem_v);

                fmha::ldgdepbar<USE_LDGSTS>(); // 0: K, V done.
            }

            uint32_t smem_nvvm_k_b = __nvvm_get_smem_pointer(&smem_->k_b[0]);
            uint32_t smem_nvvm_do_b = __nvvm_get_smem_pointer(&smem_->do_b[0]);
            uint32_t smem_nvvm_q_b = __nvvm_get_smem_pointer(&smem_->q_b[0]);
            // dK = dP' x Q
            Compute_tile_dk ctile_dk(0, smem_nvvm_q_b);
            ctile_dk.clear();
            // dV = S' x dO
            Compute_tile_dv ctile_dv(0, smem_nvvm_do_b);
            ctile_dv.clear();

            // dQ = dP x K: tile is cleared for ci=0 in compute_single_tile()
            Compute_tile_dq ctile_dq(0, smem_nvvm_k_b);

            uint32_t smem_nvvm_q_a = smem_q.smem_;
            uint32_t smem_nvvm_do_a = smem_do.smem_;
            uint32_t smem_nvvm_kt_b = smem_k.smem_ + (ci % 2) * Smem_tile_o_b::BYTES_PER_BUFFER;
            uint32_t smem_nvvm_vt_b = smem_v.smem_ + (ci % 2) * Smem_tile_o_b::BYTES_PER_BUFFER;
            if (ci < steps - 1)
            {
                fmha::depbar_<USE_LDGSTS, 1>();
                __syncthreads(); // K done.
            }
            else
            {
                fmha::depbar_<USE_LDGSTS, 0>();
                __syncthreads(); // K done.
            }
            // This will view the transposed logical [64, 64] as [64, 4, 8, 2] and store as [64, 4, 2, 8].
            Transposer::template transpose_<false>(tidx_, smem_nvvm_kt_b, smem_nvvm_k_b);

            for (int ri = 0; ri < steps; ++ri)
            {

                // P  =  Q x K'
                Compute_tile_p ctile_p(smem_nvvm_q_a, smem_nvvm_kt_b);
                // dS = dO x V'
                Compute_tile_ds ctile_ds(smem_nvvm_do_a, smem_nvvm_vt_b);

                // No need to clear P. We prime p's accs to deal with masking.
                // ctile_p.clear();
                ctile_ds.clear();

                constexpr int ELTS_PER_TILE_PER_THREAD = Cta_tile_p::M * Cta_tile_p::N / Kernel_traits::THREADS;
                Philox ph(seed, binfo_.tidx_global + (ri * steps + ci) * ELTS_PER_TILE_PER_THREAD, offset);
                bool first_col = ci == 0;
                bool last_col = ci == steps - 1;
                if (!(ci == 0 && ri == 0))
                    __syncthreads();
                if (first_col && last_col)
                {
                    compute_single_tile<true, true>(ctile_dk, ctile_dv, ctile_p, ctile_ds, ctile_dq, smem_nvvm_do_a,
                        smem_nvvm_do_b, smem_nvvm_q_a, smem_nvvm_q_b, tidx_, ri, ci, gmem_dq8, ph);
                }
                else if (first_col && !last_col)
                {
                    compute_single_tile<true, false>(ctile_dk, ctile_dv, ctile_p, ctile_ds, ctile_dq, smem_nvvm_do_a,
                        smem_nvvm_do_b, smem_nvvm_q_a, smem_nvvm_q_b, tidx_, ri, ci, gmem_dq8, ph);
                }
                else if (!first_col && last_col)
                {
                    compute_single_tile<false, true>(ctile_dk, ctile_dv, ctile_p, ctile_ds, ctile_dq, smem_nvvm_do_a,
                        smem_nvvm_do_b, smem_nvvm_q_a, smem_nvvm_q_b, tidx_, ri, ci, gmem_dq8, ph);
                }
                else
                { //! first_col && !last_col
                    compute_single_tile<false, false>(ctile_dk, ctile_dv, ctile_p, ctile_ds, ctile_dq, smem_nvvm_do_a,
                        smem_nvvm_do_b, smem_nvvm_q_a, smem_nvvm_q_b, tidx_, ri, ci, gmem_dq8, ph);
                }

                smem_nvvm_q_a += Smem_tile_q_a::BYTES_PER_BUFFER;
                smem_nvvm_do_a += Smem_tile_q_a::BYTES_PER_BUFFER;

                m_ptr_ += Cta_tile_p::M;
                zi_ptr_ += Cta_tile_p::M;
                dq_tmp_ += STRIDE_DQ_TILE;
            }
// Determine local amax and scale.
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {
#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M * 2; mi++)
                {
                    float& dk = ctile_dk.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi);
                    float& dv = ctile_dv.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi);
                    // Unscale acc.
                    dk *= d_scale_dp_qkv_;
                    dv *= d_scale_s_do_;
                    // Collect AMAX.
                    // amax_dqkv_ = fmaxf(amax_dqkv_, fabsf(dk));
                    // amax_dqkv_ = fmaxf(amax_dqkv_, fabsf(dv));

                    amax_dqkv_ = max3Pos_(amax_dqkv_, fabsf(dk), fabsf(dv));
                    // Scaling before conversion happens in the gmem_tile when quantizing
                }
            }
            // Convert and store.
            gmem_dk8.store(ctile_dk.acc_);
            gmem_dk8.move();

            gmem_dv8.store(ctile_dv.acc_);
            gmem_dv8.move();

            // Reset stats ptrs.
            m_ptr_ -= steps * Cta_tile_p::M;
            zi_ptr_ -= steps * Cta_tile_p::M;
            dq_tmp_ -= steps * STRIDE_DQ_TILE;
        }

        // Assumes proper initialization of amax pointer by the user.
        // We're making use of amax >= 0 to save a branch for atomicMax.
        // Also the compiler generates REDUX for intra-warp reduction and calls atomicMax only once per warp.
        atomicMaxFloatPos_(params_.amax_dp, amax_dp_);
        atomicMaxFloatPos_(params_.amax_dqkv, amax_dqkv_);
    }

    // template<bool DESER_DQ, bool SER_DQ, bool MASK>
    template <bool IS_FIRST_COL, bool IS_LAST_COL>
    __device__ inline void compute_single_tile(Compute_tile_dk& ctile_dk, Compute_tile_dv& ctile_dv,
        Compute_tile_p& ctile_p, Compute_tile_ds& ctile_ds, Compute_tile_dq& ctile_dq, uint32_t const smem_nvvm_do_a,
        uint32_t const smem_nvvm_do_b, uint32_t const smem_nvvm_q_a, uint32_t const smem_nvvm_q_b, int const tidx,
        int const ri, int const ci, Gmem_tile_dq& gmem_dq8, Philox& ph)
    {

        if (IS_FIRST_COL)
        {
            ctile_dq.clear();
        }
        else
        {
            deserialize_acc(ctile_dq.acc_[0][0], dq_tmp_);
        }

#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                int row = ri * Cta_tile_p::M + row_local_ + mi * 8;
                int col = ci * Cta_tile_p::N + qid_ * 2 + ni * 8;

                float& p0 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                float& p1 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                bool v0 = row < binfo_.actual_seqlen && (col + 0) < binfo_.actual_seqlen;
                bool v1 = row < binfo_.actual_seqlen && (col + 1) < binfo_.actual_seqlen;

                p0 = v0 ? 0.f : -std::numeric_limits<float>::infinity();
                p1 = v1 ? 0.f : -std::numeric_limits<float>::infinity();
            }
        }

        float p_max[2], p_sum[2], doo_sum[2];
        p_max[0] = m_ptr_[row_local_ + 0];
        p_max[1] = m_ptr_[row_local_ + 8];

        p_sum[0] = zi_ptr_[row_local_ + 0];
        p_sum[1] = zi_ptr_[row_local_ + 8];

        static_assert(Compute_tile_p::MMAS_N == 1);
        static_assert(Compute_tile_p::MMAS_K == 2);

        static_assert(Compute_tile_dq::MMAS_N == 1);
        static_assert(Compute_tile_dq::MMAS_K == 2);

        // Q row-major => Q col-major.
        Transposer::template transpose_<false>(tidx, smem_nvvm_q_a, smem_nvvm_q_b);
        // dO row-major => dO col-major.
        Transposer::template transpose_<false>(tidx, smem_nvvm_do_a, smem_nvvm_do_b);

        // uint32_t smem_nvvm_kt_b = __nvvm_get_smem_pointer(&smem_->kt_b) + (ci % 2) * Smem_tile_o_b::BYTES_PER_BUFFER;
        // uint32_t smem_nvvm_k_b  = __nvvm_get_smem_pointer(&smem_->k_b[0]);
        // Transposer::template transpose_<false>(tidx_, smem_nvvm_kt_b, smem_nvvm_k_b);
        //  Broadcast diag_doo.
        doo_sum[0] = smem_->diag_doo[ri * Cta_tile_p::M + row_local_ + 0];
        doo_sum[1] = smem_->diag_doo[ri * Cta_tile_p::M + row_local_ + 8];

        // Fence syncs STS with GMMA to make transposed tiles visible.
        fmha::fence_view_async_shared();

        fmha::warpgroup_arrive();
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K - 1; ++ki)
        {
            ctile_p.compute(ki);
        }
        ctile_p.compute(Mma_tile_p::MMAS_K - 1, true, false);

#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K - 1; ++ki)
        {
            ctile_ds.compute(ki);
        }
        ctile_ds.compute(Mma_tile_p::MMAS_K - 1, true, false);

        fmha::warpgroup_wait<1>();

        static_assert(Mma_tile_p::CORES_M == 2);
        static_assert(Mma_tile_p::CORES_N == 8);

        // Recomputed masked S.
        Acc_p s_[1][1];
        float const scale_bmm_q_k = params_.scale_bmm_q_k * d_scale_q_k_;
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {

                float const alpha = p_sum[mi];
                float const beta = p_max[mi];

                float p0 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                float p1 = ctile_p.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                // Unscale acc and apply attention scale.
                p0 *= scale_bmm_q_k;
                p1 *= scale_bmm_q_k;
                // Recompute softmax.
                float s0 = alpha * expf(p0 - beta);
                float s1 = alpha * expf(p1 - beta);

                s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0) = s0;
                s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1) = s1;
            }
        }

        fmha::warpgroup_wait<0>();

        Acc_p dp_[1][1];
        uint32_t s_half[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N];
        uint32_t dp_half[Mma_tile_p::CORES_M][Mma_tile_p::CORES_N];
        float const& p_keep = params_.p_keep;
        float const& rp_keep = params_.rp_keep;

        int pos = 1;
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {

            float4 const tmp = uniform4(ph());
            float const rand[4] = {tmp.x, tmp.y, tmp.z, tmp.w};
            // bool keep[4] = {false, false, false, false};
            // #pragma unroll
            // for( int ii = 0; ii < 4; ii++ ) {
            //     keep[ii] = (dmask & pos) != 0;

            //    pos <<= 1;
            //}

#pragma unroll
            for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
            {
                // Dropout mask.
                float const dmask0 = rand[mi * 2 + 0] <= p_keep ? rp_keep : 0.f;
                float const dmask1 = rand[mi * 2 + 1] <= p_keep ? rp_keep : 0.f;

                // const float dmask0 = keep[mi * 2 + 0]? rp_keep : 0.f;
                // const float dmask1 = keep[mi * 2 + 1]? rp_keep : 0.f;

                float const gamma = doo_sum[mi];

                float& ds0 = ctile_ds.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                float& ds1 = ctile_ds.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                float s0 = s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                float s1 = s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                // Unscale acc.
                ds0 *= d_scale_do_v_;
                ds1 *= d_scale_do_v_;
                // Bprop through dropout.
                ds0 *= dmask0;
                ds1 *= dmask1;
                // Bprop through softmax.
                float dp0 = (ds0 - gamma) * s0 * params_.scale_bmm_q_k;
                float dp1 = (ds1 - gamma) * s1 * params_.scale_bmm_q_k;
                // Collect AMAX.
                amax_dp_ = fmaxf(amax_dp_, fmaxf(fabsf(dp0), fabsf(dp1)));
                // amax_dp_ = max3Pos_(amax_dp_, fabsf(dp0), fabsf(dp1));
                //  Store back for debug.
                dp_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0) = dp0;
                dp_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1) = dp1;
                // Scale for conversion. TODO half2 since we're anyway converting?
                dp0 *= q_scale_dp_;
                dp1 *= q_scale_dp_;
                // Convert for transpose.
                dp_half[mi][ni] = fmha::float2_to_half2(dp0, dp1);

                // Reapply dropout.
                float d0 = s0 * dmask0;
                float d1 = s1 * dmask1;

                s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0) = dmask0;
                s_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1) = dmask1;

                // Scale for conversion. TODO half2 since we're anyway converting?
                d0 *= q_scale_s_;
                d1 *= q_scale_s_;
                // Convert for transpose.
                s_half[mi][ni] = fmha::float2_to_half2(d0, d1);
            }
        }

        // Transpose: Store s_half.
        uint32_t smem_nvvm_s_t = __nvvm_get_smem_pointer(&smem_->s_t[0]);
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N / 2; ni++)
        {
            uint4 data;
            data.x = s_half[0][ni * 2 + 0];
            data.y = s_half[0][ni * 2 + 1];
            data.z = s_half[1][ni * 2 + 0];
            data.w = s_half[1][ni * 2 + 1];
            fmha::stsmt(smem_nvvm_s_t + write_offset_ + ni * 16 * BYTES_PER_ROW, data);
        }

        Fragment_dp frag_dp[Mma_tile_dq::MMAS_K];
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N / 2 / 2; ni++)
        {
            frag_dp[ni].reg(0) = fmha::half4_to_fp8x4<A_type_dq>(dp_half[0][4 * ni + 0], dp_half[0][4 * ni + 1]);
            frag_dp[ni].reg(1) = fmha::half4_to_fp8x4<A_type_dq>(dp_half[1][4 * ni + 0], dp_half[1][4 * ni + 1]);
            frag_dp[ni].reg(2) = fmha::half4_to_fp8x4<A_type_dq>(dp_half[0][4 * ni + 2], dp_half[0][4 * ni + 3]);
            frag_dp[ni].reg(3) = fmha::half4_to_fp8x4<A_type_dq>(dp_half[1][4 * ni + 2], dp_half[1][4 * ni + 3]);
        }

        // Transpose: Store dp_half.
        uint32_t smem_nvvm_dp_t = __nvvm_get_smem_pointer(&smem_->dp_t[0]);
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N / 2; ni++)
        {
            uint4 data;
            data.x = dp_half[0][ni * 2 + 0];
            data.y = dp_half[0][ni * 2 + 1];
            data.z = dp_half[1][ni * 2 + 0];
            data.w = dp_half[1][ni * 2 + 1];
            fmha::stsmt(smem_nvvm_dp_t + write_offset_ + ni * 16 * BYTES_PER_ROW, data);
        }
        __syncthreads();

        fmha::warpgroup_arrive();

#pragma unroll
        for (int ki = 0; ki < Mma_tile_dq::MMAS_K - 1; ++ki)
        {
            ctile_dq.fill_frag_a(frag_dp[ki]);
            ctile_dq.compute(ki);
        }
        ctile_dq.fill_frag_a(frag_dp[Mma_tile_dq::MMAS_K - 1]);
        ctile_dq.compute(Mma_tile_dq::MMAS_K - 1, true, true);

        fmha::warpgroup_wait<0>();

        uint32_t read_s_t = smem_nvvm_s_t + read_offset_;
        uint32_t read_dp_t = smem_nvvm_dp_t + read_offset_;

        static_assert(Mma_tile_p::CORES_N == 8);

        Fragment_st frag_s_t[Mma_tile_dv::MMAS_K];
        Fragment_dpt frag_dp_t[Mma_tile_dk::MMAS_K];
        static_assert(Mma_tile_dk::MMAS_K == 2);
        static_assert(Mma_tile_dk::MMAS_K == Mma_tile_p::CORES_N / 4);
        static_assert(Mma_tile_dk::MMAS_M == 1);
        static_assert(Mma_tile_dv::MMAS_K == 2);
        static_assert(Mma_tile_dv::MMAS_K == Mma_tile_p::CORES_N / 4);
        static_assert(Mma_tile_dv::MMAS_M == 1);
#pragma unroll
        for (int ii = 0; ii < Mma_tile_p::CORES_N / 2 / 2; ii++)
        {
            uint4 data_s[2];
            uint4 data_dp[2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++)
            {
                int ni = ii * 2 + jj;
                fmha::ldsm(data_s[jj], read_s_t);
                fmha::ldsm(data_dp[jj], read_dp_t);

                uint32_t step = ((ni & 0x01) * 2 + 1) * 2 * BYTES_PER_STS;
                read_s_t ^= step;
                read_dp_t ^= step;

#if 0
                uint32_t *ptr_s = reinterpret_cast<uint32_t*>(params_.print_buf);
                uint32_t *ptr_dp = ptr_s + 64 * 32;
                // DEBUG Store the transposed FP16 tile
                int r = warp * 16 + quad;
                int c = ni   *  8 + qid;

                ptr_s[(r + 0) * 32 + c + 0] = data_s[jj].x;
                ptr_s[(r + 8) * 32 + c + 0] = data_s[jj].y;
                ptr_s[(r + 0) * 32 + c + 4] = data_s[jj].z;
                ptr_s[(r + 8) * 32 + c + 4] = data_s[jj].w;

                ptr_dp[(r + 0) * 32 + c + 0] = data_dp[jj].x;
                ptr_dp[(r + 8) * 32 + c + 0] = data_dp[jj].y;
                ptr_dp[(r + 0) * 32 + c + 4] = data_dp[jj].z;
                ptr_dp[(r + 8) * 32 + c + 4] = data_dp[jj].w;
#endif
            }

            frag_s_t[ii].reg(0) = fmha::half4_to_fp8x4<A_type_dv>(data_s[0].x, data_s[0].z);
            frag_s_t[ii].reg(1) = fmha::half4_to_fp8x4<A_type_dv>(data_s[0].y, data_s[0].w);
            frag_s_t[ii].reg(2) = fmha::half4_to_fp8x4<A_type_dv>(data_s[1].x, data_s[1].z);
            frag_s_t[ii].reg(3) = fmha::half4_to_fp8x4<A_type_dv>(data_s[1].y, data_s[1].w);

            frag_dp_t[ii].reg(0) = fmha::half4_to_fp8x4<A_type_dk>(data_dp[0].x, data_dp[0].z);
            frag_dp_t[ii].reg(1) = fmha::half4_to_fp8x4<A_type_dk>(data_dp[0].y, data_dp[0].w);
            frag_dp_t[ii].reg(2) = fmha::half4_to_fp8x4<A_type_dk>(data_dp[1].x, data_dp[1].z);
            frag_dp_t[ii].reg(3) = fmha::half4_to_fp8x4<A_type_dk>(data_dp[1].y, data_dp[1].w);
        }

        if (IS_LAST_COL)
        {
// Determine local amax and scale.
#pragma unroll
            for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
            {

#pragma unroll
                for (int mi = 0; mi < Mma_tile_p::CORES_M; mi++)
                {
                    float& dq0 = ctile_dq.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 0);
                    float& dq1 = ctile_dq.acc_[0][0].elt(ni * Mma_tile_p::CORES_M * 2 + mi * 2 + 1);
                    // Unscale acc.
                    dq0 *= d_scale_dp_qkv_;
                    dq1 *= d_scale_dp_qkv_;
                    // Collect AMAX.
                    // amax_dqkv_ = fmaxf(amax_dqkv_, fmaxf(fabsf(dq0), fabsf(dq1)));
                    amax_dqkv_ = max3Pos_(amax_dqkv_, fabsf(dq0), fabsf(dq1));
                    // Scaling for conversion is done by the Gmem_tile.
                }
            }

            gmem_dq8.store(ctile_dq.acc_);
            gmem_dq8.move();
        }
        else
        {
            serialize_acc(dq_tmp_, ctile_dq.acc_[0][0]);
        }

        fmha::warpgroup_arrive();
#pragma unroll
        for (int ki = 0; ki < Mma_tile_dv::MMAS_K - 1; ++ki)
        {
            ctile_dv.fill_frag_a(frag_s_t[ki]);
            ctile_dv.compute(ki);

            ctile_dk.fill_frag_a(frag_dp_t[ki]);
            ctile_dk.compute(ki);
        }
        // Last GMMA increments score board.
        ctile_dv.fill_frag_a(frag_s_t[Mma_tile_dv::MMAS_K - 1]);
        ctile_dv.compute(Mma_tile_dv::MMAS_K - 1, true, true);

        ctile_dk.fill_frag_a(frag_dp_t[Mma_tile_dk::MMAS_K - 1]);
        ctile_dk.compute(Mma_tile_dk::MMAS_K - 1, true, true);

        fmha::warpgroup_wait<0>();

#if defined(FMHA_TRAIN_OPS_DEBUG_HOPPER_DGRAD)
        enum
        {
            BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8
        };

        using Gmem_tile_p = fmha::Gmem_tile_ps_hopper<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
        using Gmem_tile_ds = fmha::Gmem_tile_ps_hopper<Traits_ds, Cta_tile_ds, BITS_PER_ELT_P>;
        constexpr float one_f = 1.f;
        uint32_t scale_ = reinterpret_cast<uint32_t const&>(one_f);
        size_t sbhs_tile_offset_bytes = ri * Cta_tile_p::M * params_.ds_stride_in_bytes + ci * Cta_tile_p::N * 4;
        Gmem_tile_p gmem_s(static_cast<char*>(params_.s_ptr) + sbhs_tile_offset_bytes, params_.ds_stride_in_bytes,
            params_.s * 4, scale_, tidx_);

        Gmem_tile_ds gmem_ds(static_cast<char*>(params_.ds_ptr) + sbhs_tile_offset_bytes, params_.ds_stride_in_bytes,
            params_.s * 4, scale_, tidx_);

        Gmem_tile_p gmem_dp(static_cast<char*>(params_.dp_ptr) + sbhs_tile_offset_bytes, params_.ds_stride_in_bytes,
            params_.s * 4, scale_, tidx_);

        gmem_s.store(s_);

        gmem_ds.store(ctile_ds.acc_);

        gmem_dp.store(dp_);
#endif
    }

    __device__ inline void compute_single_diag_doo(
        float* diag_doo, Compute_tile_t& ctile_t, int const row_local, int const qid)
    {

        fmha::warpgroup_arrive();
#pragma unroll
        for (int ki = 0; ki < Mma_tile_p::MMAS_K - 1; ++ki)
        {
            ctile_t.compute(ki);
        }
        ctile_t.compute(Mma_tile_p::MMAS_K - 1, true, true);
        fmha::warpgroup_wait<0>();

        // Extract diag_doo = (dO * O).sum(-1).
        int c0 = qid * 2 + 0;
        int c1 = qid * 2 + 1;
#pragma unroll
        for (int ni = 0; ni < Mma_tile_p::CORES_N; ni++)
        {
            auto& acc = ctile_t.acc_[0][0];
            if (c0 == row_local)
            {
                diag_doo[row_local + 0] = acc.elt(ni * Mma_tile_p::CORES_M * 2 + 0 * 2 + 0) * d_scale_do_o_;
            }
            if (c1 == row_local)
            {
                diag_doo[row_local + 0] = acc.elt(ni * Mma_tile_p::CORES_M * 2 + 0 * 2 + 1) * d_scale_do_o_;
            }
            if (c0 == row_local + 8)
            {
                diag_doo[row_local + 8] = acc.elt(ni * Mma_tile_p::CORES_M * 2 + 1 * 2 + 0) * d_scale_do_o_;
            }
            if (c1 == row_local + 8)
            {
                diag_doo[row_local + 8] = acc.elt(ni * Mma_tile_p::CORES_M * 2 + 1 * 2 + 1) * d_scale_do_o_;
            }
            c0 += 8;
            c1 += 8;
        }
    }

    Fmha_dgrad_params params_;
    fused_multihead_attention::Block_info_padded<Kernel_traits::THREADS> const binfo_;
    int const tidx_;

    char* smem_aligned_;
    Shared_storage* smem_;

    float* m_ptr_;
    float* zi_ptr_;

    float* dq_tmp_;

    // uint32_t smem_nvvm_s_t;
    // uint32_t smem_nvvm_dp_t;

    // int row_, col_;

    uint32_t read_offset_;
    uint32_t write_offset_;
    int row_local_;
    int qid_;
    // int n_valid_;

    // Keeps track of max abs of dP.
    float amax_dp_;
    // Keeps track of max abs of dQKV.
    float amax_dqkv_;

    // U N S C A L E   F A C T O R S.
    float d_scale_do_o_;   // unscale Dg = dO x O'
    float d_scale_q_k_;    // unscale  P = Q  x K'
    float d_scale_do_v_;   // unscale dS = dO x V'
    float d_scale_s_do_;   // unscale dV = S  x dO
    float d_scale_dp_qkv_; // unscale dQ = dP  x K and dK = dP' x Q

    // S C A L E   F A C T O R S.
    float q_scale_dqkv_; // scale dQKV
    float q_scale_s_;    // scale dP
    float q_scale_dp_;   // scale dP
};

} // namespace dgrad
} // namespace hopper
} // namespace fmha
