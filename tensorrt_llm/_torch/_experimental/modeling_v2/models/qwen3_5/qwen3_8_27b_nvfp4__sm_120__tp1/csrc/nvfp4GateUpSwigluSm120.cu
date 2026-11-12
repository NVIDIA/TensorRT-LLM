/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// SM120 NVFP4 W4A4 gate_up GEMM with SwiGLU + NVFP4 (group-16, E4M3 SF,
// 128x4-swizzled) quantize in the epilogue: it writes down_proj's FP4 input
// and SF directly instead of the [M, 2I] BF16 gate_up output.
//
// Mainloop: identical to nvfp4_nvfp4_gemm_template_sm120.h (Sm120
// OpClassBlockScaledTensorOp, cooperative, StaticPersistentScheduler,
// swizzle 1, raster heuristic). Epilogue: register-only operator epilogue.
//
// Weight layout (gate_up rows permuted at weight load): inside every
// 128-row tile t, row c = 32a + 16nw + 8b + 2lq + p holds gate (p = 0) or up
// (p = 1) of output j = 64t + 32nw + 16(a/2) + 4lq + 2(a%2) + b. With the SM120
// tiled MMA (warps 4x2, N permutation (8,2,2):(1,16,8)) each thread's
// accumulator pair is (gate_j, up_j) and a lane quad owns two whole 16-wide
// SF blocks per row.
//
// Per-element math is a bitwise copy of the unfused path (nvfp4_gemm with BF16
// output, then fused_silu_mul_nvfp4):
//   LinCombPerColBias (null bias): bf16_rn(alpha * acc + 0)
//   fused_silu_mul_nvfp4.py: h = bf16_rn((u + 0) * (sigmoid(g) * g)) with
//   tl.sigmoid lowered to div.full(1, ex2.approx(g * -log2e) + 1)
//   quant tail (cvt_warp_fp16_to_fp4): rcp.approx.ftz, mul.rn,
//   cvt.rn.satfinite.e4m3x2, cvt.rn.satfinite.e2m1x2.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4GateUpSwigluSm120.h"
#include "tensorrt_llm/common/assert.h"

#ifndef PLACEHOLDER_KERNELS

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/archCondition.h"

namespace cutlass::gemm::collective
{
// SM120 block-scaled TMA mainloop (Base, from the CollectiveBuilder) whose mma()
// issues only the MMAs of tiled-MMA M repeat 0 (CTA tile rows 0-63). For problems
// with M <= 64 rows 64-127 are TMA zero fill and never stored, so the MMAs on them
// are dead work: at decode M <= 64 they are half the tensor-pipe time of the
// 128-row tile. Everything else is a verbatim copy of Base::mma(); every kept
// accumulator sees the same k_block sequence, so the result is bitwise the same.
// The skipped accumulators keep the zeros of clear(accum).
template <class Base>
struct Sm120BlockScaledLowerHalfM : Base
{
    using typename Base::SmemLayoutA;
    using typename Base::SmemLayoutB;
    using typename Base::SmemLayoutSFA;
    using typename Base::SmemLayoutSFB;
    using typename Base::TiledMma;
    using typename Base::SmemCopyAtomA;
    using typename Base::SmemCopyAtomB;
    using typename Base::SmemCopyAtomSFA;
    using typename Base::SmemCopyAtomSFB;
    using typename Base::DispatchPolicy;
    using typename Base::MainloopPipeline;
    using typename Base::PipelineState;
    using typename Base::TensorStorage;
    using typename Base::Params;

    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum,
        int k_tile_count, int thread_idx, TensorStorage& shared_tensors, [[maybe_unused]] Params const& params)
    {
        using namespace cute;

        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(decltype(size<1>(accum))::value == 2, "lower-half mainloop needs MMA_M == 2 (128-row tile)");

        clear(accum);

        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
        Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
        Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

        Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

        Tensor tCrSFA = this->partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
        Tensor tCrSFB = this->partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        auto tile_shape_mnk = tile_shape(tiled_mma);
        auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{}, this->get_layoutSFA_TV(tiled_mma),
            make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
        Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

        auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{}, this->get_layoutSFB_TV(tiled_mma),
            make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
        Tensor tCsSFB = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
        Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));

        auto K_BLOCK_MAX = size<2>(tCrA);

        int read_stage = smem_pipe_read.index();
        auto tCsA_stage = tCsA(_, _, _, read_stage);
        auto tCsB_stage = tCsB(_, _, _, read_stage);
        auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);
        auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

        auto copy_kblock = [&](auto k_block)
        {
            copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block), tCrB_copy_view(_, _, k_block));
            using MMAOp = typename TiledMma::MMA_Op;
            fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
            fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));
            copy(tCsSFA_stage(_, _, k_block), tCrSFA_copy_view(_, _, k_block));
            copy(tCsSFB_stage(_, _, k_block), tCrSFB_copy_view(_, _, k_block));
        };

        auto gemm_kblock = [&](auto k_block)
        {
            // (V,M) x (V,N) => (V,M,N) restricted to M repeat 0: (V) x (V) => (V) per N repeat.
            auto tA = make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block));
            auto tB = make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block));
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<1>(tB); ++n)
            {
                auto acc_n = accum(_, 0, n);
                cute::gemm(tiled_mma, tA(_, 0), tB(_, n), acc_n);
            }
        };

        pipeline.consumer_wait(smem_pipe_read);

        copy_kblock(_0{});
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 1; --k_tile_count)
        {
            for_each(make_int_sequence<K_BLOCK_MAX>{},
                [&](auto k_block)
                {
                    auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

                    if (k_block == K_BLOCK_MAX - 1)
                    {
                        cutlass::arch::NamedBarrier::sync(
                            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                        pipeline.consumer_release(smem_pipe_read);
                        ++smem_pipe_read;
                        read_stage = smem_pipe_read.index();
                        tCsA_stage = tCsA(_, _, _, read_stage);
                        tCsB_stage = tCsB(_, _, _, read_stage);
                        tCsSFA_stage = tCsSFA(_, _, _, read_stage);
                        tCsSFB_stage = tCsSFB(_, _, _, read_stage);
                        pipeline.consumer_wait(smem_pipe_read);
                    }

                    copy_kblock(k_block_next);
                    gemm_kblock(k_block);
                });
        }

        for_each(make_int_sequence<K_BLOCK_MAX>{},
            [&](auto k_block)
            {
                auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

                if (k_block == K_BLOCK_MAX - 1)
                {
                    cutlass::arch::NamedBarrier::sync(
                        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                    pipeline.consumer_release(smem_pipe_read);
                    ++smem_pipe_read;
                }

                if (k_block_next > 0)
                {
                    copy_kblock(k_block_next);
                }
                gemm_kblock(k_block);
            });
    }
};
} // namespace cutlass::gemm::collective

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace swigluNvfp4Sm120
{
using namespace cute;

__device__ __forceinline__ float rcpApproxFtz(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float mulRn(float a, float b)
{
    float y;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float addRn(float a, float b)
{
    float y;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float ex2Approx(float a)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
    return y;
}

__device__ __forceinline__ float divFull(float a, float b)
{
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float bf16Round(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint32_t cvtE4m3(float x)
{
    uint32_t r;
    asm("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\ncvt.u32.u16 %0, t;\n}" : "=r"(r) : "f"(x), "f"(0.0f));
    return r & 0xff;
}

__device__ __forceinline__ float e4m3ToF32(uint32_t b)
{
    uint32_t h2;
    uint16_t v = static_cast<uint16_t>(b);
    asm("{\ncvt.rn.f16x2.e4m3x2 %0, %1;\n}" : "=r"(h2) : "h"(v));
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffff)));
}

__device__ __forceinline__ uint32_t cvtE2m1x4(float e0, float e1, float e2, float e3)
{
    uint32_t r;
    asm("{\n.reg .b8 b0;\n.reg .b8 b1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "mov.b32 %0, {b0, b1, b0, b1};\n}"
        : "=r"(r)
        : "f"(e0), "f"(e1), "f"(e2), "f"(e3));
    return r & 0xffff;
}

__device__ __forceinline__ float swigluElem(float alpha, float accGate, float accUp)
{
    // gate_up GEMM epilogue (LinCombPerColBias, null bias, beta 0) -> BF16.
    float const g = bf16Round(__fmaf_rn(alpha, accGate, 0.0f));
    float const u = bf16Round(__fmaf_rn(alpha, accUp, 0.0f));
    // tl.sigmoid as Triton lowers it for fused_silu_mul_nvfp4_kernel.
    float const sig = divFull(1.0f, addRn(ex2Approx(mulRn(g, -1.44269502162933349609375f)), 1.0f));
    return bf16Round(mulRn(addRn(u, 0.0f), mulRn(sig, g)));
}

template <class StrideC_, class StrideD_>
class SwigluNvfp4Epilogue
{
public:
    using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
    using DispatchPolicy = EpilogueSchedule;
    using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 1, float, float>;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementC = void;
    using StrideC = StrideC_;
    using ElementD = cutlass::bfloat16_t;
    using StrideD = StrideD_;
    using GmemElementC = ElementD;
    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    struct SharedStorage
    {
    };

    using TensorStorage = SharedStorage;

    struct Arguments
    {
        float const* alpha_ptr = nullptr;
        float const* out_scale_ptr = nullptr;
        uint8_t* fp4_ptr = nullptr;
        uint8_t* sf_ptr = nullptr;
        int num_k_tiles = 0;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(
        ProblemShape const&, Arguments const&, void*, cudaStream_t, cutlass::CudaHostAdapter* = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const&, Arguments const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE
    SwigluNvfp4Epilogue(Params const& params_, SharedStorage const& = SharedStorage())
        : params(params_)
    {
    }

    CUTLASS_DEVICE bool is_source_needed()
    {
        return false;
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout,
        class TiledMma, class ResidueMNK>
    CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK, BlockCoordMNKL blk_coord_mnkl,
        cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TiledMma tiled_mma, ResidueMNK, int thread_idx, char*)
    {
        constexpr int BLK_M = size<0>(BlockShapeMNK{});
        constexpr int BLK_N = size<1>(BlockShapeMNK{});
        static_assert(BLK_N == 128, "tile N must be 128");
        int const M = get<0>(problem_shape_mnkl);
        int const N = get<1>(problem_shape_mnkl);
        auto m_coord = get<0>(blk_coord_mnkl);
        auto n_coord = get<1>(blk_coord_mnkl);

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor cD = make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{}));
        Tensor tCcD = thr_mma.partition_C(cD);
        static_assert(size<0>(FrgLayout{}) == 4, "accumulator fragment must be (2,2)");
        constexpr int MMA_M = size<1>(FrgLayout{});
        constexpr int MMA_N = size<2>(FrgLayout{});
        static_assert(MMA_N == 8, "MMA_N must be 8");

        float const alpha = *params.alpha_ptr;
        float const s = *params.out_scale_ptr;
        float const r6 = rcpApproxFtz(6.0f);
        float const rs = rcpApproxFtz(s);

        int const n0 = get<1>(tCcD(0, 0, 0));
        int const nw = n0 / 16;
        int const lq = (n0 % 16) / 2;
        int const d = N / 2;
        int const m_pad = (M + 127) / 128 * 128;
        int const j_base = int(n_coord) * (BLK_N / 2) + 32 * nw + 4 * lq;
        int const m_base = int(m_coord) * BLK_M;
        int const nkt = params.num_k_tiles;

        CUTLASS_PRAGMA_UNROLL
        for (int im = 0; im < MMA_M; ++im)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int r2 = 0; r2 < 2; ++r2)
            {
                int const row = m_base + get<0>(tCcD(2 * r2, im, 0));
                // MMA_N index k covers tile col offset 32a + 8b (static) ->
                // output j_base + 16(a/2) + 2(a%2) + b.
                float hv[2][4];
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < MMA_N; ++k)
                {
                    // Column offset ck = 32a + 8b of MMA_N index k = 2a + b (the
                    // 8-col atom repeat b is the fastest RestN mode, then the
                    // 32-col permutation tile a). Compile-time after unrolling,
                    // so hv[] stays in registers (reading ck from tCcD made the
                    // indices runtime and put hv in local memory).
                    int const a_ = k / 2;
                    int const b_ = k % 2;
                    hv[a_ / 2][2 * (a_ % 2) + b_]
                        = swigluElem(alpha, accumulators(2 * r2, im, k), accumulators(2 * r2 + 1, im, k));
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sb = 0; sb < 2; ++sb)
                {
                    float vmax = 0.0f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int e = 0; e < 4; ++e)
                    {
                        vmax = fmaxf(vmax, fabsf(hv[sb][e]));
                    }
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 1));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 2));
                    float const sfv = mulRn(s, mulRn(vmax, r6));
                    uint32_t const sfb = cvtE4m3(sfv);
                    float const sff = e4m3ToF32(sfb);
                    float const osc = (vmax != 0.0f) ? rcpApproxFtz(mulRn(sff, rs)) : 0.0f;
                    uint32_t const packed = cvtE2m1x4(
                        mulRn(hv[sb][0], osc), mulRn(hv[sb][1], osc), mulRn(hv[sb][2], osc), mulRn(hv[sb][3], osc));
                    int const j = j_base + 16 * sb;
                    if (row < M)
                    {
                        *reinterpret_cast<uint16_t*>(params.fp4_ptr + size_t(row) * (d / 2) + j / 2)
                            = static_cast<uint16_t>(packed);
                    }
                    if (lq == 0 && row < m_pad)
                    {
                        // get_sf_out_offset_128x4, batch 0; SF padding rows zeroed.
                        int const c = j / 16;
                        size_t const off = (size_t(row / 128) * nkt + c / 4) * 512 + (row % 32) * 16
                            + ((row % 128) / 32) * 4 + c % 4;
                        params.sf_ptr[off] = row < M ? static_cast<uint8_t>(sfb) : uint8_t(0);
                    }
                }
            }
        }
    }

private:
    Params params;
};

// LowerHalfM: mainloop skips the MMAs of tile rows 64-127 (valid only for M <= 64).
template <int TM, int TN, int TK, bool LowerHalfM = false>
struct SwigluNvfp4GemmSm120
{
    using CTAShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Arch = cutlass::arch::Sm120;
    using ElementA = cutlass::float_e2m1_t;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB = cutlass::float_e2m1_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;
    using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using StrideCD = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
    using EpiOp = SwigluNvfp4Epilogue<StrideCD, StrideCD>;
    using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<EpiOp>;
    using BuiltMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<Arch, cutlass::arch::OpClassBlockScaledTensorOp,
            ElementPairA, LayoutA, AlignmentA, ElementPairB, LayoutB, AlignmentB, float, CTAShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    static_assert(!LowerHalfM || TM == 128, "LowerHalfM needs a 128-row CTA tile");
    using CollectiveMainloop = cute::conditional_t<LowerHalfM,
        cutlass::gemm::collective::Sm120BlockScaledLowerHalfM<BuiltMainloop>, BuiltMainloop>;
    using BaseKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, cutlass::gemm::StaticPersistentScheduler>;

    struct GemmKernel : BaseKernel
    {
        using typename BaseKernel::Params;

        CUTLASS_DEVICE
        void operator()(Params const& params, char* smem_buf)
        {
            if constexpr (tensorrt_llm::kernels::arch::is_major_v<12>)
            {
                this->BaseKernel::operator()(params, smem_buf);
            }
            else
            {
                if (cute::thread0())
                {
                    printf("%s : This kernel shall only run on SM12x devices.\n", __PRETTY_FUNCTION__);
                    __trap();
                }
            }
        }
    };

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Returns the workspace size when A, B and fp4 are all null.
template <int TM, int TN, int TK, bool LowerHalfM = false>
size_t runSwigluNvfp4GemmSm120(void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* alpha, float const* out_scale, uint8_t* fp4, uint8_t* sf, int m, int n, int k, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(!LowerHalfM || m <= 64, "[SwiGLU NVFP4 GEMM] lower-half tile needs m <= 64, got %d", m);
    using G = SwigluNvfp4GemmSm120<TM, TN, TK, LowerHalfM>;
    using Gemm = typename G::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    int const d = n / 2;
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<typename G::ElementA const*>(A);
    args.mainloop.ptr_B = static_cast<typename G::ElementB const*>(B);
    args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(input_sf);
    args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(weight_sf);
    args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
    args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
    args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
    args.epilogue.alpha_ptr = alpha;
    args.epilogue.out_scale_ptr = out_scale;
    args.epilogue.fp4_ptr = fp4;
    args.epilogue.sf_ptr = sf;
    args.epilogue.num_k_tiles = (d / 16 + 3) / 4;
    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
    {
        args.scheduler.max_swizzle_size = 1;
    }
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>)
    {
        using Enum_t = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = Enum_t::Heuristic;
    }
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    Gemm gemm;
    if (!A && !B && !fp4)
    {
        return gemm.get_workspace_size(args);
    }
    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
    static int const maxSmem = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    TLLM_CHECK_WITH_INFO(smem_size <= maxSmem, "[SwiGLU NVFP4 GEMM] SMEM %d exceeds %d", smem_size, maxSmem);
    TLLM_CHECK_WITH_INFO(gemm.can_implement(args) == cutlass::Status::kSuccess,
        "[SwiGLU NVFP4 GEMM] cannot implement m=%d n=%d k=%d", m, n, k);
    size_t const ws = gemm.get_workspace_size(args);
    TLLM_CHECK_WITH_INFO(ws <= workspaceBytes, "[SwiGLU NVFP4 GEMM] workspace %zu > %zu", ws, workspaceBytes);
    auto st = gemm.initialize(args, workspace, stream);
    TLLM_CHECK_WITH_INFO(st == cutlass::Status::kSuccess, "[SwiGLU NVFP4 GEMM] initialize failed: %s",
        cutlass::cutlassGetStatusString(st));
    st = gemm.run(args, workspace, stream, nullptr, enablePdl);
    TLLM_CHECK_WITH_INFO(
        st == cutlass::Status::kSuccess, "[SwiGLU NVFP4 GEMM] run failed: %s", cutlass::cutlassGetStatusString(st));
    return ws;
}

} // namespace swigluNvfp4Sm120
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END

#endif // PLACEHOLDER_KERNELS

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

size_t nvfp4GateUpSwigluSm120(void const* A, void const* B, void const* inputSf, void const* weightSf,
    float const* alpha, float const* outScale, uint8_t* fp4Out, uint8_t* sfOut, int m, int n, int k, int tile,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, bool enablePdl)
{
#ifdef PLACEHOLDER_KERNELS
    TLLM_THROW("[SwiGLU NVFP4 GEMM] not compiled for this architecture.");
#else
    if (tile == 0)
    {
        return swigluNvfp4Sm120::runSwigluNvfp4GemmSm120<256, 128, 128>(A, B, inputSf, weightSf, alpha, outScale,
            fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
    }
    if (tile == 2)
    {
        // 128x128x256 with the MMAs of tile rows 64-127 skipped (m <= 64 only).
        return swigluNvfp4Sm120::runSwigluNvfp4GemmSm120<128, 128, 256, true>(A, B, inputSf, weightSf, alpha, outScale,
            fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
    }
    return swigluNvfp4Sm120::runSwigluNvfp4GemmSm120<128, 128, 256>(
        A, B, inputSf, weightSf, alpha, outScale, fp4Out, sfOut, m, n, k, workspace, workspaceBytes, stream, enablePdl);
#endif
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
