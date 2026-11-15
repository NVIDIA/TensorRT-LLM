/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * reference: https://github.com/deepseek-ai/FlashMLA (csrc/kerutils)
 */

// The subset of FlashMLA's kerutils device helpers that the CSA2 sparse decode kernel uses: tcgen05 UMMA issue
// without elect_one_sync (and with the .ws variant), TMEM 32dp32b loads/stores, TMA gather4 and packed fp32x2 math.
// Every function here emits SM100-family instructions; callers only reach them from code compiled for that family.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseDecode
{

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

// tcgen05.mma.ws.cta_group::1.kind::f16 with A in TMEM and B from a shared memory descriptor. CuTe's UMMA atoms do
// not cover .ws and elect inside the atom; this atom is issued by one already elected thread.
template <class AType, class BType, class CType, int M, int N, cute::UMMA::Major AMajor, cute::UMMA::Major BMajor>
struct MmaWsTsBf16NoElect
{
    static_assert(M == 32 || M == 64 || M == 128, "M must be 32, 64 or 128 for a 1-CTA .ws MMA");
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128 or 256 for a 1-CTA .ws MMA");

    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint32_t const& tmemA, uint64_t const& descB, uint32_t const& tmemC,
        uint32_t const& scaleC, uint64_t const& idescE)
    {
#if defined(__CUDA_ARCH__)
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; \n\t"
            "}\n"
            :
            : "r"(tmemC), "r"(tmemA), "l"(descB), "r"(uint32_t(idescE >> 32)), "r"(scaleC));
#endif
    }
};

// tcgen05.mma.ws.cta_group::1.kind::f16 with A and B from shared memory descriptors.
template <class AType, class BType, class CType, int M, int N, cute::UMMA::Major AMajor, cute::UMMA::Major BMajor>
struct MmaWsSsBf16NoElect
{
    static_assert(M == 32 || M == 64 || M == 128, "M must be 32, 64 or 128 for a 1-CTA .ws MMA");
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128 or 256 for a 1-CTA .ws MMA");

    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint64_t const& descA, uint64_t const& descB, uint32_t const& tmemC,
        uint32_t const& scaleC, uint64_t const& idescE)
    {
#if defined(__CUDA_ARCH__)
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
            "}\n"
            :
            : "r"(tmemC), "l"(descA), "l"(descB), "r"(uint32_t(idescE >> 32)), "r"(scaleC));
#endif
    }
};

} // namespace kernels::csa2SparseDecode

TRTLLM_NAMESPACE_END

namespace cute
{

template <class AType, class BType, class CType, int M, int N, UMMA::Major AMajor, UMMA::Major BMajor>
struct MMA_Traits<
    tensorrt_llm::kernels::csa2SparseDecode::MmaWsTsBf16NoElect<AType, BType, CType, M, N, AMajor, BMajor>>
{
    using ValTypeD = CType;
    using ValTypeA = AType;
    using ValTypeB = BType;
    using ValTypeC = CType;
    static_assert(cute::sizeof_bits_v<AType> == 16 && cute::sizeof_bits_v<BType> == 16, "16-bit A/B types only");

    // A in TMEM is really "duplicated"; NonInterleaved gives the same column addressing for this use.
    using FrgTypeA = UMMA::tmem_frg_1sm<AType, AType, UMMA::TmemAllocMode::NonInterleaved>;
    using FrgTypeB = UMMA::smem_desc<BMajor>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<CType>;

    // Logical shape-K is always 256 bits.
    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<AType, BType, CType, M, N, AMajor, BMajor>();

    template <class TD, class DLayout, class TA, class ALayout_, class TB, class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits, Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout_> const& A, Tensor<TB, BLayout_> const& B, Tensor<TC, CLayout_> const& C)
    {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint32_t tmemA = raw_pointer_cast(A.data());
        uint64_t descB = B[0];
        uint32_t tmemC = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

        tensorrt_llm::kernels::csa2SparseDecode::MmaWsTsBf16NoElect<AType, BType, CType, M, N, AMajor, BMajor>::fma(
            tmemA, descB, tmemC, uint32_t(traits.accumulate_), idesc);
    }
};

template <class AType, class BType, class CType, int M, int N, UMMA::Major AMajor, UMMA::Major BMajor>
struct MMA_Traits<
    tensorrt_llm::kernels::csa2SparseDecode::MmaWsSsBf16NoElect<AType, BType, CType, M, N, AMajor, BMajor>>
{
    using ValTypeD = CType;
    using ValTypeA = AType;
    using ValTypeB = BType;
    using ValTypeC = CType;
    static_assert(cute::sizeof_bits_v<AType> == 16 && cute::sizeof_bits_v<BType> == 16, "16-bit A/B types only");

    using FrgTypeA = UMMA::smem_desc<AMajor>;
    using FrgTypeB = UMMA::smem_desc<BMajor>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<CType>;

    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

    UMMA::InstrDescriptor idesc_ = UMMA::make_instr_desc<AType, BType, CType, M, N, AMajor, BMajor>();

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB, class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits, Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout_> const& A, Tensor<TB, BLayout_> const& B, Tensor<TC, CLayout_> const& C)
    {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TA>::value, "Expected desc registers in MMA_Atom::call");
        static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint64_t descA = A[0];
        uint64_t descB = B[0];
        uint32_t tmemC = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

        tensorrt_llm::kernels::csa2SparseDecode::MmaWsSsBf16NoElect<AType, BType, CType, M, N, AMajor, BMajor>::fma(
            descA, descB, tmemC, uint32_t(traits.accumulate_), idesc);
    }
};

} // namespace cute

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseDecode
{

// TMA gather4: four whole rows (row coordinates are int32) of a 2D tensor map into shared memory.
CUTE_DEVICE void tmaGather4(
    void const* descPtr, transac_bar_t& mbar, void* smemPtr, int colIdx, int4 rowIdxs, int64_t cacheHint)
{
    uint32_t const smemAddr = cute::cast_smem_ptr_to_uint(smemPtr);
    uint32_t const mbarAddr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_"
        "hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smemAddr), "l"(descPtr), "r"(colIdx), "r"(rowIdxs.x), "r"(rowIdxs.y), "r"(rowIdxs.z), "r"(rowIdxs.w),
        "r"(mbarAddr), "l"(cacheHint)
        : "memory");
}

CUTE_DEVICE float2 float2Add(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("add.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE float2 float2Mul(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// a * b + c
CUTE_DEVICE float2 float2Fma(float2 const& a, float2 const& b, float2 const& c)
{
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

// tcgen05.commit.cta_group::1: arrive on bar once the MMAs issued by this thread complete.
CUTE_DEVICE void ummaArriveNoElect(transac_bar_t& bar)
{
    uint32_t const barAddr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" : : "r"(barAddr));
}

__device__ __forceinline__ void tcgen05BeforeThreadSync()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05AfterThreadSync()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// 32 data-path lanes x 32 bit x 32 columns from TMEM into 32 registers.
__device__ __forceinline__ void tmemLoad32dp32b32x(uint32_t tmemStart, void* data)
{
    uint32_t* d = static_cast<uint32_t*>(data);
    [&]<size_t... Is>(cute::index_sequence<Is...>)
    { cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmemStart, d[Is]...); }(cute::make_index_sequence<32>{});
}

// tcgen05.ld.red exists on SM103 only; the SM100-family build of the kernels does without it.
#if defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030)
#define CSA2_SPARSE_DECODE_TMEM_LOAD_MAX 1
#endif

//! As tmemLoad32dp32b32x, and the maximum magnitude of the 32 loaded FP32 values (SM103 tcgen05.ld.red).
__device__ __forceinline__ void tmemLoad32dp32b32xMaxAbs(uint32_t tmemStart, void* data, float& maxAbs)
{
#if defined(CSA2_SPARSE_DECODE_TMEM_LOAD_MAX)
    uint32_t* d = static_cast<uint32_t*>(data);
    asm volatile(
        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.abs.f32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]), "=r"(d[8]),
        "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]), "=r"(d[16]),
        "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]),
        "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31]), "=f"(maxAbs)
        : "r"(tmemStart));
#else
    __trap();
#endif
}

// 32 registers into 32 data-path lanes x 32 bit x 32 columns of TMEM.
__device__ __forceinline__ void tmemStore32dp32b32x(uint32_t tmemStart, void const* data)
{
    uint32_t const* d = static_cast<uint32_t const*>(data);
    [&]<size_t... Is>(cute::index_sequence<Is...>)
    { cute::SM100_TMEM_STORE_32dp32b32x::copy(d[Is]..., tmemStart); }(cute::make_index_sequence<32>{});
}

// UMMA with both operands in shared memory (CTA-local tiles); clearAccum overwrites C with the first K step.
template <typename TiledMma, typename TensorA, typename TensorB, typename TensorFragC>
CUTE_DEVICE void utcmmaSs(TiledMma& tiledMma, TensorA sA, TensorB sB, TensorFragC tCFrag, bool clearAccum)
{
    using namespace cute;
    tiledMma.accumulate_ = clearAccum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thrMma = tiledMma.get_slice(_0{});
    auto sAFrag = thrMma.partition_fragment_A(sA);
    auto sBFrag = thrMma.partition_fragment_B(sB);
    static_assert(size<2>(sAFrag) == size<2>(sBFrag));
    static_assert(size<1>(sAFrag) == size<1>(tCFrag));
    static_assert(size<1>(sBFrag) == size<2>(tCFrag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sAFrag); ++k)
    {
        cute::gemm(tiledMma, sAFrag(_, _, k), sBFrag(_, _, k), tCFrag);
        tiledMma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// UMMA with A in TMEM and B in shared memory.
template <typename TiledMma, typename TensorA, typename TensorB, typename TensorFragC>
CUTE_DEVICE void utcmmaTs(TiledMma& tiledMma, TensorA tAFrag, TensorB sB, TensorFragC tCFrag, bool clearAccum)
{
    using namespace cute;
    tiledMma.accumulate_ = clearAccum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thrMma = tiledMma.get_slice(_0{});
    auto sBFrag = thrMma.partition_fragment_B(sB);
    static_assert(size<2>(tAFrag) == size<2>(sBFrag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tAFrag); ++k)
    {
        cute::gemm(tiledMma, tAFrag(_, _, k), sBFrag(_, _, k), tCFrag);
        tiledMma.accumulate_ = UMMA::ScaleOut::One;
    }
}

} // namespace kernels::csa2SparseDecode

TRTLLM_NAMESPACE_END
