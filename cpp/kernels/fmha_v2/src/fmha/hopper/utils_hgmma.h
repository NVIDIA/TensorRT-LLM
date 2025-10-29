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

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp16 Accumulator
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, bool TA, bool TB>
struct Hgmma_fp16
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<8, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[2])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16\n"
            "{\n"
            "   %0, %1\n"
            "}, %2, %3, 1, 1, 1, %4, %5;\n"

            : "+r"(acc[0]), "+r"(acc[1])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x32x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<32, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[8])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 \n"
            "},\n"
            "  %8, %9, 1, 1, 1, %10, %11;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<64, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[16])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15 \n"
            "},\n"
            "  %16, %17, 1, 1, 1, %18, %19;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<128, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[32])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31 \n"
            "},\n"
            "  %32, %33, 1, 1, 1, %34, %35;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x192x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<192, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[48])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n192k16.f16.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47 \n"
            "},\n"
            "  %48, %49, 1, 1, 1, %50, %51;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp16<256, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[64])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47,\n"
            "  %48, %49, %50, %51, %52, %53, %54, %55,\n"
            "  %56, %57, %58, %59, %60, %61, %62, %63 \n"
            "},\n"
            "  %64, %65, 1, 1, 1, %66, %67;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB, int N, bool /*ignored*/>
inline __device__ void hgmma_fp16(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[N / 4])
{
    Hgmma_fp16<N, TA, TB>::mma(desc_a, desc_b, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp32 Accumulator
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, bool TA, bool TB>
struct Hgmma_fp32
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp32<8, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[4])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, %6, %7;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp32<64, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[32])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31 \n"
            "},\n"
            "  %32, %33, 1, 1, 1, %34, %35;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp32<128, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[64])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47,\n"
            "  %48, %49, %50, %51, %52, %53, %54, %55,\n"
            "  %56, %57, %58, %59, %60, %61, %62, %63 \n"
            "},\n"
            "  %64, %65, 1, 1, 1, %66, %67;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x192x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp32<192, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[96])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16\n"
            "{\n"
            "    %0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,\n"
            "    %8,   %9,  %10,  %11,  %12,  %13,  %14,  %15,\n"
            "   %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,\n"
            "   %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,\n"
            "   %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,\n"
            "   %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,\n"
            "   %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,\n"
            "   %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,\n"
            "   %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,\n"
            "   %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,\n"
            "   %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,\n"
            "   %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95 \n"
            "},\n"
            "  %96, %97, 1, 1, 1, %98, %99;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63]), "+r"(acc[64]), "+r"(acc[65]), "+r"(acc[66]), "+r"(acc[67]), "+r"(acc[68]), "+r"(acc[69]),
            "+r"(acc[70]), "+r"(acc[71]), "+r"(acc[72]), "+r"(acc[73]), "+r"(acc[74]), "+r"(acc[75]), "+r"(acc[76]),
            "+r"(acc[77]), "+r"(acc[78]), "+r"(acc[79]), "+r"(acc[80]), "+r"(acc[81]), "+r"(acc[82]), "+r"(acc[83]),
            "+r"(acc[84]), "+r"(acc[85]), "+r"(acc[86]), "+r"(acc[87]), "+r"(acc[88]), "+r"(acc[89]), "+r"(acc[90]),
            "+r"(acc[91]), "+r"(acc[92]), "+r"(acc[93]), "+r"(acc[94]), "+r"(acc[95])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB>
struct Hgmma_fp32<256, TA, TB>
{
    static inline __device__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[128])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_a = TA ? 1 : 0;
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
            "{\n"
            "    %0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,\n"
            "    %8,   %9,  %10,  %11,  %12,  %13,  %14,  %15,\n"
            "   %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,\n"
            "   %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,\n"
            "   %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,\n"
            "   %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,\n"
            "   %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,\n"
            "   %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,\n"
            "   %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,\n"
            "   %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,\n"
            "   %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,\n"
            "   %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,\n"
            "   %96,  %97,  %98,  %99, %100, %101, %102, %103,\n"
            "  %104, %105, %106, %107, %108, %109, %110, %111,\n"
            "  %112, %113, %114, %115, %116, %117, %118, %119,\n"
            "  %120, %121, %122, %123, %124, %125, %126, %127 \n"
            "},\n"
            "  %128, %129, 1, 1, 1, %130, %131;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63]), "+r"(acc[64]), "+r"(acc[65]), "+r"(acc[66]), "+r"(acc[67]), "+r"(acc[68]), "+r"(acc[69]),
            "+r"(acc[70]), "+r"(acc[71]), "+r"(acc[72]), "+r"(acc[73]), "+r"(acc[74]), "+r"(acc[75]), "+r"(acc[76]),
            "+r"(acc[77]), "+r"(acc[78]), "+r"(acc[79]), "+r"(acc[80]), "+r"(acc[81]), "+r"(acc[82]), "+r"(acc[83]),
            "+r"(acc[84]), "+r"(acc[85]), "+r"(acc[86]), "+r"(acc[87]), "+r"(acc[88]), "+r"(acc[89]), "+r"(acc[90]),
            "+r"(acc[91]), "+r"(acc[92]), "+r"(acc[93]), "+r"(acc[94]), "+r"(acc[95]), "+r"(acc[96]), "+r"(acc[97]),
            "+r"(acc[98]), "+r"(acc[99]), "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103]),
            "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107]), "+r"(acc[108]), "+r"(acc[109]),
            "+r"(acc[110]), "+r"(acc[111]), "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115]),
            "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119]), "+r"(acc[120]), "+r"(acc[121]),
            "+r"(acc[122]), "+r"(acc[123]), "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
            : "l"(desc_a), "l"(desc_b), "n"(trans_a), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TA, bool TB, int N, bool /*ignored*/>
inline __device__ void hgmma_fp32(uint64_t desc_a, uint64_t desc_b, uint32_t (&acc)[N / 2])
{
    Hgmma_fp32<N, TA, TB>::mma(desc_a, desc_b, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp16 Accumulator, where A is coming from RF
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, bool TB>
struct Hgmma_rfa_fp16
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<8, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[2])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, %6, 1, 1, 1, %7;\n"

            : "+r"(acc[0]), "+r"(acc[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_a), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x16x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<16, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[4])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 "
            "{ %0,  %1,  %2,  %3 },\n"
            "{ %4, %5, %6, %7 }, %8, 1, 1, 1, %9;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x32x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<32, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[8])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16 "
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7 },\n"
            "{ %8, %9, %10, %11 }, %12, 1, 1, 1, %13;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<64, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[16])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
            "{"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15 \n"
            "},\n"
            "{ %16, %17, %18, %19 }, %20, 1, 1, 1, %21;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<128, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[32])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
            "{"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31 \n"
            "},\n"
            "{ %32, %33, %34, %35 }, %36, 1, 1, 1, %37;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x192x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<192, TB>
{
    static inline __device__ void mma(const uint32_t (&a)[4], uint64_t desc_b, uint32_t (&acc)[48])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n192k16.f16.f16.f16 "
            "{"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47 \n"
            "},\n"
            "{ %48, %49, %50, %51 }, %52, 1, 1, 1, %53;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp16<256, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[64])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
            "{"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47,\n"
            "  %48, %49, %50, %51, %52, %53, %54, %55,\n"
            "  %56, %57, %58, %59, %60, %61, %62, %63 \n"
            "},\n"
            "{ %64, %65, %66, %67 }, %68, 1, 1, 1, %69;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB, int N, bool /*ignored*/>
inline __device__ void hgmma_rfa_fp16(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[N / 4])
{
    Hgmma_rfa_fp16<N, TB>::mma(a, desc_b, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp32 Accumulator, where A is coming from RF
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N, bool TB>
struct Hgmma_rfa_fp32
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<8, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[4])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
            "{\n"
            "  %0, %1, %2, %3\n"
            "}\n,"
            "{ %4, %5, %6, %7 }, %8, 1, 1, 1, %9;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x32x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<32, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[16])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15 \n"
            "},\n"
            "{ %16, %17, %18, %19 }, %20, 1, 1, 1, %21;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<64, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[32])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31 \n"
            "},\n"
            "{ %32, %33, %34, %35 }, %36, 1, 1, 1, %37;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<128, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[64])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
            "{\n"
            "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,\n"
            "   %8,  %9, %10, %11, %12, %13, %14, %15,\n"
            "  %16, %17, %18, %19, %20, %21, %22, %23,\n"
            "  %24, %25, %26, %27, %28, %29, %30, %31,\n"
            "  %32, %33, %34, %35, %36, %37, %38, %39,\n"
            "  %40, %41, %42, %43, %44, %45, %46, %47,\n"
            "  %48, %49, %50, %51, %52, %53, %54, %55,\n"
            "  %56, %57, %58, %59, %60, %61, %62, %63 \n"
            "},\n"
            "{ %64, %65, %66, %67 }, %68, 1, 1, 1, %69;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x192x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<192, TB>
{
    static inline __device__ void mma(const uint32_t (&a)[4], uint64_t desc_b, uint32_t (&acc)[96])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16\n"
            "{\n"
            "    %0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,\n"
            "    %8,   %9,  %10,  %11,  %12,  %13,  %14,  %15,\n"
            "   %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,\n"
            "   %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,\n"
            "   %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,\n"
            "   %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,\n"
            "   %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,\n"
            "   %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,\n"
            "   %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,\n"
            "   %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,\n"
            "   %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,\n"
            "   %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95 \n"
            "},\n"
            "{ %96, %97, %98, %99 }, %100, 1, 1, 1, %101;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63]), "+r"(acc[64]), "+r"(acc[65]), "+r"(acc[66]), "+r"(acc[67]), "+r"(acc[68]), "+r"(acc[69]),
            "+r"(acc[70]), "+r"(acc[71]), "+r"(acc[72]), "+r"(acc[73]), "+r"(acc[74]), "+r"(acc[75]), "+r"(acc[76]),
            "+r"(acc[77]), "+r"(acc[78]), "+r"(acc[79]), "+r"(acc[80]), "+r"(acc[81]), "+r"(acc[82]), "+r"(acc[83]),
            "+r"(acc[84]), "+r"(acc[85]), "+r"(acc[86]), "+r"(acc[87]), "+r"(acc[88]), "+r"(acc[89]), "+r"(acc[90]),
            "+r"(acc[91]), "+r"(acc[92]), "+r"(acc[93]), "+r"(acc[94]), "+r"(acc[95])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB>
struct Hgmma_rfa_fp32<256, TB>
{
    static inline __device__ void mma(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[128])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
        int const trans_b = TB ? 1 : 0;
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
            "{\n"
            "    %0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,\n"
            "    %8,   %9,  %10,  %11,  %12,  %13,  %14,  %15,\n"
            "   %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,\n"
            "   %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,\n"
            "   %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,\n"
            "   %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,\n"
            "   %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,\n"
            "   %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,\n"
            "   %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,\n"
            "   %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,\n"
            "   %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,\n"
            "   %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,\n"
            "   %96,  %97,  %98,  %99, %100, %101, %102, %103,\n"
            "  %104, %105, %106, %107, %108, %109, %110, %111,\n"
            "  %112, %113, %114, %115, %116, %117, %118, %119,\n"
            "  %120, %121, %122, %123, %124, %125, %126, %127 \n"
            "},\n"
            "{ %128, %129, %130, %131 }, %132, 1, 1, 1, %133;\n"

            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]),
            "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]),
            "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]),
            "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
            "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]), "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]),
            "+r"(acc[35]), "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]), "+r"(acc[40]), "+r"(acc[41]),
            "+r"(acc[42]), "+r"(acc[43]), "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]), "+r"(acc[48]),
            "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]), "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
            "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]), "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]),
            "+r"(acc[63]), "+r"(acc[64]), "+r"(acc[65]), "+r"(acc[66]), "+r"(acc[67]), "+r"(acc[68]), "+r"(acc[69]),
            "+r"(acc[70]), "+r"(acc[71]), "+r"(acc[72]), "+r"(acc[73]), "+r"(acc[74]), "+r"(acc[75]), "+r"(acc[76]),
            "+r"(acc[77]), "+r"(acc[78]), "+r"(acc[79]), "+r"(acc[80]), "+r"(acc[81]), "+r"(acc[82]), "+r"(acc[83]),
            "+r"(acc[84]), "+r"(acc[85]), "+r"(acc[86]), "+r"(acc[87]), "+r"(acc[88]), "+r"(acc[89]), "+r"(acc[90]),
            "+r"(acc[91]), "+r"(acc[92]), "+r"(acc[93]), "+r"(acc[94]), "+r"(acc[95]), "+r"(acc[96]), "+r"(acc[97]),
            "+r"(acc[98]), "+r"(acc[99]), "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103]),
            "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107]), "+r"(acc[108]), "+r"(acc[109]),
            "+r"(acc[110]), "+r"(acc[111]), "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115]),
            "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119]), "+r"(acc[120]), "+r"(acc[121]),
            "+r"(acc[122]), "+r"(acc[123]), "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "n"(trans_b));
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool TB, int N, bool /*ignored*/>
inline __device__ void hgmma_rfa_fp32(uint32_t const (&a)[4], uint64_t desc_b, uint32_t (&acc)[N / 2])
{
    Hgmma_rfa_fp32<N, TB>::mma(a, desc_b, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
