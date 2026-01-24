// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze
//   and Meng, Chenlin and Zhu, Jun-Yan and Han, Song}, journal={arXiv preprint arXiv:2411.05007}, year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "gemm_base.cuh"
#include "lora.cuh"

// #include "gemm_w4a4_block.cuh"

namespace nunchaku::kernels
{

template <typename Config>
class GEMM_W4A4;

#ifndef __INTELLISENSE__
template <typename Config>
class GEMM_W4A4 : public GEMMBase<Config>
{
#else
template <>
class GEMM_W4A4<GEMMConfig_W4A4_FP16> : public GEMMBase<GEMMConfig_W4A4_FP16>
{
    using Config = GEMMConfig_W4A4_FP16;
#endif

public:
    IMPORT_GEMM_BASE(Config);

public:
    // micro-scales for FP4 MMA
    // each uint32_t is a 4*32 matrix of scales (for MMA of 64*32)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    static constexpr bool FP4_AVAILABLE = true;
#else
    static constexpr bool FP4_AVAILABLE = false;
#endif

    __device__ __forceinline__ static void trap_no_fp4()
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        {
            printf("FP4 is not available on this device\n");
        }
        __syncthreads();
        __nanosleep(1000000);
        __trap();
    }

    static_assert(WARP_N % 32 == 0);
    static_assert(WARP_M % 32 == 0);

    static constexpr int WMSCALES_PACK_SIZE = clamp(WARP_N / 32, 1, 4);
    static constexpr int WMSCALES_NUM_PACKS = ceilDiv(WARP_N / 32, WMSCALES_PACK_SIZE);
    static constexpr int WMSCALES_VALID_LANES = WARP_SIZE;

    static constexpr int AMSCALES_PACK_SIZE = clamp(WARP_M / 32, 1, 4);
    static constexpr int AMSCALES_NUM_PACKS = ceilDiv(WARP_M / 32, AMSCALES_PACK_SIZE);
    static constexpr int AMSCALES_VALID_LANES = WARP_SIZE;

    struct packed_wmscale_t
    {
        uint32_t data[WMSCALES_PACK_SIZE];
    };

    struct packed_amscale_t
    {
        uint32_t data[AMSCALES_PACK_SIZE];
    };

    using amscale_warp = std::array<packed_amscale_t, AMSCALES_NUM_PACKS>;
    using wmscale_warp = std::array<packed_wmscale_t, WMSCALES_NUM_PACKS>;

    // amscales: [M / BLOCK_M, K / group size, NUM_WARPS, AMSCALES_NUM_PACKS, WARP_SIZE] of packed_amscale_t
    __device__ __forceinline__ static void load_amscale(
        packed_amscale_t const* ptr, int group, amscale_warp& out, bool pred)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;
#pragma unroll
        for (int i = 0; i < AMSCALES_NUM_PACKS; i++)
        {
            out[i] = load_pred(&ptr[(group * NUM_WARPS + warpId) * AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES
                                   + i * AMSCALES_VALID_LANES + laneId],
                pred);
        }
    }

    // wmscales: [N / BLOCK_N, 1, K / group size, WMSCALES_NUM_PACKS, WMSCALES_VALID_LANES] of packed_wmscale_t
    __device__ __forceinline__ static void load_wmscale(
        packed_wmscale_t const* ptr, int group, wmscale_warp& out, bool pred)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
#pragma unroll
        for (int i = 0; i < WMSCALES_NUM_PACKS; i++)
        {
            out[i] = load_pred(&ptr[(group * WMSCALES_NUM_PACKS + i) * WMSCALES_VALID_LANES + laneId], pred);
        }
    }

    __device__ __forceinline__ static void quantize_w4a4_fp4_from_fpsum_warp(
        const packed_fpsum_t (&fpsum)[INSN_K / INSN_N], packed_act_t& output, uint32_t& output_scale, int ida)
    {

        constexpr int NUM_GROUPS = 4;
        static_assert(NUM_GROUPS == INSN_K / INSN_N);

        constexpr float QVALUE_MAX = 6.0f;
        constexpr float RECPI_QVALUE_MAX = 1 / QVALUE_MAX;
        constexpr float MSCALE_MAX = 448.0f;

        int const laneId = threadIdx.x % WARP_SIZE;

        // 0 for row 0-7; 1 for row 8-15
        // each half2_t represents a 8*8 matrix
        half2_t input[2][INSN_K / INSN_N * 2];
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++)
        {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }

        auto maxabs = [](half2_t val) ALWAYSINLINE
        {
            val = __habs2(val);
            return __hmax(val.x, val.y);
        };

        // each half_t represents maxvalue in a 8*16 matrix
        half_t maxvalue[2][NUM_GROUPS];
#pragma unroll
        for (int i = 0; i < NUM_GROUPS; i++)
        {
            maxvalue[0][i] = __hmax(maxabs(input[0][i * 2]), maxabs(input[0][i * 2 + 1]));
            maxvalue[1][i] = __hmax(maxabs(input[1][i * 2]), maxabs(input[1][i * 2 + 1]));
        }
#pragma unroll
        for (int mask = 2; mask > 0; mask /= 2)
        {
#pragma unroll
            for (int i = 0; i < NUM_GROUPS; i++)
            {
                maxvalue[0][i] = __hmax(maxvalue[0][i], __shfl_xor_sync(~0, maxvalue[0][i], mask));
                maxvalue[1][i] = __hmax(maxvalue[1][i], __shfl_xor_sync(~0, maxvalue[1][i], mask));
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical maxvalue now

        float scale[2][NUM_GROUPS];
        float rscale[2][NUM_GROUPS];
#pragma unroll
        for (int i = 0; i < NUM_GROUPS; i++)
        {
            scale[0][i] = fminf(float(maxvalue[0][i]) * RECPI_QVALUE_MAX, MSCALE_MAX);
            scale[1][i] = fminf(float(maxvalue[1][i]) * RECPI_QVALUE_MAX, MSCALE_MAX);
            // TODO: check whether (1 / scale) or (1 / fp8scale) is better
            rscale[0][i] = cuda_frcp(scale[0][i]);
            rscale[1][i] = cuda_frcp(scale[1][i]);
        }
        uint32_t fp8scale[2];
        fp8scale[0] = quantize_float4_fp8(make_float4(scale[0][0], scale[0][1], scale[0][2], scale[0][3]));
        fp8scale[1] = quantize_float4_fp8(make_float4(scale[1][0], scale[1][1], scale[1][2], scale[1][3]));

        /**
         * output_scale pack format: (ida=0)
         * lane 0 => row 0 if ida==0
         * lane 1 => row 8 if ida==0
         * lane 2 => row 0 if ida==1
         * lane 3 => row 8 if ida==1
         * ...
         * lane i => quad (i/4) => row (i/4+8*(i%2)) if (i%4/2==ida) => srclane i, index i%2
         */
        if (laneId % 4 / 2 == ida)
        {
            output_scale = (laneId % 2 == 0) ? fp8scale[0] : fp8scale[1];
        }

        uint32_t qpacks[2][INSN_K / INSN_M * 2];
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 2; j++)
            {
                float2 fval = half22float2(input[j][i]) * make_float2(rscale[j][i / 2], rscale[j][i / 2]);
                qpacks[j][i] = quantize_float2_fp4(fval) << (laneId % 4 * 8);
            }
        }

#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2)
        {
#pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++)
            {
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if (laneId % 4 == i)
            {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    // m16n16k64 MMA
    // ida, idb in {0, 1}
    __device__ __forceinline__ static packed_f32psum_t mma_fp4(
        packed_act_t act, packed_wgt_t wgt, packed_f32psum_t psum, uint32_t amscale, uint32_t wmscale, int ida, int idb)
    {
        packed_f32psum_t out;
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "{%14}, {%15, %16}, "
            "{%17}, {%18, %19};"
            : "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
            : "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w), "r"(wgt.x), "r"(wgt.y), "f"(psum.data[0]),
            "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3]), "r"(amscale), "n"(0), "h"((short) ida),
            "r"(wmscale), "n"(0), "h"((short) (idb * 2)));
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "{%14}, {%15, %16}, "
            "{%17}, {%18, %19};"
            : "=f"(out.data[4]), "=f"(out.data[5]), "=f"(out.data[6]), "=f"(out.data[7])
            : "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w), "r"(wgt.z), "r"(wgt.w), "f"(psum.data[4]),
            "f"(psum.data[5]), "f"(psum.data[6]), "f"(psum.data[7]), "r"(amscale), "n"(0), "h"((short) ida),
            "r"(wmscale), "n"(0), "h"((short) (idb * 2 + 1)));
        return out;
    }

    __device__ __forceinline__ static void compute_fp4(
        act_warp A, wgt_warp W, amscale_warp amscale, wmscale_warp wmscale, f32psum_warp& psum)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

#pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++)
        {
#pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++)
            {
                psum[i * WARP_N_TILES + j] = mma_fp4(A[i], W[j], psum[i * WARP_N_TILES + j],
                    amscale[i / 2 / AMSCALES_PACK_SIZE].data[i / 2 % AMSCALES_PACK_SIZE],
                    wmscale[j / 2 / WMSCALES_PACK_SIZE].data[j / 2 % WMSCALES_PACK_SIZE], i % 2, j % 2);
            }
        }
    }

    template <typename Epilogue, bool USE_ALPHA>
    __device__ __forceinline__ static void gemm_w4a4_fp4_block(const BlockInfo binfo, packed_act_t const* act,
        packed_wgt_t const* wgt, packed_amscale_t const* ascales, packed_wmscale_t const* wscales,
        float alpha, // per-tensor scale of weight
        int M, int N, int K, Epilogue::Arguments const& epilogueArgs, bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        act_warp A[NUM_STAGES];           // 8 * 2
        wgt_warp W[NUM_STAGES];           // 32 * 2
        amscale_warp amscale[NUM_STAGES]; // 1 * 2
        wmscale_warp wmscale[NUM_STAGES]; // 4 * 2
        f32psum_warp fpsum;               // 128

        for (int k = 0; k < NUM_STAGES - 1; k++)
        {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_amscale(ascales, k, amscale[k], true);
            load_wmscale(wscales, k, wmscale[k], true);
        }

#pragma unroll
        for (auto& pack : fpsum)
        {
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                pack.data[i] = 0;
            }
        }

        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES)
        {
#pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++)
            {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_amscale(ascales, nextk, amscale[idx], pred);
                load_wmscale(wscales, nextk, wmscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute_fp4(A[k2], W[k2], amscale[k2], wmscale[k2], fpsum);

                if (alwaysfalse)
                {
                    dummy = clock();
                }

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

        if constexpr (USE_ALPHA)
        {
#pragma unroll
            for (auto& pack : fpsum)
            {
#pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    pack.data[i] *= alpha;
                }
            }
        }

        auto f16psum = packed_fp32_to_fp16(fpsum);

        CHECK_NAN(f16psum, "f16psum");

        Epilogue()(binfo, f16psum, M, N, K, epilogueArgs);
    }

    template <typename Epilogue, bool USE_ALPHA>
    struct gemm_w4a4_fp4_kernel
    {
        static constexpr int MIN_ARCH = 1200;

        __device__ void operator()(packed_act_t const* act, packed_wgt_t const* wgt, packed_amscale_t const* ascales,
            packed_wmscale_t const* wscales, float alpha, int M, int N, int K, Epilogue::Arguments epilogueArgs,
            bool swapBlockXY, bool alwaysfalse)
        {
            BlockInfo binfo = {
                .bm = (int) blockIdx.x,
                .bn = (int) blockIdx.y,
                .numBlocksM = (int) gridDim.x,
                .numBlocksN = (int) gridDim.y,
            };

            if (swapBlockXY)
            {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            int const bm = binfo.bm;
            int const bn = binfo.bn;

            if constexpr (FP4_AVAILABLE)
            {
                gemm_w4a4_fp4_block<Epilogue, USE_ALPHA>(binfo,
                    act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                    wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                    ascales + bm * (K / WARP_K) * NUM_WARPS * AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES,
                    wscales + bn * (K / WARP_K) * WMSCALES_NUM_PACKS * WMSCALES_VALID_LANES, alpha, M, N, K,
                    epilogueArgs, alwaysfalse);
            }
            else
            {
                trap_no_fp4();
            }
        }
    };

public:
    template <bool ACT_UNSIGNED>
    __device__ __forceinline__ static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt)
    {
        packed_psum_t psum;

        uint4 out1 = mma_m16n8kx_s32common<mma_helper::s4u4<ACT_UNSIGNED>, mma_helper::s4>(
            act, uint2(wgt.x, wgt.y), uint4(0, 0, 0, 0));
        uint4 out2 = mma_m16n8kx_s32common<mma_helper::s4u4<ACT_UNSIGNED>, mma_helper::s4>(
            act, uint2(wgt.z, wgt.w), uint4(0, 0, 0, 0));
        psum.data[0] = out1.x;
        psum.data[1] = out1.y;
        psum.data[2] = out1.z;
        psum.data[3] = out1.w;
        psum.data[4] = out2.x;
        psum.data[5] = out2.y;
        psum.data[6] = out2.z;
        psum.data[7] = out2.w;

        return psum;
    }

    // template<bool si>
    template <bool use_unsigned>
    __device__ __forceinline__ static void quantize_w4a4_from_fpsum_warp(
        const packed_fpsum_t (&fpsum)[INSN_K / INSN_N], packed_act_t& output, half_t* output_scale)
    {
        int const laneId = threadIdx.x % WARP_SIZE;

        constexpr float QVALUE_MAX_SIGNED = 7.0f;
        constexpr float QVALUE_MAX_UNSIGNED = 15.0f;
        constexpr float RECPI_QVALUE_MAX_SIGNED = 1 / QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX_UNSIGNED = 1 / QVALUE_MAX_UNSIGNED;

        constexpr float QVALUE_MAX = use_unsigned ? QVALUE_MAX_UNSIGNED : QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX = use_unsigned ? RECPI_QVALUE_MAX_UNSIGNED : RECPI_QVALUE_MAX_SIGNED;
        // constexpr int QUANTIZE_BITMASK = 0xf;

        // 0 for row 0-7; 1 for row 8-15
        half2_t input[2][INSN_K / INSN_N * 2];

#pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++)
        {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }

        half_t maxvalue[2];
        maxvalue[0] = 0;
        maxvalue[1] = 0;
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++)
        {
            half2_t abs0 = __habs2(input[0][i]);
            half2_t abs1 = __habs2(input[1][i]);
            maxvalue[0] = __hmax(maxvalue[0], __hmax(abs0.x, abs0.y));
            maxvalue[1] = __hmax(maxvalue[1], __hmax(abs1.x, abs1.y));
        }
#pragma unroll
        for (int mask = 2; mask > 0; mask /= 2)
        {
            maxvalue[0] = __hmax(maxvalue[0], __shfl_xor_sync(~0, maxvalue[0], mask));
            maxvalue[1] = __hmax(maxvalue[1], __shfl_xor_sync(~0, maxvalue[1], mask));
        }
        maxvalue[0] = __shfl_sync(~0, maxvalue[0], laneId / 4 * 4);
        maxvalue[1] = __shfl_sync(~0, maxvalue[1], laneId / 4 * 4);

        float scale[2];
        // scale[0] = float(maxvalue[0]) / QVALUE_MAX;
        // scale[1] = float(maxvalue[1]) / QVALUE_MAX;
        scale[0] = float(maxvalue[0]) * RECPI_QVALUE_MAX;
        scale[1] = float(maxvalue[1]) * RECPI_QVALUE_MAX;
        if (laneId % 4 == 0)
        {
            output_scale[laneId / 4] = half_t(scale[0]);
            output_scale[laneId / 4 + 8] = half_t(scale[1]);
        }

        float rscale[2];
        // rscale[0] = QVALUE_MAX / float(maxvalue[0]);
        // rscale[1] = QVALUE_MAX / float(maxvalue[1]);
        rscale[0] = cuda_frcp(scale[0]);
        rscale[1] = cuda_frcp(scale[1]);

        uint32_t qpacks[2][INSN_K / INSN_M * 2];
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 2; j++)
            {
                // half2_t hval = __hmul2(input[j][i], half2_t(rscale[j], rscale[j]));
                // float2 fval = half22float2(hval);
                float2 fval = half22float2(input[j][i]) * make_float2(rscale[j], rscale[j]);
                qpacks[j][i] = quantize_float2<4, use_unsigned>(fval) << (laneId % 4 * 8);
            }
        }

        // 2 * 8 * 2 = 32 instructions => 256 cycles
#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2)
        {
#pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++)
            {
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if (laneId % 4 == i)
            {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    /**
     * each warp quantizes a INSN_M * INSN_K (16 * 64) matrix
     * input is per-warp (in global memory)
     * output is per-thread (in regs)
     * output_scale is per-warp (in shared memory)
     * shmem must be at least INSN_M * INSN_K * sizeof(element) (16 * 64 * 0.5 = 512 Bytes)
     * default to quantize activation, if quantize weight, input should be column-majored and output should be
     * transposed ({x, y, z, w} = {x, z, y, w})
     */
    __device__ __forceinline__ static void quantize_w4a4_warp(
        half_t const* input, int stride, packed_act_t& output, half_t* output_scale, void* shmem)
    {
        int const laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 4;
        constexpr int QVALUE_MAX = 7; // 4 bit => [-8, 7]

        // 1 lane = 1 pack
        // 1 warp = 32 lanes = 32 packs = 1 packwarp
        // a pack is {a0, ..., a7} in figure
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ex2#mma-16864-a PACK_SIZE * 4 =
        // INSN_K / 2
        constexpr int PACK_SIZE = INSN_K / 8; // = 8 for 4bit
        constexpr int NUM_PACKS_PER_ROW = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input = std::array<half_t, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++)
        {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i] = load(reinterpret_cast<packed_input const*>(input + rowId * stride + colId));
        }

        // find max
        half_t maxvalue[NUM_PACKWARPS];
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++)
        {
            maxvalue[i] = __habs(packs[i][0]);
#pragma unroll
            for (int j = 1; j < PACK_SIZE; j++)
            {

                maxvalue[i] = __hmax(maxvalue[i], __habs(packs[i][j]));
            }
        }

        // warp reduce (max)
#pragma unroll
        for (int mask = NUM_PACKS_PER_ROW / 2; mask > 0; mask /= 2)
        {
#pragma unroll
            for (int i = 0; i < NUM_PACKWARPS; i++)
            {
                maxvalue[i] = __hmax(maxvalue[i], __shfl_xor_sync(~0, maxvalue[i], mask));
            }
        }

        // broadcast (max)
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++)
        {
            maxvalue[i] = __shfl_sync(~0, maxvalue[i], laneId / NUM_PACKS_PER_ROW * NUM_PACKS_PER_ROW);
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t& mat = *reinterpret_cast<matrix_t*>(shmem);
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++)
        {
            half_t scale = maxvalue[i] / half_t(QVALUE_MAX);
            half_t rscale = half_t(QVALUE_MAX) / maxvalue[i];
            if (laneId % NUM_PACKS_PER_ROW == 0)
            {
                output_scale[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW] = scale;
            }

            uint32_t qpack = 0;
// #pragma unroll
//         for (int j = 0; j < PACK_SIZE; j++) {
//             int intvalue = __half2int_rn(packs[i][j] / scale);
//             intvalue = clamp(intvalue, -QVALUE_MAX, QVALUE_MAX);
//             qpack |= (intvalue & QUANTIZE_BITMASK) << (QUANTIZE_BITWIDTH * j);
//         }
#pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2)
            {
                half2_t hval = __hmul2(half2_t(rscale, rscale), half2_t(packs[i][j], packs[i][j + 1]));
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(half22float2(hval)) << (j * QUANTIZE_BITWIDTH);
            }
            mat[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW][laneId % NUM_PACKS_PER_ROW] = qpack;
        }
        __syncwarp();

        // convert to imma format
        int row = laneId % 16;
        int col = laneId / 16 * 4;
        ldmatrix(&mat[row][col], output);

        __syncwarp();
    }

    // each thread block (1 warp) quantize WARP_M * WARP_K tile (32 * 64)
    struct quantize_w4a4_act_kernel
    {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;

        __device__ void operator()(half_t const* input, packed_act_t* output, packed_ascale_t* oscales, int K)
        {
            int const laneId = threadIdx.x % WARP_SIZE;

            int const bm = blockIdx.x / (BLOCK_M / WARP_M);
            int const bk = blockIdx.y;
            int const warpId = blockIdx.x % (BLOCK_M / WARP_M);

            int const row = blockIdx.x * WARP_M;
            int const col = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_M];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_M_TILES; tileId++)
            {
                packed_act_t tmpout;

                quantize_w4a4_warp(
                    input + (row + tileId * INSN_M) * K + col, K, tmpout, oscale_shmem + tileId * INSN_M, tmp_shmem);

                store(&output[(((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * WARP_M_TILES + tileId) * WARP_SIZE
                          + laneId],
                    tmpout);
            }

            // if (threadIdx.x == 0) {
            //     printf("Block (%d, %d) => offset = %d\n", blockIdx.x, blockIdx.y, (bm * K / WARP_K + bk) * NUM_WARPS
            //     + warpId);
            // }
            pack_ascales(oscale_shmem,
                &oscales[((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
        }
    };

    // each thread block (1 warp) quantize WARP_N * WARP_K tile (128 * 64)
    struct quantize_w4a4_wgt_kernel
    {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;

        __device__ void operator()(half_t const* input, packed_wgt_t* output, packed_wscale_t* oscales, int K)
        {
            int const laneId = threadIdx.x % WARP_SIZE;

            int const bn = blockIdx.x / (BLOCK_N / WARP_N);
            int const bk = blockIdx.y;

            int const col = blockIdx.x * WARP_N;
            int const row = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_N];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_N_TILES; tileId++)
            {
                packed_wgt_t tmpout;

                quantize_w4a4_warp(
                    input + (col + tileId * INSN_N) * K + row, K, tmpout, oscale_shmem + tileId * INSN_N, tmp_shmem);

                std::swap(tmpout.y, tmpout.z);

                store(&output[((bn * K / WARP_K + bk) * WARP_N_TILES + tileId) * WARP_SIZE + laneId], tmpout);
            }

            pack_wscales(oscale_shmem, &oscales[(bn * K / WARP_K + bk) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES]);
        }
    };

    struct i2f_sm80
    {
        __device__ __forceinline__ static float2 int2float2(int x, int y)
        {
            return make_float2(int2float_fast(x), int2float_fast(y));
        }

        __device__ __forceinline__ static half2_t int2half2(int x, int y)
        {
            return float22half2<half2_t>(int2float2(x, y));
        }
    };

    struct i2f_sm75
    {
        __device__ __forceinline__ static float2 int2float2(int x, int y)
        {
            return make_float2(int2float_fast(x), int2float_fast(y));
        }

        __device__ __forceinline__ static half2_t int2half2(int x, int y)
        {
            return half2(__int2half_rn(x), __int2half_rn(y));
        }
    };

    struct i2f_sm75_fast
    {
        __device__ __forceinline__ static float2 int2float2(int x, int y)
        {
            return make_float2(int2float_fast(x), int2float_fast(y));
        }

        __device__ __forceinline__ static half2_t int2half2(int x, int y)
        {
            return int2half2_fast_512(x, y);
        }
    };

    template <bool ACT_UNSIGNED, typename T>
    __device__ __forceinline__ static void compute(
        act_warp A, wgt_warp W, ascale_warp ascale, wscale_warp wscale, T& fpsum)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
        using int2half2 = i2f_sm80;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
        using int2half2 = std::conditional_t<Config::FASTER_I2F, i2f_sm75_fast, i2f_sm75>;
        ;
#else
    using int2half2 = Base::i2f_normal;
#endif

        Base::template apply_scales<int2half2>(
            [&](int i, int j) { return mma<ACT_UNSIGNED>(A[i], W[j]); }, ascale, wscale, fpsum);
    }

    __device__ __forceinline__ static void checkNan(fpsum_warp fpsum, char const* info = "")
    {
#if ENABLE_NAN_CHECK
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        for (int i = 0; i < fpsum.size(); i++)
        {
            for (int j = 0; j < 4; j++)
            {
                bool abnormal = !isfinite((float) fpsum[i].data[j].x) || !isfinite((float) fpsum[i].data[j].y);
                if (abnormal)
                {
                    printf(
                        "abnormal value detected at block.x=%d block.y=%d warpId=%d laneId=%d fpsum_warp (%s) i=%d "
                        "j=%d data.x=%f data.y=%f\n",
                        blockIdx.x, blockIdx.y, warpId, laneId, info, i, j, (float) fpsum[i].data[j].x,
                        (float) fpsum[i].data[j].y);
                    __trap();
                }
            }
        }
#endif
    }

    __device__ __forceinline__ static void checkNan(packed_f32psum_t fpsum, char const* info = "")
    {
#if ENABLE_NAN_CHECK
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 8; j++)
        {
            bool abnormal = !isfinite(fpsum.data[j]);
            if (abnormal)
            {
                printf(
                    "abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_f32psum_t (%s) j=%d data=%f\n",
                    blockIdx.x, blockIdx.y, warpId, laneId, info, j, fpsum.data[j]);
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__ static void checkNan(packed_fpsum_t fpsum, char const* info = "")
    {
#if ENABLE_NAN_CHECK
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 4; j++)
        {
            bool abnormal = !isfinite((float) fpsum.data[j].x) || !isfinite((float) fpsum.data[j].y);
            if (abnormal)
            {
                printf(
                    "abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) j=%d data.x=%f "
                    "data.y=%f\n",
                    blockIdx.x, blockIdx.y, warpId, laneId, info, j, (float) fpsum.data[j].x, (float) fpsum.data[j].y);
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__ static void checkNan(float data, char const* info = "")
    {
#if ENABLE_NAN_CHECK
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        bool abnormal = !isfinite(data);
        if (abnormal)
        {
            printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) data=%f\n",
                blockIdx.x, blockIdx.y, warpId, laneId, info, data);
            __trap();
        }
#endif
    }

    // out: [M / BLOCK_M, N / BLOCK_N, NUM_WARPS, 1, NUM_M_TILES, NUM_N_TILES, WARP_SIZE] of fpsum_warp
    template <typename Epilogue, bool ACT_UNSIGNED, bool USE_FP32_ACCUM>
    __device__ __forceinline__ static void gemm_w4a4_block(const BlockInfo binfo, packed_act_t const* act,
        packed_wgt_t const* wgt, packed_ascale_t const* ascales, packed_wscale_t const* wscales,
        // const packed_wscale_t *bias_ptr,
        // half_t *out,
        int M, int N, int K, Epilogue::Arguments const& epilogueArgs, bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

#if 0
        fpsum_warp fpsum;
        GEMM_W4A4_Block<Config>()(act, wgt, ascales, wscales, K, fpsum, alwaysfalse);
#else
        act_warp A[NUM_STAGES];                                             // 8
        wgt_warp W[NUM_STAGES];                                             // 32
        ascale_warp ascale[NUM_STAGES];                                     // 1
        wscale_warp wscale[NUM_STAGES];                                     // 2
        std::conditional_t<USE_FP32_ACCUM, f32psum_warp, fpsum_warp> fpsum; // 64

        // load_wscale<true>(wscales, wscale[0], true);
        // load_wscale<false>(wscales, wscale[1], true);
        // load_wscale<false>(wscales, wscale[2], true);

        for (int k = 0; k < NUM_STAGES - 1; k++)
        {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_ascale(ascales, k, M, ascale[k], true);
            load_wscale(wscales, k, N, wscale[k], true);
        }

        for (auto& pack : fpsum)
        {
            if constexpr (USE_FP32_ACCUM)
            {
                for (int i = 0; i < 8; i++)
                {
                    pack.data[i] = 0;
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    pack.data[i].x = 0;
                    pack.data[i].y = 0;
                }
            }
        }

        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES)
        {
#pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++)
            {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_ascale(ascales, nextk, M, ascale[idx], pred);
                load_wscale(wscales, nextk, N, wscale[idx], pred);
                // load_wscale<false>(wscales, wscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute<ACT_UNSIGNED>(A[k2], W[k2], ascale[k2], wscale[k2], fpsum);

                // #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                if (alwaysfalse)
                {
                    dummy = clock();
                }
                // #endif

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

#endif

        fpsum_warp f16psum;
        if constexpr (USE_FP32_ACCUM)
        {
            f16psum = packed_fp32_to_fp16(fpsum);
        }
        else
        {
            f16psum = fpsum;
        }

        CHECK_NAN(f16psum, "f16psum");

        Epilogue()(binfo, f16psum, M, N, K, epilogueArgs);
    }

    template <bool FUSE_GELU, bool USE_UNSIGNED, bool USE_FP4>
    struct EpilogueQuantize
    {
        using oscales_t = typename std::conditional_t<USE_FP4, packed_amscale_t, packed_ascale_t>;

        struct Arguments
        {
            packed_act_t* qout;
            oscales_t* oscales;

            half_t shift_value;
            packed_wscale_t const* smooth_factor;
        };

        static constexpr int NUM_PACKS = INSN_K / INSN_N;
        static constexpr int NUM_GROUPS = WARP_N_TILES / NUM_PACKS;

        __device__ __forceinline__ void apply_quantize(fpsum_warp fpsum, int M, int N, int K, packed_act_t* qout,
            oscales_t* oscales, half_t shift_value, packed_wscale_t const* smooth_factor)
        {
            int const laneId = threadIdx.x % WARP_SIZE;
            int const warpId = threadIdx.x / WARP_SIZE;

            __shared__ half_t oscale_shmem[NUM_WARPS][WARP_M];

            wscale_warp smooth;
            load_wscale(smooth_factor, 0, N, smooth, true);

#pragma unroll
            for (int group = 0; group < NUM_GROUPS; group++)
            {
                amscale_warp omscale;

#pragma unroll
                for (int i = 0; i < WARP_M_TILES; i++)
                {
                    packed_fpsum_t tmp[NUM_PACKS];

#pragma unroll
                    for (int j = 0; j < NUM_PACKS; j++)
                    {
                        half2_t ws1 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4, laneId);
                        half2_t ws2 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4 + 2, laneId);
#pragma unroll
                        for (int k = 0; k < 4; k++)
                        {
                            half2_t src = fpsum[i * WARP_N_TILES + group * NUM_PACKS + j].data[k];
                            half2_t& dst = tmp[j].data[k];

                            // dst.x = gelu(src.x);
                            // dst.y = gelu(src.y);
                            if constexpr (FUSE_GELU)
                            {
                                dst = gelu_half2(src);
                            }
                            else
                            {
                                dst = src;
                            }

                            dst += half2_t(shift_value, shift_value);
                            // dst = src;
                        }

                        tmp[j].data[0] = h2div(tmp[j].data[0], ws1);
                        tmp[j].data[1] = h2div(tmp[j].data[1], ws1);
                        tmp[j].data[2] = h2div(tmp[j].data[2], ws2);
                        tmp[j].data[3] = h2div(tmp[j].data[3], ws2);
                    }

                    packed_act_t qresult;
                    if constexpr (USE_FP4)
                    {
                        quantize_w4a4_fp4_from_fpsum_warp(
                            tmp, qresult, omscale[i / 2 / AMSCALES_PACK_SIZE].data[i / 2 % AMSCALES_PACK_SIZE], i % 2);
                    }
                    else
                    {
                        quantize_w4a4_from_fpsum_warp<USE_UNSIGNED>(tmp, qresult, &oscale_shmem[warpId][i * INSN_M]);
                    }
                    store(&qout[((group * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId], qresult);
                }

                if constexpr (USE_FP4)
                {
#pragma unroll
                    for (int k = 0; k < AMSCALES_NUM_PACKS; k++)
                    {
                        store(&oscales[((group * NUM_WARPS + warpId) * AMSCALES_NUM_PACKS + k) * AMSCALES_VALID_LANES
                                  + laneId],
                            omscale[k]);
                    }
                }
                if constexpr (!USE_FP4)
                {
                    __syncwarp();
                    pack_ascales(&oscale_shmem[warpId][0],
                        &oscales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
                    __syncwarp();
                }
            }
        }

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments const& args)
        {
            int const bm = binfo.bm;
            int const bn = binfo.bn;

            if constexpr (!USE_FP4 || FP4_AVAILABLE)
            {
                apply_quantize(fpsum, M, N, K,
                    args.qout + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                    args.oscales
                        + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS
                            * (USE_FP4 ? AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES
                                       : ASCALES_NUM_PACKS * ASCALES_VALID_LANES),
                    args.shift_value, args.smooth_factor + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES);
            }
            else
            {
                trap_no_fp4();
            }
        }
    };

    // using EpilogueQuantizeFuseGelu = EpilogueQuantize<true>;

    template <typename Epilogue, bool ACT_UNSIGNED>
    struct gemm_w4a4_kernel
    {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr int MAX_ARCH = Config::FASTER_I2F ? 750 : INT_MAX; // FASTER_I2F is only needed on sm_75

        __device__ void operator()(packed_act_t const* act, packed_wgt_t const* wgt, packed_ascale_t const* ascales,
            packed_wscale_t const* wscales, int M, int N, int K, Epilogue::Arguments epilogueArgs, bool swapBlockXY,
            bool alwaysfalse)
        {
            // printf("Device sizeof(args) = %d", (int)sizeof(epilogueArgs));

            BlockInfo binfo = {
                .bm = (int) blockIdx.x,
                .bn = (int) blockIdx.y,
                .numBlocksM = (int) gridDim.x,
                .numBlocksN = (int) gridDim.y,
            };

            if (swapBlockXY)
            {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            int const bm = binfo.bm;
            int const bn = binfo.bn;

            // bool fusequant = !out;

            gemm_w4a4_block<Epilogue, ACT_UNSIGNED, false>(binfo,
                act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                ascales + bm * (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                // bias ? bias + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES : nullptr,
                // out + (bm * BLOCK_M * N) + bn * BLOCK_N,
                // out + (bm * N / BLOCK_N + bn) * NUM_WARPS * WARP_M_TILES * WARP_N_TILES * WARP_SIZE,
                M, N, K, epilogueArgs, alwaysfalse);
        }
    };

    template <bool fuse_glu, bool use_fp4>
    struct quantize_w4a4_fuse_lora_kernel
    {
        using oscales_t = typename std::conditional_t<use_fp4, packed_amscale_t, packed_ascale_t>;

        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr size_t SHMEM_PER_WARP
            = ceilDiv<size_t>(Base::template load_act_to_fpsum<fuse_glu>::SHMEM_SIZE, 128) * 128;
        static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * NUM_WARPS;

        struct Arguments
        {
            half_t const* input;
            packed_wscale_t const* smooth_factor;
            packed_act_t* output;
            oscales_t* oscales;
            packed_fpsum_t const* lora_wgt_down;
            float* lora_act;

            int lora_rank;

            // aligned to BLOCK_M and BLOCK_N
            int M, N; // N should be the actual K in the next GEMM (needs /2 if fuse_glu)
            // the actual M and N   (no need to /2 if fuse_glu)
            int actualM, actualN;

            bool alwaysfalse;
        };

        __device__ __forceinline__ void operator()(Arguments args)
        {
            const BlockInfo binfo = {
                .bm = (int) blockIdx.x,
                .bn = (int) blockIdx.y,
                .numBlocksM = (int) gridDim.x,
                .numBlocksN = (int) gridDim.y,
            };

            int const bm = binfo.bm;
            int const bn = binfo.bn;
            int const warpId = threadIdx.x / WARP_SIZE;

            int const m_offset = bm * BLOCK_M + warpId * WARP_M;
            int const n_offset = bn * BLOCK_N * (fuse_glu ? 2 : 1);

            extern __shared__ uint8_t shmem[];

            fpsum_warp fpsum;

            Base::template load_act_to_fpsum<fuse_glu>()(args.input + m_offset * args.actualN + n_offset, args.actualN,
                args.actualM - m_offset, args.actualN - n_offset, fpsum, shmem + warpId * SHMEM_PER_WARP
                // args.smooth_factor ? args.smooth_factor + n_offset : nullptr
            );

            CHECK_NAN(fpsum, "fpsum");
            // for (int i = 0; i < 16; i++) {
            //     printf("bm=%d bn=%d warp=%d lane=%d fpsum[%d][0:1]=%f %f\n",
            //         bm, bn, warpId, threadIdx.x % WARP_SIZE, i,
            //         (float)fpsum[i].data[0].x, (float)fpsum[i].data[0].y);
            // }

            using EpilogueLoraDown = typename Lora<Config>::EpilogueLoraDown;

            EpilogueLoraDown()(binfo, fpsum, args.M, args.N, 0,
                typename EpilogueLoraDown::Arguments{
                    .lora_wgt_down = args.lora_wgt_down,
                    .lora_act = args.lora_act,
                    .rank = args.lora_rank,
                    .alwaysfalse = args.alwaysfalse,
                });

            EpilogueQuantize<false, false, use_fp4>()(binfo, fpsum, args.M, args.N, 0,
                typename EpilogueQuantize<false, false, use_fp4>::Arguments{.qout = args.output,
                    .oscales = args.oscales,
                    .shift_value = 0,
                    .smooth_factor = args.smooth_factor});
        }
    };
};

}; // namespace nunchaku::kernels
