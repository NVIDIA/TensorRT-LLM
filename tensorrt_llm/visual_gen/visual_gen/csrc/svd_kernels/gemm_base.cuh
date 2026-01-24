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

#include "common.h"

#include "dispatch_utils.h"
#include "gemm_utils.cuh"
#include "mma_earlycuda.cuh"
#include "utils.cuh"

#pragma nv_diag_suppress 177

#ifdef _MSC_VER
#define ALWAYSINLINE [[msvc::forceinline]]
#else
#define ALWAYSINLINE __attribute__((always_inline))
#endif

// #define ENABLE_NAN_CHECK 1
#if ENABLE_NAN_CHECK
#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define CHECK_NAN(data, name) checkNan(data, name " at " STRINGIZE(__LINE__))
#else
#define CHECK_NAN(data, name)
#endif

namespace nunchaku::kernels
{

template <bool bf16, bool faster_i2f = false>
class GEMMConfig_W4A4
{
public:
    // BE CAREFUL: weights need to be repacked when the tiling size changes

    static constexpr int BLOCK_M = 256;
    static constexpr int BLOCK_N = 128;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int INSN_M = 16;
    static constexpr int INSN_N = 16;
    static constexpr int INSN_K = 64;

    // faster i2f conversion on sm_75
    // may generate incorrect results in certain circumstances
    static constexpr bool FASTER_I2F = faster_i2f;

    using half_t = typename std::conditional_t<bf16, __nv_bfloat16, half>;
    using half2_t = typename std::conditional_t<bf16, __nv_bfloat162, half2>;
};

using GEMMConfig_W4A4_FP16 = GEMMConfig_W4A4<false>;
using GEMMConfig_W4A4_BF16 = GEMMConfig_W4A4<true>;
using GEMMConfig_W4A4_FP16_FasterI2F = GEMMConfig_W4A4<false, true>;

class GEMMConfig_W8A8
{
public:
    static constexpr int BLOCK_M = 256;
    static constexpr int BLOCK_N = 128;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int INSN_M = 16;
    static constexpr int INSN_N = 16;
    static constexpr int INSN_K = 32;

#if 0
    using half_t  = half;
    using half2_t = half2;
#else
    using half_t = __nv_bfloat16;
    using half2_t = __nv_bfloat162;
#endif
};

template <class Config>
class GEMMBase : public Config
{
public:
    using Config::BLOCK_M;
    using Config::BLOCK_N;
    using Config::WARP_SIZE;
    using Config::NUM_WARPS;
    using Config::INSN_M;
    using Config::INSN_N;
    using Config::INSN_K;

    using typename Config::half_t;
    using typename Config::half2_t;

    static constexpr int WARP_M = BLOCK_M / NUM_WARPS;
    static constexpr int WARP_N = BLOCK_N;
    static constexpr int WARP_K = INSN_K;

    static constexpr int WARP_M_TILES = WARP_M / INSN_M;
    static constexpr int WARP_N_TILES = WARP_N / INSN_N;
    static constexpr int WARP_K_TILES = WARP_K / INSN_K;

    /**
     * refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16864-c
     *
     * wscales store order: (pack = 4)
     *  0   1   8   9   <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  2   3   10  11  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
     *  4   5   12  13  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
     *  6   7   14  15  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *
     *  16  17  24  25  <-- load by lane 4, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  22  23  30  31  <-- load by lane 7, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *  ... ...
     *  112 113 120 121 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  118 119 126 127 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *
     * wscales store order: (pack = 8)
     *  0   1   8   9   16  17  24  25  <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  2   3   10  11  18  19  26  27  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
     *  4   5   12  13  20  21  28  29  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
     *  6   7   14  15  22  23  30  31  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *
     *  224 225 232 233 240 241 248 249 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  230 231 238 239 246 247 254 255 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *
     * {k}-th wscale used by lane {i} => {k // (WSCALES_PACK_SIZE * WARP_SIZE)}-th pack, in lane {4*(k //
     * WSCALES_PACK_SIZE) + i % 4}, element {k % WSCALES_PACK_SIZE}
     *
     * max pack size set to 8 since max load size is 16 bytes / lane
     * min pack size set to 2 since shuffle granularity is 32b 2*half
     * */
    static constexpr int WSCALES_PACK_SIZE = clamp(WARP_N / WARP_SIZE, 4 / sizeof(half), 16 / sizeof(half));
    static constexpr int WSCALES_NUM_PACKS = ceilDiv(WARP_N, (WSCALES_PACK_SIZE * WARP_SIZE));
    static constexpr int WSCALES_VALID_LANES = std::min(WARP_SIZE, WARP_N / WSCALES_PACK_SIZE);

    /**
     * ascales store order: (pack = 2)
     *  0   8   <-- load by lane 0, broadcast to lane {0, 1, 2, 3} (4x)
     *  1   9   <-- load by lane 1, broadcast to lane {4, 5, 6, 7} (4x)
     *  2   10
     *  ...
     *  6   14
     *  7   15  <-- load by lane 7, broadcast to lane {28, 29, 30, 31} (4x)
     *  ... ...
     *  48  56  <-- load by lane 24, broadcast to lane {0, 1, 2, 3} (4x)
     *  49  57
     *  ...
     *  54  62
     *  55  63  <-- load by lane 31, broadcast to lane {28, 29, 30, 31}  (4x)
     *
     * {k}-th wscale used by lane {i} => {k // (ASCALES_PACK_SIZE * WARP_SIZE)}-th pack, in lane {8*(k //
     * ASCALES_PACK_SIZE) + i // 4}, element {k % ASCALES_PACK_SIZE}
     */
    static constexpr int ASCALES_PACK_SIZE = clamp(WARP_M / WARP_SIZE, 4 / sizeof(half), 16 / sizeof(half));
    static constexpr int ASCALES_NUM_PACKS = ceilDiv(WARP_M, (ASCALES_PACK_SIZE * WARP_SIZE));
    static constexpr int ASCALES_VALID_LANES = std::min(WARP_SIZE, WARP_M / ASCALES_PACK_SIZE);

    using packed_act_t = uint4;
    using packed_wgt_t = uint4;

    struct alignas(32) packed_psum_t
    {
        int data[8];
    };

    struct alignas(16) packed_fpsum_t
    {
        half2_t data[4];
    };

    struct alignas(8) packed_gated_fpsum_t
    {
        half_t data[4];
    };

    // 16 * 16 matrix
    struct alignas(32) packed_f32psum_t
    {
        float data[8];

        static constexpr packed_f32psum_t zeros()
        {
            packed_f32psum_t result;
            for (int i = 0; i < 8; i++)
            {
                result.data[i] = 0;
            }
            return result;
        };
    };

    struct packed_wscale_t
    {
        half2_t data[WSCALES_PACK_SIZE / 2];
    };

    struct packed_ascale_t
    {
        half2_t data[ASCALES_PACK_SIZE / 2];
    };

    using act_warp = std::array<packed_act_t, WARP_M_TILES>;
    using wgt_warp = std::array<packed_wgt_t, WARP_N_TILES>;
    using ascale_warp = std::array<packed_ascale_t, ASCALES_NUM_PACKS>;
    using wscale_warp = std::array<packed_wscale_t, WSCALES_NUM_PACKS>;
    using fpsum_warp = std::array<packed_fpsum_t, WARP_M_TILES * WARP_N_TILES>;
    using f32psum_warp = std::array<packed_f32psum_t, WARP_M_TILES * WARP_N_TILES>;
    using gated_fpsum_warp = std::array<packed_gated_fpsum_t, WARP_M_TILES * WARP_N_TILES>;

    struct BlockInfo
    {
        int bm;
        int bn;
        int numBlocksM;
        int numBlocksN;
    };

    __device__ __forceinline__ static packed_f32psum_t mma_f16xf16_f32(
        packed_fpsum_t a, packed_fpsum_t b, packed_f32psum_t psum)
    {
        static_assert(std::is_same_v<half_t, half> || std::is_same_v<half_t, __nv_bfloat16>);

        static constexpr bool is_bf16 = std::is_same_v<half_t, __nv_bfloat16>;

        uint4 out1 = mma_m16n8k16_f32f16f16f32<is_bf16>(kernels::bit_cast<uint4>(a),
            kernels::bit_cast<uint2>(std::array<half2_t, 2>(b.data[0], b.data[1])),
            kernels::bit_cast<uint4>(float4(psum.data[0], psum.data[1], psum.data[2], psum.data[3])));
        uint4 out2 = mma_m16n8k16_f32f16f16f32<is_bf16>(kernels::bit_cast<uint4>(a),
            kernels::bit_cast<uint2>(std::array<half2_t, 2>(b.data[2], b.data[3])),
            kernels::bit_cast<uint4>(float4(psum.data[4], psum.data[5], psum.data[6], psum.data[7])));
        psum.data[0] = kernels::bit_cast<float>(out1.x);
        psum.data[1] = kernels::bit_cast<float>(out1.y);
        psum.data[2] = kernels::bit_cast<float>(out1.z);
        psum.data[3] = kernels::bit_cast<float>(out1.w);
        psum.data[4] = kernels::bit_cast<float>(out2.x);
        psum.data[5] = kernels::bit_cast<float>(out2.y);
        psum.data[6] = kernels::bit_cast<float>(out2.z);
        psum.data[7] = kernels::bit_cast<float>(out2.w);

        return psum;
    }

    __device__ __forceinline__ static packed_fpsum_t packed_fp32_to_fp16(packed_f32psum_t input)
    {
        packed_fpsum_t results;
        for (int i = 0; i < 4; i++)
        {
            results.data[i] = float22half2<half2_t>(float2(input.data[i * 2], input.data[i * 2 + 1]));
        }
        return results;
    }

    __device__ __forceinline__ static packed_f32psum_t packed_fp16_to_fp32(packed_fpsum_t input)
    {
        packed_f32psum_t results;
        for (int i = 0; i < 4; i++)
        {
            float2 tmp = half22float2(input.data[i]);
            results.data[i * 2] = tmp.x;
            results.data[i * 2 + 1] = tmp.y;
        }
        return results;
    }

    __device__ __forceinline__ static fpsum_warp packed_fp32_to_fp16(f32psum_warp input)
    {
        fpsum_warp results;
#pragma unroll
        for (int i = 0; i < results.size(); i++)
        {
            results[i] = packed_fp32_to_fp16(input[i]);
        }
        return results;
    }

    __device__ __forceinline__ static f32psum_warp packed_fp16_to_fp32(fpsum_warp input)
    {
        f32psum_warp results;
#pragma unroll
        for (int i = 0; i < results.size(); i++)
        {
            results[i] = packed_fp16_to_fp32(input[i]);
        }
        return results;
    }

    // activation: row major, [M / BLOCK_M, K / WARP_K, NUM_WARPS, WARP_M_TILES, WARP_SIZE] of packed_act_t
    __device__ __forceinline__ static void load_act(packed_act_t const* act, int k, int K, act_warp& out, bool pred)
    {
        int laneId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
#pragma unroll
        for (int i = 0; i < WARP_M_TILES; i++)
        {
            // if (pred) {
            //  out[i] = load(&act[((warpId * WARP_M_TILES + i) * K / WARP_K + k) * WARP_SIZE + laneId]);
            out[i] = load_pred(&act[((k * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId], pred);
            //}
        }
    }

    // weight: column major: [N / BLOCK_N, 1, K / WARP_K, WARP_N_TILES, WARP_SIZE] of packed_wgt_t
    __device__ __forceinline__ static void load_wgt(packed_wgt_t const* wgt, int k, int K, wgt_warp& out, bool pred)
    {
        int laneId = threadIdx.x % WARP_SIZE;

        // const packed_wgt_t *ptr = &wgt[(0 * K / WARP_K + k) * WARP_SIZE + laneId];
        packed_wgt_t const* ptr = &wgt[(0 + k * WARP_N_TILES) * WARP_SIZE + laneId];
        // int offset = K / WARP_K * WARP_SIZE;
#pragma unroll
        for (int i = 0; i < WARP_N_TILES; i++)
        {
            // if (pred) {
            //  out[i] = load(&wgt[(i * K / WARP_K + k) * WARP_SIZE + laneId]);
            //  out[i] = load(&wgt[(i + k * WARP_N_TILES) * WARP_SIZE + laneId]);
            out[i] = load_pred(&ptr[i * WARP_SIZE], pred);
            // ptr += offset;
            //}
        }
    }

    // ascales: row major [M / BLOCK_M, K / group size, NUM_WARPS, ASCALES_NUM_PACKS, ASCALES_VALID_LANES] of
    // packed_ascale_t
    __device__ __forceinline__ static void load_ascale(
        packed_ascale_t const* ascales, int group, int M, ascale_warp& out, bool pred)
    {
        int laneId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
#pragma unroll
        for (int i = 0; i < ASCALES_NUM_PACKS; i++)
        {
            // if (pred && laneId < ASCALES_VALID_LANES) {
            // out[i] = ascales[(group * M / WARP_M + warpId) * ASCALES_VALID_LANES * ASCALES_NUM_PACKS + i *
            // ASCALES_VALID_LANES + laneId];
            out[i] = load_pred(&ascales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES
                                   + i * ASCALES_VALID_LANES + laneId],
                pred && laneId < ASCALES_VALID_LANES);

            // }
        }
    }

    // wscales: column major [N / BLOCK_N, K / group size, 1, WSCALES_NUM_PACKS, WSCALES_VALID_LANES] of packed_wscale_t
    // </del>
    __device__ __forceinline__ static void load_wscale(
        packed_wscale_t const* wscales, int group, int N, wscale_warp& out, bool pred)
    {
        int laneId = threadIdx.x % WARP_SIZE;

        // static_assert(WSCALES_NUM_PACKS * WSCALES_VALID_LANES == 32);
        // static_assert(sizeof(packed_wscale_t) == 8);

        // const packed_wscale_t *ptr = &wscales[(group * WSCALES_NUM_PACKS + 0) * WSCALES_VALID_LANES + laneId];
        // // const packed_wscale_t *ptr = (const packed_wscale_t *)((const char *)wscales) + ((group *
        // WSCALES_NUM_PACKS + 0) * WSCALES_VALID_LANES + laneId) * sizeof(packed_wscale_t);

#pragma unroll
        for (int i = 0; i < WSCALES_NUM_PACKS; i++)
        {
            // if (pred && laneId < WSCALES_VALID_LANES) {

            // out[i] = wscales[group * N / WARP_N * WSCALES_VALID_LANES * WSCALES_NUM_PACKS + i * WSCALES_VALID_LANES +
            // laneId]; out[i] = load(&wscales[group * N / WARP_N * WSCALES_VALID_LANES * WSCALES_NUM_PACKS + i *
            // WSCALES_VALID_LANES + laneId]);
            out[i] = load_pred(&wscales[(group * WSCALES_NUM_PACKS + i) * WSCALES_VALID_LANES + laneId],
                pred && laneId < WSCALES_VALID_LANES);
            // out[i] = load(&ptr[i * WSCALES_VALID_LANES]);
            // }
        }
    }

    // get {k}-th and {k+1}-th wscale from the block, k must be multiples of 2, k must be uniform across all lanes
    __device__ __forceinline__ static half2_t broadcast_wscale(wscale_warp block, int k, int laneId)
    {
        int const packIdx = k / (WSCALES_PACK_SIZE * WARP_SIZE);
        int const srcLane = 4 * (k / WSCALES_PACK_SIZE) + laneId % 4;
        int const elementIdx = k % WSCALES_PACK_SIZE / 2;
        return __shfl_sync(~0, block[packIdx].data[elementIdx], srcLane);
    }

    // get {k}-th and {k+1}-th ascale from the block, k must be multiples of 2, k must be uniform across all lanes
    __device__ __forceinline__ static half2_t broadcast_ascale(ascale_warp block, int k, int laneId)
    {
        int const packIdx = k / (ASCALES_PACK_SIZE * WARP_SIZE);
        int const srcLane = 8 * (k / ASCALES_PACK_SIZE) + laneId / 4;
        int const elementIdx = k % ASCALES_PACK_SIZE / 2;
        return __shfl_sync(~0, block[packIdx].data[elementIdx], srcLane);
    }

    struct i2f_normal
    {
        __device__ __forceinline__ static float2 int2float2(int x, int y)
        {
            return make_float2(__int2float_rn(x), __int2float_rn(y));
        }

        __device__ __forceinline__ static half2_t int2half2(int x, int y)
        {
            return float22half2<half2_t>(int2float2(x, y));
        }
    };

    template <typename i2f = i2f_normal, typename F>
    __device__ __forceinline__ static void apply_scales(
        F&& getpsum, ascale_warp ascale, wscale_warp wscale, fpsum_warp& fpsum)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        half2_t asx[WARP_M_TILES];
        half2_t asy[WARP_M_TILES];

        for (int i = 0; i < WARP_M_TILES; i++)
        {
            half2_t as = broadcast_ascale(ascale, i * 2, laneId);
            asx[i] = half2_t(as.x, as.x);
            asy[i] = half2_t(as.y, as.y);
        }

        for (int j = 0; j < WARP_N_TILES; j++)
        {
            half2_t ws1 = broadcast_wscale(wscale, j * 4, laneId);
            half2_t ws2 = broadcast_wscale(wscale, j * 4 + 2, laneId);

            for (int i = 0; i < WARP_M_TILES; i++)
            {
                auto& fsum = fpsum[i * WARP_N_TILES + j];

                packed_psum_t psum = getpsum(i, j);

                // constexpr int target = 0;
                // if (threadIdx.x == 3 && j == 1 && i == 0) {

                //     printf("before ws2 = %f %f fsum.data[%d] = %f %f\n", (float)ws2.x, (float)ws2.y, target,
                //     (float)fsum.data[target].x, (float)fsum.data[target].y);
                // }

                fsum.data[0] = __hfma2(i2f::int2half2(psum.data[0], psum.data[1]), __hmul2(asx[i], ws1), fsum.data[0]);
                fsum.data[1] = __hfma2(i2f::int2half2(psum.data[2], psum.data[3]), __hmul2(asy[i], ws1), fsum.data[1]);
                fsum.data[2] = __hfma2(i2f::int2half2(psum.data[4], psum.data[5]), __hmul2(asx[i], ws2), fsum.data[2]);
                fsum.data[3] = __hfma2(i2f::int2half2(psum.data[6], psum.data[7]), __hmul2(asy[i], ws2), fsum.data[3]);

                // if (threadIdx.x == 3 && j == 1 && i == 0) {
                //     printf("before ws2 = %f %f fsum.data[%d] = %f %f\n", (float)ws2.x, (float)ws2.y, target,
                //     (float)fsum.data[target].x, (float)fsum.data[target].y);
                // }
            }
        }
    }

    template <typename i2f = i2f_normal, typename F>
    __device__ __forceinline__ static void apply_scales(
        F&& getpsum, ascale_warp ascale, wscale_warp wscale, f32psum_warp& fpsum)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        float2 asx[WARP_M_TILES];
        float2 asy[WARP_M_TILES];

        for (int i = 0; i < WARP_M_TILES; i++)
        {
            half2_t as = broadcast_ascale(ascale, i * 2, laneId);
            asx[i] = half22float2(half2_t(as.x, as.x));
            asy[i] = half22float2(half2_t(as.y, as.y));
        }

        auto fma2 = [](float2 a, float2 b, float& cx, float& cy) ALWAYSINLINE
        {
            cx += a.x * b.x;
            cy += a.y * b.y;
        };

        for (int j = 0; j < WARP_N_TILES; j++)
        {
            float2 ws1 = half22float2(broadcast_wscale(wscale, j * 4, laneId));
            float2 ws2 = half22float2(broadcast_wscale(wscale, j * 4 + 2, laneId));

            for (int i = 0; i < WARP_M_TILES; i++)
            {
                auto& fsum = fpsum[i * WARP_N_TILES + j];

                packed_psum_t psum = getpsum(i, j);

                fma2(i2f::int2float2(psum.data[0], psum.data[1]), asx[i] * ws1, fsum.data[0], fsum.data[1]);
                fma2(i2f::int2float2(psum.data[2], psum.data[3]), asy[i] * ws1, fsum.data[2], fsum.data[3]);
                fma2(i2f::int2float2(psum.data[4], psum.data[5]), asx[i] * ws2, fsum.data[4], fsum.data[5]);
                fma2(i2f::int2float2(psum.data[6], psum.data[7]), asy[i] * ws2, fsum.data[6], fsum.data[7]);
            }
        }
    }

    /**
     * input: WARP_M of half (in shared memory, per-warp)
     * output: [..., ASCALES_NUM_PACKS, ASCALES_VALID_LANES] in global memory, per-warp
     */
    __device__ __forceinline__ static void pack_ascales(half_t const* input, packed_ascale_t* output)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
#pragma unroll
        for (int j = 0; j < ASCALES_NUM_PACKS; j++)
        {
            if (laneId < ASCALES_VALID_LANES)
            {
                packed_ascale_t tmp;
#pragma unroll
                for (int i = 0; i < ASCALES_PACK_SIZE; i += 2)
                {
                    tmp.data[i / 2].x = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE
                        + laneId % 8 + i * 8];
                    tmp.data[i / 2].y = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE
                        + laneId % 8 + (i + 1) * 8];
                }
                output[j * ASCALES_VALID_LANES + laneId] = tmp;
            }
        }
    }

    /**
     * input: WARP_N of half (in shared memory, per-warp)
     * output: [..., WSCALES_NUM_PACKS, WSCALES_VALID_LANES] in global memory, per-warp
     */
    __device__ __forceinline__ static void pack_wscales(half_t const* input, packed_wscale_t* output)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
#pragma unroll
        for (int j = 0; j < WSCALES_NUM_PACKS; j++)
        {
            if (laneId < WSCALES_VALID_LANES)
            {
                packed_wscale_t tmp;
#pragma unroll
                for (int i = 0; i < WSCALES_PACK_SIZE; i += 2)
                {
                    tmp.data[i / 2] = *reinterpret_cast<half2_t const*>(&input[j * WSCALES_PACK_SIZE * WARP_SIZE
                        + laneId / 4 * 4 * WSCALES_PACK_SIZE + laneId % 4 * 2 + i * 4]);
                }
                store(&output[j * WSCALES_VALID_LANES + laneId], tmp);
            }
        }
    }

    struct unpack_fpsum
    {
        // +8 to prevent bank conflicts
        using matrix_t = half_t[8][WARP_N + 8];

        static constexpr int SHMEM_SIZE = sizeof(matrix_t);
        static constexpr int PACK_SIZE = WARP_N / WARP_SIZE;
        using pack_t = std::array<half_t, PACK_SIZE>;

        // F (int rowId, pack_t &pack)
        template <typename... F>
        __device__ __forceinline__ void operator()(
            fpsum_warp fpsum, half_t* output, int stride, int maxRows, int maxCols, void* shmem, F&&... plugins)
        {
            int const laneId = threadIdx.x % WARP_SIZE;

            matrix_t& mat = *reinterpret_cast<matrix_t*>(shmem);

            // pack_t reduce_tmp;
            // constexpr bool enableReduce = !std::is_void_v<FuncReduce>;

            // if constexpr (enableReduce) {
            //     reduce_tmp.fill(reduce_initval);
            //     // reduce_tmp = load<true>(reinterpret_cast<pack_t *>(&reduce_result[laneId * PACK_SIZE]));
            // }
            // auto doReduce = [&reduce_tmp](pack_t pack) {
            //     if constexpr (enableReduce) {
            //         for (int i = 0; i < PACK_SIZE; i++) {
            //             reduce_tmp[i] = FuncReduce()(reduce_tmp[i], pack[i]);
            //         }
            //     }
            // };

#pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++)
            {
#pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++)
                {
                    packed_fpsum_t& fsum = fpsum[i * WARP_N_TILES + j];
                    int row = laneId / 4;
                    int col = laneId % 4 * 2 + j * INSN_N;
                    *reinterpret_cast<half2_t*>(&mat[row][col + 0]) = fsum.data[0];
                    *reinterpret_cast<half2_t*>(&mat[row][col + 8]) = fsum.data[2];
                }
                __syncwarp();

#pragma unroll
                for (int row = 0; row < 8; row++)
                {
                    pack_t pack = *reinterpret_cast<pack_t*>(&mat[row][laneId * PACK_SIZE]);

                    // if constexpr (enableReduce) {
                    //     doReduce(pack);
                    // }

                    (plugins(i * INSN_M + row, pack), ...);

                    bool pred = i * INSN_M + row < maxRows && laneId * PACK_SIZE < maxCols;
                    // if (pred) {
                    store_pred(reinterpret_cast<pack_t*>(&output[(i * INSN_M + row) * stride + laneId * PACK_SIZE]),
                        pack, pred);
                    // }
                }
                __syncwarp();

#pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++)
                {
                    packed_fpsum_t& fsum = fpsum[i * WARP_N_TILES + j];
                    int row = laneId / 4;
                    int col = laneId % 4 * 2 + j * INSN_N;
                    *reinterpret_cast<half2_t*>(&mat[row][col + 0]) = fsum.data[1];
                    *reinterpret_cast<half2_t*>(&mat[row][col + 8]) = fsum.data[3];
                }
                __syncwarp();

#pragma unroll
                for (int row = 0; row < 8; row++)
                {
                    pack_t pack = *reinterpret_cast<pack_t*>(&mat[row][laneId * PACK_SIZE]);

                    // if constexpr (enableReduce) {
                    //     doReduce(pack);
                    // }

                    (plugins(i * INSN_M + 8 + row, pack), ...);

                    bool pred = i * INSN_M + 8 + row < maxRows && laneId * PACK_SIZE < maxCols;
                    // if (pred) {
                    store_pred(reinterpret_cast<pack_t*>(&output[(i * INSN_M + 8 + row) * stride + laneId * PACK_SIZE]),
                        pack, pred);
                    // }
                }
                __syncwarp();
            }
            // if (enableReduce) {
            //     store<true>(reinterpret_cast<pack_t *>(&reduce_result[laneId * PACK_SIZE]), reduce_tmp);
            // }
        }
    };

    // loads act of [WARP_M, WARP_N] and stores to fpsum_warp
    // [WARP_M, WARP_N * 2] when fuse_glu
    template <bool fuse_glu>
    struct load_act_to_fpsum
    {
        using matrix_t = half_t[INSN_M][WARP_N + 8];
        static constexpr size_t SHMEM_SIZE = sizeof(matrix_t);

        __device__ __forceinline__ void operator()(
            half_t const* input, int stride, int maxRows, int maxCols, fpsum_warp& out, void* shmem)
        {
            int const laneId = threadIdx.x % WARP_SIZE;

            matrix_t& mat = *reinterpret_cast<matrix_t*>(shmem);

            constexpr int PACK_SIZE = WARP_N / WARP_SIZE;
            using packed_input = std::array<half_t, PACK_SIZE>;
            using packed_raw_input = std::array<half2_t, PACK_SIZE>;

#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++)
            {
#pragma unroll
                for (int row = 0; row < INSN_M; row++)
                {
                    packed_input pack;
                    // TODO: numCols not multiples of PACK_SIZE
                    if constexpr (fuse_glu)
                    {
                        packed_raw_input raw;
                        raw.fill(half2_t(0, 0));
                        bool pred = (m * INSN_M + row) < maxRows && laneId * PACK_SIZE * 2 < maxCols;
                        if (pred)
                        {
                            raw = load(reinterpret_cast<packed_raw_input const*>(
                                input + (m * INSN_M + row) * stride + laneId * PACK_SIZE * 2));
                        }
#pragma unroll
                        for (int j = 0; j < PACK_SIZE; j++)
                        {
                            pack[j] = raw[j].x * silu(raw[j].y);
                        }
                    }
                    else
                    {
                        pack.fill(half_t(0));
                        bool pred = (m * INSN_M + row) < maxRows && laneId * PACK_SIZE < maxCols;
                        if (pred)
                        {
                            pack = load(reinterpret_cast<packed_input const*>(
                                input + (m * INSN_M + row) * stride + laneId * PACK_SIZE));
                        }
                    }
                    store<true>(reinterpret_cast<packed_input*>(&mat[row][laneId * PACK_SIZE]), pack);
                }
                __syncwarp();

                for (int n = 0; n < WARP_N_TILES; n++)
                {
                    int const row = laneId % 16;
                    int const col = n * INSN_N + laneId / 16 * 8;
                    uint4 tmp;
                    ldmatrix(&mat[row][col], tmp);
                    *reinterpret_cast<uint4*>(&out[m * WARP_N_TILES + n]) = tmp;
                }
                __syncwarp();
            }
        }
    };

    template <typename F>
    __device__ __forceinline__ static fpsum_warp apply_act(fpsum_warp fpsum, F func)
    {
        fpsum_warp result;
#pragma unroll
        for (int i = 0; i < WARP_M_TILES; i++)
        {
#pragma unroll
            for (int j = 0; j < WARP_N_TILES; j++)
            {
#pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    half2_t& dst = result[i * WARP_N_TILES + j].data[k];
                    half2_t src = fpsum[i * WARP_N_TILES + j].data[k];
                    dst.x = func(src.x);
                    dst.y = func(src.y);
                }
            }
        }
        return result;
    }

    struct EpilogueDefault
    {
        struct Arguments
        {
            half_t* out;
            int actualM, actualN;
        };

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments const& args)
        {
            int const warpId = threadIdx.x / WARP_SIZE;

            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];

            int const m_offset = binfo.bm * BLOCK_M + warpId * WARP_M;
            int const n_offset = binfo.bn * BLOCK_N;

            unpack_fpsum()(fpsum, args.out + m_offset * args.actualN + n_offset, args.actualN, args.actualM - m_offset,
                args.actualN - n_offset, shmem[warpId],
                [&](int rowId, unpack_fpsum::pack_t& pack) ALWAYSINLINE
                {
                    if constexpr (std::is_same_v<half_t, half>)
                    {
#pragma unroll
                        for (int i = 0; i < pack.size(); i++)
                        {
                            pack[i] = __hmin(pack[i], (half) 65504);
                            pack[i] = __hmax(pack[i], (half) -65504);
                        }
                    }
                });
        }
    };

    struct EpilogueNop
    {
        // workaround for layout mismatch between host and device code
        struct Arguments
        {
            size_t unused;
        };

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments const& args)
        {
        }
    };

    template <bool USE_BIAS = true, bool USE_SCALE = false>
    struct EpilogueBias
    {
        struct Arguments
        {
            packed_wscale_t const* bias; // [N / BLOCK_N, WSCALES_NUM_PACKS, WSCALES_VALID_LANES] of packed_wscale_t
            packed_wscale_t const* scale;
        };

        __device__ __forceinline__ void apply_bias(
            fpsum_warp& fpsum, int M, int N, int K, packed_wscale_t const* bias, packed_wscale_t const* scale)
        {
            int const laneId = threadIdx.x % WARP_SIZE;

            // if (laneId == 0) {
            //     printf("block.x=%d block.y=%d warpId=%d bias=%p\n", blockIdx.x, blockIdx.y, threadIdx.x / WARP_SIZE,
            //     bias);
            // }

            wscale_warp b, s;
            if constexpr (USE_BIAS)
            {
                load_wscale(bias, 0, N, b, true);
            }
            if constexpr (USE_SCALE)
            {
                load_wscale(scale, 0, N, s, true);
            }

            for (int j = 0; j < WARP_N_TILES; j++)
            {
                half2_t b1, b2;
                half2_t s1, s2;
                if constexpr (USE_BIAS)
                {
                    b1 = broadcast_wscale(b, j * 4, laneId);
                    b2 = broadcast_wscale(b, j * 4 + 2, laneId);
                }
                if constexpr (USE_SCALE)
                {
                    s1 = broadcast_wscale(s, j * 4, laneId);
                    s2 = broadcast_wscale(s, j * 4 + 2, laneId);
                }

                for (int i = 0; i < WARP_M_TILES; i++)
                {
                    auto& fsum = fpsum[i * WARP_N_TILES + j];

                    if constexpr (USE_SCALE && USE_BIAS)
                    {
                        fsum.data[0] = __hfma2(fsum.data[0], s1, b1);
                        fsum.data[1] = __hfma2(fsum.data[1], s1, b1);
                        fsum.data[2] = __hfma2(fsum.data[2], s2, b2);
                        fsum.data[3] = __hfma2(fsum.data[3], s2, b2);
                    }
                    else if constexpr (USE_SCALE)
                    {
                        fsum.data[0] = __hmul2(fsum.data[0], s1);
                        fsum.data[1] = __hmul2(fsum.data[1], s1);
                        fsum.data[2] = __hmul2(fsum.data[2], s2);
                        fsum.data[3] = __hmul2(fsum.data[3], s2);
                    }
                    else if constexpr (USE_BIAS)
                    {
                        fsum.data[0] = __hadd2(fsum.data[0], b1);
                        fsum.data[1] = __hadd2(fsum.data[1], b1);
                        fsum.data[2] = __hadd2(fsum.data[2], b2);
                        fsum.data[3] = __hadd2(fsum.data[3], b2);
                    }
                }
            }
        }

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp& fpsum, int M, int N, int K, Arguments const& args)
        {
            int const bn = binfo.bn;
            if constexpr (USE_BIAS || USE_SCALE)
            {
                apply_bias(fpsum, M, N, K, args.bias + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                    args.scale + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES);
            }
        }
    };

    struct EpilogueSilu
    {
        struct Arguments
        {
            size_t unused;
        };

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp& fpsum, int M, int N, int K, Arguments const& args)
        {
            fpsum = apply_act(fpsum, [](half_t x) { return silu(x); });
        }
    };

    template <typename... Epilogues>
    struct EpilogueCombination
    {
        using Arguments = std::tuple<typename Epilogues::Arguments...>;

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp& fpsum, int M, int N, int K, Arguments const& args)
        {
            // this function makes intellisense crashes :(
#if __INTELLISENSE__
            __trap(); // should not happen when actually compiling
#else
            std::tuple<Epilogues...> epilogues;
            auto run = [&]<size_t idx>()
            { std::get<idx>(epilogues).operator()(binfo, fpsum, M, N, K, std::get<idx>(args)); };
            auto foreach = [&]<size_t... Is>(std::index_sequence<Is...>) { (run.template operator()<Is>(), ...); };
            foreach (std::make_index_sequence<sizeof...(Epilogues)>())
                ;
#endif
        }
    };
};

#define IMPORT_GEMM_BASE(config)                                                                                       \
    using Base = GEMMBase<config>;                                                                                     \
    using Base::BLOCK_M;                                                                                               \
    using Base::BLOCK_N;                                                                                               \
    using Base::WARP_SIZE;                                                                                             \
    using Base::NUM_WARPS;                                                                                             \
    using Base::INSN_M;                                                                                                \
    using Base::INSN_N;                                                                                                \
    using Base::INSN_K;                                                                                                \
    using typename Base::half_t;                                                                                       \
    using typename Base::half2_t;                                                                                      \
    using Base::WARP_M;                                                                                                \
    using Base::WARP_N;                                                                                                \
    using Base::WARP_K;                                                                                                \
    using Base::WARP_M_TILES;                                                                                          \
    using Base::WARP_N_TILES;                                                                                          \
    using Base::WARP_K_TILES;                                                                                          \
    using Base::WSCALES_PACK_SIZE;                                                                                     \
    using Base::WSCALES_NUM_PACKS;                                                                                     \
    using Base::WSCALES_VALID_LANES;                                                                                   \
    using Base::ASCALES_PACK_SIZE;                                                                                     \
    using Base::ASCALES_NUM_PACKS;                                                                                     \
    using Base::ASCALES_VALID_LANES;                                                                                   \
    using typename Base::packed_act_t;                                                                                 \
    using typename Base::packed_wgt_t;                                                                                 \
    using typename Base::packed_psum_t;                                                                                \
    using typename Base::packed_fpsum_t;                                                                               \
    using typename Base::packed_gated_fpsum_t;                                                                         \
    using typename Base::packed_f32psum_t;                                                                             \
    using typename Base::packed_wscale_t;                                                                              \
    using typename Base::packed_ascale_t;                                                                              \
    using typename Base::act_warp;                                                                                     \
    using typename Base::wgt_warp;                                                                                     \
    using typename Base::ascale_warp;                                                                                  \
    using typename Base::wscale_warp;                                                                                  \
    using typename Base::fpsum_warp;                                                                                   \
    using typename Base::f32psum_warp;                                                                                 \
    using typename Base::gated_fpsum_warp;                                                                             \
    using typename Base::BlockInfo;                                                                                    \
    using typename Base::unpack_fpsum;                                                                                 \
    using typename Base::EpilogueDefault;                                                                              \
    using typename Base::EpilogueNop;                                                                                  \
    template <bool USE_BIAS, bool USE_SCALE>                                                                           \
    using EpilogueBias = typename Base::EpilogueBias<USE_BIAS, USE_SCALE>;                                             \
    using Base::mma_f16xf16_f32;                                                                                       \
    using Base::packed_fp32_to_fp16;                                                                                   \
    using Base::packed_fp16_to_fp32;                                                                                   \
    using Base::load_act;                                                                                              \
    using Base::load_wgt;                                                                                              \
    using Base::load_ascale;                                                                                           \
    using Base::load_wscale;                                                                                           \
    using Base::broadcast_wscale;                                                                                      \
    using Base::broadcast_ascale;                                                                                      \
    using Base::apply_scales;                                                                                          \
    using Base::pack_ascales;                                                                                          \
    using Base::pack_wscales;                                                                                          \
    using Base::apply_act;

template <typename kernel>
constexpr int min_arch()
{
    if constexpr (requires { kernel::MIN_ARCH; })
    {
        return kernel::MIN_ARCH;
    }
    else
    {
        return 0;
    }
}

template <typename kernel>
constexpr int max_arch()
{
    if constexpr (requires { kernel::MAX_ARCH; })
    {
        return kernel::MAX_ARCH;
    }
    else
    {
        return INT_MAX;
    }
}

template <typename kernel, typename... T>
__global__ static void invoke_kernel(T... args)
{
#ifdef __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ >= min_arch<kernel>() && __CUDA_ARCH__ <= max_arch<kernel>())
    {
        kernel()(args...);
    }
    else
    {
        trap_unsupported_arch();
    }
#else
    // ???
    kernel()(args...);
#endif
}

template <typename T>
__global__ static void test_sizeof_device()
{
    printf("sizeof on device = %d\n", (int) sizeof(T));
}

template <typename T>
static void test_sizeof_host()
{
    printf("sizeof on host = %d\n", (int) sizeof(T));
}

template <typename T>
static void test_sizeof()
{
    printf("typeid = %s\n", typeid(T).name());
    test_sizeof_host<T>();
    test_sizeof_device<T><<<1, 1>>>();
    checkCUDA(cudaDeviceSynchronize());
}

}; // namespace nunchaku::kernels
