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

namespace nunchaku::kernels
{

template <typename Config>
class Lora;

#ifndef __INTELLISENSE__
template <typename Config>
class Lora : public GEMMBase<Config>
{
#else
template <>
class Lora<GEMMConfig_W4A4_FP16> : public GEMMBase<GEMMConfig_W4A4_FP16>
{
    using Config = GEMMConfig_W4A4_FP16;
#endif
public:
    IMPORT_GEMM_BASE(Config);

public:
    static constexpr int MAX_RANK = 1024;
    static constexpr int WARP_R = 16;

    // static constexpr int LORA_RANK = rank;
    static constexpr int LORA_M_TILES = WARP_M / 16;
    static constexpr int LORA_R_TILES = WARP_R / 16;
    static constexpr int LORA_N_TILES = WARP_N / 16;

    static_assert(LORA_M_TILES == WARP_M_TILES);
    static_assert(LORA_N_TILES == WARP_N_TILES);

    // lora_down: [WARP_M, WARP_N] x [WARP_N, R] (row-wise) = [WARP_M, R]
    // lora up:   [WARP_M, R]      x [WARP_N, R] (col-wise) = [WARP_M, WARP_N]
    // we use fp32 for lora activation since there's no bf16 reduction in sm_89 :(

    using lora_act_warp = std::array<packed_f32psum_t, LORA_M_TILES * LORA_R_TILES>;
    using lora_act16_warp = std::array<packed_fpsum_t, LORA_M_TILES * LORA_R_TILES>;
    using lora_wgt_warp = std::array<packed_fpsum_t, LORA_N_TILES * LORA_R_TILES>;

    using scale_t = std::array<float, MAX_RANK / 16>;

    // lora_wgt:   [N / 16, rank / WARP_R, LORA_R_TILES, WARP_SIZE] of packed_fpsum_t
    //             [N / 16, rank / 16, WARP_SIZE]
    __device__ __forceinline__ static void load_lora_wgt(
        packed_fpsum_t const* ptr, int rtile, int rank, lora_wgt_warp& result, bool pred)
    {
        int const laneId = threadIdx.x % WARP_SIZE;

        packed_fpsum_t const* ptr_lane = &ptr[rtile * LORA_R_TILES * WARP_SIZE + laneId];
        int const stride_ntile = rank / 16 * WARP_SIZE;

        unrolled_loop<LORA_N_TILES>(
            [&]<int n>()
            {
                unrolled_loop<LORA_R_TILES>(
                    [&]<int r>()
                    {
                        constexpr int roffset = r * WARP_SIZE;
                        const int noffset = n * stride_ntile;
                        result[n * LORA_R_TILES + r] = load_pred(ptr_lane + noffset + roffset, pred);
                    });
            });
    }

    // lora_act: [M / BLOCK_M, rank / WARP_R, NUM_WARPS, LORA_M_TILES, LORA_R_TILES, 8, WARP_SIZE] of float
    __device__ __forceinline__ static void load_lora_act(float const* ptr, int rtile, lora_act_warp& result, bool pred)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        float const* ptrlane
            = &ptr[(rtile * NUM_WARPS + warpId) * LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE + laneId];

        unrolled_loop<LORA_M_TILES>(
            [&]<int m>()
            {
                unrolled_loop<LORA_R_TILES>(
                    [&]<int r>
                    {
                        constexpr int i = m * LORA_R_TILES + r;
                        unrolled_loop<8>(
                            [&]<int j>()
                            {
                                constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                                result[i].data[j]
                                    = load_pred(ptrlane + offset, pred); // * scales[rtile * LORA_R_TILES + r];
                            });
                        // CHECK_NAN(tmp, "load_lora_act.tmp");
                    });
            });
    }

    // no vector reduction in sm_89 :(
    __device__ __forceinline__ static void reduce_lora_act(float* ptr, int rtile, lora_act_warp val, bool pred)
    {
        int const laneId = threadIdx.x % WARP_SIZE;
        int const warpId = threadIdx.x / WARP_SIZE;

        float* ptrlane = &ptr[(rtile * NUM_WARPS + warpId) * LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE + laneId];

        unrolled_loop<LORA_M_TILES * LORA_R_TILES>(
            [&]<int i>()
            {
                unrolled_loop<8>(
                    [&]<int j>()
                    {
                        constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                        reduce_add_pred(&ptrlane[offset], val[i].data[j], pred);
                    });
            });
    }

    // __device__ __forceinline__
    // static void reduce_lora_act(float *ptr, lora_act_warp val, int m) {
    //     const int laneId = threadIdx.x % WARP_SIZE;

    //     float *ptrlane = ptr + laneId + m * LORA_R_TILES * 8 * WARP_SIZE;

    //     unrolled_loop<LORA_R_TILES>([&]<int r>() {
    //         unrolled_loop<8>([&]<int j>() {
    //             constexpr int offset = r * 8 * WARP_SIZE + j * WARP_SIZE;
    //             reduce_add(&ptrlane[offset], val[m * LORA_R_TILES + r].data[j]);
    //         });
    //     });
    // }

    struct EpilogueLoraUp
    {
        struct Arguments
        {
            float const* lora_act;
            packed_fpsum_t const* lora_wgt_up;
            int rank;

            scale_t scales;

            bool alwaysfalse;
        };

        __device__ __forceinline__ static void apply_lora_up(fpsum_warp& fpsum, float const* act,
            packed_fpsum_t const* wgt, scale_t const& scales, int rank, bool alwaysfalse)
        {
            constexpr int NUM_STAGES = 2;

            int const laneId = threadIdx.x % WARP_SIZE;
            int const warpId = threadIdx.x / WARP_SIZE;

            lora_act_warp lora_act[NUM_STAGES]; // 32
            lora_wgt_warp lora_wgt[NUM_STAGES]; // 64

            int dummy = 0;

#pragma unroll
            for (int k = 0; k < NUM_STAGES - 1; k++)
            {
                // we have rank > 0
                bool const pred = k == 0 ? true : k < rank / WARP_R;
                load_lora_act(act, 0, lora_act[k], pred);
                load_lora_wgt(wgt, 0, rank, lora_wgt[k], pred);
            }

            f32psum_warp f32psum = packed_fp16_to_fp32(fpsum); // 128

            auto compute = [&scales](lora_act_warp A, lora_wgt_warp W, f32psum_warp& f32psum, int rtile) ALWAYSINLINE
            {
                lora_act16_warp A_fp16;
                for (int m = 0; m < LORA_M_TILES; m++)
                {
                    for (int r = 0; r < LORA_R_TILES; r++)
                    {
                        packed_f32psum_t pack = A[m * LORA_R_TILES + r];
#pragma unroll
                        for (int j = 0; j < 8; j++)
                        {
                            pack.data[j] *= scales[rtile * LORA_R_TILES + r];
                        }
                        A_fp16[m * LORA_R_TILES + r] = packed_fp32_to_fp16(pack);
                    }
                }
                for (int m = 0; m < LORA_M_TILES; m++)
                {
                    for (int n = 0; n < LORA_N_TILES; n++)
                    {
                        for (int r = 0; r < LORA_R_TILES; r++)
                        {
                            CHECK_NAN(lora_act[m * LORA_R_TILES + r], "lora_act");
                            CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "lora_wgt");
                            f32psum[m * WARP_N_TILES + n] = mma_f16xf16_f32(
                                A_fp16[m * LORA_R_TILES + r], W[n * LORA_R_TILES + r], f32psum[m * WARP_N_TILES + n]);
                        }
                    }
                }
            };

            for (int k1 = 0; k1 < rank / WARP_R; k1 += NUM_STAGES)
            {
#pragma unroll
                for (int k2 = 0; k2 < NUM_STAGES; k2++)
                {
                    if (k1 + k2 >= rank / WARP_R)
                    {
                        break;
                    }

                    int nextk = k1 + k2 + NUM_STAGES - 1;
                    int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                    bool pred = nextk < rank / WARP_R;

                    if (alwaysfalse)
                    {
                        act += kernels::bit_cast<int>(lora_act[k2][0].data[0]);
                    }

                    if (alwaysfalse)
                    {
                        dummy = clock();
                    }

                    load_lora_act(act, nextk, lora_act[idx], pred);
                    load_lora_wgt(wgt, nextk, rank, lora_wgt[idx], pred);

                    compute(lora_act[k2], lora_wgt[k2], f32psum, k1 + k2);
                }
            }

            // NVCC does not know rank > 0 :(
            // it will generate a branch instruction to skip the initial load
            // the branch splits the basic blocks and prevents the overlap of memory access and computing
            // (packed_fp16_to_fp32) add fake dependency of loaded data so NVCC will not skip the load
#pragma unroll
            for (int k = 0; k < NUM_STAGES - 1; k++)
            {
#pragma unroll
                for (auto&& data : lora_act[k])
                {
#pragma unroll
                    for (int i = 0; i < 8; i++)
                    {
                        dummy ^= kernels::bit_cast<int>(data.data[i]);
                    }
                }
#pragma unroll
                for (auto&& data : lora_wgt[k])
                {
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        dummy ^= kernels::bit_cast<int>(data.data[i]);
                    }
                }
            }

            unused_var(dummy, alwaysfalse);

            fpsum = packed_fp32_to_fp16(f32psum);
        }

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp& fpsum, int M, int N, int K, Arguments const& args)
        {
            int const bm = binfo.bm;
            int const bn = binfo.bn;

            CHECK_NAN(fpsum, "fpsum");

            apply_lora_up(fpsum,
                args.lora_act + bm * (args.rank / WARP_R) * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE),
                args.lora_wgt_up + bn * (BLOCK_N / 16) * (args.rank / 16) * WARP_SIZE, args.scales, args.rank,
                args.alwaysfalse);

            CHECK_NAN(fpsum, "fpsum");
        }
    };

    struct EpilogueLoraDown
    {
        struct Arguments
        {
            packed_fpsum_t const* lora_wgt_down;
            float* lora_act;

            int rank;

            bool alwaysfalse;
        };

        __device__ __forceinline__ static void apply_lora_down(
            fpsum_warp& fpsum, float* act, packed_fpsum_t const* wgt, int rank, bool alwaysfalse)
        {
            constexpr int NUM_STAGES = 2;

            int const laneId = threadIdx.x % WARP_SIZE;
            int const warpId = threadIdx.x / WARP_SIZE;

            lora_wgt_warp lora_wgt[NUM_STAGES]; // 64

#pragma unroll
            for (int k = 0; k < NUM_STAGES - 1; k++)
            {
                // we have rank > 0
                bool pred = k == 0 ? true : k < rank / WARP_R;
                load_lora_wgt(wgt, 0, rank, lora_wgt[k], pred);
            }

            auto compute = [](lora_wgt_warp W, fpsum_warp fpsum) -> lora_act_warp
            {
                lora_act_warp lora_act;
                lora_act.fill(packed_f32psum_t::zeros());

#pragma unroll
                for (int m = 0; m < LORA_M_TILES; m++)
                {
#pragma unroll
                    for (int n = 0; n < LORA_N_TILES; n++)
                    {
#pragma unroll
                        for (int r = 0; r < LORA_R_TILES; r++)
                        {
                            auto& psum = lora_act[m * LORA_R_TILES + r];

                            CHECK_NAN(fpsum[m * WARP_N_TILES + n], "apply_lora_down.fpsum");
                            CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "apply_lora_down.lora_wgt");

                            psum = mma_f16xf16_f32(fpsum[m * WARP_N_TILES + n], W[n * LORA_R_TILES + r], psum);

                            CHECK_NAN(psum, "apply_lora_down.psum");
                        }
                    }
                }

                return lora_act;
            };

            int dummy = 0;

            for (int k1 = 0; k1 < rank / WARP_R; k1 += NUM_STAGES)
            {
#pragma unroll
                for (int k2 = 0; k2 < NUM_STAGES; k2++)
                {
                    if (k1 + k2 >= rank / WARP_R)
                    {
                        break;
                    }

                    int nextk = k1 + k2 + NUM_STAGES - 1;
                    int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                    bool pred = nextk < rank / WARP_R;

                    if (alwaysfalse)
                    {
                        wgt += kernels::bit_cast<int>(lora_wgt[k2][0].data[0]);
                    }

                    if (alwaysfalse)
                    {
                        dummy = clock();
                    }

                    load_lora_wgt(wgt, nextk, rank, lora_wgt[idx], pred);

                    if (alwaysfalse)
                    {
                        dummy = clock();
                    }

                    lora_act_warp lora_act = compute(lora_wgt[k2], fpsum);

                    reduce_lora_act(act, k1 + k2, lora_act, true);
                }
            }

#pragma unroll
            for (int k = 0; k < NUM_STAGES - 1; k++)
            {
#pragma unroll
                for (auto&& data : lora_wgt[k])
                {
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        dummy ^= kernels::bit_cast<int>(data.data[i]);
                    }
                }
            }

            unused_var(dummy, alwaysfalse);
        }

        __device__ __forceinline__ void operator()(
            const BlockInfo binfo, fpsum_warp& fpsum, int M, int N, int K, Arguments const& args)
        {
            int const bm = binfo.bm;
            int const bn = binfo.bn;

            apply_lora_down(fpsum,
                args.lora_act + bm * (args.rank / WARP_R) * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE),
                args.lora_wgt_down + bn * (BLOCK_N / 16) * (args.rank / 16) * WARP_SIZE, args.rank, args.alwaysfalse);
        }
    };
};

}; // namespace nunchaku::kernels
