// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
//   journal={arXiv preprint arXiv:2411.05007},
//   year={2024}
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

namespace nunchaku::kernels {

template<typename Config>
class Epilogues;

#ifndef __INTELLISENSE__
template<typename Config>
class Epilogues : public GEMMBase<Config> {
#else
template<>
class Epilogues<GEMMConfig_W4A4_FP16> : public GEMMBase<GEMMConfig_W4A4_FP16> {
    using Config = GEMMConfig_W4A4_FP16;
#endif
public:
    IMPORT_GEMM_BASE(Config);

public:
    struct EpilogueGelu {
        struct Arguments {
            size_t unused;
        };

        // static constexpr float SHIFT_VALUE = 0.171875f;

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M, int N, int K, const Arguments &args) {
#pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
#pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        half2_t &data = fpsum[i * WARP_N_TILES + j].data[k];
                        data          = gelu_half2(data);
                        // data = __hadd2(data, half2_t(SHIFT_VALUE, SHIFT_VALUE));
                    }
                }
            }
        }
    };

    // template<int PoolSize = 128>
    struct EpilogueQKVProj {
        struct Arguments {
            half_t *out;
            int actualM, actualN;

            half_t *pool_out;               // [M / PoolSize, N]
            const float *rotary_emb;        // [M, HEAD_DIM / 2, ROTARY_EMB_NUM_ELEMENTS]
            const half_t *rmsnorm_weight_q; // [HEAD_DIM]
            const half_t *rmsnorm_weight_k; // [HEAD_DIM]
            float epsilon;
        };

        static constexpr int HEAD_DIM           = 128;
        static constexpr int NUM_HEADS_PER_WARP = WARP_N / HEAD_DIM;

        static constexpr int PoolSize            = 128;
        static constexpr int NUM_WARPS_PER_POOL  = PoolSize / WARP_M;
        static constexpr int NUM_POOLS_PER_BLOCK = BLOCK_M / PoolSize;

        static constexpr int ROTARY_EMB_NUM_ELEMENTS = 2; // 1 for theta, 2 for {sin, cos} pair

        __device__ __forceinline__ static void apply(fpsum_warp fpsum,
                                                     half_t *out,
                                                     int M,
                                                     int N,
                                                     int K,
                                                     half_t *pool_out,
                                                     const float *rotary_emb,
                                                     const half_t *rmsnorm_weight,
                                                     float epsilon,
                                                     int maxRows) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];

            constexpr int PACK_SIZE = unpack_fpsum::PACK_SIZE;
            using pack_t            = unpack_fpsum::pack_t;

            using pack_rope_t            = std::array<float, PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS>;
            constexpr int LANES_PER_HEAD = HEAD_DIM / PACK_SIZE;

            pack_t reduce_tmp;
            __shared__ alignas(128) pack_t pool[NUM_WARPS];

            // load rmsnorm scales
            pack_t rms;
            if (laneId < LANES_PER_HEAD) {
                rms = load(reinterpret_cast<const pack_t *>(&rmsnorm_weight[laneId * PACK_SIZE]));
            }
            if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                for (int i = 0; i < PACK_SIZE; i++) {
                    rms[i] = __shfl_sync(~0, rms[i], laneId % LANES_PER_HEAD);
                }
            }

            const float *rotary_emb_base_addr = &rotary_emb[(warpId * WARP_M) * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS +
                                                            laneId * PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS];

            CHECK_NAN(fpsum, "fpsum");

            unpack_fpsum()(fpsum,
                           out + warpId * WARP_M * N,
                           N,
                           maxRows - warpId * WARP_M,
                           INT_MAX,
                           shmem[warpId],
                           [&](int rowId, pack_t &pack) ALWAYSINLINE {
                               // load rope
                               pack_rope_t rope;
                               if (laneId < LANES_PER_HEAD) {
                                   // freq = load(reinterpret_cast<pack_freq_t *>(&freqs_cis[(warpId * WARP_M + rowId) *
                                   // HEAD_DIM * 2 + laneId * PACK_SIZE * 2]));
                                   rope = load(reinterpret_cast<const pack_rope_t *>(
                                       &rotary_emb_base_addr[rowId * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS]));
                               }
                               if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                                   for (int i = 0; i < rope.size(); i++) {
                                       rope[i] = __shfl_sync(~0, rope[i], laneId % LANES_PER_HEAD);
                                   }
                               }

                               // rmsnorm
                               float sqrsum = 0.0f;
                               for (int i = 0; i < PACK_SIZE; i++) {
                                   sqrsum += float(pack[i]) * float(pack[i]);
                                   CHECK_NAN(sqrsum, "sqrsum");
                               }
#pragma unroll
                               for (int mask = LANES_PER_HEAD / 2; mask > 0; mask /= 2) {
                                   sqrsum += __shfl_xor_sync(~0, sqrsum, mask);
                               }
                               sqrsum /= HEAD_DIM;
                               float coef = cuda_frsqrt(sqrsum + epsilon);
                               CHECK_NAN(coef, "coef");

                               for (int i = 0; i < PACK_SIZE; i++) {
                                   pack[i] *= coef * float(rms[i]);

                                   CHECK_NAN(rms[i], "rms.wgt");
                                   CHECK_NAN(pack[i], "rms.out");
                               }

#if 1
                               // rope
                               for (int i = 0; i < PACK_SIZE; i += 2) {
                                   float2 pack2 = half22float2(half2_t(pack[i], pack[i + 1]));

                                   CHECK_NAN(freq[i].x, "rope.freq");
                                   CHECK_NAN(freq[i].y, "rope.freq");
                                   CHECK_NAN(freq[i + 1].x, "rope.freq");
                                   CHECK_NAN(freq[i + 1].y, "rope.freq");

                                   // half2_t tmp = __hmul2(freq[i], pack2);
                                   // tmp = __hfma2(freq[i+1], pack2, tmp);
                                   // pack[i] = tmp.x;
                                   // pack[i+1] = tmp.y;

                                   // printf("block.x=%d block.y=%d warpId=%d rowId=%d (%d) freqs = %f %f %f %f\n",
                                   //     blockIdx.x, blockIdx.y, warpId, rowId,
                                   //     blockIdx.x * BLOCK_M + warpId * WARP_M + rowId,
                                   //     (float)freq[i].x, (float)freq[i].y, (float)freq[i+1].x, (float)freq[i+1].y
                                   // );
                                   // __trap();

                                   // half2_t tmp = __hmul2(half2_t(pack2.x, pack2.x), freq[i]);
                                   // tmp = __hfma2(half2_t(pack2.y, pack2.y), freq[i+1], tmp);
                                   // pack[i] = tmp.x;
                                   // pack[i+1] = tmp.y;

                                   float sin, cos;

                                   if constexpr (ROTARY_EMB_NUM_ELEMENTS == 1) {
                                       sin = cuda_sin(rope[i / 2]);
                                       cos = cuda_cos(rope[i / 2]);
                                   }
                                   if constexpr (ROTARY_EMB_NUM_ELEMENTS == 2) {
                                       sin = rope[i];
                                       cos = rope[i + 1];
                                   }

                                   // pack[i]   = pack2.x * freq[i].x   + pack2.y * freq[i].y;
                                   // pack[i+1] = pack2.x * freq[i+1].x + pack2.y * freq[i+1].y;

                                   pack[i]     = half_t(pack2.x * cos - pack2.y * sin);
                                   pack[i + 1] = half_t(pack2.x * sin + pack2.y * cos);

                                   CHECK_NAN(pack[i], "rope.out");
                                   CHECK_NAN(pack[i + 1], "rope.out");
                               }
#endif

                               // mean pool
                               for (int i = 0; i < PACK_SIZE; i++) {
                                   reduce_tmp[i] += pack[i];
                               }
                           });

            if (!pool_out) {
                return;
            }

            store<true>(&pool[warpId], reduce_tmp);
            __syncthreads();

            if (warpId < NUM_POOLS_PER_BLOCK) {
                const int row = warpId * NUM_WARPS_PER_POOL;
                reduce_tmp    = load<true>(&pool[row]);

                for (int i = 1; i < NUM_WARPS_PER_POOL; i++) {
                    pack_t pack = load<true>(&pool[row + i]);
                    for (int j = 0; j < PACK_SIZE; j++) {
                        reduce_tmp[j] += pack[j];
                    }
                }
                for (int j = 0; j < PACK_SIZE; j++) {
                    reduce_tmp[j] /= PoolSize;
                }

                store(reinterpret_cast<pack_t *>(pool_out + warpId * N), reduce_tmp);
            }
            __syncthreads();
        }

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, const Arguments &args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            assert(binfo.numBlocksN % 3 == 0);
            const bool is_q = bn < binfo.numBlocksN / 3;
            const bool is_k = !is_q && bn < binfo.numBlocksN / 3 * 2;

            assert(!args.pool_out || args.actualM == M);
            assert(args.actualN == N);

            if (is_q || is_k) {
                apply(fpsum,
                      args.out + bm * BLOCK_M * args.actualN + bn * BLOCK_N,
                      M,
                      N,
                      K,
                      args.pool_out ? args.pool_out + bm * BLOCK_M / PoolSize * N : nullptr,
                      args.rotary_emb + bm * BLOCK_M * (HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS),
                      is_q ? args.rmsnorm_weight_q : args.rmsnorm_weight_k,
                      args.epsilon,
                      args.actualM - bm * BLOCK_M);
            } else {
                EpilogueDefault()(binfo,
                                  fpsum,
                                  M,
                                  N,
                                  K,
                                  typename EpilogueDefault::Arguments{
                                      .out     = args.out,
                                      .actualM = args.actualM,
                                      .actualN = args.actualN,
                                  });
            }
        }
    };

    struct EpilogueRMSNormRope {
        static constexpr int HEAD_DIM              = 128;
        static constexpr int NUM_HEADS_PER_WARP    = WARP_N / HEAD_DIM;
        static constexpr int WARP_N_TILES_PER_HEAD = WARP_N_TILES / NUM_HEADS_PER_WARP;

        static constexpr int ROTARY_EMB_NUM_ELEMENTS = 2;

        using packed_rotemb_t                    = float4;
        static constexpr int WARP_N_ROTEMB_TILES = WARP_N_TILES / NUM_HEADS_PER_WARP * 2;
        using rotemb_warp = std::array<packed_rotemb_t, WARP_M_TILES * WARP_N_ROTEMB_TILES>; // 128 regs

        struct Arguments {
            // **packed** [M, HEAD_DIM] float => [M // 16, HEAD_DIM // 8, WARP_SIZE] of packed_rotemb_t
            // aka [M // BLOCK_M, NUM_WARPS, WARP_M_TILES, WARP_N_TILES // NUM_HEADS_PER_WARP * 2, WARP_SIZE]
            const packed_rotemb_t *rotary_emb;
            const half_t *rmsnorm_weight_q; // [HEAD_DIM]
            const half_t *rmsnorm_weight_k; // [HEAD_DIM]
            float epsilon;
        };

        __device__ __forceinline__ static rotemb_warp load_rotemb(const packed_rotemb_t *ptr_rotemb) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            rotemb_warp rotemb;
            const packed_rotemb_t *ptrlane =
                &ptr_rotemb[warpId * WARP_M_TILES * WARP_N_ROTEMB_TILES * WARP_SIZE + laneId];

            unrolled_loop<WARP_M_TILES>([&]<int i>() {
                unrolled_loop<WARP_N_ROTEMB_TILES>([&]<int j>() {
                    constexpr int offset                = (i * WARP_N_ROTEMB_TILES + j) * WARP_SIZE;
                    rotemb[i * WARP_N_ROTEMB_TILES + j] = load(&ptrlane[offset]);
                });
            });

            return rotemb;
        }

        __device__ __forceinline__ static void load_rmsnorm(const half_t *ptr_rmsnorm_weight, half_t *shmem) {
            const int laneId = threadIdx.x % WARP_SIZE;

            static constexpr int PACK_SIZE = HEAD_DIM / WARP_SIZE;
            using packed_t                 = std::array<half_t, PACK_SIZE>;

            packed_t pack = load(reinterpret_cast<const packed_t *>(ptr_rmsnorm_weight + laneId * PACK_SIZE));
            store<true>(reinterpret_cast<packed_t *>(shmem + laneId * PACK_SIZE), pack);
        }

        __device__ __forceinline__ static packed_fpsum_t load_rmsnorm_from_shmem(half_t *shmem, int n) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int col    = n * INSN_N + laneId / 16 * 8; // lane 0-15: n*16+0, lane 16-31: n*16+8
            uint4 tmp;
            ldmatrix(shmem + col, tmp);
            return kernels::bit_cast<packed_fpsum_t>(tmp);
        }

        __device__ __forceinline__ static void
        apply(fpsum_warp &fpsum, const packed_rotemb_t *ptr_rotemb, const half_t *ptr_rmsnorm_weight, float epsilon) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            __shared__ half_t shmem_rmsnorm[NUM_WARPS][HEAD_DIM];
            load_rmsnorm(ptr_rmsnorm_weight, &shmem_rmsnorm[warpId][0]);
            __syncwarp();

            rotemb_warp rotemb = load_rotemb(ptr_rotemb);

            float rmsnorm_coef[NUM_HEADS_PER_WARP][WARP_M_TILES][2];

            auto sqr = [](half2_t val) ALWAYSINLINE {
                float2 fval = half22float2(val);
                return fval.x * fval.x + fval.y * fval.y;
            };

#pragma unroll
            for (int head = 0; head < NUM_HEADS_PER_WARP; head++) {
                const int n_offset = head * WARP_N_TILES_PER_HEAD;

#pragma unroll
                for (int m = 0; m < WARP_M_TILES; m++) {
                    float sqrsum[2] = {0.0f, 0.0f};
#pragma unroll
                    for (int n = 0; n < WARP_N_TILES_PER_HEAD; n++) {
                        sqrsum[0] += sqr(fpsum[m * WARP_N_TILES + n + n_offset].data[0]);
                        sqrsum[1] += sqr(fpsum[m * WARP_N_TILES + n + n_offset].data[1]);
                        sqrsum[0] += sqr(fpsum[m * WARP_N_TILES + n + n_offset].data[2]);
                        sqrsum[1] += sqr(fpsum[m * WARP_N_TILES + n + n_offset].data[3]);
                    }
#pragma unroll
                    for (int mask = 1; mask <= 2; mask *= 2) {
                        sqrsum[0] += __shfl_xor_sync(~0, sqrsum[0], mask);
                        sqrsum[1] += __shfl_xor_sync(~0, sqrsum[1], mask);
                    }
                    rmsnorm_coef[head][m][0] = cuda_frsqrt(sqrsum[0] / HEAD_DIM + epsilon);
                    rmsnorm_coef[head][m][1] = cuda_frsqrt(sqrsum[1] / HEAD_DIM + epsilon);
                }
            }

#pragma unroll
            for (int head = 0; head < NUM_HEADS_PER_WARP; head++) {
                const int n_offset = head * WARP_N_TILES_PER_HEAD;

#pragma unroll
                for (int n = 0; n < WARP_N_TILES_PER_HEAD; n++) {
                    packed_f32psum_t rms = packed_fp16_to_fp32(load_rmsnorm_from_shmem(&shmem_rmsnorm[warpId][0], n));
#pragma unroll
                    for (int m = 0; m < WARP_M_TILES; m++) {
                        packed_f32psum_t pack = packed_fp16_to_fp32(fpsum[m * WARP_N_TILES + n + n_offset]);
                        pack.data[0] *= rmsnorm_coef[head][m][0] * rms.data[0];
                        pack.data[1] *= rmsnorm_coef[head][m][0] * rms.data[1];
                        pack.data[2] *= rmsnorm_coef[head][m][1] * rms.data[2];
                        pack.data[3] *= rmsnorm_coef[head][m][1] * rms.data[3];
                        pack.data[4] *= rmsnorm_coef[head][m][0] * rms.data[4];
                        pack.data[5] *= rmsnorm_coef[head][m][0] * rms.data[5];
                        pack.data[6] *= rmsnorm_coef[head][m][1] * rms.data[6];
                        pack.data[7] *= rmsnorm_coef[head][m][1] * rms.data[7];

                        auto rope = [](float &x, float &y, float sin, float cos) ALWAYSINLINE {
                            float ix = x, iy = y;
                            x = ix * cos - iy * sin;
                            y = ix * sin + iy * cos;
                        };

                        {
                            packed_rotemb_t sincos = rotemb[m * WARP_N_ROTEMB_TILES + n * 2];
                            rope(pack.data[0], pack.data[1], sincos.x, sincos.y);
                            rope(pack.data[2], pack.data[3], sincos.z, sincos.w);
                        }
                        {
                            packed_rotemb_t sincos = rotemb[m * WARP_N_ROTEMB_TILES + n * 2 + 1];
                            rope(pack.data[4], pack.data[5], sincos.x, sincos.y);
                            rope(pack.data[6], pack.data[7], sincos.z, sincos.w);
                        }

                        fpsum[m * WARP_N_TILES + n + n_offset] = packed_fp32_to_fp16(pack);
                    }
                }
            }
        }

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M, int N, int K, const Arguments &args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            assert(binfo.numBlocksN % 3 == 0);
            const bool is_q = bn < binfo.numBlocksN / 3;
            const bool is_k = !is_q && bn < binfo.numBlocksN / 3 * 2;

            if (is_q || is_k) {
                apply(fpsum,
                      args.rotary_emb + bm * NUM_WARPS * WARP_M_TILES * WARP_N_ROTEMB_TILES * WARP_SIZE,
                      is_q ? args.rmsnorm_weight_q : args.rmsnorm_weight_k,
                      args.epsilon);
            }
        }
    };

    struct EpiloguePackQKV {
        using attn_half_t  = half;
        using attn_half2_t = half2;
        using packed_qkv_t = uint4;

        static constexpr int HEAD_DIM  = 128;
        static constexpr int INSN_K_QK = 16;
        static constexpr int INSN_K_PV = 16;

        struct Arguments {
            packed_qkv_t *out_q, *out_k, *out_v;
            int actualM;

            // !!! stride in number of packed_qkv_t !!!
            int strideHead_q;
            int strideHead_k;
            int strideHead_v;
        };

        __device__ __forceinline__ static attn_half2_t convert_half2(half2_t input) {
            if constexpr (std::is_same_v<half2_t, attn_half2_t>) {
                return input;
            } else {
                float2 fval = half22float2(input);
                return float22half2<attn_half2_t>(fval);
            }
        }

        __device__ __forceinline__ static packed_qkv_t pack_q(packed_fpsum_t input) {
            packed_qkv_t output;
            output.x = kernels::bit_cast<int>(convert_half2(input.data[0]));
            output.y = kernels::bit_cast<int>(convert_half2(input.data[1]));
            output.z = kernels::bit_cast<int>(convert_half2(input.data[2]));
            output.w = kernels::bit_cast<int>(convert_half2(input.data[3]));
            return output;
        }

        __device__ __forceinline__ static packed_qkv_t pack_k(packed_fpsum_t input) {
            packed_qkv_t output;
            output.x = kernels::bit_cast<int>(convert_half2(input.data[0]));
            output.y = kernels::bit_cast<int>(convert_half2(input.data[2]));
            output.z = kernels::bit_cast<int>(convert_half2(input.data[1]));
            output.w = kernels::bit_cast<int>(convert_half2(input.data[3]));
            return output;
        }

        __device__ __forceinline__ static packed_qkv_t pack_v(packed_fpsum_t input) {
            packed_qkv_t output;
            output.x = kernels::bit_cast<int>(convert_half2(movmatrix(input.data[0])));
            output.y = kernels::bit_cast<int>(convert_half2(movmatrix(input.data[1])));
            output.z = kernels::bit_cast<int>(convert_half2(movmatrix(input.data[2])));
            output.w = kernels::bit_cast<int>(convert_half2(movmatrix(input.data[3])));
            return output;
        }

        __device__ __forceinline__ static void mask(packed_qkv_t &pack, uint32_t maskVal, int m, int maxRows) {
            const int laneId = threadIdx.x % WARP_SIZE;
            if (m * INSN_M + laneId / 4 >= maxRows) {
                pack.x = maskVal;
                pack.z = maskVal;
            }
            if (m * INSN_M + laneId / 4 + 8 >= maxRows) {
                pack.y = maskVal;
                pack.w = maskVal;
            }
        }

        // qkv: [batch, head, bm, NUM_WARPS, WARP_M_TILES, WARP_N_TILES, WARP_SIZE] of packed_qkv_t
        template<typename F>
        __device__ __forceinline__ static void
        apply(fpsum_warp &fpsum, packed_qkv_t *ptr_output, int maxRows, F &&funcPack, attn_half2_t maskVal) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            static_assert(HEAD_DIM == WARP_N);

            packed_qkv_t *ptrlane = &ptr_output[((warpId * WARP_M_TILES + 0) * WARP_N_TILES + 0) * WARP_SIZE + laneId];

            unrolled_loop<WARP_M_TILES>([&]<int m>() ALWAYSINLINE {
                unrolled_loop<WARP_N_TILES>([&]<int n>() ALWAYSINLINE {
                    packed_qkv_t pack = funcPack(fpsum[m * WARP_N_TILES + n]);
                    mask(pack, kernels::bit_cast<uint32_t>(maskVal), m, maxRows - warpId * WARP_M);
                    store(&ptrlane[(m * WARP_N_TILES + n) * WARP_SIZE], pack);
                });
            });
        }

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, const Arguments &args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            assert(binfo.numBlocksN % 3 == 0);
            const int numBlocksQ = binfo.numBlocksN / 3;
            const bool is_q      = bn < numBlocksQ;
            const bool is_k      = !is_q && bn < numBlocksQ * 2;

            // bn is head_id (assume HEAD_DIM == WARP_N)
            int head_id, strideHead;
            if (is_q) {
                head_id    = bn;
                strideHead = args.strideHead_q;
            } else if (is_k) {
                head_id    = bn - numBlocksQ;
                strideHead = args.strideHead_k;
            } else {
                head_id    = bn - numBlocksQ * 2;
                strideHead = args.strideHead_v;
            }

            int block_offset = head_id * strideHead + bm * NUM_WARPS * WARP_M_TILES * WARP_N_TILES * WARP_SIZE;
            int maxRows      = args.actualM - bm * BLOCK_M;

            // static constexpr float neginf = -std::numeric_limits<float>::infinity();

            if (is_q) {
                apply(fpsum, args.out_q + block_offset, maxRows, pack_q, attn_half2_t(0.0f, 0.0f));
            } else if (is_k) {
                apply(fpsum, args.out_k + block_offset, maxRows, pack_k, attn_half2_t(NAN, NAN));
            } else {
                apply(fpsum, args.out_v + block_offset, maxRows, pack_v, attn_half2_t(0.0f, 0.0f));
            }
        }
    };

    struct EpilogueLiteLA {

        __device__ __forceinline__ static packed_f32psum_t
        mma_litela(packed_fpsum_t k, packed_fpsum_t v, packed_f32psum_t psum) {
            for (int i = 0; i < 4; i++) {
                k.data[i] = movmatrix(k.data[i]);
                v.data[i] = movmatrix(v.data[i]);
            }
            std::swap(v.data[1], v.data[2]);
            return mma_f16xf16_f32(v, k, psum);
        }

        static constexpr int LITELA_HEAD_DIM = 32;
        static constexpr int LITELA_K_TILES  = LITELA_HEAD_DIM / 16;
        static constexpr int LITELA_V_TILES  = LITELA_HEAD_DIM / 16;

        static constexpr int SHMEM_SIZE = NUM_WARPS * (LITELA_HEAD_DIM + 1) * (LITELA_HEAD_DIM + 8) * sizeof(float);

        // out_vk: [batch_size, num_heads, head_dim + 1, head_dim]
        __device__ __forceinline__ static void
        apply_litela(const BlockInfo binfo, fpsum_warp fpsum, float *out_vk, int num_blocks_per_batch) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            using vk_t = float[NUM_WARPS][LITELA_HEAD_DIM + 1][LITELA_HEAD_DIM + 8];
            extern __shared__ uint8_t shmem[];

            vk_t &shmem_vk = *reinterpret_cast<vk_t *>(shmem);

            static_assert(sizeof(vk_t) == SHMEM_SIZE);
            static_assert(WARP_N == BLOCK_N);
            assert(binfo.numBlocksN % 3 == 0);

            const int num_heads = binfo.numBlocksN / 3 * 2 * (WARP_N / (LITELA_HEAD_DIM * 2));
            const int batch_id  = binfo.bm / num_blocks_per_batch;

            for (int head_id = 0; head_id < WARP_N / (LITELA_HEAD_DIM * 2); head_id++) {
                const int global_head_id =
                    (binfo.bn - binfo.numBlocksN / 3) * (WARP_N / (LITELA_HEAD_DIM * 2)) + head_id;
                float *out_vk_current_head =
                    out_vk + (batch_id * num_heads + global_head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM;

                for (int i = laneId; i < sizeof(shmem_vk) / sizeof(float) / NUM_WARPS; i += WARP_SIZE) {
                    *((&shmem_vk[warpId][0][0]) + i) = 0;
                }
                __syncwarp();

                for (int tile_v = 0; tile_v < LITELA_V_TILES; tile_v++) {
                    for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                        packed_f32psum_t attn_sum = {0};
                        for (int i = 0; i < WARP_M_TILES; i++) {
                            packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                            packed_fpsum_t v = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 +
                                                     LITELA_HEAD_DIM / 16 + tile_v];
                            for (int j = 0; j < 4; j++) {
                                k.data[j] = __hmax2(k.data[j], half2_t(0, 0)); // relu
                            }
                            attn_sum = mma_litela(k, v, attn_sum);
                        }

                        const int row = tile_v * 16 + laneId / 4;
                        const int col = tile_k * 16 + laneId % 4 * 2;

                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 8][col + 0] = attn_sum.data[2];
                        shmem_vk[warpId][row + 8][col + 1] = attn_sum.data[3];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                        shmem_vk[warpId][row + 8][col + 8] = attn_sum.data[6];
                        shmem_vk[warpId][row + 8][col + 9] = attn_sum.data[7];
                    }
                }
                for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                    packed_f32psum_t attn_sum = {0};
                    for (int i = 0; i < WARP_M_TILES; i++) {
                        packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                        packed_fpsum_t v = {};
                        for (int j = 0; j < 4; j++) {
                            k.data[j] = __hmax2(k.data[j], half2_t(0, 0)); // relu
                        }
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            v.data[i] = half2_t(1, 1);
                        }
                        // if (laneId < 4) {
                        //     v.data[0] = half2_t(1, 1);
                        //     v.data[2] = half2_t(1, 1);
                        // }
                        // if (laneId % 4 == 0) {
                        //     v.data[0] = half2_t(1, 0);
                        //     v.data[1] = half2_t(1, 0);
                        // }
                        attn_sum = mma_litela(k, v, attn_sum);
                    }
                    const int row = LITELA_HEAD_DIM + laneId / 4;
                    const int col = tile_k * 16 + laneId % 4 * 2;

                    if (laneId < 4) {
                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                    }
                }
                __syncthreads();

                for (int i = warpId; i < LITELA_HEAD_DIM + 1; i += NUM_WARPS) {
                    for (int j = laneId; j < LITELA_HEAD_DIM; j += WARP_SIZE) {
                        float sum = 0;
                        for (int k = 0; k < NUM_WARPS; k++) {
                            sum += shmem_vk[k][i][j];
                        }
                        reduce_add(&out_vk_current_head[i * LITELA_HEAD_DIM + j], sum);
                    }
                }
                __syncthreads();
            }
        }

        struct Arguments {
            half_t *out_q;
            float *out_vk;
            int num_blocks_per_batch;
            int actualM;
        };

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, const Arguments &args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            if (bn < binfo.numBlocksN / 3) {
                fpsum = apply_act(fpsum, [](half_t x) { return __hmax(x, 0); }); // relu
                return EpilogueDefault()(binfo,
                                         fpsum,
                                         M,
                                         N / 3,
                                         K,
                                         typename EpilogueDefault::Arguments{
                                             .out     = args.out_q,
                                             .actualM = args.actualM,
                                             .actualN = N / 3,
                                         });
            }

            return apply_litela(binfo, fpsum, args.out_vk, args.num_blocks_per_batch);
        }

        // each thread block mults BlockSize*HEAD_DIM q and (HEAD_DIM+1)*HEAD_DIM vk, in-place writes back to q
        // q:   [batch_size, #blocks, block_size, #heads, HEAD_DIM]
        // vk:  [batch_size, #heads, HEAD_DIM+1, HEAD_DIM]
        struct vk_mul_q_kernel {
            static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
            // FIXME FIXME FIXME
            __device__ void operator()(half_t *q, const float *vk, float eps, int num_tokens) {
                const int block_id = blockIdx.x;
                const int head_id  = blockIdx.y;
                const int batch_id = blockIdx.z;

                const int num_blocks = gridDim.x;
                const int num_heads  = gridDim.y;
                const int block_size = blockDim.x;

                bool pred = block_id * block_size + threadIdx.x < num_tokens;

                half_t *localq =
                    &q[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads + head_id) *
                       LITELA_HEAD_DIM];
                const float *localvk = &vk[(batch_id * num_heads + head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM];
                // half_t *localout = &out[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads
                // + head_id) * LITELA_HEAD_DIM];

                using packed_q  = std::array<half_t, 8>;
                using packed_vk = std::array<float, 4>;

                half_t qblock[LITELA_HEAD_DIM];
                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    if (pred) {
                        *reinterpret_cast<packed_q *>(&qblock[i]) =
                            load(reinterpret_cast<const packed_q *>(&localq[i]));
                    }
                }

                float outblock[LITELA_HEAD_DIM + 1];
#pragma unroll
                for (int j = 0; j < LITELA_HEAD_DIM + 1; j++) {
                    outblock[j] = 0;
#pragma unroll
                    for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_vk) / sizeof(float)) {
                        packed_vk vkpack = load(reinterpret_cast<const packed_vk *>(&localvk[j * LITELA_HEAD_DIM + i]));
#pragma unroll
                        for (int k = 0; k < vkpack.size(); k++) {
                            outblock[j] += (float)qblock[i + k] * vkpack[k];
                        }
                    }
                }

                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    packed_q opack;
                    for (int k = 0; k < opack.size(); k++) {
                        opack[k] = __fdividef(outblock[i + k], outblock[LITELA_HEAD_DIM] + eps);
                    }
                    if (pred) {
                        store(reinterpret_cast<packed_q *>(&localq[i]), opack);
                    }
                }
            }
        };
    };

    template<typename Epilogue>
    struct test_epilogue_kernel {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr size_t SHMEM_PER_WARP =
            ceilDiv<size_t>(Base::template load_act_to_fpsum<false>::SHMEM_SIZE, 128) * 128;
        static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * NUM_WARPS;

        struct Arguments {
            const half_t *input;
            half_t *output;

            // aligned to BLOCK_M and BLOCK_N
            int M, N;
            int actualM, actualN;

            typename Epilogue::Arguments argsEpilogue;
        };

        __device__ __forceinline__ void operator()(Arguments args) {
            const BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .bn         = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            const int bm     = binfo.bm;
            const int bn     = binfo.bn;
            const int warpId = threadIdx.x / WARP_SIZE;

            const int m_offset = bm * BLOCK_M + warpId * WARP_M;
            const int n_offset = bn * BLOCK_N;

            extern __shared__ uint8_t shmem[];

            fpsum_warp fpsum;

            Base::template load_act_to_fpsum<false>()(args.input + m_offset * args.actualN + n_offset,
                                                      args.actualN,
                                                      args.actualM - m_offset,
                                                      args.actualN - n_offset,
                                                      fpsum,
                                                      shmem + warpId * SHMEM_PER_WARP);

            Epilogue()(binfo, fpsum, args.M, args.N, 0, args.argsEpilogue);

            EpilogueDefault()(binfo,
                              fpsum,
                              args.M,
                              args.N,
                              0,
                              typename EpilogueDefault::Arguments{
                                  .out     = args.output,
                                  .actualM = args.actualM,
                                  .actualN = args.actualN,
                              });
        }
    };
};

}; // namespace nunchaku::kernels
