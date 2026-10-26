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

#include "tensorrt_llm/kernels/glmMoeDsa/glmFusedAGemm.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cstdint>
#include <cuda.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

extern "C"
{
    __device__ uint32_t __nvvm_get_smem_pointer(void*);
}

namespace
{

using bf16_t = __nv_bfloat16;

// Warp w issues the tensor copies of its two register stages w * kFusedAMma8PaceNs after the copy of its shared-memory
// stage (1.75 us over the CTA's eight warps); the CTA's share of the L2 prefetch ranges follows once its staged tiles
// are in registers.
constexpr int kFusedAMma8PaceNs = 250;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ void initialize_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
    int thread_count = 1)                                  // Thread count expected to arrive/wait on this barrier
{
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(thread_count));
}

// Barrier wait
__device__ void wait_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
    int phase_bit)                                   // Current phase bit the barrier waiting to flip
{
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_ptr),
        "r"(phase_bit));
}

// Prefetches the ranges into L2 in 16 KB pieces. Piece q of the concatenated ranges goes to CTA q % gridDim.x: an SM
// issues bulk prefetches at a limited rate, so every CTA takes an equal share.
__device__ void prefetch_ranges_l2(FusedAL2Prefetch const& ranges, int local_tid, int thread_cnt)
{
    constexpr size_t piece_bytes = 16384;
    size_t piece_cnt[FusedAL2Prefetch::kMaxRanges];
    size_t total_piece_cnt = 0;
#pragma unroll
    for (int i = 0; i < FusedAL2Prefetch::kMaxRanges; i++)
    {
        piece_cnt[i] = (ranges.bytes[i] + piece_bytes - 1) / piece_bytes;
        total_piece_cnt += piece_cnt[i];
    }
    for (size_t q = blockIdx.x + static_cast<size_t>(local_tid) * gridDim.x; q < total_piece_cnt;
         q += static_cast<size_t>(thread_cnt) * gridDim.x)
    {
        size_t piece_idx = q;
        int range_idx = 0;
        while (piece_idx >= piece_cnt[range_idx])
        {
            piece_idx -= piece_cnt[range_idx];
            range_idx++;
        }
        size_t const offset = piece_idx * piece_bytes;
        size_t const remaining = ranges.bytes[range_idx] - offset;
        uint32_t const bytes = static_cast<uint32_t>(remaining < piece_bytes ? remaining : piece_bytes) & ~15u;
        if (bytes > 0)
        {
            char const* gmem_ptr = static_cast<char const*>(ranges.ptr[range_idx]) + offset;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(gmem_ptr), "r"(bytes) : "memory");
        }
    }
}

// Tensor copies completing on an mbarrier, ldmatrix and HMMA on shared addresses.
__device__ __forceinline__ void mma8_expect_tx(uint64_t* smem_barrier, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(__nvvm_get_smem_pointer(smem_barrier)),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void mma8_tma_load_2d(
    void* smem_dst, CUtensorMap const* tensor_map, int c0, int c1, uint64_t* smem_barrier)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
            "r"(__nvvm_get_smem_pointer(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(tensor_map)), "r"(c0), "r"(c1), "r"(__nvvm_get_smem_pointer(smem_barrier))
        : "memory");
}

__device__ __forceinline__ void mma8_tma_load_3d(
    void* smem_dst, CUtensorMap const* tensor_map, int c0, int c1, int c2, uint64_t* smem_barrier)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];\n" ::"r"(__nvvm_get_smem_pointer(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(tensor_map)), "r"(c0), "r"(c1), "r"(c2),
        "r"(__nvvm_get_smem_pointer(smem_barrier))
        : "memory");
}

__device__ __forceinline__ void mma8_ldsm_x4(uint32_t smem_addr, uint32_t (&r)[4])
{
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr));
}

__device__ __forceinline__ void mma8_hmma(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t mma8_globaltimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}
#endif

// The GLM-5 fused-A GEMM: gemm_k 6144, 16 weight rows per CTA, up to 8 tokens, with all eight warps issuing the
// HMMAs. output: [gemm_n, gemm_m]; tma_a: the tensor map of the weights [gemm_m, gemm_k]; tma_b: that of the
// activations [gemm_n, gemm_k] (makeFusedAMma8Map).
//
// Stage s is the k range [256 s, 256 s + 256); warp w owns stages w, w + 8 and w + 16 and waits only on its own data.
// Before the grid dependency wait (no kernel of the step writes the weights): the weight tile of stage w is copied into
// shared memory (one 3D tensor copy: four 64-wide k boxes of the CTA's 16 rows, 128-byte swizzle), the weights of
// stages w + 8 and w + 16 reach registers, and the l2_prefetch ranges go to L2. After the wait: the activation tile of
// each of the warp's stages (one 3D copy at 8 tokens, else four 2D boxes of gemm_n rows; tile rows past gemm_n only
// feed HMMA columns that are never stored). The dependent launch is triggered right after the wait.
//
// The register stages permute the k order inside each 32-wide k pair alike for both operands: lane (g = lane / 4,
// c = lane % 4) holds k [8 c, 8 c + 8) of weight rows g and g + 8 and reads k [8 c, 8 c + 8) of token g from the
// activation tile; HMMA step t of the pair takes words 2 t and 2 t + 1 of each. Each warp keeps two accumulators
// (16-wide k step i of a stage adds into i % 2); the eight warp partials are summed in warp order. The products are
// those of a plain GEMM, summed in this order in FP32 and rounded to BF16 once.
//
// The weights of stages w + 8 and w + 16 come in through shared memory, still before the wait: two more 3D tensor
// copies per warp into the space of the activation tiles (those land only after the wait) extended by 32 KB, read into
// registers once they land. Tensor copies leave the SM's load path to a co-resident kernel (the post-MoE all-reduce hop
// that fused-A follows). Lane 0 of warp w issues its two register-stage copies w * kFusedAMma8PaceNs after its stage-w
// copy (%globaltimer), and the l2_prefetch ranges go out only once the CTA's staged tiles are in registers: the CTAs'
// weights reach HBM over 7 * kFusedAMma8PaceNs ahead of the ranges, and the hop beside fused-A keeps more of the HBM
// bandwidth. Issue times do not change the outputs.
template <int gemm_m, bool l2_prefetch>
__global__ __launch_bounds__(256, 1) void fused_a_gemm_mma8_kernel(bf16_t* output, int gemm_n,
    FusedAL2Prefetch l2_prefetch_ranges, const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int gemm_k = kFusedAHiddenSize;
    constexpr int warp_cnt = 8;
    constexpr int tile_m = kFusedARowsPerCta;
    constexpr int tile_n = kFusedAMaxTokens;
    constexpr int tile_k = 256;
    constexpr int box_k = 64;
    constexpr int boxes = tile_k / box_k;
    constexpr int pair_cnt = tile_k / 32;
    constexpr int acc_cnt = 2;
    static_assert(gemm_k == 3 * warp_cnt * tile_k, "warp w owns stages w, w + 8 and w + 16");
    static_assert(gemm_m % tile_m == 0);
    constexpr uint32_t a_box_bytes = tile_m * box_k * sizeof(bf16_t);
    constexpr uint32_t b_box_bytes = tile_n * box_k * sizeof(bf16_t);
    constexpr uint32_t a_tile_bytes = boxes * a_box_bytes;
    constexpr uint32_t b_tile_bytes = boxes * b_box_bytes;

    // 1024-byte aligned (the 128-byte swizzle atoms): stage barriers, 8 weight tiles, 24 activation tiles. Before the
    // wait, the 16 register-stage weight tiles sit where the activation tiles land after it (warp w's stages w + 8 and
    // w + 16 at tiles 2 w and 2 w + 1), barrier 24 + w for warp w's two.
    extern __shared__ char smem[];
    char* smem_base = smem + ((1024u - (__nvvm_get_smem_pointer(smem) & 1023u)) & 1023u);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem_base);
    uint64_t* staged = full + 3 * warp_cnt;
    char* smem_a = smem_base + 1024;
    char* smem_b = smem_a + warp_cnt * a_tile_bytes;

    int const warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int const lane = threadIdx.x % 32;
    int const m0 = blockIdx.x * tile_m;
    int const g = lane / 4;
    int const c = lane % 4;
    uint32_t const b_tx_bytes = boxes * gemm_n * box_k * sizeof(bf16_t);
    bool const b_3d = gemm_n == tile_n;

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 3 * warp_cnt; i++)
        {
            // Stage s's barrier: the weight and the activation copies (stages 0-7), the activation copy (8-23).
            initialize_barrier(full + i, i < warp_cnt ? 2 : 1);
        }
        for (int i = 0; i < warp_cnt; i++)
        {
            initialize_barrier(staged + i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();

    if (lane == 0)
    {
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&tma_a)) : "memory");
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&tma_b)) : "memory");
        mma8_expect_tx(full + warp, a_tile_bytes);
        mma8_tma_load_3d(smem_a + warp * a_tile_bytes, &tma_a, 0, m0, warp * boxes, full + warp);
        mma8_expect_tx(staged + warp, 2 * a_tile_bytes);
        uint64_t const due = mma8_globaltimer() + static_cast<uint64_t>(warp) * kFusedAMma8PaceNs;
        while (mma8_globaltimer() < due)
        {
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            mma8_tma_load_3d(smem_b + (2 * warp + j) * a_tile_bytes, &tma_a, 0, m0, (warp + warp_cnt * (j + 1)) * boxes,
                staged + warp);
        }
    }
    uint4 a_reg[2][pair_cnt][2];
    // Every weight copy of the CTA is queued ahead of its share of the L2 prefetch list.
    __syncthreads();
    // The staged tiles into registers: lane (g, c) takes k [8 c, 8 c + 8) + 32 p of rows g and g + 8, i.e. 16-byte
    // chunk (p % 2) * 4 + c of box p / 2 (128-byte rows, 128-byte swizzle).
    wait_barrier(staged + warp, 0);
#pragma unroll
    for (int j = 0; j < 2; j++)
    {
        uint32_t const tile = __nvvm_get_smem_pointer(smem_b + (2 * warp + j) * a_tile_bytes);
#pragma unroll
        for (int p = 0; p < pair_cnt; p++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                uint4& v = a_reg[j][p][h];
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                             : "r"(tile + (p / 2) * a_box_bytes + (g + 8 * h) * 128 + ((((p % 2) * 4 + c) ^ g) << 4)));
            }
        }
    }
    // The activation copies after the wait overwrite the staged tiles.
    __syncthreads();
    if constexpr (l2_prefetch)
    {
        prefetch_ranges_l2(l2_prefetch_ranges, threadIdx.x, blockDim.x);
    }
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    if (lane == 0)
    {
        // The generic-proxy reads of the staged tiles before the tensor copies that overwrite them.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int j = 0; j < 3; j++)
        {
            int const stage = warp + warp_cnt * j;
            mma8_expect_tx(full + stage, b_tx_bytes);
            if (b_3d)
            {
                mma8_tma_load_3d(smem_b + stage * b_tile_bytes, &tma_b, 0, 0, stage * boxes, full + stage);
            }
            else
            {
#pragma unroll
                for (int q = 0; q < boxes; q++)
                {
                    mma8_tma_load_2d(smem_b + stage * b_tile_bytes + q * b_box_bytes, &tma_b,
                        stage * tile_k + q * box_k, 0, full + stage);
                }
            }
        }
    }

    float acc[acc_cnt][4] = {};
    // Stage w from shared memory. ldmatrix lanes: weight row (lane % 16) at k (lane / 16) * 8 of a 16-wide step;
    // token (lane % 8) at k (lane / 8) * 8 of a pair of steps.
    {
        wait_barrier(full + warp, 0);
        int const a_row = lane % 16;
        int const a_k = (lane / 16) * 8;
        int const b_row = lane % 8;
        int const b_k = (lane / 8) * 8;
        uint32_t const a_base = __nvvm_get_smem_pointer(smem_a + warp * a_tile_bytes);
        uint32_t const b_base = __nvvm_get_smem_pointer(smem_b + warp * b_tile_bytes);
#pragma unroll
        for (int s = 0; s < tile_k / 16; s += 2)
        {
            int const q = s / 4;
            int const kk = (s % 4) * 16;
            uint32_t b[4];
            mma8_ldsm_x4(b_base + q * b_box_bytes + b_row * 128 + ((((kk + b_k) >> 3) ^ (b_row & 7)) << 4), b);
#pragma unroll
            for (int t = 0; t < 2; t++)
            {
                uint32_t a[4];
                mma8_ldsm_x4(
                    a_base + q * a_box_bytes + a_row * 128 + ((((kk + 16 * t + a_k) >> 3) ^ (a_row & 7)) << 4), a);
                mma8_hmma(acc[(s + t) % acc_cnt], a, b[2 * t], b[2 * t + 1]);
            }
        }
    }
    // Stages w + 8 and w + 16: weights from registers, activations from shared memory in the same permuted k order.
#pragma unroll
    for (int j = 0; j < 2; j++)
    {
        int const stage = warp + warp_cnt * (j + 1);
        wait_barrier(full + stage, 0);
        uint32_t const b_base = __nvvm_get_smem_pointer(smem_b + stage * b_tile_bytes);
#pragma unroll
        for (int p = 0; p < pair_cnt; p++)
        {
            int const chunk = (p % 2) * 4 + c;
            uint4 bq;
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(bq.x), "=r"(bq.y), "=r"(bq.z), "=r"(bq.w)
                         : "r"(b_base + (p / 2) * b_box_bytes + g * 128 + ((chunk ^ (g & 7)) << 4)));
            uint32_t const a0[4] = {a_reg[j][p][0].x, a_reg[j][p][1].x, a_reg[j][p][0].y, a_reg[j][p][1].y};
            uint32_t const a1[4] = {a_reg[j][p][0].z, a_reg[j][p][1].z, a_reg[j][p][0].w, a_reg[j][p][1].w};
            mma8_hmma(acc[(2 * p) % acc_cnt], a0, bq.x, bq.y);
            mma8_hmma(acc[(2 * p + 1) % acc_cnt], a1, bq.z, bq.w);
        }
    }

    // Warp partials into the warp's consumed stage w + 8 activation tile (no copy is pending there), then the sum
    // over the warps in warp order. Output (row r, token n) is register (n % 2) + 2 (r / 8) of lane
    // (r % 8) * 4 + n / 2.
    __syncwarp();
    float* partial = reinterpret_cast<float*>(smem_b + (warp + warp_cnt) * b_tile_bytes);
#pragma unroll
    for (int r = 0; r < 4; r++)
    {
        partial[lane * 4 + r] = acc[0][r] + acc[1][r];
    }
    __syncthreads();
    if (threadIdx.x < tile_m * tile_n)
    {
        int const r = threadIdx.x % tile_m;
        int const n = threadIdx.x / tile_m;
        int const src = ((r % 8) * 4 + n / 2) * 4 + (n % 2) + 2 * (r / 8);
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < warp_cnt; w++)
        {
            v += reinterpret_cast<float const*>(smem_b + (w + warp_cnt) * b_tile_bytes)[src];
        }
        if (n < gemm_n)
        {
            output[static_cast<int64_t>(n) * gemm_m + m0 + r] = __float2bfloat16_rn(v);
        }
    }
#endif
}

// Tensor map of a row-major BF16 [rows, cols] matrix with the 128-byte swizzle: dims (k within a 64-wide box, row, box)
// and box (64, box_rows, 4) when box3d (one copy loads [4 boxes][box_rows][64]), else dims (k, row) and box
// (64, box_rows).
CUtensorMap makeFusedAMma8Map(void const* ptr, int rows, int cols, int box_rows, bool box3d)
{
    CUtensorMap map;
    cuuint64_t const dims3[3] = {64, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(cols / 64)};
    cuuint64_t const strides3[2] = {static_cast<cuuint64_t>(cols) * sizeof(bf16_t), 64 * sizeof(bf16_t)};
    cuuint32_t const box3[3] = {64, static_cast<cuuint32_t>(box_rows), 4};
    cuuint64_t const dims2[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    cuuint64_t const strides2[1] = {static_cast<cuuint64_t>(cols) * sizeof(bf16_t)};
    cuuint32_t const box2[2] = {64, static_cast<cuuint32_t>(box_rows)};
    cuuint32_t const elem_strides[3] = {1, 1, 1};
    TLLM_CU_CHECK(tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(&map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, box3d ? 3 : 2, const_cast<void*>(ptr), box3d ? dims3 : dims2,
        box3d ? strides3 : strides2, box3d ? box3 : box2, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

// weight: [gemm_m, 6144]; hiddenStates: [numTokens, 6144], numTokens in [1, 8].
template <int gemm_m>
void launchFusedAMma8(bf16_t* output, bf16_t const* hiddenStates, bf16_t const* weight, int numTokens,
    FusedAL2Prefetch const* l2Prefetch, cudaStream_t stream)
{
    // Alignment slack, stage barriers, 8 weight tiles of 16 x 256, and the space of the 24 activation tiles of 8 x 256
    // extended to the 16 register-stage weight tiles of 16 x 256.
    constexpr int kSmemBytes = 1024 + 1024 + 8 * 16 * 256 * sizeof(bf16_t) + 16 * 16 * 256 * sizeof(bf16_t);
    CUtensorMap const tmaA = makeFusedAMma8Map(weight, gemm_m, kFusedAHiddenSize, kFusedARowsPerCta, true);
    CUtensorMap const tmaB
        = makeFusedAMma8Map(hiddenStates, numTokens, kFusedAHiddenSize, numTokens, numTokens == kFusedAMaxTokens);
    auto* kernel
        = l2Prefetch != nullptr ? &fused_a_gemm_mma8_kernel<gemm_m, true> : &fused_a_gemm_mma8_kernel<gemm_m, false>;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    cudaLaunchConfig_t config;
    config.gridDim = dim3(gemm_m / kFusedARowsPerCta, 1, 1);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel, output, numTokens, l2Prefetch != nullptr ? *l2Prefetch : FusedAL2Prefetch{}, tmaA, tmaB));
}

} // namespace

void invokeFusedAGemm(__nv_bfloat16* output, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight,
    int numTokens, int outFeatures, FusedAL2Prefetch const* l2Prefetch, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(numTokens >= 1 && numTokens <= kFusedAMaxTokens,
        "The GLM fused-A GEMM takes 1 to %d tokens, got %d", kFusedAMaxTokens, numTokens);
    switch (outFeatures)
    {
    case kFusedAOutFeatures:
        launchFusedAMma8<kFusedAOutFeatures>(output, hiddenStates, weight, numTokens, l2Prefetch, stream);
        break;
    case kFusedAIndexerOutFeatures:
        launchFusedAMma8<kFusedAIndexerOutFeatures>(output, hiddenStates, weight, numTokens, l2Prefetch, stream);
        break;
    default:
        TLLM_THROW("The GLM fused-A GEMM serves %d or %d output features, got %d", kFusedAOutFeatures,
            kFusedAIndexerOutFeatures, outFeatures);
    }
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
