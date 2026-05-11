/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 */

#include "fusedCatFp4.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Constants (fixed for DSV3.2 FP4 indexer: head_dim=128, per-block-32 quant).
//
// INV_FP4_E2M1_MAX / MIN_AMAX mirror DeepGEMM's per_token_cast_to_fp4 reference
// (use_ue8m0=True, gran_k=32, use_packed_ue8m0=True) at
// tensorrt_llm/deep_gemm/utils/math.py. Keep in sync with that helper when
// bumping DeepGEMM — test_fused_cat_fp4_matches_deepgemm is the guard.
constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int ELEMS_PER_THREAD = 4;                           // 128 / 32 = 4 elements per thread.
constexpr int ROWS_PER_BLOCK = 8;                             // 8 rows per CTA (mirrors fusedCatFp8).
constexpr int FP4_BLOCK_SIZE = 32;                            // One UE8M0 scale per 32 elements.
constexpr int LANES_PER_FP4_BLOCK = 8;                        // 32 elements / 4 per lane = 8 lanes.
constexpr int FP4_BLOCKS_PER_ROW = HEAD_DIM / FP4_BLOCK_SIZE; // = 4.
constexpr float INV_FP4_E2M1_MAX = 1.0f / 6.0f;               // 1 / max representable FP4 E2M1 magnitude (6.0).
constexpr float MIN_AMAX = 1.0e-12f;                          // Floor on amax to keep ratio normal in ceil_to_ue8m0.

/// Helper union for vectorized BF16 loads (4 BF16 values = 8 bytes).
union BF16x4
{
    int2 vec;
    __nv_bfloat162 bf16x2[2];
};

/// FP4 E2M1 quantize a single scaled value.
/// Returns a 4-bit code matching DeepGEMM's table:
///   magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
///   sign bit 3 set iff value < 0 and magnitude code != 0.
__device__ __forceinline__ uint32_t quantizeFp4E2M1(float scaled)
{
    float ax = fminf(fabsf(scaled), 6.0f);
    // Strict `>` matches Triton reference and DeepGEMM (boundary values round down).
    uint32_t idx = static_cast<uint32_t>(
        (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f));
    uint32_t code = idx & 0x7u;
    uint32_t sign = (scaled < 0.0f && idx != 0u) ? 1u : 0u;
    return code | (sign << 3);
}

/// Fused kernel: cat + per-block-32 FP4 E2M1 quantization (UE8M0 scales).
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)
/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
///
/// Thread layout within a warp (handles one 128-element row):
///   - Thread t handles elements [4t, 4t+1, 4t+2, 4t+3].
///   - FP4 block g = t / 8 (group of 8 lanes); lane-in-group = t % 8.
///   - Per-block amax reduces across the 8 lanes of a group via shuffle (offsets 1, 2, 4).
///   - UE8M0 scale bit-trick runs on every lane; all 8 lanes in a group see
///     identical amax and thus produce the same scale.
///   - Output bytes: lane t writes packed[row*64 + 2t .. +1].
///   - Output scale: lane 0 gathers the 4 UE8M0 exponent bytes from lanes
///     {0, 8, 16, 24} and writes one int32.
__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedCatFp4Kernel(int8_t* __restrict__ packed_out,
    int32_t* __restrict__ scale_out, __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t pe_row_stride, int32_t nope_row_stride)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    // ---- Stage 1: Load + Concat (vectorized 8-byte loads) ----
    // pe_dim is guaranteed a multiple of ELEMS_PER_THREAD by the host check,
    // so each thread's 4 elements come entirely from pe or entirely from nope.
    float v0, v1, v2, v3;
    {
        int base = lane * ELEMS_PER_THREAD;
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_row_stride;
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_row_stride;

        bool from_pe = (base < pe_dim);
        __nv_bfloat16 const* src = from_pe ? pe_row : nope_row;
        int col = from_pe ? base : (base - pe_dim);

        BF16x4 loaded;
        loaded.vec = *reinterpret_cast<int2 const*>(src + col);

        float2 f0 = __bfloat1622float2(loaded.bf16x2[0]);
        float2 f1 = __bfloat1622float2(loaded.bf16x2[1]);
        v0 = f0.x;
        v1 = f0.y;
        v2 = f1.x;
        v3 = f1.y;
    }

    // ---- Stage 2: Per-block-32 amax reduction (within a group of 8 lanes) ----
    // __shfl_xor_sync with offsets 1, 2, 4 stays inside each group of 8 lanes,
    // so after three rounds all 8 lanes in a group share the same group-max.
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = local_max;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
    amax = fmaxf(amax, MIN_AMAX);

    // ---- Stage 3: UE8M0 scale via IEEE 754 bit manipulation ----
    // scale = 2^ceil(log2(amax / 6)). With MIN_AMAX = 1e-12, ratio is always
    // normal (exp >= 1), so no explicit clamp needed.
    float ratio = amax * INV_FP4_E2M1_MAX;
    uint32_t bits = __float_as_uint(ratio);
    uint32_t exp_bits = bits & 0x7F800000u;
    if ((bits & 0x007FFFFFu) != 0u)
    {
        exp_bits += 0x00800000u;
    }
    float scale = __uint_as_float(exp_bits);

    // ---- Stage 4: FP4 E2M1 quantize ----
    // Use IEEE float division (compiles to div.rn.f32) for bit-exact parity
    // with DeepGEMM's `x / scale` reference. Reciprocal approximation could
    // drift at exact midpoints even though the relative error is below the
    // FP4 resolution on average.
    uint32_t c0 = quantizeFp4E2M1(v0 / scale);
    uint32_t c1 = quantizeFp4E2M1(v1 / scale);
    uint32_t c2 = quantizeFp4E2M1(v2 / scale);
    uint32_t c3 = quantizeFp4E2M1(v3 / scale);

    // ---- Stage 5: Nibble pack (even → low, odd → high) ----
    uint8_t byte0 = static_cast<uint8_t>(c0 | (c1 << 4));
    uint8_t byte1 = static_cast<uint8_t>(c2 | (c3 << 4));

    // ---- Stage 6: Store packed bytes (2 bytes per lane) ----
    int base_out = row * (HEAD_DIM / 2) + lane * 2;
    packed_out[base_out + 0] = static_cast<int8_t>(byte0);
    packed_out[base_out + 1] = static_cast<int8_t>(byte1);

    // ---- Stage 7: Pack the 4 UE8M0 exponent bytes into one int32 ----
    // All 8 lanes in a group share the same scale, so every lane has a valid
    // exp byte for its block. Shuffle the leader-lane values (0, 8, 16, 24)
    // to lane 0 and combine little-endian.
    uint32_t my_exp = (__float_as_uint(scale) >> 23) & 0xFFu;
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, my_exp, 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, my_exp, 8);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, my_exp, 16);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, my_exp, 24);
    if (lane == 0)
    {
        uint32_t packed_scale = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        scale_out[row] = static_cast<int32_t>(packed_scale);
    }
}

} // anonymous namespace

void invokeFusedCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedCatFp4: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim, "fusedCatFp4: pe_dim (%d) + nope_dim (%d) != head_dim (%d)",
        pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: pe_dim (%d) must be a multiple of %d for vectorized access", pe_dim, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(
        pe_row_stride >= pe_dim, "fusedCatFp4: pe_row_stride (%d) must be >= pe_dim (%d)", pe_row_stride, pe_dim);
    TLLM_CHECK_WITH_INFO(nope_row_stride >= nope_dim, "fusedCatFp4: nope_row_stride (%d) must be >= nope_dim (%d)",
        nope_row_stride, nope_dim);
    TLLM_CHECK_WITH_INFO(pe_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: pe_row_stride (%d) must be a multiple of %d for aligned vectorized access", pe_row_stride,
        ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(nope_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: nope_row_stride (%d) must be a multiple of %d for aligned vectorized access", nope_row_stride,
        ELEMS_PER_THREAD);

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK); // 256 threads per block

    fusedCatFp4Kernel<<<grid, block, 0, stream>>>(
        packed_out, scale_out, pe, nope, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride);

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
