/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/kernels/inverseRopeFp8QuantKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

// Fixed by the DSv4 inverse-RoPE op contract (matches the Triton reference):
//   * QUANT_GROUP_SIZE = 128
//   * ROPE_DIM         = 64   (with HALF_ROPE = 32, NEOX layout)
//   * Rope occupies the LAST ROPE_DIM elements of the head (i.e. the second
//     half of the final quant chunk). Earlier chunks are pure nope.
//   * Per-lane-per-chunk: 4 fp32 elements, full warp covers all 128 elts.
constexpr int kQuantGroupSize = 128;
constexpr int kRopeDim = 64;
constexpr int kHalfRope = kRopeDim / 2;                             // 32
constexpr int kCsStride = 2 * kHalfRope;                            // 64 fp32 per position (cos block + sin block)
constexpr int kRopeStartInChunk = kQuantGroupSize - kRopeDim;       // 64; rope is [64,128) of last chunk
constexpr int kWarpSize = 32;
constexpr int kEltsPerThreadPerChunk = kQuantGroupSize / kWarpSize; // 4
constexpr float kFp8Max = 448.0f;
constexpr float kFp8InvMax = 1.0f / 448.0f;
constexpr float kEps = 1e-12f;

// Per-warp per-chunk processing. Each chunk = 128 elements = 32 lanes × 4 elts.
// Caller passes 4 bf16 inputs (already loaded into registers), and the
// chunk-relative `is_rope_chunk` flag controls whether the second half of
// the chunk gets the inverse-NEOX rotation. Within the rope chunk the
// elements [kRopeStartInChunk, 128) carry the rope, partner across the
// rope halves is lane ^ 8.
//
// On exit: x_quant_lo / x_quant_hi hold the packed fp8x2 outputs to store
// for this chunk, and `scale_out` is the per-chunk fp32 scale (written
// only by lane 0).
template <bool IS_NEOX>
__device__ __forceinline__ void process_chunk(__nv_bfloat162 in_lo, __nv_bfloat162 in_hi,
    float const* __restrict__ cs_row, bool is_rope_chunk, __nv_fp8x2_e4m3& out_lo, __nv_fp8x2_e4m3& out_hi,
    float& scale_out)
{
    int const lane = threadIdx.x & 31;
    int const e0_in_chunk = lane * kEltsPerThreadPerChunk; // [0..128) in steps of 4

    float x[4];
    x[0] = __bfloat162float(in_lo.x);
    x[1] = __bfloat162float(in_lo.y);
    x[2] = __bfloat162float(in_hi.x);
    x[3] = __bfloat162float(in_hi.y);

    // NEOX layout: partner across the two rope halves = `lane XOR 8`. All
    // 32 lanes must issue the shuffle (mask=0xffffffff). Skipped under
    // GPT-J / interleaved layout where the partner is intra-lane.
    unsigned part_lo_u = 0u, part_hi_u = 0u;
    if (IS_NEOX)
    {
        int const partner_lane = lane ^ 8;
        unsigned in_lo_u = *reinterpret_cast<unsigned*>(&in_lo);
        unsigned in_hi_u = *reinterpret_cast<unsigned*>(&in_hi);
        part_lo_u = __shfl_sync(0xFFFFFFFFu, in_lo_u, partner_lane);
        part_hi_u = __shfl_sync(0xFFFFFFFFu, in_hi_u, partner_lane);
    }

    bool const lane_in_rope = is_rope_chunk && (lane >= 16);

    if (IS_NEOX && lane_in_rope)
    {
        // Inverse-NEOX rope: lanes 16..23 own the first rope half
        // (rope_local 0..31, chunk-local 64..95); lanes 24..31 own the
        // second half (rope_local 32..63, chunk-local 96..127).
        bool const lane_in_first_half = (lane < 24);
        __nv_bfloat162 p2_lo = *reinterpret_cast<__nv_bfloat162*>(&part_lo_u);
        __nv_bfloat162 p2_hi = *reinterpret_cast<__nv_bfloat162*>(&part_hi_u);

        float xp[4];
        xp[0] = __bfloat162float(p2_lo.x);
        xp[1] = __bfloat162float(p2_lo.y);
        xp[2] = __bfloat162float(p2_hi.x);
        xp[3] = __bfloat162float(p2_hi.y);

        int const cs_base = e0_in_chunk - kRopeStartInChunk - (lane_in_first_half ? 0 : kHalfRope);
        float const sign = lane_in_first_half ? 1.0f : -1.0f;
        // 16-byte (float4) coalesced load -- cs_base is always a multiple of 4.
        float4 const cos4 = *reinterpret_cast<float4 const*>(cs_row + cs_base);
        float4 const sin4 = *reinterpret_cast<float4 const*>(cs_row + kHalfRope + cs_base);
        x[0] = x[0] * cos4.x + sign * sin4.x * xp[0];
        x[1] = x[1] * cos4.y + sign * sin4.y * xp[1];
        x[2] = x[2] * cos4.z + sign * sin4.z * xp[2];
        x[3] = x[3] * cos4.w + sign * sin4.w * xp[3];
    }
    else if (!IS_NEOX && lane_in_rope)
    {
        // Interleaved (GPT-J) inverse RoPE -- partners are adjacent pairs
        // (x[2i], x[2i+1]) within the rope segment. Each lane's 4 elements
        // form two intra-lane pairs (x[0],x[1]) and (x[2],x[3]); no
        // cross-lane shuffle is needed.
        //   for rope_local r:  cs_idx = r >> 1
        //   r even: new = x[r]*cos[cs_idx] + x[r+1]*sin[cs_idx]
        //   r odd : new = x[r]*cos[cs_idx] - x[r-1]*sin[cs_idx]
        // Lane t in [16,31] owns rope_local in [(t-16)*4, (t-16)*4+4), so
        // cs_idx ∈ {(t-16)*2, (t-16)*2 + 1} for the two intra-lane pairs.
        int const cs_base = (lane - 16) * 2;
        float2 const cos2 = *reinterpret_cast<float2 const*>(cs_row + cs_base);
        float2 const sin2 = *reinterpret_cast<float2 const*>(cs_row + kHalfRope + cs_base);
        float const x0_new = x[0] * cos2.x + x[1] * sin2.x;
        float const x1_new = x[1] * cos2.x - x[0] * sin2.x;
        float const x2_new = x[2] * cos2.y + x[3] * sin2.y;
        float const x3_new = x[3] * cos2.y - x[2] * sin2.y;
        x[0] = x0_new;
        x[1] = x1_new;
        x[2] = x2_new;
        x[3] = x3_new;
    }

    // Per-chunk warp absmax (128 elements).
    float local_max = fmaxf(fabsf(x[0]), fmaxf(fabsf(x[1]), fmaxf(fabsf(x[2]), fabsf(x[3]))));
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFFu, local_max, mask));
    }
    float const block_max = fmaxf(local_max, kEps);
    scale_out = block_max * kFp8InvMax;
    float const inv_scale = __fdividef(kFp8Max, block_max);

    out_lo = __nv_fp8x2_e4m3(float2{x[0] * inv_scale, x[1] * inv_scale});
    out_hi = __nv_fp8x2_e4m3(float2{x[2] * inv_scale, x[3] * inv_scale});
}

// Baseline 1-warp-per-token variant. Each warp handles one (head, token)
// pair and emits CHUNKS_PER_HEAD fp8 chunks + scales. The kernel hardcodes
// the DSv4 layout: rope_dim=64 lives in the second half of the last chunk.
template <int BLOCK_TOKENS, int CHUNKS_PER_HEAD, bool IS_NEOX>
__global__ __launch_bounds__(BLOCK_TOKENS* kWarpSize) void inverseRopeFp8QuantKernel(
    __nv_bfloat16 const* __restrict__ o_ptr,              //
    int64_t const* __restrict__ positions_ptr,            //
    float const* __restrict__ cs_cache_ptr,               //
    __nv_fp8_e4m3* __restrict__ fp8_ptr,                  //
    float* __restrict__ scale_ptr,                        //
    int num_tokens, int scale_buf_m, int heads_per_group, //
    int o_stride_token, int o_stride_head,                //
    int fp8_stride_group, int fp8_stride_token,           //
    int scale_stride_group, int scale_stride_k)
{
    constexpr int HEAD_DIM = CHUNKS_PER_HEAD * kQuantGroupSize;
    int const warp_id = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const e0_in_chunk = lane * kEltsPerThreadPerChunk;
    int const pid_token = blockIdx.x * BLOCK_TOKENS + warp_id;
    int const head_idx = blockIdx.y; // global head: [0, num_heads)
    int const g_idx = head_idx / heads_per_group;
    int const h_in_g = head_idx - g_idx * heads_per_group;
    int const qb_base = h_in_g * CHUNKS_PER_HEAD;

    if (pid_token >= scale_buf_m)
        return;

    if (pid_token >= num_tokens)
    {
        // Pad row in [num_tokens, pad_up(num_tokens, 4)): zero out all
        // CHUNKS_PER_HEAD scale slots for this (group, head_in_group).
        if (lane == 0)
        {
            float* sc_grp = scale_ptr + static_cast<long>(g_idx) * scale_stride_group;
#pragma unroll
            for (int c = 0; c < CHUNKS_PER_HEAD; ++c)
            {
                sc_grp[(qb_base + c) * scale_stride_k + pid_token] = 0.0f;
            }
        }
        return;
    }

    long const pos = positions_ptr[pid_token];
    // Input layout: [num_tokens, num_heads, head_dim] flat — global head_idx works.
    auto const* in_row
        = o_ptr + static_cast<long>(pid_token) * o_stride_token + static_cast<long>(head_idx) * o_stride_head;
    // Output layout: fp8_buf [n_groups, num_tokens, heads_per_group * head_dim]
    // and scale_buf [n_groups, heads_per_group*chunks, pad_up(T,4)] — both with
    // n_groups outermost (the BMM consumer's expected (G, T, K) view).
    auto* fp8_row = fp8_ptr + static_cast<long>(g_idx) * fp8_stride_group
        + static_cast<long>(pid_token) * fp8_stride_token + static_cast<long>(h_in_g) * HEAD_DIM;
    float* sc_grp = scale_ptr + static_cast<long>(g_idx) * scale_stride_group;
    auto const* cs_row = cs_cache_ptr + pos * kCsStride;

#pragma unroll
    for (int c = 0; c < CHUNKS_PER_HEAD; ++c)
    {
        // Load this chunk's 4 bf16 per lane.
        auto const* in_pair = reinterpret_cast<__nv_bfloat162 const*>(in_row + c * kQuantGroupSize + e0_in_chunk);
        __nv_bfloat162 in_lo = in_pair[0];
        __nv_bfloat162 in_hi = in_pair[1];

        bool const is_rope = (c == CHUNKS_PER_HEAD - 1);
        __nv_fp8x2_e4m3 out_lo, out_hi;
        float scale;
        process_chunk<IS_NEOX>(in_lo, in_hi, cs_row, is_rope, out_lo, out_hi, scale);

        auto* out_pair = reinterpret_cast<__nv_fp8x2_e4m3*>(fp8_row + c * kQuantGroupSize + e0_in_chunk);
        out_pair[0] = out_lo;
        out_pair[1] = out_hi;

        if (lane == 0)
        {
            sc_grp[(qb_base + c) * scale_stride_k + pid_token] = scale;
        }
    }
}

// Software-pipelined variant: each warp processes TOKENS_PER_WARP tokens
// with explicit double-buffered load/compute interleaving. Hides L1
// scoreboard stalls by overlapping the next iter's input/cs LDGs with the
// current iter's compute+store. TPW=2 is the sweet spot — TPW>=4 spills
// registers and drops occupancy.
template <int BLOCK_TOKENS, int CHUNKS_PER_HEAD, int TOKENS_PER_WARP, bool IS_NEOX>
__global__ __launch_bounds__(BLOCK_TOKENS* kWarpSize) void inverseRopeFp8QuantKernelPipelined(
    __nv_bfloat16 const* __restrict__ o_ptr,              //
    int64_t const* __restrict__ positions_ptr,            //
    float const* __restrict__ cs_cache_ptr,               //
    __nv_fp8_e4m3* __restrict__ fp8_ptr,                  //
    float* __restrict__ scale_ptr,                        //
    int num_tokens, int scale_buf_m, int heads_per_group, //
    int o_stride_token, int o_stride_head,                //
    int fp8_stride_group, int fp8_stride_token,           //
    int scale_stride_group, int scale_stride_k)
{
    constexpr int HEAD_DIM = CHUNKS_PER_HEAD * kQuantGroupSize;
    int const warp_id = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const e0_in_chunk = lane * kEltsPerThreadPerChunk;
    int const head_idx = blockIdx.y;
    int const g_idx = head_idx / heads_per_group;
    int const h_in_g = head_idx - g_idx * heads_per_group;
    int const qb_base = h_in_g * CHUNKS_PER_HEAD;
    int const pid_token_base = blockIdx.x * (BLOCK_TOKENS * TOKENS_PER_WARP) + warp_id * TOKENS_PER_WARP;

    // ---- Stage 1: issue all input + position LDGs up front --------------
    __nv_bfloat162 in_lo_arr[TOKENS_PER_WARP][CHUNKS_PER_HEAD];
    __nv_bfloat162 in_hi_arr[TOKENS_PER_WARP][CHUNKS_PER_HEAD];
    long pos_arr[TOKENS_PER_WARP];
    bool valid[TOKENS_PER_WARP];
    bool in_range[TOKENS_PER_WARP];

#pragma unroll
    for (int t = 0; t < TOKENS_PER_WARP; ++t)
    {
        int const pid = pid_token_base + t;
        in_range[t] = pid < scale_buf_m;
        valid[t] = pid < num_tokens;
        if (valid[t])
        {
            pos_arr[t] = positions_ptr[pid];
            auto const* in_row
                = o_ptr + static_cast<long>(pid) * o_stride_token + static_cast<long>(head_idx) * o_stride_head;
#pragma unroll
            for (int c = 0; c < CHUNKS_PER_HEAD; ++c)
            {
                auto const* in_pair
                    = reinterpret_cast<__nv_bfloat162 const*>(in_row + c * kQuantGroupSize + e0_in_chunk);
                in_lo_arr[t][c] = in_pair[0];
                in_hi_arr[t][c] = in_pair[1];
            }
        }
    }

    // ---- Stage 2: compute + store, one token at a time -----------------
#pragma unroll
    for (int t = 0; t < TOKENS_PER_WARP; ++t)
    {
        int const pid = pid_token_base + t;
        if (!in_range[t])
            continue;

        float* sc_grp = scale_ptr + static_cast<long>(g_idx) * scale_stride_group;
        if (!valid[t])
        {
            if (lane == 0)
            {
#pragma unroll
                for (int c = 0; c < CHUNKS_PER_HEAD; ++c)
                {
                    sc_grp[(qb_base + c) * scale_stride_k + pid] = 0.0f;
                }
            }
            continue;
        }

        auto const* cs_row = cs_cache_ptr + pos_arr[t] * kCsStride;
        auto* fp8_row = fp8_ptr + static_cast<long>(g_idx) * fp8_stride_group
            + static_cast<long>(pid) * fp8_stride_token + static_cast<long>(h_in_g) * HEAD_DIM;

#pragma unroll
        for (int c = 0; c < CHUNKS_PER_HEAD; ++c)
        {
            bool const is_rope = (c == CHUNKS_PER_HEAD - 1);
            __nv_fp8x2_e4m3 out_lo, out_hi;
            float scale;
            process_chunk<IS_NEOX>(in_lo_arr[t][c], in_hi_arr[t][c], cs_row, is_rope, out_lo, out_hi, scale);

            auto* out_pair = reinterpret_cast<__nv_fp8x2_e4m3*>(fp8_row + c * kQuantGroupSize + e0_in_chunk);
            out_pair[0] = out_lo;
            out_pair[1] = out_hi;

            if (lane == 0)
            {
                sc_grp[(qb_base + c) * scale_stride_k + pid] = scale;
            }
        }
    }
}

template <int CHUNKS_PER_HEAD, bool IS_NEOX>
inline void dispatchByM(int num_tokens, int num_heads, int heads_per_group, int scale_buf_m,              //
    __nv_bfloat16 const* o_p, int64_t const* pos_p, float const* cs_p, __nv_fp8_e4m3* fp8_p, float* sc_p, //
    int o_stride_token, int o_stride_head, int fp8_stride_group, int fp8_stride_token,                    //
    int scale_stride_group, int scale_stride_k, cudaStream_t stream)
{
    // BTM=4 with TPW=2 above M=4096, otherwise baseline. Tuned on B200.
    if (num_tokens >= 4096)
    {
        constexpr int BTM = 4;
        constexpr int TPW = 2;
        int const tokens_per_cta = BTM * TPW;
        int const grid_x = (scale_buf_m + tokens_per_cta - 1) / tokens_per_cta;
        dim3 grid(grid_x, num_heads);
        dim3 block(BTM * kWarpSize);
        inverseRopeFp8QuantKernelPipelined<BTM, CHUNKS_PER_HEAD, TPW, IS_NEOX><<<grid, block, 0, stream>>>( //
            o_p, pos_p, cs_p, fp8_p, sc_p,                                                                  //
            num_tokens, scale_buf_m, heads_per_group,                                                       //
            o_stride_token, o_stride_head, fp8_stride_group, fp8_stride_token,                              //
            scale_stride_group, scale_stride_k);
    }
    else
    {
        constexpr int BTM = 4;
        int const grid_x = (scale_buf_m + BTM - 1) / BTM;
        dim3 grid(grid_x, num_heads);
        dim3 block(BTM * kWarpSize);
        inverseRopeFp8QuantKernel<BTM, CHUNKS_PER_HEAD, IS_NEOX><<<grid, block, 0, stream>>>( //
            o_p, pos_p, cs_p, fp8_p, sc_p,                                                    //
            num_tokens, scale_buf_m, heads_per_group,                                         //
            o_stride_token, o_stride_head, fp8_stride_group, fp8_stride_token,                //
            scale_stride_group, scale_stride_k);
    }
}

} // namespace

void invokeInverseRopeFp8Quant(void const* o, //
    void const* positions,                    //
    void const* cos_sin_cache,                //
    void* fp8_out,                            //
    void* scale_out,                          //
    int num_tokens,                           //
    int num_heads,                            //
    int heads_per_group,                      //
    int chunks_per_head,                      //
    bool is_neox,                             //
    int scale_buf_m,                          //
    int o_stride_token,                       //
    int o_stride_head,                        //
    int fp8_stride_group,                     //
    int fp8_stride_token,                     //
    int scale_stride_group,                   //
    int scale_stride_k,                       //
    cudaStream_t stream)
{
    auto const* o_p = reinterpret_cast<__nv_bfloat16 const*>(o);
    auto const* pos_p = reinterpret_cast<int64_t const*>(positions);
    auto const* cs_p = reinterpret_cast<float const*>(cos_sin_cache);
    auto* fp8_p = reinterpret_cast<__nv_fp8_e4m3*>(fp8_out);
    auto* sc_p = reinterpret_cast<float*>(scale_out);

#define TRTLLM_INV_ROPE_DISPATCH_CHUNK(CHUNKS, NEOX)                                                                   \
    dispatchByM<(CHUNKS), (NEOX)>(num_tokens, num_heads, heads_per_group, scale_buf_m, o_p, pos_p, cs_p, fp8_p, sc_p,  \
        o_stride_token, o_stride_head, fp8_stride_group, fp8_stride_token, scale_stride_group, scale_stride_k, stream)

    if (is_neox)
    {
        switch (chunks_per_head)
        {
        case 1: TRTLLM_INV_ROPE_DISPATCH_CHUNK(1, true); break;
        case 2: TRTLLM_INV_ROPE_DISPATCH_CHUNK(2, true); break;
        case 3: TRTLLM_INV_ROPE_DISPATCH_CHUNK(3, true); break;
        case 4: TRTLLM_INV_ROPE_DISPATCH_CHUNK(4, true); break;
        default: break;
        }
    }
    else
    {
        switch (chunks_per_head)
        {
        case 1: TRTLLM_INV_ROPE_DISPATCH_CHUNK(1, false); break;
        case 2: TRTLLM_INV_ROPE_DISPATCH_CHUNK(2, false); break;
        case 3: TRTLLM_INV_ROPE_DISPATCH_CHUNK(3, false); break;
        case 4: TRTLLM_INV_ROPE_DISPATCH_CHUNK(4, false); break;
        default: break;
        }
    }

#undef TRTLLM_INV_ROPE_DISPATCH_CHUNK
}

} // namespace kernels

TRTLLM_NAMESPACE_END
