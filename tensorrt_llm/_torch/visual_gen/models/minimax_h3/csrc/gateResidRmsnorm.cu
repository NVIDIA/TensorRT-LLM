/*
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
 */

// MiniMax-H3 gated residual + RMSNorm + modulation, with per-row gate and modulator rows selected by an index tensor.
//   h = bf16(x + attn * gate[idx])                                   (fp32 math, sum of squares before the round)
//   y = bf16(h * rstd * w * (1 + scale[idx]) + shift[idx])           (fp32 math, one rounding)

#include "gateResidRmsnorm.h"

namespace tensorrt_llm::visual_gen::minimax_h3
{

namespace
{

constexpr int kBlockSize = 128;

__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// One row per CTA. x is bulk-loaded into shared memory; the attn and gate chunks are loaded into registers while that
// load is in flight. h overwrites x in shared memory for the output pass, which reads the L2-resident weight and
// modulator rows directly.
template <int D, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void gateResidRmsnormIndexedShiftScaleKernel(
    __nv_bfloat16 const* __restrict__ x, __nv_bfloat16 const* __restrict__ attn, __nv_bfloat16 const* __restrict__ w,
    __nv_bfloat16 const* __restrict__ gate, __nv_bfloat16 const* __restrict__ scale,
    __nv_bfloat16 const* __restrict__ shift, int64_t const* __restrict__ idx, int64_t gateStride, int64_t modStride,
    __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ hOut, float eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    constexpr int kChunkElems = 8;
    constexpr int kWarps = BLOCK_SIZE / 32;
    constexpr int kChunks = (D + BLOCK_SIZE * kChunkElems - 1) / (BLOCK_SIZE * kChunkElems);
    constexpr uint32_t kRowBytes = D * sizeof(__nv_bfloat16);
    static_assert(D % kChunkElems == 0 && kRowBytes % 16 == 0, "D must be a multiple of 8");
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of the warp size");

    asm volatile("griddepcontrol.wait;");

    int const tid = threadIdx.x;
    int64_t const rowBase = static_cast<int64_t>(blockIdx.x) * D;

    extern __shared__ __align__(16) unsigned char smemRaw[];
    __nv_bfloat16* smemX = reinterpret_cast<__nv_bfloat16*>(smemRaw);
    float* warpSums = reinterpret_cast<float*>(smemRaw + kRowBytes);
    __shared__ alignas(8) uint64_t mbar;

    uint32_t const mbarSmem = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    if (tid == 0)
    {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(mbarSmem) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(mbarSmem), "r"(kRowBytes) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(smemX))),
                     "l"(reinterpret_cast<uint64_t>(x + rowBase)), "r"(kRowBytes), "r"(mbarSmem)
                     : "memory");
    }
    __syncthreads();

    int64_t const row = idx[blockIdx.x];
    int64_t const gateBase = row * gateStride;
    int64_t const modBase = row * modStride;

    uint4 aCache[kChunks];
    uint4 gCache[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const e = c * BLOCK_SIZE * kChunkElems + tid * kChunkElems;
        if (e >= D)
            continue;
        aCache[c] = *reinterpret_cast<uint4 const*>(&attn[rowBase + e]);
        gCache[c] = *reinterpret_cast<uint4 const*>(&gate[gateBase + e]);
    }

    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], 0;\n"
        "@P1 bra DONE;\n"
        "bra     WAIT_LOOP;\n"
        "DONE:\n"
        "}\n" ::"r"(mbarSmem)
        : "memory");

    // Each thread overwrites only the shared-memory chunks it reads.
    float sum2 = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const e = c * BLOCK_SIZE * kChunkElems + tid * kChunkElems;
        if (e >= D)
            continue;
        uint4 const xv = *reinterpret_cast<uint4 const*>(&smemX[e]);
        __nv_bfloat162 const* xb = reinterpret_cast<__nv_bfloat162 const*>(&xv);
        __nv_bfloat162 const* ab = reinterpret_cast<__nv_bfloat162 const*>(&aCache[c]);
        __nv_bfloat162 const* gb = reinterpret_cast<__nv_bfloat162 const*>(&gCache[c]);
        uint4 hv;
        __nv_bfloat162* hb = reinterpret_cast<__nv_bfloat162*>(&hv);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 const xf = __bfloat1622float2(xb[i]);
            float2 const af = __bfloat1622float2(ab[i]);
            float2 const gf = __bfloat1622float2(gb[i]);
            float2 const nf = make_float2(xf.x + af.x * gf.x, xf.y + af.y * gf.y);
            sum2 += nf.x * nf.x + nf.y * nf.y;
            hb[i] = __float22bfloat162_rn(nf);
        }
        *reinterpret_cast<uint4*>(&hOut[rowBase + e]) = hv;
        *reinterpret_cast<uint4*>(&smemX[e]) = hv;
    }

    sum2 = warpReduceSum(sum2);
    if ((tid & 31) == 0)
        warpSums[tid >> 5] = sum2;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < kWarps; i++)
        total += warpSums[i];
    float const rstd = rsqrtf(total / static_cast<float>(D) + eps);

#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const e = c * BLOCK_SIZE * kChunkElems + tid * kChunkElems;
        if (e >= D)
            continue;
        uint4 const hv = *reinterpret_cast<uint4 const*>(&smemX[e]);
        uint4 const wv = *reinterpret_cast<uint4 const*>(&w[e]);
        uint4 const sv = *reinterpret_cast<uint4 const*>(&scale[modBase + e]);
        uint4 const tv = *reinterpret_cast<uint4 const*>(&shift[modBase + e]);
        __nv_bfloat162 const* hb = reinterpret_cast<__nv_bfloat162 const*>(&hv);
        __nv_bfloat162 const* wb = reinterpret_cast<__nv_bfloat162 const*>(&wv);
        __nv_bfloat162 const* sb = reinterpret_cast<__nv_bfloat162 const*>(&sv);
        __nv_bfloat162 const* tb = reinterpret_cast<__nv_bfloat162 const*>(&tv);
        uint4 out;
        __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 const hf = __bfloat1622float2(hb[i]);
            float2 const wf = __bfloat1622float2(wb[i]);
            float2 const sf = __bfloat1622float2(sb[i]);
            float2 const tf = __bfloat1622float2(tb[i]);
            ob[i] = __float22bfloat162_rn(
                make_float2(hf.x * rstd * wf.x * (1.0f + sf.x) + tf.x, hf.y * rstd * wf.y * (1.0f + sf.y) + tf.y));
        }
        *reinterpret_cast<uint4*>(&y[rowBase + e]) = out;
    }

    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

} // namespace

cudaError_t invokeGateResidRmsnormIndexedShiftScale(__nv_bfloat16* y, __nv_bfloat16* h, __nv_bfloat16 const* x,
    __nv_bfloat16 const* attn, __nv_bfloat16 const* w, __nv_bfloat16 const* gate, __nv_bfloat16 const* scale,
    __nv_bfloat16 const* shift, int64_t const* idx, int64_t gateStride, int64_t modStride, int rows, float eps,
    cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cfg.gridDim = dim3(static_cast<unsigned>(rows));
    cfg.blockDim = dim3(kBlockSize);
    cfg.dynamicSmemBytes = kGateResidRmsnormHidden * sizeof(__nv_bfloat16) + (kBlockSize / 32) * sizeof(float);
    return cudaLaunchKernelEx(&cfg, gateResidRmsnormIndexedShiftScaleKernel<kGateResidRmsnormHidden, kBlockSize>, x,
        attn, w, gate, scale, shift, idx, gateStride, modStride, y, h, eps);
}

} // namespace tensorrt_llm::visual_gen::minimax_h3
