// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Packed-QKV prep for Wan fused FP8 attention: QK RMSNorm + RoPE + FP8 quant (norm_rope_quant),
// V amax and scale (v_scale), and the LSE merge of partial attention outputs (lse_combine).

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace
{

template <typename T>
__device__ __forceinline__ T warpReduceSumT(T v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warpReduceMaxF(float v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

} // namespace

// Ring depth of the bulk-copy pipeline (rows in flight per block).
constexpr int kNrqStages = 3;
// Pipelined kernel: 128 threads, 8 bf16 per thread per row pass.
constexpr int kHeadDim = 128;
constexpr int kPipeThreads = 128;
constexpr int kVec = 8;
constexpr int kRowChunk = kPipeThreads * kVec; // elements per row pass

namespace
{

// float to bf16 (RN) and back in one cvt: zero low half.
__device__ __forceinline__ float roundBf16(float v)
{
    __nv_bfloat162 const p = __floats2bfloat162_rn(0.f, v);
    return __uint_as_float(*reinterpret_cast<uint32_t const*>(&p));
}

// 8 values: bf16 round (single cvt), scale, saturating e4m3 store.
__device__ __forceinline__ void storeFp8x8(float const (&x)[8], __nv_fp8_e4m3* dst, float mul)
{
    __nv_fp8x2_storage_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = roundBf16(x[2 * i]);
        float b = roundBf16(x[2 * i + 1]);
        packed[i] = __nv_cvt_float2_to_fp8x2(make_float2(a * mul, b * mul), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(packed);
}

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(smemAddr(bar)), "r"(phase)
            : "memory");
    }
}

// Async bulk global-to-shared copy; completion counted on the mbarrier.
__device__ __forceinline__ void bulkLoad(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar))
        : "memory");
}

// Q/K chunk: RMSNorm weight, RoPE (fp32 cos/sin), bf16 round, FP8 quant.
template <bool INTERLEAVE>
__device__ __forceinline__ void normRopeChunk(uint4 const in_vec, uint4 const w_vec, float const* cosp,
    float const* sinp, int row_lane, float rms_rcp, __nv_fp8_e4m3* dst, float mul)
{
    constexpr int CHUNK_ELEMS = kVec;
    float cos_vals[CHUNK_ELEMS];
    float sin_vals[CHUNK_ELEMS];
    float4 const* cs = reinterpret_cast<float4 const*>(cosp);
    float4 const* ss = reinterpret_cast<float4 const*>(sinp);
    float4 c0 = cs[0], c1 = cs[1];
    float4 s0 = ss[0], s1 = ss[1];
    cos_vals[0] = c0.x;
    cos_vals[1] = c0.y;
    cos_vals[2] = c0.z;
    cos_vals[3] = c0.w;
    cos_vals[4] = c1.x;
    cos_vals[5] = c1.y;
    cos_vals[6] = c1.z;
    cos_vals[7] = c1.w;
    sin_vals[0] = s0.x;
    sin_vals[1] = s0.y;
    sin_vals[2] = s0.z;
    sin_vals[3] = s0.w;
    sin_vals[4] = s1.x;
    sin_vals[5] = s1.y;
    sin_vals[6] = s1.z;
    sin_vals[7] = s1.w;

    float elements[CHUNK_ELEMS];
    float w_vals[CHUNK_ELEMS];
    uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
    uint const* w_uints = reinterpret_cast<uint const*>(&w_vec);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
        elements[2 * i] = xv.x;
        elements[2 * i + 1] = xv.y;
        float2 wv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&w_uints[i]));
        w_vals[2 * i] = wv.x;
        w_vals[2 * i + 1] = wv.y;
    }

#pragma unroll
    for (int i = 0; i < CHUNK_ELEMS; i++)
        elements[i] *= rms_rcp * w_vals[i];

    if constexpr (INTERLEAVE)
    {
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i += 2)
        {
            float const x = elements[i], y = elements[i + 1];
            elements[i] = x * cos_vals[i] + (-y) * sin_vals[i];
            elements[i + 1] = y * cos_vals[i + 1] + x * sin_vals[i + 1];
        }
    }
    else
    {
        constexpr int xor_mask = kHeadDim / 16;
        bool const negate = ((row_lane & xor_mask) == 0);
        unsigned const activeMask = __activemask();
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
        {
            float p = __shfl_xor_sync(activeMask, elements[i], xor_mask);
            if (negate)
            {
                p = -p;
            }
            elements[i] = elements[i] * cos_vals[i] + p * sin_vals[i];
        }
    }

    // Round to bf16 first to match the unfused path.
    storeFp8x8(elements, dst, mul);
}

// V chunk: bf16 to FP8 with the V multiplier.
__device__ __forceinline__ void quantChunk(uint4 const in_vec, __nv_fp8_e4m3* dst, float mul)
{
    uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
    float elements[8];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
        elements[2 * i] = xv.x;
        elements[2 * i + 1] = xv.y;
    }
    storeFp8x8(elements, dst, mul);
}

// Per-row sum of squares over this thread's Q and K chunks (original order).
template <int NCH>
__device__ __forceinline__ void rowSumSq(
    __nv_bfloat16 const* sq, __nv_bfloat16 const* sk, int lane_in_row, int N, float& q_sum2, float& k_sum2)
{
#pragma unroll
    for (int chunk = 0; chunk < NCH; chunk++)
    {
        int const elemBase = chunk * kRowChunk + lane_in_row * kVec;
        if (elemBase >= N)
            continue;
        uint4 const qv = *reinterpret_cast<uint4 const*>(&sq[elemBase]);
        uint4 const kv = *reinterpret_cast<uint4 const*>(&sk[elemBase]);
        uint const* qu = reinterpret_cast<uint const*>(&qv);
        uint const* ku = reinterpret_cast<uint const*>(&kv);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 qv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&qu[i]));
            float2 kv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&ku[i]));
            q_sum2 += qv2.x * qv2.x + qv2.y * qv2.y;
            k_sum2 += kv2.x * kv2.x + kv2.y * kv2.y;
        }
    }
}

} // namespace

// Persistent 128-thread blocks; each row (qkv + shared fp32 cos/sin) is streamed in by bulk copies.
template <bool INTERLEAVE, int NCH>
__global__ void __launch_bounds__(kPipeThreads) fusedDiTQKNormRopeFp8QuantPipeKernel(__nv_bfloat16 const* qkv,
    __nv_fp8_e4m3* q_out, __nv_fp8_e4m3* k_out, __nv_fp8_e4m3* v_out, float const* quant_mul, int const num_heads,
    float const eps, __nv_bfloat16 const* q_weight, __nv_bfloat16 const* k_weight, float const* cos_emb,
    float const* sin_emb, int const num_tokens, int const cos_seq_per_batch)
{
    constexpr int WARPS_PER_ROW = kPipeThreads / 32;
    constexpr int NST = kNrqStages;
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif
    int const tid = threadIdx.x;
    int const row_warp = tid >> 5;
    int const row_lane = tid & 31;
    int const N = num_heads * kHeadDim;
    uint32_t const rowBytes = 3 * N * sizeof(__nv_bfloat16);
    uint32_t const stageBytes = rowBytes + 2 * kHeadDim * sizeof(float);
    extern __shared__ __align__(128) unsigned char ring[];
    __shared__ __align__(8) uint64_t bars[NST];
    __shared__ float warp_sums[2 * WARPS_PER_ROW];
    int const stride = gridDim.x;
    auto issue = [&](int st, int t)
    {
        unsigned char* dst = ring + st * stageBytes;
        int const ct = (cos_seq_per_batch > 0) ? (t % cos_seq_per_batch) : t;
        mbarExpectTx(&bars[st], stageBytes);
        bulkLoad(dst, qkv + static_cast<int64_t>(t) * 3 * N, rowBytes, &bars[st]);
        bulkLoad(dst + rowBytes, cos_emb + static_cast<int64_t>(ct) * kHeadDim, kHeadDim * sizeof(float), &bars[st]);
        bulkLoad(dst + rowBytes + kHeadDim * sizeof(float), sin_emb + static_cast<int64_t>(ct) * kHeadDim,
            kHeadDim * sizeof(float), &bars[st]);
    };
    if (tid == 0)
    {
#pragma unroll
        for (int st = 0; st < NST; st++)
            mbarInit(&bars[st], 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
        for (int st = 0; st < NST; st++)
        {
            int const t = blockIdx.x + st * stride;
            if (t < num_tokens)
                issue(st, t);
        }
    }
    // Norm weights stay in registers across rows.
    uint4 q_w_cache[NCH], k_w_cache[NCH];
#pragma unroll
    for (int chunk = 0; chunk < NCH; chunk++)
    {
        int const elemBase = chunk * kRowChunk + tid * kVec;
        if (elemBase >= N)
            continue;
        uint4 const qw = *reinterpret_cast<uint4 const*>(&q_weight[elemBase]);
        uint4 const kw = *reinterpret_cast<uint4 const*>(&k_weight[elemBase]);
        q_w_cache[chunk] = qw;
        k_w_cache[chunk] = kw;
    }
    float const q_mul = quant_mul[0], k_mul = quant_mul[1], v_mul = quant_mul[2];
    __syncthreads();
    int it = 0;
    for (int tokenIdx = blockIdx.x; tokenIdx < num_tokens; tokenIdx += stride, it++)
    {
        int const st = it % NST;
        mbarWait(&bars[st], (it / NST) & 1);
        __nv_bfloat16 const* srow = reinterpret_cast<__nv_bfloat16 const*>(ring + st * stageBytes);
        float const* scos = reinterpret_cast<float const*>(ring + st * stageBytes + rowBytes);
        float const* ssin = scos + kHeadDim;
        float q_sum2 = 0.0f, k_sum2 = 0.0f;
        rowSumSq<NCH>(srow, srow + N, tid, N, q_sum2, k_sum2);
        q_sum2 = warpReduceSumT(q_sum2);
        k_sum2 = warpReduceSumT(k_sum2);
        if (row_lane == 0)
        {
            warp_sums[2 * row_warp + 0] = q_sum2;
            warp_sums[2 * row_warp + 1] = k_sum2;
        }
        __syncthreads();
        float q_total = 0.0f, k_total = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_ROW; w++)
        {
            q_total += warp_sums[2 * w + 0];
            k_total += warp_sums[2 * w + 1];
        }
        float const q_rms_rcp = rsqrtf(q_total / static_cast<float>(N) + eps);
        float const k_rms_rcp = rsqrtf(k_total / static_cast<float>(N) + eps);
        int64_t const outBase = static_cast<int64_t>(tokenIdx) * N;
#pragma unroll
        for (int chunk = 0; chunk < NCH; chunk++)
        {
            int const elemBase = chunk * kRowChunk + tid * kVec;
            if (elemBase >= N)
                continue;
            int const cosOff = elemBase % kHeadDim;
            normRopeChunk<INTERLEAVE>(*reinterpret_cast<uint4 const*>(srow + elemBase), q_w_cache[chunk], scos + cosOff,
                ssin + cosOff, row_lane, q_rms_rcp, q_out + outBase + elemBase, q_mul);
            normRopeChunk<INTERLEAVE>(*reinterpret_cast<uint4 const*>(srow + N + elemBase), k_w_cache[chunk],
                scos + cosOff, ssin + cosOff, row_lane, k_rms_rcp, k_out + outBase + elemBase, k_mul);
            quantChunk(*reinterpret_cast<uint4 const*>(srow + 2 * N + elemBase), v_out + outBase + elemBase, v_mul);
        }
        // Stage and warp sums fully consumed; refill the stage.
        __syncthreads();
        if (tid == 0 && tokenIdx + NST * stride < num_tokens)
            issue(st, tokenIdx + NST * stride);
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Per-block |V| amax partials; one warp per token, kVU loads in flight.
constexpr int kVU = 10;

__global__ void __launch_bounds__(256) vAmaxRowKernel(__nv_bfloat16 const* qkv, int64_t tokens, int N, float* out)
{
    int const lane = threadIdx.x & 31;
    int64_t const nWarps = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
    int const cpr = N / 8;
    float m = 0.f;
    for (int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; t < tokens; t += nWarps)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(qkv + t * 3 * N + 2 * N);
        for (int c0 = 0; c0 < cpr; c0 += 32 * kVU)
        {
            uint4 v[kVU];
#pragma unroll
            for (int u = 0; u < kVU; u++)
            {
                int const c = c0 + u * 32 + lane;
                v[u] = c < cpr ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < kVU; u++)
            {
                __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&v[u]);
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    float2 f = __bfloat1622float2(h[j]);
                    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }
    }
    m = warpReduceMaxF(m);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0)
        sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMaxF(m);
        if (threadIdx.x == 0)
        {
            out[blockIdx.x] = m;
        }
    }
}

// Max of per-block partials, then the V scale and its reciprocal.
__global__ void vScaleReduceKernel(float const* partial, int n, float* amax, float* mul_v, float* scale_v)
{
    float m = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, partial[i]);
    m = warpReduceMaxF(m);
    __shared__ float sm[32];
    if ((threadIdx.x & 31) == 0)
        sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMaxF(m);
        if (threadIdx.x == 0)
        {
            amax[0] = m;
            float const s = fmaxf(m, 1e-12f) / 448.f;
            *scale_v = s;
            *mul_v = 1.f / s;
        }
    }
}

// Resident-block grid for the amax kernel.
int vAmaxGrid(int sms)
{
    static int perSm = 0;
    if (perSm == 0)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, vAmaxRowKernel, 256, 0);
    return std::max(1, perSm) * sms;
}

template <typename K, typename... Args>
void launchPdl(K kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args)
{
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

// Shared fp32 cos rows, chunk count as a template argument.
template <bool IL, int NCH>
void launchNrq(int tokens, int H, cudaStream_t stream, __nv_bfloat16 const* qkv, __nv_fp8_e4m3* q8, __nv_fp8_e4m3* k8,
    __nv_fp8_e4m3* v8, float const* mul, float eps, __nv_bfloat16 const* qw, __nv_bfloat16 const* kw, float const* cosp,
    float const* sinp, int cspb)
{
    int const N = H * kHeadDim;
    auto* kern = fusedDiTQKNormRopeFp8QuantPipeKernel<IL, NCH>;
    size_t const smem = kNrqStages * (3 * N * sizeof(__nv_bfloat16) + 2 * kHeadDim * sizeof(float));
    cudaFuncSetAttribute(reinterpret_cast<void const*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int perSm = 0, sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kern, kPipeThreads, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int const grid = std::min(tokens, std::max(1, perSm) * sms);
    launchPdl(kern, dim3(grid), dim3(kPipeThreads), smem, stream, qkv, q8, k8, v8, mul, H, eps, qw, kw, cosp, sinp,
        tokens, cspb);
}

template <bool IL>
void dispatchNrq(int nch, int tokens, int H, cudaStream_t stream, __nv_bfloat16 const* qkv, __nv_fp8_e4m3* q8,
    __nv_fp8_e4m3* k8, __nv_fp8_e4m3* v8, float const* mul, float eps, __nv_bfloat16 const* qw, __nv_bfloat16 const* kw,
    float const* cosp, float const* sinp, int cspb)
{
#define NRQ_CASE(C)                                                                                                    \
    case C: launchNrq<IL, C>(tokens, H, stream, qkv, q8, k8, v8, mul, eps, qw, kw, cosp, sinp, cspb); break
    switch (nch)
    {
        NRQ_CASE(1);
        NRQ_CASE(2);
        NRQ_CASE(3);
        NRQ_CASE(4);
        NRQ_CASE(5);
        NRQ_CASE(6);
        NRQ_CASE(7);
        NRQ_CASE(8);
    default: TORCH_CHECK(false, "bad chunk count");
    }
#undef NRQ_CASE
}

// Merges NUM_PARTS partial outputs by LSE; stats are float2 {max, sum}, one warp per row-head.
template <int NUM_PARTS>
__global__ void lseCombineKernel(
    __nv_bfloat16 const* o, float2 const* stats, __nv_bfloat16* out, int64_t rowsHeads, int64_t partStride)
{
    int64_t const rh = blockIdx.x * (int64_t) (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (rh >= rowsHeads)
        return;
    int const lane = threadIdx.x & 31;
    float lse[NUM_PARTS];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NUM_PARTS; i++)
    {
        float2 const st = stats[i * partStride + rh];
        lse[i] = st.x + logf(st.y);
        m = fmaxf(m, lse[i]);
    }
    float denom = 0.f;
#pragma unroll
    for (int i = 0; i < NUM_PARTS; i++)
    {
        lse[i] = expf(lse[i] - m);
        denom += lse[i];
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < NUM_PARTS; i++)
    {
        uint2 const v = *reinterpret_cast<uint2 const*>(o + (i * partStride + rh) * kHeadDim + lane * 4);
        __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&v);
        float2 const a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
        float const w = lse[i] / denom;
        acc[0] += w * a.x;
        acc[1] += w * a.y;
        acc[2] += w * b.x;
        acc[3] += w * b.y;
    }
    __nv_bfloat162 res[2] = {__floats2bfloat162_rn(acc[0], acc[1]), __floats2bfloat162_rn(acc[2], acc[3])};
    *reinterpret_cast<uint2*>(out + rh * kHeadDim + lane * 4) = *reinterpret_cast<uint2 const*>(res);
}

// qkv [T, 3*H*D] bf16 to q8/k8/v8 [T, H*D] fp8; cos/sin are [rows, 128] fp32 tables.
void norm_rope_quant(torch::Tensor qkv, int64_t num_heads, double eps, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos_emb, torch::Tensor sin_emb, bool interleave, int64_t cos_seq_per_batch, torch::Tensor quant_mul,
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8)
{
    TORCH_CHECK(
        qkv.is_contiguous() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2, "qkv must be contiguous 2-D bf16");
    TORCH_CHECK(cos_emb.scalar_type() == at::kFloat && sin_emb.scalar_type() == at::kFloat, "fp32 cos/sin only");
    int const H = static_cast<int>(num_heads);
    int const N = H * kHeadDim;
    TORCH_CHECK(H <= 64 && qkv.size(1) == 3 * N, "qkv must be [T, 3*H*128] with <= 64 heads");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16 && k_weight.scalar_type() == at::kBFloat16
            && q_weight.numel() == N && k_weight.numel() == N && q_weight.is_contiguous() && k_weight.is_contiguous(),
        "q/k norm weights must be contiguous bf16 [H*128]");
    TORCH_CHECK(
        cos_emb.size(-1) == kHeadDim && sin_emb.size(-1) == kHeadDim, "cos/sin must be [rows, 128] fp32 tables");
    TORCH_CHECK(cos_emb.is_contiguous() && sin_emb.is_contiguous() && cos_emb.sizes() == sin_emb.sizes()
            && cos_emb.numel() / kHeadDim >= (cos_seq_per_batch > 0 ? cos_seq_per_batch : qkv.size(0)),
        "cos/sin must be contiguous with one row per token (or per position of one batch entry)");
    TORCH_CHECK(quant_mul.scalar_type() == at::kFloat && quant_mul.numel() == 3, "quant_mul must be fp32 [3]");
    for (auto const* t : {&q8, &k8, &v8})
        TORCH_CHECK(t->scalar_type() == at::kFloat8_e4m3fn && t->is_contiguous() && t->numel() == qkv.size(0) * N,
            "q8/k8/v8 must be contiguous fp8 [T, H*128]");
    auto aligned = [](void const* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    TORCH_CHECK(aligned(qkv.data_ptr()) && aligned(cos_emb.data_ptr()) && aligned(sin_emb.data_ptr()),
        "qkv/cos/sin must be 16-byte aligned (bulk copy)");
    int const tokens = static_cast<int>(qkv.size(0));
    if (tokens == 0)
        return;
    auto const stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    int const nch = (N + kRowChunk - 1) / kRowChunk;
    auto args = std::make_tuple(nch, tokens, H, stream.stream(), reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q8.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(k8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()), quant_mul.data_ptr<float>(), static_cast<float>(eps),
        reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr()), cos_emb.data_ptr<float>(),
        sin_emb.data_ptr<float>(), static_cast<int>(cos_seq_per_batch));
    if (interleave)
        std::apply(dispatchNrq<true>, args);
    else
        std::apply(dispatchNrq<false>, args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// |V| amax of packed qkv [T, 3*H*128] bf16: amax_v[0] = amax, scale_v[0] = max(amax, 1e-12) / 448,
// mul[2] = 1 / scale_v.
void v_scale(torch::Tensor qkv, int64_t num_heads, torch::Tensor amax_v, torch::Tensor mul, torch::Tensor scale_v)
{
    int const N = static_cast<int>(num_heads) * kHeadDim;
    TORCH_CHECK(qkv.is_contiguous() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(1) == 3 * N,
        "qkv must be contiguous bf16 [T, 3*H*128]");
    TORCH_CHECK(mul.scalar_type() == at::kFloat && mul.numel() == 3 && amax_v.scalar_type() == at::kFloat
            && scale_v.scalar_type() == at::kFloat,
        "fp32 amax_v/scale_v and mul[3]");
    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, qkv.get_device());
    auto const* src = reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr());
    // Two passes: per-block partial maxima, then one reduce block.
    int const grid = vAmaxGrid(sms);
    auto partial = torch::empty({grid}, qkv.options().dtype(at::kFloat));
    vAmaxRowKernel<<<grid, 256, 0, stream>>>(src, qkv.size(0), N, partial.data_ptr<float>());
    vScaleReduceKernel<<<1, 1024, 0, stream>>>(partial.data_ptr<float>(), grid, amax_v.data_ptr<float>(),
        mul.data_ptr<float>() + 2, scale_v.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// o [N, R, H, 128] bf16, stats [N, R, H, 2] f32 to out [R, H, 128] bf16.
void lse_combine(torch::Tensor o, torch::Tensor stats, torch::Tensor out)
{
    TORCH_CHECK(o.is_contiguous() && stats.is_contiguous() && out.is_contiguous(), "contiguous inputs only");
    TORCH_CHECK(o.scalar_type() == at::kBFloat16 && stats.scalar_type() == at::kFloat && o.size(-1) == kHeadDim);
    int const n = static_cast<int>(o.size(0));
    int64_t const rowsHeads = out.numel() / kHeadDim;
    TORCH_CHECK(o.numel() == n * rowsHeads * kHeadDim && stats.numel() == n * rowsHeads * 2, "shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream(o.get_device());
    dim3 const block(256), grid((rowsHeads + 7) / 8);
    auto const* op = reinterpret_cast<__nv_bfloat16 const*>(o.data_ptr());
    auto const* sp = reinterpret_cast<float2 const*>(stats.data_ptr());
    auto* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    switch (n)
    {
    case 1: lseCombineKernel<1><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    case 2: lseCombineKernel<2><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    case 3: lseCombineKernel<3><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    case 4: lseCombineKernel<4><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    case 6: lseCombineKernel<6><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    case 8: lseCombineKernel<8><<<grid, block, 0, stream>>>(op, sp, outp, rowsHeads, rowsHeads); break;
    default: TORCH_CHECK(false, "lse_combine: unsupported partial count ", n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("norm_rope_quant", &norm_rope_quant);
    m.def("v_scale", &v_scale);
    m.def("lse_combine", &lse_combine);
}
