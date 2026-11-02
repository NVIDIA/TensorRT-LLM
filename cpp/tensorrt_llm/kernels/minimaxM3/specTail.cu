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

#include "tensorrt_llm/kernels/minimaxM3/specTail.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Small fused kernels of the MiniMax-M3 Eagle3 step (GB300, decode).
//
// chunk_norm / capture_norm apply the drafter's per-layer fc_norm RMSNorms to the captured target hidden states with
// the in-tree RMSNorm arithmetic (fp32 statistic, (x * rsqrt) * w, one bf16 rounding); step_gather / slot_scatter move
// the step's integer outputs between the spec sampler's store and the engine's input buffers.
//
// draft_input replaces a fused draft pass's embedding lookup, its two RMSNorms (input_layernorm on the embedding,
// hidden_norm on the carried hidden state, the same arithmetic) and their concatenation with one launch. It optionally
// gathers the carried hidden rows (the second pass reads the last accepted row of each request) and writes them out
// as the residual of the decoder layer.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace specTail
{
namespace
{
constexpr int kVec = 8; // bf16 per 16-byte access

__device__ __forceinline__ float block_sum(float v, float* red)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    int const warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0)
        red[warp] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w)
        s += red[w];
    return s;
}

// blockIdx.x = row, blockIdx.y = 0 (embedding half) / 1 (hidden half). H % (kThreads * kVec) == 0 is not required;
// H % kVec == 0 and H <= kMaxPerThread * kThreads * kVec are.
template <int kIters>
__global__ void __launch_bounds__(kThreads)
    draft_input_kernel(int const* __restrict__ ids, int vocab, __nv_bfloat16 const* __restrict__ table,
        __nv_bfloat16 const* __restrict__ hidden, long long const* __restrict__ hidden_rows,
        __nv_bfloat16 const* __restrict__ w_embed, __nv_bfloat16 const* __restrict__ w_hidden, float eps_embed,
        float eps_hidden, int H, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ residual)
{
    __shared__ float red[kThreads / 32];
    int const row = blockIdx.x;
    bool const is_hidden = blockIdx.y == 1;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    __nv_bfloat16 const* src;
    bool valid = true;
    if (is_hidden)
    {
        long long const r = hidden_rows ? hidden_rows[row] : row;
        src = hidden + r * H;
    }
    else
    {
        int const id = ids[row];
        valid = id >= 0 && id < vocab;
        src = table + static_cast<long long>(valid ? id : 0) * H;
    }
    float v[kIters][kVec];
    float ss = 0.f;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const c = (it * kThreads + threadIdx.x) * kVec;
        if (c < H)
        {
            uint4 const raw = *reinterpret_cast<uint4 const*>(src + c);
            __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
            {
                v[it][j] = valid ? __bfloat162float(e[j]) : 0.f;
                ss += v[it][j] * v[it][j];
            }
            if (is_hidden && residual)
                *reinterpret_cast<uint4*>(residual + static_cast<long long>(row) * H + c) = raw;
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    float const rs = rsqrtf(block_sum(ss, red) / H + (is_hidden ? eps_hidden : eps_embed));
    __nv_bfloat16 const* w = is_hidden ? w_hidden : w_embed;
    __nv_bfloat16* dst = out + static_cast<long long>(row) * 2 * H + (is_hidden ? H : 0);
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const c = (it * kThreads + threadIdx.x) * kVec;
        if (c < H)
        {
            uint4 const wr = *reinterpret_cast<uint4 const*>(w + c);
            __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wr);
            uint4 o;
            __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                oe[j] = __float2bfloat16(v[it][j] * rs * __bfloat162float(we[j]));
            *reinterpret_cast<uint4*>(dst + c) = o;
        }
    }
}

// blockIdx.x = row, blockIdx.y = chunk c: dst[row, c*H:(c+1)*H] = rmsnorm(src[row, c*H:(c+1)*H]) * w[c].
template <int kIters>
__global__ void __launch_bounds__(kThreads) chunk_norm_kernel(__nv_bfloat16 const* __restrict__ src,
    long long src_stride, __nv_bfloat16 const* __restrict__ w, float eps, int H, int C, __nv_bfloat16* __restrict__ dst)
{
    __shared__ float red[kThreads / 32];
    int const row = blockIdx.x;
    int const c = blockIdx.y;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    __nv_bfloat16 const* in = src + row * src_stride + static_cast<long long>(c) * H;
    float v[kIters][kVec];
    float ss = 0.f;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const col = (it * kThreads + threadIdx.x) * kVec;
        if (col < H)
        {
            uint4 const raw = *reinterpret_cast<uint4 const*>(in + col);
            __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
            {
                v[it][j] = __bfloat162float(e[j]);
                ss += v[it][j] * v[it][j];
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    float const rs = rsqrtf(block_sum(ss, red) / H + eps);
    __nv_bfloat16 const* wc = w + static_cast<long long>(c) * H;
    __nv_bfloat16* out = dst + (static_cast<long long>(row) * C + c) * H;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const col = (it * kThreads + threadIdx.x) * kVec;
        if (col < H)
        {
            uint4 const wr = *reinterpret_cast<uint4 const*>(wc + col);
            __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wr);
            uint4 o;
            __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                oe[j] = __float2bfloat16(v[it][j] * rs * __bfloat162float(we[j]));
            *reinterpret_cast<uint4*>(out + col) = o;
        }
    }
}

// Eagle3 hidden-state capture with the drafter's fc_norm folded in (one row per CTA):
//   dst[row, :H] = rmsnorm(bf16(h[row] + r[row])) * w     (r optional)
// with chunk_norm's arithmetic, so fc reads the same values the capture + chunk_norm pair produced.
template <int kIters>
__global__ void __launch_bounds__(kThreads) capture_norm_kernel(__nv_bfloat16 const* __restrict__ h, long long h_stride,
    __nv_bfloat16 const* __restrict__ r, long long r_stride, __nv_bfloat16 const* __restrict__ w, float eps, int H,
    __nv_bfloat16* __restrict__ dst, long long dst_stride)
{
    __shared__ float red[kThreads / 32];
    int const row = blockIdx.x;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    float v[kIters][kVec];
    float ss = 0.f;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const col = (it * kThreads + threadIdx.x) * kVec;
        if (col < H)
        {
            uint4 const raw = *reinterpret_cast<uint4 const*>(h + row * h_stride + col);
            __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
            uint4 rraw = make_uint4(0u, 0u, 0u, 0u);
            if (r != nullptr)
                rraw = *reinterpret_cast<uint4 const*>(r + row * r_stride + col);
            __nv_bfloat16 const* re = reinterpret_cast<__nv_bfloat16 const*>(&rraw);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
            {
                float const x = __bfloat162float(e[j]);
                v[it][j] = r != nullptr ? __bfloat162float(__float2bfloat16(x + __bfloat162float(re[j]))) : x;
                ss += v[it][j] * v[it][j];
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    float const rs = rsqrtf(block_sum(ss, red) / H + eps);
    __nv_bfloat16* out = dst + row * dst_stride;
#pragma unroll
    for (int it = 0; it < kIters; ++it)
    {
        int const col = (it * kThreads + threadIdx.x) * kVec;
        if (col < H)
        {
            uint4 const wr = *reinterpret_cast<uint4 const*>(w + col);
            __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wr);
            uint4 o;
            __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                oe[j] = __float2bfloat16(v[it][j] * rs * __bfloat162float(we[j]));
            *reinterpret_cast<uint4*>(out + col) = o;
        }
    }
}

// Overlap-scheduler step inputs from the previous step's device outputs (one launch for the engine's gathers):
//   ids[ids0 + p*T + t]  = new_tok[t * nt_s0 + slot_p * nt_s1]
//   draft[d0 + p*D + d]  = next_draft[slot_p * nd_s0 + d * nd_s1]
//   pos_off[i]           = i in [b0*T, (b0+P)*T) ? lens[ppi[i - b0*T]] : 0      (i < n_pos)
//   kv_off[i]            = i in [b0, b0+P)       ? lens[slot_{i-b0}] - T : 0    (i < n_kv)
__global__ void step_gather_kernel(int const* __restrict__ new_tok, long long nt_s0, long long nt_s1,
    int const* __restrict__ next_draft, long long nd_s0, long long nd_s1, int const* __restrict__ lens,
    int const* __restrict__ slots, int const* __restrict__ ppi, int P, int T, int D, int* __restrict__ ids, int ids0,
    int* __restrict__ draft, int d0, int* __restrict__ pos_off, int n_pos, int* __restrict__ kv_off, int n_kv, int b0)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const stride = gridDim.x * blockDim.x;
    for (int i = tid; i < P * T; i += stride)
    {
        int const p = i / T, t = i - p * T;
        ids[ids0 + i] = new_tok[t * nt_s0 + static_cast<long long>(slots[p]) * nt_s1];
    }
    for (int i = tid; i < P * D; i += stride)
    {
        int const p = i / D, d = i - p * D;
        draft[d0 + i] = next_draft[static_cast<long long>(slots[p]) * nd_s0 + d * nd_s1];
    }
    for (int i = tid; i < n_pos; i += stride)
    {
        int const k = i - b0 * T;
        pos_off[i] = (k >= 0 && k < P * T) ? lens[ppi[k]] : 0;
    }
    for (int i = tid; i < n_kv; i += stride)
    {
        int const k = i - b0;
        kv_off[i] = (k >= 0 && k < P) ? lens[slots[k]] - T : 0;
    }
}

// Sampler stores from one step's outputs (one launch for its index_copy_s), zero-padding to the store widths:
//   a[slot_i][c] for c < wa (from src_a[i][c] if c < sa_w), b likewise, lens[slot_i] = src_lens[i], d likewise.
__global__ void slot_scatter_kernel(long long const* __restrict__ slots, int n, int const* __restrict__ sa,
    long long sa_s0, long long sa_s1, int sa_w, int* __restrict__ da, long long da_s0, long long da_s1, int wa,
    int const* __restrict__ sb, long long sb_s0, long long sb_s1, int sb_w, int* __restrict__ db, long long db_s0,
    long long db_s1, int wb, int const* __restrict__ sl, int* __restrict__ dl, int const* __restrict__ sd,
    long long sd_s0, long long sd_s1, int sd_w, int* __restrict__ dd, long long dd_s0, long long dd_s1, int wd)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    int const i = blockIdx.x;
    if (i >= n)
        return;
    long long const s = slots[i];
    for (int c = threadIdx.x; c < wa; c += blockDim.x)
        da[s * da_s0 + c * da_s1] = c < sa_w ? sa[i * sa_s0 + c * sa_s1] : 0;
    for (int c = threadIdx.x; c < wb; c += blockDim.x)
        db[s * db_s0 + c * db_s1] = c < sb_w ? sb[i * sb_s0 + c * sb_s1] : 0;
    for (int c = threadIdx.x; c < wd; c += blockDim.x)
        dd[s * dd_s0 + c * dd_s1] = c < sd_w ? sd[i * sd_s0 + c * sd_s1] : 0;
    if (threadIdx.x == 0)
        dl[s] = sl[i];
}

cudaLaunchConfig_t small_cfg(int grid, int block, bool pdl, cudaLaunchAttribute* attr, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

int norm_iters(int H)
{
    return (H / kVec + kThreads - 1) / kThreads;
}

} // namespace

cudaError_t invokeDraftInput(int const* ids, int vocab, __nv_bfloat16 const* table, __nv_bfloat16 const* hidden,
    long long const* rows, __nv_bfloat16 const* wEmbed, __nv_bfloat16 const* wHidden, float epsEmbed, float epsHidden,
    int H, int M, __nv_bfloat16* out, __nv_bfloat16* residual, bool pdl, cudaStream_t stream)
{
    int const iters = norm_iters(H);
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = small_cfg(1, kThreads, pdl, attr, stream);
    cfg.gridDim = dim3(M, 2);
#define LAUNCH(I)                                                                                                      \
    cudaLaunchKernelEx(&cfg, draft_input_kernel<I>, ids, vocab, table, hidden, rows, wEmbed, wHidden, epsEmbed,        \
        epsHidden, H, out, residual)
    switch (iters)
    {
    case 0:
    case 1: return LAUNCH(1);
    case 2: return LAUNCH(2);
    case 3: return LAUNCH(3);
    case 4: return LAUNCH(4);
    default: return cudaErrorInvalidValue;
    }
#undef LAUNCH
}

cudaError_t invokeChunkNorm(__nv_bfloat16 const* src, long long srcStride, __nv_bfloat16 const* w, float eps, int H,
    int C, int M, __nv_bfloat16* dst, bool pdl, cudaStream_t stream)
{
    int const iters = norm_iters(H);
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = small_cfg(1, kThreads, pdl, attr, stream);
    cfg.gridDim = dim3(M, C);
#define LAUNCH(I) cudaLaunchKernelEx(&cfg, chunk_norm_kernel<I>, src, srcStride, w, eps, H, C, dst)
    switch (iters)
    {
    case 0:
    case 1: return LAUNCH(1);
    case 2: return LAUNCH(2);
    case 3: return LAUNCH(3);
    case 4: return LAUNCH(4);
    default: return cudaErrorInvalidValue;
    }
#undef LAUNCH
}

cudaError_t invokeCaptureNorm(__nv_bfloat16 const* h, long long hStride, __nv_bfloat16 const* res, long long resStride,
    __nv_bfloat16 const* w, float eps, int H, int M, __nv_bfloat16* dst, long long dstStride, cudaStream_t stream)
{
    if (M == 0)
    {
        return cudaSuccess;
    }
    int const iters = norm_iters(H);
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = small_cfg(M, kThreads, true, attr, stream);
#define LAUNCH(I)                                                                                                      \
    cudaLaunchKernelEx(&cfg, capture_norm_kernel<I>, h, hStride, res, resStride, w, eps, H, dst, dstStride)
    switch (iters)
    {
    case 0:
    case 1: return LAUNCH(1);
    case 2: return LAUNCH(2);
    case 3: return LAUNCH(3);
    case 4: return LAUNCH(4);
    default: return cudaErrorInvalidValue;
    }
#undef LAUNCH
}

cudaError_t invokeStepGather(int const* newTok, long long ntS0, long long ntS1, int const* nextDraft, long long ndS0,
    long long ndS1, int const* lens, int const* slots, int const* ppi, int P, int T, int D, int* ids, int ids0,
    int* draft, int d0, int* posOff, int nPos, int* kvOff, int nKv, int b0, int work, cudaStream_t stream)
{
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = small_cfg(std::max(1, std::min(64, (work + 255) / 256)), 256, true, attr, stream);
    return cudaLaunchKernelEx(&cfg, step_gather_kernel, newTok, ntS0, ntS1, nextDraft, ndS0, ndS1, lens, slots, ppi, P,
        T, D, ids, ids0, draft, d0, posOff, nPos, kvOff, nKv, b0);
}

cudaError_t invokeSlotScatter(long long const* slots, int n, IntView sa, IntView da, IntView sb, IntView db,
    int const* sl, int* dl, IntView sd, IntView dd, cudaStream_t stream)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = small_cfg(n, 32, true, attr, stream);
    return cudaLaunchKernelEx(&cfg, slot_scatter_kernel, slots, n, sa.data, sa.s0, sa.s1, sa.width, da.data, da.s0,
        da.s1, da.width, sb.data, sb.s0, sb.s1, sb.width, db.data, db.s0, db.s1, db.width, sl, dl, sd.data, sd.s0,
        sd.s1, sd.width, dd.data, dd.s0, dd.s1, dd.width);
}

} // namespace specTail
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
