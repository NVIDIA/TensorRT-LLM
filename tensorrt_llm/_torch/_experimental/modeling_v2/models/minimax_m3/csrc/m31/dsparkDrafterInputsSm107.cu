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

// The DSpark / DFlash drafter's step inputs (ops/dspark/dspark_drafter_inputs): CUDA C++ replacement of its
// former Triton kernels (NO TRITON in served paths), bit for bit. Measured (on
// R1): EXACT on every input class (table dtypes, odd row widths, int32 / int64 indices, invalid bonus ids, the refresh
// plane incl. non-power-of-two and > 2^31 divisors); 4 warps + PDL = median -0.19 us / call vs the Triton kernel.
// Grid (n, blockSize [+ 1 refresh plane], Z chunks): a copy CTA moves blockDim units, one per thread, in the widest
// unit the alignment allows; mask rows copy without the accepted-token chain; the table row CTA issues the slot ->
// ctxLen chain beside nacc -> accepted; CTA (r, 0, 0) writes r's scalars. The refresh plane decodes int32 offsets in
// int32 (a power-of-two divisor by shift; a divisor > 2^31 - 1 gives 0), counts per row in chunk 0 with its loads
// issued together with its decode loads, and STORES counts[r].

#include "minimax_m3/csrc/m31/dsparkDrafterInputsSm107.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{
constexpr int kMaxWarps = 8;
constexpr int kRefItems = 8; // refresh decode entries per thread
constexpr int kCnt = 32;     // refresh count entries per thread per pass (chunk 0)

struct InArgs
{
    void const* acc;
    int acc_i64;
    int64_t acc_stride;
    void const* nacc;
    int nacc_i64;
    int64_t const* slots;
    int64_t const* ctx_len;
    int64_t max_ctx;
    unsigned char const* table;
    unsigned char const* mask;
    int64_t* bonus;
    int64_t* qpos;
    int64_t* cpos;
    unsigned char* noise;
    int64_t row_bytes;
    int64_t vocab;
    int bq;
    int kp1;
    // refresh plane (refresh != 0)
    int refresh;
    void const* src;
    int src_i64;
    int64_t src_stride_row, src_stride_col;
    int32_t* tables;
    int64_t tables_stride;
    int64_t* counts;
    int width;
    int64_t divisor;
    int shift; // log2(divisor) for a power of two (31 for an int32 source with divisor > 2^31 - 1), else -1
};

__device__ __forceinline__ int64_t ld_i(void const* p, int i64, int64_t idx)
{
    return i64 ? static_cast<int64_t const*>(p)[idx] : static_cast<int64_t>(static_cast<int32_t const*>(p)[idx]);
}

template <typename T>
__device__ __forceinline__ int32_t decode(T v, int shift, T divisor)
{
    T const c = v > 0 ? v : T(0);
    if (shift >= 0)
        return static_cast<int32_t>(c >> shift);
    using UT = typename std::conditional<sizeof(T) == 8, uint64_t, uint32_t>::type;
    return static_cast<int32_t>(static_cast<T>(static_cast<UT>(c) / static_cast<UT>(divisor)));
}

template <typename T>
__device__ void refresh_chunk_t(InArgs const& a, int64_t r, int z)
{
    __shared__ int32_t warp_cnt[kMaxWarps];
    int const nthr = blockDim.x;
    int const tid = threadIdx.x;
    T const* srow = static_cast<T const*>(a.src) + r * a.src_stride_row;
    int32_t* trow = a.tables + r * a.tables_stride;
    T const divisor = static_cast<T>(a.shift >= 0 ? 1 : a.divisor);
    int const chunk = nthr * kRefItems;
    int const c0 = z * chunk;
    bool const counter = z == 0;
    T v[kRefItems];
#pragma unroll
    for (int i = 0; i < kRefItems; ++i)
    {
        int const o = c0 + i * nthr + tid;
        v[i] = o < a.width ? srow[static_cast<int64_t>(o) * a.src_stride_col] : T(-1);
    }
    T w[kCnt];
    if (counter)
    {
#pragma unroll
        for (int i = 0; i < kCnt; ++i)
        {
            int const o = chunk + i * nthr + tid;
            w[i] = o < a.width ? srow[static_cast<int64_t>(o) * a.src_stride_col] : T(-1);
        }
    }
#pragma unroll
    for (int i = 0; i < kRefItems; ++i)
    {
        int const o = c0 + i * nthr + tid;
        if (o < a.width)
            trow[o] = decode(v[i], a.shift, divisor);
    }
    if (!counter)
        return;
    int32_t cnt = 0;
#pragma unroll
    for (int i = 0; i < kRefItems; ++i)
        cnt += v[i] >= 0 ? 1 : 0;
#pragma unroll
    for (int i = 0; i < kCnt; ++i)
        cnt += w[i] >= 0 ? 1 : 0;
    for (int base = chunk + nthr * kCnt; base < a.width; base += nthr * kCnt)
    {
#pragma unroll
        for (int i = 0; i < kCnt; ++i)
        {
            int const o = base + i * nthr + tid;
            w[i] = o < a.width ? srow[static_cast<int64_t>(o) * a.src_stride_col] : T(-1);
        }
#pragma unroll
        for (int i = 0; i < kCnt; ++i)
            cnt += w[i] >= 0 ? 1 : 0;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    if ((tid & 31) == 0)
        warp_cnt[tid >> 5] = cnt;
    __syncthreads();
    if (tid == 0)
    {
        int32_t total = 0;
        for (int w2 = 0; w2 < (nthr >> 5); ++w2)
            total += warp_cnt[w2];
        a.counts[r] = static_cast<int64_t>(total);
    }
}

template <typename U>
__global__ void __launch_bounds__(32 * kMaxWarps) dspark_inputs_kernel(InArgs a)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int64_t const r = blockIdx.x;
    int const j = blockIdx.y;
    int const z = blockIdx.z;
    if (a.refresh && j == a.bq)
    {
        int const chunk = blockDim.x * kRefItems;
        int const ref_chunks = a.width > 0 ? (a.width + chunk - 1) / chunk : 1;
        if (z < ref_chunks)
        {
            if (a.src_i64)
                refresh_chunk_t<int64_t>(a, r, z);
            else
                refresh_chunk_t<int32_t>(a, r, z);
        }
        return;
    }
    int64_t const units = a.row_bytes / static_cast<int64_t>(sizeof(U));
    int64_t const u = static_cast<int64_t>(z) * blockDim.x + threadIdx.x;
    bool const scalars = j == 0 && z == 0;
    if (static_cast<int64_t>(z) * blockDim.x >= units && !scalars)
        return; // a chunk past this row's units
    U* dst = reinterpret_cast<U*>(a.noise + (r * a.bq + j) * a.row_bytes);
    if (j > 0)
    {
        // a mask row: no dependency on the request's accepted tokens
        if (u < units)
            dst[u] = reinterpret_cast<U const*>(a.mask)[u];
        return;
    }
    // j == 0: both chains in flight together (slot -> ctx_len only on the scalar CTA)
    int64_t const slot = scalars ? a.slots[r] : 0;
    int64_t const nacc = ld_i(a.nacc, a.nacc_i64, r);
    int64_t const ctx = scalars ? a.ctx_len[slot] : 0;
    int64_t const k = nacc - 1 > 0 ? nacc - 1 : 0;
    int64_t const bonus = ld_i(a.acc, a.acc_i64, r * a.acc_stride + k);
    if (u < units)
    {
        bool const valid = bonus >= 0 && bonus < a.vocab;
        dst[u] = valid ? reinterpret_cast<U const*>(a.table + bonus * a.row_bytes)[u] : U{};
    }
    if (scalars)
    {
        int64_t now = ctx + nacc;
        now = now < a.max_ctx ? now : a.max_ctx;
        if (threadIdx.x == 0)
            a.bonus[r] = bonus;
        for (int q = threadIdx.x; q < a.bq; q += blockDim.x)
            a.qpos[r * a.bq + q] = now + q;
        for (int c = threadIdx.x; c < a.kp1; c += blockDim.x)
            a.cpos[r * a.kp1 + c] = ctx + c;
    }
}

template <typename U>
int launch_t(InArgs const& a, int n, int pdl, int wpc, void* stream)
{
    int const thr = 32 * wpc;
    int64_t const units = a.row_bytes / static_cast<int64_t>(sizeof(U));
    int z = static_cast<int>((units + thr - 1) / thr);
    if (z < 1)
        z = 1;
    if (a.refresh)
    {
        int const ref = a.width > 0 ? (a.width + thr * kRefItems - 1) / (thr * kRefItems) : 1;
        z = z > ref ? z : ref;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n, a.bq + (a.refresh ? 1 : 0), z);
    cfg.blockDim = dim3(thr);
    cfg.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, dspark_inputs_kernel<U>, a));
}

int pick_and_launch(InArgs const& a, int n, int pdl, int wpc, void* stream)
{
    uintptr_t const al = reinterpret_cast<uintptr_t>(a.table) | reinterpret_cast<uintptr_t>(a.mask)
        | reinterpret_cast<uintptr_t>(a.noise) | static_cast<uintptr_t>(a.row_bytes);
    if ((al & 15) == 0)
        return launch_t<uint4>(a, n, pdl, wpc, stream);
    if ((al & 7) == 0)
        return launch_t<uint2>(a, n, pdl, wpc, stream);
    if ((al & 3) == 0)
        return launch_t<uint32_t>(a, n, pdl, wpc, stream);
    if ((al & 1) == 0)
        return launch_t<uint16_t>(a, n, pdl, wpc, stream);
    return launch_t<uint8_t>(a, n, pdl, wpc, stream);
}

int pow2_shift(int64_t d)
{
    if (d <= 0 || (d & (d - 1)) != 0)
        return -1;
    int s = 0;
    while ((int64_t(1) << s) != d)
        ++s;
    return s;
}

InArgs base_args(void const* acc, int acc_i64, int64_t acc_stride, void const* nacc, int nacc_i64, void const* slots,
    void const* ctx_len, int64_t max_ctx, void const* table, void const* mask, void* bonus, void* qpos, void* cpos,
    void* noise, int64_t row_bytes, int64_t vocab, int bq, int kp1)
{
    InArgs a = {};
    a.acc = acc;
    a.acc_i64 = acc_i64;
    a.acc_stride = acc_stride;
    a.nacc = nacc;
    a.nacc_i64 = nacc_i64;
    a.slots = static_cast<int64_t const*>(slots);
    a.ctx_len = static_cast<int64_t const*>(ctx_len);
    a.max_ctx = max_ctx;
    a.table = static_cast<unsigned char const*>(table);
    a.mask = static_cast<unsigned char const*>(mask);
    a.bonus = static_cast<int64_t*>(bonus);
    a.qpos = static_cast<int64_t*>(qpos);
    a.cpos = static_cast<int64_t*>(cpos);
    a.noise = static_cast<unsigned char*>(noise);
    a.row_bytes = row_bytes;
    a.vocab = vocab;
    a.bq = bq;
    a.kp1 = kp1;
    a.shift = -1;
    return a;
}
} // namespace

namespace
{
constexpr int kLaunchWarps = 4; // the measured variant (p1w4)
} // namespace

void invokeDsparkDrafterInputs(void const* accepted, bool acceptedIs64, int64_t acceptedStride, void const* nacc,
    bool naccIs64, int64_t const* slots, int64_t const* ctxLen, int64_t maxCtx, void const* table, void const* maskRow,
    int64_t* bonus, int64_t* qpos, int64_t* cpos, void* noise, int64_t rowBytes, int64_t vocab, int32_t n,
    int32_t blockSize, int32_t tokensPerRequest, cudaStream_t stream)
{
    if (n <= 0)
        return;
    TLLM_CHECK_WITH_INFO(blockSize >= 1 && tokensPerRequest >= 1 && rowBytes >= 0, "dspark drafter inputs: bad sizes");
    InArgs a = base_args(accepted, acceptedIs64 ? 1 : 0, acceptedStride, nacc, naccIs64 ? 1 : 0, slots, ctxLen, maxCtx,
        table, maskRow, bonus, qpos, cpos, noise, rowBytes, vocab, blockSize, tokensPerRequest);
    TLLM_CUDA_CHECK(static_cast<cudaError_t>(pick_and_launch(a, n, 1, kLaunchWarps, stream)));
}

void invokeDsparkDrafterInputsRefresh(void const* accepted, bool acceptedIs64, int64_t acceptedStride, void const* nacc,
    bool naccIs64, int64_t const* slots, int64_t const* ctxLen, int64_t maxCtx, void const* table, void const* maskRow,
    int64_t* bonus, int64_t* qpos, int64_t* cpos, void* noise, int64_t rowBytes, int64_t vocab, int32_t n,
    int32_t blockSize, int32_t tokensPerRequest, void const* src, bool srcIs64, int64_t srcStrideRow,
    int64_t srcStrideCol, int32_t* tables, int64_t tablesStride, int64_t* counts, int32_t width, int64_t divisor,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        n >= 1 && blockSize >= 1 && tokensPerRequest >= 1 && rowBytes >= 0 && divisor >= 1 && width >= 0,
        "dspark drafter inputs + refresh: bad sizes");
    InArgs a = base_args(accepted, acceptedIs64 ? 1 : 0, acceptedStride, nacc, naccIs64 ? 1 : 0, slots, ctxLen, maxCtx,
        table, maskRow, bonus, qpos, cpos, noise, rowBytes, vocab, blockSize, tokensPerRequest);
    a.refresh = 1;
    a.src = src;
    a.src_i64 = srcIs64 ? 1 : 0;
    a.src_stride_row = srcStrideRow;
    a.src_stride_col = srcStrideCol;
    a.tables = tables;
    a.tables_stride = tablesStride;
    a.counts = counts;
    a.width = width;
    a.divisor = divisor;
    a.shift = pow2_shift(divisor);
    if (!srcIs64 && divisor > 2147483647LL)
        a.shift = 31; // every clamped int32 value < 2^31 <= divisor: quotient 0, as the int64 decode
    TLLM_CUDA_CHECK(static_cast<cudaError_t>(pick_and_launch(a, n, 1, kLaunchWarps, stream)));
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
