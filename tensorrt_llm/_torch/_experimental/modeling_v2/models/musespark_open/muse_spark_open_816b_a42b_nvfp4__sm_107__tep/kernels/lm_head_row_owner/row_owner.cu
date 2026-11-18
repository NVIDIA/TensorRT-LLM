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

// The owner side of the row-owned LM head (``lm_head_row_owner``, ``owner.py``).
//
// Every rank's LM-head GEMM stores its [rows, shard] BF16 output straight into the
// owner ranks' pages (row stride ``ld`` elements): the rows of request j go to the
// rank that owns j. Between two head calls every word of every page holds the
// sentinel 0xFFFF in each BF16 half (a NaN no GEMM epilogue produces: the round-to-
// nearest conversion of any NaN is 0x7FFF), so a word that is not the sentinel has
// been published with its final value (the certified producer stores each element
// once, from registers).
//
// ownerCapKernel (owner): polls each 16-byte word of its live rows until no half is
// the sentinel, writes the soft-capped FP32 logits (the torch chain's statements,
// as the split sampler's capOf), and stores the sentinel back. Rows from `liveRows`
// to `outRows` (the owner slice's padding) are written as zeros.
// pollGatherKernel (fallback): copies every row of the stitched range into a
// contiguous BF16 row, polling each word, without resetting it (other ranks read the
// same words; the owners reset after a barrier with resetKernel).
//
// A poll that waits past the timeout is a protocol failure (a peer whose GEMM never
// came): the kernel records it in `status` and traps, which ends the CUDA context
// loudly instead of returning data.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace
{

//! Four BF16 pairs of 0xFFFF: an unpublished 16-byte word.
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
//! Status bit: a poll waited past the timeout (the kernel then trapped).
constexpr int kPollTimeout = 1;
constexpr int kThreads = 256;
//! 16-byte words each thread handles per row tile.
constexpr int kWordsPerThread = 4;

struct Cap
{
    float multiplier;
    float invCap;
    float cap;
};

//! The torch chain logits.float() * m, * fp32(1 / cap), tanh, cap * t, statement for statement.
__device__ __forceinline__ float capOf(float x, Cap const& c)
{
    float y = __fmul_rn(x, c.multiplier);
    y = __fmul_rn(y, c.invCap);
    y = tanhf(y);
    return __fmul_rn(c.cap, y);
}

__device__ __forceinline__ uint4 loadRelaxedSys(uint4 const* address)
{
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(address)
                 : "memory");
    return v;
}

__device__ __forceinline__ void storeRelaxedSys(uint4* address, uint4 v)
{
    asm volatile(
        "st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ unsigned long long globalTimerNs()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

//! Whether no BF16 half of the word is the sentinel.
__device__ __forceinline__ bool published(uint4 w)
{
    return (__vcmpeq2(w.x, kSentinel) | __vcmpeq2(w.y, kSentinel) | __vcmpeq2(w.z, kSentinel)
               | __vcmpeq2(w.w, kSentinel))
        == 0u;
}

//! Spin until the word at `address` is published; past `timeoutNs`, record and trap.
__device__ __noinline__ uint4 awaitPublished(uint4 const* address, int* status, unsigned long long timeoutNs)
{
    unsigned long long const start = globalTimerNs();
    for (int spins = 0;; ++spins)
    {
        uint4 const w = loadRelaxedSys(address);
        if (published(w))
        {
            return w;
        }
        if (spins >= 64)
        {
            __nanosleep(128);
            if (globalTimerNs() - start > timeoutNs)
            {
                atomicOr(status, kPollTimeout);
                __threadfence_system();
                __trap();
            }
        }
    }
}

__device__ __forceinline__ float bf16Lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf16Hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

struct OwnerParams
{
    //! The owner's page rows: row r at pages + r * ldWords (16-byte words).
    uint4* pages;
    int64_t ldWords;
    int liveRows;
    int outRows;
    //! 16-byte words of one row (vocab / 8).
    int words;
    //! FP32 [outRows, vocab] contiguous, or null (drain: poll and reset only).
    float4* out;
    Cap cap;
    int* status;
    unsigned long long timeoutNs;
};

//! Grid (outRows, ceil(words / (kThreads * kWordsPerThread))), block kThreads.
template <bool POLL, bool RESET>
__global__ void __launch_bounds__(kThreads) ownerCapKernel(OwnerParams p)
{
    int const row = blockIdx.x;
    int const base = blockIdx.y * kThreads * kWordsPerThread + threadIdx.x;
    float4* outRow = p.out == nullptr ? nullptr : p.out + static_cast<int64_t>(row) * p.words * 2;
    if (row >= p.liveRows)
    {
        if (outRow != nullptr)
        {
#pragma unroll
            for (int k = 0; k < kWordsPerThread; ++k)
            {
                int const w = base + k * kThreads;
                if (w < p.words)
                {
                    outRow[2 * w] = make_float4(0.f, 0.f, 0.f, 0.f);
                    outRow[2 * w + 1] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
        }
        return;
    }
    uint4* pageRow = p.pages + static_cast<int64_t>(row) * p.ldWords;
    uint4 v[kWordsPerThread];
#pragma unroll
    for (int k = 0; k < kWordsPerThread; ++k)
    {
        int const w = base + k * kThreads;
        if (w < p.words)
        {
            v[k] = loadRelaxedSys(pageRow + w);
        }
    }
#pragma unroll
    for (int k = 0; k < kWordsPerThread; ++k)
    {
        int const w = base + k * kThreads;
        if (w >= p.words)
        {
            continue;
        }
        if (POLL && !published(v[k]))
        {
            v[k] = awaitPublished(pageRow + w, p.status, p.timeoutNs);
        }
        if (outRow != nullptr)
        {
            uint4 const x = v[k];
            outRow[2 * w] = make_float4(capOf(bf16Lo(x.x), p.cap), capOf(bf16Hi(x.x), p.cap), capOf(bf16Lo(x.y), p.cap),
                capOf(bf16Hi(x.y), p.cap));
            outRow[2 * w + 1] = make_float4(capOf(bf16Lo(x.z), p.cap), capOf(bf16Hi(x.z), p.cap),
                capOf(bf16Lo(x.w), p.cap), capOf(bf16Hi(x.w), p.cap));
        }
        if (RESET)
        {
            storeRelaxedSys(pageRow + w, make_uint4(kSentinel, kSentinel, kSentinel, kSentinel));
        }
    }
}

struct GatherParams
{
    //! Row r of the stitched range at pages + r * ldWords.
    uint4 const* pages;
    int64_t ldWords;
    int rows;
    int words;
    //! BF16 [rows, vocab] contiguous.
    uint4* out;
    int* status;
    unsigned long long timeoutNs;
};

template <bool POLL>
__global__ void __launch_bounds__(kThreads) pollGatherKernel(GatherParams p)
{
    int const row = blockIdx.x;
    int const base = blockIdx.y * kThreads * kWordsPerThread + threadIdx.x;
    uint4 const* pageRow = p.pages + static_cast<int64_t>(row) * p.ldWords;
    uint4* outRow = p.out + static_cast<int64_t>(row) * p.words;
    uint4 v[kWordsPerThread];
#pragma unroll
    for (int k = 0; k < kWordsPerThread; ++k)
    {
        int const w = base + k * kThreads;
        if (w < p.words)
        {
            v[k] = loadRelaxedSys(pageRow + w);
        }
    }
#pragma unroll
    for (int k = 0; k < kWordsPerThread; ++k)
    {
        int const w = base + k * kThreads;
        if (w >= p.words)
        {
            continue;
        }
        if (POLL && !published(v[k]))
        {
            v[k] = awaitPublished(pageRow + w, p.status, p.timeoutNs);
        }
        outRow[w] = v[k];
    }
}

void checkPages(at::Tensor const& pages, int64_t rows, int64_t vocab)
{
    TORCH_CHECK(pages.is_cuda() && pages.scalar_type() == at::kShort && pages.dim() == 2,
        "pages must be an int16 [rows, ld] CUDA view");
    TORCH_CHECK(pages.size(0) >= rows && pages.size(1) >= vocab && pages.stride(1) == 1, "pages cover the rows");
    TORCH_CHECK(pages.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(pages.data_ptr()) % 16 == 0,
        "pages rows must be 16-byte aligned");
}

unsigned long long timeoutNs(int64_t timeoutMs)
{
    TORCH_CHECK(timeoutMs > 0, "timeout must be positive");
    return static_cast<unsigned long long>(timeoutMs) * 1000000ull;
}

Cap capOfArgs(double multiplier, double invCap, double cap)
{
    // Python floats holding FP32 values: the conversions are exact.
    return Cap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
}

dim3 gridOf(int64_t rows, int64_t words)
{
    int64_t const perBlock = static_cast<int64_t>(kThreads) * kWordsPerThread;
    return dim3(static_cast<unsigned>(rows), static_cast<unsigned>((words + perBlock - 1) / perBlock));
}

//! Owner: poll + soft cap + reset of `live_rows` page rows into `out` [out_rows, vocab] FP32
//! (rows past `live_rows` are zeros); without `out`, poll + reset only (drain); with
//! `poll` false, the words are taken as published (the fallback's reset after its barrier).
void ownerCap(at::Tensor pages, int64_t liveRows, std::optional<at::Tensor> out, int64_t vocab, double multiplier,
    double invCap, double cap, at::Tensor status, int64_t timeoutMs, bool poll)
{
    at::cuda::CUDAGuard guard(pages.device());
    TORCH_CHECK(vocab > 0 && vocab % 8 == 0, "vocab must be a multiple of 8");
    checkPages(pages, liveRows, vocab);
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    OwnerParams p{};
    p.pages = reinterpret_cast<uint4*>(pages.data_ptr());
    p.ldWords = pages.stride(0) / 8;
    p.liveRows = static_cast<int>(liveRows);
    p.words = static_cast<int>(vocab / 8);
    p.cap = capOfArgs(multiplier, invCap, cap);
    p.status = status.data_ptr<int>();
    p.timeoutNs = timeoutNs(timeoutMs);
    int64_t outRows = liveRows;
    if (out.has_value())
    {
        at::Tensor const& o = *out;
        TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kFloat && o.dim() == 2 && o.is_contiguous(),
            "out must be a contiguous FP32 [rows, vocab] tensor");
        TORCH_CHECK(o.size(1) == vocab && o.size(0) >= liveRows, "out covers the live rows");
        outRows = o.size(0);
        p.out = reinterpret_cast<float4*>(o.data_ptr());
    }
    p.outRows = static_cast<int>(outRows);
    if (outRows == 0)
    {
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 const grid = gridOf(outRows, p.words);
    if (poll)
    {
        ownerCapKernel<true, true><<<grid, kThreads, 0, stream>>>(p);
    }
    else
    {
        TORCH_CHECK(!out.has_value(), "an unpolled call only resets");
        ownerCapKernel<false, true><<<grid, kThreads, 0, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! Fallback: every row of the stitched range `pages` [rows, ld] into `out` [rows, vocab] BF16, polled, not reset.
void pollGather(at::Tensor pages, at::Tensor out, at::Tensor status, int64_t timeoutMs)
{
    at::cuda::CUDAGuard guard(pages.device());
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous() && out.element_size() == 2,
        "out must be a contiguous 16-bit [rows, vocab] tensor");
    int64_t const rows = out.size(0);
    int64_t const vocab = out.size(1);
    TORCH_CHECK(vocab % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out rows of 16-byte words");
    checkPages(pages, rows, vocab);
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    if (rows == 0)
    {
        return;
    }
    GatherParams p{};
    p.pages = reinterpret_cast<uint4 const*>(pages.data_ptr());
    p.ldWords = pages.stride(0) / 8;
    p.rows = static_cast<int>(rows);
    p.words = static_cast<int>(vocab / 8);
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.status = status.data_ptr<int>();
    p.timeoutNs = timeoutNs(timeoutMs);
    pollGatherKernel<true><<<gridOf(rows, p.words), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

TORCH_LIBRARY(musespark_open_sm107_row_owner, m)
{
    m.def(
        "owner_cap(Tensor(a!) pages, int live_rows, Tensor(b!)? out, int vocab, float multiplier, float inv_cap, "
        "float cap, Tensor(c!) status, int timeout_ms, bool poll) -> ()");
    m.def("poll_gather(Tensor pages, Tensor(a!) out, Tensor(b!) status, int timeout_ms) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_sm107_row_owner, CUDA, m)
{
    m.impl("owner_cap", &ownerCap);
    m.impl("poll_gather", &pollGather);
}
