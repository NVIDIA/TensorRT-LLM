/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The decode step's pre-layer chain in one launch: the replicated token-embedding gather,
 * the weightless embedding RMSNorm, the exact FP32 widening of its BF16 output (the residual stream)
 * and layer 0's gemma pre-QKV RMSNorm of that BF16 output (layer 0's attention input).
 *
 * The built-in chain is four kernels: ``_replicated_embedding_gather_kernel``, flashinfer's
 * CuTe-DSL ``RMSNormKernel`` (weight ones, weight_bias 0), the same kernel with layer 0's weight and
 * weight_bias 1, and the exact BF16 -> FP32 widening. This kernel computes the same values with the same
 * floating-point operations in the same order:
 *
 * - gather: integer moves of the table row (``-0.0`` written as ``+0.0``, a ``+0.0`` row for an id
 *   outside the table), as ``_replicated_embedding_gather_kernel``;
 * - each RMSNorm, as flashinfer's ``RMSNormKernel`` compiles for an 8192-wide BF16 row (one CTA of
 *   128 threads per row; ``ops/allreduce/drafter_hop.py`` documents the same statements for its
 *   fused-add sibling): thread ``t`` owns the 64 columns ``8 t + 1024 b + e``; one FMA chain
 *   ``s = fma(x, x, s)`` from ``+0`` over ``b`` then ``e`` ascending; a butterfly over the warp's
 *   lanes (offsets 1, 2, 4, 8, 16); the same butterfly over the four warp sums padded with
 *   ``+0``; ``rstd = rsqrt.approx.ftz(fma(s, 2^-13, eps))``; ``y = (x * rstd) * (w + weight_bias)``,
 *   rounded once to BF16. Every operation is round-to-nearest without contraction (``__fmul_rn`` /
 *   ``__fadd_rn`` / ``__fmaf_rn``), so the compiler can neither fuse nor reorder them;
 * - the widening: the BF16 bits shifted into an FP32 word (exact).
 *
 * PDL: launched with programmatic stream serialization; ``griddepcontrol.wait`` precedes the first
 * global read (the ids are a predecessor's output), and the dependents are released right after
 * it (a dependent reads this kernel's outputs only after its own wait, which returns once this grid
 * completed).
 *
 * The variants (``pre_variant``; ``pre`` is variant 0, the kernel above):
 *
 * - variant 1: the two norm weights are parameters (no predecessor writes them), so each thread
 *   loads its 2 x 8 weight vectors into registers before ``griddepcontrol.wait``; the kernel's
 *   dependent chain is then the ids load, the table gather and the stores. Same statements on the
 *   same values otherwise.
 * - variant 2: variant 1 with each row split over a cluster of four 32-thread CTAs, CTA ``c`` of
 *   the cluster being warp ``c`` of the 128-thread row: thread ``t = 32 c + lane`` owns the same
 *   64 columns and forms the same per-thread FMA chain and warp butterfly; lane ``c'`` of every CTA
 *   stores its warp sum into slot ``c`` of CTA ``c'``'s shared array (distributed shared memory),
 *   a cluster barrier (release / acquire) publishes the four slots, and every CTA then runs the
 *   second butterfly over the four warp sums padded with ``+0`` as variant 0 does. Bit for bit the
 *   variant-0 sums; no CTA touches another CTA's shared memory after the second cluster barrier.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <torch/library.h>

#include <cstdint>

namespace
{

constexpr int kHidden = 8192;
constexpr int kThreads = 128;
constexpr int kVec = 8;
constexpr int kBlocks = kHidden / (kThreads * kVec); // 8
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float bf16Bits(uint32_t bits16)
{
    return __uint_as_float(bits16 << 16);
}

__device__ __forceinline__ uint32_t roundBf16(float v)
{
    uint16_t h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return static_cast<uint32_t>(h);
}

__device__ __forceinline__ float rsqrtApproxFtz(float v)
{
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
    return r;
}

//! flashinfer's row reduction of one 128-thread row: the warp butterfly, the four warp sums
//! through shared memory (padded with +0 beyond them), the same butterfly; every thread gets it.
__device__ __forceinline__ float rowSum(float s, float* red)
{
#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1 << i));
    }
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    if (lane == 0)
    {
        red[warp] = s;
    }
    __syncthreads();
    float v = lane < kWarps ? red[lane] : 0.0f;
#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1 << i));
    }
    return v;
}

template <typename Id>
__global__ void __launch_bounds__(kThreads) pfPreKernel(uint16_t const* __restrict__ table, Id const* __restrict__ ids,
    int64_t vocab, uint16_t const* __restrict__ embedWeight, float embedEps, float embedBias,
    uint16_t const* __restrict__ qkvWeight, float qkvEps, float qkvBias, uint16_t* __restrict__ attnOut,
    float* __restrict__ residualOut)
{
    __shared__ float red[2][kWarps];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    int const row = blockIdx.x;
    int const t = threadIdx.x;
    int64_t const token = static_cast<int64_t>(ids[row]);
    bool const valid = token >= 0 && token < vocab;

    // The gather: 8 columns per 16-byte load at 8 t + 1024 b.
    float x[kBlocks][kVec];
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (valid)
        {
            v = *reinterpret_cast<uint4 const*>(table + token * kHidden + kVec * t + kThreads * kVec * b);
        }
        uint32_t const w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            uint32_t lo = w[k] & 0xffffu;
            uint32_t hi = w[k] >> 16;
            lo = (lo & 0x7fffu) == 0u ? 0u : lo;
            hi = (hi & 0x7fffu) == 0u ? 0u : hi;
            x[b][2 * k] = bf16Bits(lo);
            x[b][2 * k + 1] = bf16Bits(hi);
        }
    }

    // The embedding norm.
    float s = 0.0f;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
#pragma unroll
        for (int e = 0; e < kVec; ++e)
        {
            s = __fmaf_rn(x[b][e], x[b][e], s);
        }
    }
    float const rstd = rsqrtApproxFtz(__fmaf_rn(rowSum(s, red[0]), 0x1p-13f, embedEps));

    // The embedding norm's BF16 output (stored widened as the residual), and its sum of squares.
    float s2 = 0.0f;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        int const column = kVec * t + kThreads * kVec * b;
        uint4 const wv = *reinterpret_cast<uint4 const*>(embedWeight + column);
        uint32_t const ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float const g0 = __fadd_rn(bf16Bits(ww[k] & 0xffffu), embedBias);
            float const g1 = __fadd_rn(bf16Bits(ww[k] >> 16), embedBias);
            x[b][2 * k] = bf16Bits(roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k], rstd), g0)));
            x[b][2 * k + 1] = bf16Bits(roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k + 1], rstd), g1)));
        }
#pragma unroll
        for (int e = 0; e < kVec; ++e)
        {
            s2 = __fmaf_rn(x[b][e], x[b][e], s2);
        }
        float4* res = reinterpret_cast<float4*>(residualOut + static_cast<int64_t>(row) * kHidden + column);
        res[0] = make_float4(x[b][0], x[b][1], x[b][2], x[b][3]);
        res[1] = make_float4(x[b][4], x[b][5], x[b][6], x[b][7]);
    }

    // Layer 0's pre-QKV norm of the BF16 embedding.
    float const rstd2 = rsqrtApproxFtz(__fmaf_rn(rowSum(s2, red[1]), 0x1p-13f, qkvEps));
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        int const column = kVec * t + kThreads * kVec * b;
        uint4 const wv = *reinterpret_cast<uint4 const*>(qkvWeight + column);
        uint32_t const ww[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t packed[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float const g0 = __fadd_rn(bf16Bits(ww[k] & 0xffffu), qkvBias);
            float const g1 = __fadd_rn(bf16Bits(ww[k] >> 16), qkvBias);
            uint32_t const lo = roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k], rstd2), g0));
            uint32_t const hi = roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k + 1], rstd2), g1));
            packed[k] = lo | (hi << 16);
        }
        *reinterpret_cast<uint4*>(attnOut + static_cast<int64_t>(row) * kHidden + column)
            = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

//! ``rowSum`` over a row split across a cluster of ``kWarps`` one-warp CTAs (variant 2): the warp
//! butterfly, the warp sum stored into slot ``block_rank`` of every CTA's ``red``, a cluster barrier,
//! then ``rowSum``'s second butterfly over the four slots padded with +0. ``first``: wait for the entry
//! arrive (every CTA of the cluster has started) before the first remote store.
__device__ __forceinline__ float rowSumCluster(float s, float* red, bool first)
{
    namespace cg = cooperative_groups;
#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1 << i));
    }
    int const lane = threadIdx.x & 31;
    cg::cluster_group cluster = cg::this_cluster();
    int const warp = static_cast<int>(cluster.block_rank());
    if (first)
    {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
    if (lane < kWarps)
    {
        *cluster.map_shared_rank(red + warp, lane) = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    float v = lane < kWarps ? red[lane] : 0.0f;
#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1 << i));
    }
    return v;
}

//! Variants 1 (``kSplit`` 1: one 128-thread CTA per row) and 2 (``kSplit`` 4: a cluster of four
//! one-warp CTAs per row): ``pfPreKernel``'s statements with the norm weights loaded before the wait.
template <typename Id, int kSplit>
__global__ void __launch_bounds__(kThreads / kSplit) pfPreHoistKernel(uint16_t const* __restrict__ table,
    Id const* __restrict__ ids, int64_t vocab, uint16_t const* __restrict__ embedWeight, float embedEps,
    float embedBias, uint16_t const* __restrict__ qkvWeight, float qkvEps, float qkvBias,
    uint16_t* __restrict__ attnOut, float* __restrict__ residualOut)
{
    static_assert(kSplit == 1 || kSplit == kWarps, "one CTA per row, or one CTA per warp of the row");
    __shared__ float red[2][kWarps];
    int const row = blockIdx.x / kSplit;
    int const t
        = kSplit == 1 ? static_cast<int>(threadIdx.x) : static_cast<int>((blockIdx.x % kSplit) * 32 + threadIdx.x);
    if constexpr (kSplit > 1)
    {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
    // The norm weights: parameters, read before the dependency wait.
    uint4 ew[kBlocks];
    uint4 qw[kBlocks];
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        int const column = kVec * t + kThreads * kVec * b;
        ew[b] = *reinterpret_cast<uint4 const*>(embedWeight + column);
        qw[b] = *reinterpret_cast<uint4 const*>(qkvWeight + column);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    int64_t const token = static_cast<int64_t>(ids[row]);
    bool const valid = token >= 0 && token < vocab;

    float x[kBlocks][kVec];
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (valid)
        {
            v = *reinterpret_cast<uint4 const*>(table + token * kHidden + kVec * t + kThreads * kVec * b);
        }
        uint32_t const w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            uint32_t lo = w[k] & 0xffffu;
            uint32_t hi = w[k] >> 16;
            lo = (lo & 0x7fffu) == 0u ? 0u : lo;
            hi = (hi & 0x7fffu) == 0u ? 0u : hi;
            x[b][2 * k] = bf16Bits(lo);
            x[b][2 * k + 1] = bf16Bits(hi);
        }
    }

    float s = 0.0f;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
#pragma unroll
        for (int e = 0; e < kVec; ++e)
        {
            s = __fmaf_rn(x[b][e], x[b][e], s);
        }
    }
    float sum;
    if constexpr (kSplit == 1)
    {
        sum = rowSum(s, red[0]);
    }
    else
    {
        sum = rowSumCluster(s, red[0], true);
    }
    float const rstd = rsqrtApproxFtz(__fmaf_rn(sum, 0x1p-13f, embedEps));

    float s2 = 0.0f;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        int const column = kVec * t + kThreads * kVec * b;
        uint32_t const ww[4] = {ew[b].x, ew[b].y, ew[b].z, ew[b].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float const g0 = __fadd_rn(bf16Bits(ww[k] & 0xffffu), embedBias);
            float const g1 = __fadd_rn(bf16Bits(ww[k] >> 16), embedBias);
            x[b][2 * k] = bf16Bits(roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k], rstd), g0)));
            x[b][2 * k + 1] = bf16Bits(roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k + 1], rstd), g1)));
        }
#pragma unroll
        for (int e = 0; e < kVec; ++e)
        {
            s2 = __fmaf_rn(x[b][e], x[b][e], s2);
        }
        float4* res = reinterpret_cast<float4*>(residualOut + static_cast<int64_t>(row) * kHidden + column);
        res[0] = make_float4(x[b][0], x[b][1], x[b][2], x[b][3]);
        res[1] = make_float4(x[b][4], x[b][5], x[b][6], x[b][7]);
    }

    float sum2;
    if constexpr (kSplit == 1)
    {
        sum2 = rowSum(s2, red[1]);
    }
    else
    {
        sum2 = rowSumCluster(s2, red[1], false);
    }
    float const rstd2 = rsqrtApproxFtz(__fmaf_rn(sum2, 0x1p-13f, qkvEps));
#pragma unroll
    for (int b = 0; b < kBlocks; ++b)
    {
        int const column = kVec * t + kThreads * kVec * b;
        uint32_t const ww[4] = {qw[b].x, qw[b].y, qw[b].z, qw[b].w};
        uint32_t packed[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float const g0 = __fadd_rn(bf16Bits(ww[k] & 0xffffu), qkvBias);
            float const g1 = __fadd_rn(bf16Bits(ww[k] >> 16), qkvBias);
            uint32_t const lo = roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k], rstd2), g0));
            uint32_t const hi = roundBf16(__fmul_rn(__fmul_rn(x[b][2 * k + 1], rstd2), g1));
            packed[k] = lo | (hi << 16);
        }
        *reinterpret_cast<uint4*>(attnOut + static_cast<int64_t>(row) * kHidden + column)
            = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

void preVariant(at::Tensor const& table, at::Tensor const& ids, at::Tensor const& embedWeight, double embedEps,
    double embedBias, at::Tensor const& qkvWeight, double qkvEps, double qkvBias, at::Tensor const& attnOut,
    at::Tensor const& residualOut, bool pdl, int64_t variant);

void pre(at::Tensor const& table, at::Tensor const& ids, at::Tensor const& embedWeight, double embedEps,
    double embedBias, at::Tensor const& qkvWeight, double qkvEps, double qkvBias, at::Tensor const& attnOut,
    at::Tensor const& residualOut, bool pdl)
{
    preVariant(table, ids, embedWeight, embedEps, embedBias, qkvWeight, qkvEps, qkvBias, attnOut, residualOut, pdl, 0);
}

void preVariant(at::Tensor const& table, at::Tensor const& ids, at::Tensor const& embedWeight, double embedEps,
    double embedBias, at::Tensor const& qkvWeight, double qkvEps, double qkvBias, at::Tensor const& attnOut,
    at::Tensor const& residualOut, bool pdl, int64_t variant)
{
    TORCH_CHECK(variant >= 0 && variant <= 2, "pf pre: variant 0, 1 or 2");
    c10::cuda::CUDAGuard guard(table.device());
    TORCH_CHECK(table.is_cuda() && ids.is_cuda() && embedWeight.is_cuda() && qkvWeight.is_cuda() && attnOut.is_cuda()
            && residualOut.is_cuda(),
        "pf pre: CUDA tensors only");
    TORCH_CHECK(
        table.scalar_type() == at::kBFloat16 && table.dim() == 2 && table.size(1) == kHidden && table.is_contiguous(),
        "pf pre: the table must be a contiguous [vocab, 8192] BF16 tensor");
    TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "pf pre: int32 or int64 ids");
    TORCH_CHECK(ids.is_contiguous() && ids.dim() == 1, "pf pre: ids must be a contiguous 1-D tensor");
    int64_t const rows = ids.numel();
    TORCH_CHECK(rows >= 1 && rows <= 65535, "pf pre: 1 .. 65535 rows");
    for (at::Tensor const* w : {&embedWeight, &qkvWeight})
    {
        TORCH_CHECK(w->scalar_type() == at::kBFloat16 && w->numel() == kHidden && w->is_contiguous(),
            "pf pre: the norm weights must be contiguous [8192] BF16 tensors");
    }
    TORCH_CHECK(attnOut.scalar_type() == at::kBFloat16 && attnOut.dim() == 2 && attnOut.size(0) == rows
            && attnOut.size(1) == kHidden && attnOut.is_contiguous(),
        "pf pre: attn_out must be a contiguous [rows, 8192] BF16 tensor");
    TORCH_CHECK(residualOut.scalar_type() == at::kFloat && residualOut.dim() == 2 && residualOut.size(0) == rows
            && residualOut.size(1) == kHidden && residualOut.is_contiguous(),
        "pf pre: residual_out must be a contiguous [rows, 8192] FP32 tensor");
    for (at::Tensor const* x : {&table, &embedWeight, &qkvWeight, &attnOut, &residualOut})
    {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x->data_ptr()) % 16 == 0, "pf pre: misaligned operand");
    }
    int const split = variant == 2 ? kWarps : 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(rows * split));
    config.blockDim = dim3(kThreads / split);
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = static_cast<unsigned>(split);
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = variant == 2 ? 2 : 1;
    auto const* tablePtr = reinterpret_cast<uint16_t const*>(table.data_ptr());
    auto const* embedPtr = reinterpret_cast<uint16_t const*>(embedWeight.data_ptr());
    auto const* qkvPtr = reinterpret_cast<uint16_t const*>(qkvWeight.data_ptr());
    auto* attnPtr = reinterpret_cast<uint16_t*>(attnOut.data_ptr());
    auto* residualPtr = reinterpret_cast<float*>(residualOut.data_ptr());
    int64_t const vocab = table.size(0);
    auto const launch = [&](auto kernel, auto const* idsPtr)
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, tablePtr, idsPtr, vocab, embedPtr,
            static_cast<float>(embedEps), static_cast<float>(embedBias), qkvPtr, static_cast<float>(qkvEps),
            static_cast<float>(qkvBias), attnPtr, residualPtr));
    };
    if (ids.scalar_type() == at::kInt)
    {
        auto const* idsPtr = reinterpret_cast<int32_t const*>(ids.data_ptr());
        if (variant == 0)
        {
            launch(pfPreKernel<int32_t>, idsPtr);
        }
        else if (variant == 1)
        {
            launch(pfPreHoistKernel<int32_t, 1>, idsPtr);
        }
        else
        {
            launch(pfPreHoistKernel<int32_t, kWarps>, idsPtr);
        }
    }
    else
    {
        auto const* idsPtr = reinterpret_cast<int64_t const*>(ids.data_ptr());
        if (variant == 0)
        {
            launch(pfPreKernel<int64_t>, idsPtr);
        }
        else if (variant == 1)
        {
            launch(pfPreHoistKernel<int64_t, 1>, idsPtr);
        }
        else
        {
            launch(pfPreHoistKernel<int64_t, kWarps>, idsPtr);
        }
    }
}

} // namespace

TORCH_LIBRARY(musespark_open_pf_pre, m)
{
    m.def(
        "pre(Tensor table, Tensor ids, Tensor embed_weight, float embed_eps, float embed_bias, Tensor qkv_weight, "
        "float qkv_eps, float qkv_bias, Tensor(a!) attn_out, Tensor(b!) residual_out, bool pdl) -> ()");
    m.def(
        "pre_variant(Tensor table, Tensor ids, Tensor embed_weight, float embed_eps, float embed_bias, "
        "Tensor qkv_weight, float qkv_eps, float qkv_bias, Tensor(a!) attn_out, Tensor(b!) residual_out, bool pdl, "
        "int variant) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_pf_pre, CUDA, m)
{
    m.impl("pre", &pre);
    m.impl("pre_variant", &preVariant);
}
