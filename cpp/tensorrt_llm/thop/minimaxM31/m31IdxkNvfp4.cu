// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MiniMax-M3.1 NVFP4 index-K cache (agg TP2, opt-in): the writer and the prefill gather.
//
// Page layout (the vLLM / MSA Q8KV4 contract, one 128-token page, one index head, head_dim 128):
//   bytes [0, 8192): packed E2M1 codes, token t at t * 64, low nibble = even element;
//   bytes [8192, 9216): E4M3 scales of 16-element groups, token t group g at 8192 + t * 8 + g (linear).
// No global scale. A page may sit at any 16-byte-aligned stride.
//
// Writer (unit-scale recipe of vLLM's fused qknorm/rope/kv-insert kernel, for inputs exactly representable in bf16:
// bf16 rows, or E4M3 rows, which are a subset):
//   sf = E4M3_RN_satfinite(max(RN(amax * RN(1/6)), 2^-9)); q = RZ(x * rcp.approx.ftz(sf));
//   code = cvt.rn.satfinite.e2m1x2.f32(q); a zero magnitude is stored as +0.
// Gather: each element becomes E4M3_RN_satfinite(e2m1 * scale) (exact fp32 product, one rounding), the key the
// Q8KV4 decode indexer dequantizes to off SM107, so the prefill proxy scores the same keys vLLM's prefill does.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/library.h>

#include <cstdint>

namespace
{

constexpr int kPage = 128;
constexpr int kDim = 128;
constexpr int kDataBytes = kPage * kDim / 2;                         // 8192
constexpr int kScaleBytesPerToken = kDim / 16;                       // 8
constexpr int kPageBytes = kDataBytes + kPage * kScaleBytesPerToken; // 9216

__device__ __forceinline__ float rcpApproxFtz(float a)
{
    float b;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// Zero is stored as +0: clear the sign of every nibble whose magnitude is 0.
__device__ __forceinline__ uint16_t clearNegZeroNibbles(uint16_t packed)
{
    return packed & (((packed & 0x7777u) + 0x7777u) | 0x7777u);
}

__device__ __forceinline__ float loadElem(__nv_bfloat16 const* p)
{
    return __bfloat162float(*p);
}

__device__ __forceinline__ float loadElem(__nv_fp8_e4m3 const* p)
{
    return float(*p);
}

// One warp per token: lane l holds elements [4l, 4l + 4) of group l / 4.
template <typename T>
__global__ void writeKernel(T const* __restrict__ rows, int64_t rowStride, int64_t const* __restrict__ loc64,
    int32_t const* __restrict__ loc32, int numTokens, uint8_t* __restrict__ pages, int64_t pageStride,
    int64_t rowsPageStride, bool rowsBySlot)
{
    int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int const lane = threadIdx.x & 31;
    if (warp >= numTokens)
    {
        return;
    }
    int64_t const slot = loc64 != nullptr ? loc64[warp] : static_cast<int64_t>(loc32[warp]);
    if (slot < 0)
    {
        return;
    }
    // rowsBySlot: the rows are an FP8 index-K pool [pages, 1, 128, 128] (the producer's sink) read at the token's slot.
    T const* src = rowsBySlot ? rows + (slot / kPage) * rowsPageStride + (slot % kPage) * rowStride + lane * 4
                              : rows + warp * rowStride + lane * 4;
    float x[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        x[i] = loadElem(src + i);
        amax = fmaxf(amax, fabsf(x[i]));
    }
    // Group max over the 4 lanes of the 16-element group.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    constexpr float kRcp6 = 0x1.555556p-3f; // RN(1 / 6)
    __nv_fp8_e4m3 const sf(fmaxf(__fmul_rn(amax, kRcp6), 0x1p-9f));
    float const sfRcp = rcpApproxFtz(float(sf));
    float q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        q[i] = __fmul_rz(x[i], sfRcp);
    }
    uint16_t packed;
    // Low nibble = even element; cvt packs its first source in the high nibble.
    asm("{\n"
        "  .reg .b8 lo, hi;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3;\n"
        "  mov.b16 %0, {lo, hi};\n"
        "}"
        : "=h"(packed)
        : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]));
    packed = clearNegZeroNibbles(packed);
    int64_t const page = slot / kPage;
    int const t = static_cast<int>(slot % kPage);
    uint8_t* base = pages + page * pageStride;
    reinterpret_cast<uint16_t*>(base + t * (kDim / 2))[lane] = packed;
    if ((lane & 3) == 0)
    {
        base[kDataBytes + t * kScaleBytesPerToken + (lane >> 2)] = sf.__x;
    }
}

// E2M1 code -> value, arithmetically (no table): magnitude m = code & 7 is 0.5 * m for m < 2, else
// (1 + (m & 1) / 2) * 2^((m >> 1) - 1); bit 3 is the sign, applied by a multiply so code 8 stays -0.
__device__ __forceinline__ float e2m1ToFloat(uint32_t code)
{
    uint32_t const m = code & 7u;
    float const normal = __uint_as_float((((m >> 1) + 126u) << 23) | ((m & 1u) << 22));
    float const mag = m < 2u ? 0.5f * static_cast<float>(m) : normal;
    return mag * ((code & 8u) ? -1.f : 1.f);
}

// Two E2M1 codes (low nibble = even element) times an E4M3 scale -> two E4M3 bytes (even element low): the exact fp32
// product, clamped to +-448, rounded once.
__device__ __forceinline__ uint32_t dequantPair(uint32_t byte, float scale)
{
    float lo = e2m1ToFloat(byte & 0xfu) * scale;
    float hi = e2m1ToFloat(byte >> 4) * scale;
    lo = fminf(fmaxf(lo, -448.f), 448.f);
    hi = fminf(fmaxf(hi, -448.f), 448.f);
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

// One block per gathered page, 512 threads: thread i takes packed bytes [16 i, 16 i + 16) = elements
// [32 (i % 4), 32 (i % 4) + 32) of token i / 4 (two 16-element groups) and stores 32 E4M3 bytes.
__global__ void gatherKernel(uint8_t const* __restrict__ pages, int64_t pageStride, int32_t const* __restrict__ pageIds,
    __nv_fp8_e4m3* __restrict__ out)
{
    int const dst = blockIdx.x;
    int const tid = threadIdx.x;
    int const token = tid >> 2;
    int const quarter = tid & 3;
    uint4* o = reinterpret_cast<uint4*>(out + static_cast<int64_t>(dst) * kPage * kDim + token * kDim + quarter * 32);
    int const page = pageIds[dst];
    if (page < 0)
    {
        o[0] = make_uint4(0, 0, 0, 0);
        o[1] = make_uint4(0, 0, 0, 0);
        return;
    }
    uint8_t const* base = pages + static_cast<int64_t>(page) * pageStride;
    uint4 const d = *reinterpret_cast<uint4 const*>(base + tid * 16);
    uint16_t const sc
        = *reinterpret_cast<uint16_t const*>(base + kDataBytes + token * kScaleBytesPerToken + quarter * 2);
    __nv_fp8_e4m3 s0, s1;
    s0.__x = static_cast<uint8_t>(sc & 0xffu);
    s1.__x = static_cast<uint8_t>(sc >> 8);
    float const scale0 = float(s0);
    float const scale1 = float(s1);
    uint32_t const in[4] = {d.x, d.y, d.z, d.w};
    uint32_t res[8];
#pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        // Word w holds elements [8 w, 8 w + 8): group 0 for w < 2, group 1 after.
        float const scale = w < 2 ? scale0 : scale1;
        uint32_t const v = in[w];
        res[2 * w] = dequantPair(v & 0xffu, scale) | (dequantPair((v >> 8) & 0xffu, scale) << 16);
        res[2 * w + 1] = dequantPair((v >> 16) & 0xffu, scale) | (dequantPair(v >> 24, scale) << 16);
    }
    o[0] = make_uint4(res[0], res[1], res[2], res[3]);
    o[1] = make_uint4(res[4], res[5], res[6], res[7]);
}

void checkPages(at::Tensor const& pages)
{
    TORCH_CHECK(pages.is_cuda() && pages.scalar_type() == at::kByte, "NVFP4 index-K pages must be a CUDA uint8 tensor");
    TORCH_CHECK(pages.numel() > 0 && pages.dim() >= 2, "NVFP4 index-K pages must be [pages, ...]");
    int64_t perPage = 1;
    for (int d = 1; d < pages.dim(); ++d)
    {
        perPage *= pages.size(d);
    }
    TORCH_CHECK(perPage == kPageBytes, "an NVFP4 index-K page must hold 9216 bytes, got ", perPage);
    // Within a page the bytes must be contiguous; the page stride is free (16-byte aligned).
    int64_t expect = 1;
    for (int d = pages.dim() - 1; d >= 1; --d)
    {
        TORCH_CHECK(pages.size(d) == 1 || pages.stride(d) == expect, "an NVFP4 index-K page must be contiguous");
        expect *= pages.size(d);
    }
    TORCH_CHECK(pages.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(pages.data_ptr()) % 16 == 0,
        "NVFP4 index-K pages must be 16-byte aligned");
}

void launchWrite(at::Tensor const& rows, int64_t rowStride, int64_t rowsPageStride, bool rowsBySlot,
    at::Tensor const& outCacheLoc, int numTokens, at::Tensor& pages);

void nvfp4IdxkWrite(at::Tensor const& rows, at::Tensor const& outCacheLoc, at::Tensor pages)
{
    checkPages(pages);
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.size(1) == kDim && rows.stride(1) == 1,
        "index-K rows must be [T, 128] with unit inner stride");
    TORCH_CHECK(rows.scalar_type() == at::kBFloat16 || rows.scalar_type() == at::kFloat8_e4m3fn,
        "index-K rows must be bf16 or float8_e4m3fn");
    TORCH_CHECK(rows.stride(0) % 8 == 0 || rows.size(0) <= 1, "index-K row stride must keep 8-byte alignment");
    TORCH_CHECK(outCacheLoc.is_cuda() && outCacheLoc.dim() == 1 && outCacheLoc.is_contiguous(),
        "out_cache_loc must be a contiguous CUDA vector");
    TORCH_CHECK(outCacheLoc.scalar_type() == at::kInt || outCacheLoc.scalar_type() == at::kLong,
        "out_cache_loc must be int32 or int64");
    int const numTokens = static_cast<int>(rows.size(0));
    TORCH_CHECK(outCacheLoc.size(0) >= numTokens, "out_cache_loc is shorter than the rows");
    launchWrite(rows, rows.stride(0), 0, false, outCacheLoc, numTokens, pages);
}

// The producer's FP8 sink [pages, 1, 128, 128] holds each token's E4M3 index-K at its slot: quantize the first
// num_tokens slots of out_cache_loc from there (slots < 0 skipped).
void nvfp4IdxkWriteFromSink(at::Tensor const& sink, at::Tensor const& outCacheLoc, int64_t numTokens, at::Tensor pages)
{
    checkPages(pages);
    TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kFloat8_e4m3fn && sink.dim() == 4 && sink.size(1) == 1
            && sink.size(2) == kPage && sink.size(3) == kDim && sink.stride(3) == 1 && sink.stride(2) == kDim,
        "the index-K sink must be FP8 [pages, 1, 128, 128] with contiguous rows");
    TORCH_CHECK(sink.size(0) <= pages.size(0), "the index-K sink holds more pages than the NVFP4 cache");
    TORCH_CHECK(outCacheLoc.is_cuda() && outCacheLoc.dim() == 1 && outCacheLoc.is_contiguous()
            && outCacheLoc.size(0) >= numTokens && numTokens >= 0,
        "out_cache_loc must be a contiguous CUDA vector of at least num_tokens slots");
    TORCH_CHECK(outCacheLoc.scalar_type() == at::kInt || outCacheLoc.scalar_type() == at::kLong,
        "out_cache_loc must be int32 or int64");
    launchWrite(sink, sink.stride(2), sink.stride(0), true, outCacheLoc, static_cast<int>(numTokens), pages);
}

void launchWrite(at::Tensor const& rows, int64_t rowStride, int64_t rowsPageStride, bool rowsBySlot,
    at::Tensor const& outCacheLoc, int numTokens, at::Tensor& pages)
{
    if (numTokens == 0)
    {
        return;
    }
    at::cuda::CUDAGuard guard(rows.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t const* loc64 = outCacheLoc.scalar_type() == at::kLong ? outCacheLoc.data_ptr<int64_t>() : nullptr;
    int32_t const* loc32 = outCacheLoc.scalar_type() == at::kInt ? outCacheLoc.data_ptr<int32_t>() : nullptr;
    constexpr int kThreads = 256;
    int const blocks = (numTokens * 32 + kThreads - 1) / kThreads;
    if (rows.scalar_type() == at::kBFloat16)
    {
        writeKernel<__nv_bfloat16>
            <<<blocks, kThreads, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(rows.data_ptr()), rowStride, loc64,
                loc32, numTokens, pages.data_ptr<uint8_t>(), pages.stride(0), rowsPageStride, rowsBySlot);
    }
    else
    {
        writeKernel<__nv_fp8_e4m3>
            <<<blocks, kThreads, 0, stream>>>(reinterpret_cast<__nv_fp8_e4m3 const*>(rows.data_ptr()), rowStride, loc64,
                loc32, numTokens, pages.data_ptr<uint8_t>(), pages.stride(0), rowsPageStride, rowsBySlot);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4IdxkGatherFp8(at::Tensor const& pages, at::Tensor const& pageIds, at::Tensor out)
{
    checkPages(pages);
    TORCH_CHECK(pageIds.is_cuda() && pageIds.scalar_type() == at::kInt && pageIds.dim() == 1 && pageIds.is_contiguous(),
        "page_ids must be a contiguous CUDA int32 vector");
    int const n = static_cast<int>(pageIds.size(0));
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat8_e4m3fn && out.is_contiguous()
            && out.numel() >= static_cast<int64_t>(n) * kPage * kDim,
        "out must be a contiguous CUDA float8_e4m3fn tensor of at least [N, 128, 128]");
    if (n == 0)
    {
        return;
    }
    at::cuda::CUDAGuard guard(pages.device());
    gatherKernel<<<n, 512, 0, at::cuda::getCurrentCUDAStream()>>>(pages.data_ptr<uint8_t>(), pages.stride(0),
        pageIds.data_ptr<int32_t>(), reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

TORCH_LIBRARY_FRAGMENT(m31idxk, m)
{
    m.def("nvfp4_idxk_write(Tensor rows, Tensor out_cache_loc, Tensor(a!) pages) -> ()");
    m.def("nvfp4_idxk_write_from_sink(Tensor sink, Tensor out_cache_loc, int num_tokens, Tensor(a!) pages) -> ()");
    m.def("nvfp4_idxk_gather_fp8(Tensor pages, Tensor page_ids, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(m31idxk, CUDA, m)
{
    m.impl("nvfp4_idxk_write", &nvfp4IdxkWrite);
    m.impl("nvfp4_idxk_write_from_sink", &nvfp4IdxkWriteFromSink);
    m.impl("nvfp4_idxk_gather_fp8", &nvfp4IdxkGatherFp8);
}
