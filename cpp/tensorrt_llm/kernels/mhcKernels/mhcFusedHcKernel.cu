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

// Single-launch fused hyper-connection boundary op.
//
// This implementation wraps the SM100 tcgen05-based residual-out variant
// (fused_tf32_pmap_gemm_rout_atomic_impl) to produce residual_cur, D, and
// sqr_sum in a single kernel launch; the big-fuse postlogue kernel then
// consumes (D, sqr_sum, residual_cur) to emit (post_mix_cur, comb_mix_cur,
// layer_input_cur). Semantically identical to
//
//   residual_cur = prev_mHC.post_mapping(x_prev, residual_prev, ...)
//   post_mix_cur, comb_mix_cur, layer_input_cur = self.pre_mapping(residual_cur)
//
// but exposed as a single entry point.

#include "fused_tf32_pmap_gemm.cuh"
#include "mhcKernels.h"
#include "mhc_fused_fma.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc
{

// ---- Single-launch workspace zero kernel -----------------------------------
//
// Replaces 2-3 separate cudaMemsetAsync calls for the atomic accumulator
// workspaces (y_acc, r_acc, optional done_counter). Avoids per-memset launch
// latency that is visible at small M / high-frequency inference.
namespace
{

__global__ void fhcZeroWorkspacesKernel(float* __restrict__ y_acc, uint32_t y_elems, float* __restrict__ r_acc,
    uint32_t r_elems, int* __restrict__ done_counter, uint32_t done_elems)
{
    uint32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const stride = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < y_elems; i += stride)
    {
        y_acc[i] = 0.0f;
    }
    for (uint32_t i = tid; i < r_elems; i += stride)
    {
        r_acc[i] = 0.0f;
    }
    if (done_counter != nullptr)
    {
        for (uint32_t i = tid; i < done_elems; i += stride)
        {
            done_counter[i] = 0;
        }
    }
}

inline void fhcZeroWorkspaces(float* y_acc, uint32_t y_elems, float* r_acc, uint32_t r_elems, int* done_counter,
    uint32_t done_elems, cudaStream_t stream)
{
    uint32_t const total = y_elems + r_elems + (done_counter != nullptr ? done_elems : 0u);
    if (total == 0u)
    {
        return;
    }
    constexpr uint32_t kBlock = 256;
    // Cap grid so we don't over-launch for small workspaces; 148 SMs on B200.
    uint32_t const num_blocks = min(static_cast<uint32_t>((total + kBlock - 1) / kBlock), 148u * 8u);
    fhcZeroWorkspacesKernel<<<num_blocks, kBlock, 0, stream>>>(
        y_acc, y_elems, r_acc, r_elems, done_counter, done_elems);
}

} // namespace

// ---- mHC fused kernel shape constants (mirrors the Python module) ----
static constexpr uint32_t FHC_SHAPE_N = 24;  // HC_MULT * (2 + HC_MULT) = 4 * 6 = 24
static constexpr uint32_t FHC_HIDDEN = 4096; // only this hidden size is currently wired up
static constexpr uint32_t FHC_HC_MULT = 4;
static constexpr uint32_t FHC_BLOCK_M = 64;
static constexpr uint32_t FHC_BLOCK_N = 32;
static constexpr uint32_t FHC_BLOCK_K = 64;
static constexpr uint32_t FHC_SWIZZLE_CD = 128;
static constexpr uint32_t FHC_N_B_STAGES = 12;
static constexpr uint32_t FHC_N_INPUT_STG = 2;
static constexpr uint32_t FHC_NUM_MMA_TH = 128;
static constexpr uint32_t FHC_NUM_PMAP_TH = 128;

static CUtensorMap makeTma2D(void* base, CUtensorMapDataType dtype, uint64_t gmemInner, uint64_t gmemOuter,
    uint32_t smemInner, uint32_t smemOuter, uint64_t gmemOuterStrideBytes, uint32_t swizzleBytes, uint32_t elemBytes)
{
    CUtensorMap tm;
    if (swizzleBytes != 0)
    {
        smemInner = swizzleBytes / elemBytes;
    }
    uint64_t gmemDims[2] = {gmemInner, gmemOuter};
    uint32_t smemDims[2] = {smemInner, smemOuter};
    uint64_t gmemStrides[1] = {gmemOuterStrideBytes};
    uint32_t elemStrides[2] = {1, 1};
    CUtensorMapSwizzle swizzle = (swizzleBytes == 128) ? CU_TENSOR_MAP_SWIZZLE_128B
        : (swizzleBytes == 64)                         ? CU_TENSOR_MAP_SWIZZLE_64B
        : (swizzleBytes == 32)                         ? CU_TENSOR_MAP_SWIZZLE_32B
                                                       : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult rc = cuTensorMapEncodeTiled(&tm, dtype, 2, base, gmemDims, gmemStrides, smemDims, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TLLM_CHECK_WITH_INFO(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed (%d)", static_cast<int>(rc));
    return tm;
}

static constexpr uint32_t fhcSmemSize()
{
    constexpr uint32_t SMEM_CD = FHC_BLOCK_M * FHC_SWIZZLE_CD;
    constexpr uint32_t SMEM_B = FHC_BLOCK_N * FHC_BLOCK_K * sizeof(float);
    constexpr uint32_t SMEM_RES_ISTG = FHC_BLOCK_M * FHC_HC_MULT * FHC_BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_X_ISTG = FHC_BLOCK_M * FHC_BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_POST = FHC_BLOCK_M * FHC_HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_COMB = FHC_BLOCK_M * FHC_HC_MULT * FHC_HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_RC = FHC_HC_MULT * FHC_BLOCK_M * FHC_BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t kNumCast = 4;
    constexpr uint32_t barriers = 2 * FHC_N_B_STAGES + 2 * FHC_N_INPUT_STG + 2 * kNumCast + 1;
    // 4 bytes for the tmem ptr word + 32 bytes padding for alignment headroom.
    return SMEM_CD + FHC_N_B_STAGES * SMEM_B + FHC_N_INPUT_STG * (SMEM_RES_ISTG + SMEM_X_ISTG) + SMEM_POST + SMEM_COMB
        + SMEM_RC + barriers * 8 + 4 + 32;
}

using FusedRoutFn = void (*)(
    uint32_t, CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, float*, float const*, float const*, float*);

template <uint32_t KS>
static FusedRoutFn fhcInstance()
{
    return &fused_mhc::fused_tf32_pmap_gemm_rout_atomic_impl<FHC_SHAPE_N, FHC_HIDDEN, FHC_HC_MULT, FHC_BLOCK_M,
        FHC_BLOCK_N, FHC_BLOCK_K, FHC_SWIZZLE_CD, FHC_N_B_STAGES, FHC_N_INPUT_STG, FHC_NUM_MMA_TH, FHC_NUM_PMAP_TH, KS,
        /*kEarlyRelease=*/false>;
}

static FusedRoutFn pickFhc(uint32_t ks)
{
    switch (ks)
    {
    case 1: return fhcInstance<1>();
    case 2: return fhcInstance<2>();
    case 4: return fhcInstance<4>();
    case 8: return fhcInstance<8>();
    case 16: return fhcInstance<16>();
    case 32: return fhcInstance<32>();
    case 64: return fhcInstance<64>();
    default: TLLM_CHECK_WITH_INFO(false, "mhcFusedHcLaunch: unsupported kNumSplits=%u", ks); return nullptr;
    }
}

// Heuristic split-K selection driven by M. Keeps large-M waves compute-bound
// (low splits) and low-M waves fully occupied (high splits).
//
// NOTE: split > 1 introduces atomic accumulation into D and sqr_sum. Atomic
// ordering is non-deterministic across runs, which breaks CUDA-graph bit-exact
// replay equality. Consumers that require bit-exact determinism should keep
// M small enough to land in the splits=1 bucket, or supply an explicit
// deterministic variant if needed.
static uint32_t pickKSplits(int M)
{
    if (M <= 512)
        return 16;
    if (M <= 1024)
        return 8;
    if (M <= 2048)
        return 4;
    if (M <= 4096)
        return 2;
    return 1;
}

static int selectBigFuseBS(int M)
{
    if (M >= 4096)
        return 128;
    if (M >= 256)
        return 256;
    return 512;
}

void mhcFusedHcLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev, float const* post_mix_prev,
    float const* comb_mix_prev, float const* w_t, float const* hc_scale, float const* hc_base,
    __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur, __nv_bfloat16* layer_input_cur,
    float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult, int num_k_splits,
    int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hidden_size == static_cast<int>(FHC_HIDDEN),
        "mhcFusedHcLaunch: hidden_size=%d not supported (only %u)", hidden_size, FHC_HIDDEN);
    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);

    constexpr uint32_t SHAPE_K = FHC_HC_MULT * FHC_HIDDEN;

    uint32_t const m_u = static_cast<uint32_t>(M);
    uint32_t const ks = (num_k_splits > 0) ? static_cast<uint32_t>(num_k_splits) : pickKSplits(M);
    int const bs = (bigfuse_block_size > 0) ? bigfuse_block_size : selectBigFuseBS(M);

    // ---- Zero workspace buffers (atomic accumulators) ----
    fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * FHC_SHAPE_N, r_acc_workspace,
        static_cast<uint32_t>(M), /*done_counter=*/nullptr, /*done_elems=*/0, stream);

    // ---- Build TMA descriptors ----
    CUtensorMap desc_res = makeTma2D(const_cast<__nv_bfloat16*>(residual_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SHAPE_K, m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_x = makeTma2D(const_cast<__nv_bfloat16*>(x_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, FHC_HIDDEN,
        m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(FHC_HIDDEN) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_b = makeTma2D(const_cast<float*>(w_t), CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, SHAPE_K, FHC_SHAPE_N,
        FHC_BLOCK_K, FHC_BLOCK_N, static_cast<uint64_t>(SHAPE_K) * sizeof(float),
        /*swizzleBytes=*/128, sizeof(float));

    CUtensorMap desc_res_out = makeTma2D(residual_cur, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, SHAPE_K, m_u, FHC_BLOCK_K,
        /*smemOuter=*/16, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    // ---- Step 1: fused post-mapping + TF32 GEMM + sqrsum + residual_out ----
    constexpr uint32_t fused_smem = fhcSmemSize();
    FusedRoutFn fa = pickFhc(ks);
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void const*>(fa), cudaFuncAttributeMaxDynamicSharedMemorySize, fused_smem));

    uint32_t const m_tiles = (m_u + FHC_BLOCK_M - 1) / FHC_BLOCK_M;
    dim3 const grid(m_tiles * ks);
    dim3 const block(FHC_NUM_MMA_TH + FHC_NUM_PMAP_TH);
    fa<<<grid, block, fused_smem, stream>>>(
        m_u, desc_res, desc_x, desc_b, desc_res_out, y_acc_workspace, post_mix_prev, comb_mix_prev, r_acc_workspace);

    // ---- Step 2: big-fuse postlogue (RMS + sigmoid + Sinkhorn + pre-apply) ----
    // Delegate to mhcBigFuseLaunch (defined in mhcKernels.cu) to avoid
    // instantiating the mhcBigFuseKernel template in this TU.
    mhcBigFuseLaunch(y_acc_workspace, r_acc_workspace, residual_cur, hc_scale, hc_base, post_mix_cur, comb_mix_cur,
        layer_input_cur, M, /*K=*/static_cast<int>(SHAPE_K), hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, /*num_splits=*/1, /*block_size=*/bs, stream);
}

// ===================================================================
// FMA-path fused hyper-connection boundary launcher.
//
// Uses `fused_fma_kernels::fused_pmap_gemm_fma_ksplit<N_PER_BLOCK, NUM_K_SPLITS, 0, true>`
// which computes pmap inline in registers, emits residual_cur to HBM, and
// writes split GEMM partials to Yp[ks, M, N] / Rp[ks, M].  The same bigfuse
// kernel used by mhc_pre_mapping consumes those split buffers via num_splits.
// ===================================================================

using FmaKsplitFn = void (*)(__nv_bfloat16 const*, __nv_bfloat16 const*, float const*, float const*, float const*,
    float*, float*, int, int, int, __nv_bfloat16*);

template <int TN, int KS>
static FmaKsplitFn fhcFmaInstance()
{
    return &fused_fma_kernels::fused_pmap_gemm_fma_ksplit<TN, KS, /*BF16_VEC_OVERRIDE=*/0, /*WRITE_RESIDUAL=*/true>;
}

// Valid (tile_n, num_k_splits) combinations the fused_hc FMA path supports.
// Keep this limited to the small/mid-M sweet spots from profile_fair_report v4.
static FmaKsplitFn pickFhcFma(int tile_n, int ks)
{
#define FHCFMA_CASE(TN, KS)                                                                                            \
    if (tile_n == (TN) && ks == (KS))                                                                                  \
    return fhcFmaInstance<TN, KS>()

    FHCFMA_CASE(1, 1);
    FHCFMA_CASE(1, 2);
    FHCFMA_CASE(1, 4);
    FHCFMA_CASE(1, 8);
    FHCFMA_CASE(2, 1);
    FHCFMA_CASE(2, 2);
    FHCFMA_CASE(2, 4);
    FHCFMA_CASE(2, 8);
    FHCFMA_CASE(3, 1);
    FHCFMA_CASE(3, 2);
    FHCFMA_CASE(3, 4);
    FHCFMA_CASE(4, 1);
    FHCFMA_CASE(4, 2);
    FHCFMA_CASE(6, 1);
    FHCFMA_CASE(8, 1);
    FHCFMA_CASE(12, 1);
    FHCFMA_CASE(24, 1);
#undef FHCFMA_CASE
    TLLM_CHECK_WITH_INFO(false, "mhcFusedHcFmaLaunch: unsupported (tile_n=%d, ks=%d)", tile_n, ks);
    return nullptr;
}

void mhcFusedHcFmaLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev, float const* post_mix_prev,
    float const* comb_mix_prev, float const* w_t, float const* hc_scale, float const* hc_base,
    __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur, __nv_bfloat16* layer_input_cur,
    float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult, int tile_n, int num_k_splits,
    int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcFmaLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);
    TLLM_CHECK_WITH_INFO(
        FHC_SHAPE_N % tile_n == 0, "mhcFusedHcFmaLaunch: SHAPE_N=%u not divisible by tile_n=%d", FHC_SHAPE_N, tile_n);

    int const K = hc_mult * hidden_size;
    int const N = static_cast<int>(FHC_SHAPE_N);

    // ---- Step 1: fused pmap + GEMM + sqrsum + residual_cur (FMA ksplit) ----
    FmaKsplitFn fn = pickFhcFma(tile_n, num_k_splits);
    dim3 const grid(static_cast<unsigned>(M), static_cast<unsigned>(N / tile_n), static_cast<unsigned>(num_k_splits));
    dim3 const block(256);
    fn<<<grid, block, 0, stream>>>(residual_prev, x_prev, post_mix_prev, comb_mix_prev, w_t, y_acc_workspace,
        r_acc_workspace, hidden_size, N, K, residual_cur);

    // ---- Step 2: big-fuse postlogue (reduces ks splits internally) ----
    mhcBigFuseLaunch(y_acc_workspace, r_acc_workspace, residual_cur, hc_scale, hc_base, post_mix_cur, comb_mix_cur,
        layer_input_cur, M, /*K=*/K, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, /*num_splits=*/num_k_splits, bigfuse_block_size, stream);
}

// ===================================================================
// All-in-one single-kernel fused hyper-connection launcher (TF32 tcgen05).
//
// Wraps fused_allinone_tf32_pmap_gemm_atomic_impl: pmap + GEMM + bigfuse
// all fused into one kernel.  Phase 3 elects last-home CTA per m-block via
// atomic done_counter; Phase 4 runs bigfuse inline on the elected CTA.
// ===================================================================

// Path D (all-in-one) reuses Path B's SMEM layout: CD + B stages + res/x
// stages + post + comb + rc (HC_MULT slices) + barriers. The inline bigfuse
// tail uses static __shared__ scratch (s_pre_mix, s_is_last) which is NOT
// counted in the dynamic SMEM budget, so this size matches fhcSmemSize().
static constexpr uint32_t fhcAllInOneSmemSize()
{
    return fhcSmemSize();
}

using FusedAllInOneFn = void (*)(uint32_t, CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, __nv_bfloat16 const*,
    __nv_bfloat16*, float*, float*, int*, float const*, float const*, float const*, float const*, float*, float*, float,
    float, float, float, uint32_t);

template <uint32_t KS>
static FusedAllInOneFn fhcAllInOneInstance()
{
    return &fused_mhc::fused_allinone_tf32_pmap_gemm_atomic_impl<FHC_SHAPE_N, FHC_HIDDEN, FHC_HC_MULT, FHC_BLOCK_M,
        FHC_BLOCK_N, FHC_BLOCK_K, FHC_SWIZZLE_CD, FHC_N_B_STAGES, FHC_N_INPUT_STG, FHC_NUM_MMA_TH, FHC_NUM_PMAP_TH, KS>;
}

static FusedAllInOneFn pickFhcAllInOne(uint32_t ks)
{
    switch (ks)
    {
    case 1: return fhcAllInOneInstance<1>();
    case 2: return fhcAllInOneInstance<2>();
    case 4: return fhcAllInOneInstance<4>();
    case 8: return fhcAllInOneInstance<8>();
    case 16: return fhcAllInOneInstance<16>();
    case 32: return fhcAllInOneInstance<32>();
    case 64: return fhcAllInOneInstance<64>();
    default: TLLM_CHECK_WITH_INFO(false, "mhcFusedHcAllInOneLaunch: unsupported kNumSplits=%u", ks); return nullptr;
    }
}

void mhcFusedHcAllInOneLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int num_k_splits, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hidden_size == static_cast<int>(FHC_HIDDEN),
        "mhcFusedHcAllInOneLaunch: hidden_size=%d not supported (only %u)", hidden_size, FHC_HIDDEN);
    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcAllInOneLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);

    constexpr uint32_t SHAPE_K = FHC_HC_MULT * FHC_HIDDEN;

    uint32_t const m_u = static_cast<uint32_t>(M);
    uint32_t const ks = (num_k_splits > 0) ? static_cast<uint32_t>(num_k_splits) : 1u;
    uint32_t const m_tiles = (m_u + FHC_BLOCK_M - 1) / FHC_BLOCK_M;

    // ---- Zero workspace buffers (atomic accumulators + done counter) ----
    fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * FHC_SHAPE_N, r_acc_workspace,
        static_cast<uint32_t>(M), done_counter_workspace, m_tiles, stream);

    // ---- Build TMA descriptors ----
    CUtensorMap desc_res = makeTma2D(const_cast<__nv_bfloat16*>(residual_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SHAPE_K, m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_x = makeTma2D(const_cast<__nv_bfloat16*>(x_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, FHC_HIDDEN,
        m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(FHC_HIDDEN) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_b = makeTma2D(const_cast<float*>(w_t), CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, SHAPE_K, FHC_SHAPE_N,
        FHC_BLOCK_K, FHC_BLOCK_N, static_cast<uint64_t>(SHAPE_K) * sizeof(float),
        /*swizzleBytes=*/128, sizeof(float));

    CUtensorMap desc_res_out = makeTma2D(residual_cur, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, SHAPE_K, m_u, FHC_BLOCK_K,
        /*smemOuter=*/16, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    // ---- Launch the single all-in-one kernel ----
    constexpr uint32_t fused_smem = fhcAllInOneSmemSize();
    FusedAllInOneFn fa = pickFhcAllInOne(ks);
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void const*>(fa), cudaFuncAttributeMaxDynamicSharedMemorySize, fused_smem));

    dim3 const grid(m_tiles * ks);
    dim3 const block(FHC_NUM_MMA_TH + FHC_NUM_PMAP_TH);
    fa<<<grid, block, fused_smem, stream>>>(m_u, desc_res, desc_x, desc_b, desc_res_out, residual_cur, layer_input_cur,
        y_acc_workspace, r_acc_workspace, done_counter_workspace, post_mix_prev, comb_mix_prev, hc_scale, hc_base,
        post_mix_cur, comb_mix_cur, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        static_cast<uint32_t>(sinkhorn_repeat));
}

// ===================================================================
// All-in-one single-kernel fused hyper-connection launcher (FMA).
//
// Wraps fused_pmap_gemm_fma_allinone<TN, KS, TM>: pmap + FMA GEMM + bigfuse
// all fused into one kernel.  Same last-home CTA election pattern as the
// TF32 all-in-one path.  Preferred for small-M (M <= 32).
// ===================================================================

using FmaAllInOneFn = void (*)(__nv_bfloat16 const*, __nv_bfloat16 const*, float const*, float const*, float const*,
    float const*, float const*, __nv_bfloat16*, float*, float*, __nv_bfloat16*, float*, float*, int*, int, int, int,
    float, float, float, float, int);

template <int TN, int KS, int TM>
static FmaAllInOneFn fhcFmaAllInOneInstance()
{
    return &fused_fma_kernels::fused_pmap_gemm_fma_allinone<TN, KS, TM, /*FULL_N=*/24, /*BF16_VEC_OVERRIDE=*/0>;
}

static FmaAllInOneFn pickFhcFmaAllInOne(int tile_n, int ks, int tile_m)
{
#define FHC_FMA_AIO_CASE(TN, KS, TM)                                                                                   \
    if (tile_n == (TN) && ks == (KS) && tile_m == (TM))                                                                \
    return fhcFmaAllInOneInstance<TN, KS, TM>()

    FHC_FMA_AIO_CASE(1, 1, 1);
    FHC_FMA_AIO_CASE(1, 2, 1);
    FHC_FMA_AIO_CASE(2, 1, 1);
    FHC_FMA_AIO_CASE(2, 2, 1);
    FHC_FMA_AIO_CASE(3, 1, 1);
    FHC_FMA_AIO_CASE(4, 1, 1);
    FHC_FMA_AIO_CASE(6, 1, 1);
    FHC_FMA_AIO_CASE(8, 1, 1);
    FHC_FMA_AIO_CASE(12, 1, 1);
    FHC_FMA_AIO_CASE(24, 1, 1);
    FHC_FMA_AIO_CASE(1, 1, 2);
    FHC_FMA_AIO_CASE(1, 2, 2);
    FHC_FMA_AIO_CASE(2, 1, 2);
    FHC_FMA_AIO_CASE(2, 2, 2);
    FHC_FMA_AIO_CASE(3, 1, 2);
    FHC_FMA_AIO_CASE(4, 1, 2);
    FHC_FMA_AIO_CASE(6, 1, 2);
    FHC_FMA_AIO_CASE(8, 1, 2);
    FHC_FMA_AIO_CASE(12, 1, 2);
    FHC_FMA_AIO_CASE(24, 1, 2);
    FHC_FMA_AIO_CASE(1, 1, 4);
    FHC_FMA_AIO_CASE(1, 2, 4);
    FHC_FMA_AIO_CASE(2, 1, 4);
    FHC_FMA_AIO_CASE(2, 2, 4);
    FHC_FMA_AIO_CASE(3, 1, 4);
    FHC_FMA_AIO_CASE(4, 1, 4);
    FHC_FMA_AIO_CASE(6, 1, 4);
    FHC_FMA_AIO_CASE(8, 1, 4);
    FHC_FMA_AIO_CASE(12, 1, 4);
    FHC_FMA_AIO_CASE(24, 1, 4);
#undef FHC_FMA_AIO_CASE
    TLLM_CHECK_WITH_INFO(
        false, "mhcFusedHcFmaAllInOneLaunch: unsupported (tile_n=%d, ks=%d, tile_m=%d)", tile_n, ks, tile_m);
    return nullptr;
}

void mhcFusedHcFmaAllInOneLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int tile_n, int num_k_splits, int tile_m, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcFmaAllInOneLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);
    TLLM_CHECK_WITH_INFO(FHC_SHAPE_N % tile_n == 0,
        "mhcFusedHcFmaAllInOneLaunch: SHAPE_N=%u not divisible by tile_n=%d", FHC_SHAPE_N, tile_n);

    int const K = hc_mult * hidden_size;
    int const N = static_cast<int>(FHC_SHAPE_N);
    int const m_batches = (M + tile_m - 1) / tile_m;

    // ---- Zero workspace buffers (atomic accumulators + done counter) ----
    fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * static_cast<uint32_t>(N), r_acc_workspace,
        static_cast<uint32_t>(M), done_counter_workspace, static_cast<uint32_t>(m_batches), stream);

    FmaAllInOneFn fn = pickFhcFmaAllInOne(tile_n, num_k_splits, tile_m);
    dim3 const grid(
        static_cast<unsigned>(m_batches), static_cast<unsigned>(N / tile_n), static_cast<unsigned>(num_k_splits));
    dim3 const block(256);
    fn<<<grid, block, 0, stream>>>(residual_prev, x_prev, post_mix_prev, comb_mix_prev, w_t, hc_scale, hc_base,
        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace, r_acc_workspace,
        done_counter_workspace, M, K, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat);
}

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END
