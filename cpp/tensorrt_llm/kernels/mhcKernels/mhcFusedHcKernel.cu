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
#include <list>
#include <unordered_map>

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
// HC_MULT * (2 + HC_MULT) = 4 * 6 = 24.
static constexpr uint32_t FHC_SHAPE_N = 24;
static constexpr uint32_t FHC_HC_MULT = 4;
static constexpr uint32_t FHC_HIDDEN_FLASH = 4096;
static constexpr uint32_t FHC_HIDDEN_PRO = 7168;
static constexpr uint32_t FHC_BLOCK_M = 64;
static constexpr uint32_t FHC_BLOCK_N = 32;
static constexpr uint32_t FHC_BLOCK_K = 64;
static constexpr uint32_t FHC_SWIZZLE_CD = 128;
// Rebalanced from N_B=12 / N_INPUT=2: SASS PC-sampling on M=4096 KS=2 showed
// ~25% of all stalls landed on a single NANOSLEEP.SYNCS (mbarrier wait) — the
// pmap warp was input-buffer-starved with only 2 TMA stages for 56 h_tiles.
// Trade 5 B-stages (-40 KiB) for +1 input stage (+40 KiB), keeping total SMEM
// at 226 KiB. N_B=7 still leaves >= HC_MULT slack for the MMA pipeline.
static constexpr uint32_t FHC_N_B_STAGES = 7;
static constexpr uint32_t FHC_N_INPUT_STG = 3;
static constexpr uint32_t FHC_NUM_MMA_TH = 128;
static constexpr uint32_t FHC_NUM_PMAP_TH = 128;

template <uint32_t Hidden>
static constexpr bool isSupportedFhcHidden()
{
    return Hidden == FHC_HIDDEN_FLASH || Hidden == FHC_HIDDEN_PRO;
}

static bool isSupportedFhcHiddenRuntime(int hidden_size)
{
    return hidden_size == static_cast<int>(FHC_HIDDEN_FLASH) || hidden_size == static_cast<int>(FHC_HIDDEN_PRO);
}

// Validate the tcgen05 MMA fused-HC compile-time shape contract. Hidden must
// be divisible into BLOCK_K tiles, kNumSplits must evenly divide those tiles,
// and the hidden dimension must be a multiple of BF16_VEC_LI (per-thread vector
// load granularity in the Phase 4 layer_input loop). The (Hidden % team-stride)
// alignment is no longer required: the layer_input loop has a scalar-vec tail
// that handles the residue after the vectorized main loop. Keep this in sync
// with the Python tactic filter (_fused_hc_mma_ks_supported in mhc_cuda.py).
template <uint32_t Hidden, uint32_t KS>
static constexpr bool isSupportedFhcMmaKS()
{
    static_assert(isSupportedFhcHidden<Hidden>(), "Unsupported fused-HC hidden size");
    static_assert(KS > 0, "kNumSplits must be positive");

    constexpr uint32_t hTilesPerHc = Hidden / FHC_BLOCK_K;
    constexpr uint32_t bf16VecLi = 8;

    return Hidden % FHC_BLOCK_K == 0 && hTilesPerHc % KS == 0 && Hidden % bf16VecLi == 0;
}

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

// ---- TMA descriptor cache --------------------------------------------------
//
// cuTensorMapEncodeTiled is a host-side call that takes ~1-2 µs per descriptor.
// Each fused_hc launch builds 4 descriptors (residual_in, x_in, W,
// residual_cur), so the per-call descriptor build is 4-8 µs — 25-50% of total
// wall time at small M (M ≤ 64).
//
// Cache scope: per-host-thread (`thread_local`). Same host thread launching to
// multiple CUDA streams shares one cache (descriptor content depends only on
// (device, ptr, shape), not on the stream the kernel runs on). Multi-thread
// callers each get their own cache.
//
// Multi-GPU: device_id is part of the key so a host thread that switches CUDA
// devices never reuses a descriptor across address spaces.
//
// CUDA-graph capture: cuTensorMapEncodeTiled is a pure host function that does
// not record any stream operation, so cache miss inside capture is safe. The
// descriptor is passed by value as __grid_constant__; the recorded graph node
// holds those bytes and replays correctly under workspace-stable replay
// (already enforced by _FusedHcWorkspaceCache in mhc_cuda.py).
//
// Eviction: LRU bounded to kTmaDescCacheCap entries per thread. Eager mode
// without CUDA-graph capture sees the PyTorch caching allocator hand out
// fresh `base` pointers when the workspace cache misses, so the unbounded
// version would grow on every shape transition. 128 entries × ~256 B = ~32 KB
// per host thread — fits in L1, sized to cover the working set of any single
// model (~4-8 distinct shapes × 4 descriptors each = O(20) live, with
// headroom for shape transitions).
namespace
{
constexpr size_t kTmaDescCacheCap = 128;

struct TmaDescKey
{
    void const* base; // GMEM base pointer (device pointer)
    uint64_t gmemInner;
    uint64_t gmemOuter;
    uint32_t smemInner;
    uint32_t smemOuter;
    uint64_t gmemOuterStrideBytes;
    uint32_t swizzleBytes;
    uint32_t elemBytes;
    CUtensorMapDataType dtype;
    int device_id; // protect against cross-device pointer aliasing

    bool operator==(TmaDescKey const& o) const noexcept
    {
        return base == o.base && gmemInner == o.gmemInner && gmemOuter == o.gmemOuter && smemInner == o.smemInner
            && smemOuter == o.smemOuter && gmemOuterStrideBytes == o.gmemOuterStrideBytes
            && swizzleBytes == o.swizzleBytes && elemBytes == o.elemBytes && dtype == o.dtype
            && device_id == o.device_id;
    }
};

struct TmaDescKeyHash
{
    size_t operator()(TmaDescKey const& k) const noexcept
    {
        size_t h = reinterpret_cast<uintptr_t>(k.base);
        h = h * 1099511628211ull + k.gmemInner;
        h = h * 1099511628211ull + k.gmemOuter;
        h = h * 1099511628211ull + (static_cast<uint64_t>(k.swizzleBytes) << 16 | k.elemBytes);
        h = h * 1099511628211ull + static_cast<uint64_t>(k.device_id);
        return h;
    }
};

// Standard hash+intrusive-LRU container. The list keeps the eviction order
// (front=MRU, back=LRU) and the map points at the list node for O(1) move-to-
// front on hit. On miss we evict from the back if at capacity, then push the
// new entry to the front.
struct TmaDescCache
{
    using ListIt = std::list<std::pair<TmaDescKey, CUtensorMap>>::iterator;
    std::list<std::pair<TmaDescKey, CUtensorMap>> order;
    std::unordered_map<TmaDescKey, ListIt, TmaDescKeyHash> index;
};

CUtensorMap getCachedTma2D(void* base, CUtensorMapDataType dtype, uint64_t gmemInner, uint64_t gmemOuter,
    uint32_t smemInner, uint32_t smemOuter, uint64_t gmemOuterStrideBytes, uint32_t swizzleBytes, uint32_t elemBytes)
{
    static thread_local TmaDescCache cache;
    int device_id = 0;
    cudaGetDevice(&device_id);
    TmaDescKey const key{base, gmemInner, gmemOuter, smemInner, smemOuter, gmemOuterStrideBytes, swizzleBytes,
        elemBytes, dtype, device_id};
    auto it = cache.index.find(key);
    if (it != cache.index.end())
    {
        // Move to front (MRU) and copy out by value before any future insert
        // invalidates the list node.
        cache.order.splice(cache.order.begin(), cache.order, it->second);
        return it->second->second;
    }
    CUtensorMap const tm = makeTma2D(
        base, dtype, gmemInner, gmemOuter, smemInner, smemOuter, gmemOuterStrideBytes, swizzleBytes, elemBytes);
    if (cache.order.size() >= kTmaDescCacheCap)
    {
        // Evict LRU. unordered_map.erase by key is O(1) avg.
        cache.index.erase(cache.order.back().first);
        cache.order.pop_back();
    }
    cache.order.emplace_front(key, tm);
    cache.index.emplace(key, cache.order.begin());
    return tm;
}

} // namespace

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

template <uint32_t Hidden, uint32_t KS>
static FusedRoutFn fhcInstance()
{
    static_assert(isSupportedFhcHidden<Hidden>(), "Unsupported fused-HC hidden size");
    static_assert(isSupportedFhcMmaKS<Hidden, KS>(), "Unsupported fused-HC MMA kNumSplits for hidden size");
    return &fused_mhc::fused_tf32_pmap_gemm_rout_atomic_impl<FHC_SHAPE_N, Hidden, FHC_HC_MULT, FHC_BLOCK_M, FHC_BLOCK_N,
        FHC_BLOCK_K, FHC_SWIZZLE_CD, FHC_N_B_STAGES, FHC_N_INPUT_STG, FHC_NUM_MMA_TH, FHC_NUM_PMAP_TH, KS,
        /*kEarlyRelease=*/false>;
}

template <uint32_t Hidden, uint32_t KS>
static FusedRoutFn fhcInstanceIfSupported()
{
    if constexpr (isSupportedFhcMmaKS<Hidden, KS>())
    {
        return fhcInstance<Hidden, KS>();
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "mhcFusedHcLaunch: unsupported kNumSplits=%u for hidden_size=%u", KS, Hidden);
        return nullptr;
    }
}

template <uint32_t Hidden>
static FusedRoutFn pickFhc(uint32_t ks)
{
    switch (ks)
    {
    case 1: return fhcInstanceIfSupported<Hidden, 1>();
    case 2: return fhcInstanceIfSupported<Hidden, 2>();
    case 4: return fhcInstanceIfSupported<Hidden, 4>();
    case 7: return fhcInstanceIfSupported<Hidden, 7>();
    case 8: return fhcInstanceIfSupported<Hidden, 8>();
    case 14: return fhcInstanceIfSupported<Hidden, 14>();
    case 16: return fhcInstanceIfSupported<Hidden, 16>();
    case 28: return fhcInstanceIfSupported<Hidden, 28>();
    case 32: return fhcInstanceIfSupported<Hidden, 32>();
    case 56: return fhcInstanceIfSupported<Hidden, 56>();
    case 64: return fhcInstanceIfSupported<Hidden, 64>();
    case 112: return fhcInstanceIfSupported<Hidden, 112>();
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

template <uint32_t Hidden>
static void mhcFusedHcLaunchImpl(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult,
    int num_k_splits, int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, __nv_bfloat16 const* norm_weight, float norm_eps,
    cudaStream_t stream)
{
    if (M <= 0)
        return;

    static_assert(isSupportedFhcHidden<Hidden>(), "Unsupported fused-HC hidden size");
    TLLM_CHECK_WITH_INFO(hidden_size == static_cast<int>(Hidden),
        "mhcFusedHcLaunch: dispatched Hidden=%u but got hidden_size=%d", Hidden, hidden_size);
    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);

    constexpr uint32_t SHAPE_K = FHC_HC_MULT * Hidden;

    uint32_t const m_u = static_cast<uint32_t>(M);
    uint32_t const ks = (num_k_splits > 0) ? static_cast<uint32_t>(num_k_splits) : pickKSplits(M);
    int const bs = (bigfuse_block_size > 0) ? bigfuse_block_size : selectBigFuseBS(M);

    // ---- Zero workspace buffers (atomic accumulators) ----
    // KS=1 takes the direct-store path in the kernel epilogue (no atomicAdd,
    // each (m,n) written by exactly one CTA), so pre-zeroing is unnecessary.
    if (ks > 1)
    {
        fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * FHC_SHAPE_N, r_acc_workspace,
            static_cast<uint32_t>(M), /*done_counter=*/nullptr, /*done_elems=*/0, stream);
    }

    // ---- Build TMA descriptors (cached by ptr+shape) ----
    CUtensorMap desc_res = getCachedTma2D(const_cast<__nv_bfloat16*>(residual_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SHAPE_K, m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_x = getCachedTma2D(const_cast<__nv_bfloat16*>(x_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, Hidden,
        m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(Hidden) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_b = getCachedTma2D(const_cast<float*>(w_t), CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, SHAPE_K, FHC_SHAPE_N,
        FHC_BLOCK_K, FHC_BLOCK_N, static_cast<uint64_t>(SHAPE_K) * sizeof(float),
        /*swizzleBytes=*/128, sizeof(float));

    CUtensorMap desc_res_out = getCachedTma2D(residual_cur, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, SHAPE_K, m_u, FHC_BLOCK_K,
        /*smemOuter=*/16, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    // ---- Step 1: fused post-mapping + TF32 GEMM + sqrsum + residual_out ----
    constexpr uint32_t fused_smem = fhcSmemSize();
    FusedRoutFn fa = pickFhc<Hidden>(ks);
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
        hc_post_mult_value, sinkhorn_repeat, /*num_splits=*/1, /*block_size=*/bs, norm_weight, norm_eps, stream);
}

void mhcFusedHcLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev, float const* post_mix_prev,
    float const* comb_mix_prev, float const* w_t, float const* hc_scale, float const* hc_base,
    __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur, __nv_bfloat16* layer_input_cur,
    float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult, int num_k_splits,
    int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, __nv_bfloat16 const* norm_weight, float norm_eps, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(isSupportedFhcHiddenRuntime(hidden_size),
        "mhcFusedHcLaunch: unsupported hidden_size=%d; supported hidden sizes are 4096 and 7168", hidden_size);

    switch (hidden_size)
    {
    case static_cast<int>(FHC_HIDDEN_FLASH):
        mhcFusedHcLaunchImpl<FHC_HIDDEN_FLASH>(x_prev, residual_prev, post_mix_prev, comb_mix_prev, w_t, hc_scale,
            hc_base, residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace, r_acc_workspace, M,
            hidden_size, hc_mult, num_k_splits, bigfuse_block_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, stream);
        return;
    case static_cast<int>(FHC_HIDDEN_PRO):
        mhcFusedHcLaunchImpl<FHC_HIDDEN_PRO>(x_prev, residual_prev, post_mix_prev, comb_mix_prev, w_t, hc_scale,
            hc_base, residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace, r_acc_workspace, M,
            hidden_size, hc_mult, num_k_splits, bigfuse_block_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, stream);
        return;
    default: return;
    }
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
    int sinkhorn_repeat, __nv_bfloat16 const* norm_weight, float norm_eps, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcFmaLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);
    TLLM_CHECK_WITH_INFO(hidden_size % static_cast<int>(FHC_BLOCK_K) == 0,
        "mhc fused-HC FMA path requires hidden_size to be divisible by %u, got %d", FHC_BLOCK_K, hidden_size);
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
        sinkhorn_repeat, /*num_splits=*/num_k_splits, bigfuse_block_size, norm_weight, norm_eps, stream);
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
    __nv_bfloat16*, float*, float*, int*, float const*, float const*, float const*, float const*, float*, float*,
    __nv_bfloat16 const*, float, float, float, float, float, uint32_t);

template <uint32_t Hidden, uint32_t KS, bool kFuseNorm>
static FusedAllInOneFn fhcAllInOneInstance()
{
    static_assert(isSupportedFhcHidden<Hidden>(), "Unsupported fused-HC hidden size");
    static_assert(isSupportedFhcMmaKS<Hidden, KS>(), "Unsupported fused-HC MMA kNumSplits for hidden size");
    return &fused_mhc::fused_allinone_tf32_pmap_gemm_atomic_impl<FHC_SHAPE_N, Hidden, FHC_HC_MULT, FHC_BLOCK_M,
        FHC_BLOCK_N, FHC_BLOCK_K, FHC_SWIZZLE_CD, FHC_N_B_STAGES, FHC_N_INPUT_STG, FHC_NUM_MMA_TH, FHC_NUM_PMAP_TH, KS,
        kFuseNorm>;
}

template <uint32_t Hidden, uint32_t KS, bool kFuseNorm>
static FusedAllInOneFn fhcAllInOneInstanceIfSupported()
{
    if constexpr (isSupportedFhcMmaKS<Hidden, KS>())
    {
        return fhcAllInOneInstance<Hidden, KS, kFuseNorm>();
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "mhcFusedHcAllInOneLaunch: unsupported kNumSplits=%u for hidden_size=%u", KS, Hidden);
        return nullptr;
    }
}

template <uint32_t Hidden, bool kFuseNorm>
static FusedAllInOneFn pickFhcAllInOne(uint32_t ks)
{
    switch (ks)
    {
    case 1: return fhcAllInOneInstanceIfSupported<Hidden, 1, kFuseNorm>();
    case 2: return fhcAllInOneInstanceIfSupported<Hidden, 2, kFuseNorm>();
    case 4: return fhcAllInOneInstanceIfSupported<Hidden, 4, kFuseNorm>();
    case 7: return fhcAllInOneInstanceIfSupported<Hidden, 7, kFuseNorm>();
    case 8: return fhcAllInOneInstanceIfSupported<Hidden, 8, kFuseNorm>();
    case 14: return fhcAllInOneInstanceIfSupported<Hidden, 14, kFuseNorm>();
    case 16: return fhcAllInOneInstanceIfSupported<Hidden, 16, kFuseNorm>();
    case 28: return fhcAllInOneInstanceIfSupported<Hidden, 28, kFuseNorm>();
    case 32: return fhcAllInOneInstanceIfSupported<Hidden, 32, kFuseNorm>();
    case 56: return fhcAllInOneInstanceIfSupported<Hidden, 56, kFuseNorm>();
    case 64: return fhcAllInOneInstanceIfSupported<Hidden, 64, kFuseNorm>();
    case 112: return fhcAllInOneInstanceIfSupported<Hidden, 112, kFuseNorm>();
    default: TLLM_CHECK_WITH_INFO(false, "mhcFusedHcAllInOneLaunch: unsupported kNumSplits=%u", ks); return nullptr;
    }
}

template <uint32_t Hidden>
static void mhcFusedHcAllInOneLaunchImpl(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int num_k_splits, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, __nv_bfloat16 const* norm_weight, float norm_eps,
    cudaStream_t stream)
{
    if (M <= 0)
        return;

    static_assert(isSupportedFhcHidden<Hidden>(), "Unsupported fused-HC hidden size");
    TLLM_CHECK_WITH_INFO(hidden_size == static_cast<int>(Hidden),
        "mhcFusedHcAllInOneLaunch: dispatched Hidden=%u but got hidden_size=%d", Hidden, hidden_size);
    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcAllInOneLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);

    constexpr uint32_t SHAPE_K = FHC_HC_MULT * Hidden;

    uint32_t const m_u = static_cast<uint32_t>(M);
    uint32_t const ks = (num_k_splits > 0) ? static_cast<uint32_t>(num_k_splits) : 1u;
    uint32_t const m_tiles = (m_u + FHC_BLOCK_M - 1) / FHC_BLOCK_M;

    // ---- Zero workspace buffers (atomic accumulators + done counter) ----
    // KS=1 takes the direct-store path (no atomic on y_acc/r_acc) and skips
    // the done_counter atomic election (Phase 3 just __threadfence_block +
    // __syncthreads). All three buffers are unused at KS=1 → skip the zero.
    if (ks > 1)
    {
        fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * FHC_SHAPE_N, r_acc_workspace,
            static_cast<uint32_t>(M), done_counter_workspace, m_tiles, stream);
    }

    // ---- Build TMA descriptors (cached by ptr+shape) ----
    CUtensorMap desc_res = getCachedTma2D(const_cast<__nv_bfloat16*>(residual_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SHAPE_K, m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_x = getCachedTma2D(const_cast<__nv_bfloat16*>(x_prev), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, Hidden,
        m_u, FHC_BLOCK_K, FHC_BLOCK_M, static_cast<uint64_t>(Hidden) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    CUtensorMap desc_b = getCachedTma2D(const_cast<float*>(w_t), CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, SHAPE_K, FHC_SHAPE_N,
        FHC_BLOCK_K, FHC_BLOCK_N, static_cast<uint64_t>(SHAPE_K) * sizeof(float),
        /*swizzleBytes=*/128, sizeof(float));

    CUtensorMap desc_res_out = getCachedTma2D(residual_cur, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, SHAPE_K, m_u, FHC_BLOCK_K,
        /*smemOuter=*/16, static_cast<uint64_t>(SHAPE_K) * sizeof(__nv_bfloat16),
        /*swizzleBytes=*/128, sizeof(__nv_bfloat16));

    // ---- Launch the single all-in-one kernel ----
    // Dispatch on `norm_weight != nullptr` to a kFuseNorm=true instance that
    // inlines the next-layer RMSNorm into Phase 4's layer_input write.
    bool const fuse_norm = (norm_weight != nullptr);
    constexpr uint32_t fused_smem = fhcAllInOneSmemSize();
    FusedAllInOneFn fa = fuse_norm ? pickFhcAllInOne<Hidden, /*kFuseNorm=*/true>(ks)
                                   : pickFhcAllInOne<Hidden, /*kFuseNorm=*/false>(ks);
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void const*>(fa), cudaFuncAttributeMaxDynamicSharedMemorySize, fused_smem));

    dim3 const grid(m_tiles * ks);
    dim3 const block(FHC_NUM_MMA_TH + FHC_NUM_PMAP_TH);
    fa<<<grid, block, fused_smem, stream>>>(m_u, desc_res, desc_x, desc_b, desc_res_out, residual_cur, layer_input_cur,
        y_acc_workspace, r_acc_workspace, done_counter_workspace, post_mix_prev, comb_mix_prev, hc_scale, hc_base,
        post_mix_cur, comb_mix_cur, norm_weight, norm_eps, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        static_cast<uint32_t>(sinkhorn_repeat));
}

void mhcFusedHcAllInOneLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int num_k_splits, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, __nv_bfloat16 const* norm_weight, float norm_eps,
    cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(isSupportedFhcHiddenRuntime(hidden_size),
        "mhcFusedHcAllInOneLaunch: unsupported hidden_size=%d; supported hidden sizes are 4096 and 7168", hidden_size);

    switch (hidden_size)
    {
    case static_cast<int>(FHC_HIDDEN_FLASH):
        mhcFusedHcAllInOneLaunchImpl<FHC_HIDDEN_FLASH>(x_prev, residual_prev, post_mix_prev, comb_mix_prev, w_t,
            hc_scale, hc_base, residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace,
            r_acc_workspace, done_counter_workspace, M, hidden_size, hc_mult, num_k_splits, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, stream);
        return;
    case static_cast<int>(FHC_HIDDEN_PRO):
        mhcFusedHcAllInOneLaunchImpl<FHC_HIDDEN_PRO>(x_prev, residual_prev, post_mix_prev, comb_mix_prev, w_t, hc_scale,
            hc_base, residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace, r_acc_workspace,
            done_counter_workspace, M, hidden_size, hc_mult, num_k_splits, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, stream);
        return;
    default: return;
    }
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
    float, float, float, float, int, __nv_bfloat16 const*, float);

template <int TN, int KS, int TM, bool kFuseNorm>
static FmaAllInOneFn fhcFmaAllInOneInstance()
{
    return &fused_fma_kernels::fused_pmap_gemm_fma_allinone<TN, KS, TM, /*FULL_N=*/24, /*BF16_VEC_OVERRIDE=*/0,
        kFuseNorm>;
}

template <bool kFuseNorm>
static FmaAllInOneFn pickFhcFmaAllInOne(int tile_n, int ks, int tile_m)
{
#define FHC_FMA_AIO_CASE(TN, KS, TM)                                                                                   \
    if (tile_n == (TN) && ks == (KS) && tile_m == (TM))                                                                \
    return fhcFmaAllInOneInstance<TN, KS, TM, kFuseNorm>()

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
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat, __nv_bfloat16 const* norm_weight,
    float norm_eps, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(hc_mult == static_cast<int>(FHC_HC_MULT),
        "mhcFusedHcFmaAllInOneLaunch: hc_mult=%d not supported (only %u)", hc_mult, FHC_HC_MULT);
    TLLM_CHECK_WITH_INFO(hidden_size % static_cast<int>(FHC_BLOCK_K) == 0,
        "mhc fused-HC FMA path requires hidden_size to be divisible by %u, got %d", FHC_BLOCK_K, hidden_size);
    TLLM_CHECK_WITH_INFO(FHC_SHAPE_N % tile_n == 0,
        "mhcFusedHcFmaAllInOneLaunch: SHAPE_N=%u not divisible by tile_n=%d", FHC_SHAPE_N, tile_n);

    int const K = hc_mult * hidden_size;
    int const N = static_cast<int>(FHC_SHAPE_N);
    int const m_batches = (M + tile_m - 1) / tile_m;

    // ---- Zero workspace buffers (atomic accumulators + done counter) ----
    // NOTE: Path F's kernel epilogue always uses atomicAdd into y_acc/r_acc
    // (no KS=1 direct-store fast path like the MMA Path D in
    // mhcFusedHcAllInOneLaunchImpl), so we cannot skip the zero even at
    // num_k_splits == 1. Adding the direct-store branch would require kernel
    // changes in fused_pmap_gemm_fma_allinone; deferred since Path F is the
    // small-M (M <= 32) winner where the workspace zero is ~1 µs anyway.
    fhcZeroWorkspaces(y_acc_workspace, static_cast<uint32_t>(M) * static_cast<uint32_t>(N), r_acc_workspace,
        static_cast<uint32_t>(M), done_counter_workspace, static_cast<uint32_t>(m_batches), stream);

    bool const fuse_norm = (norm_weight != nullptr);
    FmaAllInOneFn fn = fuse_norm ? pickFhcFmaAllInOne</*kFuseNorm=*/true>(tile_n, num_k_splits, tile_m)
                                 : pickFhcFmaAllInOne</*kFuseNorm=*/false>(tile_n, num_k_splits, tile_m);
    dim3 const grid(
        static_cast<unsigned>(m_batches), static_cast<unsigned>(N / tile_n), static_cast<unsigned>(num_k_splits));
    dim3 const block(256);
    fn<<<grid, block, 0, stream>>>(residual_prev, x_prev, post_mix_prev, comb_mix_prev, w_t, hc_scale, hc_base,
        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, y_acc_workspace, r_acc_workspace,
        done_counter_workspace, M, K, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, norm_weight, norm_eps);
}

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END
