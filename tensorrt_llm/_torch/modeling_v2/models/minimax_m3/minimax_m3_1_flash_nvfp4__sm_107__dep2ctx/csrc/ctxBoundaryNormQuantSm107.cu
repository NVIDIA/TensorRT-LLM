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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryNormQuantSm107.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <algorithm>
#include <cstdlib>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{
namespace
{

// The device code (kernel, ring and scale-tile staging); it calls quantization.cuh's helpers unqualified.
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryNormQuantSm107.cuh"
// DEP2CTX_TOPK_FOLD (K5 U2): the next-layer boundary with the MegaMoE top-k reduce + shared add folded in; it reuses
// the helpers above and leaves ctxBoundaryNormQuantSm107.cuh untouched.
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryTopkNormQuantSm107.cuh"
// CBNQ_P2 (K5 U3): the ring kernel at hidden / 16 threads x two chunks, two CTAs per SM on sm_107.
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryNormQuantP2Sm107.cuh"
// CBNQ_TOPK_DUAL: the top-k fold at two 384-thread row groups per 768-thread CTA on sm_107.
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryTopkNormQuantDualSm107.cuh"

constexpr int kMaxDevices = 64;

// L6 (PREFILL_PLAN_v2 row 6): an sm_107 SM holds at most 1024 resident threads (ptxas: "threads per SM out of range,
// .minnctapersm ignored" for (768, 2) on sm_107a), so two 768-thread ring CTAs never share an SM there, whatever the
// registers (36-40 in the sm_100f cubin) or the shared memory (75.7 KB of 227 KB): the occupancy API reports one CTA
// per SM. A deeper ring (4, 6 or 8 stages: 6 = the 2 x 3 rows the two-CTA design planned, 147 KB at hidden 6144) puts
// more rows in flight per SM, when the per-block shared-memory opt-in holds it, with the maximum shared-memory
// carveout. The depth only moves where a row lands in shared memory (stage = consumed % stages, parity =
// consumed / stages); the rows, their order per CTA, the grid and every arithmetic step are unchanged, so every output
// is the three-stage launch's, bit for bit. Where the thread limit admits two CTAs per SM (sm_100 / sm_103: 2048
// threads per SM), the depth also steps down until both CTAs' rings still fit the SM's shared memory (4 stages at
// hidden 6144), so a deeper ring never costs a resident CTA. The default depth is kCbnqL6DefaultStages = 3, the
// previous launch exactly (no carveout attribute: the carveout only serves a ring deeper than 3); the deeper rings
// measured no faster.
constexpr int kCbnqL6DefaultStages = 3;
// Static shared memory of the kernel (cuobjdump: 1152 bytes, the block-reduce scratch included) with headroom; the ring
// depth steps down until dynamic + this bound fits the per-block opt-in.
constexpr int kCbnqStaticSmemBound = 2048;
// Shared memory the driver reserves per resident CTA (cudaDevAttrReservedSharedMemoryPerBlock on sm_10x).
constexpr int kCbnqReservedSmemPerCta = 1024;

// CBNQ_P2 (K5 U3): where the hidden size splits into whole 32-chunk warps per half (hidden % 512 == 0, hidden / 16
// <= 384 threads), the thread limit holds the one-chunk kernel to one CTA per SM (sm_107: 1024 threads per SM) and the
// device holds two P2 CTAs per SM, the launch runs ctxBoundaryNormQuantP2Kernel
// (ctxBoundaryNormQuantP2Sm107.cuh): hidden / 16 threads x two 8-element chunks, so two rows' serial chains overlap on
// an sm_107 SM, which one 768-thread CTA cannot do; every output bit for bit the one-chunk kernel's. Its ring is
// kCbnqP2DefaultStages = 3 deep (2 x 76 KB per SM at hidden 6144; the K5 MB chose 3 over 4 stages);
// else the L6 / previous launch below. The maximum shared-memory carveout is always set for it.
// The DEP2CTX_TOPK_FOLD kernel is not a P2 flavor (its 72 KB stages leave one per CTA).
constexpr int kCbnqP2DefaultStages = 3;

struct CbnqLaunchChoice
{
    int stages;    // requested ring depth (3, 4, 6 or 8) of the one-chunk launch
    bool carveout; // set cudaFuncAttributePreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared
    bool p2;       // try the P2 launch first
    int p2Stages;  // its requested ring depth (3 or 4)
};

//! The launch the params ask for. The defaults are the serving launch: the three-stage one-chunk ring (no carveout)
//! behind the P2 launch at three stages. twoCta = false forces the previous launch (three stages, no carveout)
//! whatever depth is asked; the carveout only serves a ring deeper than the previous launch's.
CbnqLaunchChoice cbnqLaunchChoice(CtxBoundaryNormQuantParams const& p)
{
    int const p2Stages = (p.p2Stages == 3 || p.p2Stages == 4) ? p.p2Stages : kCbnqP2DefaultStages;
    if (!p.twoCta)
    {
        return CbnqLaunchChoice{kCbnqStages, false, p.p2, p2Stages};
    }
    int const stages
        = (p.stages == 3 || p.stages == 4 || p.stages == 6 || p.stages == 8) ? p.stages : kCbnqL6DefaultStages;
    return CbnqLaunchChoice{stages, stages > kCbnqStages, p.p2, p2Stages};
}

//! Whether the P2 kernel serves this hidden size: whole warps per half and a virtual warp count of at most 32.
bool cbnqP2Supported(int hidden)
{
    return hidden % 512 == 0 && hidden / (kCbnqP2Chunks * kCbnqPacked) <= kCbnqP2MaxThreads;
}

//! SM count, compute-capability major, per-block shared-memory opt-in and (optionally) resident threads and shared
//! memory per SM of the current device (cached per device).
cudaError_t deviceShape(
    int* device, int* sms, int* major, int* smemMax, int* threadsPerSm = nullptr, int* smemPerSm = nullptr)
{
    static int sSms[kMaxDevices] = {};
    static int sMajor[kMaxDevices] = {};
    static int sSmemMax[kMaxDevices] = {};
    static int sThreads[kMaxDevices] = {};
    static int sSmemSm[kMaxDevices] = {};
    cudaError_t e = cudaGetDevice(device);
    if (e != cudaSuccess)
    {
        return e;
    }
    if (*device < 0 || *device >= kMaxDevices)
    {
        return cudaErrorNotSupported;
    }
    if (sSms[*device] == 0)
    {
        int count = 0;
        int cc = 0;
        int smemMax = 0;
        int threads = 0;
        int smemSm = 0;
        if ((e = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, *device)) != cudaSuccess
            || (e = cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, *device)) != cudaSuccess
            || (e = cudaDeviceGetAttribute(&smemMax, cudaDevAttrMaxSharedMemoryPerBlockOptin, *device)) != cudaSuccess
            || (e = cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, *device)) != cudaSuccess
            || (e = cudaDeviceGetAttribute(&smemSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, *device))
                != cudaSuccess)
        {
            return e;
        }
        sMajor[*device] = cc;
        sSmemMax[*device] = smemMax;
        sThreads[*device] = threads;
        sSmemSm[*device] = smemSm;
        sSms[*device] = count;
    }
    *sms = sSms[*device];
    *major = sMajor[*device];
    *smemMax = sSmemMax[*device];
    if (threadsPerSm != nullptr)
    {
        *threadsPerSm = sThreads[*device];
    }
    if (smemPerSm != nullptr)
    {
        *smemPerSm = sSmemSm[*device];
    }
    return cudaSuccess;
}

//! The ring depth a one-chunk launch uses: `requested`, stepped down (8 -> 6 -> 4 -> 3) until the ring fits the
//! per-block opt-in and the CTAs the thread limit admits per SM (at most kCbnqMinBlocks) still fit the SM's shared
//! memory, so the depth never lowers the resident CTAs. Three stages (the previous launch) are always taken.
int cbnqLaunchStages(int requested, bool add, int hidden, int smemMax, int threadsPerSm, int smemPerSm)
{
    int const resident = std::min(kCbnqMinBlocks, threadsPerSm / (hidden / kCbnqPacked));
    for (int const s : {8, 6, 4})
    {
        int const perCta = cbnqSmemBytes(s, add, hidden) + kCbnqStaticSmemBound;
        if (s <= requested && perCta <= smemMax && resident * (perCta + kCbnqReservedSmemPerCta) <= smemPerSm)
        {
            return s;
        }
    }
    return kCbnqStages;
}

CbnqArgs makeArgs(CtxBoundaryNormQuantParams const& p)
{
    CbnqArgs a{};
    a.inter = p.inter;
    a.residual = p.residual;
    a.weight = p.weight;
    a.normOut = p.normOut;
    a.qOut = p.qOut;
    a.sfOut = p.sfOut;
    a.f32Out = p.f32Out;
    a.fp4Out = p.fp4Out;
    a.fp4SfOut = p.fp4SfOut;
    a.fp4GlobalScale = p.fp4GlobalScale;
    a.tailIds = p.tailIds;
    a.tailWeights = p.tailWeights;
    a.fp4StrideWords = p.fp4StrideWords;
    a.fp4SfStride = p.fp4SfStride;
    a.tailIdsStride = p.tailIdsStride;
    a.tailWeightsStride = p.tailWeightsStride;
    a.tailIdsRows = p.tailIdsRows;
    a.tailWeightsRows = p.tailWeightsRows;
    a.tailTopk = p.tailTopk;
    a.rows = p.rows;
    a.hidden = p.hidden;
    a.eps = p.eps;
    return a;
}

//! Launch shape of one flavor at ring depth kStages (CTAs per SM from the occupancy API with the ring's dynamic shared
//! memory, cached per device and hidden size), then the launch itself unless `launch` is false. `carveout` sets the
//! maximum shared-memory carveout once per (instantiation, device, hidden) with the shared-memory attribute.
template <int kStages, bool kAdd, bool kNorm, bool kMx, bool kF32, bool kFp4>
cudaError_t runFlavor(CtxBoundaryNormQuantParams const& p, bool carveout, bool enablePdl, cudaStream_t stream,
    bool launch, int* ctasPerSm, int* gridOut, int* smemOut)
{
    auto kernel = &ctxBoundaryNormQuantKernel<kStages, kAdd, kNorm, kMx, kF32, kFp4>;
    static int sHidden[kMaxDevices] = {};
    static int sCtas[kMaxDevices] = {};
    int device = 0;
    int sms = 0;
    int major = 0;
    int smemMax = 0;
    cudaError_t e = deviceShape(&device, &sms, &major, &smemMax);
    if (e != cudaSuccess)
    {
        return e;
    }
    if (major != 10)
    {
        return cudaErrorNotSupported;
    }
    int const block = p.hidden / kCbnqPacked;
    int const smem = cbnqSmemBytes(kStages, kAdd, p.hidden);
    int ctas = sHidden[device] == p.hidden ? sCtas[device] : 0;
    if (ctas == 0)
    {
        if ((e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess
            || (carveout
                && (e = cudaFuncSetAttribute(
                        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared))
                    != cudaSuccess)
            || (e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, block, smem)) != cudaSuccess)
        {
            return e;
        }
        if (ctas < 1)
        {
            return cudaErrorInvalidConfiguration;
        }
        sCtas[device] = ctas;
        sHidden[device] = p.hidden;
    }
    int const grid = std::max(1, std::min(cbnqNumQuads(p.rows), ctas * sms));
    if (ctasPerSm != nullptr)
    {
        *ctasPerSm = ctas;
    }
    if (gridOut != nullptr)
    {
        *gridOut = grid;
    }
    if (smemOut != nullptr)
    {
        *smemOut = smem;
    }
    if (!launch)
    {
        return cudaSuccess;
    }
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enablePdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, makeArgs(p));
}

template <int kStages>
cudaError_t dispatchFlavor(
    CtxBoundaryNormQuantParams const& p, bool c, bool pdl, cudaStream_t st, bool go, int* cps, int* g, int* sm)
{
    bool const norm = p.normOut != nullptr;
    bool const f32 = p.f32Out != nullptr;
    bool const fp4 = p.fp4Out != nullptr;
    if (p.residual == nullptr)
    {
        // Norm only, from rows that already hold inter + residual.
        if (!norm || p.qOut != nullptr || f32 || fp4)
        {
            return cudaErrorInvalidValue;
        }
        return runFlavor<kStages, false, true, false, false, false>(p, c, pdl, st, go, cps, g, sm);
    }
    if (p.qOut == nullptr || p.sfOut == nullptr
        || (fp4 && (p.fp4SfOut == nullptr || p.fp4GlobalScale == nullptr || p.fp4SfStride % 16 != 0)))
    {
        return cudaErrorInvalidValue;
    }
    switch ((norm ? 4 : 0) | (f32 ? 2 : 0) | (fp4 ? 1 : 0))
    {
    case 0: return runFlavor<kStages, true, false, true, false, false>(p, c, pdl, st, go, cps, g, sm);
    case 1: return runFlavor<kStages, true, false, true, false, true>(p, c, pdl, st, go, cps, g, sm);
    case 2: return runFlavor<kStages, true, false, true, true, false>(p, c, pdl, st, go, cps, g, sm);
    case 3: return runFlavor<kStages, true, false, true, true, true>(p, c, pdl, st, go, cps, g, sm);
    case 4: return runFlavor<kStages, true, true, true, false, false>(p, c, pdl, st, go, cps, g, sm);
    case 5: return runFlavor<kStages, true, true, true, false, true>(p, c, pdl, st, go, cps, g, sm);
    case 6: return runFlavor<kStages, true, true, true, true, false>(p, c, pdl, st, go, cps, g, sm);
    default: return runFlavor<kStages, true, true, true, true, true>(p, c, pdl, st, go, cps, g, sm);
    }
}

//! CBNQ_P2: launch shape of one P2 flavor at ring depth kStages, then the launch itself unless `launch` is false.
//! Sets `*declined` (and launches nothing) when the device does not hold kCbnqP2MinBlocks of these CTAs per SM (or
//! the ring does not fit the per-block opt-in); the caller then takes another form. The occupancy is cached per
//! (instantiation, device, hidden) as in runFlavor.
template <int kStages, bool kAdd, bool kNorm, bool kMx, bool kF32, bool kFp4>
cudaError_t runFlavorP2(CtxBoundaryNormQuantParams const& p, bool enablePdl, cudaStream_t stream, bool launch,
    int* ctasPerSm, int* gridOut, int* smemOut, bool* declined)
{
    auto kernel = &ctxBoundaryNormQuantP2Kernel<kStages, kAdd, kNorm, kMx, kF32, kFp4>;
    static int sHidden[kMaxDevices] = {};
    static int sCtas[kMaxDevices] = {};
    int device = 0;
    int sms = 0;
    int major = 0;
    int smemMax = 0;
    cudaError_t e = deviceShape(&device, &sms, &major, &smemMax);
    if (e != cudaSuccess)
    {
        return e;
    }
    if (major != 10)
    {
        return cudaErrorNotSupported;
    }
    int const block = p.hidden / (kCbnqP2Chunks * kCbnqPacked);
    int const smem = cbnqSmemBytes(kStages, kAdd, p.hidden);
    if (smem + kCbnqStaticSmemBound > smemMax)
    {
        *declined = true;
        return cudaSuccess;
    }
    int ctas = sHidden[device] == p.hidden ? sCtas[device] : 0;
    if (ctas == 0)
    {
        if ((e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess
            || (e = cudaFuncSetAttribute(
                    kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared))
                != cudaSuccess
            || (e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, block, smem)) != cudaSuccess)
        {
            return e;
        }
        ctas = std::max(ctas, 1); // cached: 1 (any value below kCbnqP2MinBlocks) marks a declined shape
        sCtas[device] = ctas;
        sHidden[device] = p.hidden;
    }
    if (ctas < kCbnqP2MinBlocks)
    {
        *declined = true;
        return cudaSuccess;
    }
    int const grid = std::max(1, std::min(cbnqNumQuads(p.rows), ctas * sms));
    if (ctasPerSm != nullptr)
    {
        *ctasPerSm = ctas;
    }
    if (gridOut != nullptr)
    {
        *gridOut = grid;
    }
    if (smemOut != nullptr)
    {
        *smemOut = smem;
    }
    if (!launch)
    {
        return cudaSuccess;
    }
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enablePdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, makeArgs(p));
}

//! dispatchFlavor's flavor table for the P2 kernel (the caller ran checkFlavor, dispatchFlavor's argument checks).
template <int kStages>
cudaError_t dispatchFlavorP2(
    CtxBoundaryNormQuantParams const& p, bool pdl, cudaStream_t st, bool go, int* cps, int* g, int* sm, bool* d)
{
    bool const norm = p.normOut != nullptr;
    bool const f32 = p.f32Out != nullptr;
    bool const fp4 = p.fp4Out != nullptr;
    if (p.residual == nullptr)
    {
        return runFlavorP2<kStages, false, true, false, false, false>(p, pdl, st, go, cps, g, sm, d);
    }
    switch ((norm ? 4 : 0) | (f32 ? 2 : 0) | (fp4 ? 1 : 0))
    {
    case 0: return runFlavorP2<kStages, true, false, true, false, false>(p, pdl, st, go, cps, g, sm, d);
    case 1: return runFlavorP2<kStages, true, false, true, false, true>(p, pdl, st, go, cps, g, sm, d);
    case 2: return runFlavorP2<kStages, true, false, true, true, false>(p, pdl, st, go, cps, g, sm, d);
    case 3: return runFlavorP2<kStages, true, false, true, true, true>(p, pdl, st, go, cps, g, sm, d);
    case 4: return runFlavorP2<kStages, true, true, true, false, false>(p, pdl, st, go, cps, g, sm, d);
    case 5: return runFlavorP2<kStages, true, true, true, false, true>(p, pdl, st, go, cps, g, sm, d);
    case 6: return runFlavorP2<kStages, true, true, true, true, false>(p, pdl, st, go, cps, g, sm, d);
    default: return runFlavorP2<kStages, true, true, true, true, true>(p, pdl, st, go, cps, g, sm, d);
    }
}

//! dispatchFlavor's argument checks: cudaErrorInvalidValue for a combination of outputs no flavor writes.
cudaError_t checkFlavor(CtxBoundaryNormQuantParams const& p)
{
    bool const norm = p.normOut != nullptr;
    bool const f32 = p.f32Out != nullptr;
    bool const fp4 = p.fp4Out != nullptr;
    if (p.residual == nullptr)
    {
        return !norm || p.qOut != nullptr || f32 || fp4 ? cudaErrorInvalidValue : cudaSuccess;
    }
    if (p.qOut == nullptr || p.sfOut == nullptr
        || (fp4 && (p.fp4SfOut == nullptr || p.fp4GlobalScale == nullptr || p.fp4SfStride % 16 != 0)))
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

//! One launch (or its shape) with an explicit choice; the public entry points pass cbnqLaunchChoice(). The P2 form
//! first (at its depth, then 3), else the one-chunk form at L6's depth. `chunks` / `stages` (optional) report the form.
cudaError_t dispatch(CtxBoundaryNormQuantParams const& p, CbnqLaunchChoice choice, bool enablePdl, cudaStream_t stream,
    bool launch, int* ctasPerSm, int* grid, int* smem, int* chunks = nullptr, int* stagesOut = nullptr)
{
    if (!ctxBoundaryNormQuantSupported(p.hidden) || p.rows < 0)
    {
        return cudaErrorNotSupported;
    }
    int device = 0;
    int sms = 0;
    int major = 0;
    int smemMax = 0;
    int threadsPerSm = 0;
    int smemPerSm = 0;
    cudaError_t const e = deviceShape(&device, &sms, &major, &smemMax, &threadsPerSm, &smemPerSm);
    if (e != cudaSuccess)
    {
        return e;
    }
    // P2 only where the thread limit holds the one-chunk kernel below its kCbnqMinBlocks CTAs per SM (sm_107 at hidden
    // 6144: 1024 / 768 = 1); elsewhere (sm_100 / sm_103, 2048 threads per SM) that kernel already runs two per SM.
    if (choice.p2 && cbnqP2Supported(p.hidden) && threadsPerSm / (p.hidden / kCbnqPacked) < kCbnqMinBlocks)
    {
        cudaError_t const ce = checkFlavor(p);
        if (ce != cudaSuccess)
        {
            return ce;
        }
        for (int const s : {choice.p2Stages, kCbnqStages})
        {
            bool declined = false;
            cudaError_t const pe = s == 4
                ? dispatchFlavorP2<4>(p, enablePdl, stream, launch, ctasPerSm, grid, smem, &declined)
                : dispatchFlavorP2<kCbnqStages>(p, enablePdl, stream, launch, ctasPerSm, grid, smem, &declined);
            if (!declined)
            {
                if (chunks != nullptr)
                {
                    *chunks = kCbnqP2Chunks;
                }
                if (stagesOut != nullptr)
                {
                    *stagesOut = s;
                }
                return pe;
            }
            if (s == kCbnqStages)
            {
                break; // a 3-stage request declined once: no second try at the same depth
            }
        }
    }
    int const stages
        = cbnqLaunchStages(choice.stages, p.residual != nullptr, p.hidden, smemMax, threadsPerSm, smemPerSm);
    if (chunks != nullptr)
    {
        *chunks = 1;
    }
    if (stagesOut != nullptr)
    {
        *stagesOut = stages;
    }
    switch (stages)
    {
    case 8: return dispatchFlavor<8>(p, choice.carveout, enablePdl, stream, launch, ctasPerSm, grid, smem);
    case 6: return dispatchFlavor<6>(p, choice.carveout, enablePdl, stream, launch, ctasPerSm, grid, smem);
    case 4: return dispatchFlavor<4>(p, choice.carveout, enablePdl, stream, launch, ctasPerSm, grid, smem);
    default: return dispatchFlavor<kCbnqStages>(p, choice.carveout, enablePdl, stream, launch, ctasPerSm, grid, smem);
    }
}

bool aligned16(void const* p)
{
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

// CBNQ_TOPK_DUAL: the fold's ring stage leaves one 768-thread CTA per SM, so only one row's serial chain runs
// per SM where the boundary kernels run two (CBNQ_P2). Where the thread limit also holds the one-chunk kernel to one
// CTA per SM (sm_107) and the hidden size splits into whole 32-chunk warps per half (cbnqP2Supported), the fold runs
// ctxBoundaryTopkNormQuantDualKernel (ctxBoundaryTopkNormQuantDualSm107.cuh): the same CTA, grid and ring, two
// hidden / 16-thread row groups with two chunks per thread, every output bit for bit the one-group fold's. The params'
// dual = false (the CBNQ_TOPK_DUAL kill switch) keeps the one-group fold.

//! Whether the fold takes the dual-group form on this device and hidden size (with `enabled` the params' dual).
bool cbnqTopkDualForm(bool enabled, int hidden, int threadsPerSm)
{
    return enabled && cbnqP2Supported(hidden) && threadsPerSm / (hidden / kCbnqPacked) < kCbnqMinBlocks;
}

//! DEP2CTX_TOPK_FOLD: launch shape of one fold flavor (as runFlavor: CTAs per SM from the occupancy API with the
//! ring's dynamic shared memory, cached per device and hidden size; the maximum shared-memory carveout, since one
//! 768-thread CTA holds a ~218 KB ring), then the launch itself unless `launch` is false. The ring is kStages deep
//! when it fits the per-block opt-in, else cudaErrorNotSupported.
template <int kStages, bool kNorm, bool kDual>
cudaError_t runTopkFlavor(CtxBoundaryTopkNormQuantParams const& p, bool enablePdl, cudaStream_t stream, bool launch,
    int* ctasPerSm, int* gridOut, int* smemOut)
{
    auto kernel = kDual ? &ctxBoundaryTopkNormQuantDualKernel<kStages, kCbnqTopk, kNorm>
                        : &ctxBoundaryTopkNormQuantKernel<kStages, kCbnqTopk, kNorm>;
    static int sHidden[kMaxDevices] = {};
    static int sCtas[kMaxDevices] = {};
    int device = 0;
    int sms = 0;
    int major = 0;
    int smemMax = 0;
    cudaError_t e = deviceShape(&device, &sms, &major, &smemMax);
    if (e != cudaSuccess)
    {
        return e;
    }
    int const smem = cbnqTopkSmemBytes(kStages, kCbnqTopk, p.hidden);
    if (major != 10 || smem + kCbnqStaticSmemBound > smemMax)
    {
        return cudaErrorNotSupported;
    }
    int const block = p.hidden / kCbnqPacked;
    int ctas = sHidden[device] == p.hidden ? sCtas[device] : 0;
    if (ctas == 0)
    {
        if ((e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess
            || (e = cudaFuncSetAttribute(
                    kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared))
                != cudaSuccess
            || (e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, block, smem)) != cudaSuccess)
        {
            return e;
        }
        if (ctas < 1)
        {
            return cudaErrorInvalidConfiguration;
        }
        sCtas[device] = ctas;
        sHidden[device] = p.hidden;
    }
    int const grid = std::max(1, std::min(cbnqNumQuads(p.rows), ctas * sms));
    if (ctasPerSm != nullptr)
    {
        *ctasPerSm = ctas;
    }
    if (gridOut != nullptr)
    {
        *gridOut = grid;
    }
    if (smemOut != nullptr)
    {
        *smemOut = smem;
    }
    if (!launch)
    {
        return cudaSuccess;
    }
    CbnqTopkArgs a{};
    a.shared = p.shared;
    a.plane = p.plane;
    a.scores = p.scores;
    a.residual = p.residual;
    a.weight = p.weight;
    a.normOut = p.normOut;
    a.qOut = p.qOut;
    a.sfOut = p.sfOut;
    a.rows = p.rows;
    a.hidden = p.hidden;
    a.eps = p.eps;
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enablePdl;
    config.attrs = attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, a);
}

cudaError_t dispatchTopk(CtxBoundaryTopkNormQuantParams const& p, bool dualEnabled, bool enablePdl, cudaStream_t stream,
    bool launch, int* ctasPerSm, int* grid, int* smem, int* dualOut = nullptr)
{
    if (!ctxBoundaryNormQuantSupported(p.hidden) || p.rows < 0 || p.topk != kCbnqTopk)
    {
        return cudaErrorNotSupported;
    }
    if (p.shared == nullptr || p.plane == nullptr || p.scores == nullptr || p.residual == nullptr || p.weight == nullptr
        || p.qOut == nullptr || p.sfOut == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    // The ring's bulk copies and the vector loads / stores need 16-byte aligned rows (rows are hidden x 2 bytes, a
    // multiple of 512; a row's scores are 16 bytes).
    if (!aligned16(p.shared) || !aligned16(p.plane) || !aligned16(p.scores) || !aligned16(p.residual)
        || !aligned16(p.weight) || !aligned16(p.qOut) || !aligned16(p.sfOut)
        || (p.normOut != nullptr && !aligned16(p.normOut)))
    {
        return cudaErrorNotSupported;
    }
    int device = 0;
    int sms = 0;
    int major = 0;
    int smemMax = 0;
    int threadsPerSm = 0;
    cudaError_t const e = deviceShape(&device, &sms, &major, &smemMax, &threadsPerSm);
    if (e != cudaSuccess)
    {
        return e;
    }
    bool const dual = cbnqTopkDualForm(dualEnabled, p.hidden, threadsPerSm);
    if (dualOut != nullptr)
    {
        *dualOut = dual ? 1 : 0;
    }
    if (dual)
    {
        return p.normOut != nullptr
            ? runTopkFlavor<kCbnqStages, true, true>(p, enablePdl, stream, launch, ctasPerSm, grid, smem)
            : runTopkFlavor<kCbnqStages, false, true>(p, enablePdl, stream, launch, ctasPerSm, grid, smem);
    }
    return p.normOut != nullptr
        ? runTopkFlavor<kCbnqStages, true, false>(p, enablePdl, stream, launch, ctasPerSm, grid, smem)
        : runTopkFlavor<kCbnqStages, false, false>(p, enablePdl, stream, launch, ctasPerSm, grid, smem);
}

} // namespace

bool ctxBoundaryNormQuantSupported(int hidden)
{
    return hidden % 256 == 0 && hidden >= 256 && hidden / kCbnqPacked <= kCbnqMaxThreads;
}

cudaError_t invokeCtxBoundaryNormQuant(CtxBoundaryNormQuantParams const& params, bool enablePdl, cudaStream_t stream)
{
    return dispatch(params, cbnqLaunchChoice(params), enablePdl, stream, /*launch=*/true, nullptr, nullptr, nullptr);
}

cudaError_t ctxBoundaryNormQuantLaunchShape(
    CtxBoundaryNormQuantParams const& params, int* ctasPerSm, int* grid, int* smemBytes)
{
    return dispatch(params, cbnqLaunchChoice(params), false, nullptr, /*launch=*/false, ctasPerSm, grid, smemBytes);
}

cudaError_t ctxBoundaryNormQuantLaunchForm(CtxBoundaryNormQuantParams const& params, int* chunksPerThread,
    int* ctasPerSm, int* grid, int* smemBytes, int* stages)
{
    return dispatch(params, cbnqLaunchChoice(params), false, nullptr, /*launch=*/false, ctasPerSm, grid, smemBytes,
        chunksPerThread, stages);
}

cudaError_t invokeCtxBoundaryTopkNormQuant(
    CtxBoundaryTopkNormQuantParams const& params, bool enablePdl, cudaStream_t stream)
{
    return dispatchTopk(params, params.dual, enablePdl, stream, /*launch=*/true, nullptr, nullptr, nullptr);
}

cudaError_t ctxBoundaryTopkNormQuantLaunchShape(
    CtxBoundaryTopkNormQuantParams const& params, int* ctasPerSm, int* grid, int* smemBytes)
{
    return dispatchTopk(params, params.dual, false, nullptr, /*launch=*/false, ctasPerSm, grid, smemBytes);
}

cudaError_t ctxBoundaryTopkNormQuantLaunchForm(
    CtxBoundaryTopkNormQuantParams const& params, int* dual, int* ctasPerSm, int* grid, int* smemBytes)
{
    return dispatchTopk(params, params.dual, false, nullptr, /*launch=*/false, ctasPerSm, grid, smemBytes, dual);
}

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END
