/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <vector>

namespace tkc = tensorrt_llm::cutlass_extensions;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::FP4GemmType;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kMxfp8LargeMMin = 6553;
constexpr int64_t kMxfp8M8kBucket = 8192;
constexpr int64_t kMxfp8M16kMin = 13106;
constexpr int64_t kMxfp8M16kBucket = 16384;
constexpr int64_t kMxfp8M32kMin = 19659;
constexpr int64_t kMxfp8M32kBucket = 32768;
constexpr int64_t kMxfp8TacticCacheMiss = -2;

int getMxfp8SmVersion()
{
    // PyExecutor binds one GPU architecture per rank.
    static int const smVersion = tensorrt_llm::common::getSMVersion();
    return smVersion;
}

int64_t getMxfp8TuningBucket(int64_t const m)
{
    if (m < kMxfp8LargeMMin)
    {
        return m;
    }
    if (m <= kMxfp8M8kBucket)
    {
        return kMxfp8M8kBucket;
    }
    if (m >= kMxfp8M16kMin && m <= kMxfp8M16kBucket)
    {
        return kMxfp8M16kBucket;
    }
    if (m >= kMxfp8M32kMin && m <= kMxfp8M32kBucket)
    {
        return kMxfp8M32kBucket;
    }
    return m;
}

using Mxfp8TacticCacheKey = std::tuple<int, at::ScalarType, int64_t, int64_t, int64_t>;

struct Mxfp8TacticCacheEntry
{
    tkc::CutlassGemmConfig config;
    int64_t tactic;
};

using Mxfp8TacticCache = std::map<Mxfp8TacticCacheKey, Mxfp8TacticCacheEntry>;

Mxfp8TacticCache& getMxfp8TacticCache()
{
    static Mxfp8TacticCache cache;
    return cache;
}

std::shared_mutex& getMxfp8TacticCacheMutex()
{
    static std::shared_mutex mutex;
    return mutex;
}

Mxfp8TacticCacheKey makeMxfp8TacticCacheKey(
    int64_t const m, int64_t const n, int64_t const k, at::ScalarType const outputDtype)
{
    return {getMxfp8SmVersion(), outputDtype, getMxfp8TuningBucket(m), n, k};
}

std::optional<Mxfp8TacticCacheEntry> findMxfp8TacticCacheEntry(
    int64_t const m, int64_t const n, int64_t const k, at::ScalarType const outputDtype)
{
    std::shared_lock<std::shared_mutex> lock(getMxfp8TacticCacheMutex());
    auto const& cache = getMxfp8TacticCache();
    auto const iterator = cache.find(makeMxfp8TacticCacheKey(m, n, k, outputDtype));
    if (iterator == cache.end())
    {
        return std::nullopt;
    }
    return iterator->second;
}

void cacheMxfp8Tactic(int64_t const m, int64_t const n, int64_t const k, at::ScalarType const outputDtype,
    tkc::CutlassGemmConfig const& config, int64_t const tactic)
{
    std::unique_lock<std::shared_mutex> lock(getMxfp8TacticCacheMutex());
    getMxfp8TacticCache().insert_or_assign(
        makeMxfp8TacticCacheKey(m, n, k, outputDtype), Mxfp8TacticCacheEntry{config, tactic});
}

void clearMxfp8CachedTactics()
{
    std::unique_lock<std::shared_mutex> lock(getMxfp8TacticCacheMutex());
    getMxfp8TacticCache().clear();
}

tkc::CutlassGemmConfig getDefaultMxfp8GemmConfig()
{
    // Reuse the same default tile/cluster as MXFP8xMXFP4 -- the B operand is
    // 2x wider in MXFP8xMXFP8, but the same 4x4 cluster/256x256 tile shape is
    // a reasonable Blackwell fallback before startup tuning populates the
    // native tactic cache.
    return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x256B, tkc::MainloopScheduleType::AUTO,
        tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_4x4x1);
}

template <typename T>
void runMxfp8Gemm(at::Tensor& out, at::Tensor const& act, at::Tensor const& weight, at::Tensor const& actScale,
    at::Tensor const& weightScale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k,
    tkc::CutlassGemmConfig const& gemmConfig)
{
    CutlassFp4GemmRunner<T, FP4GemmType::W8A8_MXFP8_MXFP8> gemmRunner;
    int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, /*batch_count=*/1);

    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, act.device(), std::nullopt);

    gemmRunner.gemm(out.data_ptr(), act.const_data_ptr(), weight.const_data_ptr(), actScale.const_data_ptr(),
        weightScale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, /*batch_count=*/1, gemmConfig,
        reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(act.get_device()));
}

at::Tensor mxfp8Mxfp8GemmImpl(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, at::Tensor const& globalScale, std::optional<c10::ScalarType> outDtype,
    tkc::CutlassGemmConfig const* gemmConfig, bool const useTacticCache)
{
    CHECK_INPUT(act, torch::kFloat8_e4m3fn);
    CHECK_INPUT(weight, torch::kFloat8_e4m3fn);
    CHECK_INPUT(actScale, SF_DTYPE);
    CHECK_INPUT(weightScale, SF_DTYPE);
    CHECK_INPUT(globalScale, at::ScalarType::Float);

    TORCH_CHECK(act.dim() == 2, "act must be a 2D tensor [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor [N, K]");

    int64_t const m = act.sizes()[0];
    int64_t const k = act.sizes()[1];
    int64_t const n = weight.sizes()[0];
    TORCH_CHECK(
        weight.sizes()[1] == k, "act and weight K dims must match: act K=", k, ", weight K=", weight.sizes()[1]);

    // K must be divisible by the UE8M0 block size (32) for both A and B.
    constexpr int kBlock = 32;
    TORCH_CHECK(k % kBlock == 0, "K (", k, ") must be divisible by MXFP8 block size ", kBlock);
    // N must also be aligned to the kernel's tile-N alignment requirement.
    constexpr int kAlignmentN = 32;
    TORCH_CHECK(n % kAlignmentN == 0, "N (", n, ") must be divisible by ", kAlignmentN);

    auto const chosenDtype = outDtype.value_or(torch::kBFloat16);
    TORCH_CHECK(chosenDtype == torch::kFloat || chosenDtype == torch::kHalf || chosenDtype == torch::kBFloat16,
        "out_dtype must be one of fp16/bf16/fp32 (default bf16).");

    at::Tensor out = at::detail::empty_cuda({m, n}, chosenDtype, act.device(), std::nullopt);

    auto const cachedEntry = useTacticCache ? findMxfp8TacticCacheEntry(m, n, k, chosenDtype) : std::nullopt;
    auto const config = gemmConfig != nullptr
        ? *gemmConfig
        : (cachedEntry.has_value() ? cachedEntry->config : getDefaultMxfp8GemmConfig());
    switch (chosenDtype)
    {
    case at::ScalarType::Half:
        runMxfp8Gemm<half>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
        break;
    case at::ScalarType::BFloat16:
#ifdef ENABLE_BF16
        runMxfp8Gemm<__nv_bfloat16>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled to run MXFP8xMXFP8 GEMM with bf16 output.");
#endif
        break;
    case at::ScalarType::Float:
        runMxfp8Gemm<float>(out, act, weight, actScale, weightScale, globalScale, m, n, k, config);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

} // namespace

// MXFP8 (e4m3 + UE8M0 1x32 block scales) x MXFP8 (e4m3 + UE8M0 1x32 block
// scales) GEMM on Blackwell sm_100/103.
//
// Operands (matching the CUTLASS block-scaled tensor-op convention):
//   act:          [M, K] Float8_e4m3fn, row-major.
//   actScale:     1D uint8 (UE8M0), swizzled layout produced by
//                 torch.ops.trtllm.mxfp8_quantize(input, swizzedLayout=True).
//   weight:       [N, K] Float8_e4m3fn, expected to be column-major in memory.
//                 The caller is responsible for ensuring the weight tensor is
//                 contiguous in the column-major sense that CUTLASS expects.
//   weightScale:  1D uint8 (UE8M0), swizzled layout produced by
//                 torch.ops.trtllm.block_scale_interleave(scale).
//   globalScale:  [1] float -- alpha multiplier baked into the epilogue.
//                 For pure MXFP8xMXFP8 this is usually [1.0].
//   out_dtype:    fp16 / bf16 / fp32 output element type.
at::Tensor mxfp8_mxfp8_gemm(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, at::Tensor const& globalScale, std::optional<c10::ScalarType> outDtype)
{
    return mxfp8Mxfp8GemmImpl(act, actScale, weight, weightScale, globalScale, outDtype, /*gemmConfig=*/nullptr,
        /*useTacticCache=*/true);
}

//! Profiles native MXFP8 GEMM tactics and registers selected tactics in the
//! serving cache.
class MXFP8GemmRunner : public torch::CustomClassHolder
{
public:
    //! Constructs a runner for the requested output element type.
    //! \param outputDtype Output type; supported values are FP16, BF16, and FP32.
    explicit MXFP8GemmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
        TORCH_CHECK(outputDtype == torch::kFloat || outputDtype == torch::kHalf || outputDtype == torch::kBFloat16,
            "output_dtype must be one of fp16/bf16/fp32.");
        mConfigs = CutlassFp4GemmRunner<half, FP4GemmType::W8A8_MXFP8_MXFP8>{}.getConfigs();
    }

    //! Runs one MXFP8 GEMM with a selected compiled tactic.
    //! \param act Row-major MXFP8 activation tensor with shape [M, K].
    //! \param actScale Swizzled UE8M0 activation scales.
    //! \param weight MXFP8 weight tensor with logical shape [N, K].
    //! \param weightScale Swizzled UE8M0 weight scales.
    //! \param globalScale FP32 scalar tensor applied by the epilogue.
    //! \param configIdx Compiled tactic index, or -1 for the generic fallback.
    //! \return Output tensor with shape [M, N].
    at::Tensor runGemm(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
        at::Tensor const& weightScale, at::Tensor const& globalScale, int64_t configIdx) const
    {
        auto const config = configIdx == -1 ? getDefaultMxfp8GemmConfig() : getConfig(configIdx);
        return mxfp8Mxfp8GemmImpl(
            act, actScale, weight, weightScale, globalScale, mOutputDtype, &config, /*useTacticCache=*/false);
    }

    //! Registers a compiled tactic for a serving shape.
    void registerTactic(int64_t const m, int64_t const n, int64_t const k, int64_t const configIdx) const
    {
        tkc::CutlassGemmConfig const config = configIdx == -1 ? getDefaultMxfp8GemmConfig() : getConfig(configIdx);
        cacheMxfp8Tactic(m, n, k, mOutputDtype, config, configIdx);
    }

    //! Returns the registered tactic for a serving shape, or the cache-miss
    //! sentinel.
    int64_t getCachedTactic(int64_t const m, int64_t const n, int64_t const k) const
    {
        auto const entry = findMxfp8TacticCacheEntry(m, n, k, mOutputDtype);
        return entry.has_value() ? entry->tactic : kMxfp8TacticCacheMiss;
    }

    //! Removes every registered MXFP8 serving tactic.
    void clearTacticCache() const
    {
        clearMxfp8CachedTactics();
    }

    //! Returns the number of compiled native tactics available for profiling.
    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    tkc::CutlassGemmConfig const& getConfig(int64_t const configIdx) const
    {
        TORCH_CHECK(configIdx >= 0 && configIdx < getNumConfigs(), "MXFP8 config index ", configIdx,
            " is out of range [0, ", getNumConfigs(), ").");
        return mConfigs.at(configIdx);
    }

    at::ScalarType mOutputDtype;
    std::vector<tkc::CutlassGemmConfig> mConfigs;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::MXFP8GemmRunner>("MXFP8GemmRunner")
        .def(torch::init<at::ScalarType>())
        .def("run_gemm", &tensorrt_llm::torch_ext::MXFP8GemmRunner::runGemm)
        .def("get_num_configs", &tensorrt_llm::torch_ext::MXFP8GemmRunner::getNumConfigs)
        .def("register_tactic", &tensorrt_llm::torch_ext::MXFP8GemmRunner::registerTactic)
        .def("get_cached_tactic", &tensorrt_llm::torch_ext::MXFP8GemmRunner::getCachedTactic)
        .def("clear_tactic_cache", &tensorrt_llm::torch_ext::MXFP8GemmRunner::clearTacticCache);

    m.def(
        "mxfp8_mxfp8_gemm(Tensor act, Tensor actScale, Tensor weight, Tensor "
        "weightScale, "
        "Tensor globalScale, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mxfp8_mxfp8_gemm", &tensorrt_llm::torch_ext::mxfp8_mxfp8_gemm);
}
