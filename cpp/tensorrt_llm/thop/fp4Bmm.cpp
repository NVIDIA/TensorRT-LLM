/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp4_gemm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <c10/util/Exception.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <type_traits>

namespace tkc = tensorrt_llm::cutlass_extensions;
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::internal_cutlass_kernels::CutlassFp4GemmRunnerInterface;

namespace torch_ext
{

namespace
{

tkc::CutlassGemmConfig getDefaultGemmConfig(int64_t m, int64_t n, int64_t k)
{
    return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B, tkc::MainloopScheduleType::AUTO,
        tkc::EpilogueScheduleType::AUTO, tkc::ClusterShape::ClusterShape_1x1x1);
}

template <typename T>
void runBmm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k, int64_t batch_count,
    tkc::CutlassGemmConfig const& gemmConfig)
{
    CutlassFp4GemmRunner<T> gemmRunner;
    int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, batch_count);

    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, mat1.device(), std::nullopt);

    gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
        mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, batch_count, gemmConfig,
        reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(mat1.get_device()));
}

// mat1: [B, M, K / 2], FLOAT4_E2M1X2
// mat2: [B, N, K / 2], FLOAT4_E2M1X2
// out:  [B, M, N], fp16/bf16/fp32
// mat1Scale: B * ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// mat2Scale: B * ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
// Only NVFP4 is currently supported
at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype, tkc::CutlassGemmConfig const* maybe_config = nullptr)
{
    CHECK_INPUT(mat1, FLOAT4_E2M1X2);
    CHECK_INPUT(mat2, FLOAT4_E2M1X2);

    CHECK_INPUT(mat1Scale, SF_DTYPE);
    CHECK_INPUT(mat2Scale, SF_DTYPE);

    CHECK_INPUT(globalScale, at::ScalarType::Float);

    TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");

    TORCH_CHECK(mat1.dim() == 3, "mat1 must be a batch of matrices");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batch of matrices");
    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 must have the same number of batches");
    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[1], "x",
        mat1.sizes()[2], " and ", mat2.sizes()[1], "x", mat2.sizes()[2], ")");

    auto const m = mat1.sizes()[1];
    auto const n = mat2.sizes()[1];
    auto const k = mat1.sizes()[2] * 2;

    auto const batch_count = mat1.sizes()[0];

    auto config = maybe_config ? *maybe_config : getDefaultGemmConfig(m, n, k);

    constexpr int alignment = 32;
    TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got mat1 shape: (",
        mat1.sizes()[0], "x", mat1.sizes()[1], "), k: ", k, ".");
    TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got mat2 shape: (",
        mat2.sizes()[0], "x", mat2.sizes()[1], ").");

    if (!out_dtype)
    {
        out_dtype = torch::kHalf;
    }
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
        "out_dtype must be one of fp16/bf16/fp32. It defaults to fp16.");

    at::Tensor out = at::detail::empty_cuda({batch_count, m, n}, out_dtype.value(), mat1.device(), std::nullopt);

    switch (out_dtype.value())
    {
    case at::ScalarType::Half:
        runBmm<half>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, batch_count, config);
        break;
    case at::ScalarType::BFloat16:
        runBmm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, batch_count, config);
        break;
    case at::ScalarType::Float:
        runBmm<float>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, batch_count, config);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

struct Fp4BmmId
{
    int64_t n;
    int64_t k;
    int64_t b;
    c10::ScalarType out_dtype;

    Fp4BmmId(int64_t n_, int64_t k_, int64_t b_, c10::ScalarType out_dtype_)
        : n(n_)
        , k(k_)
        , b(b_)
        , out_dtype(out_dtype_)
    {
    }

    Fp4BmmId()
        : n(-1)
        , k(-1)
        , b(-1)
        , out_dtype(c10::ScalarType::Half)
    {
    }

    bool operator==(Fp4BmmId const& other) const
    {
        return n == other.n && k == other.k && b == other.b && out_dtype == other.out_dtype;
    }
};

struct Fp4BmmIdHash
{
    std::size_t operator()(Fp4BmmId const& id) const
    {
        auto h1 = std::hash<int64_t>{}(id.n);
        auto h2 = std::hash<int64_t>{}(id.k);
        auto h3 = std::hash<int64_t>{}(id.b);
        auto h4 = std::hash<int>{}(static_cast<int>(id.out_dtype));
        return h1 ^ h2 ^ h3 ^ h4;
    }
};

void eventDelete(cudaEvent_t event)
{
    // Do not use check_cuda_error here; we need to swallow all
    // exceptions in destructors.
    if (event != nullptr)
    {
        cudaEventDestroy(event);
    }
}

float profileConfigForProblem(CutlassFp4GemmRunnerInterface& gemmRunner, int64_t m, int64_t n, int64_t k,
    int64_t batch_count, at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, at::Tensor& workspace, int64_t wsBytes,
    tkc::CutlassGemmConfig const& config)
{
    constexpr int warmup = 5;
    constexpr int runs = 10;

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    auto runBmm = [&]
    {
        gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
            mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, batch_count, config,
            reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, stream);
    };

    for (int i = 0; i < warmup; ++i)
    {
        runBmm();
    }

    // Wrap start/stop in unique ptrs so we don't leak memory if there are errors.
    cudaEvent_t start;
    tensorrt_llm::common::check_cuda_error(cudaEventCreate(&start));
    std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, decltype(&eventDelete)> start_ptr{start, &eventDelete};

    cudaEvent_t stop;
    tensorrt_llm::common::check_cuda_error(cudaEventCreate(&stop));
    std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, decltype(&eventDelete)> stop_ptr{stop, &eventDelete};

    tensorrt_llm::common::check_cuda_error(cudaStreamSynchronize(stream));
    tensorrt_llm::common::check_cuda_error(cudaEventRecord(start, stream));

    for (int i = 0; i < runs; ++i)
    {
        runBmm();
    }

    tensorrt_llm::common::check_cuda_error(cudaEventRecord(stop, stream));

    tensorrt_llm::common::check_cuda_error(cudaEventSynchronize(stop));

    float elapsed;
    tensorrt_llm::common::check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));

    return elapsed / runs;
}

std::pair<tkc::CutlassGemmConfig, int64_t> runProfilingFor(
    int64_t m, Fp4BmmId const& gemmId, CutlassFp4GemmRunnerInterface& gemmRunner)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    float bestTime = std::numeric_limits<float>::max();
    tkc::CutlassGemmConfig bestConfig;
    int64_t bestIdx{};

    auto configs = gemmRunner.getConfigs();

    // Check alignment here to avoid confusing errors from failing inside the kernel.
    constexpr int alignment = 32;
    TORCH_CHECK(gemmId.k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got ", gemmId.k);
    TORCH_CHECK(gemmId.n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got ", gemmId.n);

    constexpr auto randMin = std::numeric_limits<uint8_t>::min();
    constexpr auto randMax = std::numeric_limits<uint8_t>::max();

    auto const batch_count = gemmId.b;

    // Note that we have to use randint() as the fp4/sf dtypes are really just aliases for integer types.
    at::Tensor mat1 = at::randint(randMin, randMax, {batch_count, m, gemmId.k / 2}, at::ScalarType::Byte, std::nullopt,
        torch::kCUDA, std::nullopt)
                          .view(FLOAT4_E2M1X2);
    at::Tensor mat2 = at::randint(randMin, randMax, {batch_count, gemmId.n, gemmId.k / 2}, at::ScalarType::Byte,
        std::nullopt, torch::kCUDA, std::nullopt)
                          .view(FLOAT4_E2M1X2);
    at::Tensor out = at::randn({batch_count, m, gemmId.n}, gemmId.out_dtype, std::nullopt, torch::kCUDA, std::nullopt);

    constexpr int sfVecSize = 16;

    auto const scaleColSize = gemmId.k / sfVecSize;
    auto mat1ScaleNumElems = tensorrt_llm::computeSFSize(m, scaleColSize) * batch_count;
    at::Tensor mat1Scale = at::randint(
        randMin, randMax, {mat1ScaleNumElems}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                               .view(SF_DTYPE);

    auto mat2ScaleNumElems = tensorrt_llm::computeSFSize(gemmId.n, scaleColSize) * batch_count;
    at::Tensor mat2Scale = at::randint(
        randMin, randMax, {mat2ScaleNumElems}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                               .view(SF_DTYPE);

    at::Tensor globalScale = at::randn({1}, at::ScalarType::Float, std::nullopt, torch::kCUDA, std::nullopt);

    int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, gemmId.n, gemmId.k, batch_count);
    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    for (int64_t i = 0; i < static_cast<int64_t>(configs.size()); ++i)
    {
        auto& config = configs[i];
        try
        {
            float const time = profileConfigForProblem(gemmRunner, m, gemmId.n, gemmId.k, batch_count, out, mat1, mat2,
                mat1Scale, mat2Scale, globalScale, workspace, wsBytes, config);

            if (time < bestTime)
            {
                bestTime = time;
                bestConfig = config;
                bestIdx = i;
            }
        }
        catch (std::exception const& e)
        {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << config.toString() << "\n (for"
                << " m=" << m << ", n=" << gemmId.n << ", k=" << gemmId.k << ")"
                << ", reason: \"" << e.what() << "\". Skipped";
            TLLM_LOG_TRACE(msg.str());
            cudaGetLastError(); // Reset the last cudaError to cudaSuccess.
        }
    }

    if (std::isinf(bestTime))
    {
        std::ostringstream msg;
        msg << "Have not found any valid GEMM config for shape ("
            << "m=" << m << ", n=" << gemmId.n << ", k=" << gemmId.k << "). Will try to use default or fail at"
            << "runtime";
        TLLM_LOG_WARNING(msg.str());
        return {getDefaultGemmConfig(m, gemmId.n, gemmId.k), -1};
    }
    return {bestConfig, bestIdx};
}

class Fp4BmmProfiler
{
public:
    // Maps values of M to the best config and the config's index.
    using MProfileMap = std::unordered_map<int64_t, std::pair<tkc::CutlassGemmConfig, int64_t>>;

    void profileTactics(int64_t m, Fp4BmmId const& gemmId, CutlassFp4GemmRunnerInterface& gemmRunner)
    {
        if (getBestConfigImpl(m, gemmId) != std::nullopt)
        {
            return;
        }

        auto bestConfigAndIdx = runProfilingFor(m, gemmId, gemmRunner);
        setBestConfig(m, gemmId, bestConfigAndIdx);
    }

    std::optional<tkc::CutlassGemmConfig> getBestConfig(int64_t m, Fp4BmmId const& gemmId)
    {
        auto result = getBestConfigImpl(m, gemmId);
        if (result != std::nullopt)
        {
            return result->first;
        }
        return std::nullopt;
    }

    std::optional<int64_t> getBestConfigIdx(int64_t m, Fp4BmmId const& gemmId)
    {
        auto result = getBestConfigImpl(m, gemmId);
        if (result != std::nullopt)
        {
            return result->second;
        }
        return std::nullopt;
    }

private:
    std::optional<std::pair<tkc::CutlassGemmConfig, int64_t>> getBestConfigImpl(int64_t m, Fp4BmmId const& gemmId)
    {
        auto profileMapIt = mProfileMap.find(gemmId);
        if (profileMapIt == mProfileMap.end())
        {
            return std::nullopt;
        }

        auto& mValToConfig = profileMapIt->second;
        auto configIt = mValToConfig.find(m);
        if (configIt != mValToConfig.end())
        {
            return configIt->second;
        }
        return std::nullopt;
    }

    void setBestConfig(
        int64_t m, Fp4BmmId const& gemmId, std::pair<tkc::CutlassGemmConfig, int64_t> const& configAndIdx)
    {
        // Note that profileMapIt will point to the existing map associated
        // with gemmId if such a map exists.
        auto [profileMapIt, _] = mProfileMap.try_emplace(gemmId, MProfileMap{});
        auto& profileMap = profileMapIt->second;
        profileMap.emplace(m, configAndIdx);
    }

    std::unordered_map<Fp4BmmId, MProfileMap, Fp4BmmIdHash> mProfileMap;
};

} // namespace

at::Tensor fp4_bmm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype)
{
    // The functional version of this op does not do any profiling; use the profiler class below instead for
    // better performance.
    // Note that we can still add a heuristic here.
    return fp4_bmm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, sfUseUE8M0, out_dtype);
}

class FP4BmmRunner : public torch::CustomClassHolder
{
public:
    explicit FP4BmmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
        if (outputDtype == at::ScalarType::Half)
        {
            mGemmRunner = std::make_unique<CutlassFp4GemmRunner<half>>();
        }
        else if (outputDtype == at::ScalarType::Float)
        {
            mGemmRunner = std::make_unique<CutlassFp4GemmRunner<float>>();
        }
#ifdef ENABLE_BF16
        else if (outputDtype == at::ScalarType::BFloat16)
        {
            mGemmRunner = std::make_unique<CutlassFp4GemmRunner<__nv_bfloat16>>();
        }
#endif
        else
        {
            C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
        }
        mConfigs = mGemmRunner->getConfigs();
    }

    static c10::intrusive_ptr<FP4BmmRunner> getInstance(c10::ScalarType outputDtype)
    {
        static std::mutex instance_mutex;
        static std::unordered_map<c10::ScalarType, c10::intrusive_ptr<FP4BmmRunner>> dtype_to_instance;

        std::lock_guard lock(instance_mutex);
        auto instance_it = dtype_to_instance.find(outputDtype);

        if (instance_it != dtype_to_instance.end())
        {
            return instance_it->second;
        }

        auto result = c10::make_intrusive<FP4BmmRunner>(outputDtype);
        dtype_to_instance.emplace(outputDtype, result);
        return result;
    }

    void runProfile(int64_t n, int64_t k, int64_t b, std::vector<int64_t> buckets)
    {
        TORCH_CHECK(!buckets.empty(), "At least one bucket must be specified");

        std::lock_guard lk(mProfilerMutex);

        mBuckets = std::move(buckets);
        std::sort(mBuckets.begin(), mBuckets.end());

        for (auto m : mBuckets)
        {
            TORCH_CHECK(m > 0, "Bucket sizes must be positive.");
            auto gemmId = Fp4BmmId(n, k, b, mOutputDtype);
            TORCH_CHECK(mGemmRunner != nullptr);
            mGemmProfiler.profileTactics(m, gemmId, *mGemmRunner);
        }
    }

    int64_t getBestConfigId(int64_t m, int64_t n, int64_t k, int64_t b)
    {
        std::lock_guard lk(mProfilerMutex);

        auto bucketIt = std::upper_bound(mBuckets.begin(), mBuckets.end(), m);

        if (bucketIt == mBuckets.begin())
        {
            return getConfigIdForBucket(mBuckets[0], n, k, b);
        }

        return getConfigIdForBucket(*(bucketIt - 1), n, k, b);
    }

    at::Tensor runBmm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0, int64_t configIdx) const
    {
        TORCH_CHECK(configIdx >= 0 && configIdx < static_cast<int64_t>(mConfigs.size()));
        auto const& config = mConfigs.at(configIdx);
        return fp4_bmm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, sfUseUE8M0, mOutputDtype, &config);
    }

private:
    int64_t getConfigIdForBucket(int64_t bucket, int64_t n, int64_t k, int64_t b)
    {
        auto gemmId = Fp4BmmId(n, k, b, mOutputDtype);
        auto configIdx = mGemmProfiler.getBestConfigIdx(bucket, gemmId);
        TORCH_CHECK(configIdx != std::nullopt, "Need to run profiling before getting best config");
        return *configIdx;
    }

    std::mutex mProfilerMutex;
    Fp4BmmProfiler mGemmProfiler{};
    std::shared_ptr<CutlassFp4GemmRunnerInterface> mGemmRunner{nullptr};
    std::vector<tkc::CutlassGemmConfig> mConfigs;
    at::ScalarType mOutputDtype;
    std::vector<int64_t> mBuckets;
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::FP4BmmRunner>("FP4BmmRunner")
        .def_static("get_instance", &torch_ext::FP4BmmRunner::getInstance)
        .def("run_bmm_kmajor", &torch_ext::FP4BmmRunner::runBmm)
        .def("run_profile", &torch_ext::FP4BmmRunner::runProfile)
        .def("get_best_config_id", &torch_ext::FP4BmmRunner::getBestConfigId);

    m.def(
        "fp4_bmm_kmajor(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor globalScale, "
        "bool sfUseUE8M0, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_bmm_kmajor", &torch_ext::fp4_bmm);
}
