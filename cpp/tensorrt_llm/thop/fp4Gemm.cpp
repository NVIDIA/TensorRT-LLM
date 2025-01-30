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

#include <cuda_fp16.h>

#include <cstdint>
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
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, int64_t m, int64_t n, int64_t k,
    tkc::CutlassGemmConfig const& gemmConfig)
{
    CutlassFp4GemmRunner<T> gemmRunner;
    int64_t wsBytes = gemmRunner.getWorkspaceSize(m, n, k);

    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, mat1.device(), std::nullopt);

    gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
        mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, gemmConfig,
        reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, at::cuda::getCurrentCUDAStream(mat1.get_device()));
}

// mat1: [M, K / 2], FLOAT4_E2M1X2
// mat2: [N, K / 2], FLOAT4_E2M1X2
// out: [M, N], fp16/bf16/fp32
// mat1Scale: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// mat2Scale: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
// globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
// Only NVFP4 is currently supported
at::Tensor fp4_gemm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype, tkc::CutlassGemmConfig const* maybe_config = nullptr)
{
    CHECK_INPUT(mat1, FLOAT4_E2M1X2);
    CHECK_INPUT(mat2, FLOAT4_E2M1X2);

    CHECK_INPUT(mat1Scale, SF_DTYPE);
    CHECK_INPUT(mat2Scale, SF_DTYPE);

    CHECK_INPUT(globalScale, at::ScalarType::Float);

    TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");

    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

    auto const m = mat1.sizes()[0];
    auto const n = mat2.sizes()[0];
    auto const k = mat1.sizes()[1] * 2;

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

    at::Tensor out = at::detail::empty_cuda({m, n}, out_dtype.value(), mat1.device(), std::nullopt);

    switch (out_dtype.value())
    {
    case at::ScalarType::Half:
        runGemm<half>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, config);
        break;
    case at::ScalarType::BFloat16:
        runGemm<__nv_bfloat16>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, config);
        break;
    case at::ScalarType::Float:
        runGemm<float>(out, mat1, mat2, mat1Scale, mat2Scale, globalScale, m, n, k, config);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "out_dtype must be one of fp16/bf16/fp32.");
    }
    return out;
}

struct Fp4GemmId
{
    int n;
    int k;
    c10::ScalarType out_dtype;

    Fp4GemmId(int n_, int k_, c10::ScalarType out_dtype_)
        : n(n_)
        , k(k_)
        , out_dtype(out_dtype_)
    {
    }

    Fp4GemmId()
        : n(-1)
        , k(-1)
        , out_dtype(c10::ScalarType::Half)
    {
    }

    bool operator==(Fp4GemmId const& other) const
    {
        return n == other.n && k == other.k && out_dtype == other.out_dtype;
    }
};

struct Fp4GemmIdHash
{
    std::size_t operator()(Fp4GemmId const& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        auto h3 = std::hash<int>{}(static_cast<int>(id.out_dtype));
        return h1 ^ h2 ^ h3;
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

float profileConfigForProblem(CutlassFp4GemmRunnerInterface& gemmRunner, int m, int n, int k, at::Tensor& out,
    at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale, at::Tensor const& mat2Scale,
    at::Tensor const& globalScale, at::Tensor& workspace, int wsBytes, tkc::CutlassGemmConfig const& config)
{
    constexpr int warmup = 5;
    constexpr int runs = 10;

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    auto runGemm = [&]
    {
        gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(), mat1Scale.const_data_ptr(),
            mat2Scale.const_data_ptr(), globalScale.data_ptr<float>(), m, n, k, config,
            reinterpret_cast<char*>(workspace.data_ptr()), wsBytes, stream);
    };

    for (int i = 0; i < warmup; ++i)
    {
        runGemm();
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
        runGemm();
    }

    tensorrt_llm::common::check_cuda_error(cudaEventRecord(stop, stream));

    tensorrt_llm::common::check_cuda_error(cudaEventSynchronize(stop));

    float elapsed;
    tensorrt_llm::common::check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));

    return elapsed / runs;
}

std::pair<tkc::CutlassGemmConfig, int64_t> runProfilingFor(
    int m, Fp4GemmId const& gemmId, CutlassFp4GemmRunnerInterface& gemmRunner)
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

    // Note that we have to use randint() as the fp4/sf dtypes are really just aliases for integer types.
    at::Tensor mat1 = at::randint(
        randMin, randMax, {m, gemmId.k / 2}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                          .view(FLOAT4_E2M1X2);
    at::Tensor mat2 = at::randint(
        randMin, randMax, {gemmId.n, gemmId.k / 2}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                          .view(FLOAT4_E2M1X2);
    at::Tensor out = at::randn({m, gemmId.n}, gemmId.out_dtype, std::nullopt, torch::kCUDA, std::nullopt);

    constexpr int sfVecSize = 16;

    auto const scaleColSize = gemmId.k / sfVecSize;
    auto mat1ScaleNumElems = tensorrt_llm::computeSFSize(m, scaleColSize);
    at::Tensor mat1Scale = at::randint(
        randMin, randMax, {mat1ScaleNumElems}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                               .view(SF_DTYPE);

    auto mat2ScaleNumElems = tensorrt_llm::computeSFSize(gemmId.n, scaleColSize);
    at::Tensor mat2Scale = at::randint(
        randMin, randMax, {mat2ScaleNumElems}, at::ScalarType::Byte, std::nullopt, torch::kCUDA, std::nullopt)
                               .view(SF_DTYPE);

    at::Tensor globalScale = at::randn({1}, at::ScalarType::Float, std::nullopt, torch::kCUDA, std::nullopt);

    int64_t wsBytes = gemmRunner.getWorkspaceSize(m, gemmId.n, gemmId.k);
    at::Tensor workspace = at::detail::empty_cuda({wsBytes}, at::ScalarType::Char, torch::kCUDA, std::nullopt);

    for (int64_t i = 0; i < configs.size(); ++i)
    {
        auto& config = configs[i];
        try
        {
            float time = profileConfigForProblem(gemmRunner, m, gemmId.n, gemmId.k, out, mat1, mat2, mat1Scale,
                mat2Scale, globalScale, workspace, wsBytes, config);

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
            << "m=" << m << ", n=" << gemmId.n << ", k=" << gemmId.k << "). Will try to use default or fail at runtime";
        TLLM_LOG_WARNING(msg.str());
        return {getDefaultGemmConfig(m, gemmId.n, gemmId.k), -1};
    }
    return {bestConfig, bestIdx};
}

class Fp4GemmProfiler
{
public:
    // Maps values of M to the best config and the config's index.
    using MProfileMap = std::unordered_map<int, std::pair<tkc::CutlassGemmConfig, int64_t>>;

    void profileTactics(int m, Fp4GemmId const& gemmId, CutlassFp4GemmRunnerInterface& gemmRunner)
    {
        if (getBestConfigImpl(m, gemmId) != std::nullopt)
        {
            return;
        }

        auto bestConfigAndIdx = runProfilingFor(m, gemmId, gemmRunner);
        setBestConfig(m, gemmId, bestConfigAndIdx);
    }

    std::optional<tkc::CutlassGemmConfig> getBestConfig(int m, Fp4GemmId const& gemmId)
    {
        auto result = getBestConfigImpl(m, gemmId);
        if (result != std::nullopt)
        {
            return result->first;
        }
        return std::nullopt;
    }

    std::optional<int64_t> getBestConfigIdx(int m, Fp4GemmId const& gemmId)
    {
        auto result = getBestConfigImpl(m, gemmId);
        if (result != std::nullopt)
        {
            return result->second;
        }
        return std::nullopt;
    }

private:
    std::optional<std::pair<tkc::CutlassGemmConfig, int64_t>> getBestConfigImpl(int m, Fp4GemmId const& gemmId)
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

    void setBestConfig(int m, Fp4GemmId const& gemmId, std::pair<tkc::CutlassGemmConfig, int64_t> const& configAndIdx)
    {
        // Note that profileMapIt will point to the existing map associated
        // with gemmId if such a map exists.
        auto [profileMapIt, _] = mProfileMap.try_emplace(gemmId, MProfileMap{});
        auto& profileMap = profileMapIt->second;
        profileMap.emplace(m, configAndIdx);
    }

    std::unordered_map<Fp4GemmId, MProfileMap, Fp4GemmIdHash> mProfileMap;
};

} // namespace

at::Tensor fp4_gemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
    at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
    std::optional<c10::ScalarType> out_dtype)
{
    // The functional version of this op does not do any profiling; use the profiler class below instead for
    // better performance.
    // Note that we can still add a heuristic here.
    return fp4_gemm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, sfUseUE8M0, out_dtype);
}

class FP4GemmRunner : public torch::CustomClassHolder
{
public:
    explicit FP4GemmRunner(at::ScalarType outputDtype)
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

    static c10::intrusive_ptr<FP4GemmRunner> getInstance(c10::ScalarType outputDtype)
    {
        static std::mutex instance_mutex;
        static std::unordered_map<c10::ScalarType, c10::intrusive_ptr<FP4GemmRunner>> dtype_to_instance;

        std::lock_guard lock(instance_mutex);
        auto instance_it = dtype_to_instance.find(outputDtype);

        if (instance_it != dtype_to_instance.end())
        {
            return instance_it->second;
        }

        auto result = c10::make_intrusive<FP4GemmRunner>(outputDtype);
        dtype_to_instance.emplace(outputDtype, result);
        return result;
    }

    void runProfile(int64_t n, int64_t k, std::vector<int64_t> buckets)
    {
        TORCH_CHECK(buckets.size() > 0, "At least one bucket must be specified");

        std::lock_guard lk(mProfilerMutex);

        mBuckets = std::move(buckets);
        std::sort(mBuckets.begin(), mBuckets.end());

        for (auto m : mBuckets)
        {
            TORCH_CHECK(m > 0, "Bucket sizes must be positive.");
            auto gemmId = Fp4GemmId(n, k, mOutputDtype);
            TORCH_CHECK(mGemmRunner != nullptr);
            mGemmProfiler.profileTactics(m, gemmId, *mGemmRunner);
        }
    }

    int64_t getBestConfigId(int64_t m, int64_t n, int64_t k)
    {
        std::lock_guard lk(mProfilerMutex);

        auto bucketIt = std::upper_bound(mBuckets.begin(), mBuckets.end(), m);

        if (bucketIt == mBuckets.begin())
        {
            return getConfigIdForBucket(mBuckets[0], n, k);
        }
        else
        {
            return getConfigIdForBucket(*(bucketIt - 1), n, k);
        }
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0, int64_t configIdx) const
    {
        TORCH_CHECK(configIdx >= 0 && configIdx < mConfigs.size());
        auto const& config = mConfigs.at(configIdx);
        return fp4_gemm_impl(mat1, mat2, mat1Scale, mat2Scale, globalScale, sfUseUE8M0, mOutputDtype, &config);
    }

private:
    int64_t getConfigIdForBucket(int64_t bucket, int64_t n, int64_t k)
    {
        auto gemmId = Fp4GemmId(n, k, mOutputDtype);
        auto configIdx = mGemmProfiler.getBestConfigIdx(bucket, gemmId);
        TORCH_CHECK(configIdx != std::nullopt, "Need to run profiling before getting best config");
        return *configIdx;
    }

    std::mutex mProfilerMutex;
    Fp4GemmProfiler mGemmProfiler{};
    std::shared_ptr<CutlassFp4GemmRunnerInterface> mGemmRunner{nullptr};
    std::vector<tkc::CutlassGemmConfig> mConfigs;
    at::ScalarType mOutputDtype;
    std::vector<int64_t> mBuckets;
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::FP4GemmRunner>("FP4GemmRunner")
        .def_static("get_instance", &torch_ext::FP4GemmRunner::getInstance)
        .def("run_gemm", &torch_ext::FP4GemmRunner::runGemm)
        .def("run_profile", &torch_ext::FP4GemmRunner::runProfile)
        .def("get_best_config_id", &torch_ext::FP4GemmRunner::getBestConfigId);

    m.def(
        "fp4_gemm(Tensor mat1, Tensor mat2, Tensor mat1Scale, Tensor mat2Scale, Tensor globalScale, bool sfUseUE8M0, "
        "ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_gemm", &torch_ext::fp4_gemm);
}
