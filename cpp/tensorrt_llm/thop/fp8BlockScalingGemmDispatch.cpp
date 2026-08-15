/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/fp8BlockScalingGemmDispatch.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/version.h"

#include <ATen/cuda/CUDAContext.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
constexpr char const* kCachePathEnv = "TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_CACHE";
constexpr char const* kSmallMEnv = "TRTLLM_FP8_BLOCK_SCALING_GEMM_SMALL_M";
constexpr char const* kDebugEnv = "TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_DEBUG";
constexpr char const* kBuildIdEnv = "TRTLLM_BUILD_ID";
constexpr int64_t kDefaultSmallM = 512;
constexpr size_t kMaxDevices = 128;

enum class ActivationScaleLayout : uint8_t
{
    Unsupported = 0,
    Logical = 1,
    Padded = 2,
    Transposed = 4,
};

struct ShapeKey
{
    int64_t m;
    int64_t n;
    int64_t k;

    bool operator==(ShapeKey const& other) const
    {
        return m == other.m && n == other.n && k == other.k;
    }
};

struct ShapeKeyHash
{
    size_t operator()(ShapeKey const& key) const
    {
        auto hash = std::hash<int64_t>{}(key.m);
        hash ^= std::hash<int64_t>{}(key.n) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int64_t>{}(key.k) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct DispatchState
{
    std::string cachePath;
    std::string smallMConfig;
    std::string debugConfig;
    int device{-1};
    int64_t smallM{kDefaultSmallM};
    bool debug{false};
    std::unordered_map<ShapeKey, uint8_t, ShapeKeyHash> deepGemmLayouts;

    bool matches(
        std::string_view path, std::string_view smallMValue, std::string_view debugValue, int deviceIndex) const
    {
        return std::string_view(cachePath) == path && std::string_view(smallMConfig) == smallMValue
            && std::string_view(debugConfig) == debugValue && device == deviceIndex;
    }
};

std::mutex gStateMutex;
std::array<std::shared_ptr<DispatchState const>, kMaxDevices> gStates;
std::string gDeepGemmVersion;

std::string_view envValue(char const* name)
{
    auto const* value = std::getenv(name);
    return value == nullptr ? std::string_view{} : std::string_view(value);
}

std::filesystem::path getSharedLibraryPath()
{
#if defined(__linux__)
    Dl_info info{};
    if (dladdr(reinterpret_cast<void const*>(&getSharedLibraryPath), &info) != 0 && info.dli_fname != nullptr)
    {
        return std::filesystem::path(info.dli_fname);
    }
#endif
    return {};
}

std::string computeRuntimeBuildId()
{
    auto const path = getSharedLibraryPath();
    std::ifstream input(path, std::ios::binary);
    if (path.empty() || !input)
    {
        return "unknown";
    }

    constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    uint64_t hash = kFnvOffset;
    std::array<char, 64 * 1024> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        for (std::streamsize index = 0; index < input.gcount(); ++index)
        {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(index)]);
            hash *= kFnvPrime;
        }
    }

    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string currentRuntimeBuildId()
{
    auto const configured = envValue(kBuildIdEnv);
    if (!configured.empty())
    {
        return std::string(configured);
    }
    static std::string const buildId = computeRuntimeBuildId();
    return buildId;
}

std::string getDeviceClass(std::string deviceName)
{
    auto upperDeviceName = deviceName;
    std::transform(upperDeviceName.begin(), upperDeviceName.end(), upperDeviceName.begin(),
        [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
    for (auto const* deviceClass : {"GH200", "H200", "H100", "B200"})
    {
        if (upperDeviceName.find(deviceClass) != std::string::npos)
        {
            return deviceClass;
        }
    }
    return deviceName;
}

int64_t parseSmallM(std::string_view configured)
{
    if (configured.empty())
    {
        return kDefaultSmallM;
    }
    try
    {
        size_t parsed = 0;
        auto const value = std::stoll(std::string(configured), &parsed);
        TORCH_CHECK(
            parsed == configured.size() && value >= 0, kSmallMEnv, " must be a non-negative integer, got ", configured);
        return value;
    }
    catch (std::exception const&)
    {
        TORCH_CHECK(false, kSmallMEnv, " must be a non-negative integer, got ", configured);
    }
    return kDefaultSmallM;
}

ActivationScaleLayout parseActivationLayout(std::string const& layout)
{
    if (layout == "logical_m_k_blocks")
    {
        return ActivationScaleLayout::Logical;
    }
    if (layout == "trt_padded_1d")
    {
        return ActivationScaleLayout::Padded;
    }
    if (layout == "trt_transposed_k_m")
    {
        return ActivationScaleLayout::Transposed;
    }
    return ActivationScaleLayout::Unsupported;
}

bool cacheIdentityMatches(nlohmann::json const& identity, int device, std::string const& deepGemmVersion)
{
    cudaDeviceProp properties{};
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    auto const sm = properties.major * 10 + properties.minor;
#if defined(TRTLLM_ENABLE_DEEP_GEMM_THOP)
    constexpr bool kDeepGemmBuilt = true;
#else
    constexpr bool kDeepGemmBuilt = false;
#endif
    auto const deepGemmAvailable = kDeepGemmBuilt && sm == 90;
    auto const expectedCandidates = deepGemmAvailable ? std::vector<std::string>{"sm90_trt", "sm90_deep_gemm_1d2d"}
                                                      : std::vector<std::string>{"sm" + std::to_string(sm) + "_trt"};
    auto const buildId = currentRuntimeBuildId();
    return buildId != "unknown" && identity.at("sm").get<int>() == sm
        && identity.at("device_class").get<std::string>() == getDeviceClass(properties.name)
        && identity.at("trtllm_version").get<std::string>() == executor::kTensorRtLlmVersion
        && identity.at("trtllm_build_id").get<std::string>() == buildId
        && identity.at("deep_gemm_version").get<std::string>() == deepGemmVersion
        && identity.at("deep_gemm_available").get<bool>() == deepGemmAvailable
        && identity.at("policy_version").get<int>() == 1
        && identity.at("backend_candidates").get<std::vector<std::string>>() == expectedCandidates;
}

void loadCache(DispatchState& state, std::string const& deepGemmVersion)
{
    if (state.cachePath.empty())
    {
        return;
    }
    try
    {
        std::ifstream input(state.cachePath);
        TORCH_CHECK(input, "cannot open cache file");
        auto const payload = nlohmann::json::parse(input);
        TORCH_CHECK(payload.at("schema_version").get<int>() == 1, "unsupported cache schema");
        TORCH_CHECK(cacheIdentityMatches(payload.at("identity"), state.device, deepGemmVersion),
            "cache identity does not match this runtime");
        auto const deepGemmAvailable = payload.at("identity").at("deep_gemm_available").get<bool>();

        std::unordered_set<std::string> exactKeys;
        for (auto const& entry : payload.at("entries"))
        {
            auto const m = entry.at("m").get<int64_t>();
            auto const n = entry.at("n").get<int64_t>();
            auto const k = entry.at("k").get<int64_t>();
            auto const activationLayout = entry.at("activation_scale_layout").get<std::string>();
            auto const weightLayout = entry.at("weight_scale_layout").get<std::string>();
            auto const matrixLayout = entry.at("matrix_layout").get<std::string>();
            auto const backend = entry.at("backend").get<std::string>();
            TORCH_CHECK(backend == "trtllm" || backend == "deep_gemm", "unsupported backend ", backend);
            TORCH_CHECK(
                backend != "deep_gemm" || deepGemmAvailable, "DeepGEMM cache entry is not valid for this runtime");
            auto const exactKey = std::to_string(m) + ":" + std::to_string(n) + ":" + std::to_string(k) + ":"
                + activationLayout + ":" + weightLayout + ":" + matrixLayout;
            TORCH_CHECK(exactKeys.insert(exactKey).second, "duplicate cache key");

            auto const layout = parseActivationLayout(activationLayout);
            if (backend == "deep_gemm" && layout != ActivationScaleLayout::Unsupported
                && weightLayout == "logical_n_k_blocks" && matrixLayout == "k_major_contiguous")
            {
                state.deepGemmLayouts[{m, n, k}] |= static_cast<uint8_t>(layout);
            }
        }
    }
    catch (c10::Error const& error)
    {
        state.deepGemmLayouts.clear();
        TLLM_LOG_WARNING("Ignoring invalid FP8 block-scaling dispatch cache %s: %s", state.cachePath.c_str(),
            error.what_without_backtrace());
    }
    catch (std::exception const& error)
    {
        state.deepGemmLayouts.clear();
        TLLM_LOG_WARNING(
            "Ignoring invalid FP8 block-scaling dispatch cache %s: %s", state.cachePath.c_str(), error.what());
    }
}

bool isStreamCapturing(cudaStream_t stream)
{
    cudaStreamCaptureStatus captureStatus{};
    TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
    return captureStatus != cudaStreamCaptureStatusNone;
}

std::shared_ptr<DispatchState const> getDispatchState(int device)
{
    TORCH_CHECK(device >= 0 && static_cast<size_t>(device) < gStates.size(), "Unsupported CUDA device index ", device);
    auto const cachePath = envValue(kCachePathEnv);
    auto const smallMConfig = envValue(kSmallMEnv);
    auto const debugConfig = envValue(kDebugEnv);
    auto* stateSlot = &gStates[static_cast<size_t>(device)];
    auto state = std::atomic_load(stateSlot);
    if (state != nullptr && state->matches(cachePath, smallMConfig, debugConfig, device))
    {
        return state;
    }
    if (isStreamCapturing(at::cuda::getCurrentCUDAStream(device)))
    {
        return {};
    }

    std::lock_guard<std::mutex> lock(gStateMutex);
    state = std::atomic_load(stateSlot);
    if (state != nullptr && state->matches(cachePath, smallMConfig, debugConfig, device))
    {
        return state;
    }

    auto next = std::make_shared<DispatchState>();
    next->cachePath = cachePath;
    next->smallMConfig = smallMConfig;
    next->debugConfig = debugConfig;
    next->device = device;
    next->smallM = parseSmallM(smallMConfig);
    next->debug = debugConfig == "1";
    loadCache(*next, gDeepGemmVersion);
    std::atomic_store(stateSlot, std::shared_ptr<DispatchState const>(next));
    return next;
}

ActivationScaleLayout classifyActivationScaleLayout(torch::Tensor const& scale, int64_t m, int64_t kBlocks)
{
    if (scale.dim() == 2 && scale.size(0) == m && scale.size(1) == kBlocks)
    {
        return ActivationScaleLayout::Logical;
    }
    if (scale.dim() == 2 && scale.size(0) == kBlocks && scale.size(1) >= m)
    {
        return ActivationScaleLayout::Transposed;
    }
    auto const mPadded = ((m + 3) / 4) * 4;
    if (scale.dim() == 1 && scale.numel() >= kBlocks * mPadded)
    {
        return ActivationScaleLayout::Padded;
    }
    return ActivationScaleLayout::Unsupported;
}

void logDecisionOnce(DispatchState const& state, ShapeKey const& key, char const* backend, char const* reason)
{
    if (!state.debug)
    {
        return;
    }
    static std::mutex logMutex;
    static std::unordered_set<std::string> logged;
    auto const signature = std::to_string(key.m) + ":" + std::to_string(key.n) + ":" + std::to_string(key.k) + ":"
        + backend + ":" + reason;
    std::lock_guard<std::mutex> lock(logMutex);
    if (logged.insert(signature).second)
    {
        TLLM_LOG_WARNING("FP8 block-scaling dispatch debug shape=%lldx%lldx%lld backend=%s reason=%s",
            static_cast<long long>(key.m), static_cast<long long>(key.n), static_cast<long long>(key.k), backend,
            reason);
    }
}

} // namespace

void configureFp8BlockScalingGemmDispatch(std::string const& deepGemmVersion)
{
    std::lock_guard<std::mutex> lock(gStateMutex);
    gDeepGemmVersion = deepGemmVersion;
    for (auto& state : gStates)
    {
        std::atomic_store(&state, std::shared_ptr<DispatchState const>{});
    }
}

std::string fp8BlockScalingGemmRuntimeBuildId()
{
    return currentRuntimeBuildId();
}

bool shouldUseDeepGemm(torch::Tensor const& mat1, torch::Tensor const& mat2, torch::Tensor const& mat1Scale,
    torch::Tensor const& mat2Scale)
{
#if !defined(TRTLLM_ENABLE_DEEP_GEMM_THOP)
    static_cast<void>(mat1);
    static_cast<void>(mat2);
    static_cast<void>(mat1Scale);
    static_cast<void>(mat2Scale);
    return false;
#else
    if (!mat1.is_cuda() || mat1.dim() != 2 || mat2.dim() != 2 || mat1.size(0) <= 0 || mat2.size(0) <= 0
        || mat1.size(1) <= 0 || mat1.size(1) != mat2.size(1))
    {
        return false;
    }
    ShapeKey const key{mat1.size(0), mat2.size(0), mat1.size(1)};
    auto const state = getDispatchState(mat1.get_device());
    if (state == nullptr)
    {
        return false;
    }
    if (key.m <= state->smallM)
    {
        logDecisionOnce(*state, key, "trtllm", "small_m");
        return false;
    }
    if (key.m == 65536 && key.n == 3072 && key.k == 3072)
    {
        logDecisionOnce(*state, key, "trtllm", "denylist");
        return false;
    }
    auto const entry = state->deepGemmLayouts.find(key);
    if (entry == state->deepGemmLayouts.end())
    {
        logDecisionOnce(*state, key, "trtllm", "cache_miss");
        return false;
    }

    if (isStreamCapturing(at::cuda::getCurrentCUDAStream(mat1.get_device())))
    {
        logDecisionOnce(*state, key, "trtllm", "capture");
        return false;
    }

    if (key.k <= 0 || key.k % 128 != 0)
    {
        logDecisionOnce(*state, key, "trtllm", "unsupported_layout");
        return false;
    }
    auto const kBlocks = key.k / 128;
    auto const activationLayout = classifyActivationScaleLayout(mat1Scale, key.m, kBlocks);
    auto const matchingLayout = (entry->second & static_cast<uint8_t>(activationLayout)) != 0;
    auto const sameDevice
        = mat1.device() == mat2.device() && mat1.device() == mat1Scale.device() && mat1.device() == mat2Scale.device();
    auto const compatible = matchingLayout && sameDevice && mat1.is_cuda() && mat2.is_cuda()
        && mat1.scalar_type() == at::ScalarType::Float8_e4m3fn && mat2.scalar_type() == at::ScalarType::Float8_e4m3fn
        && mat1Scale.scalar_type() == at::ScalarType::Float && mat2Scale.scalar_type() == at::ScalarType::Float
        && mat1.stride(1) == 1 && mat2.stride(1) == 1 && mat1Scale.is_contiguous() && mat2Scale.is_contiguous()
        && mat2Scale.dim() == 2 && mat2Scale.size(0) == (key.n + 127) / 128 && mat2Scale.size(1) == kBlocks;
    if (!compatible)
    {
        logDecisionOnce(*state, key, "trtllm", "unsupported_layout");
        return false;
    }
    logDecisionOnce(*state, key, "deep_gemm", "cache_hit");
    return true;
#endif
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
