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
#include "gemmAllReducePlugin.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/plugins/common/pluginUtils.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::plugins
{
void GemmAllReducePluginProfiler::serializeToOwnFile(GemmIdCore gemmId)
{
    std::vector<char> file_buf(getSerializationSize(gemmId));
    char* begin = file_buf.data();
    char* end = file_buf.data();
    serialize(end, gemmId);
    assert(end == begin + file_buf.size());

    auto fileName = getCacheFileName(gemmId);
    std::ofstream file(fileName, std::ios::binary);
    TLLM_CHECK(file.is_open());
    file.write(begin, file_buf.size());
    file.flush();
    file.close();
}

void GemmAllReducePluginProfiler::deserializeFromOwnFile(GemmIdCore gemmId, GemmDims problemShape)
{
    auto fileName = getCacheFileName(gemmId);
    std::ifstream file(fileName, std::ios::binary);
    TLLM_CHECK(file.is_open());
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    TLLM_CHECK(size > 0);
    file.seekg(0, std::ios::beg);

    std::vector<char> file_buf(size);
    file.read(file_buf.data(), size);
    file.close();

    char const* begin = const_cast<char const*>(file_buf.data());
    char const* end = begin;
    deserialize(end, problemShape, gemmId);
    assert(end == begin + size);
}

bool GemmAllReducePluginProfiler::useProfiler()
{
    // char const* envDir = getenv("GEMM_AR_PLUGIN_PROFILE_DIR");
    // return envDir != nullptr;
    // TODO(xsimmons): currently the profiler does not add any perf gain
    // due to static heuristics being sufficient. We can re-enable this
    // when we need more configurations.
    return false;
}

std::string GemmAllReducePluginProfiler::getCacheFileName(GemmIdCore gemmId)
{
    std::stringstream fileName;
    char const* envDir = getenv("GEMM_AR_PLUGIN_PROFILE_DIR");
    std::string directory = envDir ? std::string(envDir) : "/tmp/";
    fileName << directory + "/gemm-AR";
    fileName << "-n" << std::to_string(gemmId.n);
    fileName << "-k" << std::to_string(gemmId.k);
    fileName << "-" << tc::getDtypeString(gemmId.dtype);
    fileName << ".prof_cache";
    return fileName.str();
}

void GemmAllReducePluginProfiler::runTactic(int m, int n, int k,
    cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig const& tactic, char* workspace,
    cudaStream_t const& stream)
{
    const size_t dtype_size = tc::getDTypeSize(mType);
    char* inputA = workspace;
    char* inputB = inputA + m * k * dtype_size;
    char* outputD = inputB + n * k * dtype_size;
    char* inputSFA = outputD + m * n * dtype_size;
    char* inputSFB = inputSFA + m * k * dtype_size;
    std::set<int> tpGroup = {0};

    // Run on single-GPU
    cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args;
    args.argProblemShape(m, n, k, 1)
        .argA((void*) inputA)
        .argB((void*) inputB)
        .argD((void*) outputD, /*output_mc=*/nullptr)
        .argAScale((void*) inputSFA)
        .argBScale((void*) inputSFB)
        .argRanks(0, tpGroup)
        .argAlpha(1.f)
        .argBeta(0.f) // no bias
        .argLaunchConfig(tactic);

    TLLM_CHECK(mRunner != nullptr);
    mRunner->run(args, stream);
}

void GemmAllReducePluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    TLLM_CHECK(maxM != 0);
    TLLM_CHECK(n != 0);
    TLLM_CHECK(k != 0);
    // mType refers to the output data type
    // WARNING: This code assumes that the output precision is >= to input precision
    const size_t dtype_size = tc::getDTypeSize(mType);
    size_t bytes = 0;
    bytes += maxM * k * dtype_size; // A
    bytes += n * k * dtype_size;    // B
    // No C
    // Note that D is typically IPC, however, when tuning GEMM we need it to run on single GPU
    bytes += maxM * n * dtype_size; // D
    // scale tensors for A & B - will at most be same size as A/B
    bytes += maxM * k * dtype_size; // A
    bytes += n * k * dtype_size;    // B

    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig> GemmAllReducePluginProfiler::getTactics(
    int m, int n, int k) const
{
    TLLM_CHECK(mRunner != nullptr);
    return mRunner->getSupportedLaunchConfigs();
}
} // namespace tensorrt_llm::plugins
