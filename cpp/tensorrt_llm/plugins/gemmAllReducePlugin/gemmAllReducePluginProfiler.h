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
#pragma once

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
#else
#include "allreduce_gemm_runner.h"
#endif
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"

namespace tensorrt_llm::plugins
{
/*
 * Used for tuning to find best GEMM configs for different problem shapes.
 * WARNING: Tuning GEMM+AR kernel may not be fully representable of real
 * multi-GPU workloads as tuning only runs on single-GPU.
 * IMPORTANT: TRT-LLM does not support deterministic tuning across ranks.
 * Because of this, we have to serialize/deserialize our own configuration file.
 */

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
namespace cutlass_kernels = ::tensorrt_llm::kernels::opened_cutlass_kernels;
#else
namespace cutlass_kernels = ::tensorrt_llm::kernels::cutlass_kernels;
#endif
class GemmAllReducePluginProfiler
    : public GemmPluginProfiler<cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig,
          std::shared_ptr<cutlass_kernels::GemmAllReduceImplInterface>, GemmIdCore, GemmIdCoreHash>
{
public:
    void serializeToOwnFile(GemmIdCore gemmId);

    void deserializeFromOwnFile(GemmIdCore gemmId, GemmDims problemShape);

    bool useProfiler();

protected:
    ////////////////////////////////////
    // GemmPluginProfiler methods
    ////////////////////////////////////
    void runTactic(int m, int n, int k, cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig const& tactic,
        char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig> getTactics(
        int m, int n, int k) const override;

private:
    static std::string getCacheFileName(GemmIdCore gemmId);
};

} // namespace tensorrt_llm::plugins
