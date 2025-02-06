/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "tensorrt_llm/kernels/cutlass_kernels/allreduce_gemm/allreduce_gemm_runner.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;

namespace tensorrt_llm::plugins
{
/*
 * Used for tuning to find best GEMM configs for different problem shapes.
 * WARNING: Tuning GEMM+AR kernel may not be fully representable of real
 * multi-GPU workloads as tuning only runs on single-GPU.
 */
class GemmAllReducePluginProfiler : public GemmPluginProfiler<GemmAllReduceImplInterface::LaunchConfig,
                                        std::shared_ptr<GemmAllReduceImplInterface>, GemmIdCore, GemmIdCoreHash>
{
public:
    void serializeToOwnFile(GemmIdCore gemmId);

    void deserializeFromOwnFile(GemmIdCore gemmId, GemmDims problemShape);

protected:
    ////////////////////////////////////
    // GemmPluginProfiler methods
    ////////////////////////////////////
    void runTactic(int m, int n, int k, GemmAllReduceImplInterface::LaunchConfig const& tactic, char* workspace,
        cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<GemmAllReduceImplInterface::LaunchConfig> getTactics(int m, int n, int k) const override;

private:
    static std::string getCacheFileName(GemmIdCore gemmId);
};

} // namespace tensorrt_llm::plugins
