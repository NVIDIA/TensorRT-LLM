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

#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#if defined(USING_OSS_CUTLASS_FP4_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
#else
#include "fp4_gemm.h"
#endif

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

#if defined(USING_OSS_CUTLASS_FP4_GEMM)
using Fp4GemmRunnerPtr = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunnerInterface>;
#else
using Fp4GemmRunnerPtr
    = std::shared_ptr<tensorrt_llm::kernels::internal_cutlass_kernels::CutlassFp4GemmRunnerInterface>;
#endif

class Fp4GemmPluginProfiler : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                  Fp4GemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    tensorrt_llm::common::QuantMode mQuantMode;
};

class Fp4GemmPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<Fp4GemmPluginProfiler>;

    Fp4GemmPlugin() = delete;

    Fp4GemmPlugin(int sfVecSize, nvinfer1::DataType OutputType, PluginProfilerPtr const& pluginProfiler);

    Fp4GemmPlugin(void const* data, size_t length, PluginProfilerPtr const& pluginProfiler);

    ~Fp4GemmPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    };

    IndexType getInputSFTensorIdx() const
    {
        return 1;
    };

    IndexType getWeightsTensorIdx() const
    {
        return 2;
    };

    IndexType getWeightsSFTensorIdx() const
    {
        return 3;
    };

    IndexType getGlobalSFTensorIdx() const
    {
        return 4;
    }

    void init(nvinfer1::DataType type);
    void configGemm();

    Fp4GemmRunnerPtr mGemmRunner;
    PluginProfilerPtr mPluginProfiler;

    int mSfVecSize;
    nvinfer1::DataType mOutputType;
    size_t m_workspaceMaxSize;
    GemmDims mDims{};
    GemmIdCore mGemmId{};
};

class Fp4GemmPluginCreator : public BaseCreator
{
public:
    Fp4GemmPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<Fp4GemmPluginProfiler> mGemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
