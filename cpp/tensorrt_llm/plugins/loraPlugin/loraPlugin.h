/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_LORA_PLUGIN_H
#define TRT_LORA_PLUGIN_H
#include "tensorrt_llm/kernels/lora/lora.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include <cassert>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class LoraPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<CublasLtGemmPluginProfiler>;
    using ImplPtr = std::shared_ptr<kernels::LoraImpl>;
    using Config = cublasLtMatmulHeuristicResult_t;

    LoraPlugin() = delete;

    LoraPlugin(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB, int num_lora_modules,
        nvinfer1::DataType type, PluginProfilerPtr const& profiler, bool remove_input_padding, int max_low_rank,
        int weight_index);

    LoraPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~LoraPlugin() override = default;

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
    int64_t getNumTokens(nvinfer1::PluginTensorDesc const* input_tensors) const;
    void init();

    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    }

    IndexType getHostRequestTypesIdx() const
    {
        return 1;
    }

    IndexType getLoraRanksIdx() const
    {
        return 2;
    }

    IndexType getLoraWeightsPtrsIdx() const
    {
        return 2 + mNumLoraModules;
    }

    IndexType getHostContextLengthsIdx() const
    {
        TLLM_CHECK(mRemoveInputPadding);
        return 2 + mNumLoraModules + mNumLoraModules;
    }

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    const std::string mLayerName;

    std::vector<int> mOutHiddenSizes;
    int mTransA;
    int mTransB;
    nvinfer1::DataType mType;
    bool mRemoveInputPadding;
    int mNumLoraModules;
    int mInHiddenSize;
    int mMaxLowRank;
    int mWeightIndex;

    std::vector<void const*> mExpandLoraWeightPtrs{};
    std::vector<int32_t> mExpandLoraRanks{};

    GemmDims mDims{};
    GemmIdCublas mGemmId{};

    PluginProfilerPtr mPluginProfiler;
    ImplPtr mLoraImpl;
};

class LoraPluginCreator : public BaseCreator
{
public:
    LoraPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<CublasLtGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins

#endif // TRT_LORA_PLUGIN_H
