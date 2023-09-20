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
#ifndef TRT_SMOOTH_QUANT_GEMM_PLUGIN_H
#define TRT_SMOOTH_QUANT_GEMM_PLUGIN_H

#include "NvInferPlugin.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

using perfMapType = std::unordered_map<int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig>;

class SmoothQuantGemmPlugin : public IPluginV2DynamicExt
{
public:
    SmoothQuantGemmPlugin() = delete;

    SmoothQuantGemmPlugin(tensorrt_llm::common::QuantMode quantMode, nvinfer1::DataType type);

    SmoothQuantGemmPlugin(const void* data, size_t length);

    ~SmoothQuantGemmPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    void init(nvinfer1::DataType type);

    void setProblemSize(int minM, int maxM, int n, int k);
    void configGemm();
    void setSelectedTactics(const perfMapType& selected_tactics_map);
    void setMaxM(int maxM);

    void allocateTmpData();
    void freeTmpData();

private:
    const std::string mLayerName;
    std::string mNamespace;

    std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface> m_sqGemmRunner;
    tensorrt_llm::common::QuantMode mQuantMode;
    int m_workspaceMaxSize;
    int mMaxM{-1};
    int mMinM{-1};
    int mN{-1};
    int mK{-1};

    int8_t* mATmp{nullptr};
    int8_t* mBTmp{nullptr};
    void* mCTmp{nullptr};
    float* mAlphaRowTmp{nullptr};
    float* mAlphaColTmp{nullptr};
    char* mWorkspaceTmp{nullptr};

    nvinfer1::DataType mType;
};

class SmoothQuantGemmPluginCreator : public IPluginCreator
{
public:
    SmoothQuantGemmPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SMOOTH_QUANT_GEMM_PLUGIN_H
