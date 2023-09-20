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
#ifndef TRT_WEIGHT_ONLY_GROUPWISE_QUANT_MATMUL_PLUGIN_H
#define TRT_WEIGHT_ONLY_GROUPWISE_QUANT_MATMUL_PLUGIN_H

#include "NvInferPlugin.h"
#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyGroupwiseMatrixVectorMultiplication.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace nvinfer1
{
namespace plugin
{

class WeightOnlyGroupwiseQuantMatmulPlugin : public IPluginV2DynamicExt
{
public:
    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    WeightOnlyGroupwiseQuantMatmulPlugin(nvinfer1::DataType type, int quant_algo, int group_size);

    WeightOnlyGroupwiseQuantMatmulPlugin(const void* data, size_t length);

    ~WeightOnlyGroupwiseQuantMatmulPlugin() override = default;

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
    // group_size: 64, 128
    void init(nvinfer1::DataType type, int quant_algo, int group_size);

private:
    const std::string mLayerName;
    std::string mNamespace;

    std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>
        m_weightOnlyGroupwiseGemmRunner;
    int m_workspaceMaxSize;
    nvinfer1::DataType mType;

    // When M is smaller than this value, we trigger a fast path
    // I.e. a tailored kernel instead of cutlass.
    static constexpr int SMALL_M_FAST_PATH = 5;

    int mQuantAlgo;

    // Flags for indicating whether the corresponding inputs are applied in mQuantAlgo
    // mQuantAlgo = pre_quant_scale * PRE_SCALE_QUANT + zero * ZER0 + bias * BIAS
    // Here pre_quant_scale, zero and bias are boolean type
    static constexpr int BIAS = int(1) << 0;
    static constexpr int ZER0 = int(1) << 1;
    static constexpr int PRE_SCALE_QUANT = int(1) << 2;

    int mGroupSize;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;
};

class WeightOnlyGroupwiseQuantMatmulPluginCreator : public IPluginCreator
{
public:
    WeightOnlyGroupwiseQuantMatmulPluginCreator();

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

#endif // TRT_WEIGHT_ONLY_GROUPWISE_QUANT_MATMUL_PLUGIN_H
