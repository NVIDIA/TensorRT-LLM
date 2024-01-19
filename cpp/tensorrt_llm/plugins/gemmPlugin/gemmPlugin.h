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
#ifndef TRT_GEMM_PLUGIN_H
#define TRT_GEMM_PLUGIN_H
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

using CublasGemmWrapper = tensorrt_llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class CublasLtGemmPluginProfiler
    : public GemmPluginProfiler<cublasLtMatmulHeuristicResult_t, CublasGemmWrapperPtr, GemmIdCublas, GemmIdCublasHash>
{
public:
    using Config = cublasLtMatmulHeuristicResult_t;

    void setTranspose(bool transposeA, bool transposeB)
    {
        mTransA = transposeA;
        mTransB = transposeB;
    }

    void setOutputType(nvinfer1::DataType type)
    {
        mOutputType = type;
    }

protected:
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    bool checkTactic(int m, int n, int k, const Config& tactic) const override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    bool mTransA;
    bool mTransB;
    nvinfer1::DataType mOutputType;

    static constexpr size_t ALIGNMENT = 256;
};

class GemmPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<CublasLtGemmPluginProfiler>;

    GemmPlugin() = delete;

    GemmPlugin(int transA, int transB, nvinfer1::DataType type, bool useFp8, const PluginProfilerPtr& profiler);

    GemmPlugin(const void* data, size_t length, const PluginProfilerPtr& profiler);

    ~GemmPlugin() override = default;

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

private:
    void init();
    void configGemm();
    void setGemmConfig();

private:
    const std::string mLayerName;

    int mTransA;
    int mTransB;
    nvinfer1::DataType mType;
    nvinfer1::DataType mOutputType;

    // @fixme: seems this is shared across multiple clones.
    // If we deep copy the wrapper inside clone(), then we may avoid the mutex inside the wrapper?
    CublasGemmWrapperPtr mCublasWrapper;

    GemmDims mDims{};
    GemmIdCublas mGemmId{};
    bool mUseFp8{false};

    PluginProfilerPtr mPluginProfiler;
};

class GemmPluginCreator : public BaseCreator
{
public:
    GemmPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<CublasLtGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins

#endif // TRT_GEMM_PLUGIN_H
