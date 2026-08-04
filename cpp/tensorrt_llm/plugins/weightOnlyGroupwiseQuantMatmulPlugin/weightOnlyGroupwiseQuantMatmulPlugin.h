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
#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv//kernelLauncher.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <cuda_runtime.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace tensorrt_llm::plugins
{

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;
using KernelType = tensorrt_llm::kernels::weight_only::KernelType;

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

    void setCudaKernelType(KernelType cudaKernelType, int arch)
    {
        mCudaKernelType = cudaKernelType;
        mArch = arch;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

    bool checkTactic(int m, int n, int k, Config const& tactic) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
    KernelType mCudaKernelType;
    int mArch;
};

class WeightOnlyGroupwiseQuantMatmulPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;

    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    WeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, float alpha, PluginProfilerPtr const& profiler);

    WeightOnlyGroupwiseQuantMatmulPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~WeightOnlyGroupwiseQuantMatmulPlugin() override = default;

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
    // group_size: 64, 128
    void init(nvinfer1::DataType type, int quant_algo, int group_size, float alpha);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    bool mCudaKernelEnabled;
    tensorrt_llm::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    // When M is smaller than this value, we trigger a fast path
    // I.e. a tailored kernel instead of cutlass.

    int mQuantAlgo;

    int mGroupSize;

    float mAlpha = 1.0f;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class WeightOnlyGroupwiseQuantMatmulPluginCreator : public BaseCreator
{
public:
    WeightOnlyGroupwiseQuantMatmulPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
