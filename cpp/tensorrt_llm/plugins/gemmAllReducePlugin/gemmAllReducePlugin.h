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

#include "gemmAllReducePluginProfiler.h"
#include "gemmAllReducePluginResource.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

using namespace nvinfer1;

using nvinfer1::DataType;
#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
namespace cutlass_kernels = ::tensorrt_llm::kernels::opened_cutlass_kernels;
#else
namespace cutlass_kernels = ::tensorrt_llm::kernels::cutlass_kernels;
#endif

using LaunchConfig = typename cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig;

namespace tensorrt_llm::plugins
{
struct GemmAllReducePluginOptions
{
    // Don't need to specify problem shape, this
    // is specified in configurePlugin
    DataType typeA;
    DataType typeB;
    DataType typeD;
    int transA;
    int transB;
    float alpha;
    // ranks participating in collective
    std::set<int> group;
    int groupSize;
    // Set in configurePlugin during build phase
    GemmDims maxProblemShape;
    bool deserialize; // used for profiler instantiation
    int8_t hasSFA = 0;
    int8_t hasSFB = 0;
    int8_t alphaIsPtr = 0;
};

class GemmAllReducePlugin : public BasePlugin
{
    friend class GemmAllReducePluginCreator;

public:
    ~GemmAllReducePlugin() override = default;

    //////////////////////////////////
    // IPluginV2DynamicExt Methods
    //////////////////////////////////
    IPluginV2DynamicExt* clone() const noexcept override;

    DimsExprs getOutputDimensions(
        int outputIndex, DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // inOut[0] -> activation
    // inOut[1] -> weight
    // inOut[2] -> result
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    // in[0] -> activation
    // in[1] -> weight
    // no bias needed
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    // in[0] -> activation
    // in[1] -> weight
    // out[0] -> result_uc
    // out[1] -> result_mc
    int enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    //////////////////////////////////
    // IPluginV2Ext Methods
    //////////////////////////////////
    DataType getOutputDataType(int index, DataType const* inputTypes, int nbInputs) const noexcept override;

    //////////////////////////////////
    // IPluginV2 Methods
    //////////////////////////////////
    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    int getNbOutputs() const noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;

private:
    explicit GemmAllReducePlugin(GemmAllReducePluginOptions const& options);
    // Parameterized constructor
    explicit GemmAllReducePlugin(void const* data, size_t length);

    void allocatePersistentWorkspace();

    LaunchConfig getStaticHeuristicLaunchConfig(int M) const;

    // Params that are initialized during constructor
    using KeyType = std::tuple<DataType, DataType, DataType>;
    using ValueType = std::function<cutlass_kernels::GemmAllReduceImplInterface*()>;
    GemmAllReducePluginOptions mOptions;
    int mRank = 0;

    enum TensorArg
    {
        IN_ACTIVATION,
        IN_ACTIVATION_SF,
        IN_WEIGHT,
        IN_WEIGHT_SF,
        IN_ALPHA,
        OUT_D_UC,
        OUT_D_MC,
        OUT_D_IPC
    };

    std::unordered_map<int, TensorArg> mArgMap;
    std::unordered_map<TensorArg, int> mArgInvMap;
    int mNbInputs = 0;
    int mNbOutputs = 0;

    std::map<KeyType, ValueType> mTypedInstantiators;
    std::string mWorkspaceKey;
    std::shared_ptr<cutlass_kernels::GemmAllReduceImplInterface> mGemm;
    // Params that are initialized during configurePlugin()
    GemmAllReducePersistentWorkspace* mWorkspace = nullptr;

    // Used for selecting best GEMM for given problem shapes
    GemmIdCore mGemmId{};
    GemmPluginProfilerManager<GemmAllReducePluginProfiler> mGemmPluginProfileManager;
    std::shared_ptr<GemmAllReducePluginProfiler> mProfiler;
};

class GemmAllReducePluginCreator : public BaseCreator
{
public:
    GemmAllReducePluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
