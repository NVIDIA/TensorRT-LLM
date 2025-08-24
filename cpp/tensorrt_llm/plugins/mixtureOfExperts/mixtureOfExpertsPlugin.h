/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#ifndef TRT_MIXTURE_OF_EXPERTS_PLUGIN_H
#define TRT_MIXTURE_OF_EXPERTS_PLUGIN_H

#include "NvInferPlugin.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/cutlass_kernel_selector.h"
#if defined(USING_OSS_CUTLASS_MOE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#else
#include "moe_kernels.h"
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/lora/lora.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/cudaStreamPlugin/cudaStreamPlugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using MoeMinLatencyParams = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::MoeMinLatencyParams;
using MOEParallelismConfig = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::MOEParallelismConfig;
using QuantParams = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::QuantParams;
using MoeGemmId = CUTLASS_MOE_GEMM_NAMESPACE::MoeGemmId;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using ActivationParams = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::ActivationParams;
using TmaWarpSpecializedGroupedGemmInput = CUTLASS_MOE_GEMM_NAMESPACE::TmaWarpSpecializedGroupedGemmInput;
using CUTLASS_MOE_GEMM_NAMESPACE::isGatedActivation;

class MixtureOfExpertsGemmProfiler;
using MixtureOfExpertsPluginProfilerPtr = std::shared_ptr<MixtureOfExpertsGemmProfiler>;
using GroupwiseQuantAlgo = tensorrt_llm::common::GroupwiseQuantAlgo;

struct GemmIDMoe
{
    int gemm_idx;
    int num_experts{};
    int experts_per_token{};
    kernels::MOEParallelismConfig parallelism_config{};
    int64_t hidden{};
    int64_t inter{};
    int64_t group_size{};
    ActivationType actfn{};
    nvinfer1::DataType dtype{};
    nvinfer1::DataType wdtype{};
    tensorrt_llm::common::QuantMode quant_mode;
    bool determinism_mode = false;

    bool operator==(GemmIDMoe const& id) const
    {
        return id.gemm_idx == gemm_idx && id.num_experts == num_experts && id.experts_per_token == experts_per_token
            && id.parallelism_config == parallelism_config && id.hidden == hidden && id.inter == inter
            && id.group_size == group_size && id.actfn == actfn && id.dtype == dtype && id.wdtype == wdtype
            && id.quant_mode == quant_mode && id.determinism_mode == determinism_mode;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIDMoe const& id)
    {
        out << "gemm idx, experts, experts_per_token, parallelism_config, hidden, inter, group_size, actfn, dtype, "
               "weight "
               "type, parallelism mode, determinism mode="

            << id.gemm_idx << "," << id.num_experts << "," << id.experts_per_token << "," << id.parallelism_config
            << "," << id.hidden << "," << id.inter << "," << id.group_size << "," << static_cast<int>(id.actfn) << ","
            << static_cast<int>(id.dtype) << "," << static_cast<int>(id.wdtype) << "," << id.quant_mode.value() << ","
            << id.determinism_mode;
        return out;
    }
};

// Hash of GemmIDMoe
struct GemmIDMoeHash
{
    std::size_t operator()(GemmIDMoe const& id) const
    {
        size_t hash = std::hash<int>{}(id.gemm_idx);
        hash ^= std::hash<int>{}(id.num_experts);
        hash ^= std::hash<int>{}(id.experts_per_token);
        hash ^= std::hash<int>{}(id.parallelism_config.tp_size);
        hash ^= std::hash<int>{}(id.parallelism_config.ep_size);
        hash ^= std::hash<int>{}(id.parallelism_config.tp_rank);
        hash ^= std::hash<int>{}(id.parallelism_config.ep_rank);
        hash ^= std::hash<int>{}(id.hidden);
        hash ^= std::hash<int>{}(id.inter);
        hash ^= std::hash<int>{}(id.group_size);
        hash ^= std::hash<int>{}(static_cast<int>(id.actfn));
        hash ^= std::hash<int>{}(static_cast<int>(id.dtype));
        hash ^= std::hash<int>{}(static_cast<int>(id.wdtype));
        hash ^= std::hash<int>{}(static_cast<int>(id.quant_mode.value()));
        return hash;
    }
};

class MixtureOfExpertsPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    using LoraPluginProfilerPtr = std::shared_ptr<CublasLtGemmPluginProfiler>;
    using LoraImplPtr = std::shared_ptr<tensorrt_llm::kernels::LoraImpl>;
    MixtureOfExpertsPlugin() = delete;
    MixtureOfExpertsPlugin(bool remove_input_padding, int number_of_experts, int experts_per_token,
        int expert_hidden_size, int expert_inter_size, int groupwise_quant_algo, int group_size,
        ActivationType activation_type, nvinfer1::DataType type, nvinfer1::DataType weight_type,
        nvinfer1::DataType output_type, tensorrt_llm::common::QuantMode quant_mode, bool use_final_scales,
        bool use_bias, int tp_size, int tp_rank, int ep_size, int ep_rank, bool force_determinism, int side_stream_id,
        MixtureOfExpertsPluginProfilerPtr gemm_profiler_ptr, bool use_lora, nvinfer1::DataType lora_type,
        LoraPluginProfilerPtr lora_profiler, int max_low_rank);
    MixtureOfExpertsPlugin(void const* data, size_t length, MixtureOfExpertsPluginProfilerPtr gemm_profiler_ptr,
        LoraPluginProfilerPtr lora_profiler);
    MixtureOfExpertsPlugin(MixtureOfExpertsPlugin const&);

    void init();

    ~MixtureOfExpertsPlugin() override = default;

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

    int getNbOutputs() const noexcept override
    {
        return 1 + useSideStream();
    }

    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    friend class MixtureOfExpertsGemmProfiler;
    std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> mMOERunner{};
    int mNumExperts{};
    int mExpertsPerToken{};
    int64_t mExpertHiddenSize{};
    int64_t mExpertInterSize{};
    int64_t mGroupwiseQuantAlgo{};
    int64_t mGroupSize{};
    ActivationType mActivationType;
    nvinfer1::DataType mType{};
    nvinfer1::DataType mWeightType{};
    nvinfer1::DataType mOutputType{};
    tensorrt_llm::common::QuantMode mQuantMode;
    bool mUseFinalScales{};
    bool mUseBias{};
    MOEParallelismConfig mParallelismConfig{};

    GemmDims mDims{};
    bool mUseDeterministicKernels = false;
    int mSideStreamId = 0;

    int mDebugStallMain = 0;
    int mDebugStallSide = 0;

    GemmIDMoe mGemmId1{};
    GemmIDMoe mGemmId2{};

    MixtureOfExpertsPluginProfilerPtr mGemmProfiler;

    // lora related
    bool mUseLora{};
    nvinfer1::DataType mLoraType{};
    int mMaxLowRank{};
    bool mRemoveInputPadding{};

    LoraImplPtr mLoraImpl1;
    LoraImplPtr mLoraImpl2;

    GemmIdCublas mLoraGemmId1{};
    GemmIdCublas mLoraGemmId2{};
    LoraPluginProfilerPtr mLoraProfiler;

    std::vector<void const*> mLoraExpandFC1WeightPtrs{};
    std::vector<void const*> mLoraExpandFC2WeightPtrs{};
    std::vector<void const*> mLoraExpandGatedWeightPtrs{};
    std::vector<int32_t> mLoraExpandFC1Ranks{};
    std::vector<int32_t> mLoraExpandFC2Ranks{};
    std::vector<int32_t> mLoraExpandGatedRanks{};

    cudaEvent_t mMemcpyEvent;
    nvinfer1::pluginInternal::SideStream* mSideStreamPtr;

    // The below are not serialised
    std::string const mLayerName{};
    std::string mNamespace{};

    struct WorkspaceInfo
    {
        void* workspace{};
        void* src_to_dest_map{};
        void* lora_workspace{};
        size_t size{};
    };

    int64_t getNumTokens(nvinfer1::PluginTensorDesc const* input_tensor) const;
    WorkspaceInfo setupWorkspace(void* base_ptr, int64_t num_tokens, int num_reqs = 0) const;

    MOEParallelismConfig getParallelismConfig() const;
    QuantParams getQuantParams(nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs,
        int scale_1_idx = -1, int scale_2_idx = -1, int scale_3_idx = -1, int scale_4_idx = -1, int scale_5_idx = -1,
        int scale_6_idx = -1, int scale_7_idx = -1, int scale_8_idx = -1) const;

    int getNumLoraRequests(nvinfer1::PluginTensorDesc const* input_tensor) const;
    tensorrt_llm::kernels::LoraParams getLoraParams(
        nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs, void* workspace);

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

    using IndexType = std::int32_t;

    // Inputs
    constexpr static IndexType getInputTensorIndex()
    {
        return 0;
    }

    constexpr static IndexType getExpertWeights1Index()
    {
        return getInputTensorIndex() + 1;
    }

    constexpr static IndexType getExpertWeights2Index()
    {
        return getExpertWeights1Index() + 1;
    }

    constexpr static IndexType getTokenSelectedExpertsIndex()
    {
        return getExpertWeights2Index() + 1;
    }

    // Conditional inputs, we only allocate a new index if actually used
    bool hasBias() const
    {
        return mUseBias;
    }

    bool hasFinalScales() const
    {
        return mUseFinalScales;
    }

    bool hasExpertIntQuantScales() const
    {
        return mQuantMode.hasInt4Weights() || mQuantMode.hasInt8Weights();
    }

    bool hasExpertFp8QuantScales() const
    {
        return mQuantMode.hasFp8Qdq();
    }

    bool hasExpertFp8FinalQuantScales() const
    {
        return hasExpertFp8QuantScales() && mOutputType == nvinfer1::DataType::kFP8;
    }

    bool hasFP4QuantScales() const
    {
        return mQuantMode.hasNvfp4();
    }

    bool hasGroupwiseIntQuantScales() const
    {
        return mGroupwiseQuantAlgo > 0;
    }

    bool hasExpertWeightQuantZeros() const
    {
        return mGroupwiseQuantAlgo & GroupwiseQuantAlgo::ZERO;
    }

    bool hasExpertPrequantScales() const
    {
        return mGroupwiseQuantAlgo & GroupwiseQuantAlgo::PRE_QUANT_SCALE;
    }

    bool hasGroupwiseFp8Alpha() const
    {
        return mGroupwiseQuantAlgo & GroupwiseQuantAlgo::FP8_ALPHA;
    }

    bool useSideStream() const
    {
        return mSideStreamId > 0;
    }

    bool hasLora() const
    {
        return mUseLora;
    }

    bool hasGatedLoraWeightsAndRanks() const
    {
        return mUseLora && isGatedActivation(mActivationType);
    }

    IndexType getTokenFinalScalesIndex() const
    {
        return getTokenSelectedExpertsIndex() + hasFinalScales();
    }

    IndexType getExpertBias1Index() const
    {
        return getTokenFinalScalesIndex() + hasBias();
    }

    IndexType getExpertBias2Index() const
    {
        return getExpertBias1Index() + hasBias();
    }

    /*
     * Weight-Only int quant scales
     */
    IndexType getExpertIntQuantScale1Index() const
    {
        return getExpertBias2Index() + hasExpertIntQuantScales();
    }

    IndexType getExpertIntQuantScale2Index() const
    {
        return getExpertIntQuantScale1Index() + hasExpertIntQuantScales();
    }

    /*
     * FP8 Quant Scales
     */
    IndexType getExpertFP8Dequant1Index() const
    {
        return getExpertIntQuantScale2Index() + hasExpertFp8QuantScales();
    }

    IndexType getExpertFP8Quant2Index() const
    {
        return getExpertFP8Dequant1Index() + hasExpertFp8QuantScales();
    }

    IndexType getExpertFP8Dequant2Index() const
    {
        return getExpertFP8Quant2Index() + hasExpertFp8QuantScales();
    }

    IndexType getExpertFP8QuantFinalIndex() const
    {
        return getExpertFP8Dequant2Index() + hasExpertFp8FinalQuantScales();
    }

    IndexType getInputFP8DequantIndex() const
    {
        return getExpertFP8QuantFinalIndex() + (hasExpertFp8QuantScales() && hasLora());
    }

    /*
     * FP4 Quant Scales
     */
    IndexType getFP4GlobalActSF1Index() const
    {
        return getInputFP8DequantIndex() + hasFP4QuantScales();
    }

    IndexType getFP4WeightSF1Index() const
    {
        return getFP4GlobalActSF1Index() + hasFP4QuantScales();
    }

    IndexType getFP4GlobalSF1Index() const
    {
        return getFP4WeightSF1Index() + hasFP4QuantScales();
    }

    IndexType getFP4GlobalActSF2Index() const
    {
        return getFP4GlobalSF1Index() + hasFP4QuantScales();
    }

    IndexType getFP4WeightSF2Index() const
    {
        return getFP4GlobalActSF2Index() + hasFP4QuantScales();
    }

    IndexType getFP4GlobalSF2Index() const
    {
        return getFP4WeightSF2Index() + hasFP4QuantScales();
    }

    /*
     * Groupwise Params
     */
    IndexType getExpertPrequantScales1Index() const
    {
        return getFP4GlobalSF2Index() + hasExpertPrequantScales();
    }

    IndexType getExpertPrequantScales2Index() const
    {
        return getExpertPrequantScales1Index() + hasExpertPrequantScales();
    }

    IndexType getExpertIntQuantZeros1Index() const
    {
        return getExpertPrequantScales2Index() + hasExpertWeightQuantZeros();
    }

    IndexType getExpertIntQuantZeros2Index() const
    {
        return getExpertIntQuantZeros1Index() + hasExpertWeightQuantZeros();
    }

    IndexType getExpertFp8Alpha1Index() const
    {
        return getExpertIntQuantZeros2Index() + hasGroupwiseFp8Alpha();
    }

    IndexType getExpertFp8Alpha2Index() const
    {
        return getExpertFp8Alpha1Index() + hasGroupwiseFp8Alpha();
    }

    /*
     * LoRA params
     */
    IndexType getLoraFC1WeightPtrsIndex() const
    {
        return getExpertFp8Alpha2Index() + hasLora();
    }

    IndexType getLoraFC1RanksIndex() const
    {
        return getLoraFC1WeightPtrsIndex() + hasLora();
    }

    IndexType getLoraFC2WeightPtrsIndex() const
    {
        return getLoraFC1RanksIndex() + hasLora();
    }

    IndexType getLoraFC2RanksIndex() const
    {
        return getLoraFC2WeightPtrsIndex() + hasLora();
    }

    IndexType getLoraGatedWeightPtrsIndex() const
    {
        return getLoraFC2RanksIndex() + hasGatedLoraWeightsAndRanks();
    }

    IndexType getLoraGatedRanksIndex() const
    {
        return getLoraGatedWeightPtrsIndex() + hasGatedLoraWeightsAndRanks();
    }

    IndexType getHostRequestTypeIndex() const
    {
        return getLoraGatedRanksIndex() + hasLora();
    }

    IndexType getHostContextLengthIndex() const
    {
        return getHostRequestTypeIndex() + (mRemoveInputPadding && hasLora());
    }

    IndexType getInputDummyTensorIndex() const
    {
        return getHostContextLengthIndex() + useSideStream();
    }

    IndexType getNbInputs() const
    {
        return getInputDummyTensorIndex() + 1;
    }

    // Outputs
    constexpr static IndexType getOutputTensorIndex()
    {
        return 0;
    }

    IndexType getOutputDummyTensorIndex() const
    {
        return getOutputTensorIndex() + useSideStream();
    }

    /**
     * Get the index of the expert shape tuple that represents the inner dimension
     */
    int getGemmShapeInnerDimIndex() const
    {
        // In weight only mode the shape is transposed
        return hasExpertIntQuantScales() ? 1 : 2;
    }

    /**
     * Get the index of the expert shape tuple that represents the outer dimension
     */
    int getGemmShapeOuterDimIndex() const
    {
        // In weight only mode the shape is transposed
        return hasExpertIntQuantScales() ? 2 : 1;
    }

    /**
     * Get quantization dimension scaling factor
     */
    std::pair<int, int> getWeightPackedElements() const
    {
        if (mGroupwiseQuantAlgo == 0)
        {
            return {1, mQuantMode.hasInt4Weights() ? 2 : 1};
        }
        else
        {
            return {1, 4};
        }
    }
};

class MixtureOfExpertsGemmProfiler
    : public tensorrt_llm::plugins::GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
          MixtureOfExpertsPlugin*, GemmIDMoe, GemmIDMoeHash>
{
public:
    MixtureOfExpertsGemmProfiler()
    {
        // NOTE: Do not access mPlugin here, since we are called from the constructor before all fields are init
    }

    void setGemmToProfile(kernels::GemmProfilerBackend::GemmToProfile gemm_to_profile)
    {
        // Just set the backend directly. This will just be reused in checkInit().
        backend.mGemmToProfile = gemm_to_profile;
        // We need to set the backend to reinitialise itself with the new GEMM
        init_backend = false;
    }

    void setMaxProfileM(int maxProfileM)
    {
        mMaxProfileM = maxProfileM;
    }

    virtual int getMaxProfileM() const override
    {
        return mMaxProfileM;
    }

protected:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;
    void computeTmpSize(size_t maxM, size_t n, size_t k) override;
    std::vector<Config> getTactics(int m, int n, int k) const override;
    void initTmpData(int maxM, int n, int k, char* workspace, size_t size, cudaStream_t stream) override;

    void checkInit();

    bool init_backend = false;
    kernels::GemmProfilerBackend backend{};

private:
    int mMaxProfileM = 0;
};

class MixtureOfExpertsPluginCreator : public nvinfer1::IPluginCreator
{
public:
    MixtureOfExpertsPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    GemmPluginProfilerManager<MixtureOfExpertsGemmProfiler> moePluginProfiler;
    GemmPluginProfilerManager<CublasLtGemmPluginProfiler> loraPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace tensorrt_llm::plugins

#endif // TRT_MIXTURE_OF_EXPERTS_PLUGIN_H
