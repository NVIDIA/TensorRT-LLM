/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::plugins;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::common::QuantMode;
using tensorrt_llm::common::nextWorkspacePtr;
using tensorrt_llm::common::calculateTotalWorkspaceSize;
using tensorrt_llm::plugins::MixtureOfExpertsPluginCreator;
using tensorrt_llm::plugins::MixtureOfExpertsPlugin;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* MIXTURE_OF_EXPERTS_PLUGIN_VERSION{"1"};
static char const* MIXTURE_OF_EXPERTS_PLUGIN_NAME{"MixtureOfExperts"};
nvinfer1::PluginFieldCollection MixtureOfExpertsPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> MixtureOfExpertsPluginCreator::mPluginAttributes;

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(bool remove_input_padding, int number_of_experts, int top_k,
    int expert_hidden_size, int expert_inter_size, tensorrt_llm::ActivationType activation_type,
    nvinfer1::DataType type, nvinfer1::DataType weight_type, nvinfer1::DataType output_type, QuantMode quant_mode,
    bool use_finished, bool use_bias, int tp_size, int tp_rank, int ep_size, int ep_rank,
    MOEExpertScaleNormalizationMode normalization_mode, float sparse_mixer_epsilon, bool force_determinism,
    int side_stream_id, MixtureOfExpertsPluginProfilerPtr gemm_profiler_ptr, bool use_lora,
    nvinfer1::DataType lora_type, LoraPluginProfilerPtr lora_profiler, int max_low_rank)
    : mNumExperts(number_of_experts)
    , mK(top_k)
    , mExpertHiddenSize(expert_hidden_size)
    , mExpertInterSize(expert_inter_size)
    , mActivationType(activation_type)
    , mType(type)
    , mWeightType(weight_type)
    , mOutputType(output_type)
    , mQuantMode(quant_mode)
    , mUseFinished(use_finished)
    , mUseBias(use_bias)
    , mParallelismConfig(MOEParallelismConfig{tp_size, tp_rank, ep_size, ep_rank})
    , mNormalizationMode(normalization_mode)
    , mSparseMixerEpsilon(sparse_mixer_epsilon)
    , mUseDeterministicKernels(force_determinism)
    , mSideStreamId(side_stream_id)
    , mGemmProfiler(std::move(gemm_profiler_ptr))
    , mUseLora(use_lora)
    , mLoraType(lora_type)
    , mMaxLowRank(max_low_rank)
    , mRemoveInputPadding(remove_input_padding)
    , mLoraProfiler(std::move(lora_profiler))
{
    init();
}

tensorrt_llm::plugins::MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(MixtureOfExpertsPlugin const& other)
    : mMOERunner()
    , mNumExperts(other.mNumExperts)
    , mK(other.mK)
    , mExpertHiddenSize(other.mExpertHiddenSize)
    , mExpertInterSize(other.mExpertInterSize)
    , mActivationType(other.mActivationType)
    , mType(other.mType)
    , mWeightType(other.mWeightType)
    , mOutputType(other.mOutputType)
    , mQuantMode(other.mQuantMode)
    , mUseFinished(other.mUseFinished)
    , mUseBias(other.mUseBias)
    , mParallelismConfig(other.mParallelismConfig)
    , mNormalizationMode(other.mNormalizationMode)
    , mSparseMixerEpsilon(other.mSparseMixerEpsilon)
    , mDims(other.mDims)
    , mUseDeterministicKernels(other.mUseDeterministicKernels)
    , mSideStreamId(other.mSideStreamId)
    , mGemmId1(other.mGemmId1)
    , mGemmId2(other.mGemmId2)
    , mGemmProfiler(other.mGemmProfiler)
    , mUseLora(other.mUseLora)
    , mLoraType(other.mLoraType)
    , mMaxLowRank(other.mMaxLowRank)
    , mRemoveInputPadding(other.mRemoveInputPadding)
    , mLoraImpl1(other.mLoraImpl1)
    , mLoraImpl2(other.mLoraImpl2)
    , mLoraGemmId1(other.mLoraGemmId1)
    , mLoraGemmId2(other.mLoraGemmId2)
    , mLoraProfiler(other.mLoraProfiler)
    , mLayerName(other.mLayerName)
    , mNamespace(other.mNamespace)
{
    init();
}

size_t MixtureOfExpertsPlugin::getSerializationSize() const noexcept
{
    size_t size = sizeof(mRemoveInputPadding) + sizeof(mNumExperts) + sizeof(mK) + sizeof(mExpertHiddenSize)
        + sizeof(mExpertInterSize) + sizeof(mActivationType) + sizeof(mType) + sizeof(mWeightType) + sizeof(mOutputType)
        + sizeof(QuantMode::BaseType) + sizeof(mUseFinished) + sizeof(mUseBias) + sizeof(mParallelismConfig)
        + sizeof(mNormalizationMode) + sizeof(mSparseMixerEpsilon) + sizeof(mDims) + sizeof(mUseDeterministicKernels)
        + sizeof(mSideStreamId) + mGemmProfiler->getSerializationSize(mGemmId1)
        + mGemmProfiler->getSerializationSize(mGemmId2) + sizeof(mUseLora) + sizeof(mLoraType) + sizeof(mMaxLowRank);

    if (hasLora())
    {
        size += mLoraProfiler->getSerializationSize(mLoraGemmId1);
        size += mLoraProfiler->getSerializationSize(mLoraGemmId2);
    }

    return size;
}

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(void const* data, size_t length,
    MixtureOfExpertsPluginProfilerPtr gemm_profiler_ptr, LoraPluginProfilerPtr lora_profiler)
    : mGemmProfiler(gemm_profiler_ptr)
    , mLoraProfiler(lora_profiler)
{
    char const* d = reinterpret_cast<char const*>(data);
    char const* a = d;
    read(d, mRemoveInputPadding);
    read(d, mNumExperts);
    read(d, mK);
    read(d, mExpertHiddenSize);
    read(d, mExpertInterSize);
    read(d, mActivationType);
    read(d, mType);
    read(d, mWeightType);
    read(d, mOutputType);
    QuantMode::BaseType quant_mode;
    read(d, quant_mode);
    mQuantMode = QuantMode{quant_mode};
    read(d, mUseFinished);
    read(d, mUseBias);
    read(d, mParallelismConfig);
    read(d, mNormalizationMode);
    read(d, mSparseMixerEpsilon);
    read(d, mDims);
    read(d, mUseDeterministicKernels);
    read(d, mSideStreamId);
    read(d, mUseLora);
    read(d, mLoraType);
    read(d, mMaxLowRank);

    // Call init before deserialising the profiler to initialize mGemmId
    init();
    mGemmProfiler->deserialize(d, mDims, mGemmId1);
    mGemmProfiler->deserialize(d, mDims, mGemmId2);

    if (hasLora())
    {
        mLoraProfiler->deserialize(d, mDims, mLoraGemmId1);
        mLoraProfiler->deserialize(d, mDims, mLoraGemmId2);
    }

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void MixtureOfExpertsPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    char* a = d;

    write(d, mRemoveInputPadding);
    write(d, mNumExperts);
    write(d, mK);
    write(d, mExpertHiddenSize);
    write(d, mExpertInterSize);
    write(d, mActivationType);
    write(d, mType);
    write(d, mWeightType);
    write(d, mOutputType);
    write(d, mQuantMode.value());
    write(d, mUseFinished);
    write(d, mUseBias);
    write(d, mParallelismConfig);
    write(d, mNormalizationMode);
    write(d, mSparseMixerEpsilon);
    write(d, mDims);
    write(d, mUseDeterministicKernels);
    write(d, mSideStreamId);
    write(d, mUseLora);
    write(d, mLoraType);
    write(d, mMaxLowRank);

    mGemmProfiler->serialize(d, mGemmId1);
    mGemmProfiler->serialize(d, mGemmId2);

    if (hasLora())
    {
        mLoraProfiler->serialize(d, mLoraGemmId1);
        mLoraProfiler->serialize(d, mLoraGemmId2);
    }

    TLLM_CHECK(d == a + getSerializationSize());
}

void MixtureOfExpertsPlugin::init()
{
    TLLM_CHECK_WITH_INFO(mType == DataType::kFP8 || mType == DataType::kFP4 || mOutputType == mType,
        "MOE plugin only supports a different output type for FP4/FP8");
    TLLM_CHECK_WITH_INFO(mType != DataType::kFP8 || tensorrt_llm::common::getSMVersion() >= 89,
        "MoE FP8 is not supported for architectures less than SM89");
    TLLM_CHECK_WITH_INFO(mType != DataType::kFP4 || tensorrt_llm::common::getSMVersion() >= 100,
        "MoE FP4 is not supported for architectures less than SM100");

    TLLM_CHECK_WITH_INFO(!hasLora() || mLoraType == mOutputType, "The LoraType need to keep same with moe OutputType.");

    if (mWeightType == nvinfer1::DataType::kINT8 && mQuantMode.hasInt4Weights())
    {
        mWeightType = DataType::kINT4;
    }

    if (mType == DataType::kHALF && mWeightType == DataType::kHALF)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<half, half>>();
    }
    else if (mType == DataType::kFLOAT && mWeightType == DataType::kFLOAT)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<float, float>>();
    }
    else if (mType == DataType::kHALF && mWeightType == DataType::kINT8)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<half, uint8_t>>();
    }
    else if (mType == DataType::kHALF && mWeightType == DataType::kINT4)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16 && mWeightType == DataType::kBF16)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
    else if (mType == DataType::kBF16 && mWeightType == DataType::kINT8)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t>>();
    }
    else if (mType == DataType::kBF16 && mWeightType == DataType::kINT4)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
    }
#endif

    // Templated lambda for picking the right output type for fp8/fp4
    auto switch_output_type = [&](auto&& argType)
    {
        using Type = std::decay_t<decltype(argType)>;
        switch (mOutputType)
        {
        case nvinfer1::DataType::kFP4: // INT64 == FP4
        case nvinfer1::DataType::kFP8:
            // TODO We need an atomic FP8 reduction for the finalize fusions
            TLLM_THROW("Outputting %d directly is not currently supported", static_cast<int>(mOutputType));
            // mMOERunner = std::make_unique<CutlassMoeFCRunner<Type, Type>>();
            break;
        case nvinfer1::DataType::kHALF:
            mMOERunner = std::make_unique<CutlassMoeFCRunner<Type, Type, half, half>>();
            break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            mMOERunner = std::make_unique<CutlassMoeFCRunner<Type, Type, __nv_bfloat16, __nv_bfloat16>>();
            break;
#endif
        default:
            TLLM_THROW(
                "Invalid output type %d specified for %d", static_cast<int>(mOutputType), static_cast<int>(mType));
        }
    };
#ifdef ENABLE_FP8
    if (mType == DataType::kFP8 && mWeightType == DataType::kFP8)
    {
        switch_output_type(__nv_fp8_e4m3{});
    }
#endif
#ifdef ENABLE_FP4
    if (mType == DataType::kFP4 && mWeightType == DataType::kFP4)
    {
        switch_output_type(__nv_fp4_e2m1{});
    }
#endif

    if (!mMOERunner)
    {
        TLLM_THROW(
            "Could not construct the mixture of experts plugin with the requested input combination Activation: %d "
            "Weight: %d Output: %d",
            static_cast<int>(mType), static_cast<int>(mWeightType), static_cast<int>(mOutputType));
    }

    mMOERunner->use_deterministic_hopper_reduce_ = mK > 2 && mUseDeterministicKernels;

    mGemmId1 = GemmIDMoe{1, mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode, mMOERunner->use_deterministic_hopper_reduce_};
    mGemmId2 = GemmIDMoe{2, mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode, mMOERunner->use_deterministic_hopper_reduce_};
    mGemmProfiler->setMaxProfileM(16384 * mNumExperts / mK);

    if (hasLora())
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        auto cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
        mLoraGemmId1 = GemmIdCublas(mExpertInterSize, mExpertHiddenSize, mLoraType, false, true, mLoraType);
        mLoraGemmId2 = GemmIdCublas(mExpertHiddenSize, mExpertInterSize, mLoraType, false, true, mLoraType);
        std::vector<int> loraOutSizes1 = {static_cast<int>(mExpertInterSize)};
        mLoraImpl1 = std::make_shared<LoraImpl>(
            mExpertHiddenSize, loraOutSizes1, false, true, 1, mLoraType, mMaxLowRank, cublasWrapper);
        std::vector<int> loraOutSizes2 = {static_cast<int>(mExpertHiddenSize)};
        mLoraImpl2 = std::make_shared<LoraImpl>(
            mExpertInterSize, loraOutSizes2, false, true, 1, mLoraType, mMaxLowRank, cublasWrapper);

        TLLM_CUDA_CHECK(cudaEventCreate(&mMemcpyEvent));
    }
    mSideStreamPtr = nullptr;
    mDebugStallMain = tensorrt_llm::runtime::utils::stallStream("TLLM_DEBUG_MOE_STALL_MAIN");
    mDebugStallSide = tensorrt_llm::runtime::utils::stallStream("TLLM_DEBUG_MOE_STALL_SIDE");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* MixtureOfExpertsPlugin::clone() const noexcept
{
    auto* plugin = new MixtureOfExpertsPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs MixtureOfExpertsPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == getOutputTensorIndex() || outputIndex == getOutputDummyTensorIndex());
    return inputs[getInputTensorIndex()];
}

bool MixtureOfExpertsPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < getNbInputs() + getNbOutputs());
    TLLM_CHECK_WITH_INFO(
        nbInputs == getNbInputs(), "Required input to plugin is missing. Expected %d Got %d", getNbInputs(), nbInputs);
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing. Expected %d Got %d",
        getNbOutputs(), nbOutputs);

    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (pos == getExpertWeights1Index() || pos == getExpertWeights2Index())
    {
        auto normalized_weight_type
            = mWeightType == nvinfer1::DataType::kINT4 ? nvinfer1::DataType::kINT8 : mWeightType;
        return inOut[pos].type == normalized_weight_type;
    }
    else if (pos == getFinishedTensorIndex() && hasFinishedTensor())
    {
        return inOut[pos].type == DataType::kBOOL;
    }
    else if (pos == getRoutingTensorIndex())
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    else if (pos == getExpertBias1Index() || pos == getExpertBias2Index())
    {
        return inOut[pos].type == mOutputType;
    }
    else if (pos == nbInputs + getOutputTensorIndex())
    {
        return inOut[pos].type == mOutputType;
    }
    else if (useSideStream() && pos == nbInputs + getOutputDummyTensorIndex())
    {
        return inOut[pos].type == inOut[getInputDummyTensorIndex()].type;
    }
    else if (useSideStream() && pos == getInputDummyTensorIndex())
    {
        return true;
    }
    else if (hasExpertFp8QuantScales() && getExpertFP8Dequant1Index() <= pos && pos <= getExpertFP8QuantFinalIndex())
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    else if (hasExpertIntQuantScales() && getExpertIntQuantScale1Index() <= pos
        && pos <= getExpertIntQuantScale2Index())
    {
        return inOut[pos].type == mOutputType;
    }
    else if (hasFP4QuantScales() && getFP4GlobalActSF1Index() <= pos && pos <= getFP4GlobalSF2Index())
    {
        if (pos == getFP4WeightSF1Index() || pos == getFP4WeightSF2Index())
            return inOut[pos].type == nvinfer1::DataType::kFP8;
        else
            return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }
    else if (hasLora() && hasExpertFp8QuantScales() && pos == getInputFP8DequantIndex())
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }
    else if (hasLora() && pos == getHostRequestTypeIndex())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (hasLora() && (pos == getLoraFC1RanksIndex() || pos == getLoraFC2RanksIndex()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (hasGatedLoraWeightsAndRanks() && pos == getLoraGatedRanksIndex())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (hasLora() && (pos == getLoraFC1WeightPtrsIndex() || pos == getLoraFC2WeightPtrsIndex()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else if (hasGatedLoraWeightsAndRanks() && pos == getLoraGatedWeightPtrsIndex())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else if (hasLora() && mRemoveInputPadding && pos == getHostContextLengthIndex())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (hasFP4QuantScales() && pos == getInputTensorIndex())
    {
        return inOut[pos].type == mOutputType;
    }
    else
    {
        return inOut[pos].type == mType;
    }

    return false;
}

void MixtureOfExpertsPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    TLLM_CHECK_WITH_INFO(
        nbInputs == getNbInputs(), "Required input to plugin is missing. Expected %d Got %d", getNbInputs(), nbInputs);
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing. Expected %d Got %d",
        getNbOutputs(), nbOutputs);

    auto in_tensor = in[getInputTensorIndex()];

    auto const minM
        = std::accumulate(in_tensor.min.d, in_tensor.min.d + in_tensor.min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM
        = std::accumulate(in_tensor.max.d, in_tensor.max.d + in_tensor.max.nbDims - 1, 1, std::multiplies<int>());

    auto weights_1 = in[getExpertWeights1Index()];
    auto weights_2 = in[getExpertWeights2Index()];
    int inner_dim_idx = getGemmShapeInnerDimIndex();
    int const maxK = weights_1.max.d[inner_dim_idx];
    int const maxN = weights_2.max.d[inner_dim_idx];
    int const minK = weights_1.min.d[inner_dim_idx];
    int const minN = weights_2.min.d[inner_dim_idx];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");
    TLLM_CHECK_WITH_INFO(maxK == mExpertHiddenSize && maxN == mExpertInterSize,
        "Configured tensor sizes %dx%d does not match constructor param size %ldx%ld", maxK, maxN, mExpertHiddenSize,
        mExpertInterSize);

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId1 = GemmIDMoe{1, mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode};
    mGemmId2 = GemmIDMoe{2, mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode};

    if (hasLora())
    {
        auto const N = utils::computeNDimension(true, in[getHostRequestTypeIndex()].max);
        mLoraGemmId1 = GemmIdCublas(N, mExpertHiddenSize, mLoraType, false, true, mLoraType);
        mLoraGemmId2 = GemmIdCublas(N, mExpertInterSize, mLoraType, false, true, mLoraType);
    }
}

auto MixtureOfExpertsPlugin::setupWorkspace(void* base_ptr, int64_t num_tokens, int num_reqs) const -> WorkspaceInfo
{
    size_t moe_workspace_size
        = mMOERunner->getWorkspaceSize(num_tokens, mExpertHiddenSize, mExpertInterSize, mNumExperts, mK,
            mActivationType, mNormalizationMode, mParallelismConfig, hasLora(), /*use_fp8_block_scaling=*/false);

    // Output of post-softmax routing probabilities
    size_t scale_probabilities_size = num_tokens * mNumExperts * sizeof(float);

    // Permutation map
    size_t src_to_dest_map_size = mK * num_tokens * sizeof(int);

    // Selected expert map
    size_t selected_expert_size = mK * num_tokens * sizeof(int);

    size_t lora_workspace_size = 0;
    if (hasLora())
    {
        int64_t num_reqs_lora = std::min(num_tokens * mK, static_cast<int64_t>(num_reqs * mNumExperts));
        lora_workspace_size = std::max(mLoraImpl1->getWorkspaceSize(num_tokens * mK, num_reqs_lora, mLoraType),
            mLoraImpl2->getWorkspaceSize(num_tokens * mK, num_reqs_lora, mLoraType));
    }

    std::vector<size_t> workspaces{
        moe_workspace_size,
        scale_probabilities_size,
        src_to_dest_map_size,
        selected_expert_size,
        lora_workspace_size,
    };

    WorkspaceInfo info{};
    info.size = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    if (base_ptr)
    {
        info.workspace = base_ptr;
        info.scale_probs = nextWorkspacePtr((int8_t*) info.workspace, moe_workspace_size);
        info.src_to_dest_map = nextWorkspacePtr((int8_t*) info.scale_probs, scale_probabilities_size);
        info.selected_experts = nextWorkspacePtr((int8_t*) info.src_to_dest_map, src_to_dest_map_size);
        info.lora_workspace = nextWorkspacePtr((int8_t*) info.selected_experts, selected_expert_size);
    }

    return info;
}

int64_t MixtureOfExpertsPlugin::getNumTokens(nvinfer1::PluginTensorDesc const* input_tensors) const
{
    int ndim = input_tensors[getInputTensorIndex()].dims.nbDims;
    TLLM_CHECK_WITH_INFO(
        3 == ndim || 2 == ndim, "hidden_state dimension should be either 2 [b*s, hidden], or 3 [b, s, hidden]");
    int64_t num_tokens = input_tensors[getInputTensorIndex()].dims.d[0];
    if (ndim == 3)
    {
        num_tokens *= input_tensors[getInputTensorIndex()].dims.d[1];
    }
    return num_tokens;
}

size_t MixtureOfExpertsPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    TLLM_CHECK_WITH_INFO(
        nbInputs == getNbInputs(), "Required input to plugin is missing. Expected %d Got %d", getNbInputs(), nbInputs);
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing. Expected %d Got %d",
        getNbOutputs(), nbOutputs);

    if (useSideStream())
    {
        return 0;
    }
    int const num_tokens = getNumTokens(inputs);
    int const num_lora_reqs = getNumLoraRequests(inputs);
    return setupWorkspace(nullptr, num_tokens, num_lora_reqs).size;
}

MOEParallelismConfig MixtureOfExpertsPlugin::getParallelismConfig() const
{
    return mParallelismConfig;
}

QuantParams tensorrt_llm::plugins::MixtureOfExpertsPlugin::getQuantParams(nvinfer1::PluginTensorDesc const* inputDesc,
    void const* const* inputs, int scale_1_idx, int scale_2_idx, int scale_3_idx, int scale_4_idx, int scale_5_idx,
    int scale_6_idx) const
{
    void const* scale_1 = scale_1_idx >= 0 ? inputs[scale_1_idx] : nullptr;
    void const* scale_2 = scale_2_idx >= 0 ? inputs[scale_2_idx] : nullptr;
    void const* scale_3 = scale_3_idx >= 0 ? inputs[scale_3_idx] : nullptr;
    void const* scale_4 = scale_4_idx >= 0 ? inputs[scale_4_idx] : nullptr;
    void const* scale_5 = scale_5_idx >= 0 ? inputs[scale_5_idx] : nullptr;
    void const* scale_6 = scale_6_idx >= 0 ? inputs[scale_6_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_1 = scale_1_idx >= 0 ? &inputDesc[scale_1_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_2 = scale_2_idx >= 0 ? &inputDesc[scale_2_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_3 = scale_3_idx >= 0 ? &inputDesc[scale_3_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_4 = scale_4_idx >= 0 ? &inputDesc[scale_4_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_5 = scale_5_idx >= 0 ? &inputDesc[scale_5_idx] : nullptr;
    nvinfer1::PluginTensorDesc const* desc_6 = scale_6_idx >= 0 ? &inputDesc[scale_6_idx] : nullptr;
    auto const gated_inter_size = isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;
    auto const experts_per_node = mNumExperts / mParallelismConfig.ep_size;
    if (hasExpertIntQuantScales())
    {
        TLLM_CHECK(scale_1 && scale_2);
        TLLM_CHECK(!scale_3 && !scale_4 && !scale_5 && !scale_6);
        TLLM_CHECK(desc_1->dims.nbDims == 2);
        TLLM_CHECK(desc_2->dims.nbDims == 2);
        TLLM_CHECK_WITH_INFO(desc_1->dims.d[0] == experts_per_node, "Incorrect number of experts in int quant scale");
        TLLM_CHECK(desc_1->dims.d[1] == gated_inter_size);
        TLLM_CHECK_WITH_INFO(desc_2->dims.d[0] == experts_per_node, "Incorrect number of experts in int quant scale");
        TLLM_CHECK(desc_2->dims.d[1] == mExpertHiddenSize);
        return QuantParams::Int(scale_1, scale_2);
    }
    else if (hasExpertFp8QuantScales())
    {
        TLLM_CHECK(scale_1 && scale_2 && scale_3);
        TLLM_CHECK(scale_4 || !hasExpertFp8FinalQuantScales());
        TLLM_CHECK((scale_5 != nullptr) == hasLora());
        TLLM_CHECK(!scale_6);
        TLLM_CHECK(desc_1->dims.nbDims == 2);
        TLLM_CHECK(desc_2->dims.nbDims == 1);
        TLLM_CHECK(desc_3->dims.nbDims == 2);
        TLLM_CHECK_WITH_INFO(
            desc_1->dims.d[0] == experts_per_node && desc_1->dims.d[1] == 1, "Incorrect shape for weight FP8 scale");
        TLLM_CHECK(desc_2->dims.d[0] == 1);
        TLLM_CHECK_WITH_INFO(
            desc_3->dims.d[0] == experts_per_node && desc_3->dims.d[1] == 1, "Incorrect shape for weight FP8 scale");
        return QuantParams::FP8(static_cast<float const*>(scale_1), static_cast<float const*>(scale_2),
            static_cast<float const*>(scale_3), static_cast<float const*>(scale_4), static_cast<float const*>(scale_5));
    }
    else if (hasFP4QuantScales())
    {
        TLLM_CHECK(scale_1 && scale_2 && scale_3 && scale_4 && scale_5 && scale_6);
        TLLM_CHECK(desc_1->dims.nbDims == 1);
        TLLM_CHECK(desc_2->dims.nbDims == 3);
        TLLM_CHECK(desc_3->dims.nbDims == 1);
        TLLM_CHECK(desc_4->dims.nbDims == 1);
        TLLM_CHECK(desc_5->dims.nbDims == 3);
        TLLM_CHECK(desc_6->dims.nbDims == 1);
        TLLM_CHECK(desc_1->dims.d[0] == 1);
        TLLM_CHECK_WITH_INFO(desc_2->dims.d[0] == experts_per_node && desc_2->dims.d[1] == gated_inter_size
                && desc_2->dims.d[2] == mExpertHiddenSize / TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize,
            "Incorrect shape for FP4 scale");
        TLLM_CHECK_WITH_INFO(desc_3->dims.d[0] == experts_per_node, "Incorrect shape for FP4 scale");
        TLLM_CHECK(desc_4->dims.d[0] == 1);
        TLLM_CHECK_WITH_INFO(desc_5->dims.d[0] == experts_per_node && desc_5->dims.d[1] == mExpertHiddenSize
                && desc_5->dims.d[2] == mExpertInterSize / TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize,
            "Incorrect shape for FP4 scale");
        TLLM_CHECK_WITH_INFO(desc_6->dims.d[0] == experts_per_node, "Incorrect shape for FP4 scale");
        return QuantParams::FP4(static_cast<float const*>(scale_1),
            static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(scale_2),
            static_cast<float const*>(scale_3), static_cast<float const*>(scale_4),
            static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(scale_5),
            static_cast<float const*>(scale_6));
    }
    return {};
}

int MixtureOfExpertsPlugin::getNumLoraRequests(nvinfer1::PluginTensorDesc const* input_tensors) const
{
    if (!hasLora())
        return 0;
    int num_reqs = input_tensors[getLoraFC1RanksIndex()].dims.d[0];
    return num_reqs;
}

LoraParams MixtureOfExpertsPlugin::getLoraParams(
    nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs, void* workspace)
{
    TLLM_CHECK(hasLora());

    int const num_reqs = getNumLoraRequests(inputDesc);
    int64_t const num_tokens = getNumTokens(inputDesc);
    bool is_gated_actiation = isGatedActivation(mActivationType);

    mLoraExpandFC1WeightPtrs.clear();
    mLoraExpandFC2WeightPtrs.clear();
    mLoraExpandFC1Ranks.clear();
    mLoraExpandFC2Ranks.clear();

    mLoraExpandFC1WeightPtrs.reserve(num_tokens * 2);
    mLoraExpandFC2WeightPtrs.reserve(num_tokens * 2);
    mLoraExpandFC1Ranks.reserve(num_tokens);
    mLoraExpandFC2Ranks.reserve(num_tokens);

    if (is_gated_actiation)
    {
        mLoraExpandGatedWeightPtrs.clear();
        mLoraExpandGatedRanks.clear();
        mLoraExpandGatedWeightPtrs.reserve(num_tokens * 2);
        mLoraExpandGatedRanks.reserve(num_tokens);
    }

    int const seq_len = mRemoveInputPadding ? 0 : inputDesc[getInputTensorIndex()].dims.d[1];
    int32_t const* req_types = static_cast<int32_t const*>(inputs[getHostRequestTypeIndex()]);
    int32_t const* host_context_lens
        = mRemoveInputPadding ? static_cast<int32_t const*>(inputs[getHostContextLengthIndex()]) : nullptr;

    auto const fc1_lora_weight_ptrs = static_cast<void const* const*>(inputs[getLoraFC1WeightPtrsIndex()]);
    auto const fc1_lora_ranks = static_cast<int32_t const*>(inputs[getLoraFC1RanksIndex()]);

    auto const fc2_lora_weight_ptrs = static_cast<void const* const*>(inputs[getLoraFC2WeightPtrsIndex()]);
    auto const fc2_lora_ranks = static_cast<int32_t const*>(inputs[getLoraFC2RanksIndex()]);

    auto const gated_lora_weight_ptrs
        = is_gated_actiation ? static_cast<void const* const*>(inputs[getLoraGatedWeightPtrsIndex()]) : nullptr;
    auto const gated_lora_ranks
        = is_gated_actiation ? static_cast<int32_t const*>(inputs[getLoraGatedRanksIndex()]) : nullptr;

    int idx = 0;
    for (int req_id = 0; req_id < num_reqs; req_id++)
    {
        RequestType const reqType = static_cast<RequestType const>(req_types[req_id]);
        if (reqType == RequestType::kGENERATION)
        {
            // lora_weight_ptrs has 3 pointers for each module: A,B, and an optional DoRA magnitude
            // the current LoRA implementation does not apply DoRA scaling, so the magnitude is ignored
            mLoraExpandFC1WeightPtrs.push_back(fc1_lora_weight_ptrs[req_id * 3]);
            mLoraExpandFC1WeightPtrs.push_back(fc1_lora_weight_ptrs[req_id * 3 + 1]);
            mLoraExpandFC1Ranks.push_back(fc1_lora_ranks[req_id]);

            mLoraExpandFC2WeightPtrs.push_back(fc2_lora_weight_ptrs[req_id * 3]);
            mLoraExpandFC2WeightPtrs.push_back(fc2_lora_weight_ptrs[req_id * 3 + 1]);
            mLoraExpandFC2Ranks.push_back(fc2_lora_ranks[req_id]);

            if (is_gated_actiation)
            {
                mLoraExpandGatedWeightPtrs.push_back(gated_lora_weight_ptrs[req_id * 3]);
                mLoraExpandGatedWeightPtrs.push_back(gated_lora_weight_ptrs[req_id * 3 + 1]);
                mLoraExpandGatedRanks.push_back(gated_lora_ranks[req_id]);
            }

            idx += 1;
        }
        else
        {
            int context_len = (mRemoveInputPadding ? host_context_lens[req_id] : seq_len);

            for (int context_id = 0; context_id < context_len; context_id++)
            {
                mLoraExpandFC1WeightPtrs.push_back(fc1_lora_weight_ptrs[req_id * 3]);
                mLoraExpandFC1WeightPtrs.push_back(fc1_lora_weight_ptrs[req_id * 3 + 1]);
                mLoraExpandFC1Ranks.push_back(fc1_lora_ranks[req_id]);

                mLoraExpandFC2WeightPtrs.push_back(fc2_lora_weight_ptrs[req_id * 3]);
                mLoraExpandFC2WeightPtrs.push_back(fc2_lora_weight_ptrs[req_id * 3 + 1]);
                mLoraExpandFC2Ranks.push_back(fc2_lora_ranks[req_id]);

                if (is_gated_actiation)
                {
                    mLoraExpandGatedWeightPtrs.push_back(gated_lora_weight_ptrs[req_id * 3]);
                    mLoraExpandGatedWeightPtrs.push_back(gated_lora_weight_ptrs[req_id * 3 + 1]);
                    mLoraExpandGatedRanks.push_back(gated_lora_ranks[req_id]);
                }
            }
            idx += context_len;
        }
    }

    TLLM_CHECK_WITH_INFO(idx == num_tokens, fmtstr("idx %d num_tokens %ld", idx, num_tokens));

    return LoraParams(num_reqs, mLoraExpandFC1Ranks.data(), mLoraExpandFC1WeightPtrs.data(), mLoraExpandFC2Ranks.data(),
        mLoraExpandFC2WeightPtrs.data(), mLoraImpl1, mLoraImpl2, workspace, &mMemcpyEvent, mLoraExpandGatedRanks.data(),
        mLoraExpandGatedWeightPtrs.data());
}

int MixtureOfExpertsPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace_ptr,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    int64_t const num_tokens = getNumTokens(inputDesc);
    int64_t const num_reqs = getNumLoraRequests(inputDesc);
    int64_t const num_not_finished = num_tokens; // TODO Take this as an input

    if (useSideStream())
    {
        // Prepare the side stream
        if (!mSideStreamPtr)
        {
            auto const resource_name = nvinfer1::pluginInternal::SideStream::getResourceKey(mSideStreamId);
            nvinfer1::pluginInternal::SideStream side_stream{};
            mSideStreamPtr = reinterpret_cast<nvinfer1::pluginInternal::SideStream*>(
                getPluginRegistry()->acquirePluginResource(resource_name.c_str(), &side_stream));
        }
        // Debug the code with the main stream stalled (only executed when the environment variable
        // TLLM_DEBUG_MOE_STALL_MAIN is set and has a positive value)
        mSideStreamPtr->stallMainStream("TLLM_DEBUG_MOE_STALL_MAIN", stream, mDebugStallMain);
        // The side stream waits for the inputs managed by the main stream to be ready
        mSideStreamPtr->waitMainStreamOnSideStream(stream);
        // Provide data dependency for the shared experts running after this plugin by copying inputs on the main stream
        size_t count = 1;
        for (int i = 0; i < inputDesc[getInputDummyTensorIndex()].dims.nbDims; ++i)
        {
            count *= inputDesc[getInputDummyTensorIndex()].dims.d[i];
        }
        count *= tensorrt_llm::runtime::BufferDataType(inputDesc[getInputDummyTensorIndex()].type).getSize();
        TLLM_CUDA_CHECK(cudaMemcpyAsync(outputs[getOutputDummyTensorIndex()], inputs[getInputDummyTensorIndex()], count,
            cudaMemcpyDeviceToDevice, stream));
        // Switch from the main stream to the side stream
        stream = mSideStreamPtr->getStream();
        // The workspace is managed by the side stream (otherwise, the lifetime of workspace may be incorrect)
        auto const workspace_size = setupWorkspace(nullptr, num_tokens, num_reqs).size;
        workspace_ptr = mSideStreamPtr->getWorkspacePtr(workspace_size);
    }
    auto workspace = setupWorkspace(workspace_ptr, num_tokens, num_reqs);

    auto w1_desc = inputDesc[getExpertWeights1Index()];
    auto w2_desc = inputDesc[getExpertWeights2Index()];
    TLLM_CHECK(w1_desc.dims.nbDims == 3);
    auto const experts_per_node = mNumExperts / mParallelismConfig.ep_size;
    TLLM_CHECK(w1_desc.dims.d[0] == experts_per_node);
    TLLM_CHECK(w2_desc.dims.nbDims == 3);
    TLLM_CHECK(w2_desc.dims.d[0] == experts_per_node);

    auto [inner_packed_elements, outer_packed_elements] = getWeightPackedElements();
    int inner_dim_idx = getGemmShapeInnerDimIndex();
    int outer_dim_idx = getGemmShapeOuterDimIndex();
    TLLM_CHECK(w1_desc.dims.d[inner_dim_idx] * inner_packed_elements == mExpertHiddenSize);
    if (isGatedActivation(mActivationType))
    {
        TLLM_CHECK(w1_desc.dims.d[outer_dim_idx] * outer_packed_elements == mExpertInterSize * 2);
    }
    else
    {
        TLLM_CHECK(w1_desc.dims.d[outer_dim_idx] * outer_packed_elements == mExpertInterSize);
    }

    TLLM_CHECK(w2_desc.dims.d[inner_dim_idx] * inner_packed_elements == mExpertInterSize);
    TLLM_CHECK(w2_desc.dims.d[outer_dim_idx] * outer_packed_elements == mExpertHiddenSize);

    QuantParams quant_params{};
    if (hasExpertIntQuantScales())
    {
        quant_params
            = getQuantParams(inputDesc, inputs, getExpertIntQuantScale1Index(), getExpertIntQuantScale2Index());
    }
    else if (hasExpertFp8QuantScales())
    {
        quant_params = getQuantParams(inputDesc, inputs, //
            getExpertFP8Dequant1Index(),                 //
            getExpertFP8Quant2Index(),                   //
            getExpertFP8Dequant2Index(),                 //
            hasExpertFp8FinalQuantScales() ? getExpertFP8QuantFinalIndex() : -1,
            hasLora() ? getInputFP8DequantIndex() : -1);
    }
    else if (hasFP4QuantScales())
    {
        quant_params = getQuantParams(inputDesc, inputs, //
            getFP4GlobalActSF1Index(),                   //
            getFP4WeightSF1Index(),                      //
            getFP4GlobalSF1Index(),                      //
            getFP4GlobalActSF2Index(),                   //
            getFP4WeightSF2Index(),                      //
            getFP4GlobalSF2Index()                       //
        );
    }

    LoraParams lora_params{};

    if (hasLora())
    {
        lora_params = getLoraParams(inputDesc, inputs, workspace.lora_workspace);
        auto lora_gemm1 = mLoraProfiler->getBestConfig(num_tokens, mLoraGemmId1);
        auto lora_gemm2 = mLoraProfiler->getBestConfig(num_tokens, mLoraGemmId2);

        mLoraImpl1->setBestTactic(lora_gemm1);
        mLoraImpl2->setBestTactic(lora_gemm2);
    }

    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> gemm1;
    std::optional<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> gemm2;
    if (common::getEnvForceDeterministicMOE())
    {
        gemm1 = mMOERunner->getTactics()[0];
        gemm2 = mMOERunner->getTactics()[0];
    }
    else
    {
        gemm1 = mGemmProfiler->getBestConfig(num_tokens, mGemmId1);
        gemm2 = mGemmProfiler->getBestConfig(num_tokens, mGemmId2);
    }

    mMOERunner->setTactic(gemm1, gemm2);
    BlockScaleParams deepseek_params{};
    mMOERunner->runMoe(inputs[getInputTensorIndex()], static_cast<float const*>(inputs[getRoutingTensorIndex()]),
        inputs[getExpertWeights1Index()], hasBias() ? inputs[getExpertBias1Index()] : nullptr, mActivationType,
        inputs[getExpertWeights2Index()], hasBias() ? inputs[getExpertBias2Index()] : nullptr, quant_params, num_tokens,
        mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, static_cast<char*>(workspace.workspace),
        // Outputs
        outputs[getOutputTensorIndex()],
        hasFinishedTensor() ? static_cast<bool const*>(inputs[getFinishedTensorIndex()]) : nullptr, num_not_finished,
        workspace.scale_probs, static_cast<int*>(workspace.src_to_dest_map),
        static_cast<int*>(workspace.selected_experts), mSparseMixerEpsilon, mParallelismConfig, mNormalizationMode,
        hasLora(), lora_params, /*use_fp8_block_scaling=*/false, deepseek_params, stream);

    if (useSideStream())
    {
        // Debug the code with the side stream stalled (only executed when the environment variable
        // TLLM_DEBUG_MOE_STALL_SIDE is set and has a positive value)
        mSideStreamPtr->stallSideStream("TLLM_DEBUG_MOE_STALL_SIDE", mDebugStallSide);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType MixtureOfExpertsPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == getOutputTensorIndex() || index == getOutputDummyTensorIndex());
    if (useSideStream() && index == getOutputDummyTensorIndex())
    {
        return inputTypes[getInputDummyTensorIndex()];
    }
    return mOutputType;
}

// IPluginV2 Methods
char const* MixtureOfExpertsPlugin::getPluginType() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_NAME;
}

char const* MixtureOfExpertsPlugin::getPluginVersion() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_VERSION;
}

int MixtureOfExpertsPlugin::initialize() noexcept
{
    mGemmProfiler->setGemmToProfile(kernels::GemmProfilerBackend::GemmToProfile::GEMM_1);
    mGemmProfiler->profileTactics(this, mType, mDims, mGemmId1);
    mGemmProfiler->setGemmToProfile(kernels::GemmProfilerBackend::GemmToProfile::GEMM_2);
    mGemmProfiler->profileTactics(this, mType, mDims, mGemmId2);

    if (hasLora())
    {
        mLoraImpl1->setGemmConfig();
        mLoraImpl2->setGemmConfig();

        mLoraProfiler->profileTactics(mLoraImpl1->mCublasWrapper, mType, mDims, mLoraGemmId1);
        mLoraProfiler->profileTactics(mLoraImpl2->mCublasWrapper, mType, mDims, mLoraGemmId2);
    }
    return 0;
}

void MixtureOfExpertsPlugin::terminate() noexcept
{
    if (mSideStreamPtr)
    {
        auto const resource_name = nvinfer1::pluginInternal::SideStream::getResourceKey(mSideStreamId);
        getPluginRegistry()->releasePluginResource(resource_name.c_str());
        mSideStreamPtr = nullptr;
    }
}

void MixtureOfExpertsPlugin::destroy() noexcept
{
    if (hasLora())
    {
        TLLM_CUDA_CHECK(cudaEventDestroy(mMemcpyEvent));
    }
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void MixtureOfExpertsPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* MixtureOfExpertsPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

char const* MixtureOfExpertsPluginCreator::getPluginName() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_NAME;
}

char const* MixtureOfExpertsPluginCreator::getPluginVersion() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* MixtureOfExpertsPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

MixtureOfExpertsPluginCreator::MixtureOfExpertsPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("remove_input_padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("number_of_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("expert_hidden_size", nullptr, PluginFieldType::kINT32, 128));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("expert_inter_size", nullptr, PluginFieldType::kINT32, 128 * 4));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "activation_type", nullptr, PluginFieldType::kINT32, static_cast<int>(tensorrt_llm::ActivationType::Identity)));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("type_id", nullptr, PluginFieldType::kINT32, static_cast<int>(DataType::kHALF)));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("weight_type_id", nullptr, PluginFieldType::kINT32, static_cast<int>(DataType::kHALF)));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("quant_mode", nullptr, PluginFieldType::kINT32, static_cast<int>(DataType::kHALF)));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("use_finished", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("use_bias", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("tp_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("tp_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("ep_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("ep_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("normalization_mode", nullptr, PluginFieldType::kINT32,
        static_cast<int>(MOEExpertScaleNormalizationMode::NONE)));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("sparse_mixer_epsilon", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("side_stream_id", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("use_lora", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("lora_type_id", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_low_rank", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2* MixtureOfExpertsPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    nvinfer1::PluginField const* fields = fc->fields;
    int mRemoveInputPadding{};
    int mNumExperts{};
    int mK{};
    int mExpertHiddenSize{};
    int mExpertInterSize{};
    int mActivationType{};
    int mType{};
    int mWeightType{};
    int mOutputType{INT_MAX};
    int mQuantMode{};
    int mUseFinished{0};
    int mUseBias{0};
    int mTPSize{};
    int mTPRank{};
    int mEPSize{};
    int mEPRank{};
    int mNormalizationMode{};
    int mRequiresDeterminism{0};
    int mSideStreamId{0};
    int mUseLora{};
    int mLoraType{INT_MAX};
    int mMaxLowRank{0};

    float mSparseMixerEpsilon = -INFINITY;

    // Read configurations from each fields
    struct MapPair
    {
        char const* key;
        int& field;
        bool optional = false;
        bool set = false;
    };

    std::array input_map{
        MapPair{"remove_input_padding", std::ref(mRemoveInputPadding)},
        MapPair{"number_of_experts", std::ref(mNumExperts)},
        MapPair{"top_k", std::ref(mK)},
        MapPair{"expert_hidden_size", std::ref(mExpertHiddenSize)},
        MapPair{"expert_inter_size", std::ref(mExpertInterSize)},
        MapPair{"activation_type", std::ref(mActivationType)},
        MapPair{"type_id", std::ref(mType)},
        MapPair{"weight_type_id", std::ref(mWeightType)},
        MapPair{"quant_mode", std::ref(mQuantMode)},
        MapPair{"tp_size", std::ref(mTPSize)},
        MapPair{"tp_rank", std::ref(mTPRank)},
        MapPair{"ep_size", std::ref(mEPSize)},
        MapPair{"ep_rank", std::ref(mEPRank)},
        MapPair{"normalization_mode", std::ref(mNormalizationMode)},
        MapPair{"use_lora", std::ref(mUseLora)},

        // Optional
        MapPair{"use_finished", std::ref(mUseFinished), true},
        MapPair{"use_bias", std::ref(mUseBias), true},
        MapPair{"output_type_id", std::ref(mOutputType), true},
        MapPair{"force_determinism", std::ref(mRequiresDeterminism), true},
        MapPair{"side_stream_id", std::ref(mSideStreamId), true},
        MapPair{"lora_type_id", std::ref(mLoraType), true},
        MapPair{"max_low_rank", std::ref(mMaxLowRank), true},
    };
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        for (auto& item : input_map)
        {
            if (!strcmp(item.key, attrName))
            {
                TLLM_CHECK(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                TLLM_CHECK_WITH_INFO(!item.set, "Parameter %s was set twice", item.key);
                item.field = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
                item.set = true;
            }
        }

        // Non integer inputs
        if (!strcmp(attrName, "sparse_mixer_epsilon"))
        {
            TLLM_CHECK(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            mSparseMixerEpsilon = *static_cast<float const*>(fields[i].data);
        }
    }

    for (auto& item : input_map)
    {
        TLLM_CHECK_WITH_INFO(item.set || item.optional, "Parameter %s is required but not set", item.key);
    }

    // Output type is optional, if not set it to the same as mType
    if (mOutputType == INT_MAX)
    {
        mOutputType = mType;
    }

    if (mUseLora)
    {
        TLLM_CHECK_WITH_INFO(mLoraType != INT_MAX && mMaxLowRank != 0,
            "MoE fuse lora, lora_type_id and max_low_rank are required but not set");
    }

    if (static_cast<MOEExpertScaleNormalizationMode>(mNormalizationMode)
        == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        TLLM_CHECK_WITH_INFO(
            mSparseMixerEpsilon > 0, "sparse_mixer_epsilon must be set when normalization mode is SPARSE_MIXER");
    }

    try
    {
        auto gemmProfiler = moePluginProfiler.createGemmPluginProfiler(/* inference */ false);
        auto loraProfiler = loraPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);
        auto* obj = new MixtureOfExpertsPlugin(
            // Constructor parameters
            mRemoveInputPadding, mNumExperts, mK, mExpertHiddenSize, mExpertInterSize,
            static_cast<tensorrt_llm::ActivationType>(mActivationType), static_cast<nvinfer1::DataType>(mType),
            static_cast<nvinfer1::DataType>(mWeightType), static_cast<nvinfer1::DataType>(mOutputType),
            QuantMode(mQuantMode), mUseFinished != 0, mUseBias != 0, mTPSize, mTPRank, mEPSize, mEPRank,
            static_cast<MOEExpertScaleNormalizationMode>(mNormalizationMode), mSparseMixerEpsilon,
            mRequiresDeterminism != 0, mSideStreamId, gemmProfiler, mUseLora != 0,
            static_cast<nvinfer1::DataType>(mLoraType), loraProfiler, mMaxLowRank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* MixtureOfExpertsPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MixtureOfExpertsPlugin::destroy()
    try
    {
        auto gemmProfiler = moePluginProfiler.createGemmPluginProfiler(/* inference */ true);
        auto loraProfiler = loraPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);

        auto* obj = new MixtureOfExpertsPlugin(
            // Constructor parameters
            serialData, serialLength, gemmProfiler, loraProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MixtureOfExpertsPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* MixtureOfExpertsPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void MixtureOfExpertsGemmProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    checkInit();
    size_t bytes = backend.getWorkspaceSize(maxM);
    this->setTmpWorkspaceSizeInBytes(bytes);
}

void MixtureOfExpertsGemmProfiler::runTactic(int m, int n, int k, MixtureOfExpertsGemmProfiler::Config const& tactic,
    char* workspace_ptr_char, cudaStream_t const& stream)
{
    checkInit();
    backend.runProfiler(m, tactic, workspace_ptr_char, stream);
}

auto MixtureOfExpertsGemmProfiler::getTactics(int m, int n, int k) const -> std::vector<Config>
{
    assert(mRunner);
    return mRunner->mMOERunner->getTactics();
}

void MixtureOfExpertsGemmProfiler::initTmpData(
    int m, int n, int k, char* workspace, size_t ws_size, cudaStream_t stream)
{
    checkInit();
    backend.prepare(m, workspace, stream);
}

void MixtureOfExpertsGemmProfiler::checkInit()
{
    assert(mRunner);
    if (init_backend)
    {
        return;
    }
    init_backend = true;
    auto& plugin = *mRunner;
    backend.init(*plugin.mMOERunner, backend.mGemmToProfile, plugin.mType, plugin.mWeightType, plugin.mOutputType,
        plugin.mNumExperts, plugin.mK, plugin.mExpertHiddenSize, plugin.mExpertInterSize, plugin.mActivationType,
        plugin.hasBias(), plugin.hasLora(), plugin.getParallelismConfig());
}
