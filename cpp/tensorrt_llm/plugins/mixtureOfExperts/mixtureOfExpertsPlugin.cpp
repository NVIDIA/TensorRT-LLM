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
#include "tensorrt_llm/common/quantization.h"
#include <numeric>

using namespace nvinfer1;
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

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(int number_of_experts, int top_k, int expert_hidden_size,
    int expert_inter_size, tensorrt_llm::ActivationType activation_type, nvinfer1::DataType type,
    nvinfer1::DataType weight_type, nvinfer1::DataType output_type, QuantMode quant_mode, bool use_finished,
    bool use_bias, int tp_size, int tp_rank, int ep_size, int ep_rank,
    MOEExpertScaleNormalizationMode normalization_mode, MixtureOfExpertsPluginProfilerPtr plugin_profiler_ptr)
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
    , mPluginProfiler(std::move(plugin_profiler_ptr))
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
    , mDims(other.mDims)
    , mGemmId(other.mGemmId)
    , mPluginProfiler(other.mPluginProfiler)
    , mLayerName(other.mLayerName)
    , mNamespace(other.mNamespace)
{
    init();
}

size_t MixtureOfExpertsPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumExperts) + sizeof(mK) + sizeof(mExpertHiddenSize) + sizeof(mExpertInterSize)
        + sizeof(mActivationType) + sizeof(mType) + sizeof(mWeightType) + sizeof(mOutputType)
        + sizeof(QuantMode::BaseType) + sizeof(mUseFinished) + sizeof(mUseBias) + sizeof(mParallelismConfig)
        + sizeof(mNormalizationMode) + sizeof(mDims) + mPluginProfiler->getSerializationSize(mGemmId);
}

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(
    void const* data, size_t length, MixtureOfExpertsPluginProfilerPtr plugin_profiler_ptr)
    : mPluginProfiler(plugin_profiler_ptr)
{
    char const* d = reinterpret_cast<char const*>(data);
    char const* a = d;
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
    read(d, mDims);

    init();
    mPluginProfiler->deserialize(d, mDims, mGemmId);
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
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);

    assert(d == a + getSerializationSize());
}

void MixtureOfExpertsPlugin::init()
{
    TLLM_CHECK_WITH_INFO(
        mType == DataType::kFP8 || mOutputType == mType, "MOE plugin only supports a different output type for FP8");

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
        if (mQuantMode.hasInt4Weights())
        {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
        }
        else
        {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<half, uint8_t>>();
        }
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16 && mWeightType == DataType::kBF16)
    {
        mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
    else if (mType == DataType::kBF16 && mWeightType == DataType::kINT8)
    {
        if (mQuantMode.hasInt4Weights())
        {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
        }
        else
        {
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t>>();
        }
    }
#endif
#ifdef ENABLE_FP8
    else if (mType == DataType::kFP8 && mWeightType == DataType::kFP8)
    {
        switch (mOutputType)
        {
        case nvinfer1::DataType::kFP8:
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>>();
            break;
        case nvinfer1::DataType::kHALF:
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>>();
            break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            mMOERunner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>>();
            break;
#endif
        default: TLLM_THROW("Invalid output type specified for FP8");
        }
    }
#endif
    else
    {
        TLLM_THROW(
            "Could not construct the mixture of experts plugin with the requested input combination Activation: %d "
            "Weight: %d",
            static_cast<int>(mType), static_cast<int>(mWeightType));
    }

    mGemmId = GemmIDMoe{mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode};
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
    assert(outputIndex == getOutputTensorIndex());
    return inputs[getInputTensorIndex()];
}

bool MixtureOfExpertsPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < getNbInputs() + getNbOutputs());
    TLLM_CHECK_WITH_INFO(nbInputs == getNbInputs(), "Required input to plugin is missing");
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing");

    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (pos == getExpertWeights1Index() || pos == getExpertWeights2Index())
    {
        return (inOut[pos].type == mWeightType);
    }
    else if (pos == getFinishedTensorIndex() && hasFinishedTensor())
    {
        return (inOut[pos].type == DataType::kBOOL);
    }
    else if (pos == getRoutingTensorIndex())
    {
        return (inOut[pos].type == DataType::kFLOAT);
    }
    else if (pos == nbInputs + getOutputTensorIndex())
    {
        return inOut[pos].type == mOutputType;
    }
    else if (hasExpertFp8QuantScales() && getExpertFP8Dequant1Index() <= pos && pos <= getExpertFP8QuantFinalIndex())
    {
        return inOut[pos].type == DataType::kFLOAT;
    }
    else
    {
        return (inOut[pos].type == mType);
    }

    return false;
}

void MixtureOfExpertsPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    TLLM_CHECK_WITH_INFO(nbInputs == getNbInputs(), "Required input to plugin is missing");
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing");

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
    mGemmId = GemmIDMoe{mNumExperts, mK, mParallelismConfig, mExpertHiddenSize, mExpertInterSize, mActivationType,
        mType, mWeightType, mQuantMode};
}

auto MixtureOfExpertsPlugin::setupWorkspace(void* base_ptr, int64_t num_tokens) const -> WorkspaceInfo
{
    size_t dtype_size = tensorrt_llm::common::getDTypeSize(mType);

    size_t moe_workspace_size = mMOERunner->getWorkspaceSize(
        num_tokens, mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, mActivationType, mParallelismConfig);

    // Output of post-softmax routing probabilities
    size_t scale_probabilities_size = num_tokens * mNumExperts * sizeof(float);

    // Permutation map
    size_t src_to_dest_map_size = mK * num_tokens * sizeof(int);

    // Selected expert map
    size_t selected_expert_size = mK * num_tokens * sizeof(int);

    std::vector<size_t> workspaces{
        moe_workspace_size,
        scale_probabilities_size,
        src_to_dest_map_size,
        selected_expert_size,
    };

    WorkspaceInfo info{};
    info.size = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    if (base_ptr)
    {
        info.workspace = base_ptr;
        info.scale_probs = nextWorkspacePtr((int8_t*) info.workspace, moe_workspace_size);
        info.src_to_dest_map = nextWorkspacePtr((int8_t*) info.scale_probs, scale_probabilities_size);
        info.selected_experts = nextWorkspacePtr((int8_t*) info.src_to_dest_map, src_to_dest_map_size);
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
    TLLM_CHECK_WITH_INFO(nbInputs == getNbInputs(), "Required input to plugin is missing");
    TLLM_CHECK_WITH_INFO(nbOutputs == getNbOutputs(), "Required output to plugin is missing");

    int const num_tokens = getNumTokens(inputs);
    return setupWorkspace(nullptr, num_tokens).size;
}

MOEParallelismConfig MixtureOfExpertsPlugin::getParallelismConfig() const
{
    return mParallelismConfig;
}

QuantParams tensorrt_llm::plugins::MixtureOfExpertsPlugin::getQuantParams(
    void const* scale_1, void const* scale_2, void const* scale_3, void const* scale_4) const
{
    if (hasExpertIntQuantScales())
    {
        TLLM_CHECK(scale_1 && scale_2);
        return QuantParams::Int(scale_1, scale_2);
    }
    else if (hasExpertFp8QuantScales())
    {
        TLLM_CHECK(scale_1 && scale_2 && scale_3);
        TLLM_CHECK(scale_4 || !hasExpertFp8FinalQuantScales());
        return QuantParams::FP8(static_cast<float const*>(scale_1), static_cast<float const*>(scale_2),
            static_cast<float const*>(scale_3), static_cast<float const*>(scale_4));
    }
    return {};
}

int MixtureOfExpertsPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace_ptr,
    cudaStream_t stream) noexcept
{
    int64_t const num_tokens = getNumTokens(inputDesc);
    int64_t const num_not_finished = num_tokens; // TODO Take this as an input

    auto workspace = setupWorkspace(workspace_ptr, num_tokens);

    auto w1_desc = inputDesc[getExpertWeights1Index()];
    auto w2_desc = inputDesc[getExpertWeights2Index()];
    TLLM_CHECK(w1_desc.dims.nbDims == 3);
    size_t experts_per_node = mNumExperts / mParallelismConfig.ep_size;
    TLLM_CHECK(w1_desc.dims.d[0] == experts_per_node);
    TLLM_CHECK(w2_desc.dims.nbDims == 3);
    TLLM_CHECK(w2_desc.dims.d[0] == experts_per_node);

    int packed_elements = getWeightPackedElements();
    int inner_dim_idx = getGemmShapeInnerDimIndex();
    int outer_dim_idx = getGemmShapeOuterDimIndex();
    TLLM_CHECK(w1_desc.dims.d[inner_dim_idx] == mExpertHiddenSize);
    if (isGatedActivation(mActivationType))
    {
        TLLM_CHECK(w1_desc.dims.d[outer_dim_idx] * packed_elements == mExpertInterSize * 2);
    }
    else
    {
        TLLM_CHECK(w1_desc.dims.d[outer_dim_idx] * packed_elements == mExpertInterSize);
    }

    TLLM_CHECK(w2_desc.dims.d[inner_dim_idx] == mExpertInterSize);
    TLLM_CHECK(w2_desc.dims.d[outer_dim_idx] * packed_elements == mExpertHiddenSize);

    QuantParams quant_params{};
    if (hasExpertIntQuantScales())
    {
        quant_params = getQuantParams(inputs[getExpertIntQuantScale1Index()], inputs[getExpertIntQuantScale2Index()]);
    }
    else if (hasExpertFp8QuantScales())
    {
        quant_params = getQuantParams(           //
            inputs[getExpertFP8Dequant1Index()], //
            inputs[getExpertFP8Quant2Index()],   //
            inputs[getExpertFP8Dequant2Index()], //
            hasExpertFp8FinalQuantScales() ? inputs[getExpertFP8QuantFinalIndex()] : nullptr);
    }

    mMOERunner->setTactic(mPluginProfiler->getBestConfig(num_tokens, mGemmId));
    mMOERunner->runMoe(inputs[getInputTensorIndex()], static_cast<float const*>(inputs[getRoutingTensorIndex()]),
        inputs[getExpertWeights1Index()], hasBias() ? inputs[getExpertBias1Index()] : nullptr, mActivationType,
        inputs[getExpertWeights2Index()], hasBias() ? inputs[getExpertBias2Index()] : nullptr, quant_params, num_tokens,
        mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, static_cast<char*>(workspace.workspace),
        // Outputs
        outputs[getOutputTensorIndex()],
        hasFinishedTensor() ? static_cast<bool const*>(inputs[getFinishedTensorIndex()]) : nullptr, num_not_finished,
        workspace.scale_probs, static_cast<int*>(workspace.src_to_dest_map),
        static_cast<int*>(workspace.selected_experts), mParallelismConfig, mNormalizationMode, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType MixtureOfExpertsPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == getOutputTensorIndex());
    TLLM_CHECK(inputTypes[getInputTensorIndex()] == mType);
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
    mPluginProfiler->profileTactics(this, mType, mDims, mGemmId);
    return 0;
}

void MixtureOfExpertsPlugin::terminate() noexcept {}

void MixtureOfExpertsPlugin::destroy() noexcept
{
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
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2* MixtureOfExpertsPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    nvinfer1::PluginField const* fields = fc->fields;
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

    // Read configurations from each fields
    struct MapPair
    {
        char const* key;
        int& field;
        bool optional = false;
        bool set = false;
    };

    std::array input_map{
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

        // Optional
        MapPair{"use_finished", std::ref(mUseFinished), true},
        MapPair{"use_bias", std::ref(mUseBias), true},
        MapPair{"output_type_id", std::ref(mOutputType), true},
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

    try
    {
        auto pluginProfiler = moePluginProfiler.createGemmPluginProfiler(/* inference */ false);
        auto* obj = new MixtureOfExpertsPlugin(
            // Constructor parameters
            mNumExperts, mK, mExpertHiddenSize, mExpertInterSize,
            static_cast<tensorrt_llm::ActivationType>(mActivationType), static_cast<nvinfer1::DataType>(mType),
            static_cast<nvinfer1::DataType>(mWeightType), static_cast<nvinfer1::DataType>(mOutputType),
            QuantMode(mQuantMode), mUseFinished != 0, mUseBias != 0, mTPSize, mTPRank, mEPSize, mEPRank,
            static_cast<MOEExpertScaleNormalizationMode>(mNormalizationMode), pluginProfiler);
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
        auto pluginProfiler = moePluginProfiler.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new MixtureOfExpertsPlugin(
            // Constructor parameters
            serialData, serialLength, pluginProfiler);
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

std::vector<size_t> MixtureOfExpertsGemmProfiler::getProfilerWorkspaces(int maxM)
{
    auto const& plugin = *mRunner;

    size_t num_tokens = maxM;

    size_t dtype_bytes = tensorrt_llm::common::getDTypeSize(plugin.mType);
    size_t weight_bytes = tensorrt_llm::common::getDTypeSize(plugin.mWeightType);
    size_t output_bytes = tensorrt_llm::common::getDTypeSize(plugin.mOutputType);

    size_t hidden_size = plugin.mExpertHiddenSize;
    size_t inter_size = plugin.mExpertInterSize;
    size_t num_experts = plugin.mNumExperts;

    size_t fc1_out_size = inter_size;
    if (isGatedActivation(plugin.mActivationType))
    {
        fc1_out_size = inter_size * 2;
    }

    size_t input_size = hidden_size * num_tokens * dtype_bytes;
    size_t routing_weights = num_experts * num_tokens * sizeof(float);

    size_t weights_1 = hidden_size * fc1_out_size * num_experts * weight_bytes;

    size_t quant_1 = plugin.hasExpertIntQuantScales() ? fc1_out_size * num_experts * dtype_bytes : 0;
    quant_1 = plugin.hasExpertFp8QuantScales() ? num_experts * sizeof(float) : quant_1;

    size_t bias_1 = plugin.hasBias() ? fc1_out_size * num_experts * dtype_bytes : 0;

    size_t weights_2 = hidden_size * inter_size * num_experts * weight_bytes;

    size_t quant_2 = plugin.hasExpertIntQuantScales() ? hidden_size * num_experts * dtype_bytes : 0;
    quant_2 = plugin.hasExpertFp8QuantScales() ? sizeof(float) : quant_2;

    size_t bias_2 = plugin.hasBias() ? hidden_size * num_experts * dtype_bytes : 0;

    size_t quant_3 = plugin.hasExpertFp8QuantScales() ? num_experts * sizeof(float) : 0;
    size_t quant_4 = plugin.hasExpertFp8FinalQuantScales() ? sizeof(float) : 0;

    size_t output = hidden_size * num_tokens * output_bytes;

    size_t ws_size = plugin.setupWorkspace(nullptr, maxM).size;

    return {routing_weights, // Put this first because we initialise this but nothing else
        input_size, weights_1, quant_1, bias_1, weights_2, quant_2, bias_2, quant_3, quant_4, output, ws_size};
}

void MixtureOfExpertsGemmProfiler::computeTmpSize(int maxM, int n, int k)
{
    auto workspaces = getProfilerWorkspaces(maxM);
    size_t bytes = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    this->setTmpWorkspaceSizeInBytes(bytes);
}

void MixtureOfExpertsGemmProfiler::runTactic(int m, int n, int k, MixtureOfExpertsGemmProfiler::Config const& tactic,
    char* workspace_ptr_char, cudaStream_t const& stream)
{
    assert(mRunner);
    auto& plugin = *mRunner;
    auto parallelism_config = plugin.getParallelismConfig();
    int const num_tokens = m;

    int8_t* workspace_ptr = reinterpret_cast<int8_t*>(workspace_ptr_char);
    auto workspaces = getProfilerWorkspaces(m);
    auto ws_it = workspaces.begin();
    auto getNext = [&]() -> void*
    {
        assert(ws_it != workspaces.end());
        auto res = workspace_ptr;
        size_t element_size_bytes = *ws_it;
        workspace_ptr = nextWorkspacePtr(workspace_ptr, element_size_bytes);
        ws_it++;
        // Return nullptr if size is 0
        return element_size_bytes != 0 ? res : nullptr;
    };

    // Routing goes first as we need to manually initialise it in initTmpData, everything else can be uninit
    // If we didn't init routing all the values could go to one expert, causing the profile to be unreliable (e.g.
    // for expert parallelism)
    auto const* routing = static_cast<float const*>(getNext());

    void const* input = getNext();
    void const* weights_1 = getNext();
    void const* scale_1 = getNext();
    void const* bias_1 = getNext();
    void const* weights_2 = getNext();
    void const* scale_2 = getNext();
    void const* bias_2 = getNext();
    void const* scale_3 = getNext();
    void const* scale_4 = getNext();
    void* output = getNext();
    bool const* finished = nullptr; // No finished, we want to benchmark all tokens

    auto workspace = plugin.setupWorkspace(getNext(), num_tokens);

    QuantParams quant_params = plugin.getQuantParams(scale_1, scale_2, scale_3, scale_4);

    plugin.mMOERunner->is_profiler = true;
    // TODO(dastokes) We should probably profile the two GEMMs separately as the optimal config may differ
    plugin.mMOERunner->setTactic(tactic);
    plugin.mMOERunner->runMoe(input, routing, weights_1, bias_1, plugin.mActivationType, weights_2, bias_2,
        quant_params, num_tokens, plugin.mExpertHiddenSize, plugin.mExpertInterSize, plugin.mNumExperts, plugin.mK,
        static_cast<char*>(workspace.workspace),
        // Outputs
        output, finished, num_tokens, workspace.scale_probs, static_cast<int*>(workspace.src_to_dest_map),
        static_cast<int*>(workspace.selected_experts), parallelism_config, plugin.mNormalizationMode, stream);
    plugin.mMOERunner->is_profiler = false;

    sync_check_cuda_error();
}

auto MixtureOfExpertsGemmProfiler::getTactics(int m, int n, int k) const -> std::vector<Config>
{
    assert(mRunner);
    return mRunner->mMOERunner->getTactics();
}

void MixtureOfExpertsGemmProfiler::initTmpData(int m, int, int, char* workspace, size_t ws_size, cudaStream_t stream)
{
    assert(mRunner);
    auto& plugin = *mRunner;
    int num_tokens = m;
    void* routing_workspace = workspace;
    tensorrt_llm::common::check_cuda_error(cudaMemsetAsync(workspace, 0x0, ws_size, stream));
    makeLoadBalancedRoutingConfiguration(
        routing_workspace, plugin.mNumExperts, num_tokens, plugin.mK, DataType::kFLOAT, stream);
}
