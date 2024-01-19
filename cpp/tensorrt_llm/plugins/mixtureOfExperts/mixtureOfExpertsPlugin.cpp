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

static const char* MIXTURE_OF_EXPERTS_PLUGIN_VERSION{"1"};
static const char* MIXTURE_OF_EXPERTS_PLUGIN_NAME{"MixtureOfExperts"};
nvinfer1::PluginFieldCollection MixtureOfExpertsPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> MixtureOfExpertsPluginCreator::mPluginAttributes;

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(int number_of_experts, int top_k, int expert_hidden_size,
    int expert_inter_size, tensorrt_llm::ActivationType activation_type, nvinfer1::DataType type,
    nvinfer1::DataType weight_type, QuantMode quant_mode, bool use_finished, bool use_bias, int tp_size, int tp_rank,
    MOEParallelismMode parallelism_mode, MOEExpertScaleNormalizationMode normalization_mode,
    MixtureOfExpertsPluginProfilerPtr plugin_profiler_ptr)
    : mNumExperts(number_of_experts)
    , mK(top_k)
    , mExpertHiddenSize(expert_hidden_size)
    , mExpertInterSize(expert_inter_size)
    , mActivationType(activation_type)
    , mType(type)
    , mWeightType(weight_type)
    , mQuantMode(quant_mode)
    , mUseFinished(use_finished)
    , mUseBias(use_bias)
    , mTPSize(tp_size)
    , mTPRank(tp_rank)
    , mParallelismMode(parallelism_mode)
    , mNormalizationMode(normalization_mode)
    , mPluginProfiler(std::move(plugin_profiler_ptr))
{
    init();
}

tensorrt_llm::plugins::MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(const MixtureOfExpertsPlugin& other)
    : mMOERunner()
    , mNumExperts(other.mNumExperts)
    , mK(other.mK)
    , mExpertHiddenSize(other.mExpertHiddenSize)
    , mExpertInterSize(other.mExpertInterSize)
    , mActivationType(other.mActivationType)
    , mType(other.mType)
    , mWeightType(other.mWeightType)
    , mQuantMode(other.mQuantMode)
    , mUseFinished(other.mUseFinished)
    , mUseBias(other.mUseBias)
    , mTPSize(other.mTPSize)
    , mTPRank(other.mTPRank)
    , mParallelismMode(other.mParallelismMode)
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
        + sizeof(mActivationType) + sizeof(mType) + sizeof(mWeightType) + sizeof(QuantMode::BaseType)
        + sizeof(mUseFinished) + sizeof(mUseBias) + sizeof(mTPSize) + sizeof(mTPRank) + sizeof(mParallelismMode)
        + sizeof(mNormalizationMode) + sizeof(mDims) + mPluginProfiler->getSerializationSize(mGemmId);
}

MixtureOfExpertsPlugin::MixtureOfExpertsPlugin(
    const void* data, size_t length, MixtureOfExpertsPluginProfilerPtr plugin_profiler_ptr)
    : mPluginProfiler(plugin_profiler_ptr)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    read(d, mNumExperts);
    read(d, mK);
    read(d, mExpertHiddenSize);
    read(d, mExpertInterSize);
    read(d, mActivationType);
    read(d, mType);
    read(d, mWeightType);
    QuantMode::BaseType quant_mode;
    read(d, quant_mode);
    mQuantMode = QuantMode{quant_mode};
    read(d, mUseFinished);
    read(d, mUseBias);
    read(d, mTPSize);
    read(d, mTPRank);
    read(d, mParallelismMode);
    read(d, mNormalizationMode);
    read(d, mDims);

    init();
    mPluginProfiler->deserialize(d, mDims, mGemmId);
    TLLM_CHECK(d == a + length);
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
    write(d, mQuantMode.value());
    write(d, mUseFinished);
    write(d, mUseBias);
    write(d, mTPSize);
    write(d, mTPRank);
    write(d, mParallelismMode);
    write(d, mNormalizationMode);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);

    assert(d == a + getSerializationSize());
}

void MixtureOfExpertsPlugin::init()
{
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
    else
    {
        TLLM_THROW("Could not construct the mixture of experts plugin with the requested input combination");
    }

    mGemmId = GemmIDMoe{mNumExperts, mK, mExpertHiddenSize, mExpertInterSize, mActivationType, mType, mWeightType,
        mQuantMode, mParallelismMode};
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* MixtureOfExpertsPlugin::clone() const noexcept
{
    auto* plugin = new MixtureOfExpertsPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs MixtureOfExpertsPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == getOutputTensorIndex());
    return inputs[getInputTensorIndex()];
}

bool MixtureOfExpertsPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < getNbInputs() + getNbOutputs());
    TLLM_CHECK(nbInputs == getNbInputs());
    TLLM_CHECK(nbOutputs == getNbOutputs());

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
    else
    {
        return (inOut[pos].type == mType);
    }

    return false;
}

void MixtureOfExpertsPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    auto in_tensor = in[getInputTensorIndex()];

    const auto minM
        = std::accumulate(in_tensor.min.d, in_tensor.min.d + in_tensor.min.nbDims - 1, 1, std::multiplies<int>());
    const auto maxM
        = std::accumulate(in_tensor.max.d, in_tensor.max.d + in_tensor.max.nbDims - 1, 1, std::multiplies<int>());

    auto weights_1 = in[getExpertWeights1Index()];
    auto weights_2 = in[getExpertWeights2Index()];
    int inner_dim_idx = getGemmShapeInnerDimIndex();
    const int maxK = weights_1.max.d[inner_dim_idx];
    const int maxN = weights_2.max.d[inner_dim_idx];
    const int minK = weights_1.min.d[inner_dim_idx];
    const int minN = weights_2.min.d[inner_dim_idx];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");
    TLLM_CHECK_WITH_INFO(maxK == mExpertHiddenSize, "Configured tensor sizes does not match constructor param size");
    TLLM_CHECK_WITH_INFO(maxN == mExpertInterSize, "Configured tensor sizes does not match constructor param size");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = GemmIDMoe{mNumExperts, mK, mExpertHiddenSize, mExpertInterSize, mActivationType, mType, mWeightType,
        mQuantMode, mParallelismMode};
}

auto MixtureOfExpertsPlugin::setupWorkspace(void* base_ptr, int num_tokens) const -> WorkspaceInfo
{

    size_t dtype_size = tensorrt_llm::common::getDTypeSize(mType);

    size_t moe_workspace_size = mMOERunner->getWorkspaceSize(
        num_tokens, mExpertHiddenSize, mExpertInterSize, mNumExperts, mK, mActivationType, getParallelismConfig());

    // Output of post-softmax routing probabilities
    size_t scale_probabilities_size = num_tokens * mNumExperts * sizeof(float);

    // Hidden states buffer for GEMM result
    size_t fc2_output_size = mK * mExpertHiddenSize * num_tokens * dtype_size;

    // Permutation map
    size_t src_to_dest_map_size = mK * num_tokens * sizeof(int);

    // Selected expert map
    size_t selected_expert_size = mK * num_tokens * sizeof(int);

    std::vector<size_t> workspaces{
        moe_workspace_size,
        scale_probabilities_size,
        fc2_output_size,
        src_to_dest_map_size,
        selected_expert_size,
    };

    WorkspaceInfo info{};
    info.size = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    if (base_ptr)
    {
        info.workspace = base_ptr;
        info.scale_probs = nextWorkspacePtr((int8_t*) info.workspace, moe_workspace_size);
        info.fc2_output = nextWorkspacePtr((int8_t*) info.scale_probs, scale_probabilities_size);
        info.src_to_dest_map = nextWorkspacePtr((int8_t*) info.fc2_output, fc2_output_size);
        info.selected_experts = nextWorkspacePtr((int8_t*) info.src_to_dest_map, src_to_dest_map_size);
    }

    return info;
}

int MixtureOfExpertsPlugin::getNumTokens(const nvinfer1::PluginTensorDesc* input_tensors) const
{
    int ndim = input_tensors[getInputTensorIndex()].dims.nbDims;
    TLLM_CHECK_WITH_INFO(
        3 == ndim || 2 == ndim, "hidden_state dimension should be either 2 [b*s, hidden], or 3 [b, s, hidden]");
    int num_tokens = input_tensors[getInputTensorIndex()].dims.d[0];
    if (ndim == 3)
    {
        num_tokens *= input_tensors[getInputTensorIndex()].dims.d[1];
    }
    return num_tokens;
}

size_t MixtureOfExpertsPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    const int num_tokens = getNumTokens(inputs);
    return setupWorkspace(nullptr, num_tokens).size;
}

MOEParallelismConfig MixtureOfExpertsPlugin::getParallelismConfig() const
{
    switch (mParallelismMode)
    {
    case kernels::MOEParallelismMode::NONE: return {};
    case kernels::MOEParallelismMode::EXPERT_PARALLELISM:
        return MOEParallelismConfig::ExpertParallelism(mTPSize, mTPRank);
    case kernels::MOEParallelismMode::TENSOR_PARALLELISM:
        return MOEParallelismConfig::TensorParallelism(mTPSize, mTPRank);
    }
    assert(false);
    return {};
}

int MixtureOfExpertsPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace_ptr,
    cudaStream_t stream) noexcept
{
    const int num_tokens = getNumTokens(inputDesc);
    const int num_not_finished = num_tokens; // TODO Take this as an input
    auto parallelism_config = getParallelismConfig();

    auto workspace = setupWorkspace(workspace_ptr, num_tokens);

    auto w1_desc = inputDesc[getExpertWeights1Index()];
    auto w2_desc = inputDesc[getExpertWeights2Index()];
    TLLM_CHECK(w1_desc.dims.nbDims == 3);
    size_t experts_per_node = mNumExperts / parallelism_config.ep_size;
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

    mMOERunner->setTactic(mPluginProfiler->getBestConfig(num_tokens, mGemmId));
    mMOERunner->runMoe(inputs[getInputTensorIndex()], static_cast<const float*>(inputs[getRoutingTensorIndex()]),
        inputs[getExpertWeights1Index()], hasExpertQuantScales() ? inputs[getExpertQuantScale1Index()] : nullptr,
        hasBias() ? inputs[getExpertBias1Index()] : nullptr, mActivationType, inputs[getExpertWeights2Index()],
        hasExpertQuantScales() ? inputs[getExpertQuantScale2Index()] : nullptr,
        hasBias() ? inputs[getExpertBias2Index()] : nullptr, num_tokens, mExpertHiddenSize, mExpertInterSize,
        mNumExperts, mK, static_cast<char*>(workspace.workspace),
        // Outputs
        outputs[getOutputTensorIndex()], workspace.fc2_output,
        hasFinishedTensor() ? static_cast<const bool*>(inputs[getFinishedTensorIndex()]) : nullptr, num_not_finished,
        workspace.scale_probs, static_cast<int*>(workspace.src_to_dest_map),
        static_cast<int*>(workspace.selected_experts), parallelism_config, mNormalizationMode, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType MixtureOfExpertsPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == getOutputTensorIndex());
    assert(inputTypes[getInputTensorIndex()] == mType);
    return mType;
}

// IPluginV2 Methods
const char* MixtureOfExpertsPlugin::getPluginType() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_NAME;
}

const char* MixtureOfExpertsPlugin::getPluginVersion() const noexcept
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

void MixtureOfExpertsPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MixtureOfExpertsPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

const char* MixtureOfExpertsPluginCreator::getPluginName() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_NAME;
}

const char* MixtureOfExpertsPluginCreator::getPluginVersion() const noexcept
{
    return MIXTURE_OF_EXPERTS_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* MixtureOfExpertsPluginCreator::getFieldNames() noexcept
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
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "parallelism_mode", nullptr, PluginFieldType::kINT32, static_cast<int>(MOEParallelismMode::NONE)));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("normalization_mode", nullptr, PluginFieldType::kINT32,
        static_cast<int>(MOEExpertScaleNormalizationMode::NONE)));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2* MixtureOfExpertsPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    const nvinfer1::PluginField* fields = fc->fields;
    int mNumExperts{};
    int mK{};
    int mExpertHiddenSize{};
    int mExpertInterSize{};
    int mActivationType{};
    int mType{};
    int mWeightType{};
    int mQuantMode{};
    int mUseFinished{};
    int mUseBias{};
    int mTPSize{};
    int mTPRank{};
    int mParallelismMode{};
    int mNormalizationMode{};

    // Read configurations from each fields
    using MapPair = std::pair<const char*, std::reference_wrapper<int>>;
    const std::array input_map{
        MapPair{"number_of_experts", std::ref(mNumExperts)},
        MapPair{"top_k", std::ref(mK)},
        MapPair{"expert_hidden_size", std::ref(mExpertHiddenSize)},
        MapPair{"expert_inter_size", std::ref(mExpertInterSize)},
        MapPair{"activation_type", std::ref(mActivationType)},
        MapPair{"type_id", std::ref(mType)},
        MapPair{"weight_type_id", std::ref(mWeightType)},
        MapPair{"quant_mode", std::ref(mQuantMode)},
        MapPair{"use_finished", std::ref(mUseFinished)},
        MapPair{"use_bias", std::ref(mUseBias)},
        MapPair{"tp_size", std::ref(mTPSize)},
        MapPair{"tp_rank", std::ref(mTPRank)},
        MapPair{"parallelism_mode", std::ref(mParallelismMode)},
        MapPair{"normalization_mode", std::ref(mNormalizationMode)},
    };
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        for (const auto& item : input_map)
        {
            if (!strcmp(item.first, attrName))
            {
                TLLM_CHECK(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                item.second.get() = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
        }
    }
    try
    {
        auto pluginProfiler = moePluginProfiler.createGemmPluginProfiler(/* inference */ false);
        auto* obj = new MixtureOfExpertsPlugin(
            // Constructor parameters
            mNumExperts, mK, mExpertHiddenSize, mExpertInterSize,
            static_cast<tensorrt_llm::ActivationType>(mActivationType), static_cast<nvinfer1::DataType>(mType),
            static_cast<nvinfer1::DataType>(mWeightType), QuantMode(mQuantMode), mUseFinished != 0, mUseBias != 0,
            mTPSize, mTPRank, static_cast<MOEParallelismMode>(mParallelismMode),
            static_cast<MOEExpertScaleNormalizationMode>(mNormalizationMode), pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* MixtureOfExpertsPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
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
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MixtureOfExpertsPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MixtureOfExpertsPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

std::vector<size_t> MixtureOfExpertsGemmProfiler::getProfilerWorkspaces(int maxM)
{
    const auto& plugin = *mRunner;

    size_t num_tokens = maxM;

    size_t dtype_bytes = tensorrt_llm::common::getDTypeSize(plugin.mType);
    size_t weight_bytes = tensorrt_llm::common::getDTypeSize(plugin.mWeightType);

    size_t hidden_size = plugin.mExpertHiddenSize;
    size_t inter_size = plugin.mExpertInterSize;
    size_t num_experts = plugin.mNumExperts;

    size_t input_size = hidden_size * num_tokens * dtype_bytes;
    size_t routing_weights = num_experts * num_tokens * sizeof(float);

    size_t weights_1 = hidden_size * inter_size * num_experts * weight_bytes;

    size_t quant_1 = plugin.hasExpertQuantScales() ? inter_size * num_experts * dtype_bytes : 0;
    size_t bias_1 = plugin.hasBias() ? inter_size * num_experts * dtype_bytes : 0;

    size_t weights_2 = hidden_size * inter_size * num_experts * weight_bytes;

    size_t quant_2 = plugin.hasExpertQuantScales() ? hidden_size * num_experts * dtype_bytes : 0;
    size_t bias_2 = plugin.hasBias() ? hidden_size * num_experts * dtype_bytes : 0;

    size_t output = hidden_size * num_tokens * dtype_bytes;

    size_t ws_size = plugin.setupWorkspace(nullptr, maxM).size;
    return {routing_weights, // Put this first because we initialise this but nothing else
        input_size, weights_1, quant_1, bias_1, weights_2, quant_2, bias_2, output, ws_size};
}

void MixtureOfExpertsGemmProfiler::computeTmpSize(int maxM, int n, int k)
{
    auto workspaces = getProfilerWorkspaces(maxM);
    size_t bytes = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    this->setTmpWorkspaceSizeInBytes(bytes);
}

void MixtureOfExpertsGemmProfiler::runTactic(int m, int n, int k, const MixtureOfExpertsGemmProfiler::Config& tactic,
    char* workspace_ptr_char, cudaStream_t const& stream)
{
    assert(mRunner);
    auto& plugin = *mRunner;
    auto parallelism_config = plugin.getParallelismConfig();
    const int num_tokens = m;

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
    // If we didn't init routing all the values could go to one expert, causing the profile to be unreliable (e.g. for
    // expert parallelism)
    const float* routing = static_cast<const float*>(getNext());

    const void* input = getNext();
    const void* weights_1 = getNext();
    const void* scale_1 = getNext();
    const void* bias_1 = getNext();
    const void* weights_2 = getNext();
    const void* scale_2 = getNext();
    const void* bias_2 = getNext();
    void* output = getNext();
    const bool* finished = nullptr; // No finished, we want to benchmark all tokens

    auto workspace = plugin.setupWorkspace(getNext(), num_tokens);

    plugin.mMOERunner->setTactic(tactic);
    plugin.mMOERunner->runMoe(input, routing, weights_1, scale_1, bias_1, plugin.mActivationType, weights_2, scale_2,
        bias_2, num_tokens, plugin.mExpertHiddenSize, plugin.mExpertInterSize, plugin.mNumExperts, plugin.mK,
        static_cast<char*>(workspace.workspace),
        // Outputs
        output, workspace.fc2_output, finished, num_tokens, workspace.scale_probs,
        static_cast<int*>(workspace.src_to_dest_map), static_cast<int*>(workspace.selected_experts), parallelism_config,
        plugin.mNormalizationMode, stream);

    sync_check_cuda_error();
}

auto MixtureOfExpertsGemmProfiler::getTactics(int m, int n, int k) const -> std::vector<Config>
{
    assert(mRunner);
    return mRunner->mMOERunner->getTactics();
}

void MixtureOfExpertsGemmProfiler::initTmpData(int m, int, int, char* workspace, size_t, cudaStream_t stream)
{
    assert(mRunner);
    auto& plugin = *mRunner;
    int num_tokens = m;
    void* routing_workspace = workspace;
    makeLoadBalancedRoutingConfiguration(
        routing_workspace, plugin.mNumExperts, num_tokens, plugin.mK, DataType::kFLOAT, stream);
}
