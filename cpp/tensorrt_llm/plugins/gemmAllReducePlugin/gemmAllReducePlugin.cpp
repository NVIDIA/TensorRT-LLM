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
#include "gemmAllReducePlugin.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/plugins/common/pluginUtils.h"

#include <unistd.h>

static char const* GEMM_ALLREDUCE_PLUGIN_VERSION = "1";
static char const* GEMM_ALLREDUCE_PLUGIN_NAME = "GemmAllReduce";
template <nvinfer1::DataType T>
using CutlassType = ::tensorrt_llm::kernels::cutlass_kernels::CutlassType<T>;

namespace tensorrt_llm::plugins
{
template <typename K, typename V, DataType ElementA, DataType ElementB, DataType ElementD>
static std::pair<K, V> makeEntry()
{
    return {std::make_tuple(ElementA, ElementB, ElementD),
        [&]()
        {
            using GemmTraits
                = cutlass_kernels::GemmTypes<typename CutlassType<ElementA>::type, typename CutlassType<ElementB>::type,
                    typename CutlassType<ElementD>::type,                                         // C, unused
                    typename CutlassType<ElementD>::type,
                    std::conditional_t<ElementA == DataType::kFP4, cutlass::float_ue4m3_t, void>, // SFA
                    std::conditional_t<ElementB == DataType::kFP4, cutlass::float_ue4m3_t, void>, // SFB
                    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
                    cutlass::layout::RowMajor,                                                    // C, unused
                    cutlass::layout::RowMajor>;
            return new cutlass_kernels::GemmAllReduceImplRunner<GemmTraits>();
        }};
}

template <typename K, typename V>
static std::map<K, V> getTypedInstantiators()
{
    return std::map<K, V>({makeEntry<K, V, DataType::kHALF, DataType::kHALF, DataType::kHALF>(),
        makeEntry<K, V, DataType::kBF16, DataType::kBF16, DataType::kBF16>(),
        makeEntry<K, V, DataType::kFP8, DataType::kFP8, DataType::kHALF>(),
        makeEntry<K, V, DataType::kFP8, DataType::kFP8, DataType::kBF16>(),
        makeEntry<K, V, DataType::kFP4, DataType::kFP4, DataType::kHALF>(),
        makeEntry<K, V, DataType::kFP4, DataType::kFP4, DataType::kBF16>()});
}

////////////////////////////////////////////////////////////
// GemmAllReducePlugin Methods
////////////////////////////////////////////////////////////
GemmAllReducePlugin::GemmAllReducePlugin(GemmAllReducePluginOptions const& options)
    : mOptions(options)
    , mGemmId(GemmIdCore(options.maxProblemShape.n, options.maxProblemShape.k, options.typeD))
    , mProfiler(mGemmPluginProfileManager.createGemmPluginProfiler(/*inference=*/options.deserialize))
{
    // construct mapping of input/output pos to argument
    int argIdx = 0;
    // inputs
    mArgMap[argIdx++] = TensorArg::IN_ACTIVATION;
    mArgMap[argIdx++] = TensorArg::IN_WEIGHT;
    if (mOptions.hasSFA)
    {
        mArgMap[argIdx++] = TensorArg::IN_ACTIVATION_SF;
    }
    if (mOptions.hasSFB)
    {
        mArgMap[argIdx++] = TensorArg::IN_WEIGHT_SF;
    }
    if (mOptions.alphaIsPtr)
    {
        mArgMap[argIdx++] = TensorArg::IN_ALPHA;
    }
    mNbInputs = argIdx;
    // outputs
    mArgMap[argIdx++] = TensorArg::OUT_D_UC;
    mArgMap[argIdx++] = TensorArg::OUT_D_MC;
    mArgMap[argIdx++] = TensorArg::OUT_D_IPC;
    mNbOutputs = argIdx - mNbInputs;

    // Create mapping of argument to tensor pos
    for (auto const& pair : mArgMap)
    {
        mArgInvMap[pair.second] = pair.first;
    }

    // Use map instead of huge switch case
    mTypedInstantiators = getTypedInstantiators<KeyType, ValueType>();

    auto key = std::make_tuple(mOptions.typeA, mOptions.typeB, mOptions.typeD);

    TLLM_CHECK_WITH_INFO(mTypedInstantiators.count(key) > 0, "No cutlass gemm for impl.");
    mGemm = std::shared_ptr<cutlass_kernels::GemmAllReduceImplInterface>(mTypedInstantiators[key]());
}

void GemmAllReducePlugin::allocatePersistentWorkspace()
{
    TLLM_CHECK(mOptions.maxProblemShape.isInitialized());

    mWorkspaceKey = "gemm_allreduce_workspace_m" + std::to_string(mOptions.maxProblemShape.maxM);

    cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig smallest_tile_config
        = mGemm->getSupportedLaunchConfigs()[0];
    cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args;
    args.argProblemShape(mOptions.maxProblemShape.maxM, mOptions.maxProblemShape.n, mOptions.maxProblemShape.k, 1)
        .argRanks(mRank, mOptions.group)
        .argLaunchConfig(smallest_tile_config);

    TLLM_CHECK(mWorkspace == nullptr);

    // Wrap persistent workspace in IPluginResource type
    // so that clone() can be called to allocate memory
    GemmAllReducePersistentWorkspace unallocated_resource(mGemm->getPersistentWorkspace(args));

    // Register and allocate workspace
    mWorkspace = static_cast<GemmAllReducePersistentWorkspace*>(
        getPluginRegistry()->acquirePluginResource(mWorkspaceKey.c_str(), &unallocated_resource));
    TLLM_CHECK(mWorkspace != nullptr);
}

LaunchConfig GemmAllReducePlugin::getStaticHeuristicLaunchConfig(int M) const
{
    using namespace tensorrt_llm::cutlass_extensions;
    // This is only applicable when we swap and transpose A & B.
    // When M is small we want to select tile that best fits it to maximize MMA efficiency.
    auto filterByM = [&](std::vector<LaunchConfig> candidateConfigs)
    {
        std::vector<LaunchConfig> result;
        if (M <= 16)
        {
            std::copy_if(candidateConfigs.begin(), candidateConfigs.end(), std::back_inserter(result),
                [](const LaunchConfig& config)
                { return config.tile_shape == TileShape::TileShape_128x16x128 and config.transposed; });
        }
        else if (M <= 32)
        {
            std::copy_if(candidateConfigs.begin(), candidateConfigs.end(), std::back_inserter(result),
                [](const LaunchConfig& config)
                { return config.tile_shape == TileShape::TileShape_128x32x128 and config.transposed; });
        }
        else if (M <= 64)
        {
            std::copy_if(candidateConfigs.begin(), candidateConfigs.end(), std::back_inserter(result),
                [](const LaunchConfig& config)
                { return config.tile_shape == TileShape::TileShape_128x64x128 and config.transposed; });
        }
        else
        {
            std::copy_if(candidateConfigs.begin(), candidateConfigs.end(), std::back_inserter(result),
                [](const LaunchConfig& config)
                { return config.tile_shape == TileShape::TileShape_128x128x128 and config.transposed; });
        }
        // If result empty then use any.
        if (result.empty())
        {
            result = candidateConfigs;
        }
        return result;
    };

    auto bestLaunchConfigs = mGemm->getSupportedLaunchConfigs();
    bestLaunchConfigs = filterByM(bestLaunchConfigs);
    TLLM_CHECK(!bestLaunchConfigs.empty());
    // Return first one, because who knows which is best.
    return bestLaunchConfigs.front();
}

static GemmAllReducePluginOptions deserializeOptions(void const*& data, size_t length)
{
    char const* begin = reinterpret_cast<char const*>(data);
    char const*& end = reinterpret_cast<char const*&>(data);
    GemmAllReducePluginOptions options;
    options.deserialize = true;

    read(end, options.typeA);
    read(end, options.typeB);
    read(end, options.typeD);
    read(end, options.transA);
    read(end, options.transB);
    read(end, options.alpha);
    read(end, options.maxProblemShape);
    read(end, options.groupSize);
    for (int i = 0; i < options.groupSize; ++i)
    {
        int rank = -1;
        read(end, rank);
        options.group.insert(rank);
    }
    read(end, options.hasSFA);
    read(end, options.hasSFB);
    read(end, options.alphaIsPtr);

    TLLM_CHECK_WITH_INFO(end == begin + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (end - begin));

    return options;
}

GemmAllReducePlugin::GemmAllReducePlugin(void const* data, size_t length)
    : GemmAllReducePlugin(deserializeOptions(std::ref(data), length))
{
    if (mProfiler->useProfiler())
    {
        mProfiler->deserializeFromOwnFile(mGemmId, mOptions.maxProblemShape);
    }
}

//////////////////////////////////
// IPluginV2DynamicExt Methods
//////////////////////////////////
IPluginV2DynamicExt* GemmAllReducePlugin::clone() const noexcept
{
    return new GemmAllReducePlugin(*this);
}

DimsExprs GemmAllReducePlugin::getOutputDimensions(
    int outputIndex, DimsExprs const* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == mNbInputs); // number of input tensors
        TLLM_CHECK(inputs[0].nbDims == inputs[1].nbDims);
        TLLM_CHECK(outputIndex < getNbOutputs());

        // List of pointers to D on each rank
        if ((nbInputs + outputIndex) == TensorArg::OUT_D_IPC)
        {
            DimsExprs out_dims;
            out_dims.nbDims = 1;
            out_dims.d[0] = exprBuilder.constant(mOptions.groupSize);
            return out_dims;
        }

        TLLM_CHECK(mOptions.transA == false);
        TLLM_CHECK(mOptions.transB == true);

        int const nbDimsA = inputs[0].nbDims; // number of dims
        int const nbDimsB = inputs[1].nbDims;

        DimsExprs out_dims;
        // subtract 2 -> K from each input
        out_dims.nbDims = nbDimsA + nbDimsB - 2;

        if (mOptions.transA)
        {
            for (int i = 1; i < nbDimsA; ++i)
            {
                out_dims.d[i - 1] = inputs[0].d[i];
            }
        }
        else
        {
            for (int i = 0; i < nbDimsA - 1; ++i)
            {
                out_dims.d[i] = inputs[0].d[i];
            }
        }
        if (mOptions.transB)
        {
            for (int i = 0; i < nbDimsB - 1; ++i)
            {
                out_dims.d[nbDimsA - 1 + i] = inputs[1].d[i];
            }
        }
        else
        {
            for (int i = 1; i < nbDimsB; ++i)
            {
                out_dims.d[nbDimsA - 2 + i] = inputs[1].d[i];
            }
        }
        return out_dims;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmAllReducePlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // inOut[0] -> activation
    // inOut[1] -> weight
    // inOut[1+hasInputSF] -> activation_sf
    // inOut[1+hasInputSF*2] -> weight_sf
    // inOut[2+hasInputSF*2] -> output[0] = D_uc
    // inOut[3+hasInputSF*2] -> output[1] = D_mc

    TLLM_CHECK_WITH_INFO(pos < mNbInputs + mNbOutputs, "Unexpected pos: %d", pos);
    auto const& desc = inOut[pos];

    TLLM_CHECK_WITH_INFO(mArgMap.count(pos) > 0, "pos %d not found in mArgMap.", pos);
    TensorArg arg = mArgMap[pos];

    auto typeExists = [&](DataType dtype, auto idx) -> bool
    {
        for (const auto& [key, value] : mTypedInstantiators)
        {
            // key format: <ActivationType, WeightType, OutputType>
            if (std::get<decltype(idx)::value>(key) == dtype)
            {
                return true;
            }
        }
        return false;
    };

    switch (arg)
    {
    case TensorArg::IN_ACTIVATION: return typeExists(desc.type, std::integral_constant<size_t, 0>{});
    case TensorArg::IN_WEIGHT: return typeExists(desc.type, std::integral_constant<size_t, 1>{});
    case TensorArg::IN_ACTIVATION_SF:
    case TensorArg::IN_WEIGHT_SF:
        // Assumed SF for only FP4 at the moment
        return desc.type == DataType::kFP8;
    case TensorArg::IN_ALPHA: return desc.type == DataType::kFLOAT;
    case TensorArg::OUT_D_UC:
    case TensorArg::OUT_D_MC:
    case TensorArg::OUT_D_IPC: return typeExists(desc.type, std::integral_constant<size_t, 2>{});
    default: return false;
    }
}

void GemmAllReducePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // Get problem shape
    int const nbDimsA = in[0].max.nbDims;
    int const minM = utils::computeMDimension(mOptions.transA, in[0].min);
    int const maxM = utils::computeMDimension(mOptions.transA, in[0].max);
    int const N = utils::computeNDimension(mOptions.transB, in[1].max);
    int const K = mOptions.transA ? in[0].max.d[0] : in[0].max.d[nbDimsA - 1];

    TLLM_CHECK_WITH_INFO(out[0].desc.type == mOptions.typeD, "Output type mismatch.");

    // Ensure call from execution phase does
    // not override call from build phase
    if (!mOptions.maxProblemShape.isInitialized())
    {
        mOptions.maxProblemShape = {minM, maxM, N, K};
        mGemmId = {N, K, mOptions.typeD};
    }

    // Build phase doesn't have COMM_SESSION (i.e built on single rank)
    // so do not allocate persistent workspace
    if (!isBuilding())
    {
        auto getTPRank = [&]()
        {
            int rank = COMM_SESSION.getRank();
            auto it = std::find(mOptions.group.begin(), mOptions.group.end(), rank);
            TLLM_CHECK_WITH_INFO(it != mOptions.group.end(),
                "Incorrect group specified - rank " + std::to_string(rank) + " not found in group");
            return std::distance(mOptions.group.begin(), it);
        };

        mRank = getTPRank();

        if (mWorkspace == nullptr)
        {
            allocatePersistentWorkspace();
        }
    }
}

size_t GemmAllReducePlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int GemmAllReducePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs[0] -> [M(*), K]
    // inputs[1] -> [K, N]
    // outputs[0] -> [M(*), N] unicast ptr
    // outputs[1] -> [M(*), N] multicast ptr
    auto const nbDimsA = inputDesc[0].dims.nbDims;
    auto const M = utils::computeMDimension(mOptions.transA, inputDesc[0].dims);
    auto const N = utils::computeNDimension(mOptions.transB, inputDesc[1].dims);
    auto const K = mOptions.transA ? inputDesc[0].dims.d[0] : inputDesc[0].dims.d[nbDimsA - 1];

    TLLM_CHECK_WITH_INFO(M <= mOptions.maxProblemShape.maxM, "GemmAllReducePlugin M > maxM.");
    TLLM_CHECK_WITH_INFO(M > 0, "GemmAllReducePlugin M is 0.");
    TLLM_CHECK_WITH_INFO(N > 0, "GemmAllReducePlugin N is 0.");
    TLLM_CHECK_WITH_INFO(K > 0, "GemmAllReducePlugin K is 0.");
    TLLM_CHECK_WITH_INFO(mWorkspace != nullptr, "GemmAllReducePlugin workspace is null.");

    LaunchConfig bestLaunchConfig;
    if (mProfiler->useProfiler())
    {
        bestLaunchConfig = mProfiler->getBestConfig(M, mGemmId).value();
    }
    else
    {
        bestLaunchConfig = getStaticHeuristicLaunchConfig(M);
    }

    void const* activation = inputs[mArgInvMap[TensorArg::IN_ACTIVATION]];
    void const* weight = inputs[mArgInvMap[TensorArg::IN_WEIGHT]];
    void* D_out_uc = outputs[mArgInvMap[TensorArg::OUT_D_UC] - mNbInputs];
    void* D_out_mc = outputs[mArgInvMap[TensorArg::OUT_D_MC] - mNbInputs];
    void* D_out_ipc = outputs[mArgInvMap[TensorArg::OUT_D_IPC] - mNbInputs];

    TLLM_CHECK_WITH_INFO(activation != nullptr, "GemmAllReducePlugin activation is NULL");
    TLLM_CHECK_WITH_INFO(weight != nullptr, "GemmAllReducePlugin weight is NULL");
    TLLM_CHECK_WITH_INFO(D_out_uc != nullptr, "GemmAllReducePlugin out_uc is NULL");
    TLLM_CHECK_WITH_INFO(D_out_mc != nullptr, "GemmAllReducePlugin out_mc is NULL");
    TLLM_CHECK_WITH_INFO(D_out_ipc != nullptr, "GemmAllReducePlugin out_ipc is NULL");

    cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args;
    args.argProblemShape(M, N, K, 1)
        .argA(activation)
        .argB(weight)
        .argC(nullptr)
        .argD(D_out_uc, D_out_mc, (void**) D_out_ipc)
        .argRanks(mRank, mOptions.group)
        .argBeta(0.f) // no bias
        .argLaunchConfig(bestLaunchConfig)
        .argWorkspace(mWorkspace->mWorkspace.get());
    // tensor for scaling input A
    if (mOptions.hasSFA)
    {
        void const* activation_sf = inputs[mArgInvMap[TensorArg::IN_ACTIVATION_SF]];
        TLLM_CHECK_WITH_INFO(activation_sf != nullptr, "GemmAllReducePlugin activation_sf is NULL");
        args.argAScale(activation_sf);
    }
    // tensor for scaling input B
    if (mOptions.hasSFB)
    {
        void const* weight_sf = inputs[mArgInvMap[TensorArg::IN_WEIGHT_SF]];
        TLLM_CHECK_WITH_INFO(weight_sf != nullptr, "GemmAllReducePlugin weight_sf is NULL");
        args.argBScale(weight_sf);
    }
    // tensor for scaling output D
    if (mOptions.alphaIsPtr)
    {
        void const* alpha_vec = inputs[mArgInvMap[TensorArg::IN_ALPHA]];
        TLLM_CHECK_WITH_INFO(alpha_vec != nullptr, "GemmAllReducePlugin alpha_vec is NULL");
        args.argAlphaPtr(reinterpret_cast<float const*>(alpha_vec));
    }
    else
    {
        args.argAlpha(mOptions.alpha);
    }

    mGemm->run(args, stream);

    return 0;
}

//////////////////////////////////
// IPluginV2Ext Methods
//////////////////////////////////
DataType GemmAllReducePlugin::getOutputDataType(int index, DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK_WITH_INFO(index < getNbOutputs(), "Output index out of bounds: %d", index);
    return mOptions.typeD;
}

//////////////////////////////////
// IPluginV2 Methods
//////////////////////////////////
char const* GemmAllReducePlugin::getPluginType() const noexcept
{
    return GEMM_ALLREDUCE_PLUGIN_NAME;
}

char const* GemmAllReducePlugin::getPluginVersion() const noexcept
{
    return GEMM_ALLREDUCE_PLUGIN_VERSION;
}

int GemmAllReducePlugin::getNbOutputs() const noexcept
{
    return mNbOutputs;
}

int GemmAllReducePlugin::initialize() noexcept
{
    if (isBuilding() && mProfiler->useProfiler())
    {
        // TODO (xsimmons): interfaces between GemmPluginProfiler and Plugin
        // needs to be relooked at - current interface implicitly assigns runner to profiler
        // object in profileTactics()
        assert(mOptions.maxProblemShape.isInitialized());
        mProfiler->profileTactics(mGemm, mOptions.typeD, mOptions.maxProblemShape, mGemmId);
    }
    return 0;
}

void GemmAllReducePlugin::terminate() noexcept
{
    if (isBuilding()) // need this otherwise getComm will crash during build phase
    {
        return;
    }

    // free mWorkspace
    if (mWorkspace)
    {
        getPluginRegistry()->releasePluginResource(mWorkspaceKey.c_str());
        mWorkspace = nullptr;
    }
}

size_t GemmAllReducePlugin::getSerializationSize() const noexcept
{
    // cannot use sizeof(GemmAllReducePluginOptions)
    // becaused need packed attribute which doesn't work on enum
    // without making the enum also packed
    size_t size = 0;
    size += sizeof(mOptions.typeA);
    size += sizeof(mOptions.typeB);
    size += sizeof(mOptions.typeD);
    size += sizeof(mOptions.transA);
    size += sizeof(mOptions.transB);
    size += sizeof(mOptions.alpha);
    size += sizeof(mOptions.maxProblemShape);
    size += sizeof(mOptions.groupSize);
    size += mOptions.group.size() * sizeof(int);
    size += sizeof(mOptions.hasSFA);
    size += sizeof(mOptions.hasSFB);
    size += sizeof(mOptions.alphaIsPtr);
    return size;
}

void GemmAllReducePlugin::serialize(void* buffer) const noexcept
{
    char* begin = reinterpret_cast<char*>(buffer);
    char* end = reinterpret_cast<char*>(buffer);

    write(end, mOptions.typeA);
    write(end, mOptions.typeB);
    write(end, mOptions.typeD);
    write(end, mOptions.transA);
    write(end, mOptions.transB);
    write(end, mOptions.alpha);
    write(end, mOptions.maxProblemShape);
    write(end, mOptions.groupSize);
    for (auto const& rank : mOptions.group)
    {
        write(end, rank);
    }
    write(end, mOptions.hasSFA);
    write(end, mOptions.hasSFB);
    write(end, mOptions.alphaIsPtr);
    TLLM_CHECK(end == begin + getSerializationSize());

    // Profiler MNK->kernel mappings need to be deterministic and consistent across ranks
    // to ensure correct functionality (unlike standalone GEMMs).
    // Since by default each rank will generate and serialize its own profiler mapping
    // this can lead to different mappings between ranks which will result in fatal
    // error. Therefore only generate and use profiler mapping for single rank.
    if (mProfiler->useProfiler() && COMM_SESSION.getRank() == 0)
    {
        mProfiler->serializeToOwnFile(mGemmId);
    }
}

void GemmAllReducePlugin::destroy() noexcept
{
    delete this;
}

////////////////////////////////////////////////////////////
// GemmAllReducePluginCreator Methods
////////////////////////////////////////////////////////////
PluginFieldCollection GemmAllReducePluginCreator::mFC;
std::vector<PluginField> GemmAllReducePluginCreator::mPluginAttributes;

GemmAllReducePluginCreator::GemmAllReducePluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("type_a", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("type_b", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("type_d", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("transa", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("transb", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("alpha", nullptr, PluginFieldType::kFLOAT32, 1);
    mPluginAttributes.emplace_back("group", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("has_sfa", nullptr, PluginFieldType::kINT8, 1);
    mPluginAttributes.emplace_back("has_sfb", nullptr, PluginFieldType::kINT8, 1);
    mPluginAttributes.emplace_back("alpha_is_ptr", nullptr, PluginFieldType::kINT8, 1);
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GemmAllReducePluginCreator::getPluginName() const noexcept
{
    return GEMM_ALLREDUCE_PLUGIN_NAME;
}

char const* GemmAllReducePluginCreator::getPluginVersion() const noexcept
{
    return GEMM_ALLREDUCE_PLUGIN_VERSION;
}

PluginFieldCollection const* GemmAllReducePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmAllReducePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    GemmAllReducePluginOptions options;
    options.deserialize = false;

    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_a"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            options.typeA = *static_cast<DataType const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "type_b"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            options.typeB = *static_cast<DataType const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "type_d"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            options.typeD = *static_cast<DataType const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "transa"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            options.transA = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "transb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            options.transB = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "alpha"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            options.alpha = *static_cast<float const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* ranks = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                options.group.insert(ranks[j]);
            }
            options.groupSize = options.group.size();
        }
        else if (!strcmp(attrName, "has_sfa")) // passed in as input tensor
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            options.hasSFA = *static_cast<int8_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "has_sfb")) // passed in as input tensor
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            options.hasSFB = *static_cast<int8_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "alpha_is_ptr")) // passed in as input tensor
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            options.alphaIsPtr = *static_cast<int8_t const*>(fields[i].data);
        }
    }

    try
    {
        // GemmAllReducePluginCreator is unique and shared for an engine generation
        auto* obj = new GemmAllReducePlugin(options);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return nullptr;
    }
}

IPluginV2* GemmAllReducePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GemmAllReducePlugin::destroy()
    try
    {
        auto* obj = new GemmAllReducePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

} // namespace tensorrt_llm::plugins
