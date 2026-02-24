/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"
#include "RoutingKernel.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <set>
#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{

inline std::set<int32_t> computeSelectedTileN(std::vector<int32_t> const& supported_tile_nums, int64_t const num_tokens,
    int64_t const top_k, int64_t const num_local_experts)
{
    float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
    // assume supported_tile_nums is sorted
    int32_t tile_tokens_dim = std::clamp(
        torch_ext::nextPowerOfTwo(avg_tokens_per_expert), supported_tile_nums.front(), supported_tile_nums.back());
    auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);

    std::set<int32_t> selected_tile_nums;
    selected_tile_nums.insert(tile_tokens_dim);
    if (std::next(it) != supported_tile_nums.end())
    {
        selected_tile_nums.insert(*std::next(it));
        if (std::next(std::next(it)) != supported_tile_nums.end())
        {
            selected_tile_nums.insert(*std::next(std::next(it)));
        }
    }
    if (it != supported_tile_nums.begin())
    {
        selected_tile_nums.insert(*std::prev(it));
    }

    return selected_tile_nums;
}

namespace Routing
{

// The type of method in top-K routing, for use in torch custom op
// Please keep this in sync with the counterpart defined in tensorrt_llm/_torch/modules/fused_moe/routing.py
enum class RoutingMethodType : int64_t
{
    // Default: Softmax -> TopK
    Default = 0,
    // Renormalize: TopK -> Softmax
    Renormalize = 1,
    // DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = 2,
    // Llama4: Top1 -> Sigmoid
    Llama4 = 3,
    // RenormalizeNaive: Softmax -> TopK -> Renormalize
    RenormalizeNaive = 4,
    // MiniMaxM2: Sigmoid -> RoutingBiasAdd -> TopK -> Renormalize(without bias)
    MiniMax2 = 5,
    // Unspecified
    Unspecified = 6,
};

inline int32_t maybeGetMinTokenCount(int32_t numPaddedTokens, int32_t hiddenSize, int32_t dtypeSizeBits)
{
    // Pad so total size exceeds 128KiB for performance reasons
    int32_t minNumTokensRequired = common::divUp(128 * 1024 * 8, hiddenSize * dtypeSizeBits);
    return std::max(numPaddedTokens, minNumTokensRequired);
}

inline std::string serializeMoeRoutingMethodType(RoutingMethodType routingMethodType)
{
    switch (routingMethodType)
    {
    case RoutingMethodType::Default: return "Default";
    case RoutingMethodType::Renormalize: return "Renormalize";
    case RoutingMethodType::DeepSeekV3: return "DeepSeekV3";
    case RoutingMethodType::Llama4: return "Llama4";
    case RoutingMethodType::RenormalizeNaive: return "RenormalizeNaive";
    case RoutingMethodType::MiniMax2: return "MiniMax2";
    default: TLLM_CHECK_WITH_INFO(false, "Invalid routing method"); return "";
    };
}

inline int32_t getMaxNumCtasInBatchDim(int32_t numTokens, int32_t topK, int32_t numExperts, int32_t tileTokensDim)
{
    // For MoE, mNumTokens != 0 and the number of CTAs is known only at runtime.
    // We launch maximally possible number of CTAs and use ptrNumNonExitingCtas to determine
    // the actual number of CTAs to run.

    // Initialize number of tokens with the number of expanded tokens after routing.
    int32_t numRemainingTokens = numTokens * topK;
    int32_t maxNumCtasInBatchDim = 0;
    // First, distribute one token each expert until token depletion to maximize CTA tile count.
    int32_t numExpertsFilled = std::min(numExperts, numRemainingTokens);
    maxNumCtasInBatchDim += numExpertsFilled;
    numRemainingTokens -= numExpertsFilled;
    // Next, greedily pour all remaining tokens to one expert to maximize CTA tile count.
    // E.g., at this point tokens over 4 experts are [1, 1, 1, 1], and we have 4 tokens left.
    // If each CTA handles 4 tokens/expert, the greedy strategy is to pour all remaining tokens
    // to any one expert to get to the 5th CTA tile. Otherwise, we can only get 4 tiles in total.
    //
    // Another way to reason about this is to pour the remaining tokens into buckets of some fixed
    // capacity. These buckets, if full, can then be attributed to any expert; it does not have to
    // belong to the same expert every time.
    if (numRemainingTokens > 0)
    {
        // For every tileTokenDim tokens, we add an extra CTA tile in the token dimension.
        // The number of CTA tiles is given by divDown(numRemainingTokens, tokenTileDim).
        maxNumCtasInBatchDim += (numRemainingTokens / tileTokensDim);
    }
    return maxNumCtasInBatchDim;
}

inline int32_t getMaxPermutedPaddedCount(
    int32_t numTokens, int32_t expertsPerToken, int32_t numExperts, int32_t padding)
{
    int32_t maxCtas = getMaxNumCtasInBatchDim(numTokens, expertsPerToken, numExperts, padding);
    return maxCtas * padding;
}

class Runner
{
public:
    explicit Runner();

    explicit Runner(int32_t tileTokensDim);

    void run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts, int32_t topK,
        int32_t nGroups, int32_t topkGroups, int32_t localExpertOffset, int32_t localNumExperts,
        float routedScalingFactor, int32_t* routingExpertIndexes, int32_t* expertCountHistogram,
        int32_t* permutedIdxSize, int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx,
        int32_t* permutedIdxToTokenIdx, void* expertWeights, int32_t* expertIds, int32_t* numTokensPerExpert,
        int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas,
        batchedGemm::trtllm::gen::Dtype dtypeElt, bool useRoutingScalesOnInput, bool useDeepSeekFp8,
        RoutingMethodType routingMethodType, cudaStream_t stream);

private:
    int32_t mTileTokensDim;
};
} // namespace Routing

namespace PermuteGemm1
{
class Runner
{
public:
    explicit Runner(batchedGemm::trtllm::gen::Dtype dtypeAct, batchedGemm::trtllm::gen::Dtype dtypeWeights,
        bool useDeepSeekFp8, int tileTokensDim, ActType actType);

    size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
        int32_t numTokens, int32_t configIndex) const;

    [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
        int32_t numExperts, int32_t numTokens, int32_t validHiddenSize = -1, int32_t validIntermediateSize = -1) const;

    [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
        int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t validHiddenSize = -1,
        int32_t validIntermediateSize = -1) const;

    [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

    [[nodiscard]] std::string getKernelNameFromConfigIndex(int32_t configIndex) const;

    void run(void* hiddenState, void* hiddenStateScale, void* weight, void* weightScale, void* expertWeights,
        float* outputScalesScalar, float* outputScalesGateScalar, float* ptrBias, float* ptrSwiGluAlpha,
        float* ptrSwiGluBeta, float* ptrClampLimit, void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
        int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx,
        int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
        int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace, bool useRoutingScalesOnInput, int device,
        cudaStream_t stream, int32_t configIndex, int32_t validHiddenSize = -1, int32_t validIntermediateSize = -1);

private:
    batchedGemm::trtllm::gen::Dtype mDtypeAct;
    batchedGemm::trtllm::gen::Dtype mDtypeWeights;
    int32_t mTileTokensDim;
    ActType mActType;
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner mRunner;
};
} // namespace PermuteGemm1

namespace Gemm2
{
class Runner
{
public:
    explicit Runner(batchedGemm::trtllm::gen::Dtype dtypeAct, batchedGemm::trtllm::gen::Dtype dtypeWeights,
        batchedGemm::trtllm::gen::Dtype outputDtype, bool useDeepSeekFp8, int tileTokensDim);

    size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
        int32_t numTokens, int32_t configIndex) const;

    [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
        int32_t numExperts, int32_t numTokens, int32_t validHiddenSize = -1, int32_t validIntermediateSize = -1) const;

    [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
        int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t validHiddenSize = -1,
        int32_t validIntermediateSize = -1) const;

    [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

    [[nodiscard]] std::string getKernelNameFromConfigIndex(int32_t configIndex) const;

    void run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weight, void* weightScale,
        float* outputScalesScalar, float* ptrBias, void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
        int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* ptrNumNonExitingCtas,
        int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
        void* bmm2Workspace, int device, cudaStream_t stream, int32_t configIndex, int32_t validHiddenSize = -1,
        int32_t validIntermediateSize = -1);

private:
    batchedGemm::trtllm::gen::Dtype mDtypeAct;
    batchedGemm::trtllm::gen::Dtype mDtypeWeights;
    batchedGemm::trtllm::gen::Dtype mDtypeOut;
    int32_t mTileTokensDim;
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner mRunner;
};
} // namespace Gemm2

namespace MoE
{
namespace btg = batchedGemm::trtllm::gen;

struct MoERunnerArgs
{
    void* routing_logits
        = nullptr; // [num_tokens, num_experts] in float, generated after gemm(hidden_state, routing_weights)
    void* routing_bias = nullptr;  // [num_experts] in bfloat16 for now = mDtypeExpW
    void* hidden_states = nullptr; // [num_tokens, hidden_size] in fp8 = mDtypeElt
    // [hidden_size/128, num_tokens] in float for e4m3 DS recipe
    // and [num_tokens, hidden_size/16] in float for e2m1
    void* hidden_states_scale = nullptr;

    // Optional inputs:
    void* topk_weights = nullptr; // [num_tokens, top_k]  with quantized weights
    int32_t* topk_ids = nullptr;  // [num_tokens, top_k] with expert ids in int32_t

    // Gemm input:
    void* gemm1_weights = nullptr;
    void* gemm1_weights_scale = nullptr;
    void* gemm2_weights = nullptr;
    void* gemm2_weights_scale = nullptr;

    float* gemm1_bias = nullptr;
    float* gemm1_alpha = nullptr;
    float* gemm1_beta = nullptr;
    float* gemm1_clamp_limit = nullptr;
    float* gemm2_bias = nullptr;

    int32_t num_tokens{0};
    int32_t num_experts{0};
    // Hidden dimension input of MoE block. It might be padded.
    int32_t hidden_size{0};
    // Hidden dimension output of MoE block. It might be padded.
    std::optional<int32_t> output_hidden_size{std::nullopt};
    // TODO: only compiled routing kernel supports top_k = 8
    int32_t top_k{0};
    int32_t n_group{0};
    // TODO: only compiled routing kernel supports topk_group = 4
    int32_t topk_group{0};
    float routed_scaling_factor{0.0f};
    // Intermediate dimension output of MoE block. It might be padded.
    int32_t intermediate_size{0};
    int32_t local_expert_offset{0};
    int32_t local_num_experts{0};
    // TODO: support other types
    btg::Dtype mDtypeElt{btg::Dtype::Void};
    btg::Dtype mDtypeExpW{btg::Dtype::Bfloat16};
    btg::Dtype mDtypeOut{btg::Dtype::Bfloat16};
    // Unpadded dimensions.
    std::optional<int32_t> valid_intermediate_size{std::nullopt};
    std::optional<int32_t> valid_hidden_size{std::nullopt};

    // Apply routing scale factors to input activations
    bool mUseRoutingScalesOnInput{false};
    bool mUseDeepSeekFp8{false};
    float* output1_scales_scalar = nullptr;
    float* output1_scales_gate_scalar = nullptr;
    float* output2_scales_scalar = nullptr;

    // Output:
    void* output = nullptr;
    float* output_scale = nullptr;

    // finalize
    bool do_finalize{true};
};

struct MoEWorkspace
{
    // Routing intermediate outputs:
    int32_t* routing_expert_indexes = nullptr;
    int32_t* permuted_idx_size = nullptr;
    int32_t* total_num_padded_tokens = nullptr; // TODO: duplicate of permuted_idx_size
    int32_t total_max_padded_tokens{0};

    int32_t* expanded_idx_to_permuted_idx = nullptr;
    int32_t* permuted_idx_to_expanded_idx = nullptr;
    int32_t* permuted_idx_to_token_idx = nullptr;
    void* expert_weights = nullptr; // [num_tokens, top_k] in bfloat16 = mDtypeExpW

    int32_t* cta_idx_xy_to_batch_idx = nullptr;
    int32_t* cta_idx_xy_to_mn_limit = nullptr;
    int32_t* num_non_exiting_ctas = nullptr;

    void* hidden_states_scale_linear = nullptr;

    // Permute intermediate outputs:
    void* permuted_hidden_states = nullptr;
    float* permuted_hidden_states_scale = nullptr;

    // Gemm1 intermediate outputs:
    int32_t ProjUpTileN{0};
    void* gemm1_output = nullptr;
    float* gemm1_output_scale = nullptr;

    // Activation intermediate outputs:
    void* activation_output = nullptr;
    float* activation_output_scale = nullptr;

    // Gemm2 intermediate outputs:
    void* gemm2_output = nullptr;
    float* gemm2_output_scale = nullptr;

    // Finalize intermediate outputs (placeholder not used)
    void* finalize_output = nullptr;
    float* finalize_output_scale = nullptr;

    // FC1 workspace:
    void* bmm1_workspace = nullptr;

    // FC2 workspace:
    void* bmm2_workspace = nullptr;
};

// Config indices to be used with Batched GEMM runners
struct MoEConfig
{
    int64_t gemm1Config;
    int64_t gemm2Config;
};

class Runner
{
public:
    // FIXME: tileTokensDim is hardcoded for now
    Runner(batchedGemm::trtllm::gen::Dtype dtypeAct, batchedGemm::trtllm::gen::Dtype dtypeWeights, bool useDeepSeekFp8,
        int tileTokensDim = 8, ActType actType = ActType::SwiGlu);
    Runner(batchedGemm::trtllm::gen::Dtype dtypeElt, bool useDeepSeekFp8, int tileTokensDim = 8);

    void run(
        MoERunnerArgs const& args, MoEWorkspace const& workspace, int device, cudaStream_t stream, int64_t configIndex);

    [[nodiscard]] std::tuple<int32_t, int32_t> getWorkspaceSizeInBytes(
        MoERunnerArgs const& args, int64_t configIndex) const;

    [[nodiscard]] std::vector<int64_t> getValidConfigIndices(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
        int32_t numLocalExperts, int32_t numTokens, int32_t validHiddenSize = -1,
        int32_t validIntermediateSize = -1) const;

    [[nodiscard]] int64_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
        int32_t numLocalExperts, int32_t numTokens, int32_t validHiddenSize = -1,
        int32_t validIntermediateSize = -1) const;

private:
    void setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace, moe::dev::convertsf::Data& convertSfData,
        moe::dev::activation::Data& activationData, moe::dev::finalize::Data& finalizeData);

private:
    PermuteGemm1::Runner mPermuteGemm1;
    Gemm2::Runner mGemm2;
    ActType mActType;

    // This will be the cartesian product of the passing configs for gemm1 and gemm2
    // This allows us to autotune the MoE as one operation instead of tuning gemm1 and gemm2 separately
    std::vector<MoEConfig> mPassingConfigs;
};
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels

TRTLLM_NAMESPACE_END
