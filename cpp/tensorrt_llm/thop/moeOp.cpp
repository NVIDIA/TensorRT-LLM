/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#if defined(USING_OSS_CUTLASS_MOE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#else
#include "moe_gemm_kernels.h"
#include "moe_kernels.h"
#endif
// Always include the public header for moe_gemm_kernels.h
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"

#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/cutlass_kernel_selector.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/native/cuda/Resize.h>

#include <functional>
#include <map>

#define C10_THROW_ERROR_FORMATTED(ErrorType, ...)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        std::ostringstream oss;                                                                                        \
        oss << __VA_ARGS__;                                                                                            \
        C10_THROW_ERROR(ErrorType, oss.str());                                                                         \
    } while (0)

namespace torch_ext
{

namespace common = tensorrt_llm::common;
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using ActivationParams = CUTLASS_MOE_GEMM_NAMESPACE::ActivationParams;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
using MoeGemmId = CUTLASS_MOE_GEMM_NAMESPACE::MoeGemmId;
// Always use public header as it is just utility functions and types
using TmaWarpSpecializedGroupedGemmInput = tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using profiler_backend = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::GemmProfilerBackend;

class FusedMoeRunner : public torch::CustomClassHolder
{
public:
    template <typename TypeAct, typename TypeWeight, bool NeedQuant = false>
    std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> switch_output_type(c10::ScalarType output_type)
    {
        switch (output_type)
        {
        case c10::ScalarType::Long: // INT64 == FP4
        case c10::ScalarType::Float8_e4m3fn:
            // TODO We need an atomic FP8 reduction for the finalize fusions
            C10_THROW_ERROR_FORMATTED(NotImplementedError,
                "Outputting " << torch::toString(output_type) << " directly is not currently supported");
            // return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type>>();
        case c10::ScalarType::Half:
            if constexpr (NeedQuant)
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, half>>();
            }
            else
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, TypeAct>>();
            }
#ifdef ENABLE_BF16
        case c10::ScalarType::BFloat16:
            if constexpr (NeedQuant)
            {
                return std::make_unique<
                    kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, __nv_bfloat16>>();
            }
            else
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, TypeAct>>();
            }
#endif
        default:
            C10_THROW_ERROR_FORMATTED(Error,
                "Invalid output type " << torch::toString(output_type) << " specified for "
                                       << torch::toString(mActivationDtype));
        }
    };

    template <typename TypeAct>
    std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> create_weight_quant_runner()
    {
        if (isInt8Quant())
        {
            return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, uint8_t>>();
        }
        else if (isInt4Quant())
        {
#ifdef ENABLE_FP8
            if (mUseW4GroupScaling)
            {
                return std::make_unique<
                    kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, TypeAct, TypeAct>>();
            }
#endif
            return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, cutlass::uint4b_t>>();
        }
        else
        {
            C10_THROW_ERROR_FORMATTED(Error, "Unsupported weight quantization type");
        }
    }

    FusedMoeRunner(c10::ScalarType activation_dtype, c10::ScalarType weight_dtype, c10::ScalarType output_dtype,
        bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling, bool use_int8_woq_per_channel,
        bool use_mxfp8_act_scaling, bool use_fused_finalize)
    {
        mActivationDtype = activation_dtype;
        mWeightDtype = weight_dtype;
        mOutputDtype = output_dtype;
        mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
        mUseW4GroupScaling = use_w4_group_scaling;
        mUseINT8WoqPerChannel = use_int8_woq_per_channel;
        mUseMxfp8ActScaling = use_mxfp8_act_scaling;
        mUseFusedFinalize = use_fused_finalize;
        mInnerDimMultiplier = 1;

        // keep consistent with cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp
        if (mActivationDtype == c10::ScalarType::Half && mWeightDtype == c10::ScalarType::Half)
        {
            mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, half>>();
        }
#ifdef ENABLE_BF16
        else if (mActivationDtype == c10::ScalarType::BFloat16 && mWeightDtype == c10::ScalarType::BFloat16)
        {
            mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
        }
#ifdef ENABLE_FP8
        else if (mActivationDtype == c10::ScalarType::BFloat16 && mWeightDtype == c10::ScalarType::Float8_e4m3fn)
        {
            mKernelRunner = std::make_unique<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3>>();
        }
#endif
#endif

#ifdef ENABLE_FP8
        if (isFp8Quant())
        {
            mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp8_e4m3>(mOutputDtype);
        }
#endif
#ifdef ENABLE_FP4
        if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant())
        {
            mInnerDimMultiplier = 16; // 16 FP4 -> 1 LONG
            mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp4_e2m1>(mOutputDtype);
        }
        if (isNvfp4Quant())
        {
            mInnerDimMultiplier = 16; // 16 FP4 -> 1 LONG
            switch (mActivationDtype)
            {
            case c10::ScalarType::Half:
#ifdef ENABLE_BF16
            case c10::ScalarType::BFloat16:
#endif
                mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, true>(mOutputDtype);
                break;
            default: mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, false>(mOutputDtype);
            }
        }
        if (isWFP4A16Quant())
        {
            mInnerDimMultiplier = 2;
            if (mActivationDtype == c10::ScalarType::Half)
            {
                mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, __nv_fp4_e2m1>>();
            }
#ifdef ENABLE_BF16
            else if (mActivationDtype == c10::ScalarType::BFloat16)
            {
                mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1>>();
            }
#endif
        }
#endif
        if (isIntWeightOnlyQuant())
        {
            if (isInt4Quant())
            {
                mInnerDimMultiplier = 2; // 2 INT4 -> 1 INT8
            }
            switch (mActivationDtype)
            {
#ifdef ENABLE_FP8
            case c10::ScalarType::Float8_e4m3fn:
            {
                if (isInt4Quant() and mUseW4GroupScaling)
                {
                    mKernelRunner = std::make_unique<
                        kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_fp8_e4m3>>();
                }
                else
                {
                    C10_THROW_ERROR_FORMATTED(Error, "FP8 activation type is not supported for non-W4A8 quantization");
                }
                break;
            }
#endif
            case c10::ScalarType::Half: mKernelRunner = create_weight_quant_runner<half>(); break;
            case c10::ScalarType::BFloat16: mKernelRunner = create_weight_quant_runner<__nv_bfloat16>(); break;
            default: C10_THROW_ERROR_FORMATTED(Error, "Unsupported activation type for int-type weight");
            }
        }
        if (!mKernelRunner)
        {
            C10_THROW_ERROR_FORMATTED(Error,
                "Could not construct fused moe op with the requested input combination Activation: "
                    << torch::toString(mActivationDtype) << ", Weight: " << torch::toString(mWeightDtype)
                    << ", Output: " << torch::toString(mOutputDtype));
        }

        mKernelRunner->use_fused_finalize_ = mUseFusedFinalize;

        mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
        mGemm1Profiles = mKernelRunner->getTactics(MoeGemmId::GEMM_1);
        mGemm2Profiles = mKernelRunner->getTactics(MoeGemmId::GEMM_2);
    }

    ~FusedMoeRunner()
    {
        if (mProfileWorkspace != nullptr)
        {
            auto const cu_free_status = cudaFree(mProfileWorkspace);
            TORCH_CHECK(
                cu_free_status == cudaSuccess, "Can't free profile workspace during FusedMoeRunner destruction.");
        }
    }

    FusedMoeRunner(FusedMoeRunner const&) = delete;
    void operator=(FusedMoeRunner const&) = delete;

    torch::Tensor runMoe(torch::Tensor const& input, torch::Tensor const& token_selected_experts,
        torch::optional<torch::Tensor> const& token_final_scales, torch::Tensor const& fc1_expert_weights,
        torch::optional<torch::Tensor> const& fc1_expert_biases, torch::Tensor const& fc2_expert_weights,
        torch::optional<torch::Tensor> const& fc2_expert_biases,
        torch::optional<c10::ArrayRef<torch::Tensor>> const& quant_scales,
        torch::optional<torch::Tensor> const& input_sf, bool const swizzled_input_sf,
        torch::optional<torch::Tensor> const& swiglu_alpha, torch::optional<torch::Tensor> const& swiglu_beta,
        torch::optional<torch::Tensor> const& swiglu_limit, int64_t const tp_size, int64_t const tp_rank,
        int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size, int64_t const cluster_rank,
        bool const enable_alltoall, bool min_latency_mode, torch::optional<c10::ArrayRef<int64_t>> const& profile_ids,
        torch::optional<int64_t> const& unpadded_hidden_size)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        // Free the profile workspace to save memory
        freeProfileWorkspace();

        TORCH_CHECK(cluster_size == 1 && cluster_rank == 0, "smart_router is supported in min_latency mode");

        CHECK_INPUT(input, mActivationDtype)
        CHECK_INPUT(token_selected_experts, at::ScalarType::Int)
        if (token_final_scales)
        {
            CHECK_INPUT(token_final_scales.value(), at::ScalarType::Float)
        }
        CHECK_INPUT(fc1_expert_weights, mWeightDtype)
        CHECK_INPUT(fc2_expert_weights, mWeightDtype)

        TORCH_CHECK(input.dim() == 2, "input must be 2D.");
        TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");

        TORCH_CHECK(fc1_expert_weights.dim() == 3, "fc1_expert_weights must be 3D.");
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");

        if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value())
        {
            CHECK_INPUT(fc1_expert_biases.value(), mOutputDtype);
            CHECK_INPUT(fc2_expert_biases.value(), mOutputDtype);
            TORCH_CHECK(fc1_expert_biases.value().dim() == 2, "fc1_expert_biases must be 2D.");
            TORCH_CHECK(fc2_expert_biases.value().dim() == 2, "fc2_expert_biases must be 2D.");
            TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc1_expert_biases.value().sizes()[0],
                "fc1_expert_weights and fc1_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc2_expert_weights.sizes()[0] == fc2_expert_biases.value().sizes()[0],
                "fc2_expert_weights and fc2_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc1_expert_biases.value().sizes()[1] == fc1_expert_weights.sizes()[1],
                "fc1_expert_biases should match fc1_expert_weights output shape.");
            TORCH_CHECK(fc2_expert_biases.value().sizes()[1] == fc2_expert_weights.sizes()[1],
                "fc2_expert_biases should match fc2_expert_weights output shape.");
        }

        if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value())
        {
            CHECK_INPUT(fc1_expert_biases.value(), mOutputDtype);
            CHECK_INPUT(fc2_expert_biases.value(), mOutputDtype);
            TORCH_CHECK(fc1_expert_biases.value().dim() == 2, "fc1_expert_biases must be 2D.");
            TORCH_CHECK(fc2_expert_biases.value().dim() == 2, "fc2_expert_biases must be 2D.");
            TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc1_expert_biases.value().sizes()[0],
                "fc1_expert_weights and fc1_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc2_expert_weights.sizes()[0] == fc2_expert_biases.value().sizes()[0],
                "fc2_expert_weights and fc2_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc1_expert_biases.value().sizes()[1] == fc1_expert_weights.sizes()[1],
                "fc1_expert_biases should match fc1_expert_weights output shape.");
            TORCH_CHECK(fc2_expert_biases.value().sizes()[1] == fc2_expert_weights.sizes()[1],
                "fc2_expert_biases should match fc2_expert_weights output shape.");
        }

        TORCH_CHECK(input.sizes()[0] == token_selected_experts.sizes()[0],
            "input and token_selected_experts must have the same num tokens.");
        if (token_final_scales)
        {
            TORCH_CHECK(token_final_scales.value().dim() == 2, "token_selected_experts_probs must be 2D.");
            TORCH_CHECK(input.sizes()[0] == token_final_scales.value().sizes()[0],
                "input and token_selected_experts_probs must have the same num tokens.");
            TORCH_CHECK(token_selected_experts.sizes()[1] == token_final_scales.value().sizes()[1],
                "token_selected_experts and token_final_scales must have the same number of experts per token.");
        }
        TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc2_expert_weights.sizes()[0],
            "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");

        if (mUseINT8WoqPerChannel)
        {
            // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
            // [num_experts, inter_size, hidden_size]
            TORCH_CHECK(fc1_expert_weights.sizes()[2] == fc2_expert_weights.sizes()[1] * mInnerDimMultiplier * 2,
                "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");
        }
        else
        {
            TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * mInnerDimMultiplier * 2,
                "fc1_expert_weights inter size must be fc2_expert_weights inter size.");
        }

        int experts_per_token = token_selected_experts.sizes()[1];
        int64_t num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t unpadded_hidden_size_val
            = unpadded_hidden_size.has_value() ? unpadded_hidden_size.value() : hidden_size;
        int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        if (mUseINT8WoqPerChannel)
        {
            // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
            // [num_experts, inter_size, hidden_size]
            hidden_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
            inter_size = fc2_expert_weights.sizes()[1];
        }

        if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant())
        {
            // MXFP4 weights are required to bealigned to 128 bytes
            TORCH_CHECK(hidden_size % 128 == 0, "hidden_size must be divisible by 128 for MXFP4 weights");
            TORCH_CHECK(inter_size % 128 == 0, "inter_size must be divisible by 128 for MXFP4 weights");
        }
        else
        {
            // TMA requires at least 128 bit alignment
            auto min_alignment
                = 128 / (8 * std::min(c10::elementSize(mActivationDtype), c10::elementSize(mWeightDtype)));
            TORCH_CHECK(hidden_size % min_alignment == 0, "hidden_size ", hidden_size, " must be divisible by ",
                min_alignment, " for weights");
            TORCH_CHECK(inter_size % min_alignment == 0, "inter_size ", inter_size, " must be divisible by ",
                min_alignment, " for weights");
        }

        int const num_experts_on_rank = fc2_expert_weights.sizes()[0];
        auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
        ActivationType base_activation_type = ActivationType::Swiglu;
        if (swiglu_alpha.has_value())
        {
            CHECK_INPUT(swiglu_alpha.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_alpha.value().sizes()[0] == num_experts_on_rank,
                "swiglu_alpha must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        if (swiglu_beta.has_value())
        {
            CHECK_INPUT(swiglu_beta.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_beta.value().sizes()[0] == num_experts_on_rank,
                "swiglu_beta must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        if (swiglu_limit.has_value())
        {
            CHECK_INPUT(swiglu_limit.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_limit.value().sizes()[0] == num_experts_on_rank,
                "swiglu_limit must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        auto activation_params = ActivationParams(base_activation_type,
            reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().const_data_ptr() : nullptr),
            reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().const_data_ptr() : nullptr),
            reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().const_data_ptr() : nullptr));

        setRunnerProfiles(profile_ids);

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        WorkspaceInfo const& workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), base_activation_type, parallelism_config, min_latency_mode, stream);

        // output is smaller than workspace. Create output after workspace to avoid output_shape occupied a little
        // piece of memory which makes a big partition of memory segment can't be used by workspace.
        std::vector<int64_t> output_shape = {num_rows, unpadded_hidden_size_val};
        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));

        auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
        kernels::MoeMinLatencyParams min_latency_params{};

        // TODO: support lora in the future
        ::tensorrt_llm::kernels::LoraParams lora_params{};
#ifdef USING_OSS_CUTLASS_MOE_GEMM
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr, swizzled_input_sf,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(),
            fc1_expert_biases.has_value() ? fc1_expert_biases.value().const_data_ptr() : nullptr, activation_params,
            fc2_expert_weights.const_data_ptr(),
            fc2_expert_biases.has_value() ? fc2_expert_biases.value().const_data_ptr() : nullptr, quant_params,
            num_rows, hidden_size, unpadded_hidden_size_val, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace.data_ptr()),
            output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall,
            false, lora_params, mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
#else
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr, swizzled_input_sf,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(),
            fc1_expert_biases.has_value() ? fc1_expert_biases.value().const_data_ptr() : nullptr, activation_params,
            fc2_expert_weights.const_data_ptr(),
            fc2_expert_biases.has_value() ? fc2_expert_biases.value().const_data_ptr() : nullptr, quant_params,
            num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
            mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
#endif

        return output;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> runMoeMinLantency(torch::Tensor const& input,
        torch::Tensor const& token_selected_experts, torch::optional<torch::Tensor> const& token_final_scales,
        torch::Tensor const& fc1_expert_weights, torch::optional<torch::Tensor> const& fc1_expert_biases,
        torch::Tensor const& fc2_expert_weights, torch::optional<torch::Tensor> const& fc2_expert_biases,
        torch::optional<c10::ArrayRef<torch::Tensor>> const& quant_scales,
        torch::optional<torch::Tensor> const& input_sf, bool const swizzled_input_sf,
        torch::optional<torch::Tensor> const& swiglu_alpha, torch::optional<torch::Tensor> const& swiglu_beta,
        torch::optional<torch::Tensor> const& swiglu_limit, int64_t const tp_size, int64_t const tp_rank,
        int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size, int64_t const cluster_rank,
        bool const enable_alltoall, bool min_latency_mode, torch::optional<c10::ArrayRef<int64_t>> const& profile_ids,
        torch::optional<int64_t> const& unpadded_hidden_size)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        // Free the profile workspace to save memory
        freeProfileWorkspace();

        CHECK_INPUT(input, mActivationDtype)
        CHECK_INPUT(token_selected_experts, at::ScalarType::Int)
        if (token_final_scales)
        {
            CHECK_INPUT(token_final_scales.value(), at::ScalarType::Float)
        }
        CHECK_INPUT(fc1_expert_weights, mWeightDtype)
        CHECK_INPUT(fc2_expert_weights, mWeightDtype)

        TORCH_CHECK(input.dim() == 2, "input must be 2D.");
        TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");

        TORCH_CHECK(fc1_expert_weights.dim() == 3, "fc1_expert_weights must be 3D.");
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");

        if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value())
        {
            CHECK_INPUT(fc1_expert_biases.value(), mOutputDtype);
            CHECK_INPUT(fc2_expert_biases.value(), mOutputDtype);
            TORCH_CHECK(fc1_expert_biases.value().dim() == 2, "fc1_expert_biases must be 2D.");
            TORCH_CHECK(fc2_expert_biases.value().dim() == 2, "fc2_expert_biases must be 2D.");
            TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc1_expert_biases.value().sizes()[0],
                "fc1_expert_weights and fc1_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc2_expert_weights.sizes()[0] == fc2_expert_biases.value().sizes()[0],
                "fc2_expert_weights and fc2_expert_biases must have the same number of experts.");
            TORCH_CHECK(fc1_expert_biases.value().sizes()[1] == fc1_expert_weights.sizes()[1],
                "fc1_expert_biases should match fc1_expert_weights output shape.");
            TORCH_CHECK(fc2_expert_biases.value().sizes()[1] == fc2_expert_weights.sizes()[1],
                "fc2_expert_biases should match fc2_expert_weights output shape.");
        }

        TORCH_CHECK(input.sizes()[0] == token_selected_experts.sizes()[0],
            "input and token_selected_experts must have the same num tokens.");
        if (token_final_scales)
        {
            TORCH_CHECK(token_final_scales.value().dim() == 2, "token_selected_experts_probs must be 2D.");
            TORCH_CHECK(input.sizes()[0] == token_final_scales.value().sizes()[0],
                "input and token_selected_experts_probs must have the same num tokens.");
            TORCH_CHECK(token_selected_experts.sizes()[1] == token_final_scales.value().sizes()[1],
                "token_selected_experts and token_final_scales must have the same number of experts per token.");
        }
        TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc2_expert_weights.sizes()[0],
            "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
        TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * mInnerDimMultiplier * 2,
            "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");

        TORCH_CHECK(!input_sf.has_value() || isWMxfp4AMxfp8Quant() || isNvfp4Quant(),
            "Block-scaling factors provided for non block-scaling quantization");

        int experts_per_token = token_selected_experts.sizes()[1];
        int64_t num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t unpadded_hidden_size_val
            = unpadded_hidden_size.has_value() ? unpadded_hidden_size.value() : hidden_size;
        int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        int const num_experts_on_rank = fc2_expert_weights.sizes()[0];
        auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
        auto parallelism_config
            = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank, cluster_size, cluster_rank);
        ActivationType base_activation_type = ActivationType::Swiglu;
        if (swiglu_alpha.has_value())
        {
            CHECK_INPUT(swiglu_alpha.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_alpha.value().sizes()[0] == num_experts_on_rank,
                "swiglu_alpha must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        if (swiglu_beta.has_value())
        {
            CHECK_INPUT(swiglu_beta.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_beta.value().sizes()[0] == num_experts_on_rank,
                "swiglu_beta must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        if (swiglu_limit.has_value())
        {
            CHECK_INPUT(swiglu_limit.value(), at::ScalarType::Float);
            TORCH_CHECK(swiglu_limit.value().sizes()[0] == num_experts_on_rank,
                "swiglu_limit must have num_experts_on_rank elements.");
            base_activation_type = ActivationType::SwigluBias;
        }
        auto activation_params = ActivationParams(base_activation_type,
            reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().const_data_ptr() : nullptr),
            reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().const_data_ptr() : nullptr),
            reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().const_data_ptr() : nullptr));

        setRunnerProfiles(profile_ids);

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        std::vector<int64_t> output_shape = {num_rows * num_experts_on_rank, unpadded_hidden_size_val};
        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));

        auto num_active_experts_per_node = torch::empty({1}, input.options().dtype(at::ScalarType::Int));
        auto experts_to_token_score
            = torch::empty({num_experts_on_rank, num_rows}, input.options().dtype(at::ScalarType::Float));
        auto active_expert_global_ids = torch::empty({num_experts_on_rank}, input.options().dtype(at::ScalarType::Int));

        kernels::MoeMinLatencyParams min_latency_params{};
        min_latency_params.num_active_experts_per_node = static_cast<int*>(num_active_experts_per_node.data_ptr());
        min_latency_params.experts_to_token_score = static_cast<float*>(experts_to_token_score.data_ptr());
        min_latency_params.active_expert_global_ids = static_cast<int*>(active_expert_global_ids.data_ptr());

        WorkspaceInfo const& workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), base_activation_type, parallelism_config, min_latency_mode, stream);

        auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);

        // TODO: support lora in the future
        ::tensorrt_llm::kernels::LoraParams lora_params{};
#ifdef USING_OSS_CUTLASS_MOE_GEMM
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr, swizzled_input_sf,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(),
            fc1_expert_biases.has_value() ? fc1_expert_biases.value().const_data_ptr() : nullptr, activation_params,
            fc2_expert_weights.const_data_ptr(),
            fc2_expert_biases.has_value() ? fc2_expert_biases.value().const_data_ptr() : nullptr, quant_params,
            num_rows, hidden_size, unpadded_hidden_size_val, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace.data_ptr()),
            output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall,
            false, lora_params, mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
#else
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr, swizzled_input_sf,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(),
            fc1_expert_biases.has_value() ? fc1_expert_biases.value().const_data_ptr() : nullptr, activation_params,
            fc2_expert_weights.const_data_ptr(),
            fc2_expert_biases.has_value() ? fc2_expert_biases.value().const_data_ptr() : nullptr, quant_params,
            num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
            mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
#endif

        return std::make_tuple(output, num_active_experts_per_node, experts_to_token_score, active_expert_global_ids);
    }

    int64_t getTacticNum(int64_t const gemm_idx)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        TORCH_CHECK(gemm_idx == 1 || gemm_idx == 2, "gemm_idx must be 1 or 2");
        return (gemm_idx == 1) ? mGemm1Profiles.size() : mGemm2Profiles.size();
    }

    // TODO Update this to be able to tell if we are profiling swiglu bias
    void runGemmProfile(torch::Tensor const& input, torch::Tensor const& fc1_expert_weights,
        torch::optional<torch::Tensor> const& fc1_expert_biases, torch::Tensor const& fc2_expert_weights,
        torch::optional<torch::Tensor> const& fc2_expert_biases, int64_t const top_k, int64_t const tp_size,
        int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
        int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode, int64_t const gemm_idx,
        int64_t const profile_id, bool const do_preparation, int64_t const unpadded_hidden_size)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        // TODO: support profiling under fp8 block scaling in the future
        if (mUseDeepSeekFP8BlockScaling)
        {
            return;
        }

        int64_t const num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        if (mUseINT8WoqPerChannel)
        {
            // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
            // [num_experts, inter_size, hidden_size]
            hidden_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
            inter_size = fc2_expert_weights.sizes()[1];
        }
        int64_t const group_size_
            = isInt4Quant() ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size : -1;
        int64_t const group_size = isWFP4A16Quant()
            ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
            : group_size_;
        int const num_experts = static_cast<int>(fc2_expert_weights.sizes()[0] * ep_size);

        auto const gemm_to_profile
            = (gemm_idx == 1) ? profiler_backend::GemmToProfile::GEMM_1 : profiler_backend::GemmToProfile::GEMM_2;
        auto const& profiles = (gemm_idx == 1) ? mGemm1Profiles : mGemm2Profiles;

        // Get specific profile configs according to the profile_id.
        // Fallback tactic is set to be 0
        // TODO: use the best tactic id found offline for a better default inference perf
        auto const& profile = profile_id == -1 ? profiles.front() : profiles[profile_id];

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        auto const* expert_weights_ptr
            = (gemm_idx == 1) ? fc1_expert_weights.const_data_ptr() : fc2_expert_weights.const_data_ptr();

        // Preparation phase, only enabled during autotuning warmup phase.
        if (do_preparation)
        {
            // Set profiled gemm idx
            mProfiler->mGemmToProfile = gemm_to_profile;

            // mProfiler init
            auto parallelism_config = kernels::MOEParallelismConfig(static_cast<int>(tp_size),
                static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank),
                static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

            bool const USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
            bool const USE_LORA = false;
            auto activation_dtype
                = (mUseW4GroupScaling && !isWFP4A16Quant()) ? at::ScalarType::Float8_e4m3fn : mActivationDtype;
            activation_dtype = isNvfp4Quant() ? at::ScalarType::Long : activation_dtype;
#ifdef USING_OSS_CUTLASS_MOE_GEMM
            mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                tensorrt_llm::runtime::TorchUtils::dataType(activation_dtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mWeightDtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                hidden_size, unpadded_hidden_size > 0 ? unpadded_hidden_size : hidden_size, inter_size, group_size,
                ActivationType::Swiglu, USE_BIAS, USE_LORA, min_latency_mode,
                /*need_weights*/ false, parallelism_config, enable_alltoall);
#else
            mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                tensorrt_llm::runtime::TorchUtils::dataType(activation_dtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mWeightDtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                hidden_size, inter_size, group_size, ActivationType::Swiglu, USE_BIAS, USE_LORA, min_latency_mode,
                /*need_weights*/ false, parallelism_config);
#endif

            freeProfileWorkspace();
            size_t profile_workspace_size = mProfiler->getWorkspaceSize(num_rows);
            auto const cu_malloc_status = cudaMalloc(&mProfileWorkspace, profile_workspace_size);
            TORCH_CHECK(cu_malloc_status == cudaSuccess, "Can't allocate profile workspace for MoE GEMM profile.");

            mProfiler->prepare(num_rows, mProfileWorkspace, expert_weights_ptr, stream);
        }

        // Profile specific tactic. Assuming at least one preparation phase has been executed already.
        mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
    }

private:
    struct WorkspaceInfo
    {
        torch::Tensor workspace{};
        void* src_to_dest_map{};
    };

    std::mutex mMutex;
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
    std::shared_ptr<kernels::GemmProfilerBackend> mProfiler;
    c10::ScalarType mActivationDtype;
    c10::ScalarType mWeightDtype;
    c10::ScalarType mOutputDtype;
    // number of elements packed into the inner dimension of a matrix
    // e.g. 16 nvfp4 elements are packed into a single int64 element
    int64_t mInnerDimMultiplier;
    char* mProfileWorkspace = nullptr;
    std::map<cudaStream_t, WorkspaceInfo> mStreamWorkspaces;

    bool mUseDeepSeekFP8BlockScaling = false;
    bool mUseW4GroupScaling = false;
    bool mUseINT8WoqPerChannel = false;
    bool mUseMxfp8ActScaling = false;
    bool mUseFusedFinalize = true;

    using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    std::vector<Profile> mGemm1Profiles;
    std::vector<Profile> mGemm2Profiles;

    void freeProfileWorkspace()
    {
        if (mProfileWorkspace != nullptr)
        {
            auto const cu_free_status = cudaFree(mProfileWorkspace);
            TORCH_CHECK(cu_free_status == cudaSuccess,
                "Can't free profile workspace for MoE GEMM profile during memory reallocation.");
            mProfileWorkspace = nullptr;
        }
    }

    void setRunnerProfiles(torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        if (mUseDeepSeekFP8BlockScaling)
        {
            auto config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig(
                tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90::CtaShape128x16x128B,
                tensorrt_llm::cutlass_extensions::MainloopScheduleType::AUTO,
                tensorrt_llm::cutlass_extensions::EpilogueScheduleType::AUTO,
                tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_1x1x1);
            mKernelRunner->setTactic(config, config);
            return;
        }

        auto best_gemm1_profile = mGemm1Profiles.front();
        auto best_gemm2_profile = mGemm2Profiles.front();
        if (profile_ids.has_value())
        {
            TORCH_CHECK(profile_ids.value().size() == 2, "Expecting 2 profile ids");
            best_gemm1_profile
                = profile_ids.value()[0] == -1 ? best_gemm1_profile : mGemm1Profiles.at(profile_ids.value()[0]);
            best_gemm2_profile
                = profile_ids.value()[1] == -1 ? best_gemm2_profile : mGemm2Profiles.at(profile_ids.value()[1]);
        }
        mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
    }

    WorkspaceInfo const& getWorkspaceInfo(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int num_experts, int experts_per_token, ActivationType activation_type,
        kernels::MOEParallelismConfig const& parallelismConfig, bool min_latency_mode, cudaStream_t stream)
    {
        size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts,
            experts_per_token, activation_type, parallelismConfig, /* use_lora */ false, mUseDeepSeekFP8BlockScaling,
            min_latency_mode, mUseW4GroupScaling);
        size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);
        auto& workspace_info = mStreamWorkspaces[stream];

        std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

        int64_t const total_workspace_size = common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

        bool is_capturing = tensorrt_llm::common::isCapturing(stream);
        // Always allocate workspace when capturing cuda graph to avoid illegal memory access during replay
        if (is_capturing || workspace_info.workspace.numel() < total_workspace_size)
        {
            if (is_capturing)
            {
                TLLM_LOG_DEBUG(
                    "Allocating MoE workspace with %ld bytes size during cuda graph capture", total_workspace_size);
            }
            else
            {
                TLLM_LOG_DEBUG("MoE workspace size is not enough, increase the size from %ld bytes to %ld bytes",
                    workspace_info.workspace.numel(), total_workspace_size);
            }
            // Release memory first to avoid OOM.
            workspace_info = WorkspaceInfo();
            workspace_info.workspace = torch::empty({static_cast<long>(total_workspace_size)},
                torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        }
        workspace_info.src_to_dest_map
            = common::nextWorkspacePtr(static_cast<int8_t*>(workspace_info.workspace.data_ptr()), moe_workspace_size);

        return workspace_info;
    }

    kernels::QuantParams getQuantParams(int64_t const num_experts_on_rank, int64_t const hidden_size,
        int64_t const inter_size, torch::optional<c10::ArrayRef<torch::Tensor>> const& quant_scales) const
    {
        if (isFp8Quant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for fp8 quantization");
            TORCH_CHECK(quant_scales.value().size() == 4, "Expecting 4 quant scales for fp8 quantization");

            auto const fc1_dequant = quant_scales.value()[0];
            auto const fc2_quant = quant_scales.value()[1];
            auto const fc2_dequant = quant_scales.value()[2];
            auto const fc1_input_dequant = quant_scales.value()[3];

            // Check types
            CHECK_INPUT(fc1_dequant, c10::ScalarType::Float);
            CHECK_INPUT(fc2_quant, c10::ScalarType::Float);
            CHECK_INPUT(fc2_dequant, c10::ScalarType::Float);
            CHECK_INPUT(fc1_input_dequant, c10::ScalarType::Float);
            // Check ranks
            TORCH_CHECK(fc1_dequant.dim() == 1, "fc1 dequant must be 1D");
            TORCH_CHECK(fc2_quant.dim() == 0 || fc2_quant.dim() == 1, "fc2 quant must be a scalar or 1-D tensor");
            TORCH_CHECK(fc2_dequant.dim() == 1, "fc2 quant must be 1D");
            TORCH_CHECK(fc1_input_dequant.dim() == 0, "fc1 input dequant must be a scalar tensor");
            // Check shapes
            TORCH_CHECK(
                fc1_dequant.sizes()[0] == num_experts_on_rank, "fc1 dequant size must be (num_experts_on_rank,)");
            TORCH_CHECK(fc2_quant.dim() == 0 || fc2_quant.sizes()[0] == num_experts_on_rank,
                "fc2 quant must be scalar or (num_experts_on_rank,)");
            TORCH_CHECK(
                fc2_dequant.sizes()[0] == num_experts_on_rank, "fc2 dequant size must be (num_experts_on_rank,)");

            return kernels::QuantParams::FP8(static_cast<float const*>(fc1_dequant.data_ptr()),
                static_cast<float const*>(fc2_quant.data_ptr()), static_cast<float const*>(fc2_dequant.data_ptr()),
                /* fp8 output quant scale */ nullptr, static_cast<float const*>(fc1_input_dequant.data_ptr()),
                fc2_quant.dim() == 1);
        }
        else if (isWMxfp4AFp8Quant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for W4A8_MXFP4_MXF8 quantization");
            TORCH_CHECK(quant_scales.value().size() == 5, "Expecting 5 quant scales for W4A8_MXFP4_FP8 quantization");

            auto const fc1_weight_block = quant_scales.value()[0];
            auto const fc1_global = quant_scales.value()[1];
            auto const fc2_act_global = quant_scales.value()[2];
            auto const fc2_weight_block = quant_scales.value()[3];
            auto const fc2_global = quant_scales.value()[4];

            // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
            constexpr int FP8_PER_INT32 = 4;
            // Check types
            CHECK_INPUT(fc1_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc1_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_act_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc2_global, c10::ScalarType::Float);
            // Check ranks
            TORCH_CHECK(fc1_weight_block.dim() == 3, "fc1 weight block must be #D");
            TORCH_CHECK(fc1_global.dim() == 1, "fc1 global must be 1D");
            TORCH_CHECK(fc2_act_global.dim() == 0 || fc2_act_global.dim() == 1,
                "fc2 act global must be a scalar or 1-D tensor");
            TORCH_CHECK(fc2_weight_block.dim() == 3, "fc2 weight block must be 3D");
            TORCH_CHECK(fc2_global.dim() == 1, "fc2 global must be 1D");
            // Check shapes
            TORCH_CHECK(fc1_weight_block.sizes()[0] == num_experts_on_rank
                    && fc1_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                               inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX)
                            * 2
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc1_global.sizes()[0] == num_experts_on_rank, "fc1 global size must be (num_experts_on_rank,)");
            TORCH_CHECK(fc2_act_global.dim() == 0 || fc2_act_global.sizes()[0] == num_experts_on_rank,
                "fc2 act global must be scalar or (num_experts_on_rank,)");
            TORCH_CHECK(fc2_weight_block.sizes()[0] == num_experts_on_rank
                    && fc2_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX)
                    && fc2_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc2_global.sizes()[0] == num_experts_on_rank, "fc2 global size must be (num_experts_on_rank,)");

            return kernels::QuantParams::FP8MXFP4(nullptr,
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
                static_cast<float const*>(fc1_global.data_ptr()), static_cast<float const*>(fc2_act_global.data_ptr()),
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
                static_cast<float const*>(fc2_global.data_ptr()), false, fc2_act_global.dim() == 1);
        }
        else if (isWMxfp4AMxfp8Quant())
        {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for W4A8_MXFP4_MXFP8 quantization");
            TORCH_CHECK(quant_scales.value().size() == 4, "Expecting 4 quant scales for W4A8_MXFP4_MXFP8 quantization");

            auto const fc1_weight_block = quant_scales.value()[0];
            auto const fc1_global = quant_scales.value()[1];
            auto const fc2_weight_block = quant_scales.value()[2];
            auto const fc2_global = quant_scales.value()[3];

            // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
            constexpr int FP8_PER_INT32 = 4;
            CHECK_INPUT(fc1_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc1_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc2_global, c10::ScalarType::Float);
            TORCH_CHECK(fc1_weight_block.dim() == 3, "fc1 weight block must be #D");
            TORCH_CHECK(fc1_global.dim() == 1, "fc1 global must be 1D");
            TORCH_CHECK(fc2_weight_block.dim() == 3, "fc2 weight block must be 3D");
            TORCH_CHECK(fc2_global.dim() == 1, "fc2 global must be 1D");
            TORCH_CHECK(fc1_weight_block.sizes()[0] == num_experts_on_rank
                    && fc1_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                               inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX)
                            * 2
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc1_global.sizes()[0] == num_experts_on_rank, "fc1 global size must be (num_experts_on_rank,)");
            TORCH_CHECK(fc2_weight_block.sizes()[0] == num_experts_on_rank
                    && fc2_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX)
                    && fc2_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc2_global.sizes()[0] == num_experts_on_rank, "fc2 global size must be (num_experts_on_rank,)");

            return kernels::QuantParams::MXFP8MXFP4(
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
                static_cast<float const*>(fc1_global.data_ptr()),
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
                static_cast<float const*>(fc2_global.data_ptr()));
#else
            TORCH_CHECK(false, "MXFP8 x MXFP4 quantization is not supported in OSS Cutlass Moe Gemm");
#endif
        }
        else if (isNvfp4Quant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for nvfp4 quantization");
            TORCH_CHECK(quant_scales.value().size() == 6, "Expecting 6 quant scales for nvfp4 quantization");

            auto const fc1_act_global = quant_scales.value()[0];
            auto const fc1_weight_block = quant_scales.value()[1];
            auto const fc1_global = quant_scales.value()[2];
            auto const fc2_act_global = quant_scales.value()[3];
            auto const fc2_weight_block = quant_scales.value()[4];
            auto const fc2_global = quant_scales.value()[5];

            // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
            constexpr int FP8_PER_INT32 = 4;
            // Check types
            CHECK_INPUT(fc1_act_global, c10::ScalarType::Float);
            CHECK_INPUT(fc1_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc1_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_act_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc2_global, c10::ScalarType::Float);
            // Check ranks
            TORCH_CHECK(fc1_act_global.dim() == 0 || fc1_act_global.dim() == 1,
                "fc1 act global must be a scalar or 1-D tensor");
            TORCH_CHECK(fc1_weight_block.dim() == 3, "fc1 weight block must be #D");
            TORCH_CHECK(fc1_global.dim() == 1, "fc1 global must be 1D");
            TORCH_CHECK(fc2_act_global.dim() == 0 || fc2_act_global.dim() == 1,
                "fc2 act global must be a scalar or 1-D tensor");
            TORCH_CHECK(fc2_weight_block.dim() == 3, "fc2 weight block must be 3D");
            TORCH_CHECK(fc2_global.dim() == 1, "fc2 global must be 1D");
            // Check shapes
            TORCH_CHECK(fc1_act_global.dim() == 0 || fc1_act_global.sizes()[0] == num_experts_on_rank,
                "fc1 act global must be scalar or (num_experts_on_rank,)");
            TORCH_CHECK(fc1_weight_block.sizes()[0] == num_experts_on_rank
                    && fc1_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                               inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4)
                            * 2
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc1_global.sizes()[0] == num_experts_on_rank, "fc1 global size must be (num_experts_on_rank,)");
            TORCH_CHECK(fc2_act_global.dim() == 0 || fc2_act_global.sizes()[0] == num_experts_on_rank,
                "fc2 act global must be scalar or (num_experts_on_rank,)");
            TORCH_CHECK(fc2_weight_block.sizes()[0] == num_experts_on_rank
                    && fc2_weight_block.sizes()[1]
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4)
                    && fc2_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4),
                "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc2_global.sizes()[0] == num_experts_on_rank, "fc2 global size must be (num_experts_on_rank,)");

            return kernels::QuantParams::FP4(static_cast<float const*>(fc1_act_global.data_ptr()),
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
                static_cast<float const*>(fc1_global.data_ptr()), static_cast<float const*>(fc2_act_global.data_ptr()),
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
                static_cast<float const*>(fc2_global.data_ptr()), fc1_act_global.dim() == 1, fc2_act_global.dim() == 1);
        }
        else if (mUseDeepSeekFP8BlockScaling)
        {
            auto& fc1_scales = quant_scales.value()[0];
            auto& fc2_scales = quant_scales.value()[1];
            return kernels::QuantParams::FP8BlockScaling(
                static_cast<float const*>(fc1_scales.data_ptr()), static_cast<float const*>(fc2_scales.data_ptr()));
        }
        else if (isWFP4A16Quant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for weight only quantization");
            TORCH_CHECK(quant_scales.value().size() == 2, "Expecting 2 quant scales for W4A16 quantization");

            auto& fc1_weight_scales = quant_scales.value()[0];
            auto& fc2_weight_scales = quant_scales.value()[1];
            int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size;
            return kernels::QuantParams::GroupWise(group_size, static_cast<void const*>(fc1_weight_scales.data_ptr()),
                static_cast<void const*>(fc2_weight_scales.data_ptr()), nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr);
        }
        else if (isIntWeightOnlyQuant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for weight only quantization");
            if (mUseINT8WoqPerChannel)
            {
                TORCH_CHECK(
                    quant_scales.value().size() == 2, "Expecting 2 quant scales for INT8 weight only quantization");
                auto& fc1_weight_scales = quant_scales.value()[0];
                auto& fc2_weight_scales = quant_scales.value()[1];
                return kernels::QuantParams::Int(static_cast<float const*>(fc1_weight_scales.data_ptr()),
                    static_cast<float const*>(fc2_weight_scales.data_ptr()));
            }
            else if (isInt4Quant() && mUseW4GroupScaling)
            {
                TORCH_CHECK(quant_scales.value().size() == 8, "Expecting 8 quant scales for W4A8 quantization");

                auto& fc1_weight_scales = quant_scales.value()[0];
                auto& fc2_weight_scales = quant_scales.value()[1];
                auto& fc1_act_scales = quant_scales.value()[2];
                auto& fc2_act_scales = quant_scales.value()[3];
                auto& fc1_weight_zeros = quant_scales.value()[4];
                auto& fc2_weight_zeros = quant_scales.value()[5];
                auto& fc1_alpha = quant_scales.value()[6];
                auto& fc2_alpha = quant_scales.value()[7];
                int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size;
                return kernels::QuantParams::GroupWise(group_size,
                    static_cast<void const*>(fc1_weight_scales.data_ptr()),
                    static_cast<void const*>(fc2_weight_scales.data_ptr()),
                    static_cast<void const*>(fc1_act_scales.numel() > 0 ? fc1_act_scales.data_ptr() : nullptr),
                    static_cast<void const*>(fc2_act_scales.numel() > 0 ? fc2_act_scales.data_ptr() : nullptr),
                    static_cast<void const*>(fc1_weight_zeros.numel() > 0 ? fc1_weight_zeros.data_ptr() : nullptr),
                    static_cast<void const*>(fc2_weight_zeros.numel() > 0 ? fc2_weight_zeros.data_ptr() : nullptr),
                    static_cast<float const*>(fc1_alpha.numel() > 0 ? fc1_alpha.data_ptr() : nullptr),
                    static_cast<float const*>(fc2_alpha.numel() > 0 ? fc2_alpha.data_ptr() : nullptr));
            }
            else
            {
                TORCH_CHECK(false, "Unsupported weight only quantization");
            }
        }
        else
        {
            return kernels::QuantParams{};
        }
    }

    bool isFp8Quant() const
    {
        return !mUseDeepSeekFP8BlockScaling && mActivationDtype == c10::ScalarType::Float8_e4m3fn
            && mWeightDtype == c10::ScalarType::Float8_e4m3fn;
    }

    bool isNvfp4Quant() const
    {
        return mWeightDtype == c10::ScalarType::Long
            && mActivationDtype != c10::ScalarType::Float8_e4m3fn; // FP8 activation does not use FP4
    }

    bool isWFP4A16Quant() const
    {
        return mUseW4GroupScaling && mWeightDtype == c10::ScalarType::Byte;
    }

    bool isInt8Quant() const
    {
        return mWeightDtype == c10::ScalarType::Char;
    }

    bool isInt4Quant() const
    {
        return mWeightDtype == c10::ScalarType::QUInt4x2;
    }

    bool isW4AFp8Quant() const
    {
        return mActivationDtype == c10::ScalarType::Float8_e4m3fn && isInt4Quant();
    }

    bool isIntWeightOnlyQuant() const
    {
        return isInt8Quant() || isInt4Quant();
    }

    bool isWMxfp4AFp8Quant() const
    {
        return mActivationDtype == c10::ScalarType::Float8_e4m3fn && mWeightDtype == c10::ScalarType::Long
            && !mUseMxfp8ActScaling;
    }

    bool isWMxfp4AMxfp8Quant() const
    {
        return mActivationDtype == c10::ScalarType::Float8_e4m3fn && mWeightDtype == c10::ScalarType::Long
            && mUseMxfp8ActScaling;
    }
};

} // namespace torch_ext

TORCH_LIBRARY(trtllm, m)
{
    m.class_<torch_ext::FusedMoeRunner>("FusedMoeRunner")
        .def(torch::init<c10::ScalarType, c10::ScalarType, c10::ScalarType, bool, bool, bool, bool, bool>())
        .def("run_gemm_profile", &torch_ext::FusedMoeRunner::runGemmProfile)
        .def("get_tactic_num", &torch_ext::FusedMoeRunner::getTacticNum)
        .def("run_moe", &torch_ext::FusedMoeRunner::runMoe)
        .def("run_moe_min_latency", &torch_ext::FusedMoeRunner::runMoeMinLantency);
}
