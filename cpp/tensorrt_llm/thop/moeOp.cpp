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

#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/native/cuda/Resize.h>

#include <functional>

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
namespace kernels = tensorrt_llm::kernels;
using profiler_backend = kernels::GemmProfilerBackend;

class FusedMoeRunner : public torch::CustomClassHolder
{
public:
    template <typename Type, bool NeedQuant = false>
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
                return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, half, half>>();
            }
            else
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, half, Type>>();
            }
#ifdef ENABLE_BF16
        case c10::ScalarType::BFloat16:
            if constexpr (NeedQuant)
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, __nv_bfloat16, __nv_bfloat16>>();
            }
            else
            {
                return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type, __nv_bfloat16, Type>>();
            }
#endif
        default:
            C10_THROW_ERROR_FORMATTED(Error,
                "Invalid output type " << torch::toString(output_type) << " specified for "
                                       << torch::toString(mActivationDtype));
        }
    };

    FusedMoeRunner(c10::ScalarType activation_dtype, c10::ScalarType weight_dtype, c10::ScalarType output_dtype,
        bool use_fp8_block_scaling)
    {
        mActivationDtype = activation_dtype;
        mWeightDtype = weight_dtype;
        mOutputDtype = output_dtype;
        mUseFp8BlockScaling = use_fp8_block_scaling;
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
            mKernelRunner = switch_output_type<__nv_fp8_e4m3>(mOutputDtype);
        }
#endif
#ifdef ENABLE_FP4
        if (isNvfp4Quant())
        {
            mInnerDimMultiplier = 16;
            switch (mActivationDtype)
            {
            case c10::ScalarType::Half:
#ifdef ENABLE_BF16
            case c10::ScalarType::BFloat16:
#endif
                mKernelRunner = switch_output_type<__nv_fp4_e2m1, true>(mOutputDtype);
                break;
            default: mKernelRunner = switch_output_type<__nv_fp4_e2m1, false>(mOutputDtype);
            }
        }
#endif
        if (!mKernelRunner)
        {
            C10_THROW_ERROR_FORMATTED(Error,
                "Could not construct fused moe op with the requested input combination Activation: "
                    << torch::toString(mActivationDtype) << ", Weight: " << torch::toString(mWeightDtype)
                    << ", Output: " << torch::toString(mOutputDtype));
        }

        mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
        mAllProfiles = mKernelRunner->getTactics();
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
        torch::optional<torch::Tensor> token_final_scales, torch::Tensor const& fc1_expert_weights,
        torch::Tensor const& fc2_expert_weights, torch::optional<c10::ArrayRef<torch::Tensor>> quant_scales,
        torch::optional<torch::Tensor> input_sf, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size,
        int64_t const ep_rank, bool min_latency_mode, torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        // Free the profile workspace to save memory
        if (mProfileWorkspace != nullptr)
        {
            auto const cu_free_status = cudaFree(mProfileWorkspace);
            TORCH_CHECK(
                cu_free_status == cudaSuccess, "Can't free profile workspace for MoE GEMM profile before runMoe.");
            mProfileWorkspace = nullptr;
        }

        std::lock_guard<std::mutex> lock(mMutex);

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

        int experts_per_token = token_selected_experts.sizes()[1];
        int64_t num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        int const num_experts_on_rank = fc2_expert_weights.sizes()[0];
        auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
        auto activation_type = tensorrt_llm::ActivationType::Swiglu;

        setRunnerProfiles(profile_ids);

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        std::vector<int64_t> output_shape = {num_rows, hidden_size};
        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));

        WorkspaceInfo workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), activation_type, parallelism_config, min_latency_mode);

        auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
        kernels::MoeMinLatencyParams min_latency_params{};

        // TODO: support lora in the future
        kernels::LoraParams lora_params{};
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(), nullptr, activation_type, fc2_expert_weights.const_data_ptr(), nullptr,
            quant_params, num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
            mUseFp8BlockScaling, min_latency_mode, min_latency_params, stream);

        return output;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> runMoeMinLantency(torch::Tensor const& input,
        torch::Tensor const& token_selected_experts, torch::optional<torch::Tensor> token_final_scales,
        torch::Tensor const& fc1_expert_weights, torch::Tensor const& fc2_expert_weights,
        torch::optional<c10::ArrayRef<torch::Tensor>> quant_scales, torch::optional<torch::Tensor> input_sf,
        int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
        bool min_latency_mode, torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        std::lock_guard<std::mutex> lock(mMutex);

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

        int experts_per_token = token_selected_experts.sizes()[1];
        int64_t num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        int const num_experts_on_rank = fc2_expert_weights.sizes()[0];
        auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
        auto activation_type = tensorrt_llm::ActivationType::Swiglu;

        setRunnerProfiles(profile_ids);

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        std::vector<int64_t> output_shape = {num_rows * num_experts_on_rank, hidden_size};
        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));

        auto num_active_experts_per_node = torch::empty({1}, input.options().dtype(at::ScalarType::Int));
        auto experts_to_token_score
            = torch::empty({num_experts_on_rank, num_rows}, input.options().dtype(at::ScalarType::Float));
        auto active_expert_global_ids = torch::empty({num_experts_on_rank}, input.options().dtype(at::ScalarType::Int));

        kernels::MoeMinLatencyParams min_latency_params{};
        min_latency_params.num_active_experts_per_node = static_cast<int*>(num_active_experts_per_node.data_ptr());
        min_latency_params.experts_to_token_score = static_cast<float*>(experts_to_token_score.data_ptr());
        min_latency_params.active_expert_global_ids = static_cast<int*>(active_expert_global_ids.data_ptr());

        WorkspaceInfo workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), activation_type, parallelism_config, min_latency_mode);

        auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);

        // TODO: support lora in the future
        kernels::LoraParams lora_params{};
        mKernelRunner->runMoe(input.const_data_ptr(),
            input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
            reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
            token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr())
                                           : nullptr,
            fc1_expert_weights.const_data_ptr(), nullptr, activation_type, fc2_expert_weights.const_data_ptr(), nullptr,
            quant_params, num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
            mUseFp8BlockScaling, min_latency_mode, min_latency_params, stream);

        return std::make_tuple(output, num_active_experts_per_node, experts_to_token_score, active_expert_global_ids);
    }

    int64_t getTacticNum()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mAllProfiles.size();
    }

    void runGemmProfile(torch::Tensor const& input, torch::Tensor const& fc2_expert_weights, int64_t const top_k,
        int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
        bool const min_latency_mode, int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        // TODO: support profiling under fp8 block scaling in the future
        if (mUseFp8BlockScaling)
        {
            return;
        }

        int64_t const num_rows = input.sizes()[0];
        int64_t const hidden_size = fc2_expert_weights.sizes()[1];
        int64_t const inter_size = fc2_expert_weights.sizes()[2] * mInnerDimMultiplier;
        int const num_experts = static_cast<int>(fc2_expert_weights.sizes()[0] * ep_size);

        // Get specific profile configs according to the profile_id.
        // Fallback tactic is set to be 0
        // TODO: use the best tactic id found offline for a better default inference perf
        auto const& profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        // Preparation phase, only enabled during autotuning warmup phase.
        if (do_preparation)
        {
            // Set profiled gemm idx
            mProfiler->mGemmToProfile
                = (gemm_idx == 1) ? profiler_backend::GemmToProfile::GEMM_1 : profiler_backend::GemmToProfile::GEMM_2;

            // mProfiler init
            auto parallelism_config = kernels::MOEParallelismConfig(static_cast<int>(tp_size),
                static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank));

            int const GROUP_SIZE = -1;
            bool const USE_BIAS = false;
            bool const USE_LORA = false;
            mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                tensorrt_llm::runtime::TorchUtils::dataType(mActivationDtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mWeightDtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                hidden_size, inter_size, GROUP_SIZE, tensorrt_llm::ActivationType::Swiglu, USE_BIAS, USE_LORA,
                min_latency_mode, parallelism_config);

            if (mProfileWorkspace != nullptr)
            {
                auto const cu_free_status = cudaFree(mProfileWorkspace);
                TORCH_CHECK(cu_free_status == cudaSuccess,
                    "Can't free profile workspace for MoE GEMM profile during memory reallocation.");
                mProfileWorkspace = nullptr;
            }
            size_t profile_workspace_size = mProfiler->getWorkspaceSize(num_rows);
            auto const cu_malloc_status = cudaMalloc(&mProfileWorkspace, profile_workspace_size);
            TORCH_CHECK(cu_malloc_status == cudaSuccess, "Can't allocate profile workspace for MoE GEMM profile.");

            mProfiler->prepare(num_rows, mProfileWorkspace, stream);
        }

        // Profile specific tactic. Assuming at least one preparation phase has been executed already.
        mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, stream);
    }

private:
    struct WorkspaceInfo
    {
        void* workspace{};
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

    bool mUseFp8BlockScaling = false;

    using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    std::vector<Profile> mAllProfiles;

    void setRunnerProfiles(torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        if (mUseFp8BlockScaling)
        {
            auto config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig(
                tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90::CtaShape128x16x128B,
                tensorrt_llm::cutlass_extensions::MainloopScheduleType::AUTO,
                tensorrt_llm::cutlass_extensions::EpilogueScheduleType::AUTO,
                tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_1x1x1);
            mKernelRunner->setTactic(config, config);
            return;
        }

        auto best_gemm1_profile = mAllProfiles.front();
        auto best_gemm2_profile = mAllProfiles.front();
        if (profile_ids.has_value())
        {
            TORCH_CHECK(profile_ids.value().size() == 2, "Expecting 2 profile ids");
            best_gemm1_profile
                = profile_ids.value()[0] == -1 ? best_gemm1_profile : mAllProfiles.at(profile_ids.value()[0]);
            best_gemm2_profile
                = profile_ids.value()[1] == -1 ? best_gemm2_profile : mAllProfiles.at(profile_ids.value()[1]);
        }
        mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
    }

    WorkspaceInfo getWorkspaceInfo(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int num_experts, int experts_per_token, tensorrt_llm::ActivationType activation_type,
        kernels::MOEParallelismConfig const& parallelismConfig, bool min_latency_mode)
    {
        size_t moe_workspace_size
            = mKernelRunner->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, experts_per_token,
                activation_type, parallelismConfig, /* use_lora */ false, mUseFp8BlockScaling, min_latency_mode,
                /* hasExpertPrequantScales */ false);
        size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);

        std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

        size_t total_workspace_size = common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
        auto workspace = torch::empty({static_cast<long>(total_workspace_size)},
            torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        WorkspaceInfo info{};
        info.workspace = workspace.data_ptr();
        info.src_to_dest_map = common::nextWorkspacePtr(static_cast<int8_t*>(workspace.data_ptr()), moe_workspace_size);

        return info;
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

            CHECK_INPUT(fc1_dequant, c10::ScalarType::Float);
            CHECK_INPUT(fc2_quant, c10::ScalarType::Float);
            CHECK_INPUT(fc2_dequant, c10::ScalarType::Float);
            CHECK_INPUT(fc1_input_dequant, c10::ScalarType::Float);
            TORCH_CHECK(fc1_dequant.dim() == 1, "fc1 dequant must be 1D");
            TORCH_CHECK(fc2_quant.dim() == 0, "fc2 quant must be a scalar tensor");
            TORCH_CHECK(fc2_dequant.dim() == 1, "fc2 quant must be 1D");
            TORCH_CHECK(fc1_input_dequant.dim() == 0, "fc1 input dequant must be a scalar tensor");
            TORCH_CHECK(
                fc1_dequant.sizes()[0] == num_experts_on_rank, "fc1 dequant size must be (num_experts_on_rank,)");
            TORCH_CHECK(
                fc2_dequant.sizes()[0] == num_experts_on_rank, "fc2 dequant size must be (num_experts_on_rank,)");

            return kernels::QuantParams::FP8(static_cast<float const*>(fc1_dequant.data_ptr()),
                static_cast<float const*>(fc2_quant.data_ptr()), static_cast<float const*>(fc2_dequant.data_ptr()),
                /* fp8 output quant scale */ nullptr, static_cast<float const*>(fc1_input_dequant.data_ptr()));
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
            CHECK_INPUT(fc1_act_global, c10::ScalarType::Float);
            CHECK_INPUT(fc1_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc1_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_act_global, c10::ScalarType::Float);
            CHECK_INPUT(fc2_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc2_global, c10::ScalarType::Float);
            TORCH_CHECK(fc1_act_global.dim() == 0, "fc1 act global must be a scalar tensor");
            TORCH_CHECK(fc1_weight_block.dim() == 3, "fc1 weight block must be #D");
            TORCH_CHECK(fc1_global.dim() == 1, "fc1 global must be 1D");
            TORCH_CHECK(fc2_act_global.dim() == 0, "fc2 act global must be a scalar tensor");
            TORCH_CHECK(fc2_weight_block.dim() == 3, "fc2 weight block must be 3D");
            TORCH_CHECK(fc2_global.dim() == 1, "fc2 global must be 1D");
            TORCH_CHECK(fc1_weight_block.sizes()[0] == num_experts_on_rank
                    && fc1_weight_block.sizes()[1] == inter_size * 2
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize
                        == hidden_size,
                "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc1_global.sizes()[0] == num_experts_on_rank, "fc1 global size must be (num_experts_on_rank,)");
            TORCH_CHECK(fc2_weight_block.sizes()[0] == num_experts_on_rank && fc2_weight_block.sizes()[1] == hidden_size
                    && fc2_weight_block.sizes()[2] * FP8_PER_INT32
                            * tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::BlockScaleVectorSize
                        == inter_size,
                "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
                "block_scale_vector_size)");
            TORCH_CHECK(fc2_global.sizes()[0] == num_experts_on_rank, "fc2 global size must be (num_experts_on_rank,)");

            return kernels::QuantParams::FP4(static_cast<float const*>(fc1_act_global.data_ptr()),
                static_cast<tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
                static_cast<float const*>(fc1_global.data_ptr()), static_cast<float const*>(fc2_act_global.data_ptr()),
                static_cast<tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
                static_cast<float const*>(fc2_global.data_ptr()));
        }
        else if (mUseFp8BlockScaling)
        {
            auto& fc1_scales = quant_scales.value()[0];
            auto& fc2_scales = quant_scales.value()[1];
            return kernels::QuantParams::FP8BlockScaling(
                static_cast<float const*>(fc1_scales.data_ptr()), static_cast<float const*>(fc2_scales.data_ptr()));
        }
        else
        {
            return kernels::QuantParams{};
        }
    }

    bool isFp8Quant() const
    {
        return !mUseFp8BlockScaling && mActivationDtype == c10::ScalarType::Float8_e4m3fn
            && mWeightDtype == c10::ScalarType::Float8_e4m3fn;
    }

    bool isNvfp4Quant() const
    {
        return mWeightDtype == c10::ScalarType::Long;
    }
};

} // namespace torch_ext

TORCH_LIBRARY(trtllm, m)
{
    m.class_<torch_ext::FusedMoeRunner>("FusedMoeRunner")
        .def(torch::init<c10::ScalarType, c10::ScalarType, c10::ScalarType, bool>())
        .def("run_gemm_profile", &torch_ext::FusedMoeRunner::runGemmProfile)
        .def("get_tactic_num", &torch_ext::FusedMoeRunner::getTacticNum)
        .def("run_moe", &torch_ext::FusedMoeRunner::runMoe)
        .def("run_moe_min_latency", &torch_ext::FusedMoeRunner::runMoeMinLantency);
}
