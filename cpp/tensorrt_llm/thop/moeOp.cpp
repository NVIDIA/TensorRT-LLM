/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_grouped_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_problem_builder.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_slot_expand.h"

#include "cutlass/gemm_coord.h"

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cuda_graph_grouped_gemm.h"
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

TRTLLM_NAMESPACE_BEGIN

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

// Mirrors tensorrt_llm::kernels::RequestType for the LoRA host_request_types layout.
// kCONTEXT/kGENERATION encoding matches the rest of the LoRA stack (e.g. loraOp.cpp).
enum class MoeLoraRequestType : int32_t
{
    kCONTEXT = 0,
    kGENERATION = 1
};

// ---------------------------------------------------------------------------
// libtorch-bound implementation of MoeLoraGroupedGemmRunFn.
//
// This is the per-module GEMM dispatch the grouped-GEMM LoRA core uses. It builds
// the per-token problem descriptors on device via the libtorch-free
// launchMoeLoraProblemBuilder, then dispatches cudaGraph(SplitK)GroupedGemm.
// The latter allocates workspace via at::Tensor, so this function must live in
// th_common, which links libtorch. moe_kernels.cu reaches it indirectly through
// the function pointer stored in LoraParams::grouped_gemm.run; that indirection
// keeps libmoe_gemm_src.a (also linked into the TensorRT plugin shared object)
// free of a transitive dependency on libtorch.
// ---------------------------------------------------------------------------
inline void moeLoraGroupedGemmRunImpl(::tensorrt_llm::kernels::cutlass_kernels::MoeLoraGroupedGemmModule const& mod,
    int64_t num_permuted_tokens, int64_t in_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes,
    int64_t splitk_slices, void const* input_base, void* output_base, nvinfer1::DataType data_type, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(mod.permuted_ranks_dev != nullptr,
        "Grouped-GEMM LoRA module is missing permuted ranks buffer (forgot to populate grouped_gemm?).");

    // Repack the device-resident scratch into the bundle the problem-builder
    // consumes. The typed casts recover the concrete pointer types that
    // MoeLoraGroupedGemmModule stores as void* for header decoupling.
    ::tensorrt_llm::kernels::cutlass_kernels::MoeLoraGemmGroupArrays arrays{};
    arrays.problem_sizes_in = static_cast<cutlass::gemm::GemmCoord*>(mod.problem_sizes_in_dev);
    arrays.problem_sizes_out = static_cast<cutlass::gemm::GemmCoord*>(mod.problem_sizes_out_dev);
    arrays.a_ptrs_in = mod.a_ptrs_in_dev;
    arrays.b_ptrs_in = mod.b_ptrs_in_dev;
    arrays.d_ptrs_in = mod.d_ptrs_in_dev;
    arrays.b_ptrs_out = mod.b_ptrs_out_dev;
    arrays.d_ptrs_out = mod.d_ptrs_out_dev;
    arrays.lda_in = mod.lda_in_dev;
    arrays.ldb_in = mod.ldb_in_dev;
    arrays.ldd_in = mod.ldd_in_dev;
    arrays.ldb_out = mod.ldb_out_dev;
    arrays.ldd_out = mod.ldd_out_dev;
    arrays.splitk_offsets = mod.splitk_offsets_dev;

    ::tensorrt_llm::kernels::cutlass_kernels::launchMoeLoraProblemBuilder(mod.permuted_ranks_dev, mod.permuted_ptrs_dev,
        input_base, mod.lowrank_workspace_dev, output_base, num_permuted_tokens, in_hidden_size, mod.out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices, arrays, stream);
    sync_check_cuda_error(stream);

    // The cuda_graph_(split_k_)grouped_gemm wrappers accept ldc == ldd when C
    // aliases D (the no-bias case). The problem-builder produces a single
    // ldd_in / ldd_out per stage, reused for ldcGpu below.
    auto* host_max_in = static_cast<cutlass::gemm::GemmCoord*>(mod.host_max_problem_in_pinned);
    auto* host_max_out = static_cast<cutlass::gemm::GemmCoord*>(mod.host_max_problem_out_pinned);

    // kMinKN mirrors the value attention LoRA uses for kernel selection. The
    // wrappers fall back to the smaller-tile family when min(K, N) < kMinKN.
    constexpr int kMinKN = 16;

    // In-GEMM (low-rank down-projection). Each problem is a single permuted
    // token (M=1), so split-K over K offers no benefit. Use the plain grouped
    // GEMM; the split-K grouped GEMM raises an illegal instruction on SM100 when
    // reused within a process.
    ::tensorrt_llm::kernels::cudaGraphGroupedGemm(arrays.problem_sizes_in, static_cast<int>(num_permuted_tokens),
        arrays.a_ptrs_in, arrays.b_ptrs_in, arrays.d_ptrs_in, arrays.d_ptrs_in, arrays.lda_in, arrays.ldb_in,
        arrays.ldd_in, arrays.ldd_in,
        /*isLoraIn=*/true, data_type, kMinKN, host_max_in, stream);
    sync_check_cuda_error(stream);

    ::tensorrt_llm::kernels::cudaGraphGroupedGemm(arrays.problem_sizes_out, static_cast<int>(num_permuted_tokens),
        arrays.d_ptrs_in /*== a_ptrs_out*/, arrays.b_ptrs_out, arrays.d_ptrs_out, arrays.d_ptrs_out, arrays.ldd_in,
        arrays.ldb_out, arrays.ldd_out, arrays.ldd_out,
        /*isLoraIn=*/false, data_type, kMinKN, host_max_out, stream);
    sync_check_cuda_error(stream);
}

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
        bool use_mxfp8_act_scaling, bool use_mxfp8_weight_scaling, bool use_fused_finalize)
    {
        mActivationDtype = activation_dtype;
        mWeightDtype = weight_dtype;
        mOutputDtype = output_dtype;
        mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
        mUseW4GroupScaling = use_w4_group_scaling;
        mUseINT8WoqPerChannel = use_int8_woq_per_channel;
        mUseMxfp8ActScaling = use_mxfp8_act_scaling;
        mUseMxfp8WeightScaling = use_mxfp8_weight_scaling;
        mUseFusedFinalize = use_fused_finalize;
        mInnerDimMultiplier = 1;

        // MXFP8xMXFP8 grouped MoE is only meaningful for the <e4m3, e4m3>
        // template instantiation. Reject other (act, weight) dtype pairs at
        // construction time so the downstream kernel runner never sees a
        // mismatched configuration.
        TORCH_CHECK(!mUseMxfp8WeightScaling
                || (mActivationDtype == c10::ScalarType::Float8_e4m3fn
                    && mWeightDtype == c10::ScalarType::Float8_e4m3fn),
            "use_mxfp8_weight_scaling requires both activation and weight dtypes to be Float8_e4m3fn.");

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
        // Per-tensor FP8 and MXFP8xMXFP8 share the <e4m3, e4m3> instantiation;
        // use_mxfp8_weight_scaling_ selects between them at runtime.
        if (isFp8Quant() || isWMxfp8AMxfp8Quant())
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
        mKernelRunner->use_mxfp8_weight_scaling_ = mUseMxfp8WeightScaling;

        mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
        mGemm1Profiles = mKernelRunner->getTactics(MoeGemmId::GEMM_1);
        mGemm2Profiles = mKernelRunner->getTactics(MoeGemmId::GEMM_2);
        cuInit(0);
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

    // Release internal workspace buffers to free GPU memory.
    // Workspaces will be re-allocated on the next runMoe/runGemmProfile call.
    void clearWorkspaces()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mStreamWorkspaces.clear();
        freeProfileWorkspace();
    }

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
        torch::optional<int64_t> const& activation_type, torch::optional<int64_t> const& unpadded_hidden_size,
        torch::optional<int64_t> const& num_valid_tokens, torch::optional<torch::Tensor> const& out_tensor,
        bool use_dynamic_fc2_scale = false,
        // Routed-expert LoRA inputs (all optional; presence of fc1_lora_ranks activates LoRA).
        // Each *_ranks   : CPU int32  [num_seqs]
        // Each *_weights : CPU int64  [num_seqs, 3], holding (A_ptr, B_ptr, DoRA_ptr); DoRA unused.
        torch::optional<torch::Tensor> const& fc1_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& fc1_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& fc2_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& fc2_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& gated_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& gated_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& host_request_types = torch::nullopt,
        torch::optional<torch::Tensor> const& host_context_lengths = torch::nullopt, int64_t lora_max_low_rank = 0,
        // Slot-indexed CUDA-graph LoRA inputs (mutually exclusive with the per-request
        // schema above). When fc1_slot_lora_ranks is provided, the per-token expansion
        // is performed inside the op via token_to_slot[t] indexed into the slot tables.
        //   slot_*_ranks       : CPU pinned int32 [max_lora_size]
        //   slot_*_weight_ptrs : CPU pinned int64 [max_lora_size, 3]  (A, B, dora-ignored)
        //   token_to_slot      : CPU pinned int32 [>= num_tokens]
        torch::optional<torch::Tensor> const& fc1_slot_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& fc1_slot_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& fc2_slot_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& fc2_slot_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& gated_slot_lora_ranks = torch::nullopt,
        torch::optional<torch::Tensor> const& gated_slot_lora_weight_ptrs = torch::nullopt,
        torch::optional<torch::Tensor> const& token_to_slot = torch::nullopt)
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

        ActivationType base_activation_type = activation_type.has_value()
            ? static_cast<ActivationType>(activation_type.value())
            : ActivationType::Swiglu;
        if (mUseINT8WoqPerChannel)
        {
            // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
            // [num_experts, inter_size, hidden_size]
            TORCH_CHECK(fc1_expert_weights.sizes()[2] == fc2_expert_weights.sizes()[1] * mInnerDimMultiplier * 2,
                "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");
        }
        else
        {
            if (isGatedActivation(base_activation_type))
            {
                TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * mInnerDimMultiplier * 2,
                    "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");
            }
            else
            {
                TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * mInnerDimMultiplier,
                    "fc1_expert_weights inter size must be equal to fc2_expert_weights inter size.");
            }
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

        std::vector<int64_t> output_shape = {num_rows, unpadded_hidden_size_val};
        torch::Tensor output;
        if (out_tensor.has_value())
        {
            auto const& provided = out_tensor.value();
            CHECK_INPUT(provided, mOutputDtype);
            TORCH_CHECK(provided.sizes() == output_shape, "Provided out tensor has incorrect shape. Expected ",
                output_shape, ", got ", provided.sizes());
            output = provided;
        }
        else
        {
            output = torch::empty(output_shape, input.options().dtype(mOutputDtype));
        }

        // ===== Routed-expert LoRA setup =====
        // LoRA is activated by the per-request schema (fc1_lora_ranks).
        bool const lora_per_request = fc1_lora_ranks.has_value();
        bool const lora_slot_indexed = fc1_slot_lora_ranks.has_value();
        bool const lora_active = lora_per_request || lora_slot_indexed;
        bool const is_gated_act = isGatedActivation(base_activation_type);
        if (lora_active)
        {
            // The per-request and slot-indexed schemas are mutually exclusive:
            // each drives a different token->adapter expansion inside
            // buildMoeLoraParams, and supplying both is ambiguous. The Python
            // wrapper (torch_custom_ops.fused_moe) rejects this too, but the op
            // is public, so enforce it here as well for direct C++/op callers.
            TORCH_CHECK(!(lora_per_request && lora_slot_indexed),
                "MoE LoRA: the per-request (fc1_lora_ranks, ...) and slot-indexed (fc1_slot_lora_ranks, ..., "
                "token_to_slot) input schemas are mutually exclusive. Provide exactly one, not both.");
            // Conservative rejections (min-latency, alltoall, unsupported quant, graph capture).
            TORCH_CHECK(!min_latency_mode, "MoE LoRA is not supported in min-latency mode.");
            TORCH_CHECK(!enable_alltoall,
                "MoE LoRA is not supported with alltoall: the per-token adapter pointer arrays do not survive "
                "cross-rank token reshuffling.");
            bool const is_per_tensor_fp8 = isFp8Quant();
            TORCH_CHECK(mActivationDtype == c10::ScalarType::Half || mActivationDtype == c10::ScalarType::BFloat16
                    || is_per_tensor_fp8,
                "MoE LoRA only supports fp16, bf16, or per-tensor FP8 (qdq) base weights. FP8 block-scale, NVFP4, "
                "MXFP8, and integer quant are not supported.");
            TORCH_CHECK(
                mWeightDtype == c10::ScalarType::Half || mWeightDtype == c10::ScalarType::BFloat16 || is_per_tensor_fp8,
                "MoE LoRA supports unquantized fp16/bf16 or per-tensor FP8 (qdq) base expert weights only "
                "(LoRA adapters are always fp16/bf16).");
            // The grouped-GEMM core runs entirely on the stream, so CUDA-graph
            // capture is supported, but only with the slot-indexed schema. The
            // per-request schema expands adapters on the host into pinned
            // buffers, which a captured graph would freeze at capture time.
            TORCH_CHECK(!(lora_per_request && tensorrt_llm::common::isCapturing(stream)),
                "MoE LoRA: the per-request input schema is not CUDA-graph capturable (its host-side adapter "
                "expansion is frozen at capture time). Use the slot-indexed schema (fc1_slot_lora_ranks, ..., "
                "token_to_slot) for CUDA-graph decode, or run the per-request schema eagerly.");
        }
        // Build LoraParams up-front. The grouped-GEMM core uses persistent
        // device scratch wired into LoraParams::grouped_gemm; there is no
        // separate cuBLAS workspace to size.
        auto lora_params_opt = buildMoeLoraParams(fc1_lora_ranks, fc1_lora_weight_ptrs, fc2_lora_ranks,
            fc2_lora_weight_ptrs, gated_lora_ranks, gated_lora_weight_ptrs, host_request_types, host_context_lengths,
            fc1_slot_lora_ranks, fc1_slot_lora_weight_ptrs, fc2_slot_lora_ranks, fc2_slot_lora_weight_ptrs,
            gated_slot_lora_ranks, gated_slot_lora_weight_ptrs, token_to_slot,
            /*num_tokens=*/num_rows, hidden_size, inter_size, mActivationDtype, lora_max_low_rank, is_gated_act, stream,
            static_cast<int>(experts_per_token));

        WorkspaceInfo const& workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
            static_cast<int>(experts_per_token), base_activation_type, parallelism_config, min_latency_mode, stream,
            lora_active);

        auto quant_params
            = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales, base_activation_type);

        // Dynamic fc2 scale: allocate workspace buffers
        at::Tensor dynamic_fc2_amax_tensor, dynamic_fc2_alpha_tensor, dynamic_fc2_bf16_tensor;
        if (use_dynamic_fc2_scale && isNvfp4Quant() && quant_scales.has_value() && quant_scales.value().size() >= 7)
        {
            auto opts_f32 = at::TensorOptions().dtype(at::ScalarType::Float).device(input.device());
            auto opts_bf16 = at::TensorOptions().dtype(at::ScalarType::BFloat16).device(input.device());
            dynamic_fc2_amax_tensor = at::empty({1}, opts_f32);
            dynamic_fc2_alpha_tensor = at::empty({num_experts_on_rank}, opts_f32);
            int64_t expanded_rows = num_rows * static_cast<int64_t>(experts_per_token);
            dynamic_fc2_bf16_tensor = at::empty({expanded_rows, inter_size}, opts_bf16);

            quant_params.fp4.dynamic_fc2_input_scale.enabled = true;
            quant_params.fp4.dynamic_fc2_input_scale.amax = dynamic_fc2_amax_tensor.data_ptr<float>();
            quant_params.fp4.dynamic_fc2_input_scale.alpha = dynamic_fc2_alpha_tensor.data_ptr<float>();
            quant_params.fp4.dynamic_fc2_input_scale.bf16_buffer = dynamic_fc2_bf16_tensor.data_ptr();
            // 7th element: per-expert fc2_weight_scale_2 passed directly from Python
            quant_params.fp4.dynamic_fc2_input_scale.weight_scale_2
                = static_cast<float const*>(quant_scales.value()[6].data_ptr());
        }

        kernels::MoeMinLatencyParams min_latency_params{};

        // Use the populated LoraParams when LoRA is active, otherwise a default-constructed empty one.
        ::tensorrt_llm::kernels::LoraParams lora_params
            = lora_params_opt.value_or(::tensorrt_llm::kernels::LoraParams{});
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
            num_rows, num_valid_tokens.has_value() ? num_valid_tokens.value() : num_rows, hidden_size,
            unpadded_hidden_size_val, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall, lora_active,
            lora_params, mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
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
            num_rows, num_valid_tokens.has_value() ? num_valid_tokens.value() : num_rows, hidden_size, inter_size,
            num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, lora_active, lora_params,
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
        torch::optional<int64_t> const& activation_type, torch::optional<int64_t> const& unpadded_hidden_size,
        torch::optional<int64_t> const& num_valid_tokens, torch::optional<torch::Tensor> const& out_tensor)
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
        ActivationType base_activation_type = activation_type.has_value()
            ? static_cast<ActivationType>(activation_type.value())
            : ActivationType::Swiglu;
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
        torch::Tensor output;
        if (out_tensor.has_value())
        {
            auto const& provided = out_tensor.value();
            CHECK_INPUT(provided, mOutputDtype);
            TORCH_CHECK(provided.sizes() == output_shape, "Provided out tensor has incorrect shape. Expected ",
                output_shape, ", got ", provided.sizes());
            output = provided;
        }
        else
        {
            output = torch::empty(output_shape, input.options().dtype(mOutputDtype));
        }

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

        auto quant_params
            = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales, base_activation_type);

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
            num_rows, num_valid_tokens.has_value() ? num_valid_tokens.value() : num_rows, hidden_size,
            unpadded_hidden_size_val, inter_size, num_experts_total, static_cast<int>(experts_per_token),
            static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
            static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, enable_alltoall, false, lora_params,
            mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
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
            num_rows, num_valid_tokens.has_value() ? num_valid_tokens.value() : num_rows, hidden_size, inter_size,
            num_experts_total, static_cast<int>(experts_per_token),
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
        int64_t const profile_id, bool const do_preparation, int64_t const activation_type_int,
        int64_t const unpadded_hidden_size)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        // TODO: support profiling under fp8 block scaling in the future
        if (mUseDeepSeekFP8BlockScaling)
        {
            return;
        }
        ActivationType activation_type = static_cast<ActivationType>(activation_type_int);

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
                activation_type, USE_BIAS, USE_LORA, min_latency_mode,
                /*need_weights*/ false, parallelism_config, enable_alltoall, mUseMxfp8WeightScaling);
#else
            mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                tensorrt_llm::runtime::TorchUtils::dataType(activation_dtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mWeightDtype),
                tensorrt_llm::runtime::TorchUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                hidden_size, inter_size, group_size, activation_type, USE_BIAS, USE_LORA, min_latency_mode,
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
    bool mUseMxfp8WeightScaling = false;

    using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    std::vector<Profile> mGemm1Profiles;
    std::vector<Profile> mGemm2Profiles;

    // ===== Routed-expert LoRA state =====
    // Pinned-host and persistent-device buffers for the capture-safe MoE LoRA
    // path. The pinned-host tensors hold the per-token expanded LoRA tables
    // (ranks and weight-pointer pairs) so the in-op async H2D into the device
    // mirrors is graph-capturable; an async H2D from pageable host memory
    // silently becomes synchronous and breaks capture. Both tensors are sized
    // at mLoraHostBufCapacity (max_num_tokens) and reused across calls so the
    // source and destination addresses are stable across capture and replay.
    // Only the first num_tokens entries are valid each call.
    at::Tensor mLoraExpandFC1RanksPinned;        // [max_num_tokens]      int32
    at::Tensor mLoraExpandFC1WeightPtrsPinned;   // [max_num_tokens * 2]  int64 (A, B)
    at::Tensor mLoraExpandFC2RanksPinned;        // [max_num_tokens]      int32
    at::Tensor mLoraExpandFC2WeightPtrsPinned;   // [max_num_tokens * 2]  int64
    at::Tensor mLoraExpandGatedRanksPinned;      // [max_num_tokens]      int32
    at::Tensor mLoraExpandGatedWeightPtrsPinned; // [max_num_tokens * 2]  int64
    at::Tensor mLoraExpandFC1RanksDevice;
    at::Tensor mLoraExpandFC1WeightPtrsDevice;
    at::Tensor mLoraExpandFC2RanksDevice;
    at::Tensor mLoraExpandFC2WeightPtrsDevice;
    at::Tensor mLoraExpandGatedRanksDevice;
    at::Tensor mLoraExpandGatedWeightPtrsDevice;
    // Tracks how many entries were populated this call so the H2D copies only
    // the live portion. Per module; gated may be inactive for non-gated layers.
    int64_t mLoraExpandFC1Size = 0;
    int64_t mLoraExpandFC2Size = 0;
    int64_t mLoraExpandGatedSize = 0;
    // Highest max_num_tokens we have reserved storage for. Callers can bump this
    // via reserveLoraHostBuffers() before CUDA graph capture so subsequent
    // resizes do not reallocate and invalidate the captured copy addresses.
    int64_t mLoraHostBufCapacity = 0;

    // Set once a CUDA-graph capture has been observed on the LoRA path. After
    // that, growing the persistent scratch is forbidden even outside capture,
    // since a captured graph keeps replaying against the freed addresses.
    // Mutable so the const capture-safety check can record it.
    mutable bool mLoraCaptureObserved = false;

    // ---- Slot-indexed (CUDA-graph) device-source buffers ----
    // launchMoeLoraSlotExpand reads these device-resident slot tables and
    // token_to_slot and writes the per-source-token (rank, A_ptr, B_ptr) tables
    // into the mLoraExpand*Device mirrors above. A captured async H2D refreshes
    // them each step from the caller's stable pinned-host buffers, so replaying
    // the graph picks up new adapter assignments without re-capture.
    // token_to_slot is sized at max_num_tokens; the slot tables at max_lora_size.
    at::Tensor mLoraTokenToSlotDevice;    // [max_num_tokens]       int32
    at::Tensor mLoraSlotFC1RanksDevice;   // [max_lora_size]        int32
    at::Tensor mLoraSlotFC1PtrsDevice;    // [max_lora_size * 3]    int64
    at::Tensor mLoraSlotFC2RanksDevice;   // [max_lora_size]        int32
    at::Tensor mLoraSlotFC2PtrsDevice;    // [max_lora_size * 3]    int64
    at::Tensor mLoraSlotGatedRanksDevice; // [max_lora_size]        int32
    at::Tensor mLoraSlotGatedPtrsDevice;  // [max_lora_size * 3]    int64
    // Highest max_lora_size we have reserved slot-table storage for.
    int64_t mLoraSlotTableCapacity = 0;

    // Persistent device-resident scratch backing the capture-safe MoE LoRA
    // path. One LoraGroupedGemmBuffers per module (fc1, fc2, gated). All
    // at::Tensor members are allocated by ensureLoraDeviceScratch and reused
    // across calls so the addresses baked into a captured graph remain valid
    // for replay. Pointers from these tensors are packed into
    // LoraParams::grouped_gemm by buildMoeLoraParams when the grouped-GEMM core is used.
    struct LoraGroupedGemmBuffers
    {
        // Per-permuted-row (rank, A_ptr + offset, B_ptr + offset).
        at::Tensor permuted_ranks; // int32  [P_max]
        at::Tensor permuted_ptrs;  // int64  [2 * P_max]

        // Grouped-GEMM bundle. Concrete types restored at the LoraParams boundary.
        at::Tensor problem_sizes_in;  // int8   [P_max * sizeof(GemmCoord)]
        at::Tensor problem_sizes_out; // int8   [P_max * sizeof(GemmCoord)]
        at::Tensor a_ptrs_in;         // int64  [P_max]
        at::Tensor b_ptrs_in;         // int64  [P_max]
        at::Tensor d_ptrs_in;         // int64  [P_max]
        at::Tensor b_ptrs_out;        // int64  [P_max]
        at::Tensor d_ptrs_out;        // int64  [P_max]
        at::Tensor lda_in;            // int64  [P_max]
        at::Tensor ldb_in;            // int64  [P_max]
        at::Tensor ldd_in;            // int64  [P_max]
        at::Tensor ldb_out;           // int64  [P_max]
        at::Tensor ldd_out;           // int64  [P_max]
        at::Tensor splitk_offsets;    // int64  [P_max + 1]

        // GEMM data-flow buffers. The split-K in-GEMM's partial-sum scratch is
        // allocated internally by cuda_graph_split_k_grouped_gemm, so only the
        // low-rank intermediate is owned here.
        at::Tensor lowrank_workspace; // dtype  [P_max * max_lora_rank]

        // Pinned-host single GemmCoord upper bounds; required by the
        // cuda_graph_*_grouped_gemm wrappers for kernel selection.
        at::Tensor host_max_problem_in;  // int8 pinned [sizeof(GemmCoord)]
        at::Tensor host_max_problem_out; // int8 pinned [sizeof(GemmCoord)]
    };

    LoraGroupedGemmBuffers mFc1DeviceBuf;
    LoraGroupedGemmBuffers mFc2DeviceBuf;
    LoraGroupedGemmBuffers mGatedDeviceBuf;

    // Tracks the shape parameters baked into the current scratch
    // allocation. (Re)allocation is required if any of these grows or if
    // the dtype changes.
    int64_t mLoraDeviceScratchCapacity = 0; // P_max = max(num_tokens * top_k)
    int64_t mLoraDeviceScratchMaxLoraRank = 0;
    int64_t mLoraDeviceScratchDtypeBytes = 0;
    int64_t mLoraDeviceScratchSplitKSlices = 0;
    bool mLoraDeviceScratchHasGated = false;

    // Split-K slice count for the grouped-GEMM low-rank in-GEMM. Shared between
    // buildMoeLoraParams (lazy sizing) and reserveLoraHostBuffers (warmup
    // pre-sizing) so both reserve the same amount of scratch.
    static constexpr int64_t kGroupedGemmSplitKSlices = 16;

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
        kernels::MOEParallelismConfig const& parallelismConfig, bool min_latency_mode, cudaStream_t stream,
        bool use_lora = false)
    {
        size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts,
            experts_per_token, activation_type, parallelismConfig, use_lora, mUseDeepSeekFP8BlockScaling,
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

    // ===== LoRA helpers =====

    // Map a torch dtype to the TRT-LLM nvinfer1::DataType used to size the
    // grouped-GEMM low-rank scratch. Kept as a const member (not static) so the
    // FP8 case can read mOutputDtype to pick the fp16/bf16 LoRA compute dtype.
    nvinfer1::DataType loraTypeFromActDtype(c10::ScalarType dtype) const
    {
        switch (dtype)
        {
        case c10::ScalarType::Half: return nvinfer1::DataType::kHALF;
        case c10::ScalarType::Float: return nvinfer1::DataType::kFLOAT;
#ifdef ENABLE_BF16
        case c10::ScalarType::BFloat16: return nvinfer1::DataType::kBF16;
#endif
#ifdef ENABLE_FP8
        case c10::ScalarType::Float8_e4m3fn:
            TORCH_CHECK(mOutputDtype != c10::ScalarType::Float8_e4m3fn,
                "MoE LoRA with FP8 base activations requires an fp16/bf16 output (LoRA compute) dtype.");
            return loraTypeFromActDtype(mOutputDtype);
#endif
        default: C10_THROW_ERROR_FORMATTED(Error, "MoE LoRA only supports fp16/bf16/fp32 activation dtype.");
        }
    }

    // Expand per-request LoRA ranks and weight-pointer pairs into per-token arrays.
    // Mirrors the convention in cpp/tensorrt_llm/thop/loraOp.cpp lines 97-127 and
    // cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp.
    //
    // Inputs:
    //   ranks:                 cpu int32 [num_seqs]
    //   weight_ptrs:           cpu int64 [num_seqs, 3]  (A, B, optional-DoRA-magnitude; DoRA ignored)
    //   host_request_types:    cpu int32 [num_seqs]      (0=CONTEXT/prefill, 1=GENERATION)
    //   host_context_lengths:  cpu int32 [num_seqs]      (only read for CONTEXT requests)
    //   num_tokens:            total tokens flowing through this op (used as a consistency check)
    //
    // Outputs the two `expand_*` vectors with shapes [num_tokens] / [num_tokens * 2].
    // Writes the [num_tokens] expanded LoRA tables into the caller-owned
    // pinned-host buffers expand_ranks_data ([num_tokens] int32) and
    // expand_ptrs_data ([num_tokens * 2] int64; each pair is (A, B) as
    // raw pointer bits stored in int64). The buffers must already be
    // allocated to at least num_tokens / num_tokens * 2 elements.
    void expandPerRequestLoraTo(torch::Tensor const& ranks, torch::Tensor const& weight_ptrs,
        torch::Tensor const& host_request_types, torch::Tensor const& host_context_lengths, int64_t num_tokens,
        int32_t* expand_ranks_data, int64_t* expand_ptrs_data)
    {
        CHECK_CPU_INPUT(ranks, at::ScalarType::Int)
        CHECK_CPU_INPUT(weight_ptrs, at::ScalarType::Long)
        CHECK_CPU_INPUT(host_request_types, at::ScalarType::Int)
        CHECK_CPU_INPUT(host_context_lengths, at::ScalarType::Int)

        auto const num_seqs = static_cast<int64_t>(ranks.size(0));
        TORCH_CHECK(weight_ptrs.dim() == 2 && weight_ptrs.size(0) == num_seqs && weight_ptrs.size(1) == 3,
            "MoE LoRA weight_ptrs must have shape [num_seqs, 3] (A, B, optional DoRA); got ", weight_ptrs.sizes());
        TORCH_CHECK(host_request_types.size(0) == num_seqs, "MoE LoRA host_request_types must match ranks length. Got ",
            host_request_types.size(0), " vs ", num_seqs);
        TORCH_CHECK(host_context_lengths.size(0) == num_seqs,
            "MoE LoRA host_context_lengths must match ranks length. Got ", host_context_lengths.size(0), " vs ",
            num_seqs);

        auto const* rank_data = static_cast<int32_t const*>(ranks.data_ptr());
        auto const* ptr_data = static_cast<int64_t const*>(weight_ptrs.data_ptr());
        auto const* req_types = static_cast<int32_t const*>(host_request_types.data_ptr());
        auto const* ctx_lens = static_cast<int32_t const*>(host_context_lengths.data_ptr());

        int64_t produced = 0;
        for (int64_t req_id = 0; req_id < num_seqs; ++req_id)
        {
            int32_t const rank = rank_data[req_id];
            int64_t const a_ptr = ptr_data[req_id * 3 + 0];
            int64_t const b_ptr = ptr_data[req_id * 3 + 1];
            // ptr_data[req_id * 3 + 2] is the optional DoRA magnitude vector pointer; ignored here
            // (MoE+DoRA is rejected at load time, see tensorrt_llm/lora_manager.py).

            // Validate the raw request type before trusting it. An unexpected
            // value would otherwise fall into the CONTEXT branch and read an
            // arbitrary context length, producing a negative/garbage repeat.
            int32_t const req_type_raw = req_types[req_id];
            TORCH_CHECK(req_type_raw == static_cast<int32_t>(MoeLoraRequestType::kCONTEXT)
                    || req_type_raw == static_cast<int32_t>(MoeLoraRequestType::kGENERATION),
                "MoE LoRA host_request_types[", req_id, "] must be 0 (context) or 1 (generation); got ", req_type_raw);
            auto const req_type = static_cast<MoeLoraRequestType>(req_type_raw);
            if (req_type == MoeLoraRequestType::kCONTEXT)
            {
                TORCH_CHECK(ctx_lens[req_id] >= 0, "MoE LoRA host_context_lengths[", req_id,
                    "] must be non-negative; got ", ctx_lens[req_id]);
            }
            int64_t const repeat
                = (req_type == MoeLoraRequestType::kGENERATION) ? int64_t{1} : static_cast<int64_t>(ctx_lens[req_id]);
            // Guard the destination writes BEFORE producing them. expand_*_data
            // point at fixed-capacity pinned buffers sized for num_tokens, so a
            // malformed host_context_lengths (summing past num_tokens) must be a
            // clean error rather than an out-of-bounds write into pinned memory.
            TORCH_CHECK(repeat >= 0 && produced + repeat <= num_tokens, "MoE LoRA per-request expansion overran the ",
                num_tokens, "-token buffer at request ", req_id, " (produced ", produced, " + ", repeat,
                "). Check host_request_types / host_context_lengths against the op's token count.");
            for (int64_t i = 0; i < repeat; ++i)
            {
                int64_t const t = produced + i;
                expand_ranks_data[t] = rank;
                expand_ptrs_data[2 * t + 0] = a_ptr;
                expand_ptrs_data[2 * t + 1] = b_ptr;
            }
            produced += repeat;
        }
        TORCH_CHECK(produced == num_tokens, "MoE LoRA per-request expansion produced ", produced,
            " tokens but op input has ", num_tokens, " tokens.");
    }

    // Pre-allocate the pinned-host and persistent-device LoRA expansion
    // buffers at a fixed capacity. Required for CUDA-graph capture/replay
    // safety: the captured stream records cudaMemcpyAsync at the source
    // (pinned host) and destination (device) addresses observed during
    // capture, so those addresses must remain valid for the lifetime of the
    // graph, with no reallocation between captures.
    //
    // This method is public so it can be bound to Python; callers should
    // invoke it once during warmup before any CUDA-graph capture that
    // exercises routed-expert MoE LoRA. It is idempotent at or below the
    // current capacity.
public:
    // Pre-size every MoE-LoRA scratch buffer to the worst case so no
    // (re)allocation happens during CUDA-graph capture or replay: the
    // pinned-host and device-mirror expansion buffers (max_num_tokens), the
    // slot tables (max_lora_size), and the device grouped-GEMM scratch
    // (max_num_tokens * experts_per_token). Idempotent and grow-only.
    //
    // Call once during warmup, before any graph capture that exercises MoE
    // LoRA. experts_per_token is the routing top_k; max_lora_rank is the largest
    // LoRA rank across adapters; max_lora_size is the adapter-slot pool size;
    // has_gated is true for gated activations (e.g. SwiGLU).
    void reserveLoraHostBuffers(
        int64_t max_num_tokens, int64_t experts_per_token, int64_t max_lora_rank, int64_t max_lora_size, bool has_gated)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        TORCH_CHECK(max_num_tokens > 0, "max_num_tokens must be positive; got ", max_num_tokens);
        TORCH_CHECK(experts_per_token > 0, "experts_per_token must be positive; got ", experts_per_token);
        TORCH_CHECK(max_lora_size > 0, "max_lora_size must be positive; got ", max_lora_size);

        // Pinned-host and device-mirror expansion buffers, plus the device
        // token_to_slot mirror.
        if (max_num_tokens > mLoraHostBufCapacity)
        {
            // Reserve is meant to run during warmup (before any capture). If a
            // capture has already been observed, growing here would invalidate
            // addresses baked into the captured graph, so route it through the
            // same guard as the lazy growth paths.
            checkLoraReallocSafeDuringCapture(/*stream=*/nullptr, max_num_tokens, mLoraHostBufCapacity);
            ensureLoraExpandBuffers(max_num_tokens);
            mLoraHostBufCapacity = max_num_tokens;
        }

        // Device-resident slot tables for the slot-indexed device-expand kernel.
        if (max_lora_size > mLoraSlotTableCapacity)
        {
            checkLoraReallocSafeDuringCapture(/*stream=*/nullptr, max_lora_size, mLoraSlotTableCapacity);
            ensureLoraSlotTableBuffers(max_lora_size);
            mLoraSlotTableCapacity = max_lora_size;
        }

        // Grouped-GEMM device scratch. Pre-size it unconditionally; otherwise
        // the first capture would trigger a lazy allocation that the in-capture
        // realloc guard rejects.
        TORCH_CHECK(max_lora_rank > 0, "max_lora_rank must be positive to reserve MoE-LoRA device scratch; got ",
            max_lora_rank);
        int64_t const dtype_bytes = static_cast<int64_t>(common::getDTypeSize(loraTypeFromActDtype(mActivationDtype)));
        int64_t const capacity = max_num_tokens * experts_per_token;
        ensureLoraDeviceScratch(capacity, max_lora_rank, dtype_bytes, kGroupedGemmSplitKSlices, has_gated);
    }

private:
    // Reallocating MoE-LoRA scratch hands out fresh addresses, which silently
    // invalidates any CUDA graph that baked in the old ones. Reject reallocation
    // both while capturing and after any capture has been observed, since an
    // earlier graph keeps replaying. Callers invoke this only when reallocation
    // is imminent. No-op before the first capture (e.g. warmup pre-sizing).
    void checkLoraReallocSafeDuringCapture(cudaStream_t stream, int64_t requested, int64_t current) const
    {
        bool const capturing = (stream != nullptr && tensorrt_llm::common::isCapturing(stream));
        if (capturing)
        {
            mLoraCaptureObserved = true;
        }
        if (!capturing && !mLoraCaptureObserved)
        {
            return;
        }
        TORCH_CHECK(false, "MoE LoRA scratch (current capacity ", current, ") is too small for ", requested,
            capturing ? " entries during CUDA graph capture." : " entries after a CUDA graph capture was observed.",
            " Growing it would invalidate addresses baked into already-captured graphs. Call "
            "FusedMoeRunner.reserve_lora_host_buffers() during warmup, before capture, with the engine's worst-case "
            "shape.");
    }

    // Latch that a CUDA-graph capture has been observed as soon as any MoE LoRA
    // call runs on a capturing stream, even when no buffer growth happens during
    // that capture. This forbids later reallocation of buffers whose addresses
    // are baked into the captured graph, which would corrupt replay.
    void noteLoraCaptureState(cudaStream_t stream) const
    {
        if (stream != nullptr && tensorrt_llm::common::isCapturing(stream))
        {
            mLoraCaptureObserved = true;
        }
    }

    // Internal helper: (re)allocate the six pinned-host + six device tensor
    // pairs to hold capacity expanded tokens. Called by reserveLoraHostBuffers
    // (public warmup) and by buildMoeLoraParams (lazy on first capture-sized
    // call). The (re)allocation drops the previous storage; callers must make
    // sure any in-flight CUDA graph that references the old addresses has
    // either been destroyed or never replays again.
    void ensureLoraExpandBuffers(int64_t capacity)
    {
        auto const pinned_int_opts = at::TensorOptions().dtype(at::kInt).pinned_memory(true);
        auto const pinned_long_opts = at::TensorOptions().dtype(at::kLong).pinned_memory(true);
        auto const dev_int_opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
        auto const dev_long_opts = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);

        mLoraExpandFC1RanksPinned = at::empty({capacity}, pinned_int_opts);
        mLoraExpandFC2RanksPinned = at::empty({capacity}, pinned_int_opts);
        mLoraExpandGatedRanksPinned = at::empty({capacity}, pinned_int_opts);
        mLoraExpandFC1WeightPtrsPinned = at::empty({capacity * 2}, pinned_long_opts);
        mLoraExpandFC2WeightPtrsPinned = at::empty({capacity * 2}, pinned_long_opts);
        mLoraExpandGatedWeightPtrsPinned = at::empty({capacity * 2}, pinned_long_opts);

        mLoraExpandFC1RanksDevice = at::empty({capacity}, dev_int_opts);
        mLoraExpandFC2RanksDevice = at::empty({capacity}, dev_int_opts);
        mLoraExpandGatedRanksDevice = at::empty({capacity}, dev_int_opts);
        mLoraExpandFC1WeightPtrsDevice = at::empty({capacity * 2}, dev_long_opts);
        mLoraExpandFC2WeightPtrsDevice = at::empty({capacity * 2}, dev_long_opts);
        mLoraExpandGatedWeightPtrsDevice = at::empty({capacity * 2}, dev_long_opts);

        // Device token_to_slot mirror for the slot-indexed device-expand kernel.
        // Sized with the expand buffers (one entry per source token).
        mLoraTokenToSlotDevice = at::empty({capacity}, dev_int_opts);
    }

    // (Re)allocate the device-resident slot tables consumed by
    // launchMoeLoraSlotExpand. Sized at the adapter-slot pool (max_lora_size).
    // Idempotent / grow-only; reallocation drops previous storage, so callers
    // must not grow this while a graph referencing the old addresses can still
    // replay (the in-capture guard enforces this).
    void ensureLoraSlotTableBuffers(int64_t max_lora_size)
    {
        auto const dev_int_opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
        auto const dev_long_opts = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);

        mLoraSlotFC1RanksDevice = at::empty({max_lora_size}, dev_int_opts);
        mLoraSlotFC2RanksDevice = at::empty({max_lora_size}, dev_int_opts);
        mLoraSlotGatedRanksDevice = at::empty({max_lora_size}, dev_int_opts);
        mLoraSlotFC1PtrsDevice = at::empty({max_lora_size * 3}, dev_long_opts);
        mLoraSlotFC2PtrsDevice = at::empty({max_lora_size * 3}, dev_long_opts);
        mLoraSlotGatedPtrsDevice = at::empty({max_lora_size * 3}, dev_long_opts);
    }

    // Allocate the per-module grouped-GEMM scratch for the capture-safe LoRA
    // path. The buffers are sized in permuted tokens (P = num_tokens * top_k)
    // and the per-token LoRA rank upper bound max_lora_rank; both feed the
    // pointer-expand, problem-builder, and cuda_graph_*_grouped_gemm kernels.
    //
    // The function is idempotent at or below the current capacity and
    // reallocates only when one of (capacity, max_lora_rank, dtype_bytes,
    // splitk_slices, has_gated) grows. Reallocation drops the previous storage,
    // so callers must ensure any in-flight CUDA graph referencing the old
    // addresses has been destroyed or will not replay.
    //
    // The host-side max-problem-size pins hold one GemmCoord each; the value is
    // a worst-case upper bound, independent of per-call data.
    void ensureLoraDeviceScratch(int64_t capacity, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices,
        bool has_gated, cudaStream_t stream = nullptr)
    {
        TORCH_CHECK(capacity > 0, "grouped-GEMM capacity must be positive; got ", capacity);
        TORCH_CHECK(max_lora_rank > 0, "grouped-GEMM max_lora_rank must be positive; got ", max_lora_rank);
        TORCH_CHECK(dtype_bytes > 0, "grouped-GEMM dtype_bytes must be positive; got ", dtype_bytes);
        TORCH_CHECK(splitk_slices > 0, "grouped-GEMM splitk_slices must be positive; got ", splitk_slices);

        bool const need_resize = capacity > mLoraDeviceScratchCapacity || max_lora_rank > mLoraDeviceScratchMaxLoraRank
            || dtype_bytes != mLoraDeviceScratchDtypeBytes || splitk_slices != mLoraDeviceScratchSplitKSlices
            || (has_gated && !mLoraDeviceScratchHasGated);
        if (!need_resize)
        {
            return;
        }
        // Refuse to grow device scratch mid-capture (see helper for rationale).
        checkLoraReallocSafeDuringCapture(stream, capacity, mLoraDeviceScratchCapacity);

        // Grow each field to the requested upper bound and remember the
        // dtype/rank/splitk combo so subsequent calls can early-exit.
        int64_t const new_capacity = std::max(capacity, mLoraDeviceScratchCapacity);
        int64_t const new_max_lora_rank = std::max(max_lora_rank, mLoraDeviceScratchMaxLoraRank);
        bool const new_has_gated = mLoraDeviceScratchHasGated || has_gated;

        // c10::ScalarType for the lowrank workspace. The kernel treats the
        // buffer opaquely (per-byte stride is dtype_bytes), so we pick a
        // dtype with matching element size to keep at::Tensor accounting
        // sensible; consumers cast via .data_ptr().
        c10::ScalarType const dtype_scalar = (dtype_bytes == 2) ? at::kBFloat16
            : (dtype_bytes == 4)                                ? at::kFloat
                                                                : at::kByte;
        // Callers should pass bf16/fp16 (2 bytes). Other sizes still work at the
        // byte level, but this assertion catches accidental misuse.
        TORCH_CHECK(dtype_bytes == 1 || dtype_bytes == 2 || dtype_bytes == 4,
            "grouped-GEMM lowrank workspace dtype_bytes must be 1/2/4; got ", dtype_bytes);

        auto const dev_int8_opts = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
        auto const dev_int32_opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
        auto const dev_int64_opts = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
        auto const dev_dtype_opts = at::TensorOptions().dtype(dtype_scalar).device(at::kCUDA);
        auto const pinned_int8_opts = at::TensorOptions().dtype(at::kByte).pinned_memory(true);

        // sizeof(cutlass::gemm::GemmCoord) == sizeof(int) * 3 in practice;
        // we ask for the exact byte count at allocation time so the bound
        // tracks any cutlass struct-layout change.
        int64_t const gemm_coord_bytes = static_cast<int64_t>(sizeof(cutlass::gemm::GemmCoord));

        auto alloc_one = [&](LoraGroupedGemmBuffers& mod)
        {
            mod.permuted_ranks = at::empty({new_capacity}, dev_int32_opts);
            mod.permuted_ptrs = at::empty({new_capacity * 2}, dev_int64_opts);

            mod.problem_sizes_in = at::empty({new_capacity * gemm_coord_bytes}, dev_int8_opts);
            mod.problem_sizes_out = at::empty({new_capacity * gemm_coord_bytes}, dev_int8_opts);

            mod.a_ptrs_in = at::empty({new_capacity}, dev_int64_opts);
            mod.b_ptrs_in = at::empty({new_capacity}, dev_int64_opts);
            mod.d_ptrs_in = at::empty({new_capacity}, dev_int64_opts);
            mod.b_ptrs_out = at::empty({new_capacity}, dev_int64_opts);
            mod.d_ptrs_out = at::empty({new_capacity}, dev_int64_opts);

            mod.lda_in = at::empty({new_capacity}, dev_int64_opts);
            mod.ldb_in = at::empty({new_capacity}, dev_int64_opts);
            mod.ldd_in = at::empty({new_capacity}, dev_int64_opts);
            mod.ldb_out = at::empty({new_capacity}, dev_int64_opts);
            mod.ldd_out = at::empty({new_capacity}, dev_int64_opts);
            mod.splitk_offsets = at::empty({new_capacity + 1}, dev_int64_opts);

            mod.lowrank_workspace = at::empty({new_capacity * new_max_lora_rank}, dev_dtype_opts);

            mod.host_max_problem_in = at::empty({gemm_coord_bytes}, pinned_int8_opts);
            mod.host_max_problem_out = at::empty({gemm_coord_bytes}, pinned_int8_opts);
        };

        alloc_one(mFc1DeviceBuf);
        alloc_one(mFc2DeviceBuf);
        if (new_has_gated)
        {
            alloc_one(mGatedDeviceBuf);
        }

        mLoraDeviceScratchCapacity = new_capacity;
        mLoraDeviceScratchMaxLoraRank = new_max_lora_rank;
        mLoraDeviceScratchDtypeBytes = dtype_bytes;
        mLoraDeviceScratchSplitKSlices = splitk_slices;
        mLoraDeviceScratchHasGated = new_has_gated;
    }

    // Pack the per-module at::Tensor scratch into the typed pointer bundle
    // attached to LoraParams. The buffers are owned by FusedMoeRunner, so the
    // resulting pointers stay valid as long as the runner outlives the
    // LoraParams use. dim_a/dim_b, ranks_src_dev, and out_hidden_size are filled
    // in by buildMoeLoraParams; the output base is passed directly to
    // runMoeLoraGroupedGemmModule at the call site.
    void populateLoraGroupedGemmModule(
        LoraGroupedGemmBuffers& mod, ::tensorrt_llm::kernels::cutlass_kernels::MoeLoraGroupedGemmModule& out) const
    {
        out.permuted_ranks_dev = mod.permuted_ranks.data_ptr<int32_t>();
        out.permuted_ptrs_dev = mod.permuted_ptrs.data_ptr<int64_t>();

        out.problem_sizes_in_dev = mod.problem_sizes_in.data_ptr();
        out.problem_sizes_out_dev = mod.problem_sizes_out.data_ptr();
        out.a_ptrs_in_dev = reinterpret_cast<void**>(mod.a_ptrs_in.data_ptr<int64_t>());
        out.b_ptrs_in_dev = reinterpret_cast<void**>(mod.b_ptrs_in.data_ptr<int64_t>());
        out.d_ptrs_in_dev = reinterpret_cast<void**>(mod.d_ptrs_in.data_ptr<int64_t>());
        out.b_ptrs_out_dev = reinterpret_cast<void**>(mod.b_ptrs_out.data_ptr<int64_t>());
        out.d_ptrs_out_dev = reinterpret_cast<void**>(mod.d_ptrs_out.data_ptr<int64_t>());
        out.lda_in_dev = mod.lda_in.data_ptr<int64_t>();
        out.ldb_in_dev = mod.ldb_in.data_ptr<int64_t>();
        out.ldd_in_dev = mod.ldd_in.data_ptr<int64_t>();
        out.ldb_out_dev = mod.ldb_out.data_ptr<int64_t>();
        out.ldd_out_dev = mod.ldd_out.data_ptr<int64_t>();
        out.splitk_offsets_dev = mod.splitk_offsets.data_ptr<int64_t>();

        out.lowrank_workspace_dev = mod.lowrank_workspace.data_ptr();
        out.host_max_problem_in_pinned = mod.host_max_problem_in.data_ptr();
        out.host_max_problem_out_pinned = mod.host_max_problem_out.data_ptr();

        // out_hidden_size is set by buildMoeLoraParams; default it here.
        out.out_hidden_size = 0;
    }

    // Build a populated LoraParams from the optional CPU tensors and wire the
    // grouped-GEMM core scratch (lora_params.grouped_gemm). Returns std::nullopt
    // when LoRA is inactive (no fc1 ranks tensor). Mutates the mLoraExpand*
    // pinned tensors and queues an async H2D into the device mirrors on stream.
    std::optional<::tensorrt_llm::kernels::LoraParams> buildMoeLoraParams(
        torch::optional<torch::Tensor> const& fc1_lora_ranks,
        torch::optional<torch::Tensor> const& fc1_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& fc2_lora_ranks,
        torch::optional<torch::Tensor> const& fc2_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& gated_lora_ranks,
        torch::optional<torch::Tensor> const& gated_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& host_request_types,
        torch::optional<torch::Tensor> const& host_context_lengths,
        torch::optional<torch::Tensor> const& fc1_slot_lora_ranks,
        torch::optional<torch::Tensor> const& fc1_slot_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& fc2_slot_lora_ranks,
        torch::optional<torch::Tensor> const& fc2_slot_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& gated_slot_lora_ranks,
        torch::optional<torch::Tensor> const& gated_slot_lora_weight_ptrs,
        torch::optional<torch::Tensor> const& token_to_slot, int64_t num_tokens, int64_t hidden_size,
        int64_t inter_size, c10::ScalarType act_dtype, int64_t lora_max_low_rank, bool is_gated_activation,
        cudaStream_t stream, int experts_per_token)
    {
        bool const has_per_request = fc1_lora_ranks.has_value();
        bool const has_slot_indexed = fc1_slot_lora_ranks.has_value();
        if (!has_per_request && !has_slot_indexed)
        {
            return std::nullopt;
        }
        TORCH_CHECK(lora_max_low_rank > 0, "MoE LoRA requires lora_max_low_rank > 0; got ", lora_max_low_rank);

        // Latch capture state up front so that even a capture which needs no
        // buffer growth still forbids later reallocation of these addresses.
        noteLoraCaptureState(stream);

        // num_seqs: for the per-request path, the actual request count; for the
        // slot-indexed path, the active token count (each token is its own "seq"
        // from the kernel's per-token-pointer-array perspective).
        int64_t num_seqs = 0;
        bool has_gated = false;

        if (has_per_request)
        {
            TORCH_CHECK(
                fc1_lora_weight_ptrs.has_value() && fc2_lora_ranks.has_value() && fc2_lora_weight_ptrs.has_value(),
                "MoE LoRA requires fc1_lora_ranks/fc1_lora_weight_ptrs/fc2_lora_ranks/fc2_lora_weight_ptrs together.");
            TORCH_CHECK(host_request_types.has_value() && host_context_lengths.has_value(),
                "MoE LoRA requires host_request_types and host_context_lengths CPU tensors.");
            // For gated activations (e.g. SwiGLU) the kernel's setupLoraWorkspace
            // unconditionally dereferences gated_lora_ranks and
            // gated_lora_weight_ptrs, so the caller must provide them.
            if (is_gated_activation)
            {
                TORCH_CHECK(gated_lora_ranks.has_value() && gated_lora_weight_ptrs.has_value(),
                    "MoE LoRA with a gated activation (e.g. SwiGLU) requires gated_lora_ranks and "
                    "gated_lora_weight_ptrs to be provided alongside fc1_lora_*. The fused-MoE kernel "
                    "expects three LoRA modules per layer: fc1 (gate/SiLU), gated (up/linear) and fc2 (down).");
            }
            else
            {
                TORCH_CHECK(!gated_lora_ranks.has_value(),
                    "MoE LoRA gated_lora_* is only supported for gated activations (e.g. SwiGLU).");
            }

            num_seqs = fc1_lora_ranks->size(0);
            has_gated = is_gated_activation && gated_lora_ranks.has_value();

            // Every per-request rank must fit within lora_max_low_rank, which sizes
            // both the lowrank workspace and the max-problem hints. A larger rank
            // would make the grouped-GEMM core build GEMM problems wider than
            // the allocated scratch and write out of bounds, so reject it up front.
            auto validate_rank_tensor = [&](char const* name, torch::Tensor const& ranks_tensor)
            {
                CHECK_CPU_INPUT(ranks_tensor, at::ScalarType::Int)
                auto const* rank_data = ranks_tensor.data_ptr<int32_t>();
                for (int64_t i = 0; i < ranks_tensor.size(0); ++i)
                {
                    TORCH_CHECK(rank_data[i] >= 0 && rank_data[i] <= lora_max_low_rank, name, "[", i,
                        "]=", rank_data[i], " is outside [0, ", lora_max_low_rank, "].");
                }
            };
            validate_rank_tensor("fc1_lora_ranks", *fc1_lora_ranks);
            validate_rank_tensor("fc2_lora_ranks", *fc2_lora_ranks);
            if (has_gated)
            {
                validate_rank_tensor("gated_lora_ranks", *gated_lora_ranks);
            }

            // Ensure pinned/device buffers can hold num_tokens entries.
            // Reserve is idempotent at-or-below current capacity.
            if (num_tokens > mLoraHostBufCapacity)
            {
                checkLoraReallocSafeDuringCapture(stream, num_tokens, mLoraHostBufCapacity);
                ensureLoraExpandBuffers(num_tokens);
                mLoraHostBufCapacity = num_tokens;
            }

            expandPerRequestLoraTo(*fc1_lora_ranks, *fc1_lora_weight_ptrs, *host_request_types, *host_context_lengths,
                num_tokens, mLoraExpandFC1RanksPinned.data_ptr<int32_t>(),
                mLoraExpandFC1WeightPtrsPinned.data_ptr<int64_t>());
            expandPerRequestLoraTo(*fc2_lora_ranks, *fc2_lora_weight_ptrs, *host_request_types, *host_context_lengths,
                num_tokens, mLoraExpandFC2RanksPinned.data_ptr<int32_t>(),
                mLoraExpandFC2WeightPtrsPinned.data_ptr<int64_t>());
            mLoraExpandFC1Size = num_tokens;
            mLoraExpandFC2Size = num_tokens;
            if (has_gated)
            {
                expandPerRequestLoraTo(*gated_lora_ranks, *gated_lora_weight_ptrs, *host_request_types,
                    *host_context_lengths, num_tokens, mLoraExpandGatedRanksPinned.data_ptr<int32_t>(),
                    mLoraExpandGatedWeightPtrsPinned.data_ptr<int64_t>());
                mLoraExpandGatedSize = num_tokens;
            }
            else
            {
                mLoraExpandGatedSize = 0;
            }
        }
        else
        {
            // Slot-indexed (CUDA-graph) path. The token-level expansion is driven by
            // token_to_slot and the per-slot LoRA tables.
            TORCH_CHECK(fc1_slot_lora_weight_ptrs.has_value() && fc2_slot_lora_ranks.has_value()
                    && fc2_slot_lora_weight_ptrs.has_value() && token_to_slot.has_value(),
                "MoE LoRA slot-indexed mode requires fc1_slot_lora_ranks/fc1_slot_lora_weight_ptrs/"
                "fc2_slot_lora_ranks/fc2_slot_lora_weight_ptrs/token_to_slot together.");
            // For gated activations (e.g. SwiGLU) the kernel's setupLoraWorkspace
            // unconditionally dereferences gated_lora_ranks and
            // gated_lora_weight_ptrs, so the caller must provide the gated slot
            // tables as well.
            if (is_gated_activation)
            {
                TORCH_CHECK(gated_slot_lora_ranks.has_value() && gated_slot_lora_weight_ptrs.has_value(),
                    "MoE LoRA slot-indexed mode with a gated activation requires gated_slot_lora_ranks and "
                    "gated_slot_lora_weight_ptrs alongside fc1_slot_lora_*.");
            }
            else
            {
                TORCH_CHECK(!gated_slot_lora_ranks.has_value(),
                    "MoE LoRA gated_slot_lora_* is only supported for gated activations.");
            }
            // Ensure pinned/device buffers can hold num_tokens entries.
            // Idempotent at-or-below current capacity. Performed BEFORE the
            // first H2D so that addresses are stable for any subsequent
            // CUDA-graph capture; callers should invoke reserveLoraHostBuffers
            // during warmup to avoid the lazy reallocation here.
            if (num_tokens > mLoraHostBufCapacity)
            {
                checkLoraReallocSafeDuringCapture(stream, num_tokens, mLoraHostBufCapacity);
                ensureLoraExpandBuffers(num_tokens);
                mLoraHostBufCapacity = num_tokens;
            }

            num_seqs = num_tokens;
            has_gated = is_gated_activation && gated_slot_lora_ranks.has_value();

            // Slot tables and token_to_slot must be int32/int64 host tensors so
            // the H2D element sizes are correct and the device kernel reads the
            // expected types.
            CHECK_CPU_INPUT((*token_to_slot), at::ScalarType::Int)
            CHECK_CPU_INPUT((*fc1_slot_lora_ranks), at::ScalarType::Int)
            CHECK_CPU_INPUT((*fc1_slot_lora_weight_ptrs), at::ScalarType::Long)
            CHECK_CPU_INPUT((*fc2_slot_lora_ranks), at::ScalarType::Int)
            CHECK_CPU_INPUT((*fc2_slot_lora_weight_ptrs), at::ScalarType::Long)

            // These tensors are copied to device via async H2D. Under CUDA-graph
            // capture the source must be pinned, otherwise the copy is not
            // capturable and replay would read from a freed staging buffer. In
            // eager mode pageable host memory is fine.
            bool const slot_capturing = (stream != nullptr && tensorrt_llm::common::isCapturing(stream));
            auto check_pinned = [slot_capturing](at::Tensor const& t, char const* name)
            {
                TORCH_CHECK(!slot_capturing || t.is_pinned(), "MoE LoRA ", name,
                    " must be a pinned host tensor for captured async H2D copies.");
            };
            check_pinned(*token_to_slot, "token_to_slot");
            check_pinned(*fc1_slot_lora_ranks, "fc1_slot_lora_ranks");
            check_pinned(*fc1_slot_lora_weight_ptrs, "fc1_slot_lora_weight_ptrs");
            check_pinned(*fc2_slot_lora_ranks, "fc2_slot_lora_ranks");
            check_pinned(*fc2_slot_lora_weight_ptrs, "fc2_slot_lora_weight_ptrs");

            int64_t const num_slots = fc1_slot_lora_ranks->size(0);
            TORCH_CHECK(fc1_slot_lora_weight_ptrs->dim() == 2 && fc1_slot_lora_weight_ptrs->size(0) == num_slots
                    && fc1_slot_lora_weight_ptrs->size(1) == 3,
                "MoE LoRA fc1_slot_lora_weight_ptrs must have shape [max_lora_size, 3]; got ",
                fc1_slot_lora_weight_ptrs->sizes());
            TORCH_CHECK(fc2_slot_lora_ranks->size(0) == num_slots,
                "MoE LoRA fc2_slot_lora_ranks must match fc1 max_lora_size (", num_slots, "); got ",
                fc2_slot_lora_ranks->size(0));
            // fc2 pointer table is copied as num_slots * 3 elements below, so its
            // [max_lora_size, 3] shape must be validated before the raw H2D to
            // avoid reading past the source allocation / misaligning slots.
            TORCH_CHECK(fc2_slot_lora_weight_ptrs->dim() == 2 && fc2_slot_lora_weight_ptrs->size(0) == num_slots
                    && fc2_slot_lora_weight_ptrs->size(1) == 3,
                "MoE LoRA fc2_slot_lora_weight_ptrs must have shape [max_lora_size, 3]; got ",
                fc2_slot_lora_weight_ptrs->sizes());
            TORCH_CHECK(token_to_slot->size(0) >= num_tokens, "MoE LoRA token_to_slot length (", token_to_slot->size(0),
                ") must be >= num_tokens (", num_tokens, ").");

            // Ensure the device slot tables can hold num_slots entries.
            if (num_slots > mLoraSlotTableCapacity)
            {
                checkLoraReallocSafeDuringCapture(stream, num_slots, mLoraSlotTableCapacity);
                ensureLoraSlotTableBuffers(num_slots);
                mLoraSlotTableCapacity = num_slots;
            }

            // Capture-safe device slot->token expansion. Copy the caller's stable
            // pinned slot tables and token_to_slot into persistent device buffers
            // via captured async H2D, then run launchMoeLoraSlotExpand to produce
            // the per-source-token (rank, A, B) mirrors on-device. Both the copies
            // and the kernel are recorded into the graph, so replaying picks up
            // in-place slot-table / token_to_slot updates without re-capture
            // (mirroring the attention-LoRA device tables).
            auto h2d_slot = [&](at::Tensor const& src, at::Tensor& dst, int64_t numel)
            {
                TLLM_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                    static_cast<size_t>(numel) * src.element_size(), cudaMemcpyHostToDevice, stream));
            };
            h2d_slot(*token_to_slot, mLoraTokenToSlotDevice, num_tokens);
            h2d_slot(*fc1_slot_lora_ranks, mLoraSlotFC1RanksDevice, num_slots);
            h2d_slot(*fc1_slot_lora_weight_ptrs, mLoraSlotFC1PtrsDevice, num_slots * 3);
            h2d_slot(*fc2_slot_lora_ranks, mLoraSlotFC2RanksDevice, num_slots);
            h2d_slot(*fc2_slot_lora_weight_ptrs, mLoraSlotFC2PtrsDevice, num_slots * 3);
            if (has_gated)
            {
                CHECK_CPU_INPUT((*gated_slot_lora_ranks), at::ScalarType::Int)
                CHECK_CPU_INPUT((*gated_slot_lora_weight_ptrs), at::ScalarType::Long)
                check_pinned(*gated_slot_lora_ranks, "gated_slot_lora_ranks");
                check_pinned(*gated_slot_lora_weight_ptrs, "gated_slot_lora_weight_ptrs");
                TORCH_CHECK(gated_slot_lora_ranks->size(0) == num_slots,
                    "MoE LoRA gated_slot_lora_ranks must match fc1 max_lora_size (", num_slots, "); got ",
                    gated_slot_lora_ranks->size(0));
                TORCH_CHECK(gated_slot_lora_weight_ptrs->dim() == 2 && gated_slot_lora_weight_ptrs->size(0) == num_slots
                        && gated_slot_lora_weight_ptrs->size(1) == 3,
                    "MoE LoRA gated_slot_lora_weight_ptrs must have shape [max_lora_size, 3]; got ",
                    gated_slot_lora_weight_ptrs->sizes());
                h2d_slot(*gated_slot_lora_ranks, mLoraSlotGatedRanksDevice, num_slots);
                h2d_slot(*gated_slot_lora_weight_ptrs, mLoraSlotGatedPtrsDevice, num_slots * 3);
            }

            namespace ck = ::tensorrt_llm::kernels::cutlass_kernels;
            ck::MoeLoraSlotExpandModule fc1_slot_mod{mLoraSlotFC1RanksDevice.data_ptr<int32_t>(),
                mLoraSlotFC1PtrsDevice.data_ptr<int64_t>(), mLoraExpandFC1RanksDevice.data_ptr<int32_t>(),
                mLoraExpandFC1WeightPtrsDevice.data_ptr<int64_t>()};
            ck::MoeLoraSlotExpandModule fc2_slot_mod{mLoraSlotFC2RanksDevice.data_ptr<int32_t>(),
                mLoraSlotFC2PtrsDevice.data_ptr<int64_t>(), mLoraExpandFC2RanksDevice.data_ptr<int32_t>(),
                mLoraExpandFC2WeightPtrsDevice.data_ptr<int64_t>()};
            ck::MoeLoraSlotExpandModule gated_slot_mod{};
            if (has_gated)
            {
                gated_slot_mod = ck::MoeLoraSlotExpandModule{mLoraSlotGatedRanksDevice.data_ptr<int32_t>(),
                    mLoraSlotGatedPtrsDevice.data_ptr<int64_t>(), mLoraExpandGatedRanksDevice.data_ptr<int32_t>(),
                    mLoraExpandGatedWeightPtrsDevice.data_ptr<int64_t>()};
            }
            ck::launchMoeLoraSlotExpand(mLoraTokenToSlotDevice.data_ptr<int32_t>(), num_tokens, num_slots, fc1_slot_mod,
                fc2_slot_mod, has_gated ? &gated_slot_mod : nullptr, stream);

            // The slot path fills the device mirrors directly via the kernel, so
            // the per-request host expand buffers are unused. Mark their live
            // size 0 so the generic per-request H2D
            // below is skipped and does not clobber the kernel output.
            mLoraExpandFC1Size = 0;
            mLoraExpandFC2Size = 0;
            mLoraExpandGatedSize = 0;
        }

        // Queue an async H2D into the persistent device mirrors that
        // launchMoeLoraPointerExpand consumes. The copy source is pinned, so the
        // copy is truly async and capturable, and the destination is a
        // persistent device buffer with an address stable across captures.
        auto issue_h2d = [&](at::Tensor const& src, at::Tensor& dst, int64_t numel)
        {
            if (numel == 0)
            {
                return;
            }
            TLLM_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                static_cast<size_t>(numel) * src.element_size(), cudaMemcpyHostToDevice, stream));
        };
        issue_h2d(mLoraExpandFC1RanksPinned, mLoraExpandFC1RanksDevice, mLoraExpandFC1Size);
        issue_h2d(mLoraExpandFC1WeightPtrsPinned, mLoraExpandFC1WeightPtrsDevice, mLoraExpandFC1Size * 2);
        issue_h2d(mLoraExpandFC2RanksPinned, mLoraExpandFC2RanksDevice, mLoraExpandFC2Size);
        issue_h2d(mLoraExpandFC2WeightPtrsPinned, mLoraExpandFC2WeightPtrsDevice, mLoraExpandFC2Size * 2);
        issue_h2d(mLoraExpandGatedRanksPinned, mLoraExpandGatedRanksDevice, mLoraExpandGatedSize);
        issue_h2d(mLoraExpandGatedWeightPtrsPinned, mLoraExpandGatedWeightPtrsDevice, mLoraExpandGatedSize * 2);

        // The grouped-GEMM core reads its inputs from the device mirrors set up
        // below, so the cuBLAS LoraImpl pointers and host-sync event on
        // LoraParams (used by the TensorRT MoE plugin) are left null here.
        ::tensorrt_llm::kernels::LoraParams lora_params{
            static_cast<int>(num_seqs),
            mLoraExpandFC1RanksPinned.data_ptr<int32_t>(),
            reinterpret_cast<void const**>(mLoraExpandFC1WeightPtrsPinned.data_ptr<int64_t>()),
            mLoraExpandFC2RanksPinned.data_ptr<int32_t>(),
            reinterpret_cast<void const**>(mLoraExpandFC2WeightPtrsPinned.data_ptr<int64_t>()),
            /*fc1_lora_impl=*/nullptr,
            /*fc2_lora_impl=*/nullptr,
            /*workspace=*/nullptr,
            /*memcpy_event_ptr=*/nullptr,
            has_gated ? mLoraExpandGatedRanksPinned.data_ptr<int32_t>() : nullptr,
            has_gated ? reinterpret_cast<void const**>(mLoraExpandGatedWeightPtrsPinned.data_ptr<int64_t>()) : nullptr,
        };

        // Allocate the per-module device-resident scratch and pack its pointers
        // into lora_params.grouped_gemm. Both eager (per-request) and CUDA-graph
        // (slot-indexed) inputs feed this capture-safe core. The TensorRT MoE
        // plugin leaves grouped_gemm.enabled false and uses its own cuBLAS path.
        {
            int64_t const dtype_bytes = static_cast<int64_t>(common::getDTypeSize(loraTypeFromActDtype(act_dtype)));
            int64_t const capacity = num_tokens * static_cast<int64_t>(experts_per_token);
            // Pass stream so a mid-capture resize (which would invalidate
            // previously captured graphs) is rejected with a clear error
            // rather than silently corrupting replay.
            ensureLoraDeviceScratch(capacity, lora_max_low_rank, dtype_bytes, kGroupedGemmSplitKSlices,
                /*has_gated=*/has_gated, stream);

            auto& grouped_gemm = lora_params.grouped_gemm;
            grouped_gemm.enabled = true;
            grouped_gemm.in_hidden_size = hidden_size;
            grouped_gemm.max_lora_rank = lora_max_low_rank;
            grouped_gemm.dtype_bytes = dtype_bytes;
            grouped_gemm.splitk_slices = kGroupedGemmSplitKSlices;
            grouped_gemm.has_gated = has_gated;
            // Populate the libtorch-bound GEMM dispatch entry point so
            // runMoeLoraGroupedGemmModule in moe_kernels.cu can call through
            // it without dragging libtorch into libmoe_gemm_src.a.
            grouped_gemm.run = &moeLoraGroupedGemmRunImpl;
            populateLoraGroupedGemmModule(mFc1DeviceBuf, grouped_gemm.fc1);
            populateLoraGroupedGemmModule(mFc2DeviceBuf, grouped_gemm.fc2);
            if (has_gated)
            {
                populateLoraGroupedGemmModule(mGatedDeviceBuf, grouped_gemm.gated);
            }

            // Per-module dim_a/dim_b describe the LoRA adapter shape the
            // pointer-expand kernel offsets into; per-module out_hidden_size
            // describes the LoRA delta sink the problem-builder kernel writes
            // into. The runner passes the output base (lora_fc1_result_ /
            // lora_fc2_result_ / lora_gated_out) directly to
            // runMoeLoraGroupedGemmModule at the loraFC1/loraFC2 call sites so the
            // GEMMs land where the downstream bias/reorder kernels expect.
            //
            // For fc1 (and gated): adapter A is [hidden, rank], B is [rank, inter].
            // For fc2:             adapter A is [inter,  rank], B is [rank, hidden].
            grouped_gemm.fc1.dim_a = hidden_size;
            grouped_gemm.fc1.dim_b = inter_size;
            grouped_gemm.fc1.ranks_src_dev = mLoraExpandFC1RanksDevice.data_ptr<int32_t>();
            grouped_gemm.fc1.ptrs_src_dev = mLoraExpandFC1WeightPtrsDevice.data_ptr<int64_t>();
            grouped_gemm.fc1.out_hidden_size = inter_size;

            grouped_gemm.fc2.dim_a = inter_size;
            grouped_gemm.fc2.dim_b = hidden_size;
            grouped_gemm.fc2.ranks_src_dev = mLoraExpandFC2RanksDevice.data_ptr<int32_t>();
            grouped_gemm.fc2.ptrs_src_dev = mLoraExpandFC2WeightPtrsDevice.data_ptr<int64_t>();
            grouped_gemm.fc2.out_hidden_size = hidden_size;

            if (has_gated)
            {
                grouped_gemm.gated.dim_a = hidden_size;
                grouped_gemm.gated.dim_b = inter_size;
                grouped_gemm.gated.ranks_src_dev = mLoraExpandGatedRanksDevice.data_ptr<int32_t>();
                grouped_gemm.gated.ptrs_src_dev = mLoraExpandGatedWeightPtrsDevice.data_ptr<int64_t>();
                grouped_gemm.gated.out_hidden_size = inter_size;
            }

            // Pinned-host max-problem-size hints used by cuda_graph_*_grouped_gemm
            // for kernel selection. Values are upper bounds safe to fix at
            // warmup time (M=1 since each problem is one row; N/K depend on
            // module direction and max_lora_rank).
            auto fill_max_problem = [](void* host_ptr, int m, int n, int k)
            {
                auto* coord = static_cast<cutlass::gemm::GemmCoord*>(host_ptr);
                *coord = cutlass::gemm::GemmCoord(m, n, k);
            };
            // In-GEMM: M=1, N=max_lora_rank, K=in_dim. Out-GEMM: M=1, N=out_dim, K=max_lora_rank.
            fill_max_problem(grouped_gemm.fc1.host_max_problem_in_pinned, 1, lora_max_low_rank, hidden_size);
            fill_max_problem(grouped_gemm.fc1.host_max_problem_out_pinned, 1, inter_size, lora_max_low_rank);
            fill_max_problem(grouped_gemm.fc2.host_max_problem_in_pinned, 1, lora_max_low_rank, inter_size);
            fill_max_problem(grouped_gemm.fc2.host_max_problem_out_pinned, 1, hidden_size, lora_max_low_rank);
            if (has_gated)
            {
                fill_max_problem(grouped_gemm.gated.host_max_problem_in_pinned, 1, lora_max_low_rank, hidden_size);
                fill_max_problem(grouped_gemm.gated.host_max_problem_out_pinned, 1, inter_size, lora_max_low_rank);
            }
        }

        return lora_params;
    }

    kernels::QuantParams getQuantParams(int64_t const num_experts_on_rank, int64_t const hidden_size,
        int64_t const inter_size, torch::optional<c10::ArrayRef<torch::Tensor>> const& quant_scales,
        ActivationType base_activation_type) const
    {
        int expand_ratio = isGatedActivation(base_activation_type) ? 2 : 1;
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
                            * expand_ratio
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * expand_ratio, hidden_size // 4 // "
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
                            * expand_ratio
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * expand_ratio, hidden_size // 4 // "
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
        else if (isWMxfp8AMxfp8Quant())
        {
            // <e4m3, e4m3> with MXFP8 1x32 UE8M0 block scales on both sides;
            // SF storage is int32-packed UE8M0 (same convention as MXFP4 MoE).
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for MXFP8 x MXFP8 quantization");
            TORCH_CHECK(quant_scales.value().size() == 2,
                "Expecting 2 quant scales (fc1_weight_block, fc2_weight_block) for MXFP8 x MXFP8 quantization");

            auto const fc1_weight_block = quant_scales.value()[0];
            auto const fc2_weight_block = quant_scales.value()[1];

            CHECK_INPUT(fc1_weight_block, c10::ScalarType::Int);
            CHECK_INPUT(fc2_weight_block, c10::ScalarType::Int);
            TORCH_CHECK(fc1_weight_block.dim() == 3, "fc1 weight block must be 3D");
            TORCH_CHECK(fc2_weight_block.dim() == 3, "fc2 weight block must be 3D");

            return kernels::QuantParams::MXFP8MXFP8(
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
                static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()));
        }
        else if (isNvfp4Quant())
        {
            TORCH_CHECK(quant_scales.has_value(), "Expecting quant scales for nvfp4 quantization");
            TORCH_CHECK(quant_scales.value().size() == 6 || quant_scales.value().size() == 7,
                "Expecting 6 or 7 quant scales for nvfp4 quantization");

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
                            * expand_ratio
                    && fc1_weight_block.sizes()[2] * FP8_PER_INT32
                            * TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                        == TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                            hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4),
                "fc1 weight block size must be (num_experts_on_rank, inter_size * expand_ratio, hidden_size // 4 // "
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
                // Whether it is per-expert activation scale
                bool fc1_use_per_expert_act_scale = fc1_act_scales.numel() > hidden_size;
                bool fc2_use_per_expert_act_scale = fc2_act_scales.numel() > inter_size;
                return kernels::QuantParams::GroupWise(group_size,
                    static_cast<void const*>(fc1_weight_scales.data_ptr()),
                    static_cast<void const*>(fc2_weight_scales.data_ptr()),
                    static_cast<void const*>(fc1_act_scales.numel() > 0 ? fc1_act_scales.data_ptr() : nullptr),
                    static_cast<void const*>(fc2_act_scales.numel() > 0 ? fc2_act_scales.data_ptr() : nullptr),
                    static_cast<void const*>(fc1_weight_zeros.numel() > 0 ? fc1_weight_zeros.data_ptr() : nullptr),
                    static_cast<void const*>(fc2_weight_zeros.numel() > 0 ? fc2_weight_zeros.data_ptr() : nullptr),
                    static_cast<float const*>(fc1_alpha.numel() > 0 ? fc1_alpha.data_ptr() : nullptr),
                    static_cast<float const*>(fc2_alpha.numel() > 0 ? fc2_alpha.data_ptr() : nullptr),
                    fc1_use_per_expert_act_scale, fc2_use_per_expert_act_scale);
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
        // <e4m3, e4m3> per-tensor FP8; excludes DeepSeek block-scale FP8 and MXFP8xMXFP8.
        return !mUseDeepSeekFP8BlockScaling && !mUseMxfp8WeightScaling
            && mActivationDtype == c10::ScalarType::Float8_e4m3fn && mWeightDtype == c10::ScalarType::Float8_e4m3fn;
    }

    bool isWMxfp8AMxfp8Quant() const
    {
        // <e4m3, e4m3> with MXFP8 block-scaled weights + dynamic MXFP8 acts.
        return mUseMxfp8WeightScaling && mActivationDtype == c10::ScalarType::Float8_e4m3fn
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

TRTLLM_NAMESPACE_END

TORCH_LIBRARY(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::FusedMoeRunner>("FusedMoeRunner")
        .def(torch::init<c10::ScalarType, c10::ScalarType, c10::ScalarType, bool, bool, bool, bool, bool, bool>())
        .def("run_gemm_profile", &tensorrt_llm::torch_ext::FusedMoeRunner::runGemmProfile)
        .def("get_tactic_num", &tensorrt_llm::torch_ext::FusedMoeRunner::getTacticNum)
        .def("run_moe", &tensorrt_llm::torch_ext::FusedMoeRunner::runMoe)
        .def("run_moe_min_latency", &tensorrt_llm::torch_ext::FusedMoeRunner::runMoeMinLantency)
        .def("reserve_lora_host_buffers", &tensorrt_llm::torch_ext::FusedMoeRunner::reserveLoraHostBuffers)
        .def("clear_workspaces", &tensorrt_llm::torch_ext::FusedMoeRunner::clearWorkspaces);
}
