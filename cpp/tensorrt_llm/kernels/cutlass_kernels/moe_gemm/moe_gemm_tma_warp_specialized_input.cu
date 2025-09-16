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

#include "../include/moe_gemm_kernels.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::kernels::cutlass_kernels
{
std::array<size_t, 20> TmaWarpSpecializedGroupedGemmInput::workspaceBuffers(
    int num_experts, FpXBlockScalingType scaling_type)
{
    size_t problem_shape_size = sizeof(ProblemShape::UnderlyingProblemShape) * num_experts;
    size_t stride_act_size = std::max(sizeof(StrideA), sizeof(StrideB)) * num_experts;
    size_t stride_weight_size = std::max(sizeof(StrideA), sizeof(StrideB)) * num_experts;
    size_t stride_c_size = std::max(sizeof(StrideC), sizeof(StrideC_T)) * num_experts;
    size_t stride_d_size = std::max(sizeof(StrideD), sizeof(StrideD_T)) * num_experts;

    size_t ptr_buf_size = sizeof(void*) * num_experts;
    size_t scale_buf_size = sizeof(float*) * num_experts;

    size_t sf_act_size = sizeof(ElementSF*) * num_experts;
    size_t sf_weight_size = sizeof(ElementSF*) * num_experts;
    size_t stride_sf_act_size = scaling_type == FpXBlockScalingType::MXFPX
        ? sizeof(MXFPXBlockScaledConfig::LayoutSF) * num_experts
        : sizeof(NVFP4BlockScaledConfig::LayoutSF) * num_experts;
    size_t stride_sf_weight_size = scaling_type == FpXBlockScalingType::MXFPX
        ? sizeof(MXFPXBlockScaledConfig::LayoutSF) * num_experts
        : sizeof(NVFP4BlockScaledConfig::LayoutSF) * num_experts;

    size_t int4_groupwise_problem_shape_size
        = sizeof(INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape) * num_experts;
    size_t int4_groupwise_sf_a_size = sizeof(INT4GroupwiseParams::SFA*) * num_experts;
    size_t int4_groupwise_stride_sf_a_size = sizeof(INT4GroupwiseParams::StrideSFA) * num_experts;

    size_t ptr_token_map_size = sizeof(int**) * num_experts;

    return std::array{problem_shape_size, stride_act_size, stride_weight_size, stride_c_size, stride_d_size,
        ptr_buf_size, ptr_buf_size, ptr_buf_size, ptr_buf_size, scale_buf_size, sf_act_size, sf_weight_size,
        stride_sf_act_size, stride_sf_weight_size, int4_groupwise_problem_shape_size, int4_groupwise_sf_a_size,
        int4_groupwise_stride_sf_a_size, ptr_buf_size, scale_buf_size, ptr_token_map_size};
}

size_t TmaWarpSpecializedGroupedGemmInput::workspaceSize(int num_experts, FpXBlockScalingType scaling_type)
{
    auto buffers = workspaceBuffers(num_experts, scaling_type);
    return tensorrt_llm::common::calculateTotalWorkspaceSize(buffers.data(), buffers.size());
}

void TmaWarpSpecializedGroupedGemmInput::configureWorkspace(int8_t* start_ptr, int num_experts, void* gemm_workspace,
    size_t gemm_workspace_size, FpXBlockScalingType scaling_type)
{
    auto buffers = workspaceBuffers(num_experts, scaling_type);
    std::array<int8_t*, 20> pointers{};
    TLLM_CHECK_WITH_INFO(pointers.size() == buffers.size(), "Mismatching workspace size and number of buffers");
    for (int i = 0; i < buffers.size(); i++)
    {
        pointers[i] = start_ptr;
        start_ptr = tensorrt_llm::common::nextWorkspacePtr(start_ptr, buffers[i]);
    }

    shape_info.num_groups = num_experts;
    shape_info.problem_shapes = reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(pointers[0]);
    shape_info.host_problem_shapes = nullptr;
    stride_act = reinterpret_cast<void*>(pointers[1]);
    stride_weight = reinterpret_cast<void*>(pointers[2]);
    stride_c = reinterpret_cast<void*>(pointers[3]);
    stride_d = reinterpret_cast<void*>(pointers[4]);

    ptr_act = reinterpret_cast<void const**>(pointers[5]);
    ptr_weight = reinterpret_cast<void const**>(pointers[6]);
    ptr_c = reinterpret_cast<void const**>(pointers[7]);
    ptr_d = reinterpret_cast<void**>(pointers[8]);

    alpha_scale_ptr_array = reinterpret_cast<float const**>(pointers[9]);

    fpX_block_scaling_factors_act = reinterpret_cast<ElementSF const**>(pointers[10]);
    fpX_block_scaling_factors_weight = reinterpret_cast<ElementSF const**>(pointers[11]);

    fpX_block_scaling_factors_stride_act = pointers[12];
    fpX_block_scaling_factors_stride_weight = pointers[13];

    int4_groupwise_params.shape.problem_shapes
        = reinterpret_cast<INT4GroupwiseParams::ProblemShapeInt::UnderlyingProblemShape*>(pointers[14]);
    int4_groupwise_params.shape.host_problem_shapes = nullptr;
    int4_groupwise_params.ptr_s_a = reinterpret_cast<INT4GroupwiseParams::SFA const**>(pointers[15]);
    int4_groupwise_params.stride_s_a = reinterpret_cast<INT4GroupwiseParams::StrideSFA*>(pointers[16]);

    fused_finalize_epilogue.ptr_bias = reinterpret_cast<void const**>(pointers[17]);
    fused_finalize_epilogue.ptr_router_scales = reinterpret_cast<float const**>(pointers[18]);
    fused_finalize_epilogue.ptr_source_token_index = reinterpret_cast<int const**>(pointers[19]);

    this->gemm_workspace = reinterpret_cast<uint8_t*>(gemm_workspace);
    this->gemm_workspace_size = gemm_workspace_size;
}

void TmaWarpSpecializedGroupedGemmInput::setFinalizeFusionParams(
    void* final_output, int hidden_size, int num_output_tokens, bool use_reduction)
{
    fused_finalize_epilogue.ptr_final_output = final_output;

    fused_finalize_epilogue.stride_final_output = cutlass::make_cute_packed_stride(
        FusedFinalizeEpilogue::StrideFinalOutput{}, cute::make_shape(num_output_tokens, hidden_size, 1));
    fused_finalize_epilogue.stride_final_output_transposed = cutlass::make_cute_packed_stride(
        FusedFinalizeEpilogue::StrideFinalOutput_T{}, cute::make_shape(hidden_size, num_output_tokens, 1));

    fused_finalize_epilogue.num_rows_in_final_output = num_output_tokens;
    fused_finalize_epilogue.shape_override = hidden_size;
    fused_finalize_epilogue.use_reduction = use_reduction;
}

std::string TmaWarpSpecializedGroupedGemmInput::toString() const
{
    std::stringstream ss;
    ss << "Hopper Input Information: " << (isValid() ? "valid" : "null") << "\n";
    if (isValid())
    {
        using PrintType = void const*;
        ss << "Ptr Act: " << (PrintType) ptr_act << " with Stride: " << (PrintType) stride_act << ",\n"
           << "Ptr Weight: " << (PrintType) ptr_weight << " with Stride: " << (PrintType) stride_weight << ",\n"
           << "Ptr C: " << (PrintType) ptr_c << " with Stride: " << (PrintType) stride_c << "\n";
        ss << "Epilogue Fusion: " << (int) fusion << ",\n";
        if (fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE)
        {
            ss << "Final Output: " << (PrintType) fused_finalize_epilogue.ptr_final_output;
            ss << " with Stride: " << fused_finalize_epilogue.stride_final_output;
            ss << ",\nBias: " << (PrintType) fused_finalize_epilogue.ptr_bias;
            ss << ",\nRouter Scales: " << fused_finalize_epilogue.ptr_router_scales;
            ss << ", Source Map: " << (PrintType) fused_finalize_epilogue.ptr_source_token_index;
        }
        else
        {
            ss << "Ptr D: " << (PrintType) ptr_d;
            ss << " with Stride: " << (PrintType) stride_d;
        }
        ss << '\n';
        ss << "Alpha scale ptr: " << (PrintType) alpha_scale_ptr_array << "\n";

        ss << "FpX Block Scaling Type: " << (int) fpX_block_scaling_type << "\n";
        ss << "Fp4 Block Scaling Factors Act: " << (PrintType) fpX_block_scaling_factors_act
           << ", with Stride: " << (PrintType) fpX_block_scaling_factors_stride_act << "\n";
        ss << "Fp4 Block Scaling Factors Weight: " << (PrintType) fpX_block_scaling_factors_weight
           << ", with Stride: " << (PrintType) fpX_block_scaling_factors_stride_weight << "\n";
        ss << "Gemm Workspace: " << (PrintType) gemm_workspace << ", with Size: " << gemm_workspace_size << "\n";
    }

    return ss.str();
}
} // namespace tensorrt_llm::kernels::cutlass_kernels
