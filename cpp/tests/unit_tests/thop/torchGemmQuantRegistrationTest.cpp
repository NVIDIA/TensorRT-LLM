/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "torchFamilyRegistration.h"
#include <torch/custom_class_detail.h>

namespace
{

constexpr TorchRegistration kRegistrations[] = {
    {"tensorrt_llm::dequantize_e4m3_activation", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::dequantize_e4m3_per_tensor", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::dequantize_e4m3_weight", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::dequantize_mxe4m3_host", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::e2m1_and_ufp8sf_scale_to_float", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::e2m1_and_ufp8sf_scale_to_float_v2", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::float_to_e2m1_and_ufp8sf_scale", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::half_to_e2m1_and_ufp8sf_scale", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::quantize_e4m3_activation", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::quantize_e4m3_per_tensor", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::quantize_e4m3_weight", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::quantize_mxe4m3_host", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"tensorrt_llm::static_quantize_e4m3_activation", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::static_quantize_e4m3_per_tensor", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::static_quantize_e4m3_weight", c10::DispatchKey::CUDA, true},
    {"tensorrt_llm::vectorized_per_token_fp8_quant", c10::DispatchKey::CUDA, true},
    {"trtllm::_add_bias_and_interleave_int4s", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::_add_bias_and_interleave_int8s", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::_permute_B_rows_for_mixed_gemm", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::_subbyte_transpose", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::_symmetric_quantize_last_axis_of_batched_matrix", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::block_scale_interleave", c10::DispatchKey::CUDA, true},
    {"trtllm::block_scale_interleave_reverse", c10::DispatchKey::CUDA, true},
    {"trtllm::calculate_nvfp4_global_scale", c10::DispatchKey::CUDA, true},
    {"trtllm::cublas_mm", c10::DispatchKey::CUDA, true}, {"trtllm::cublas_scaled_mm", c10::DispatchKey::CUDA, true},
    {"trtllm::cuda_core_nvfp4_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::cuda_scaled_mm", c10::DispatchKey::CUDA, true},
    {"trtllm::cutlass_scaled_mm", c10::DispatchKey::CUDA, true},
    {"trtllm::deepseek_v4_q_norm", c10::DispatchKey::CUDA, true},
    {"trtllm::deepseek_v4_q_norm_fused_fp8", c10::DispatchKey::CUDA, true},
    {"trtllm::dsv3_fused_a_gemm_op", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_batched_quantize", c10::DispatchKey::CUDA, true}, {"trtllm::fp4_bmm", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_fp8_gemm_trtllmgen", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_gemm", c10::DispatchKey::CUDA, true}, {"trtllm::fp4_gemm_trtllmgen", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_quantize_with_reorder_residual", c10::DispatchKey::CUDA, true},
    {"trtllm::fp4_quantize_with_residual", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_batched_quantize_1x128_permute102", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_block_scaling_bmm", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_block_scaling_bmm_out", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_block_scaling_gemm_impl", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_block_scaling_moe_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_per_tensor_scaling_tllmg_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_quantize_1x128", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_quantize_1x128_cutedsl_ue8m0", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_quantize_1x128_packed_ue8m0", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_add_rms_norm_quant", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_add_rmsnorm_fp4_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_cat_fp4", c10::DispatchKey::CUDA, true}, {"trtllm::fused_cat_fp8", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_gated_rmsnorm_quant", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_relu2_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_rmsnorm_fp4_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::gptq_marlin_repack", c10::DispatchKey::CUDA, true},
    {"trtllm::group_rms_norm_base", c10::DispatchKey::CUDA, false},
    {"trtllm::group_rms_norm_heuristic", c10::DispatchKey::CUDA, false},
    {"trtllm::group_rms_norm_large_batch", c10::DispatchKey::CUDA, false},
    {"trtllm::marlin_nvfp4_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::mxfp4_dequantize_unswizzled", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::mxfp8_mxfp8_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::mxfp8_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::pack_int8_tensor_to_packed_int4", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::preprocess_weights_for_mixed_gemm", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::silu_and_mul_fp8_quantize_1x128_packed_ue8m0", c10::DispatchKey::CUDA, true},
    {"trtllm::symmetric_quantize_last_axis_of_batched_matrix", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::tinygemm2", c10::DispatchKey::CUDA, true},
    {"trtllm::unpack_int4_packed_tensor_to_int8", c10::DispatchKey::CompositeImplicitAutograd, true}};

} // namespace

TEST(TorchGemmQuantRegistration, SchemasAndDispatchKernels)
{
    expectTorchRegistrations(kRegistrations);
}

TEST(TorchGemmQuantRegistration, ExcludesUnrelatedOperators)
{
    expectOnlyTorchFamily(kRegistrations);
}

TEST(TorchGemmQuantRegistration, CustomClasses)
{
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.CublasLtFP4GemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.FP4GemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.FP8BatchedGemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.FP8RowwiseGemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.MXFP8GemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.WeightOnlyQuantGemmRunner"), nullptr);
    EXPECT_NE(torch::getCustomClass("__torch__.torch.classes.trtllm.finegrainedMixedDtypeGemmRunner"), nullptr);
}
