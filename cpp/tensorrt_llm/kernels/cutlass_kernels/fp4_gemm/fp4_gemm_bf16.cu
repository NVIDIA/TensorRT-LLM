/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "fp4_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 256, 256, 4, 4, 1, _2SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 128, 128, 128, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 128, 128, 256, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(__nv_bfloat16, 256, 128, 128, 1, 1, 1)

template class CutlassFp4GemmRunner<__nv_bfloat16>;

#endif

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
