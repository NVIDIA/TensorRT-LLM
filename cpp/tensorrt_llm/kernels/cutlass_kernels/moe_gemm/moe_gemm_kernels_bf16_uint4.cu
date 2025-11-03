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

#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h"

namespace tensorrt_llm::kernels::cutlass_kernels
{
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>;
#endif
} // namespace tensorrt_llm::kernels::cutlass_kernels
