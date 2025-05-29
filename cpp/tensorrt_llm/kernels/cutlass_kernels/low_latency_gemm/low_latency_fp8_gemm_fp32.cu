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

#include "fp8_low_latency_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

template class CutlassLowLatencyFp8GemmRunner<float>; // for compilation only

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
