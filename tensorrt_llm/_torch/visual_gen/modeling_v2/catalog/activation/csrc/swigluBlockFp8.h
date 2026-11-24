/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/visual_gen/common/namespace.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_VG_NAMESPACE_BEGIN

namespace swiglu_block_fp8
{

inline constexpr int kElemsPerLane = 16;
inline constexpr int kGroupWidth = 128;
inline constexpr int kGroupsPerWord = 4;
inline constexpr int kColsPerWarp = kGroupWidth * kGroupsPerWord;

void launchSwiGLUBlockFp8(
    void const* input, void* output, int32_t* scales, int rows, int width, int64_t paddedRows, cudaStream_t stream);

} // namespace swiglu_block_fp8

TRTLLM_VG_NAMESPACE_END
