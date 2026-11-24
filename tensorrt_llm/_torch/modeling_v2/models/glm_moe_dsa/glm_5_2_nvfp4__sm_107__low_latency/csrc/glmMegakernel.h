// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::megakernel
{
int glm52PlainLayerSmemBytes();
int glm52PlainLayerNumParams();
int64_t glm52PlainLayerMagic();
int glm52PlainLayerArenaBytes();
int glm52PlainLayerClearUnits();
int glm52PlainLayerNumCounters();
int glm52PlainLayerCandCap();
int glm52PlainLayerMaxKeys();
int glm52PlainLayerRouterLg();
int glm52PlainLayerLgBytes();
int glm52PlainLayerIdxTmap(void* out, void* pool, unsigned long long numRows);
int glm52PlainLayerWeightTable(void const* params, void* out, void* stream);
int glm52PlainLayerInit();
int glm52PlainLayerLaunch(void const* params, void const* x, void const* residual, void* xOut, void* resOut,
    void const* topk, void const* blockTable, void const* kvLen, void* stream, int pdl);
} // namespace kernels::glm_moe_dsa::megakernel

TRTLLM_MV2_NAMESPACE_END
