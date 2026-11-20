/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! What the row kernel does after the optional residual update.
enum class ResidualNormMode : int
{
    kResidualOnly = 0, //!< xOut = x + gate * (y + yBias); no normalization
    kLayerNorm = 1,    //!< LayerNorm with affine weight / bias [hidden]
    kAdaLayerNorm = 2, //!< LayerNorm (no affine) * (1 + scale[b]) + shift[b], scale / shift [batch, hidden]
    kRmsNorm = 3,      //!< weight * bf16(x * rsqrt(mean(x^2) + eps)), weight [hidden] (bf16 values in fp32), no bias
};

struct ResidualNormQuantParams
{
    //! bf16 [rows, hidden]
    void const* x = nullptr;
    //! Optional bf16 [rows, hidden] update added to x; with y, xOut = x + gate * (y + yBias).
    void const* y = nullptr;
    //! Optional bf16 [hidden] bias of y, added in bf16 before the gate.
    void const* yBias = nullptr;
    //! Optional fp32 [batch, hidden] gate.
    float const* gate = nullptr;
    //! fp32 LayerNorm weight / bias [hidden] (kLayerNorm), or AdaLN scale / shift [batch, hidden].
    float const* weight = nullptr;
    float const* bias = nullptr;
    //! fp32 [1] multiplier applied before the saturating e4m3 store (1 / dequant scale).
    float const* invOutScale = nullptr;
    //! bf16 [rows, hidden] updated residual (needs y), and e4m3 [rows, hidden] normalized output.
    void* xOut = nullptr;
    void* normed = nullptr;
    int rows = 0;
    int hidden = 0;
    //! Rows that share one batch entry of gate / scale / shift.
    int rowsPerBatch = 0;
    float eps = 1e-6F;
    ResidualNormMode mode = ResidualNormMode::kResidualOnly;
};

//! Gated residual update and LayerNorm / AdaLN + static per-tensor FP8 quantization of each row,
//! in one pass (diffusion transformer blocks). The residual rounds y + yBias to bf16 and then
//! computes x + gate * y with separately rounded multiply and add; the normalized value is rounded
//! to bf16 before the FP8 multiply.
void launchResidualNormQuant(ResidualNormQuantParams const& params, cudaStream_t stream);

//! Whether launchResidualNormQuant has a kernel for this hidden size: a multiple of 1024 up to 8192.
bool residualNormQuantSupportsHidden(int hidden);

//! Arguments of launchAdaModulate.
struct AdaModulateParams
{
    //! bf16 [rows, hidden] input and output.
    void const* x = nullptr;
    void* out = nullptr;
    //! fp32 [batch, hidden] AdaLN scale and shift.
    float const* scale = nullptr;
    float const* shift = nullptr;
    //! Rows of x and the row width.
    int64_t rows = 0;
    int hidden = 0;
    //! Rows that share one batch entry of scale / shift.
    int64_t rowsPerBatch = 0;
};

//! out = bf16(x * (1 + scale[b]) + shift[b]) per element, rounded like the eager bf16 * fp32
//! multiply, fp32 add and bf16 cast: 1 + scale, the product and the sum each round to fp32 (no
//! FMA), then the result rounds to bf16. hidden must be a multiple of 8 and every pointer
//! 16-byte aligned.
void launchAdaModulate(AdaModulateParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
