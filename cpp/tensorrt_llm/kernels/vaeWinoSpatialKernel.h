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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Spatial Winograd F(m x m, 3x3) transforms, m = 2 or 4, for channels-last bf16 or fp16 frames; the
// transformed data (V, M) is fp16, which keeps three more mantissa bits than bf16 through
// the GEMMs.
//
// A frame [H, W, C] is covered by ceil(H/m) x ceil(W/m) output tiles of m x m pixels; tile
// (i, j) is computed from the (m+2) x (m+2) input window at rows m*i-1..m*i+m, cols
// m*j-1..m*j+m (zero outside the frame, the conv's padding of 1; output pixels past the
// frame are dropped). The input transform writes, for a range of tile rows [row0, row1) of
// every frame, V[coord][frame, tile][C] = B^T d B (coord = x * (m+2) + y); the GEMMs produce
// M in the same layout; the output transform writes y[frame, m*i+a, m*j+b, C] = A^T M A.
//
// The N frames form `components` groups of N / components frames (temporal Winograd
// components); V and M are [components * (m+2)^2, (N / components) * ntilesStrip, C] with
// ntilesStrip = (row1 - row0) * ceil(W/m) and tile index = pair * ntilesStrip + (i - row0) * ceil(W/m) + j.
//
// With `stackFrames` (components == 1) the input transform instead writes V as
// [(m+2)^2, ntilesStrip, N * C], frame n at channel offset n * C: the N frames sit side by
// side along the GEMM's K axis, so one GEMM sums their contributions (the three temporal taps
// of a causal 3x3x3 conv on one output frame).
struct VaeWinoSpatialParams
{
    int frames;      // N
    int height;      // H
    int width;       // W
    int channels;
    int components;  // temporal components; frame n belongs to component n / (N / components)
    int row0;        // first tile row of the strip
    int row1;        // one past the last tile row
    int tile;        // output tile side m: 2 (F(2x2, 3x3)) or 4 (F(4x4, 3x3))
    int stackFrames; // input transform only: V layout [(m+2)^2, ntilesStrip, N * C]
    int halfFrames;  // frames (and the output transform's bias) are fp16 instead of bf16
    // Input transform with stacked frames only: the N frames may live in up to three separate
    // buffers (leading zero frames, the causal history, the new frame), frame n being frame
    // n - start of its segment; a null segment pointer stands for zero frames. 0 segments: all
    // frames in the `frames` argument.
    int segments;
    void const* segPtr[3];
    int segFrames[3];
};

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream);
// `bias` ([C] in the frames' dtype, may be null) is added to every output pixel before rounding.
void launchVaeWinoSpatialOutput(
    void const* m, void* frames, void const* bias, VaeWinoSpatialParams const& p, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
