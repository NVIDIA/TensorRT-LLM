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

#ifndef TRTLLM_FUSEDDITNORMKERNEL_H
#define TRTLLM_FUSEDDITNORMKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Single fused DiT pre-block kernel covering KA/KB/KC/KD via template flags.
//
// Pipeline (compile-time selected, all "pluggable" except RmsNorm which is always on):
//   1. Phase 0a -- cp.async x -> SMEM                                     (always)
//   2. Phase 0b -- combine modulator (table, ts) pairs into bf16 caches:
//        gate (if HAS_GATE)
//        NUM_OUT scale + shift (if HAS_MODULATE)
//   3. Phase 0c -- load attn into reg cache                                (if HAS_RESIDUAL)
//   4. Phase 1  -- x_new = x [+ attn [* gate]]; sum^2; write x_new in-place if HAS_RESIDUAL
//   5. Reduce   -- per-row warp + cross-warp reduce -> rms_rcp             (always)
//   6. Phase 2  -- write NUM_OUT outputs:
//        if HAS_MODULATE: y[k] = (1 + scale[k]) * normed + shift[k]
//        else           : y    = normed
//        HAS_QUANT=false: bf16 store to out_bf16[k]
//        HAS_QUANT=true : NVFP4 + 128x4 swizzled SF to (out_fp4[k], out_sf[k])
//
// Specializations used by LTX-2:
//   KA: HAS_RESIDUAL=false, HAS_GATE=false, HAS_MODULATE=true,  NUM_OUT=1
//   KB: HAS_RESIDUAL=true,  HAS_GATE=false, HAS_MODULATE=true,  NUM_OUT=2
//   KC: HAS_RESIDUAL=true,  HAS_GATE=true,  HAS_MODULATE=true,  NUM_OUT=1
//   KD: HAS_RESIDUAL=true,  HAS_GATE=true,  HAS_MODULATE=false, NUM_OUT=1
//
// HAS_GATE implies HAS_RESIDUAL (asserted at compile time).
//
// Modulator combine matches PyTorch eager `_get_all_ada_values` semantics:
// narrow fp32 table to bf16 first, then bf16 hw add (`__hadd2`).
//
// Tile: production hardcoded to (ROWS_PER_BLOCK=1, BLOCK_SIZE=256). NCU sweep
// over {2r256, 1r256, 1r512} on B200 at the V1 production shape showed 1r256
// gives the highest per-element bench rate. 2r256 has too much per-thread
// register pressure (KB-bf16 at 12.5% occupancy); 1r512 hits higher occupancy
// but its 16-warp CTAs starve the SM warp schedulers + pay a heavier
// __syncthreads() barrier. 1r256 is the sweet spot.
//
// Supported hidden_dim: 2048 (LTX-2 audio) and 4096 (LTX-2 video).

struct AdaLNNormParams
{
    // === Input ===
    __nv_bfloat16* x = nullptr;          // [num_tokens, D] bf16. Read always; written iff HAS_RESIDUAL.
    __nv_bfloat16 const* attn = nullptr; // HAS_RESIDUAL: [num_tokens, D] bf16

    // === Gate modulator (HAS_GATE => HAS_RESIDUAL) ===
    float const* gate_table = nullptr;      // [D] fp32, broadcast over batch
    __nv_bfloat16 const* gate_ts = nullptr; // [batch, D] bf16
    int gate_ts_stride = 0;                 // inner stride between batches

    // === Affine modulators (HAS_MODULATE) -- up to NUM_OUT={1,2} entries ===
    float const* scale_table[2] = {nullptr, nullptr};
    __nv_bfloat16 const* scale_ts[2] = {nullptr, nullptr};
    int scale_ts_stride[2] = {0, 0};
    float const* shift_table[2] = {nullptr, nullptr};
    __nv_bfloat16 const* shift_ts[2] = {nullptr, nullptr};
    int shift_ts_stride[2] = {0, 0};

    // === Outputs (NUM_OUT entries) ===
    __nv_bfloat16* out_bf16[2] = {nullptr, nullptr}; // HAS_QUANT=false
    uint32_t* out_fp4[2] = {nullptr, nullptr};       // HAS_QUANT=true
    uint32_t* out_sf[2] = {nullptr, nullptr};        // HAS_QUANT=true: SWIZZLED 128x4 SF
    float const* sf_scale[2] = {nullptr, nullptr};   // HAS_QUANT=true: scalar broadcast

    // === Shape ===
    int num_tokens = 0;
    int tokens_per_batch = 0;
    float eps = 1e-6f;
};

// Launch the unified kernel. Selects production tile (1r256) and dispatches on hidden_dim.
//
// Supported hidden_dim: 2048, 4096. Caller is responsible for populating only the params
// relevant to the chosen template configuration; unused fields can stay default-initialized.
template <bool HAS_RESIDUAL, bool HAS_GATE, bool HAS_MODULATE, int NUM_OUT, bool HAS_QUANT>
void launchFusedDiTNorm(AdaLNNormParams const& params, int hidden_dim, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_FUSEDDITNORMKERNEL_H
