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

// The DFlash 2 sampled path walk (ops/spec/draft_path_sample.py), one CTA per request.
//
// The built-in sampled walk (dflash2_sample_candidate_paths with DFlashWorker's
// sample_step) runs, per draft position t: a gather of the row edge_scores[b, t, pred, :]
// (unary + bilinear), nan_to_num, the rows' Philox offsets, and
// fusedSamplingKernel<float, BLOCK, true, true> over the [B, top_k] rows (one CTA of BLOCK
// threads per row: 1024 up to kWideBlockMaxRows rows, else 512), then the clamp of the
// drawn index, which is the next position's predecessor. Afterwards it gathers the drawn
// candidates' vocabulary ids and scatters each position's distribution into the
// slot-indexed draft_probs rows.
//
// Here one CTA of the same BLOCK threads per request runs every position: it writes the
// position's row (the same single FP32 add, then nan_to_num) and the rows' offsets to
// scratch laid out as the built-in's tensors, calls the shipped kernel body on them with
// the same parameters, and keeps the drawn index for the next position. Row b of the body
// is blockIdx.x == b in both, so the body reads, reduces, draws and writes exactly what
// the shipped kernel does for that row.

#pragma once

#include "fusedSamplingKernelsCommon.cuh"

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

struct DraftPathSampleParams
{
    //! [B, K, top_k, top_k] FP32 bilinear scores (no unary term), contiguous.
    float const* edges{nullptr};
    //! [B, K, top_k] FP32 unary logits, contiguous.
    float const* unary{nullptr};
    //! [B, K, top_k] int64 vocabulary ids, contiguous.
    int64_t const* ids{nullptr};
    //! [>= B] per-request sampling parameters, as the built-in passes them.
    float const* temperatures{nullptr};
    int32_t const* topKs{nullptr};
    float const* topPs{nullptr};
    float const* minPs{nullptr};
    //! [B] per-request Philox seed and offset-window base.
    int64_t const* seed{nullptr};
    int64_t const* offset{nullptr};
    //! The window slot of draft position 0 and the slot span: position t draws at
    //! offset + (slotBase + t) * slotSpan.
    int64_t slotBase{0};
    int64_t slotSpan{0};
    //! Scratch, position-major as the built-in's per-position tensors: [K, B, top_k] rows
    //! and probabilities, [K, B] offsets and drawn indices.
    float* rows{nullptr};
    float* probs{nullptr};
    int64_t* offsets{nullptr};
    int32_t* picks{nullptr};
    //! [B, K] int32 draft tokens.
    int32_t* tokens{nullptr};
    //! Optional slot-indexed draft distributions [slots, >= K, >= vocab] and [B] slot ids.
    float* draftProbs{nullptr};
    int64_t const* slots{nullptr};
    int64_t draftProbsSlotStride{0};
    int64_t draftProbsStepStride{0};
    int32_t numRows{0};
    int32_t numSteps{0};
    int32_t topK{0};
};

template <int BLOCK>
__global__ void draftPathSampleKernel(DraftPathSampleParams p)
{
    __shared__ FusedSamplingShared<BLOCK> shared;
    __shared__ int sPredecessor;
    int const b = blockIdx.x;
    int const tid = threadIdx.x;
    int const topK = p.topK;
    if (tid == 0)
    {
        // Position 0 replicates the anchor across the predecessor axis; the built-in reads row 0.
        sPredecessor = 0;
    }
    __syncthreads();
    for (int t = 0; t < p.numSteps; ++t)
    {
        int const predecessor = sPredecessor;
        size_t const cell = static_cast<size_t>(b) * p.numSteps + t;
        size_t const stepRows = static_cast<size_t>(t) * p.numRows;
        float* const rows = p.rows + stepRows * topK;
        if (tid < topK)
        {
            float score = p.unary[cell * topK + tid] + p.edges[(cell * topK + predecessor) * topK + tid];
            // nan_to_num(nan=0, posinf=0, neginf=0).
            if (isnan(score) || isinf(score))
            {
                score = 0.0f;
            }
            rows[static_cast<size_t>(b) * topK + tid] = score;
        }
        if (tid == 0)
        {
            p.offsets[stepRows + b] = p.offset[b] + (p.slotBase + t) * p.slotSpan;
        }
        __syncthreads();

        FusedSamplingParams fs;
        fs.logits = rows;
        fs.temperatures = p.temperatures;
        fs.topKs = p.topKs;
        fs.topPs = p.topPs;
        fs.minPs = p.minPs;
        fs.seed = reinterpret_cast<uint64_t const*>(p.seed);
        fs.offset = reinterpret_cast<uint64_t const*>(p.offsets + stepRows);
        // The built-in passes one seed and offset per row.
        fs.perRowRng = true;
        fs.outputTokens = p.picks + stepRows;
        fs.outputProbs = p.probs + stepRows * topK;
        fs.numRows = p.numRows;
        fs.vocabSize = topK;
        fusedSamplingBody<float, BLOCK, true, true>(fs, shared);
        __syncthreads();

        int index = p.picks[stepRows + b];
        // A degenerate row may yield an out-of-range draw; the built-in clamps before its gather.
        index = index < 0 ? 0 : (index > topK - 1 ? topK - 1 : index);
        if (tid == 0)
        {
            sPredecessor = index;
            p.tokens[cell] = static_cast<int32_t>(p.ids[cell * topK + index]);
        }
        if (p.draftProbs != nullptr && tid < topK)
        {
            float* const dst = p.draftProbs + p.slots[b] * p.draftProbsSlotStride + t * p.draftProbsStepStride;
            dst[p.ids[cell * topK + tid]] = p.probs[(stepRows + b) * topK + tid];
        }
        __syncthreads();
    }
}

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm
