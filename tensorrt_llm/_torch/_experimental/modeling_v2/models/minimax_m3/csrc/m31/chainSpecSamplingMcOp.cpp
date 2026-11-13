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

#include "minimax_m3/csrc/m31/chainSpecSamplingMcFiSm107.h"
#include "minimax_m3/csrc/m31/chainSpecSamplingMcSm103.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! ST-01-REJ's multi-CTA chain speculative sampling (see csrc/m31/chainSpecSamplingMcSm103.h): the inputs
//! and outputs of trtllm::chain_speculative_sampling (output_token_ids [batch, numDraft + 1], num_tokens [batch]),
//! bitwise its results, on batch x ctas_per_row CTAs. workspace: a zero-initialized byte buffer of at least
//! batch x minimax_m31_chain_speculative_sampling_mc_ws_bytes(), reused across calls; ctas_per_row divides 1024 and
//! batch x ctas_per_row must not exceed the SM count when ctas_per_row > 1.
static std::tuple<torch::Tensor, torch::Tensor> chainSpecMcImpl(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow,
    TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow const& rng)
{
    TORCH_CHECK(draftProbs.dim() == 3,
        "minimax_m31_chain_speculative_sampling_mc: draft_probs must be [batch, "
        "numDraft, vocab]");
    TORCH_CHECK(draftTokenIds.dim() == 2,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be [batch, "
        "numDraft]");
    TORCH_CHECK(targetProbs.dim() == 3,
        "minimax_m31_chain_speculative_sampling_mc: target_probs must be [batch, numDraft + 1, vocab]");
    auto const batchSize = draftProbs.size(0);
    auto const numDraft = draftProbs.size(1);
    auto const vocabSize = draftProbs.size(2);
    TORCH_CHECK(draftTokenIds.size(0) == batchSize && draftTokenIds.size(1) == numDraft,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be [batch, numDraft]");
    TORCH_CHECK(
        targetProbs.size(0) == batchSize && targetProbs.size(1) == numDraft + 1 && targetProbs.size(2) == vocabSize,
        "minimax_m31_chain_speculative_sampling_mc: target_probs must be [batch, numDraft + 1, vocab]");
    for (auto const* t : {&draftProbs, &draftTokenIds, &targetProbs, &seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == draftProbs.get_device(),
            "minimax_m31_chain_speculative_sampling_mc: inputs must be contiguous CUDA tensors on one device");
    }
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32 && targetProbs.scalar_type() == torch::kFloat32,
        "minimax_m31_chain_speculative_sampling_mc: draft_probs and target_probs must be float32");
    TORCH_CHECK(draftTokenIds.scalar_type() == torch::kInt32,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be int32");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64 && offset.scalar_type() == torch::kInt64,
        "minimax_m31_chain_speculative_sampling_mc: seed and offset must be int64 tensors");
    TORCH_CHECK(seed.numel() == batchSize && offset.numel() == batchSize,
        "minimax_m31_chain_speculative_sampling_mc: seed and offset must hold one entry per request");
    TORCH_CHECK(vocabSize % 4 == 0 && vocabSize <= TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab
            && reinterpret_cast<uintptr_t>(draftProbs.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(targetProbs.data_ptr()) % 16 == 0,
        "minimax_m31_chain_speculative_sampling_mc: the probs must be 16-byte aligned with a vocab divisible by 4 and "
        "at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab);
    TORCH_CHECK(batchSize <= 65535, "minimax_m31_chain_speculative_sampling_mc: batch must be <= 65535");
    TORCH_CHECK(ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0,
        "minimax_m31_chain_speculative_sampling_mc: ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcWorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == draftProbs.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * batchSize,
        "minimax_m31_chain_speculative_sampling_mc: workspace must be a contiguous, 16-byte aligned CUDA buffer of at "
        "least ",
        wsBytesPerRow * batchSize, " bytes on draft_probs' device");

    c10::cuda::CUDAGuard const deviceGuard{draftProbs.device()};
    TORCH_CHECK(
        ctasPerRow == 1 || batchSize * ctasPerRow <= at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
        "minimax_m31_chain_speculative_sampling_mc: batch x ctas_per_row must not exceed the SM count");
    auto outputTokenIds = torch::empty({batchSize, numDraft + 1}, draftTokenIds.options());
    auto outputNumTokens = torch::empty({batchSize}, draftTokenIds.options());
    if (batchSize == 0)
    {
        return {outputTokenIds, outputNumTokens};
    }

    tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams params;
    params.draftProbs = draftProbs.const_data_ptr<float>();
    params.draftTokenIds = draftTokenIds.const_data_ptr<int32_t>();
    params.targetProbs = targetProbs.const_data_ptr<float>();
    params.seed = seed.const_data_ptr<int64_t>();
    params.offset = offset.const_data_ptr<int64_t>();
    params.outputTokenIds = outputTokenIds.data_ptr<int32_t>();
    params.outputNumTokens = outputNumTokens.data_ptr<int32_t>();
    params.batchSize = static_cast<int32_t>(batchSize);
    params.numDraftTokens = static_cast<int32_t>(numDraft);
    params.vocabSize = static_cast<int32_t>(vocabSize);
    auto const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeChainSpecSamplingMc(params, static_cast<int>(ctasPerRow),
        workspace.data_ptr(), at::cuda::getCurrentCUDAStream(draftProbs.get_device()),
        TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcDraftRows{}, rng);
    TORCH_CHECK(
        err == cudaSuccess, "minimax_m31_chain_speculative_sampling_mc launch failed: ", cudaGetErrorString(err));
    return {outputTokenIds, outputNumTokens};
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMc(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow)
{
    return chainSpecMcImpl(draftProbs, draftTokenIds, targetProbs, seed, offset, workspace, ctasPerRow,
        TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow{});
}

//! minimax_m31_chain_speculative_sampling_mc with slot-indexed draft rows (VR REJ_SLOTS): the draft distributions are
//! draft_buf[slot_ids[r], :num_draft, :vocab] of the spec metadata's [slots, maxDraft, >= vocab] float32 buffer, read
//! in place instead of from a gathered contiguous copy (the same floats, so bitwise the gathered call's results).
//! slot_ids: int64 [batch] on the device; each must index a row of draft_buf (not checked on the host: the caller's
//! gather would index the same rows).
static std::tuple<torch::Tensor, torch::Tensor> chainSpecMcSlotsImpl(torch::Tensor const& draftBuf,
    torch::Tensor const& slotIds, int64_t numDraft, torch::Tensor const& draftTokenIds,
    torch::Tensor const& targetProbs, torch::Tensor const& seed, torch::Tensor const& offset, torch::Tensor& workspace,
    int64_t ctasPerRow, TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow const& rng)
{
    TORCH_CHECK(draftBuf.dim() == 3 && draftBuf.is_cuda() && draftBuf.is_contiguous()
            && draftBuf.scalar_type() == torch::kFloat32,
        "minimax_m31_chain_speculative_sampling_mc_slots: draft_buf must be a contiguous float32 CUDA [slots, "
        "maxDraft, "
        "vocab_stride] tensor");
    TORCH_CHECK(slotIds.dim() == 1 && slotIds.is_cuda() && slotIds.is_contiguous()
            && slotIds.scalar_type() == torch::kInt64 && slotIds.get_device() == draftBuf.get_device(),
        "minimax_m31_chain_speculative_sampling_mc_slots: slot_ids must be a contiguous int64 CUDA [batch] tensor");
    TORCH_CHECK(targetProbs.dim() == 3, "minimax_m31_chain_speculative_sampling_mc_slots: target_probs must be 3-D");
    auto const batchSize = slotIds.size(0);
    auto const vocabSize = targetProbs.size(2);
    TORCH_CHECK(
        numDraft >= 0 && numDraft <= draftBuf.size(1) && vocabSize <= draftBuf.size(2) && draftBuf.size(2) % 4 == 0,
        "minimax_m31_chain_speculative_sampling_mc_slots: num_draft / vocab exceed draft_buf (or its row is not a "
        "multiple of 4 floats)");
    TORCH_CHECK(draftTokenIds.dim() == 2 && draftTokenIds.size(0) == batchSize && draftTokenIds.size(1) == numDraft,
        "minimax_m31_chain_speculative_sampling_mc_slots: draft_token_ids must be [batch, num_draft]");
    TORCH_CHECK(targetProbs.size(0) == batchSize && targetProbs.size(1) == numDraft + 1,
        "minimax_m31_chain_speculative_sampling_mc_slots: target_probs must be [batch, num_draft + 1, vocab]");
    for (auto const* t : {&draftTokenIds, &targetProbs, &seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == draftBuf.get_device(),
            "minimax_m31_chain_speculative_sampling_mc_slots: inputs must be contiguous CUDA tensors on one device");
    }
    TORCH_CHECK(targetProbs.scalar_type() == torch::kFloat32 && draftTokenIds.scalar_type() == torch::kInt32
            && seed.scalar_type() == torch::kInt64 && offset.scalar_type() == torch::kInt64 && seed.numel() == batchSize
            && offset.numel() == batchSize,
        "minimax_m31_chain_speculative_sampling_mc_slots: dtypes / counts as "
        "minimax_m31_chain_speculative_sampling_mc");
    TORCH_CHECK(vocabSize % 4 == 0 && vocabSize <= TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab
            && reinterpret_cast<uintptr_t>(draftBuf.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(targetProbs.data_ptr()) % 16 == 0,
        "minimax_m31_chain_speculative_sampling_mc_slots: 16-byte aligned probs, vocab divisible by 4, at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab);
    TORCH_CHECK(batchSize <= 65535 && ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0,
        "minimax_m31_chain_speculative_sampling_mc_slots: batch <= 65535; ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcWorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == draftBuf.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * batchSize,
        "minimax_m31_chain_speculative_sampling_mc_slots: workspace too small or misaligned");

    c10::cuda::CUDAGuard const deviceGuard{draftBuf.device()};
    TORCH_CHECK(
        ctasPerRow == 1 || batchSize * ctasPerRow <= at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
        "minimax_m31_chain_speculative_sampling_mc_slots: batch x ctas_per_row must not exceed the SM count");
    auto outputTokenIds = torch::empty({batchSize, numDraft + 1}, draftTokenIds.options());
    auto outputNumTokens = torch::empty({batchSize}, draftTokenIds.options());
    if (batchSize == 0)
    {
        return {outputTokenIds, outputNumTokens};
    }
    tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams params;
    params.draftProbs = draftBuf.const_data_ptr<float>();
    params.draftTokenIds = draftTokenIds.const_data_ptr<int32_t>();
    params.targetProbs = targetProbs.const_data_ptr<float>();
    params.seed = seed.const_data_ptr<int64_t>();
    params.offset = offset.const_data_ptr<int64_t>();
    params.outputTokenIds = outputTokenIds.data_ptr<int32_t>();
    params.outputNumTokens = outputNumTokens.data_ptr<int32_t>();
    params.batchSize = static_cast<int32_t>(batchSize);
    params.numDraftTokens = static_cast<int32_t>(numDraft);
    params.vocabSize = static_cast<int32_t>(vocabSize);
    TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcDraftRows rows;
    rows.rowIds = slotIds.const_data_ptr<int64_t>();
    rows.rowStride = draftBuf.stride(0);
    rows.posStride = draftBuf.stride(1);
    auto const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeChainSpecSamplingMc(params, static_cast<int>(ctasPerRow),
        workspace.data_ptr(), at::cuda::getCurrentCUDAStream(draftBuf.get_device()), rows, rng);
    TORCH_CHECK(
        err == cudaSuccess, "minimax_m31_chain_speculative_sampling_mc_slots launch failed: ", cudaGetErrorString(err));
    return {outputTokenIds, outputNumTokens};
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMcSlots(torch::Tensor const& draftBuf,
    torch::Tensor const& slotIds, int64_t numDraft, torch::Tensor const& draftTokenIds,
    torch::Tensor const& targetProbs, torch::Tensor const& seed, torch::Tensor const& offset, torch::Tensor& workspace,
    int64_t ctasPerRow)
{
    return chainSpecMcSlotsImpl(draftBuf, slotIds, numDraft, draftTokenIds, targetProbs, seed, offset, workspace,
        ctasPerRow, TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow{});
}

//! FOLD_F3: the Philox window operands. Request r's offset is base[base_row + r] + add (mod 2^64; `offset` is not
//! read), and the launch writes window[:window.numel()] = base[:window.numel()] + add (at most 1024 entries).
static TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow rngWindow(
    torch::Tensor const& base, int64_t add, torch::Tensor& window, int64_t baseRow, int64_t batch, c10::Device device)
{
    TORCH_CHECK(base.is_cuda() && base.scalar_type() == torch::kInt64 && base.dim() == 1 && base.is_contiguous()
            && window.is_cuda() && window.scalar_type() == torch::kInt64 && window.dim() == 1 && window.is_contiguous()
            && base.device() == device && window.device() == device,
        "minimax_m31_chain_speculative_sampling_mc_win: base / window must be contiguous 1-D int64 tensors on the "
        "probs' device");
    TORCH_CHECK(
        baseRow >= 0 && baseRow + batch <= base.numel() && window.numel() <= base.numel() && window.numel() <= 1024,
        "minimax_m31_chain_speculative_sampling_mc_win: base must hold rows base_row .. base_row + batch and the "
        "window (at most 1024 entries)");
    // The launch writes window while every CTA reads base: an overlap would let a CTA read a rewritten offset.
    auto const* const baseBegin = reinterpret_cast<char const*>(base.const_data_ptr<int64_t>());
    auto const* const baseEnd = baseBegin + base.numel() * static_cast<int64_t>(sizeof(int64_t));
    auto const* const winBegin = reinterpret_cast<char const*>(window.const_data_ptr<int64_t>());
    auto const* const winEnd = winBegin + window.numel() * static_cast<int64_t>(sizeof(int64_t));
    TORCH_CHECK(window.numel() == 0 || winEnd <= baseBegin || baseEnd <= winBegin,
        "minimax_m31_chain_speculative_sampling_mc_win: window must not overlap base");
    TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcRngWindow rng;
    rng.base = base.const_data_ptr<int64_t>();
    rng.add = add;
    rng.baseRow = static_cast<int32_t>(baseRow);
    rng.window = window.data_ptr<int64_t>();
    rng.windowLen = static_cast<int32_t>(window.numel());
    return rng;
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMcWin(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow, torch::Tensor const& base, int64_t add,
    torch::Tensor& window, int64_t baseRow)
{
    auto const rng = rngWindow(base, add, window, baseRow, draftProbs.size(0), draftProbs.device());
    return chainSpecMcImpl(draftProbs, draftTokenIds, targetProbs, seed, offset, workspace, ctasPerRow, rng);
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMcSlotsWin(torch::Tensor const& draftBuf,
    torch::Tensor const& slotIds, int64_t numDraft, torch::Tensor const& draftTokenIds,
    torch::Tensor const& targetProbs, torch::Tensor const& seed, torch::Tensor const& offset, torch::Tensor& workspace,
    int64_t ctasPerRow, torch::Tensor const& base, int64_t add, torch::Tensor& window, int64_t baseRow)
{
    auto const rng = rngWindow(base, add, window, baseRow, slotIds.size(0), draftBuf.device());
    return chainSpecMcSlotsImpl(
        draftBuf, slotIds, numDraft, draftTokenIds, targetProbs, seed, offset, workspace, ctasPerRow, rng);
}

//! Workspace bytes per request of minimax_m31_chain_speculative_sampling_mc.
int64_t minimaxM31ChainSpecSamplingMcWsBytes()
{
    return static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcWorkspaceBytesPerRow());
}

//! The kernel minimax_m31_chain_speculative_sampling_mc launches on the current device: 3 (REJ v3, sm_107) or 2.
int64_t minimaxM31ChainSpecSamplingMcVersion()
{
    return static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcKernelVersion());
}

namespace
{

//! REJ_MC_FI's shared checks on the target probs, the draft tokens, seed / offset and the workspace (the draft rows are
//! checked by the caller); returns the batch size.
int64_t checkChainSpecMcFiInputs(char const* name, torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs,
    torch::Tensor const& seed, torch::Tensor const& offset, torch::Tensor const& workspace, int64_t numDraft,
    int64_t ctasPerRow, int64_t device)
{
    TORCH_CHECK(targetProbs.dim() == 3, name, ": target_probs must be [batch, num_draft + 1, vocab]");
    auto const batchSize = targetProbs.size(0);
    auto const vocabSize = targetProbs.size(2);
    TORCH_CHECK(targetProbs.size(1) == numDraft + 1, name, ": target_probs must be [batch, num_draft + 1, vocab]");
    TORCH_CHECK(draftTokenIds.dim() == 2 && draftTokenIds.size(0) == batchSize && draftTokenIds.size(1) == numDraft,
        name, ": draft_token_ids must be [batch, num_draft]");
    for (auto const* t : {&draftTokenIds, &targetProbs, &seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == device, name,
            ": inputs must be contiguous CUDA tensors on one device");
    }
    TORCH_CHECK(targetProbs.scalar_type() == torch::kFloat32 && draftTokenIds.scalar_type() == torch::kInt32, name,
        ": target_probs must be float32 and draft_token_ids int32");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64 && offset.scalar_type() == torch::kInt64
            && seed.numel() == batchSize && offset.numel() == batchSize,
        name, ": seed and offset must be int64 tensors with one entry per request");
    TORCH_CHECK(vocabSize % 4 == 0 && vocabSize <= TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab
            && reinterpret_cast<uintptr_t>(targetProbs.data_ptr()) % 16 == 0,
        name, ": target_probs must be 16-byte aligned with a vocab divisible by 4 and at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxVocab);
    TORCH_CHECK(numDraft >= 0 && numDraft <= TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxDraft, name,
        ": num_draft must be at most ", TRTLLM_MV2_NS::kernels::minimaxM31::kChainSpecMcMaxDraft);
    TORCH_CHECK(batchSize <= 65535 && ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0, name,
        ": batch <= 65535; ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcFiWorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == device
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * batchSize,
        name, ": workspace must be a contiguous, 16-byte aligned CUDA buffer of at least ", wsBytesPerRow * batchSize,
        " bytes on the inputs' device");
    return batchSize;
}

std::tuple<torch::Tensor, torch::Tensor> launchChainSpecMcFi(char const* name, float const* draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t batchSize, int64_t numDraft, int64_t ctasPerRow,
    TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcDraftRows const& rows)
{
    c10::cuda::CUDAGuard const deviceGuard{targetProbs.device()};
    auto outputTokenIds = torch::empty({batchSize, numDraft + 1}, draftTokenIds.options());
    auto outputNumTokens = torch::empty({batchSize}, draftTokenIds.options());
    if (batchSize == 0)
    {
        return {outputTokenIds, outputNumTokens};
    }
    tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams params;
    params.draftProbs = draftProbs;
    params.draftTokenIds = draftTokenIds.const_data_ptr<int32_t>();
    params.targetProbs = targetProbs.const_data_ptr<float>();
    params.seed = seed.const_data_ptr<int64_t>();
    params.offset = offset.const_data_ptr<int64_t>();
    params.outputTokenIds = outputTokenIds.data_ptr<int32_t>();
    params.outputNumTokens = outputNumTokens.data_ptr<int32_t>();
    params.batchSize = static_cast<int32_t>(batchSize);
    params.numDraftTokens = static_cast<int32_t>(numDraft);
    params.vocabSize = static_cast<int32_t>(targetProbs.size(2));
    auto const err
        = TRTLLM_MV2_NS::kernels::minimaxM31::invokeChainSpecSamplingMcFi(params, static_cast<int>(ctasPerRow),
            workspace.data_ptr(), at::cuda::getCurrentCUDAStream(targetProbs.get_device()), rows);
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
    return {outputTokenIds, outputNumTokens};
}

} // namespace

//! REJ_MC_FI (see csrc/m31/chainSpecSamplingMcFiSm107.h): flashinfer.sampling.chain_speculative_sampling's
//! outputs (deterministic, tensor seed / offset; tokens [batch, numDraft + 1] and num_tokens = emitted + 1 [batch]),
//! bitwise, on batch x ctas_per_row CTAs. workspace: a zero-initialized byte buffer of at least batch x
//! minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes(), reused across calls; ctas_per_row divides 1024 (no
//! co-residency limit).
std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMcFi(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow)
{
    char const* name = "minimax_m31_chain_speculative_sampling_mc_fi";
    TORCH_CHECK(draftProbs.dim() == 3 && draftProbs.is_cuda() && draftProbs.is_contiguous()
            && draftProbs.scalar_type() == torch::kFloat32
            && reinterpret_cast<uintptr_t>(draftProbs.data_ptr()) % 16 == 0,
        name, ": draft_probs must be a contiguous, 16-byte aligned float32 CUDA [batch, num_draft, vocab] tensor");
    auto const numDraft = draftProbs.size(1);
    auto const batchSize = checkChainSpecMcFiInputs(
        name, draftTokenIds, targetProbs, seed, offset, workspace, numDraft, ctasPerRow, draftProbs.get_device());
    TORCH_CHECK(draftProbs.size(0) == batchSize && draftProbs.size(2) == targetProbs.size(2), name,
        ": draft_probs must be [batch, num_draft, vocab]");
    return launchChainSpecMcFi(name, draftProbs.const_data_ptr<float>(), draftTokenIds, targetProbs, seed, offset,
        workspace, batchSize, numDraft, ctasPerRow, TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcDraftRows{});
}

//! minimax_m31_chain_speculative_sampling_mc_fi with slot-indexed draft rows: the draft distributions are
//! draft_buf[slot_ids[r], :num_draft, :vocab] of the spec metadata's [slots, maxDraft, >= vocab] float32 buffer, read
//! in place instead of from a gathered contiguous copy (the same floats, so bitwise the gathered call's results).
//! slot_ids: int64 [batch] on the device; each must index a row of draft_buf (not checked on the host: the caller's
//! gather would index the same rows).
std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMcFiSlots(torch::Tensor const& draftBuf,
    torch::Tensor const& slotIds, int64_t numDraft, torch::Tensor const& draftTokenIds,
    torch::Tensor const& targetProbs, torch::Tensor const& seed, torch::Tensor const& offset, torch::Tensor& workspace,
    int64_t ctasPerRow)
{
    char const* name = "minimax_m31_chain_speculative_sampling_mc_fi_slots";
    TORCH_CHECK(draftBuf.dim() == 3 && draftBuf.is_cuda() && draftBuf.is_contiguous()
            && draftBuf.scalar_type() == torch::kFloat32 && reinterpret_cast<uintptr_t>(draftBuf.data_ptr()) % 16 == 0
            && draftBuf.size(2) % 4 == 0,
        name,
        ": draft_buf must be a contiguous, 16-byte aligned float32 CUDA [slots, maxDraft, vocab_stride] tensor with "
        "vocab_stride divisible by 4");
    TORCH_CHECK(slotIds.dim() == 1 && slotIds.is_cuda() && slotIds.is_contiguous()
            && slotIds.scalar_type() == torch::kInt64 && slotIds.get_device() == draftBuf.get_device(),
        name, ": slot_ids must be a contiguous int64 CUDA [batch] tensor");
    auto const batchSize = checkChainSpecMcFiInputs(
        name, draftTokenIds, targetProbs, seed, offset, workspace, numDraft, ctasPerRow, draftBuf.get_device());
    TORCH_CHECK(slotIds.size(0) == batchSize && numDraft <= draftBuf.size(1) && targetProbs.size(2) <= draftBuf.size(2),
        name, ": slot_ids must hold one slot per request, and num_draft / vocab must fit draft_buf");
    TRTLLM_MV2_NS::kernels::minimaxM31::ChainSpecMcDraftRows rows;
    rows.rowIds = slotIds.const_data_ptr<int64_t>();
    rows.rowStride = draftBuf.stride(0);
    rows.posStride = draftBuf.stride(1);
    return launchChainSpecMcFi(name, draftBuf.const_data_ptr<float>(), draftTokenIds, targetProbs, seed, offset,
        workspace, batchSize, numDraft, ctasPerRow, rows);
}

//! Workspace bytes per request of minimax_m31_chain_speculative_sampling_mc_fi.
int64_t minimaxM31ChainSpecSamplingMcFiWsBytes()
{
    return static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::chainSpecSamplingMcFiWorkspaceBytesPerRow());
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_chain_speculative_sampling_mc(Tensor draft_probs, Tensor draft_token_ids, Tensor target_probs, "
        "Tensor seed, Tensor offset, Tensor(a!) workspace, int ctas_per_row) -> (Tensor, Tensor)");
    m.def(
        "minimax_m31_chain_speculative_sampling_mc_slots(Tensor draft_buf, Tensor slot_ids, int num_draft, "
        "Tensor draft_token_ids, Tensor target_probs, Tensor seed, Tensor offset, Tensor(a!) workspace, "
        "int ctas_per_row) -> (Tensor, Tensor)");
    m.def(
        "minimax_m31_chain_speculative_sampling_mc_win(Tensor draft_probs, Tensor draft_token_ids, "
        "Tensor target_probs, Tensor seed, Tensor offset, Tensor(a!) workspace, int ctas_per_row, Tensor base, "
        "int add, Tensor(b!) window, int base_row) -> (Tensor, Tensor)");
    m.def(
        "minimax_m31_chain_speculative_sampling_mc_slots_win(Tensor draft_buf, Tensor slot_ids, int num_draft, "
        "Tensor draft_token_ids, Tensor target_probs, Tensor seed, Tensor offset, Tensor(a!) workspace, "
        "int ctas_per_row, Tensor base, int add, Tensor(b!) window, int base_row) -> (Tensor, Tensor)");
    // No tensor arguments: a catch-all kernel (as moe_a2a_get_timeout), callable without a device.
    m.def("minimax_m31_chain_speculative_sampling_mc_ws_bytes() -> int");
    m.def("minimax_m31_chain_speculative_sampling_mc_version() -> int");
    m.def(
        "minimax_m31_chain_speculative_sampling_mc_fi(Tensor draft_probs, Tensor draft_token_ids, Tensor "
        "target_probs, Tensor seed, Tensor offset, Tensor(a!) workspace, int ctas_per_row) -> (Tensor, Tensor)");
    m.def(
        "minimax_m31_chain_speculative_sampling_mc_fi_slots(Tensor draft_buf, Tensor slot_ids, int num_draft, "
        "Tensor draft_token_ids, Tensor target_probs, Tensor seed, Tensor offset, Tensor(a!) workspace, "
        "int ctas_per_row) -> (Tensor, Tensor)");
    m.def("minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes() -> int");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_chain_speculative_sampling_mc", &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMc);
    m.impl("minimax_m31_chain_speculative_sampling_mc_slots",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcSlots);
    m.impl("minimax_m31_chain_speculative_sampling_mc_fi", &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcFi);
    m.impl("minimax_m31_chain_speculative_sampling_mc_fi_slots",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcFiSlots);
    m.impl(
        "minimax_m31_chain_speculative_sampling_mc_win", &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcWin);
    m.impl("minimax_m31_chain_speculative_sampling_mc_slots_win",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcSlotsWin);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("minimax_m31_chain_speculative_sampling_mc_ws_bytes",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcWsBytes);
    m.impl("minimax_m31_chain_speculative_sampling_mc_version",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcVersion);
    m.impl("minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31ChainSpecSamplingMcFiWsBytes);
}
