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

// Torch ops of the draft tail (greedy draft-token selection over a
// vocab-sharded LM head). See draftTail.h.

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/draftTail.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
namespace tk = TRTLLM_MV2_NS::draft_tail;

void fillCommon(tk::DraftTailParams& p, int64_t rows, int64_t vocabLocal, at::Tensor& tokens, at::Tensor& scratch,
    std::optional<at::Tensor> const& keysOut, int64_t vocabOffset, int64_t tpRank, int64_t tpSize, int64_t mode,
    int64_t mcPtr, int64_t ucPtr, std::optional<at::Tensor> const& peerPtrs, int64_t timeoutCycles, c10::Device device)
{
    TORCH_CHECK(rows >= 1 && rows <= tk::kMaxRows, "glm_moe_dsa_draft_tail: rows must be in [1, ", tk::kMaxRows, "]");
    TORCH_CHECK(tokens.device() == device && tokens.scalar_type() == at::kInt && tokens.is_contiguous()
            && tokens.numel() >= rows,
        "glm_moe_dsa_draft_tail: tokens must be a contiguous int32 tensor with at least rows elements");
    TORCH_CHECK(scratch.device() == device && scratch.scalar_type() == at::kLong && scratch.is_contiguous()
            && scratch.numel() >= tk::kScratchWords,
        "glm_moe_dsa_draft_tail: scratch must be a contiguous int64 tensor with ", tk::kScratchWords, " elements");
    TORCH_CHECK(vocabOffset >= 0 && vocabOffset + vocabLocal <= 0xFFFFFFFFll,
        "glm_moe_dsa_draft_tail: vocab range out of bounds");
    TORCH_CHECK(tpSize >= 1 && tpSize <= tk::kMaxRanks && tpRank >= 0 && tpRank < tpSize,
        "glm_moe_dsa_draft_tail: bad tp rank");
    TORCH_CHECK(mode >= 0 && mode <= 4, "glm_moe_dsa_draft_tail: bad exchange mode");
    p.rows = static_cast<int>(rows);
    p.vocabLocal = static_cast<int>(vocabLocal);
    p.vocabOffset = vocabOffset;
    p.tokens = tokens.data_ptr<int32_t>();
    p.scratch = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
    if (keysOut.has_value())
    {
        TORCH_CHECK(keysOut->device() == device && keysOut->scalar_type() == at::kLong && keysOut->is_contiguous()
                && keysOut->numel() >= rows,
            "glm_moe_dsa_draft_tail: keys_out must be a contiguous int64 tensor with at least rows elements");
        p.keysOut = reinterpret_cast<uint64_t*>(keysOut->data_ptr<int64_t>());
    }
    p.tpRank = static_cast<int>(tpRank);
    p.tpSize = static_cast<int>(tpSize);
    p.mode = static_cast<tk::ExchangeMode>(mode);
    p.timeoutCycles = timeoutCycles;
    if (p.mode != tk::ExchangeMode::kNone && tpSize > 1)
    {
        TORCH_CHECK(ucPtr != 0 && ucPtr % 16 == 0, "glm_moe_dsa_draft_tail: exchange needs the local unicast buffer");
        p.ucSlotsLocal = reinterpret_cast<uint64_t*>(ucPtr);
        if (p.mode == tk::ExchangeMode::kMulticast || p.mode == tk::ExchangeMode::kMulticastDirect)
        {
            TORCH_CHECK(
                mcPtr != 0 && mcPtr % 16 == 0, "glm_moe_dsa_draft_tail: multicast exchange needs the multicast buffer");
            p.mcSlots = reinterpret_cast<uint64_t*>(mcPtr);
        }
        else
        {
            TORCH_CHECK(peerPtrs.has_value() && peerPtrs->device() == device && peerPtrs->scalar_type() == at::kLong
                    && peerPtrs->is_contiguous() && peerPtrs->numel() >= tpSize,
                "glm_moe_dsa_draft_tail: unicast exchange needs a device int64 tensor of tp_size peer addresses");
            p.ucSlotsPeers = reinterpret_cast<uint64_t* const*>(peerPtrs->data_ptr<int64_t>());
        }
    }
}

} // namespace

void draftTail(at::Tensor const& hidden, at::Tensor const& weight, at::Tensor& tokens, at::Tensor& scratch,
    std::optional<at::Tensor> const& keysOut, int64_t vocabOffset, int64_t tpRank, int64_t tpSize, int64_t mode,
    int64_t mcPtr, int64_t ucPtr, std::optional<at::Tensor> const& peerPtrs, int64_t variant, int64_t numCtas,
    int64_t keepPermille, bool pdl, int64_t timeoutCycles, std::optional<at::Tensor> const& logitsOut)
{
    TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 && hidden.scalar_type() == at::kBFloat16,
        "glm_moe_dsa_draft_tail: hidden must be a 2D BF16 CUDA tensor");
    TORCH_CHECK(weight.device() == hidden.device() && weight.dim() == 2 && weight.scalar_type() == at::kBFloat16
            && weight.is_contiguous(),
        "glm_moe_dsa_draft_tail: weight must be a contiguous 2D BF16 tensor on the hidden device");
    TORCH_CHECK(hidden.size(1) == 6144 && weight.size(1) == 6144, "glm_moe_dsa_draft_tail: hidden size must be 6144");
    TORCH_CHECK(hidden.stride(1) == 1 && hidden.stride(0) % 8 == 0
            && reinterpret_cast<uintptr_t>(hidden.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 32 == 0,
        "glm_moe_dsa_draft_tail: hidden rows must be 16-byte aligned and the weight 32-byte aligned");
    tk::DraftTailParams p;
    fillCommon(p, hidden.size(0), weight.size(0), tokens, scratch, keysOut, vocabOffset, tpRank, tpSize, mode, mcPtr,
        ucPtr, peerPtrs, timeoutCycles, hidden.device());
    p.hidden = reinterpret_cast<__nv_bfloat16 const*>(hidden.data_ptr());
    p.hiddenStride = hidden.stride(0);
    p.weight = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    p.hiddenDim = static_cast<int>(hidden.size(1));
    TORCH_CHECK(
        keepPermille >= 0 && keepPermille <= 1000, "glm_moe_dsa_draft_tail: keep_permille must be in [0, 1000]");
    p.keepPermille = static_cast<int>(keepPermille);
    if (logitsOut.has_value())
    {
        TORCH_CHECK(logitsOut->device() == hidden.device() && logitsOut->scalar_type() == at::kBFloat16
                && logitsOut->dim() == 2 && logitsOut->size(0) >= hidden.size(0) && logitsOut->size(1) == weight.size(0)
                && logitsOut->stride(1) == 1,
            "glm_moe_dsa_draft_tail: logits_out must be a BF16 [rows, vocab_local] tensor with unit last stride");
        p.logitsOut = reinterpret_cast<__nv_bfloat16*>(logitsOut->data_ptr());
        p.logitsOutStride = logitsOut->stride(0);
    }
    tk::FusedLaunchConfig config;
    config.numCtas = static_cast<int>(numCtas);
    config.variant = static_cast<int>(variant);
    config.pdl = pdl;
    tk::invokeDraftTailFused(p, config, at::cuda::getCurrentCUDAStream(hidden.get_device()));
}

void draftTailFromLogits(at::Tensor const& logits, at::Tensor& tokens, at::Tensor& scratch,
    std::optional<at::Tensor> const& keysOut, int64_t vocabOffset, int64_t tpRank, int64_t tpSize, int64_t mode,
    int64_t mcPtr, int64_t ucPtr, std::optional<at::Tensor> const& peerPtrs, bool pdl, int64_t timeoutCycles)
{
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == at::kBFloat16 && logits.stride(1) == 1,
        "glm_moe_dsa_draft_tail: logits must be a 2D BF16 CUDA tensor with unit last stride");
    tk::DraftTailParams p;
    fillCommon(p, logits.size(0), logits.size(1), tokens, scratch, keysOut, vocabOffset, tpRank, tpSize, mode, mcPtr,
        ucPtr, peerPtrs, timeoutCycles, logits.device());
    p.logits = reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr());
    p.logitsStride = logits.stride(0);
    tk::invokeDraftTailFromLogits(p, pdl, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "draft_tail(Tensor hidden, Tensor weight, Tensor(a!) tokens, Tensor(b!) scratch, "
        "Tensor(c!)? keys_out, int vocab_offset, int tp_rank, int tp_size, int mode, int mc_ptr, int uc_ptr, "
        "Tensor? peer_ptrs, int variant, int num_ctas, int keep_permille, bool pdl, int timeout_cycles, "
        "Tensor(d!)? logits_out=None) -> ()");
    m.def(
        "draft_tail_from_logits(Tensor logits, Tensor(a!) tokens, Tensor(b!) scratch, Tensor(c!)? "
        "keys_out, "
        "int vocab_offset, int tp_rank, int tp_size, int mode, int mc_ptr, int uc_ptr, Tensor? peer_ptrs, bool pdl, "
        "int timeout_cycles) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("draft_tail", &TRTLLM_MV2_NS::torch_ext::draftTail);
    m.impl("draft_tail_from_logits", &TRTLLM_MV2_NS::torch_ext::draftTailFromLogits);
}
