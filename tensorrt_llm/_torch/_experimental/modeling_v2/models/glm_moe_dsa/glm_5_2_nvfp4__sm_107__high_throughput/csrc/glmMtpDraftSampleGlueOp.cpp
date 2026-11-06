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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/csrc/glmMtpDraftSampleGlue.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace sd = TRTLLM_MV2_NS::kernels::glm_rubin_mtp;

bool isAligned16(at::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

int64_t setDraftStepExchangeAdp(sd::DraftSampleGlueAdpParams& params, at::Tensor const& logits,
    at::Tensor& exchange_buffer, at::Tensor& exchange_flags, int64_t group_size, int64_t segment_rows,
    at::Tensor& draft_tokens, int64_t draft_step)
{
    char const* op = "glm_rubin_mtp_sample_glue_adp";
    auto* mcastMem = tensorrt_llm::common::findMcastDevMemBuffer(exchange_buffer.data_ptr());
    TORCH_CHECK(mcastMem != nullptr, op, ": exchange_buffer must come from a McastGPUBuffer");
    TORCH_CHECK(static_cast<int64_t>(mcastMem->getWorldSize()) == group_size, op, ": group_size ", group_size,
        " does not match the workspace group of ", mcastMem->getWorldSize(), " ranks");
    TORCH_CHECK(group_size >= 2 && group_size <= sd::kDraftSampleGlueAdpMaxRanks, op, ": group_size must be 2 to ",
        sd::kDraftSampleGlueAdpMaxRanks, ", got ", group_size);
    TORCH_CHECK(exchange_flags.element_size() == 4 && exchange_flags.numel() >= 9, op,
        ": exchange_flags must hold the nine 32-bit Lamport flags");
    TORCH_CHECK(segment_rows >= 1 && segment_rows <= sd::kDraftSampleGlueAdpMaxSegmentRows, op,
        ": segment_rows must be 1 to ", sd::kDraftSampleGlueAdpMaxSegmentRows, ", got ", segment_rows);
    CHECK_INPUT(logits, at::ScalarType::BFloat16);
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == group_size * segment_rows, op,
        ": logits must be [group_size * segment_rows, vocab_shard]");
    TORCH_CHECK(logits.size(1) % 8 == 0 && isAligned16(logits), op,
        ": logits rows must be 16-byte aligned and a multiple of 8 columns");
    int64_t const bufferBytes = exchange_buffer.numel() * exchange_buffer.element_size() / 3;
    TORCH_CHECK(bufferBytes >= logits.size(0) * group_size * sd::kDraftSampleGluePacketBytes, op,
        ": the Lamport buffer is too small for ", logits.size(0), " stacked rows");
    CHECK_INPUT(draft_tokens, at::ScalarType::Int);
    int64_t const rows = draft_tokens.dim() == 2 ? draft_tokens.size(0) : 0;
    TORCH_CHECK(rows >= 1 && rows <= segment_rows && draft_step >= 0 && draft_step < draft_tokens.size(1), op,
        ": draft_tokens must be [rows, draft_len] with 1 to segment_rows rows and draft_step inside");

    sd::DraftSampleGlueParams& step = params.step;
    step.logits = reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr());
    step.logitsStride = logits.stride(0);
    step.numRows = static_cast<int>(rows);
    step.vocabShard = static_cast<int>(logits.size(1));
    step.rank = static_cast<int>(mcastMem->getRank());
    step.tpSize = static_cast<int>(group_size);
    step.exchangeLocal = exchange_buffer.data_ptr();
    step.exchangeMcast = mcastMem->getMulticastPtr();
    step.exchangeFlags = reinterpret_cast<uint32_t*>(exchange_flags.data_ptr());
    step.draftTokens = draft_tokens.data_ptr<int32_t>();
    step.numDraftTokens = static_cast<int>(draft_tokens.size(1));
    step.draftStep = static_cast<int>(draft_step);
    params.segmentRows = static_cast<int>(segment_rows);
    return rows;
}

void setDraftStepGlue(sd::DraftSampleGlueParams& params, char const* op, int64_t rows,
    std::optional<at::Tensor> const& embed_table, std::optional<at::Tensor> const& hidden,
    std::optional<at::Tensor> const& enorm_weight, std::optional<at::Tensor> const& hnorm_weight, double eps,
    int64_t col_lo, int64_t col_hi, std::optional<at::Tensor>& glue_out)
{
    if (!glue_out.has_value())
    {
        return;
    }
    constexpr int64_t kHidden = sd::kDraftSampleGlueHidden;
    TORCH_CHECK(col_lo >= 0 && col_lo < col_hi && col_hi <= 2 * kHidden && col_lo % 8 == 0 && col_hi % 8 == 0, op,
        ": invalid eh_proj input window [", col_lo, ", ", col_hi, ")");
    TORCH_CHECK(enorm_weight.has_value() && hnorm_weight.has_value(), op, ": norm weights missing");
    for (at::Tensor const& weight : {enorm_weight.value(), hnorm_weight.value()})
    {
        CHECK_INPUT(weight, at::ScalarType::BFloat16);
        TORCH_CHECK(weight.numel() == kHidden && isAligned16(weight), op, ": norm weights must be [6144]");
    }
    TORCH_CHECK(glue_out->scalar_type() == at::ScalarType::BFloat16 && glue_out->dim() == 2 && glue_out->size(0) == rows
            && glue_out->size(1) == col_hi - col_lo && glue_out->stride(1) == 1 && glue_out->stride(0) % 8 == 0
            && isAligned16(*glue_out),
        op, ": glue_out must be a 16-byte aligned BF16 [rows, col_hi - col_lo] view");
    if (col_lo < kHidden)
    {
        TORCH_CHECK(embed_table.has_value(), op, ": the window needs the embedding table");
        CHECK_INPUT(embed_table.value(), at::ScalarType::BFloat16);
        TORCH_CHECK(embed_table->dim() == 2 && embed_table->size(1) == kHidden && isAligned16(*embed_table), op,
            ": embed_table must be [vocab, 6144]");
        params.embedTable = reinterpret_cast<__nv_bfloat16 const*>(embed_table->data_ptr());
        params.vocabSize = embed_table->size(0);
    }
    if (col_hi > kHidden)
    {
        TORCH_CHECK(hidden.has_value(), op, ": the window needs the hidden rows");
        TORCH_CHECK(hidden->scalar_type() == at::ScalarType::BFloat16 && hidden->dim() == 2 && hidden->size(0) == rows
                && hidden->size(1) == kHidden && hidden->stride(1) == 1 && hidden->stride(0) % 8 == 0
                && isAligned16(*hidden),
            op, ": hidden must be a 16-byte aligned BF16 [rows, 6144] view");
        params.hidden = reinterpret_cast<__nv_bfloat16 const*>(hidden->data_ptr());
        params.hiddenStride = hidden->stride(0);
    }
    params.enormWeight = reinterpret_cast<__nv_bfloat16 const*>(enorm_weight->data_ptr());
    params.hnormWeight = reinterpret_cast<__nv_bfloat16 const*>(hnorm_weight->data_ptr());
    params.eps = static_cast<float>(eps);
    params.colLo = static_cast<int>(col_lo);
    params.colHi = static_cast<int>(col_hi);
    params.glueOut = reinterpret_cast<__nv_bfloat16*>(glue_out->data_ptr());
    params.glueOutStride = glue_out->stride(0);
}

void setDraftStepOutputRows(sd::DraftSampleGlueParams& params, char const* op, int64_t rows,
    at::Tensor const& draft_tokens, std::optional<at::Tensor> const& accepted_tokens,
    std::optional<at::Tensor> const& batch_indices, std::optional<at::Tensor> const& num_accepted,
    std::optional<at::Tensor>& next_new_tokens)
{
    if (!next_new_tokens.has_value())
    {
        return;
    }
    TORCH_CHECK(accepted_tokens.has_value() && batch_indices.has_value() && num_accepted.has_value(), op,
        ": next_new_tokens needs the accepted tokens, batch indices and counts");
    CHECK_INPUT(next_new_tokens.value(), at::ScalarType::Int);
    CHECK_TH_CUDA(accepted_tokens.value());
    TORCH_CHECK(accepted_tokens->scalar_type() == at::ScalarType::Int && accepted_tokens->dim() == 2
            && accepted_tokens->stride(1) == 1,
        op, ": accepted_tokens must be int32 [batch, draft_len + 1] rows");
    CHECK_INPUT(batch_indices.value(), at::ScalarType::Int);
    CHECK_INPUT(num_accepted.value(), at::ScalarType::Int);
    TORCH_CHECK(next_new_tokens->numel() >= rows * (draft_tokens.size(1) + 1) && batch_indices->numel() >= rows
            && num_accepted->numel() >= rows,
        op, ": next_new_tokens must be [rows, draft_len + 1]");
    params.acceptedTokens = accepted_tokens->data_ptr<int32_t>();
    params.acceptedTokensStride = accepted_tokens->stride(0);
    params.batchIndices = batch_indices->data_ptr<int32_t>();
    params.numAccepted = num_accepted->data_ptr<int32_t>();
    params.nextNewTokens = next_new_tokens->data_ptr<int32_t>();
}

void setDraftStepState(sd::DraftSampleGlueParams& params, char const* op, int64_t rows,
    std::optional<at::Tensor> const& positions_in, std::optional<at::Tensor>& positions_out,
    std::optional<at::Tensor> const& gather_ids, std::optional<at::Tensor>& seq_lens, bool fill_seq_lens,
    std::optional<at::Tensor>& kv_lens, int64_t kv_update, std::optional<at::Tensor> const& rewind_accepted,
    int64_t draft_len)
{
    if (positions_out.has_value())
    {
        TORCH_CHECK(positions_in.has_value(), op, ": positions_out needs positions_in");
        auto const posType = positions_in->scalar_type();
        TORCH_CHECK(posType == at::ScalarType::Int || posType == at::ScalarType::Long, op,
            ": position ids must be int32 or int64");
        CHECK_INPUT(positions_in.value(), posType);
        CHECK_INPUT(positions_out.value(), posType);
        TORCH_CHECK(positions_out->numel() >= rows, op, ": positions_out holds one id per row");
        params.positionsIn = positions_in->data_ptr();
        params.positionsOut = positions_out->data_ptr();
        params.positions64 = posType == at::ScalarType::Long;
        if (gather_ids.has_value())
        {
            CHECK_INPUT(gather_ids.value(), at::ScalarType::Long);
            TORCH_CHECK(gather_ids->numel() >= rows, op, ": gather_ids holds one index per row");
            params.gatherIds = gather_ids->data_ptr<int64_t>();
        }
    }

    if (seq_lens.has_value())
    {
        CHECK_INPUT(seq_lens.value(), at::ScalarType::Int);
        TORCH_CHECK(seq_lens->numel() >= rows, op, ": seq_lens holds one length per row");
        params.seqLens = seq_lens->data_ptr<int32_t>();
        params.fillSeqLens = fill_seq_lens;
    }
    if (kv_lens.has_value())
    {
        CHECK_INPUT(kv_lens.value(), at::ScalarType::Int);
        TORCH_CHECK(kv_lens->numel() >= rows, op, ": kv_lens holds one length per row");
        TORCH_CHECK(kv_update >= 0 && kv_update <= 2, op, ": unknown kv_update ", kv_update);
        params.kvLens = kv_lens->data_ptr<int32_t>();
        params.kvUpdate = static_cast<sd::DraftSampleGlueKvUpdate>(kv_update);
        if (params.kvUpdate == sd::DraftSampleGlueKvUpdate::kRewind)
        {
            TORCH_CHECK(rewind_accepted.has_value(), op, ": kv_update 2 needs rewind_accepted");
            CHECK_INPUT(rewind_accepted.value(), at::ScalarType::Int);
            TORCH_CHECK(rewind_accepted->numel() >= rows, op, ": rewind_accepted holds one count per row");
            params.kvRewindAccepted = rewind_accepted->data_ptr<int32_t>();
        }
        params.draftLen = static_cast<int>(draft_len);
    }
}

} // namespace

void rubinMtpDraftSampleGlueAdp(at::Tensor const& logits, at::Tensor& exchange_buffer, at::Tensor& exchange_flags,
    int64_t group_size, int64_t segment_rows, at::Tensor& draft_tokens, int64_t draft_step,
    std::optional<at::Tensor> const& embed_table, std::optional<at::Tensor> const& hidden,
    std::optional<at::Tensor> const& enorm_weight, std::optional<at::Tensor> const& hnorm_weight, double eps,
    int64_t col_lo, int64_t col_hi, std::optional<at::Tensor> glue_out, std::optional<at::Tensor> const& positions_in,
    std::optional<at::Tensor> positions_out, std::optional<at::Tensor> const& gather_ids,
    std::optional<at::Tensor> seq_lens, bool fill_seq_lens, std::optional<at::Tensor> kv_lens, int64_t kv_update,
    std::optional<at::Tensor> const& rewind_accepted, int64_t draft_len,
    std::optional<at::Tensor> const& accepted_tokens, std::optional<at::Tensor> const& batch_indices,
    std::optional<at::Tensor> const& num_accepted, std::optional<at::Tensor> next_new_tokens)
{
    char const* op = "glm_rubin_mtp_sample_glue_adp";
    sd::DraftSampleGlueAdpParams params;
    int64_t const rows = setDraftStepExchangeAdp(
        params, logits, exchange_buffer, exchange_flags, group_size, segment_rows, draft_tokens, draft_step);
    setDraftStepGlue(
        params.step, op, rows, embed_table, hidden, enorm_weight, hnorm_weight, eps, col_lo, col_hi, glue_out);
    setDraftStepState(params.step, op, rows, positions_in, positions_out, gather_ids, seq_lens, fill_seq_lens, kv_lens,
        kv_update, rewind_accepted, draft_len);
    setDraftStepOutputRows(
        params.step, op, rows, draft_tokens, accepted_tokens, batch_indices, num_accepted, next_new_tokens);

    sd::invokeDraftSampleGlueAdp(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "glm_rubin_mtp_sample_glue_adp(Tensor logits, Tensor(a!) exchange_buffer, Tensor(b!) exchange_flags, int "
        "group_size, "
        "int segment_rows, Tensor(c!) draft_tokens, int draft_step, Tensor? embed_table, Tensor? hidden, "
        "Tensor? enorm_weight, Tensor? hnorm_weight, float eps, int col_lo, int col_hi, Tensor(d!)? glue_out, "
        "Tensor? positions_in, Tensor(e!)? positions_out, Tensor? gather_ids, Tensor(f!)? seq_lens, "
        "bool fill_seq_lens, Tensor(g!)? kv_lens, int kv_update, Tensor? rewind_accepted, int draft_len, "
        "Tensor? accepted_tokens, Tensor? batch_indices, Tensor? num_accepted, Tensor(h!)? next_new_tokens) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("glm_rubin_mtp_sample_glue_adp", &TRTLLM_MV2_NS::torch_ext::rubinMtpDraftSampleGlueAdp);
}
