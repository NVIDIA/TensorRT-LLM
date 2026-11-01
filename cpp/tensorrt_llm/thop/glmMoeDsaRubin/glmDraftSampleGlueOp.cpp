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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmDraftSampleGlue.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = tensorrt_llm::kernels::glm_moe_dsa;

bool isAligned16(at::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// The logits, exchange workspace and draft-token fields of both draft-step ops; slotBytes is what one rank publishes
// per row. Returns the number of rows.
int64_t setDraftStepExchange(tk::DraftSampleGlueParams& params, char const* op, at::Tensor const& logits,
    at::Tensor& exchange_buffer, at::Tensor& exchange_flags, int64_t group_size, at::Tensor& draft_tokens,
    int64_t draft_step, int64_t slotBytes)
{
    auto* mcastMem = tensorrt_llm::common::findMcastDevMemBuffer(exchange_buffer.data_ptr());
    TORCH_CHECK(mcastMem != nullptr, op, ": exchange_buffer must come from a McastGPUBuffer");
    TORCH_CHECK(static_cast<int64_t>(mcastMem->getWorldSize()) == group_size, op, ": group_size ", group_size,
        " does not match the workspace group of ", mcastMem->getWorldSize(), " ranks");
    TORCH_CHECK(exchange_flags.element_size() == 4 && exchange_flags.numel() >= 9, op,
        ": exchange_flags must hold the nine 32-bit Lamport flags");
    CHECK_INPUT(logits, at::ScalarType::BFloat16);
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= 1 && logits.size(0) <= tk::kDraftSampleGlueMaxRows, op,
        ": logits must be [rows, vocab_shard] with 1 to ", tk::kDraftSampleGlueMaxRows, " rows");
    int64_t const rows = logits.size(0);
    TORCH_CHECK(logits.size(1) % 8 == 0 && isAligned16(logits), op,
        ": logits rows must be 16-byte aligned and a multiple of 8 columns");
    int64_t const bufferBytes = exchange_buffer.numel() * exchange_buffer.element_size() / 3;
    TORCH_CHECK(
        bufferBytes >= rows * group_size * slotBytes, op, ": the Lamport buffer is too small for ", rows, " rows");
    CHECK_INPUT(draft_tokens, at::ScalarType::Int);
    TORCH_CHECK(
        draft_tokens.dim() == 2 && draft_tokens.size(0) == rows && draft_step >= 0 && draft_step < draft_tokens.size(1),
        op, ": draft_tokens must be [rows, draft_len] with draft_step inside");

    params.logits = reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr());
    params.logitsStride = logits.stride(0);
    params.numRows = static_cast<int>(rows);
    params.vocabShard = static_cast<int>(logits.size(1));
    params.rank = static_cast<int>(mcastMem->getRank());
    params.tpSize = static_cast<int>(group_size);
    params.exchangeLocal = exchange_buffer.data_ptr();
    params.exchangeMcast = mcastMem->getMulticastPtr();
    params.exchangeFlags = reinterpret_cast<uint32_t*>(exchange_flags.data_ptr());
    params.draftTokens = draft_tokens.data_ptr<int32_t>();
    params.numDraftTokens = static_cast<int>(draft_tokens.size(1));
    params.draftStep = static_cast<int>(draft_step);
    return rows;
}

// The next step's position ids, sequence / KV lengths and DSA decode metadata of both draft-step ops.
void setDraftStepState(tk::DraftSampleGlueParams& params, char const* op, int64_t rows,
    std::optional<at::Tensor> const& positions_in, std::optional<at::Tensor>& positions_out,
    std::optional<at::Tensor> const& gather_ids, std::optional<at::Tensor>& seq_lens, bool fill_seq_lens,
    std::optional<at::Tensor>& kv_lens, int64_t kv_update, std::optional<at::Tensor> const& rewind_accepted,
    int64_t draft_len, std::optional<at::Tensor> const& block_offsets, std::vector<int64_t> const& dsa_layout,
    std::optional<at::Tensor>& req_idx_per_token, std::optional<at::Tensor>& slot_mapping_fp8,
    std::optional<at::Tensor>& slot_mapping_scale, std::optional<at::Tensor>& gen_kv_indptr,
    std::optional<at::Tensor>& gen_cached_token_indptr)
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
        params.kvUpdate = static_cast<tk::DraftSampleGlueKvUpdate>(kv_update);
        if (params.kvUpdate == tk::DraftSampleGlueKvUpdate::kRewind)
        {
            TORCH_CHECK(rewind_accepted.has_value(), op, ": kv_update 2 needs rewind_accepted");
            CHECK_INPUT(rewind_accepted.value(), at::ScalarType::Int);
            TORCH_CHECK(rewind_accepted->numel() >= rows, op, ": rewind_accepted holds one count per row");
            params.kvRewindAccepted = rewind_accepted->data_ptr<int32_t>();
        }
        params.draftLen = static_cast<int>(draft_len);
    }

    if (req_idx_per_token.has_value())
    {
        TORCH_CHECK(seq_lens.has_value() && kv_lens.has_value() && block_offsets.has_value()
                && slot_mapping_fp8.has_value() && slot_mapping_scale.has_value() && gen_kv_indptr.has_value()
                && gen_cached_token_indptr.has_value() && dsa_layout.size() == 7,
            op, ": the DSA metadata needs the lengths, the block offsets, all five outputs and the layout");
        CHECK_TH_CUDA(block_offsets.value());
        TORCH_CHECK(block_offsets->scalar_type() == at::ScalarType::Int && block_offsets->dim() == 2
                && block_offsets->size(0) >= rows,
            op, ": block_offsets must be int32 [rows, max_blocks]");
        int64_t const numTokens = dsa_layout[0];
        CHECK_INPUT(req_idx_per_token.value(), at::ScalarType::Int);
        CHECK_INPUT(slot_mapping_fp8.value(), at::ScalarType::Long);
        CHECK_INPUT(slot_mapping_scale.value(), at::ScalarType::Long);
        CHECK_INPUT(gen_kv_indptr.value(), at::ScalarType::Long);
        CHECK_INPUT(gen_cached_token_indptr.value(), at::ScalarType::Long);
        TORCH_CHECK(req_idx_per_token->numel() >= numTokens && slot_mapping_fp8->numel() >= numTokens
                && slot_mapping_scale->numel() >= numTokens && gen_kv_indptr->numel() >= rows + 1
                && gen_cached_token_indptr->numel() >= rows + 1,
            op, ": DSA metadata outputs too small");
        params.blockOffsets = block_offsets->data_ptr<int32_t>();
        params.blockOffsetsStride0 = block_offsets->stride(0);
        params.blockOffsetsStride1 = block_offsets->stride(1);
        params.maxBlocks = static_cast<int>(block_offsets->size(1));
        params.numTokens = static_cast<int>(numTokens);
        params.maxQueryLanes = static_cast<int>(dsa_layout[1]);
        params.tokensPerBlock = static_cast<int>(dsa_layout[2]);
        params.dataBytesPerToken = dsa_layout[3];
        params.scaleSize = dsa_layout[4];
        params.blockStride = dsa_layout[5];
        params.scaleBaseOffset = dsa_layout[6];
        params.reqIdxPerToken = req_idx_per_token->data_ptr<int32_t>();
        params.slotMappingFp8 = slot_mapping_fp8->data_ptr<int64_t>();
        params.slotMappingScale = slot_mapping_scale->data_ptr<int64_t>();
        params.genKvIndptr = gen_kv_indptr->data_ptr<int64_t>();
        params.genCachedTokenIndptr = gen_cached_token_indptr->data_ptr<int64_t>();
    }
}

} // namespace

// One MTP draft step after the vocabulary-sharded draft LM head (see draftSampleGlue.h): the greedy draft token of
// every row into column draft_step of draft_tokens, through the dedicated MNNVL Lamport workspace exchange_buffer /
// exchange_flags; optionally the next step's eh_proj input window (glue_out), position ids, sequence and KV lengths
// (kv_update 0: none, 1: + 1, 2: - (draft_len - rewind_accepted)), the DSA decode metadata (dsa_layout = [num_tokens,
// max_query_lanes, tokens_per_block, data_bytes_per_token, scale_size, block_stride, scale_base_offset]) and the
// next-new-tokens rows after the last step.
void glmDraftSampleGlue(at::Tensor const& logits, at::Tensor& exchange_buffer, at::Tensor& exchange_flags,
    int64_t group_size, at::Tensor& draft_tokens, int64_t draft_step, std::optional<at::Tensor> const& embed_table,
    std::optional<at::Tensor> const& hidden, std::optional<at::Tensor> const& enorm_weight,
    std::optional<at::Tensor> const& hnorm_weight, double eps, int64_t col_lo, int64_t col_hi,
    std::optional<at::Tensor> glue_out, std::optional<at::Tensor> const& positions_in,
    std::optional<at::Tensor> positions_out, std::optional<at::Tensor> const& gather_ids,
    std::optional<at::Tensor> seq_lens, bool fill_seq_lens, std::optional<at::Tensor> kv_lens, int64_t kv_update,
    std::optional<at::Tensor> const& rewind_accepted, int64_t draft_len, std::optional<at::Tensor> const& block_offsets,
    std::vector<int64_t> const& dsa_layout, std::optional<at::Tensor> req_idx_per_token,
    std::optional<at::Tensor> slot_mapping_fp8, std::optional<at::Tensor> slot_mapping_scale,
    std::optional<at::Tensor> gen_kv_indptr, std::optional<at::Tensor> gen_cached_token_indptr,
    std::optional<at::Tensor> const& accepted_tokens, std::optional<at::Tensor> const& batch_indices,
    std::optional<at::Tensor> const& num_accepted, std::optional<at::Tensor> next_new_tokens)
{
    tk::DraftSampleGlueParams params;
    int64_t const rows = setDraftStepExchange(params, "glm_draft_sample_glue", logits, exchange_buffer, exchange_flags,
        group_size, draft_tokens, draft_step, tk::kDraftSampleGluePacketBytes);

    if (glue_out.has_value())
    {
        constexpr int64_t kHidden = tk::kDraftSampleGlueHidden;
        TORCH_CHECK(col_lo >= 0 && col_lo < col_hi && col_hi <= 2 * kHidden && col_lo % 8 == 0 && col_hi % 8 == 0,
            "glm_draft_sample_glue: invalid eh_proj input window [", col_lo, ", ", col_hi, ")");
        TORCH_CHECK(
            enorm_weight.has_value() && hnorm_weight.has_value(), "glm_draft_sample_glue: norm weights missing");
        for (at::Tensor const& weight : {enorm_weight.value(), hnorm_weight.value()})
        {
            CHECK_INPUT(weight, at::ScalarType::BFloat16);
            TORCH_CHECK(
                weight.numel() == kHidden && isAligned16(weight), "glm_draft_sample_glue: norm weights must be [6144]");
        }
        TORCH_CHECK(glue_out->scalar_type() == at::ScalarType::BFloat16 && glue_out->dim() == 2
                && glue_out->size(0) == rows && glue_out->size(1) == col_hi - col_lo && glue_out->stride(1) == 1
                && glue_out->stride(0) % 8 == 0 && isAligned16(*glue_out),
            "glm_draft_sample_glue: glue_out must be a 16-byte aligned BF16 [rows, col_hi - col_lo] view");
        if (col_lo < kHidden)
        {
            TORCH_CHECK(embed_table.has_value(), "glm_draft_sample_glue: the window needs the embedding table");
            CHECK_INPUT(embed_table.value(), at::ScalarType::BFloat16);
            TORCH_CHECK(embed_table->dim() == 2 && embed_table->size(1) == kHidden && isAligned16(*embed_table),
                "glm_draft_sample_glue: embed_table must be [vocab, 6144]");
            params.embedTable = reinterpret_cast<__nv_bfloat16 const*>(embed_table->data_ptr());
            params.vocabSize = embed_table->size(0);
        }
        if (col_hi > kHidden)
        {
            TORCH_CHECK(hidden.has_value(), "glm_draft_sample_glue: the window needs the hidden rows");
            TORCH_CHECK(hidden->scalar_type() == at::ScalarType::BFloat16 && hidden->dim() == 2
                    && hidden->size(0) == rows && hidden->size(1) == kHidden && hidden->stride(1) == 1
                    && hidden->stride(0) % 8 == 0 && isAligned16(*hidden),
                "glm_draft_sample_glue: hidden must be a 16-byte aligned BF16 [rows, 6144] view");
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

    setDraftStepState(params, "glm_draft_sample_glue", rows, positions_in, positions_out, gather_ids, seq_lens,
        fill_seq_lens, kv_lens, kv_update, rewind_accepted, draft_len, block_offsets, dsa_layout, req_idx_per_token,
        slot_mapping_fp8, slot_mapping_scale, gen_kv_indptr, gen_cached_token_indptr);

    if (next_new_tokens.has_value())
    {
        TORCH_CHECK(accepted_tokens.has_value() && batch_indices.has_value() && num_accepted.has_value(),
            "glm_draft_sample_glue: next_new_tokens needs the accepted tokens, batch indices and counts");
        CHECK_INPUT(next_new_tokens.value(), at::ScalarType::Int);
        CHECK_TH_CUDA(accepted_tokens.value());
        TORCH_CHECK(accepted_tokens->scalar_type() == at::ScalarType::Int && accepted_tokens->dim() == 2
                && accepted_tokens->stride(1) == 1,
            "glm_draft_sample_glue: accepted_tokens must be int32 [batch, draft_len + 1] rows");
        CHECK_INPUT(batch_indices.value(), at::ScalarType::Int);
        CHECK_INPUT(num_accepted.value(), at::ScalarType::Int);
        TORCH_CHECK(next_new_tokens->numel() >= rows * (draft_tokens.size(1) + 1) && batch_indices->numel() >= rows
                && num_accepted->numel() >= rows,
            "glm_draft_sample_glue: next_new_tokens must be [rows, draft_len + 1]");
        params.acceptedTokens = accepted_tokens->data_ptr<int32_t>();
        params.acceptedTokensStride = accepted_tokens->stride(0);
        params.batchIndices = batch_indices->data_ptr<int32_t>();
        params.numAccepted = num_accepted->data_ptr<int32_t>();
        params.nextNewTokens = next_new_tokens->data_ptr<int32_t>();
    }

    tk::invokeDraftSampleGlue(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

// One MTP draft step that also replaces the next step's eh_proj with its all-reduce and input_layernorm (see
// invokeDraftEhprojTable): the draft token and the next step's state as glm_draft_sample_glue computes them,
// residual_out = x = BF16(token_table[token] + hidden_half), one FP32 add, and normed_out = input_layernorm(x) with
// norm_weight and eps. hidden_half is this rank's FP32 [rows, 6144 / group_size] share of eh_proj's hidden half,
// token_table this rank's [vocab_shard, 6144] rows; exchange_buffer / exchange_flags are a dedicated MNNVL Lamport
// workspace of draftEhprojTableSlotBytes(group_size) bytes per rank and row.
void glmDraftEhprojTable(at::Tensor const& logits, at::Tensor& exchange_buffer, at::Tensor& exchange_flags,
    int64_t group_size, at::Tensor& draft_tokens, int64_t draft_step, at::Tensor const& hidden_half,
    at::Tensor const& token_table, at::Tensor const& norm_weight, double eps, at::Tensor& normed_out,
    at::Tensor& residual_out, std::optional<at::Tensor> const& positions_in, std::optional<at::Tensor> positions_out,
    std::optional<at::Tensor> const& gather_ids, std::optional<at::Tensor> seq_lens, bool fill_seq_lens,
    std::optional<at::Tensor> kv_lens, int64_t kv_update, std::optional<at::Tensor> const& rewind_accepted,
    int64_t draft_len, std::optional<at::Tensor> const& block_offsets, std::vector<int64_t> const& dsa_layout,
    std::optional<at::Tensor> req_idx_per_token, std::optional<at::Tensor> slot_mapping_fp8,
    std::optional<at::Tensor> slot_mapping_scale, std::optional<at::Tensor> gen_kv_indptr,
    std::optional<at::Tensor> gen_cached_token_indptr)
{
    constexpr int64_t kHidden = tk::kDraftSampleGlueHidden;
    TORCH_CHECK(tk::draftEhprojTableSupportsTpSize(static_cast<int>(group_size)),
        "glm_draft_ehproj_table: group_size must be 2 to ", tk::kDraftEhprojTableMaxRanks,
        " ranks with whole 8-column shares of the hidden size, got ", group_size);
    tk::DraftEhprojTableParams params;
    int64_t const rows = setDraftStepExchange(params.step, "glm_draft_ehproj_table", logits, exchange_buffer,
        exchange_flags, group_size, draft_tokens, draft_step, tk::draftEhprojTableSlotBytes(group_size));

    TORCH_CHECK(hidden_half.is_cuda() && hidden_half.scalar_type() == at::ScalarType::Float && hidden_half.dim() == 2
            && hidden_half.size(0) == rows && hidden_half.size(1) == kHidden / group_size && hidden_half.stride(1) == 1
            && hidden_half.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(hidden_half.data_ptr()) % 32 == 0,
        "glm_draft_ehproj_table: hidden_half must be a 32-byte aligned FP32 [rows, 6144 / group_size] view");
    CHECK_INPUT(token_table, at::ScalarType::BFloat16);
    TORCH_CHECK(token_table.dim() == 2 && token_table.size(0) == logits.size(1) && token_table.size(1) == kHidden
            && isAligned16(token_table),
        "glm_draft_ehproj_table: token_table must be [vocab_shard, 6144]");
    CHECK_INPUT(norm_weight, at::ScalarType::BFloat16);
    TORCH_CHECK(norm_weight.numel() == kHidden && isAligned16(norm_weight),
        "glm_draft_ehproj_table: norm_weight must be [6144]");
    for (at::Tensor const& out : {normed_out, residual_out})
    {
        TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::ScalarType::BFloat16 && out.dim() == 2
                && out.size(0) == rows && out.size(1) == kHidden && out.stride(1) == 1
                && out.stride(0) == normed_out.stride(0) && out.stride(0) % 8 == 0 && isAligned16(out),
            "glm_draft_ehproj_table: normed_out and residual_out must be 16-byte aligned BF16 [rows, 6144] views with "
            "one row stride");
    }
    params.hiddenHalf = hidden_half.data_ptr<float>();
    params.hiddenHalfStride = hidden_half.stride(0);
    params.tokenTable = reinterpret_cast<__nv_bfloat16 const*>(token_table.data_ptr());
    params.normWeight = reinterpret_cast<__nv_bfloat16 const*>(norm_weight.data_ptr());
    params.normEps = static_cast<float>(eps);
    params.normedOut = reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr());
    params.residualOut = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    params.outStride = normed_out.stride(0);

    setDraftStepState(params.step, "glm_draft_ehproj_table", rows, positions_in, positions_out, gather_ids, seq_lens,
        fill_seq_lens, kv_lens, kv_update, rewind_accepted, draft_len, block_offsets, dsa_layout, req_idx_per_token,
        slot_mapping_fp8, slot_mapping_scale, gen_kv_indptr, gen_cached_token_indptr);

    tk::invokeDraftEhprojTable(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_draft_sample_glue(Tensor logits, Tensor(a!) exchange_buffer, Tensor(b!) exchange_flags, int group_size, "
        "Tensor(c!) draft_tokens, int draft_step, Tensor? embed_table, Tensor? hidden, Tensor? enorm_weight, "
        "Tensor? hnorm_weight, float eps, int col_lo, int col_hi, Tensor(d!)? glue_out, Tensor? positions_in, "
        "Tensor(e!)? positions_out, Tensor? gather_ids, Tensor(f!)? seq_lens, bool fill_seq_lens, "
        "Tensor(g!)? kv_lens, int kv_update, Tensor? rewind_accepted, int draft_len, Tensor? block_offsets, "
        "int[] dsa_layout, Tensor(h!)? req_idx_per_token, Tensor(i!)? slot_mapping_fp8, "
        "Tensor(j!)? slot_mapping_scale, Tensor(k!)? gen_kv_indptr, Tensor(l!)? gen_cached_token_indptr, "
        "Tensor? accepted_tokens, Tensor? batch_indices, Tensor? num_accepted, Tensor(m!)? next_new_tokens) -> ()");
    m.def(
        "glm_draft_ehproj_table(Tensor logits, Tensor(a!) exchange_buffer, Tensor(b!) exchange_flags, int group_size, "
        "Tensor(c!) draft_tokens, int draft_step, Tensor hidden_half, Tensor token_table, Tensor norm_weight, "
        "float eps, Tensor(d!) normed_out, Tensor(e!) residual_out, Tensor? positions_in, Tensor(f!)? positions_out, "
        "Tensor? gather_ids, Tensor(g!)? seq_lens, bool fill_seq_lens, Tensor(h!)? kv_lens, int kv_update, "
        "Tensor? rewind_accepted, int draft_len, Tensor? block_offsets, int[] dsa_layout, "
        "Tensor(i!)? req_idx_per_token, Tensor(j!)? slot_mapping_fp8, Tensor(k!)? slot_mapping_scale, "
        "Tensor(l!)? gen_kv_indptr, Tensor(m!)? gen_cached_token_indptr) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_draft_sample_glue", &tensorrt_llm::torch_ext::glmDraftSampleGlue);
    m.impl("glm_draft_ehproj_table", &tensorrt_llm::torch_ext::glmDraftEhprojTable);
}
