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

#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/thop/attentionOp.h"

#include <gtest/gtest.h>

namespace tcop = tensorrt_llm::common::op;
namespace tt = tensorrt_llm::torch_ext;

namespace
{

constexpr int32_t kBatchSize = 2;
constexpr int32_t kInputSequenceLength = 11;
constexpr int32_t kCrossKvLength = 7;
constexpr int32_t kPackedTokenCount = 14;
constexpr int32_t kHeadSize = 64;

tt::StaticAttentionConfig unfusedAttentionConfig(bool crossAttention)
{
    tt::StaticAttentionConfig cfg{};
    cfg.num_heads = 1;
    cfg.num_kv_heads = 1;
    cfg.head_size = kHeadSize;
    cfg.type = tensorrt_llm::DataType::kHALF;
    cfg.tokens_per_block = 64;
    cfg.q_scaling = 1.0;
    cfg.remove_padding = true;
    cfg.cross_attention = crossAttention;
    cfg.position_embedding_type = tt::AttentionOp::PositionEmbeddingType::kRELATIVE;
    return cfg;
}

size_t expectedUnfusedContextWorkspace(bool crossAttention)
{
    constexpr size_t kElementSize = sizeof(half);
    size_t const batchSize = kBatchSize;
    size_t const inputSequenceLength = kInputSequenceLength;
    size_t const kvSequenceLength = crossAttention ? kCrossKvLength : kInputSequenceLength;
    size_t const paddedTokenCount = batchSize * inputSequenceLength;
    size_t const paddedKvTokenCount = batchSize * kvSequenceLength;

    tcop::AttentionContextWorkspaceSizes sizes{};
    sizes.attentionMask = kElementSize * paddedTokenCount * kvSequenceLength;
    sizes.cuQSeqlens = sizeof(int) * (batchSize + 1);
    sizes.cuKvSeqlens = sizes.cuQSeqlens;
    sizes.cuMaskRows = sizes.cuQSeqlens;
    sizes.qBuf = kElementSize * paddedTokenCount * kHeadSize;
    sizes.kBuf = kElementSize * paddedKvTokenCount * kHeadSize;
    sizes.vBuf = sizes.kBuf;
    sizes.qkBuf = kElementSize * batchSize * inputSequenceLength * kvSequenceLength;
    sizes.qkvBuf = kElementSize * paddedTokenCount * kHeadSize;
    sizes.qkFloatBuf = sizeof(float) * batchSize * inputSequenceLength * kvSequenceLength;
    sizes.paddingOffset = sizeof(int) * paddedTokenCount;
    sizes.encoderPaddingOffset = sizeof(int) * paddedKvTokenCount;
    sizes.tokensInfo = sizeof(int2) * kPackedTokenCount;
    return tcop::AttentionWorkspaceManager::buildContextLayout(sizes).totalSize;
}

size_t getUnfusedContextWorkspace(tt::AttentionOp const& op, bool crossAttention)
{
    tt::FmhaParams params{};
    params.qkv_or_q = at::empty({0}, at::kHalf);
    params.is_cross = crossAttention;
    return op.getWorkspaceSizeForContext(params, kBatchSize, kInputSequenceLength, kCrossKvLength, kPackedTokenCount);
}

} // namespace

TEST(AttentionOpWorkspaceTest, RaggedUnfusedSelfAttentionUsesPaddedTokenCounts)
{
    tt::AttentionOp op(unfusedAttentionConfig(false));

    EXPECT_EQ(getUnfusedContextWorkspace(op, false), expectedUnfusedContextWorkspace(false));
}

TEST(AttentionOpWorkspaceTest, RaggedUnfusedCrossAttentionUsesPaddedTokenCounts)
{
    tt::AttentionOp op(unfusedAttentionConfig(true));

    EXPECT_EQ(getUnfusedContextWorkspace(op, true), expectedUnfusedContextWorkspace(true));
}

class AttentionOpSpecDecodingTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>>
{
};

TEST_P(AttentionOpSpecDecodingTest, OnlyActiveGenerationExposesSpeculativeInputs)
{
    auto const [isGen, enabled, active] = GetParam();
    auto cfg = unfusedAttentionConfig(false);
    cfg.num_heads = 8;
    cfg.head_size = 128;
    cfg.use_kv_cache = true;
    cfg.position_embedding_type = tt::AttentionOp::PositionEmbeddingType::kLEARNED_ABSOLUTE;
    cfg.mask_type = tt::AttentionOp::AttentionMaskType::CAUSAL;
    cfg.is_spec_decoding_enabled = enabled;
    cfg.spec_decoding_target_max_gen_len = 4;
    tt::AttentionOp op(cfg);

    bool const useSpecDecoding = isGen && enabled && active;
    tt::FmhaParams params{};
    params.num_seqs = params.num_requests = params.max_num_requests = params.max_num_sequences = 1;
    params.num_tokens = useSpecDecoding ? 4 : 1;
    params.beam_width = 1;
    params.max_context_length = params.max_seq_len = 64;
    params.max_attention_window_size = params.cyclic_attention_window_size = 64;
    params.qkv_or_q = at::empty({params.num_tokens, 1280}, at::kHalf);
    params.output = at::empty({params.num_tokens, 1024}, at::kHalf);
    params.workspace = at::empty({0}, at::kByte);
    params.multi_ctas_kv_counter = at::zeros({64}, at::kByte);
    params.host_past_key_value_lengths = at::full({1}, 4, at::kInt);
    params.host_context_lengths = at::full({1}, 1, at::kInt);
    params.sequence_length = params.host_past_key_value_lengths;
    params.context_lengths = params.host_context_lengths;
    params.fwd.is_fused_qkv = true;
    params.use_spec_decoding = active;
    params.spec_decoding_target_max_draft_tokens = 3;
    params.spec_decoding_generation_lengths = at::full({1}, 4, at::kInt);
    params.spec_decoding_position_offsets = at::zeros({1, 4}, at::kInt);
    params.spec_decoding_packed_mask = at::zeros({1, 4, 1}, at::kInt);
    params.spec_decoding_bl_tree_mask_offset = at::zeros({1}, at::kLong);
    params.spec_decoding_bl_tree_mask = at::zeros({1}, at::kUInt32);
    params.spec_bl_tree_first_sparse_mask_offset_kv = at::zeros({1}, at::kInt);
    params.spec_decoding_is_generation_length_variable = true;
    params.spec_decoding_max_generation_length = 4;

    auto const source = params;
    ASSERT_EQ(op.prepare(params, isGen), 0);
    EXPECT_EQ(params.spec_decoding_is_generation_length_variable, useSpecDecoding);
    EXPECT_EQ(params.spec_decoding_max_generation_length, useSpecDecoding ? 4 : 1);
    EXPECT_EQ(params.spec_decoding_target_max_gen_len, 4);
    EXPECT_EQ(params.getSpecDecodingGenerationLengths(),
        useSpecDecoding ? source.getSpecDecodingGenerationLengths() : nullptr);
    EXPECT_EQ(
        params.getSpecDecodingPositionOffsets(), useSpecDecoding ? source.getSpecDecodingPositionOffsets() : nullptr);
    EXPECT_EQ(params.getSpecDecodingPackedMask(), useSpecDecoding ? source.getSpecDecodingPackedMask() : nullptr);
    EXPECT_EQ(
        params.getSpecDecodingBlTreeMaskOffset(), useSpecDecoding ? source.getSpecDecodingBlTreeMaskOffset() : nullptr);
    EXPECT_EQ(params.getSpecDecodingBlTreeMask(), useSpecDecoding ? source.getSpecDecodingBlTreeMask() : nullptr);
    EXPECT_EQ(params.getSpecBlTreeFirstSparseMaskOffsetKv(),
        useSpecDecoding ? source.getSpecBlTreeFirstSparseMaskOffsetKv() : nullptr);
    EXPECT_TRUE(source.spec_decoding_generation_lengths.has_value());
}

INSTANTIATE_TEST_SUITE_P(PhaseAndRuntimeFlags, AttentionOpSpecDecodingTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()));
