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

#include "tensorrt_llm/kernels/triAttentionScoreKernels/triAttentionScoreKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::tri_attention_score;

// Two ops rather than one fused op, matching the file-level granularity of
// sibling kernel wrappers (one op per kernel launch): the coefficient fold
// runs once per eviction round into persistent buffers whose plane count
// depends on the aggregation mode, while the score launch consumes those
// buffers; callers time and re-plan them independently.

namespace
{

// One dtype-parameterized operand validator; dtypeName keeps the message
// spelling (fp32 / int32 / int64) for each expected scalar type.
void checkContiguousCuda(torch::Tensor const& tensor, at::ScalarType dtype, char const* dtypeName, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == dtype, name,
        " must be a contiguous ", dtypeName, " CUDA tensor");
}

// Validate the optional per-layer dequantization scales for quantized
// (fp8/int8) KV pools and return the device pointer (nullptr when absent).
// Scales are indexed by ABSOLUTE layer id, matching layer_base_addrs and the
// calibration tables, so the extent must cover every calibrated layer. The
// positivity check runs HOST-side (one small sync per eviction round on a
// functional-only path): the |K| coefficient fold assumes
// |scale * K_q| == scale * |K_q|, which silently corrupts scores for
// scale <= 0, so a loud host error here is required rather than a
// device-side assert.
float const* checkKvScales(
    std::optional<torch::Tensor> const& kv_scales, int64_t num_calibrated_layers, char const* op_name)
{
    if (!kv_scales.has_value())
    {
        return nullptr;
    }
    checkContiguousCuda(*kv_scales, at::kFloat, "fp32", "kv_scales");
    TORCH_CHECK(kv_scales->numel() >= num_calibrated_layers, op_name,
        ": kv_scales must carry one scale per calibrated layer (absolute layer id indexed)");
    TORCH_CHECK(kv_scales->min().item<float>() > 0.0f, op_name,
        ": kv_scales must be strictly positive (the |K| dequantization fold is only valid for positive scales)");
    return kv_scales->data_ptr<float>();
}

// Fold the per-round TriAttention score coefficients into c_re/c_im/c_mlr
// (fp32, [num_offsets?, num_requests, num_calibrated_layers, heads, freqs]).
// The mean aggregation consumes offset-collapsed mean_cos/mean_sin and writes
// one plane; the max aggregation consumes omega/offsets/round_starts and
// writes one c_re/c_im plane per offset. kv_scales (quantized pools only)
// folds the per-layer dequantization scale into every coefficient table; the
// paired score op then reads raw quantized elements. This op cannot see the
// pool dtype, so presence-iff-quantized is enforced by the score op.
void triAttentionFoldScoreCoefficientsOp(torch::Tensor c_re, torch::Tensor c_im, torch::Tensor c_mlr,
    torch::Tensor q_real, torch::Tensor q_imag, torch::Tensor mlr_coef, torch::Tensor freq_scale_sq,
    std::optional<torch::Tensor> mean_cos, std::optional<torch::Tensor> mean_sin, std::optional<torch::Tensor> omega,
    std::optional<torch::Tensor> offsets, std::optional<torch::Tensor> round_starts, int64_t num_requests,
    int64_t num_calibrated_layers, int64_t num_query_heads, int64_t num_freqs, int64_t num_offsets, bool use_max,
    std::optional<torch::Tensor> kv_scales)
{
    checkContiguousCuda(c_re, at::kFloat, "fp32", "c_re");
    checkContiguousCuda(c_im, at::kFloat, "fp32", "c_im");
    checkContiguousCuda(c_mlr, at::kFloat, "fp32", "c_mlr");
    checkContiguousCuda(q_real, at::kFloat, "fp32", "q_real");
    checkContiguousCuda(q_imag, at::kFloat, "fp32", "q_imag");
    checkContiguousCuda(mlr_coef, at::kFloat, "fp32", "mlr_coef");
    checkContiguousCuda(freq_scale_sq, at::kFloat, "fp32", "freq_scale_sq");
    TORCH_CHECK(num_requests > 0 && num_calibrated_layers > 0 && num_query_heads > 0 && num_freqs > 0,
        "tri_attention_fold_score_coefficients: fold extents must be positive");
    TORCH_CHECK(num_offsets >= 1 && num_offsets <= tk::kMaxScoreOffsets,
        "tri_attention_fold_score_coefficients: num_offsets must be in [1, ", tk::kMaxScoreOffsets, "], got ",
        num_offsets);
    int64_t const total = num_requests * num_calibrated_layers * num_query_heads * num_freqs;
    int64_t const planes = use_max ? num_offsets : 1;
    TORCH_CHECK(c_re.numel() >= planes * total && c_im.numel() >= planes * total && c_mlr.numel() >= total,
        "tri_attention_fold_score_coefficients: coefficient buffers are undersized");
    int64_t const calibration = num_calibrated_layers * num_query_heads * num_freqs;
    TORCH_CHECK(q_real.numel() >= calibration && q_imag.numel() >= calibration && mlr_coef.numel() >= calibration
            && freq_scale_sq.numel() >= num_freqs,
        "tri_attention_fold_score_coefficients: calibration tensors are undersized for the fold extent");

    float const* meanCosPtr = nullptr;
    float const* meanSinPtr = nullptr;
    float const* omegaPtr = nullptr;
    float const* offsetsPtr = nullptr;
    int32_t const* roundStartsPtr = nullptr;
    if (use_max)
    {
        TORCH_CHECK(omega.has_value() && offsets.has_value() && round_starts.has_value(),
            "tri_attention_fold_score_coefficients: max aggregation requires omega, offsets, and round_starts");
        checkContiguousCuda(*omega, at::kFloat, "fp32", "omega");
        checkContiguousCuda(*offsets, at::kFloat, "fp32", "offsets");
        checkContiguousCuda(*round_starts, at::kInt, "int32", "round_starts");
        TORCH_CHECK(
            omega->numel() >= num_freqs && offsets->numel() >= num_offsets && round_starts->numel() >= num_requests,
            "tri_attention_fold_score_coefficients: max-path inputs are undersized for the folded request count");
        omegaPtr = omega->data_ptr<float>();
        offsetsPtr = offsets->data_ptr<float>();
        roundStartsPtr = round_starts->data_ptr<int32_t>();
    }
    else
    {
        TORCH_CHECK(mean_cos.has_value() && mean_sin.has_value(),
            "tri_attention_fold_score_coefficients: mean aggregation requires mean_cos and mean_sin");
        checkContiguousCuda(*mean_cos, at::kFloat, "fp32", "mean_cos");
        checkContiguousCuda(*mean_sin, at::kFloat, "fp32", "mean_sin");
        TORCH_CHECK(mean_cos->numel() >= num_requests * num_freqs && mean_sin->numel() >= num_requests * num_freqs,
            "tri_attention_fold_score_coefficients: mean_cos/mean_sin are undersized (the fold iterates one row per "
            "folded request)");
        meanCosPtr = mean_cos->data_ptr<float>();
        meanSinPtr = mean_sin->data_ptr<float>();
    }
    float const* kvScalesPtr = checkKvScales(kv_scales, num_calibrated_layers, "tri_attention_fold_score_coefficients");

    auto stream = at::cuda::getCurrentCUDAStream();
    tk::foldScoreCoefficientsLaunch(q_real.data_ptr<float>(), q_imag.data_ptr<float>(), mlr_coef.data_ptr<float>(),
        freq_scale_sq.data_ptr<float>(), meanCosPtr, meanSinPtr, omegaPtr, offsetsPtr, roundStartsPtr, kvScalesPtr,
        c_re.data_ptr<float>(), c_im.data_ptr<float>(), c_mlr.data_ptr<float>(), static_cast<int32_t>(num_requests),
        static_cast<int32_t>(num_calibrated_layers), static_cast<int32_t>(num_query_heads),
        static_cast<int32_t>(num_freqs), static_cast<int32_t>(num_offsets), use_max, stream);
}

// Score every cached decode token of every (request, layer) segment against
// the folded coefficient tables, writing fp32 [segment, head, token] rows and
// each request's decode width. pool_anchor is one of the scored layer pools:
// the kernel reads all layers through layer_base_addrs (V2 exposes each layer
// as its own storage), and the anchor only supplies their common element type
// and the device; its data is never read through this argument.
void triAttentionPagedScoreOp(torch::Tensor pool_anchor, torch::Tensor layer_base_addrs, torch::Tensor block_offsets,
    torch::Tensor seg_page_offsets, torch::Tensor seg_request_ids, torch::Tensor seg_layer_ids,
    torch::Tensor request_seq_lens, torch::Tensor valid_widths, torch::Tensor request_token_starts, torch::Tensor c_re,
    torch::Tensor c_im, torch::Tensor c_mlr, torch::Tensor out, int64_t output_width, int64_t num_layers,
    int64_t num_requests, int64_t num_calibrated_layers, int64_t num_query_heads, int64_t num_kv_heads,
    int64_t num_freqs, int64_t tokens_per_block, int64_t kv_factor, int64_t num_offsets, int64_t stride_page,
    int64_t stride_kv_head, int64_t stride_slot, int64_t stride_dim, int64_t num_segments, bool use_max,
    bool use_vectorized, std::optional<torch::Tensor> kv_scales)
{
    TORCH_CHECK(use_max || num_offsets == 1,
        "tri_attention_paged_score: mean aggregation consumes exactly one folded coefficient plane");
    checkContiguousCuda(layer_base_addrs, at::kLong, "int64", "layer_base_addrs");
    checkContiguousCuda(block_offsets, at::kInt, "int32", "block_offsets");
    checkContiguousCuda(seg_page_offsets, at::kLong, "int64", "seg_page_offsets");
    checkContiguousCuda(seg_request_ids, at::kInt, "int32", "seg_request_ids");
    checkContiguousCuda(seg_layer_ids, at::kInt, "int32", "seg_layer_ids");
    checkContiguousCuda(request_seq_lens, at::kInt, "int32", "request_seq_lens");
    checkContiguousCuda(valid_widths, at::kInt, "int32", "valid_widths");
    checkContiguousCuda(request_token_starts, at::kInt, "int32", "request_token_starts");
    checkContiguousCuda(c_re, at::kFloat, "fp32", "c_re");
    checkContiguousCuda(c_im, at::kFloat, "fp32", "c_im");
    checkContiguousCuda(c_mlr, at::kFloat, "fp32", "c_mlr");
    checkContiguousCuda(out, at::kFloat, "fp32", "out");
    TORCH_CHECK(pool_anchor.is_cuda(), "tri_attention_paged_score: pool anchor must be a CUDA tensor");

    TORCH_CHECK(num_segments > 0 && num_segments <= 65535,
        "tri_attention_paged_score: request*layer segment count exceeds the CUDA grid limit");
    TORCH_CHECK(output_width > 0 && num_layers > 0 && num_requests > 0 && num_calibrated_layers > 0
            && tokens_per_block > 0 && kv_factor > 0 && num_freqs > 0,
        "tri_attention_paged_score: geometry extents must be positive");
    TORCH_CHECK(num_kv_heads > 0 && num_query_heads % num_kv_heads == 0,
        "tri_attention_paged_score: query heads must be divisible by KV heads");
    // kMaxScoreOffsets is the per-thread accumulator budget baked into the
    // kernels. It only constrains the "max" path (one coefficient plane per
    // offset); the "mean" path always folds every offset into one plane, so
    // the default geometric offset table (which is larger) still passes here.
    TORCH_CHECK(num_offsets >= 1 && num_offsets <= tk::kMaxScoreOffsets,
        "tri_attention_paged_score: num_offsets must be in [1, ", tk::kMaxScoreOffsets, "], got ", num_offsets);
    TORCH_CHECK(seg_page_offsets.numel() >= num_segments && seg_request_ids.numel() >= num_segments
            && seg_layer_ids.numel() >= num_segments,
        "tri_attention_paged_score: segment metadata is undersized");
    TORCH_CHECK(request_seq_lens.numel() >= num_requests && valid_widths.numel() >= num_requests
            && request_token_starts.numel() >= num_requests,
        "tri_attention_paged_score: per-request metadata is undersized");
    int64_t const total = num_requests * num_calibrated_layers * num_query_heads * num_freqs;
    TORCH_CHECK(c_re.numel() >= num_offsets * total && c_im.numel() >= num_offsets * total && c_mlr.numel() >= total,
        "tri_attention_paged_score: folded coefficient buffers are undersized");
    TORCH_CHECK(out.numel() >= num_segments * num_query_heads * output_width,
        "tri_attention_paged_score: score output buffer is undersized");

    auto const dtype = pool_anchor.scalar_type();
    auto poolType = tk::PoolElementType::kBFloat16;
    if (dtype == at::kBFloat16)
    {
        poolType = tk::PoolElementType::kBFloat16;
    }
    else if (dtype == at::kHalf)
    {
        poolType = tk::PoolElementType::kHalf;
    }
    else if (dtype == at::kFloat)
    {
        poolType = tk::PoolElementType::kFloat32;
    }
    else if (dtype == at::kFloat8_e4m3fn)
    {
        poolType = tk::PoolElementType::kFloat8E4M3;
    }
    else if (dtype == at::kChar)
    {
        poolType = tk::PoolElementType::kInt8;
    }
    else
    {
        TORCH_CHECK(false, "tri_attention_paged_score: unsupported KV pool dtype ", dtype,
            " (supported: bf16, fp16, fp32, fp8_e4m3fn, int8)");
    }
    // The score kernel never applies kv_scales itself (the fold op already
    // multiplied them into the coefficient tables), but this op is the only
    // one that sees the pool dtype, so it owns the presence contract:
    // quantized elements without scales would be scored as raw integers, and
    // scales alongside float pools would silently double-scale.
    bool const quantizedPool = poolType == tk::PoolElementType::kFloat8E4M3 || poolType == tk::PoolElementType::kInt8;
    TORCH_CHECK(!quantizedPool || kv_scales.has_value(),
        "tri_attention_paged_score: quantized (fp8/int8) KV pools require per-layer kv_scales");
    TORCH_CHECK(quantizedPool || !kv_scales.has_value(),
        "tri_attention_paged_score: kv_scales are only valid for quantized (fp8/int8) KV pools");
    checkKvScales(kv_scales, num_calibrated_layers, "tri_attention_paged_score");
    TORCH_CHECK(!use_vectorized || dtype == at::kBFloat16 || dtype == at::kHalf,
        "tri_attention_paged_score: the vectorized path requires bf16 or fp16 pools");
    TORCH_CHECK(!use_vectorized || (num_freqs % 8 == 0 && stride_dim == 1),
        "tri_attention_paged_score: the vectorized path requires num_freqs % 8 == 0 and a unit frequency stride");

    auto const groupSize = static_cast<int32_t>(num_query_heads / num_kv_heads);
    bool const vectorizedGroup = groupSize == 1 || groupSize == 2 || groupSize == 4 || groupSize == 8;
    // Other GQA group sizes run the vectorized math one query head per grid.z
    // block instead of one KV head (a runtime mapping, no extra template).
    bool const zIsQueryHead = use_vectorized && !vectorizedGroup;
    TORCH_CHECK((zIsQueryHead ? num_query_heads : num_kv_heads) <= 65535,
        "tri_attention_paged_score: head count exceeds the CUDA grid limit");

    tk::FoldedScoreParams params;
    params.layerBaseAddrs = layer_base_addrs.data_ptr<int64_t>();
    params.blockOffsets = block_offsets.data_ptr<int32_t>();
    params.segPageOffsets = seg_page_offsets.data_ptr<int64_t>();
    params.segRequestIds = seg_request_ids.data_ptr<int32_t>();
    params.segLayerIds = seg_layer_ids.data_ptr<int32_t>();
    params.requestSeqLens = request_seq_lens.data_ptr<int32_t>();
    params.validWidthOut = valid_widths.data_ptr<int32_t>();
    params.requestTokenStarts = request_token_starts.data_ptr<int32_t>();
    params.cRe = c_re.data_ptr<float>();
    params.cIm = c_im.data_ptr<float>();
    params.cMlr = c_mlr.data_ptr<float>();
    params.out = out.data_ptr<float>();
    params.outputWidth = static_cast<int32_t>(output_width);
    params.numLayers = static_cast<int32_t>(num_layers);
    params.numRequests = static_cast<int32_t>(num_requests);
    params.numCalibratedLayers = static_cast<int32_t>(num_calibrated_layers);
    params.numQueryHeads = static_cast<int32_t>(num_query_heads);
    params.numKvHeads = static_cast<int32_t>(num_kv_heads);
    params.numFreqs = static_cast<int32_t>(num_freqs);
    params.tokensPerBlock = static_cast<int32_t>(tokens_per_block);
    params.kvFactor = static_cast<int32_t>(kv_factor);
    params.numOffsets = static_cast<int32_t>(num_offsets);
    params.zIsQueryHead = zIsQueryHead;
    params.stridePage = stride_page;
    params.strideKvHead = stride_kv_head;
    params.strideSlot = stride_slot;
    params.strideDim = stride_dim;

    auto stream = at::cuda::getCurrentCUDAStream();
    tk::foldedScoreLaunch(
        params, poolType, groupSize, static_cast<int32_t>(num_segments), use_vectorized, use_max, stream);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "tri_attention_fold_score_coefficients("
        "Tensor(a!) c_re, Tensor(b!) c_im, Tensor(c!) c_mlr, "
        "Tensor q_real, Tensor q_imag, Tensor mlr_coef, Tensor freq_scale_sq, "
        "Tensor? mean_cos, Tensor? mean_sin, "
        "Tensor? omega, Tensor? offsets, Tensor? round_starts, "
        "int num_requests, int num_calibrated_layers, "
        "int num_query_heads, int num_freqs, "
        "int num_offsets, bool use_max, Tensor? kv_scales=None) -> ()");

    m.def(
        "tri_attention_paged_score("
        "Tensor pool_anchor, Tensor layer_base_addrs, Tensor block_offsets, "
        "Tensor seg_page_offsets, Tensor seg_request_ids, Tensor seg_layer_ids, "
        "Tensor request_seq_lens, Tensor(a!) valid_widths, Tensor request_token_starts, "
        "Tensor c_re, Tensor c_im, Tensor c_mlr, Tensor(b!) out, "
        "int output_width, int num_layers, int num_requests, int num_calibrated_layers, "
        "int num_query_heads, int num_kv_heads, int num_freqs, int tokens_per_block, "
        "int kv_factor, int num_offsets, int stride_page, int stride_kv_head, "
        "int stride_slot, int stride_dim, int num_segments, bool use_max, bool use_vectorized, "
        "Tensor? kv_scales=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("tri_attention_fold_score_coefficients", &triAttentionFoldScoreCoefficientsOp);
    m.impl("tri_attention_paged_score", &triAttentionPagedScoreOp);
}
