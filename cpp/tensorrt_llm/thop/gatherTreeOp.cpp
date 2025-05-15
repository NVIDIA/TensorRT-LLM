/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

// Must be similar to GptDecoder<T>::gatherTree
th::Tensor gatherTree(                                    // BS: batch_size, BM: beam_width, MSL: max_seq_length
    th::Tensor& sequence_lengths,                         // [BS*BM], int
    th::Tensor& output_ids,                               // [BS, BM, MSL],int
    th::Tensor& parent_ids,                               // [BS, BM, MSL], int
    th::Tensor& end_ids,                                  // [BS*BM], int
    th::Tensor& tiled_input_lengths,                      // [BS*BM], int
    th::optional<th::Tensor> cum_log_probs_opt,           // [BS, BM], float
    th::optional<th::Tensor> log_probs_opt,               // [BS, BM, MSL], float
    th::optional<th::Tensor> log_probs_tiled_opt,         // [MSL, BS, BM], float, transpose of output_log_probs_opt
    th::optional<th::Tensor> beam_hyps_output_ids_cba,    // [BS, BM*2, MSL], int
    th::optional<th::Tensor> beam_hyps_seq_len_cba,       // [BS, BM*2], int
    th::optional<th::Tensor> beam_hyps_cum_log_probs_cba, // [BS, BM*2], float
    th::optional<th::Tensor> beam_hyps_normed_scores_cba, // [BS, BM*2], float
    th::optional<th::Tensor> beam_hyps_log_probs_cba,     // [BS, BM*2, MSL], float
    th::optional<th::Tensor> beam_hyps_min_normed_scores, // [BS], float
    th::optional<th::Tensor> beam_hyps_num_beams,         // [BS], int
    th::optional<th::Tensor> beam_hyps_is_done,           // [BS], bool
    th::optional<th::Tensor> finished,                    // [BS, BM], uint8
    th::Tensor& length_penalty,                           // [BS], float
    int64_t const batch_size,                             //
    int64_t const beam_width,                             //
    int64_t const max_seq_len,                            //
    bool const use_beam_hyps                              //
)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    th::Tensor final_output_ids = torch::zeros(
        {batch_size, beam_width, max_seq_len}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    if (use_beam_hyps && beam_width > 1)
    {
        int32_t* final_output_ids_ptr = get_ptr<int32_t>(final_output_ids);
        tk::invokeInitializeOutput(
            final_output_ids_ptr, get_ptr<int32_t>(end_ids), batch_size, beam_width, max_seq_len, stream);

        tk::BeamHypotheses bh;
        bh.nBatchSize = batch_size;
        bh.nBeamWidth = beam_width;
        bh.nMaxSeqLen = max_seq_len;
        bh.lengthPenalties = get_ptr<float>(length_penalty);
        bh.inputLengths = get_ptr<int32_t>(tiled_input_lengths);
        bh.outputIds = final_output_ids_ptr;
        bh.logProbs = log_probs_opt.has_value() ? get_ptr<float>(log_probs_opt.value()) : nullptr;
        bh.logProbsTiled = log_probs_tiled_opt.has_value() ? get_ptr<float>(log_probs_tiled_opt.value()) : nullptr;
        bh.sequenceLengths = get_ptr<int32_t>(sequence_lengths);
        bh.cumLogProbs = cum_log_probs_opt.has_value() ? get_ptr<float>(cum_log_probs_opt.value()) : nullptr;
        bh.outputIdsCBA = get_ptr<int32_t>(beam_hyps_output_ids_cba.value());
        bh.logProbsCBA = get_ptr<float>(beam_hyps_log_probs_cba.value());
        bh.sequenceLengthsCBA = get_ptr<int32_t>(beam_hyps_seq_len_cba.value());
        bh.cumLogProbsCBA = get_ptr<float>(beam_hyps_cum_log_probs_cba.value());
        bh.normedScoresCBA = get_ptr<float>(beam_hyps_normed_scores_cba.value());
        bh.numBeamsCBA = get_ptr<int32_t>(beam_hyps_num_beams.value());
        bh.minNormedScoresCBA = get_ptr<float>(beam_hyps_min_normed_scores.value());
        bh.batchDones = get_ptr<bool>(beam_hyps_is_done.value());
        bh.finished
            = reinterpret_cast<tk::FinishedState*>(get_ptr<tk::FinishedState::UnderlyingType>(finished.value()));
        bh.outputIdsUnfinish = get_ptr<int32_t>(output_ids);
        bh.parentIdsUnfinish = get_ptr<int32_t>(parent_ids);

        tk::invokeInsertUnfinishedPath(bh, stream);
        sync_check_cuda_error(stream);

        tk::invokeFinalize(bh, stream);
        sync_check_cuda_error(stream);
    }
    else if (!use_beam_hyps && beam_width > 1)
    {
        th::Tensor workspace = torch::zeros(batch_size * beam_width * max_seq_len * sizeof(int32_t),
            torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        // For sampling, it is equivalent to all parent ids are 0.
        tk::gatherTreeParam param;
        param.beams = get_ptr<int32_t>(workspace);
        // Remove prompt length if possible
        param.sequenceLengths = get_ptr<int32_t>(sequence_lengths);
        // add sequence_length 1 here because the sequence_length of time step t is t - 1
        param.maxSequenceLengthFinalStep = 1;
        // response input lengths (used to slice the ids during postprocessing), used in interactive generation
        // This feature is not supported yet, setting it to nullptr temporarily.
        param.responseInputLengths = nullptr;
        param.maxSeqLen = max_seq_len;
        param.batchSize = batch_size;
        param.beamWidth = beam_width;
        param.stepIds = get_ptr<int32_t>(output_ids);
        param.parentIds = beam_width == 1 ? nullptr : get_ptr<int32_t>(parent_ids);
        param.endTokens = get_ptr<int32_t>(end_ids);
        param.inputLengths = get_ptr<int32_t>(tiled_input_lengths);

        param.stream = stream;
        param.outputIds = get_ptr<int32_t>(final_output_ids);
        param.cumLogProbs = cum_log_probs_opt.has_value() ? get_ptr<float>(cum_log_probs_opt.value()) : nullptr;
        param.lengthPenalty = get_val<float>(length_penalty, 0);

        // NOTE: need to remove all prompt virtual tokens
        tk::invokeGatherTree(param);
        sync_check_cuda_error(stream);
    }
    else
    {
        cudaMemcpyAsync(get_ptr<int32_t>(final_output_ids), get_ptr<int32_t>(output_ids),
            sizeof(int) * batch_size * beam_width * max_seq_len, cudaMemcpyDeviceToDevice, stream);
        sync_check_cuda_error(stream);
    }
    return final_output_ids;
}

} // namespace torch_ext

static auto gather_tree = torch::RegisterOperators("tensorrt_llm::gather_tree", &torch_ext::gatherTree);
