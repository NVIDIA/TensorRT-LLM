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

#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;

namespace torch_ext
{

// this should be similar to gatherTree in cpp/tensorrt_llm/runtime/gptSession.cpp
th::Tensor gatherTree(th::Tensor& sequence_lengths, th::Tensor& output_ids, th::Tensor& parent_ids, th::Tensor& end_ids,
    th::Tensor& tiled_input_lengths, th::optional<th::Tensor> cum_log_probs_opt,
    th::optional<th::Tensor> beam_hyps_output_ids_tgt, th::optional<th::Tensor> beam_hyps_sequence_lengths_tgt,
    th::optional<th::Tensor> beam_hyps_cum_log_probs, th::optional<th::Tensor> beam_hyps_normed_scores,
    th::optional<th::Tensor> beam_hyps_log_probs, th::optional<th::Tensor> beam_hyps_min_normed_scores,
    th::optional<th::Tensor> beam_hyps_num_beams, th::optional<th::Tensor> beam_hyps_is_done,
    th::optional<th::Tensor> finished, th::Tensor& length_penalty, int64_t batch_size, int64_t beam_width,
    int64_t max_seq_len, bool use_beam_hyps)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    th::Tensor final_output_ids = torch::zeros(
        {batch_size, beam_width, max_seq_len}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    if (use_beam_hyps && beam_width > 1)
    {
        tl::kernels::invokeInitializeOutput(get_ptr<int32_t>(final_output_ids), get_ptr<int32_t>(end_ids),
            batch_size * beam_width, max_seq_len, stream);

        tl::kernels::BeamHypotheses beamHypotheses;
        beamHypotheses.sequence_lengths_src = get_ptr<int32_t>(sequence_lengths);
        beamHypotheses.parent_ids_src = get_ptr<int32_t>(parent_ids);
        beamHypotheses.output_ids_src = get_ptr<int32_t>(output_ids);
        beamHypotheses.log_probs_src = nullptr;
        beamHypotheses.max_seq_len = max_seq_len;
        beamHypotheses.length_penalties = get_ptr<float>(length_penalty);

        beamHypotheses.output_ids_tgt = get_ptr<int32_t>(beam_hyps_output_ids_tgt.value());
        beamHypotheses.sequence_lengths_tgt = get_ptr<int32_t>(beam_hyps_sequence_lengths_tgt.value());
        beamHypotheses.cum_log_probs = get_ptr<float>(beam_hyps_cum_log_probs.value());
        beamHypotheses.normed_scores = get_ptr<float>(beam_hyps_normed_scores.value());
        beamHypotheses.log_probs = get_ptr<float>(beam_hyps_log_probs.value());
        beamHypotheses.min_normed_scores = get_ptr<float>(beam_hyps_min_normed_scores.value());
        beamHypotheses.num_beams = get_ptr<int32_t>(beam_hyps_num_beams.value());
        beamHypotheses.is_done = get_ptr<bool>(beam_hyps_is_done.value());
        beamHypotheses.input_lengths = get_ptr<int32_t>(tiled_input_lengths);

        tl::kernels::invokeInsertUnfinishedPath(beamHypotheses,
            reinterpret_cast<tl::kernels::FinishedState*>(
                get_ptr<tl::kernels::FinishedState::UnderlyingType>(finished.value())),
            get_ptr<float>(cum_log_probs_opt.value()), batch_size, beam_width, stream);
        sync_check_cuda_error();

        tl::kernels::invokeFinalize(get_ptr<int32_t>(final_output_ids), get_ptr<int32_t>(sequence_lengths),
            cum_log_probs_opt.has_value() ? get_ptr<float>(cum_log_probs_opt.value()) : nullptr,
            nullptr, // output_logs
            beamHypotheses.output_ids_tgt, beamHypotheses.sequence_lengths_tgt, beamHypotheses.normed_scores,
            beamHypotheses.cum_log_probs, beamHypotheses.log_probs, beamHypotheses.num_beams,
            get_ptr<int32_t>(tiled_input_lengths), beam_width, max_seq_len, batch_size, stream);
        sync_check_cuda_error();
    }
    else if (!use_beam_hyps && beam_width > 1)
    {
        th::Tensor workspace = torch::zeros(batch_size * beam_width * max_seq_len * sizeof(int32_t),
            torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        // For sampling, it is equivalent to all parent ids are 0.
        tl::kernels::gatherTreeParam param;
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
        tl::kernels::invokeGatherTree(param);
        sync_check_cuda_error();
    }
    else
    {
        cudaMemcpyAsync(get_ptr<int32_t>(final_output_ids), get_ptr<int32_t>(output_ids),
            sizeof(int) * batch_size * beam_width * max_seq_len, cudaMemcpyDeviceToDevice, stream);
        sync_check_cuda_error();
    }
    return final_output_ids;
}

} // namespace torch_ext

static auto gather_tree = torch::RegisterOperators("tensorrt_llm::gather_tree", &torch_ext::gatherTree);
