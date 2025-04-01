/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm
{

namespace kernels
{

struct gatherTreeParam
{
    // TODO rename the parameters
    int32_t* beams = nullptr;              // [batchSize, beamWidth, maxSeqLen], workspace to put intermediate outputIds
    int32_t* sequenceLengths = nullptr;    // [batchSize, beamWidth], total lengths of each query
    int32_t maxSequenceLengthFinalStep = 0;
    int32_t const* inputLengths = nullptr; // [batchSize, beamWidth]
    // response input lengths (used to slice the ids during postprocessing)
    int32_t* responseInputLengths = nullptr;
    int32_t maxSeqLen = 0;
    int32_t batchSize = 0;
    int32_t beamWidth = 0;
    int32_t const* stepIds = nullptr;   // [maxSeqLen, batchSize, beamWidth]
    int32_t const* parentIds = nullptr; // [maxSeqLen, batchSize, beamWidth]
    int32_t const* endTokens = nullptr; // [batchSize], end token ids of each query
    int32_t* outputIds = nullptr;       // the buffer to put finalized ids
    cudaStream_t stream;
    float* cumLogProbs = nullptr;       // [batchSize, beamWidth]
    float lengthPenalty = 1.0f;
    int earlyStopping = 1;
};

/*
Do gatherTree on beam search to get final result.
*/
void invokeGatherTree(gatherTreeParam param);

void invokeInsertUnfinishedPath(BeamHypotheses& bh, cudaStream_t stream);

void invokeFinalize(BeamHypotheses& bh, cudaStream_t stream);

//! \brief invoke the kernel that Initializes the output tensor by prefilling it with end tokens.
//!
//! \param finalOutputIds The output tensor to be initialized.
//! \param endIds The tensor containing the end IDs.
//! \param batchBeam batchSize*beamWidth. inferred from finalOutputIds.shape[0] * finalOutputIds.shape[1]
//! \param maxSeqLen The maximum sequence length, inferred from the finalOutputIds.shape[3]
//! \param stream The CUDA stream on which to perform the operation.
void invokeInitializeOutput(runtime::TokenIdType* finalOutputIds, runtime::TokenIdType const* endIds,
    runtime::SizeType32 batch, runtime::SizeType32 beam, runtime::SizeType32 maxSeqLen, cudaStream_t stream);

//! \brief Copies the data from the buffers in src to dst to reduce the kernel launch overhead of individual memcpy.
//! for streaming + beam search, where we need to avoid overwriting the beam search buffers.
//!
//! \param src the source, usually the buffers in which the beam search kernels write
//! \param dst temp buffers for use in the subsequent gatherTree kernels.
//! \param srcCumLogProbs source of the cumLogProbs. Separate since it's not included in beamHypotheses.
//! \param dstCumLogProbs dst of srcCumLogProbs.
//! \param stream CUDA stream to execute the kernel
//! \param numSMs number of SMs available on the device
void invokeCopyBeamHypotheses(runtime::DecodingOutput::BeamHypotheses const& src,
    runtime::DecodingOutput::BeamHypotheses const& dst, runtime::ITensor& srcCumLogProbs,
    runtime::ITensor& dstCumLogProbs, runtime::CudaStream const& stream, int numSMs);

//! \brief Copies last numNewTokens (or 1 if numNewTokens == nullptr) tokens from outputIdsPtr
//! to nextStepIds according to sequenceLengths.
//!
//! \param nextStepIds output buffer [maxTokensPerStep, maxBatchSize, maxBeamWidth],
//! destination of the new tokens.
//! \param outputIdsPtr input buffer [maxBatchSize][maxBeamWidth, maxSeqLen],
//! array of pointers to the source of the copy.
//! \param sequenceLengths input buffer [maxBatchSize], sequence length of the request
//! in outputIdsPtr that includes all new tokens. It must be guaranteed that sequenceLengths <= maxSeqLen.
//! \param numNewTokens input buffer [maxBatchSize], optional, number of tokens to be copied.
//! If nullptr, only 1 token is copied. It must be guaranteed that numNewTokens <= sequenceLengths.
//! \param batchSlots input buffer [batchSize], address map from local index
//! to global index [0, batchSize] -> [0, maxBatchSize]
//! \param batchSize current batch size
//! \param maxBatchSize maximum batch size
//! \param beamWidth current beam width
//! \param maxSeqLen maximum sequence length
//! \param maxTokensPerStep maximum tokens per step
//! \param stream stream
void invokeCopyNextStepIds(runtime::TokenIdType* nextStepIds, runtime::TokenIdType const* const* outputIdsPtr,
    runtime::SizeType32 const* sequenceLengths, runtime::SizeType32 const* numNewTokens,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
    runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxTokensPerStep,
    cudaStream_t stream);

void invokeTransposeLogProbs(float* output_log_probs, float* output_log_probs_tiled,
    runtime::SizeType32 const* sequence_lengths, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batch_size,
    runtime::SizeType32 max_batch_size, runtime::SizeType32 beam_width, runtime::SizeType32 max_seq_len,
    cudaStream_t stream);

} // namespace kernels

namespace runtime::kernels
{
//! \brief Inserts the running beams into the finished beams stored in the CBA buffers. (beams where the most likely
//! continuation is the end token get stored separately, and another candidate next token is stored). Then sorts the
//! beams according to their cumulative log probs. Note: the kernels in gatherTree modify the buffers inplace. When
//! streaming, we use tmp buffers since beam search kernels expect ungathered data.
//!
//! \param decodingOutput contains a slice of the output buffers to gather. Also contains the
//! DecodingOutput::BeamHypotheses object with the finished beams.
//! \param decodingInput used for endIds and input lengths.
//! \param samplingConfig the usual buffer samplingConfig.
//! \param cudaStream the CUDA stream on which to perform the operation.

void gatherTree(DecodingOutput const& decodingOutput, DecodingInput const& decodingInput,
    SamplingConfig const& samplingConfig, runtime::CudaStream const& cudaStream);
} // namespace runtime::kernels

} // namespace tensorrt_llm
