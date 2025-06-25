/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "buffers.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"

#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using tr::SizeType32;

namespace tensorrt_llm::pybind::batch_manager
{

void Buffers::initBindings(pybind11::module_& m)
{
    py::class_<tb::TransformerBuffers>(m, "TransformerBuffers")
        .def(py::init<SizeType32, SizeType32, std::vector<SizeType32> const&, SizeType32, SizeType32,
                 runtime::TllmRuntime const&, runtime::ModelConfig const&, runtime::WorldConfig const&>(),
            py::arg("max_batch_size"), py::arg("max_beam_width"), py::arg("max_attention_window_vec"),
            py::arg("max_attention_window"), py::arg("sink_token_len"), py::arg("runtime"), py::arg("model_config"),
            py::arg("world_config"))
        .def("reshape", &tb::TransformerBuffers::reshape, py::arg("num_sequences"), py::arg("num_input_tokens"))
        .def("reshape_kv_tensors", &tb::TransformerBuffers::reshapeKvTensors, py::arg("max_batch_size"),
            py::arg("max_beam_width"), py::arg("max_blocks_per_seq"), py::arg("kv_cache_type"), py::arg("num_pools"),
            py::arg("buffer_manager"))
        .def("get_buffers", &tb::TransformerBuffers::getBuffers, py::arg("input_buffers"), py::arg("output_buffers"),
            py::arg("model_config"))
        .def("copy_position_ids", &tb::TransformerBuffers::copyPositionIds, py::arg("runtime"),
            py::arg("position_ids_host"), py::arg("is_chat_glm"), py::arg("decoder_position_ids"))
        .def("copy_kv_block_offsets", &tb::TransformerBuffers::copyKvBlockOffsets, py::arg("context_requests"),
            py::arg("gen_requests"), py::arg("kv_cache_manager"), py::arg("cross_kv_cache_manager"),
            py::arg("buffer_manager"))
        .def("copy_cache_indirection", &tb::TransformerBuffers::copyCacheIndirection, py::arg("gen_requests"),
            py::arg("decoder_cache_indirection_output"), py::arg("runtime"))
        .def_readwrite("past_key_value_lengths", &tb::TransformerBuffers::pastKeyValueLengths)
        .def_readwrite("position_ids", &tb::TransformerBuffers::positionIds)
        .def_readwrite("max_attention_windows", &tb::TransformerBuffers::maxAttentionWindows)
        .def_readwrite("sink_token_lengths", &tb::TransformerBuffers::sinkTokenLengths)
        .def_readwrite("cache_indirection", &tb::TransformerBuffers::cacheIndirection)
        .def_readwrite("kv_cache_block_offsets_host", &tb::TransformerBuffers::kvCacheBlockOffsetsHost)
        .def_readwrite("kv_cache_block_offsets_device", &tb::TransformerBuffers::kvCacheBlockOffsetsDevice)
        .def_readwrite("cross_kv_cache_block_pool_pointers", &tb::TransformerBuffers::crossKvCacheBlockPoolPointers)
        .def_readwrite("cross_kv_cache_block_offsets_host", &tb::TransformerBuffers::crossKvCacheBlockOffsetsHost)
        .def_readwrite("cross_kv_cache_block_offsets_device", &tb::TransformerBuffers::crossKvCacheBlockOffsetsDevice)
        .def_readwrite("cache_indir_batched_copy_src_offsets", &tb::TransformerBuffers::cacheIndirBatchedCopySrcOffsets)
        .def_readwrite("cache_indir_batched_copy_dst_offsets", &tb::TransformerBuffers::cacheIndirBatchedCopyDstOffsets)
        .def_readwrite("cache_indir_batched_copy_sizes", &tb::TransformerBuffers::cacheIndirBatchedCopySizes)
        .def_readwrite("fill_values_alt", &tb::TransformerBuffers::fillValuesAlt)
        .def_readwrite("fill_values_alt_device", &tb::TransformerBuffers::fillValuesAltDevice)
        .def_readwrite("seq_slots_alt", &tb::TransformerBuffers::seqSlotsAlt)
        .def_readwrite("seq_slots_alt_device", &tb::TransformerBuffers::seqSlotsAltDevice);

    py::classh<tb::RuntimeBuffers>(m, "RuntimeBuffers")
        .def(py::init<SizeType32, SizeType32, std::vector<SizeType32> const&, SizeType32, SizeType32,
                 runtime::TllmRuntime const&, runtime::ModelConfig const&, runtime::WorldConfig const&,
                 executor::DecodingConfig const&, bool, std::optional<SizeType32>>(),
            py::arg("max_batch_size"), py::arg("max_beam_width"), py::arg("max_attention_window_vec"),
            py::arg("max_attention_window"), py::arg("sink_token_len"), py::arg("runtime"), py::arg("model_config"),
            py::arg("world_config"), py::arg("decoding_config"), py::arg("gather_generation_logits"),
            py::arg("max_num_tokens") = std::nullopt)
        .def_readwrite("transformer_buffers", &tb::RuntimeBuffers::transformerBuffers)
        .def_readwrite("num_context_logits", &tb::RuntimeBuffers::numContextLogits)
        .def_readwrite("cache_indir_decoder_io_batched_copy_src_offsets",
            &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopySrcOffsets)
        .def_readwrite("cache_indir_decoder_io_batched_copy_dst_offsets",
            &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopyDstOffsets)
        .def_readwrite(
            "cache_indir_decoder_io_batched_copy_sizes", &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopySizes)
        .def_readwrite("logits", &tb::RuntimeBuffers::logits)
        .def_readwrite("seq_slots", &tb::RuntimeBuffers::seqSlots)
        .def_readwrite("seq_slots_device", &tb::RuntimeBuffers::seqSlotsDevice)
        .def_readwrite("sorted_seq_slots", &tb::RuntimeBuffers::sortedSeqSlots)
        .def_readwrite("seq_slot_remapping_host", &tb::RuntimeBuffers::seqSlotRemappingHost)
        .def_readwrite("seq_slot_remapping_device", &tb::RuntimeBuffers::seqSlotRemappingDevice)
        .def_readwrite("cache_indir_decoder_io_batched_copy_src_offsets_slice_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice)
        .def_readwrite("cache_indir_decoder_io_batched_copy_dst_offsets_slice_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice)
        .def_readwrite("cache_indir_decoder_io_batched_copy_copy_sizes_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopyCopySizesDevice);
}
} // namespace tensorrt_llm::pybind::batch_manager
