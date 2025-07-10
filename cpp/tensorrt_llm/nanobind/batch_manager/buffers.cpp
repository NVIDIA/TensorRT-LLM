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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"

#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

namespace nb = nanobind;
namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using tr::SizeType32;

namespace tensorrt_llm::nanobind::batch_manager
{

void Buffers::initBindings(nb::module_& m)
{
    nb::class_<tb::TransformerBuffers>(m, "TransformerBuffers")
        .def(nb::init<SizeType32, SizeType32, std::vector<SizeType32> const&, SizeType32, SizeType32,
                 runtime::TllmRuntime const&, runtime::ModelConfig const&, runtime::WorldConfig const&>(),
            nb::arg("max_batch_size"), nb::arg("max_beam_width"), nb::arg("max_attention_window_vec"),
            nb::arg("max_attention_window"), nb::arg("sink_token_len"), nb::arg("runtime"), nb::arg("model_config"),
            nb::arg("world_config"))
        .def("reshape", &tb::TransformerBuffers::reshape, nb::arg("num_sequences"), nb::arg("num_input_tokens"))
        .def("reshape_kv_tensors", &tb::TransformerBuffers::reshapeKvTensors, nb::arg("max_batch_size"),
            nb::arg("max_beam_width"), nb::arg("max_blocks_per_seq"), nb::arg("kv_cache_type"), nb::arg("num_pools"),
            nb::arg("buffer_manager"))
        .def("get_buffers", &tb::TransformerBuffers::getBuffers, nb::arg("input_buffers"), nb::arg("output_buffers"),
            nb::arg("model_config"))
        .def("copy_position_ids", &tb::TransformerBuffers::copyPositionIds, nb::arg("runtime"),
            nb::arg("position_ids_host"), nb::arg("is_chat_glm"), nb::arg("decoder_position_ids"))
        .def("copy_kv_block_offsets", &tb::TransformerBuffers::copyKvBlockOffsets, nb::arg("context_requests"),
            nb::arg("gen_requests"), nb::arg("kv_cache_manager"), nb::arg("cross_kv_cache_manager"),
            nb::arg("buffer_manager"))
        .def("copy_cache_indirection", &tb::TransformerBuffers::copyCacheIndirection, nb::arg("gen_requests"),
            nb::arg("decoder_cache_indirection_output"), nb::arg("runtime"))
        .def_rw("past_key_value_lengths", &tb::TransformerBuffers::pastKeyValueLengths)
        .def_rw("position_ids", &tb::TransformerBuffers::positionIds)
        .def_rw("max_attention_windows", &tb::TransformerBuffers::maxAttentionWindows)
        .def_rw("sink_token_lengths", &tb::TransformerBuffers::sinkTokenLengths)
        .def_rw("cache_indirection", &tb::TransformerBuffers::cacheIndirection)
        .def_rw("kv_cache_block_offsets_host", &tb::TransformerBuffers::kvCacheBlockOffsetsHost)
        .def_rw("kv_cache_block_offsets_device", &tb::TransformerBuffers::kvCacheBlockOffsetsDevice)
        .def_rw("cross_kv_cache_block_pool_pointers", &tb::TransformerBuffers::crossKvCacheBlockPoolPointers)
        .def_rw("cross_kv_cache_block_offsets_host", &tb::TransformerBuffers::crossKvCacheBlockOffsetsHost)
        .def_rw("cross_kv_cache_block_offsets_device", &tb::TransformerBuffers::crossKvCacheBlockOffsetsDevice)
        .def_rw("cache_indir_batched_copy_src_offsets", &tb::TransformerBuffers::cacheIndirBatchedCopySrcOffsets)
        .def_rw("cache_indir_batched_copy_dst_offsets", &tb::TransformerBuffers::cacheIndirBatchedCopyDstOffsets)
        .def_rw("cache_indir_batched_copy_sizes", &tb::TransformerBuffers::cacheIndirBatchedCopySizes)
        .def_rw("fill_values_alt", &tb::TransformerBuffers::fillValuesAlt)
        .def_rw("fill_values_alt_device", &tb::TransformerBuffers::fillValuesAltDevice)
        .def_rw("seq_slots_alt", &tb::TransformerBuffers::seqSlotsAlt)
        .def_rw("seq_slots_alt_device", &tb::TransformerBuffers::seqSlotsAltDevice);

    nb::class_<tb::RuntimeBuffers>(m, "RuntimeBuffers")
        .def(nb::init<SizeType32, SizeType32, std::vector<SizeType32> const&, SizeType32, SizeType32,
                 runtime::TllmRuntime const&, runtime::ModelConfig const&, runtime::WorldConfig const&,
                 executor::DecodingConfig const&, bool, std::optional<SizeType32>>(),
            nb::arg("max_batch_size"), nb::arg("max_beam_width"), nb::arg("max_attention_window_vec"),
            nb::arg("max_attention_window"), nb::arg("sink_token_len"), nb::arg("runtime"), nb::arg("model_config"),
            nb::arg("world_config"), nb::arg("decoding_config"), nb::arg("gather_generation_logits"),
            nb::arg("max_num_tokens") = std::nullopt)
        // todo: test this
        .def_prop_rw(
            "transformer_buffers", [](tb::RuntimeBuffers& self) { return self.transformerBuffers.get(); },
            [](tb::RuntimeBuffers& self, tb::TransformerBuffers* val) { self.transformerBuffers.reset(val); })
        .def_rw("num_context_logits", &tb::RuntimeBuffers::numContextLogits)
        .def_rw("cache_indir_decoder_io_batched_copy_src_offsets",
            &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopySrcOffsets)
        .def_rw("cache_indir_decoder_io_batched_copy_dst_offsets",
            &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopyDstOffsets)
        .def_rw("cache_indir_decoder_io_batched_copy_sizes", &tb::RuntimeBuffers::cacheIndirDecoderIOBatchedCopySizes)
        .def_rw("logits", &tb::RuntimeBuffers::logits)
        .def_rw("seq_slots", &tb::RuntimeBuffers::seqSlots)
        .def_rw("seq_slots_device", &tb::RuntimeBuffers::seqSlotsDevice)
        .def_rw("sorted_seq_slots", &tb::RuntimeBuffers::sortedSeqSlots)
        .def_rw("seq_slot_remapping_host", &tb::RuntimeBuffers::seqSlotRemappingHost)
        .def_rw("seq_slot_remapping_device", &tb::RuntimeBuffers::seqSlotRemappingDevice)
        .def_rw("cache_indir_decoder_io_batched_copy_src_offsets_slice_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice)
        .def_rw("cache_indir_decoder_io_batched_copy_dst_offsets_slice_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice)
        .def_rw("cache_indir_decoder_io_batched_copy_copy_sizes_device",
            &tb::RuntimeBuffers::mCacheIndirDecoderIOBatchedCopyCopySizesDevice);
}
} // namespace tensorrt_llm::nanobind::batch_manager
