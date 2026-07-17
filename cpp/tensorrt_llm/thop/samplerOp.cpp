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

// Torch custom ops for the PyTorch backend sampler (TorchSampler). Aggregates
// sampler-related ops here so follow-up work can add new sampler ops alongside
// existing ones, mirroring how specDecOp.cpp groups the speculative-decoding ops.

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/samplerKernels/toppDecayKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused post-sample runtime top-p update for the TorchSampler Top-P Decay
// feature. Applies, in place, for every sampled row whose slot is decay-active:
//   runtime_top_p[slot] = (last_token == reset_id)
//                           ? initial_top_p[slot]
//                           : max(runtime_top_p[slot] * top_p_decay[slot], top_p_min[slot])
// The decay-active gate is applied on-device via is_decay_slot, so the hot path
// needs no host-side .tolist() / set intersection.
void top_p_decay_update(th::Tensor runtime_top_p, th::Tensor initial_top_p, th::Tensor top_p_decay,
    th::Tensor top_p_min, th::Tensor reset_ids, th::Tensor is_decay_slot, th::Tensor step_tokens,
    th::Tensor sampled_slots)
{
    TORCH_CHECK(runtime_top_p.is_cuda() && initial_top_p.is_cuda() && top_p_decay.is_cuda() && top_p_min.is_cuda()
            && reset_ids.is_cuda() && is_decay_slot.is_cuda() && step_tokens.is_cuda() && sampled_slots.is_cuda(),
        "all top_p_decay_update tensors must be CUDA tensors");

    TORCH_CHECK(runtime_top_p.scalar_type() == th::kFloat32, "runtime_top_p must be float32");
    TORCH_CHECK(initial_top_p.scalar_type() == th::kFloat32, "initial_top_p must be float32");
    TORCH_CHECK(top_p_decay.scalar_type() == th::kFloat32, "top_p_decay must be float32");
    TORCH_CHECK(top_p_min.scalar_type() == th::kFloat32, "top_p_min must be float32");
    TORCH_CHECK(reset_ids.scalar_type() == th::kInt32, "reset_ids must be int32");
    TORCH_CHECK(is_decay_slot.scalar_type() == th::kBool, "is_decay_slot must be bool");
    TORCH_CHECK(step_tokens.scalar_type() == th::kInt32, "step_tokens must be int32");
    TORCH_CHECK(sampled_slots.scalar_type() == th::kInt64, "sampled_slots must be int64");

    // step_tokens is a slot-indexed 1-D view of the new-tokens buffer for a fixed
    // step/beam (new_tokens[step, :, beam]); it is strided, not contiguous. The
    // kernel gathers tokens through its element stride, avoiding a separate
    // gather + cast launch on the hot path.
    TORCH_CHECK(runtime_top_p.is_contiguous() && initial_top_p.is_contiguous() && top_p_decay.is_contiguous()
            && top_p_min.is_contiguous() && reset_ids.is_contiguous() && is_decay_slot.is_contiguous()
            && sampled_slots.is_contiguous(),
        "all top_p_decay_update tensors (except step_tokens) must be contiguous");
    TORCH_CHECK(step_tokens.dim() == 1, "step_tokens must be a 1-D (strided) view");

    auto const numSlots = runtime_top_p.size(0);
    TORCH_CHECK(initial_top_p.size(0) == numSlots && top_p_decay.size(0) == numSlots && top_p_min.size(0) == numSlots
            && reset_ids.size(0) == numSlots && is_decay_slot.size(0) == numSlots,
        "all per-slot tensors must have the same length (max_num_sequences)");
    TORCH_CHECK(step_tokens.size(0) >= numSlots, "step_tokens must cover all slots");

    auto const numSampled = sampled_slots.size(0);

    auto stream = at::cuda::getCurrentCUDAStream(runtime_top_p.get_device());
    tk::invokeToppDecayUpdate(runtime_top_p.data_ptr<float>(), initial_top_p.data_ptr<float>(),
        top_p_decay.data_ptr<float>(), top_p_min.data_ptr<float>(), reset_ids.data_ptr<int32_t>(),
        is_decay_slot.data_ptr<bool>(), step_tokens.data_ptr<int32_t>(), step_tokens.stride(0),
        sampled_slots.data_ptr<int64_t>(), static_cast<int32_t>(numSampled), stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused pre-sample per-row top-p gather for the Top-P Decay feature. Returns
//   row_top_p[i] = is_decay_slot[slots[i]] ? runtime_top_p[slots[i]] : static_top_p[i]
// replacing the index_select(runtime) + index_select(gate) + where(static) chain
// with a single launch.
th::Tensor top_p_decay_gather(
    th::Tensor runtime_top_p, th::Tensor is_decay_slot, th::Tensor static_top_p, th::Tensor slots)
{
    TORCH_CHECK(runtime_top_p.is_cuda() && is_decay_slot.is_cuda() && static_top_p.is_cuda() && slots.is_cuda(),
        "all top_p_decay_gather tensors must be CUDA tensors");
    TORCH_CHECK(runtime_top_p.scalar_type() == th::kFloat32, "runtime_top_p must be float32");
    TORCH_CHECK(is_decay_slot.scalar_type() == th::kBool, "is_decay_slot must be bool");
    TORCH_CHECK(static_top_p.scalar_type() == th::kFloat32, "static_top_p must be float32");
    TORCH_CHECK(slots.scalar_type() == th::kInt64, "slots must be int64");
    TORCH_CHECK(runtime_top_p.is_contiguous() && is_decay_slot.is_contiguous() && static_top_p.is_contiguous()
            && slots.is_contiguous(),
        "all top_p_decay_gather tensors must be contiguous");
    TORCH_CHECK(runtime_top_p.size(0) == is_decay_slot.size(0), "per-slot tensors must have the same length");
    auto const numRows = slots.size(0);
    TORCH_CHECK(static_top_p.size(0) == numRows, "static_top_p and slots must have the same length");
    // NB: every slots[i] must lie in [0, runtime_top_p.size(0)); the kernel indexes the per-slot
    // arrays with it unchecked (values live on device, so validating here would force a sync).
    // The caller (TorchSampler) guarantees this: slots come from seq_slots, which is bounded by
    // max_num_sequences -- the allocation size of the per-slot store tensors.

    auto row_top_p = th::empty_like(static_top_p);
    auto stream = at::cuda::getCurrentCUDAStream(runtime_top_p.get_device());
    tk::invokeToppDecayGather(row_top_p.data_ptr<float>(), runtime_top_p.data_ptr<float>(),
        is_decay_slot.data_ptr<bool>(), static_top_p.data_ptr<float>(), slots.data_ptr<int64_t>(),
        static_cast<int32_t>(numRows), stream);
    return row_top_p;
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "top_p_decay_update(Tensor(a!) runtime_top_p, Tensor initial_top_p, Tensor top_p_decay, Tensor top_p_min, "
        "Tensor reset_ids, Tensor is_decay_slot, Tensor step_tokens, Tensor sampled_slots) -> ()");
    m.def(
        "top_p_decay_gather(Tensor runtime_top_p, Tensor is_decay_slot, Tensor static_top_p, Tensor slots) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("top_p_decay_update", &tensorrt_llm::torch_ext::top_p_decay_update);
    m.impl("top_p_decay_gather", &tensorrt_llm::torch_ext::top_p_decay_gather);
}
