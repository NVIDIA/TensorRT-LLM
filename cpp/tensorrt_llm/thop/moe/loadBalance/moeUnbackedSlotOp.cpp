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

#include "tensorrt_llm/runtime/moeLoadBalancer/unbackedSlotMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>
#include <c10/util/intrusive_ptr.h>

#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Return whether leaving unbackedSlots unbacked saves device memory on the current device.
bool moeUnbackedSlotsSaveMemory(int64_t slotBytes, int64_t slotCount, at::IntArrayRef unbackedSlots)
{
    TORCH_CHECK(slotBytes > 0 && slotCount > 0, "expects slots of at least one byte");
    return tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().savesMemory(static_cast<size_t>(slotBytes),
        static_cast<int>(slotCount), std::vector<int>(unbackedSlots.begin(), unbackedSlots.end()));
}

// Create a tensor of slots whose unbackedSlots map a shared scratch. Releasing its storage synchronizes the device,
// so the tensor must not be freed during CUDA graph capture.
at::Tensor moeUnbackedSlotTensor(
    at::IntArrayRef sizes, at::ScalarType dtype, at::IntArrayRef unbackedSlots, bool hostAccessible)
{
    TORCH_CHECK(!sizes.empty() && sizes[0] > 0, "expects the sizes of a tensor of slots");
    size_t const byteSize = c10::multiply_integers(sizes) * c10::elementSize(dtype);
    TORCH_CHECK(byteSize > 0, "expects slots of at least one byte");
    int const slotCount = static_cast<int>(sizes[0]);
    at::Device const device(at::kCUDA, c10::cuda::current_device());
    void* devPtr = tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().allocate(
        byteSize / slotCount, slotCount, std::vector<int>(unbackedSlots.begin(), unbackedSlots.end()), hostAccessible);
    c10::DataPtr dp(
        devPtr, devPtr, [](void* ptr) { tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().free(ptr); }, device);
    auto storageImpl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), byteSize,
        std::move(dp), c10::GetAllocator(device.type()), /*resizable=*/false);
    return at::empty({0}, at::TensorOptions().dtype(dtype).device(device))
        .set_(at::Storage(storageImpl), /*storage_offset=*/0, sizes);
}

// Map device memory of their own at the slots' addresses, or the shared scratch again, and return the bytes changed.
// A changed mapping synchronizes the device, so no kernel may use the slots and no CUDA graph capture may run.
// slots concatenates the slot ids of every tensor, and slotsPerTensor[i] is how many of them belong to tensors[i].
int64_t moeUnbackedSlotsSetBacked(
    at::TensorList tensors, at::IntArrayRef slots, at::IntArrayRef slotsPerTensor, bool backed)
{
    TORCH_CHECK(
        !tensors.empty() && tensors.size() == slotsPerTensor.size(), "expects a slot count per tensor of slots");
    std::vector<tensorrt_llm::runtime::UnbackedSlotMemory::SlotRange> ranges;
    ranges.reserve(tensors.size());
    auto next = slots.begin();
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        TORCH_CHECK(tensors[i].device() == tensors.front().device() && tensors[i].device().is_cuda(),
            "expects CUDA tensors of slots on one device");
        TORCH_CHECK(slotsPerTensor[i] >= 0 && slotsPerTensor[i] <= slots.end() - next,
            "expects slotsPerTensor to sum to the slots");
        ranges.push_back({tensors[i].storage().data_ptr().get(), std::vector<int>(next, next + slotsPerTensor[i])});
        next += slotsPerTensor[i];
    }
    TORCH_CHECK(next == slots.end(), "expects slotsPerTensor to sum to the slots");
    c10::cuda::CUDAGuard const deviceGuard(tensors.front().device());
    return static_cast<int64_t>(tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().setBacked(ranges, backed));
}

// Return the device bytes that backing the slots would map now.
int64_t moeUnbackedSlotsBytesToBack(at::Tensor const& tensor, at::IntArrayRef slots)
{
    TORCH_CHECK(tensor.device().is_cuda(), "expects a CUDA tensor of slots");
    return static_cast<int64_t>(tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().bytesToBack(
        tensor.storage().data_ptr().get(), std::vector<int>(slots.begin(), slots.end())));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_unbacked_slots_save_memory(int slot_bytes, int slot_count, int[] unbacked_slots) -> bool");
    m.def(
        "moe_unbacked_slot_tensor(int[] sizes, ScalarType dtype, int[] unbacked_slots, bool host_accessible=False) -> "
        "Tensor");
    m.def(
        "moe_unbacked_slots_set_backed(Tensor(a!)[] tensors, int[] slots, int[] slots_per_tensor, bool backed) -> int");
    m.def("moe_unbacked_slots_bytes_to_back(Tensor tensor, int[] slots) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_unbacked_slots_set_backed", &tensorrt_llm::torch_ext::moeUnbackedSlotsSetBacked);
    m.impl("moe_unbacked_slots_bytes_to_back", &tensorrt_llm::torch_ext::moeUnbackedSlotsBytesToBack);
}

// No tensor argument picks the backend: the slot tensor is created on the current CUDA device.
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_unbacked_slots_save_memory", &tensorrt_llm::torch_ext::moeUnbackedSlotsSaveMemory);
    m.impl("moe_unbacked_slot_tensor", &tensorrt_llm::torch_ext::moeUnbackedSlotTensor);
}
