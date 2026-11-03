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

#include <gtest/gtest.h>

#include "tensorrt_llm/runtime/mcastGPUBuffer.h"

#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;

TEST(McastTensorOwnerTest, ViewsRetainAllocationOwner)
{
    auto backing = std::make_shared<std::vector<float>>(8);
    std::weak_ptr<std::vector<float>> const lifetime = backing;
    // Alias CPU backing storage to exercise owner lifetime without allocating multicast device memory.
    DeviceMemoryOwner owner(backing, nullptr);
    auto tensor = at::for_blob(backing->data(), {8})
                      .context(new DeviceMemoryOwner(owner), deleteMcastDevMemOwner)
                      .options(at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                      .make_tensor();
    auto view = tensor.slice(0, 2, 6).view({2, 2});
    EXPECT_NE(view.data_ptr(), tensor.data_ptr());
    auto recovered = getMcastDevMemBuffer(view);
    EXPECT_FALSE(recovered.owner_before(owner));
    EXPECT_FALSE(owner.owner_before(recovered));
    EXPECT_EQ(getMcastDevMemBuffer(view.clone()).use_count(), 0);

    tensor = at::Tensor{};
    owner.reset();
    backing.reset();
    EXPECT_FALSE(lifetime.expired());
    view = at::Tensor{};
    EXPECT_FALSE(lifetime.expired());
    recovered.reset();
    EXPECT_TRUE(lifetime.expired());
}

TEST(McastTensorOwnerTest, RejectsUnrelatedStorageContext)
{
    EXPECT_EQ(getMcastDevMemBuffer(at::Tensor{}).use_count(), 0);
    EXPECT_EQ(getMcastDevMemBuffer(at::zeros({1})).use_count(), 0);
    int context{0};
    float data{0};
    auto tensor = at::for_blob(&data, {1})
                      .context(&context, [](void*) {})
                      .options(at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                      .make_tensor();
    EXPECT_EQ(getMcastDevMemBuffer(tensor).use_count(), 0);
}

TEST(McastTensorOwnerTest, RejectsMissingOwnerContext)
{
    float data{0};
    auto tensor = at::for_blob(&data, {1})
                      .context(nullptr, deleteMcastDevMemOwner)
                      .options(at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                      .make_tensor();
    EXPECT_EQ(getMcastDevMemBuffer(tensor).use_count(), 0);
}
