/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gtest/gtest.h>

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;

TEST(UserBuffer, basic)
{
    if (!tr::ub::ub_supported())
    {
        GTEST_SKIP() << "UserBuffer is not supported";
    }
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    tr::ub::ub_initialize(world_size);
    EXPECT_EQ(tr::ub::ub_is_initialized(), true);
    EXPECT_NE(tr::ub::ub_comm(), nullptr);
    void* p0 = tr::ub::ub_allocate(1024).addr;
    void* p1 = tr::ub::ub_allocate(1024).addr;
    EXPECT_NE(p0, nullptr);
    EXPECT_NE(p1, nullptr);
    EXPECT_EQ(tr::ub::ub_get(0).invalid(), false);
    EXPECT_EQ(tr::ub::ub_get(1).invalid(), false);
    tr::ub::ub_deallocate(p0);
    tr::ub::ub_deallocate(p1);
}
