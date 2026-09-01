# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_gtest(radixTreeTest radixTreeTest.cpp)
add_gtest(blockKeyTest blockKeyTest.cpp)
add_gtest(radixBlockTreeTest radixBlockTreeTest.cpp)
add_gtest(cacheTransBufferTest cacheTransBufferTest.cpp)
add_gtest(bufferIndexHolderTest bufferIndexHolderTest.cpp)
add_gtest(capacitySchedulerTest capacitySchedulerTest.cpp)
add_gtest(contextProgressTest contextProgressTest.cu)
add_gtest(contextTransferCoordinatorTest contextTransferCoordinatorTest.cpp)
add_gtest(evictionPolicyTest evictionPolicyTest.cpp)
add_gtest(kvCacheManagerTest kvCacheManagerTest.cpp)
add_gtest(kvCacheManagerFabricMemoryTest kvCacheManagerFabricMemoryTest.cpp)
add_gtest(kvCacheManagerV2DefaultColdPageCodecTest
          kvCacheManagerV2DefaultColdPageCodecTest.cpp)
target_include_directories(
  kvCacheManagerV2DefaultColdPageCodecTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2ColdPageTest kvCacheManagerV2ColdPageTest.cpp)
target_include_directories(
  kvCacheManagerV2ColdPageTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager
          ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/sha256)
add_gtest(kvCacheManagerV2ColdPageCopyTest kvCacheManagerV2ColdPageCopyTest.cu)
target_include_directories(
  kvCacheManagerV2ColdPageCopyTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(coldPageCodecTest coldPageCodecTest.cpp)
target_include_directories(
  coldPageCodecTest PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2TypedIndexTest kvCacheManagerV2TypedIndexTest.cpp)
target_include_directories(
  kvCacheManagerV2TypedIndexTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2HostMemTest kvCacheManagerV2HostMemTest.cpp)
target_include_directories(
  kvCacheManagerV2HostMemTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2StagingBufferTest
          kvCacheManagerV2StagingBufferTest.cu)
target_include_directories(
  kvCacheManagerV2StagingBufferTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2StatsTest kvCacheManagerV2StatsTest.cpp)
target_include_directories(
  kvCacheManagerV2StatsTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager
          ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/sha256)
add_gtest(kvCacheManagerV2SlotAllocatorTest
          kvCacheManagerV2SlotAllocatorTest.cpp)
target_include_directories(
  kvCacheManagerV2SlotAllocatorTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2ReentrantSharedMutexTest
          kvCacheManagerV2ReentrantSharedMutexTest.cpp)
target_include_directories(
  kvCacheManagerV2ReentrantSharedMutexTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
add_gtest(kvCacheManagerV2ConcurrencyTest kvCacheManagerV2ConcurrencyTest.cpp)
target_include_directories(
  kvCacheManagerV2ConcurrencyTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager
          ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/sha256)

add_gtest(kvCacheManagerV2DigestPoolTest kvCacheManagerV2DigestPoolTest.cpp)
target_include_directories(
  kvCacheManagerV2DigestPoolTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager
          ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/sha256)
add_gtest(kvCacheUtilsTest kvCacheUtilsTest.cpp)
add_gtest(llmRequestTest llmRequestTest.cpp)
add_gtest(microBatchSchedulerTest microBatchSchedulerTest.cpp)
add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(staticThreadPoolTest staticThreadPoolTest.cpp)
add_gtest(rnnCacheFormatterTest rnnCacheFormatterTest.cpp)
add_gtest(agentTreeTest agentTreeTest.cpp)
add_gtest(truncateBlocksTest truncateBlocksTest.cpp)
