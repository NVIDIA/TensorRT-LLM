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

# Whole-archive keeps the poison listener's static registration in each test.
add_library(tllm_kv_cache_test_support STATIC EXCLUDE_FROM_ALL
            kvCacheManagerV2PoisonCheck.cpp)
target_link_libraries(
  tllm_kv_cache_test_support
  PUBLIC tllm::kv_cache
  PRIVATE gtest)
target_compile_features(tllm_kv_cache_test_support PRIVATE cxx_std_20)

add_tllm_gtest(
  radixTreeTest
  OWNER
  batch-manager
  SOURCES
  radixTreeTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  blockKeyTest
  OWNER
  batch-manager
  SOURCES
  blockKeyTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  radixBlockTreeTest
  OWNER
  batch-manager
  SOURCES
  radixBlockTreeTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  cacheTransBufferTest
  OWNER
  batch-manager
  SOURCES
  cacheTransBufferTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  bufferIndexHolderTest
  OWNER
  batch-manager
  SOURCES
  bufferIndexHolderTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  capacitySchedulerTest
  OWNER
  batch-manager
  SOURCES
  capacitySchedulerTest.cpp
  LINK_LIBRARIES
  tllm::batch_scheduler
  tllm::executor_request_utils
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  contextProgressTest
  OWNER
  batch-manager
  SOURCES
  contextProgressTest.cu
  LINK_LIBRARIES
  tllm::batch_scheduler
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_gtest(contextTransferCoordinatorTest contextTransferCoordinatorTest.cpp)

add_tllm_gtest(
  evictionPolicyTest
  OWNER
  batch-manager
  SOURCES
  evictionPolicyTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheManagerTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerFabricMemoryTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerFabricMemoryTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2ConfigTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2ConfigTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheManagerV2DefaultColdPageCodecTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2DefaultColdPageCodecTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2ColdPageTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2ColdPageTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2ColdPageCopyTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2ColdPageCopyTest.cu
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  coldPageCodecTest
  OWNER
  batch-manager
  SOURCES
  coldPageCodecTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2TypedIndexTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2TypedIndexTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheManagerV2HostMemTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2HostMemTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2StagingBufferTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2StagingBufferTest.cu
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2StatsTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2StatsTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2SlotAllocatorTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2SlotAllocatorTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheManagerV2ReentrantSharedMutexTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2ReentrantSharedMutexTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheManagerV2ConcurrencyTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2ConcurrencyTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  kvCacheManagerV2DigestPoolTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2DigestPoolTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  kvCacheUtilsTest
  OWNER
  batch-manager
  SOURCES
  kvCacheUtilsTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  llmRequestTest
  OWNER
  batch-manager
  SOURCES
  llmRequestTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_tllm_gtest(
  microBatchSchedulerTest
  OWNER
  batch-manager
  SOURCES
  microBatchSchedulerTest.cpp
  LINK_LIBRARIES
  tllm::batch_scheduler
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)

add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)

add_gtest(rnnCacheFormatterTest rnnCacheFormatterTest.cpp)

add_tllm_gtest(
  agentTreeTest
  OWNER
  batch-manager
  SOURCES
  agentTreeTest.cpp
  LINK_LIBRARIES
  tllm::batch_scheduler
  LABELS
  "unit;batch-manager;host")

add_tllm_gtest(
  truncateBlocksTest
  OWNER
  batch-manager
  SOURCES
  truncateBlocksTest.cpp
  LINK_LIBRARIES
  tllm::kv_cache_legacy
  LABELS
  "unit;batch-manager;gpu"
  REQUIRES_GPU)
