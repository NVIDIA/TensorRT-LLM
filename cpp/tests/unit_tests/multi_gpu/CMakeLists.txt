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

add_subdirectory(kernels)

add_tllm_gtest(cacheTransceiverTest SOURCES cacheTransceiverTest.cpp FULL_STACK
               LABELS "unit;full-stack;gpu")
target_link_libraries(cacheTransceiverTest PRIVATE ${Python3_LIBRARIES})

add_tllm_gtest(
  kvCacheManagerV2HostMemNumaTest
  OWNER
  batch-manager
  SOURCES
  kvCacheManagerV2HostMemNumaTest.cpp
  LINK_LIBRARIES
  $<LINK_LIBRARY:WHOLE_ARCHIVE,tllm_kv_cache_test_support>
  FULL_STACK
  LABELS
  "unit;full-stack;batch-manager;gpu")
target_include_directories(
  kvCacheManagerV2HostMemNumaTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
# Queries libnuma and get_mempolicy directly to see where pages landed.
target_link_libraries(kvCacheManagerV2HostMemNumaTest PRIVATE numa)

add_tllm_gtest(
  mpiUtilsTest
  OWNER
  runtime
  SOURCES
  mpiUtilsTest.cpp
  LINK_LIBRARIES
  tllm::runtime_distributed
  tllm::runtime_buffers
  LABELS
  "unit;runtime;host")
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(mpiUtilsTest PRIVATE ${NCCL_LIB})
endif()
add_tllm_gtest(
  userBufferTest
  OWNER
  kernel
  SOURCES
  userBufferTest.cpp
  LINK_LIBRARIES
  tllm::kernel_userbuffers
  tllm::runtime_distributed
  LABELS
  "unit;kernels;userbuffers;multi-gpu;gpu")
add_tllm_gtest(ncclUtilsTest SOURCES ncclUtilsTest.cpp FULL_STACK LABELS
               "unit;full-stack;gpu")
target_link_libraries(ncclUtilsTest PRIVATE ${Python3_LIBRARIES})
if(BUILD_PYT)
  target_compile_definitions(cacheTransceiverTest PUBLIC BUILD_PYT)
  target_link_libraries(cacheTransceiverTest PUBLIC ${TORCH_LIBRARIES})
endif()
if(BUILD_PYT)
  target_compile_definitions(ncclUtilsTest PUBLIC BUILD_PYT)
  target_link_libraries(ncclUtilsTest PUBLIC ${TORCH_LIBRARIES})
endif()
