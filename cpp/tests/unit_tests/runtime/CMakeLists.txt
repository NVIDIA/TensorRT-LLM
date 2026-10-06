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

add_tllm_gtest(
  bufferManagerTest
  OWNER
  runtime
  SOURCES
  bufferManagerTest.cpp
  LINK_LIBRARIES
  tllm::runtime_buffers
  LABELS
  "unit;runtime;gpu"
  REQUIRES_GPU)
add_tllm_gtest(
  cudaMemPoolTest
  OWNER
  runtime
  SOURCES
  cudaMemPoolTest.cpp
  LINK_LIBRARIES
  tllm::runtime_memory
  LABELS
  "unit;runtime;gpu"
  REQUIRES_GPU)
add_tllm_gtest(gdrcopyTest SOURCES gdrcopyTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(
  hostAccessibleDeviceAllocatorTest SOURCES
  hostAccessibleDeviceAllocatorTest.cu FULL_STACK LABELS "unit;full-stack")
add_tllm_gtest(
  iBufferTest
  OWNER
  runtime
  SOURCES
  iBufferTest.cpp
  LINK_LIBRARIES
  tllm::runtime_buffers
  LABELS
  "unit;runtime;host")
add_tllm_gtest(ipcNvlsMemoryTest SOURCES ipcNvlsMemoryTest.cpp FULL_STACK
               LABELS "unit;full-stack")
add_tllm_gtest(
  iTensorTest
  OWNER
  runtime
  SOURCES
  iTensorTest.cpp
  LINK_LIBRARIES
  tllm::runtime_buffers
  LABELS
  "unit;runtime;host")
add_tllm_gtest(loraCacheTest SOURCES loraCacheTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(loraUtilsTest SOURCES loraUtilsTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(moeLoadBalancerTest SOURCES moeLoadBalancerTest.cpp FULL_STACK
               LABELS "unit;full-stack")
add_tllm_gtest(
  runtimeMpiUtilsTest
  OWNER
  runtime
  SOURCES
  mpiUtilsTest.cpp
  LINK_LIBRARIES
  tllm::runtime_distributed
  LABELS
  "unit;runtime;host")
add_tllm_gtest(localizationTest SOURCES localizationTest.cu FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(
  localityDomainPublicConfigTest SOURCES localityDomainPublicConfigTest.cpp
  FULL_STACK LABELS "unit;full-stack")
add_tllm_gtest(sanitizerTest SOURCES sanitizerTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(
  tllmBuffersTest
  OWNER
  runtime
  SOURCES
  tllmBuffersTest.cpp
  LINK_LIBRARIES
  tllm::runtime_buffers
  LABELS
  "unit;runtime;gpu"
  REQUIRES_GPU)
add_tllm_gtest(transposeKVKernelTest SOURCES transposeKVKernelTest.cpp
               FULL_STACK LABELS "unit;full-stack")
add_tllm_gtest(utilsTest SOURCES utilsTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(
  virtualMemoryTest
  OWNER
  runtime
  SOURCES
  virtualMemoryTest.cpp
  LINK_LIBRARIES
  tllm::runtime_buffers
  LABELS
  "unit;runtime;gpu"
  REQUIRES_GPU)
add_tllm_gtest(workerPoolTest SOURCES workerPoolTest.cpp FULL_STACK LABELS
               "unit;full-stack")
add_tllm_gtest(
  worldConfigTest
  OWNER
  runtime
  SOURCES
  worldConfigTest.cpp
  LINK_LIBRARIES
  tllm::runtime_distributed
  LABELS
  "unit;runtime;host")

if(${BUILD_PYT})
  add_tllm_gtest(torchTest SOURCES torchTest.cpp FULL_STACK LABELS
                 "unit;full-stack")
  target_link_libraries(torchTest PUBLIC ${TORCH_LIBRARIES})
endif()
