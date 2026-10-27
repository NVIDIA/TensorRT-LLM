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
add_gtest(gdrcopyTest gdrcopyTest.cpp)
add_gtest(hostAccessibleDeviceAllocatorTest
          hostAccessibleDeviceAllocatorTest.cu)
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
add_gtest(ipcNvlsMemoryTest ipcNvlsMemoryTest.cpp)
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
add_gtest(loraCacheTest loraCacheTest.cpp)
add_gtest(loraUtilsTest loraUtilsTest.cpp)
add_gtest(moeLoadBalancerTest moeLoadBalancerTest.cpp)
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
add_gtest(localizationTest localizationTest.cu)
add_gtest(localityDomainPublicConfigTest localityDomainPublicConfigTest.cpp)
add_gtest(sanitizerTest sanitizerTest.cpp)
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
add_gtest(transposeKVKernelTest transposeKVKernelTest.cpp)
add_gtest(utilsTest utilsTest.cpp)
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
add_gtest(workerPoolTest workerPoolTest.cpp)
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
  add_gtest(torchTest torchTest.cpp)
  target_link_libraries(torchTest PUBLIC ${TORCH_LIBRARIES})
endif()
