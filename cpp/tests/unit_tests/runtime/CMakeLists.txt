# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

add_gtest(bufferManagerTest bufferManagerTest.cpp)
add_gtest(cudaMemPoolTest cudaMemPoolTest.cpp)
add_gtest(decodingLayerWorkspaceTest decodingLayerWorkspaceTest.cpp)
add_gtest(gdrcopyTest gdrcopyTest.cpp)
add_gtest(gptDecoderBatchedTest gptDecoderBatchedTest.cpp)
add_gtest(gptDecoderTest gptDecoderTest.cpp)
add_gtest(hostAccessibleDeviceAllocatorTest
          hostAccessibleDeviceAllocatorTest.cu)
add_gtest(iBufferTest iBufferTest.cpp)
add_gtest(iTensorTest iTensorTest.cpp)
add_gtest(loraCacheTest loraCacheTest.cpp)
add_gtest(loraManagerTest loraManagerTest.cpp)
add_gtest(loraUtilsTest loraUtilsTest.cpp)
add_gtest(medusaModuleTest medusaModuleTest.cpp)
add_gtest(moeLoadBalancerTest moeLoadBalancerTest.cpp)
add_gtest(runtimeKernelTest runtimeKernelTest.cpp)
add_gtest(samplingConfigTest samplingConfigTest.cpp)
add_gtest(samplingTest samplingTest.cpp)
add_gtest(sanitizerTest sanitizerTest.cpp)
add_gtest(tllmBuffersTest tllmBuffersTest.cpp)
add_gtest(tllmRuntimeTest tllmRuntimeTest.cpp)
add_gtest(transposeKVKernelTest transposeKVKernelTest.cpp)
add_gtest(utilsTest utilsTest.cpp)
add_gtest(virtualMemoryTest virtualMemoryTest.cpp)
add_gtest(workerPoolTest workerPoolTest.cpp)
add_gtest(worldConfigTest worldConfigTest.cpp)

if(${BUILD_PYT})
  add_gtest(torchTest torchTest.cpp)
  target_link_libraries(torchTest PUBLIC ${TORCH_LIBRARIES})
endif()
