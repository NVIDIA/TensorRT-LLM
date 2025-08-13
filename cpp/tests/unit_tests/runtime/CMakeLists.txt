# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

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
