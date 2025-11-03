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

add_gtest(executorSamplingConfigTest samplingConfigTest.cpp)
add_gtest(kvCacheConfigTest kvCacheConfigTest.cpp)
add_gtest(decodingConfigTest decodingConfigTest.cpp)
add_gtest(requestTest requestTest.cpp)
add_gtest(responseTest responseTest.cpp)

add_gtest(executorTestSmall executorTestSmall.cpp)
target_link_libraries(executorTestSmall PRIVATE testingUtils)

add_gtest(executorTestSmallArbitraryOutputTensors
          executorTestSmallArbitraryOutputTensors.cpp)
target_link_libraries(executorTestSmallArbitraryOutputTensors
                      PRIVATE testingUtils)

add_gtest(executorConfigTest executorConfigTest.cpp)
add_gtest(executorTensorTest tensorTest.cpp)
add_gtest(serializeUtilsTest serializeUtilsTest.cpp)
add_gtest(requestWithIdTest requestWithIdTest.cpp)
add_gtest(loraConfigTest loraConfigTest.cpp)
add_gtest(intervalSetTest intervalSetTest.cpp)
add_gtest(dynamicBatchTunerTest dynamicBatchTunerTest.cpp)
add_gtest(ucxCommTest ucxCommTest.cpp)
target_link_libraries(ucxCommTest PRIVATE ${Python3_LIBRARIES})
target_link_libraries(serializeUtilsTest PRIVATE ${Python3_LIBRARIES})

if(NIXL_ROOT)
  add_gtest(transferAgentTest transferAgentTest.cpp)
  add_gtest(agentCommTest agentCommTest.cpp)
  target_link_libraries(transferAgentTest PRIVATE tensorrt_llm_nixl_wrapper)
  target_link_libraries(agentCommTest PRIVATE tensorrt_llm_nixl_wrapper
                                              ${Python3_LIBRARIES})
endif()
