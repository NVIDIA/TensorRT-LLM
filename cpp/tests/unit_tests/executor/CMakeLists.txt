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

# Skip MOONCAKE related tests on Rocky8
set(IS_ROCKY8 FALSE)
if(EXISTS "/etc/redhat-release")
  set(IS_ROCKY8 TRUE)
endif()

if(NIXL_ROOT OR (MOONCAKE_ROOT AND NOT IS_ROCKY8))
  add_gtest(agentCommTest agentCommTest.cpp)
  add_gtest(transferAgentTest transferAgentTest.cpp)

  if(NIXL_ROOT)
    target_link_libraries(transferAgentTest PRIVATE tensorrt_llm_nixl_wrapper)
    target_link_libraries(agentCommTest PRIVATE tensorrt_llm_nixl_wrapper
                                                ${Python3_LIBRARIES})
    target_compile_definitions(transferAgentTest PRIVATE TEST_NIXL_BACKEND=1)
    target_compile_definitions(agentCommTest PRIVATE TEST_NIXL_BACKEND=1)
  endif()

  if(MOONCAKE_ROOT)
    target_link_libraries(transferAgentTest
                          PRIVATE tensorrt_llm_mooncake_wrapper)
    target_link_libraries(agentCommTest PRIVATE tensorrt_llm_mooncake_wrapper
                                                ${Python3_LIBRARIES})
    target_compile_definitions(transferAgentTest
                               PRIVATE TEST_MOONCAKE_BACKEND=1)
    target_compile_definitions(agentCommTest PRIVATE TEST_MOONCAKE_BACKEND=1)
  endif()
endif()
