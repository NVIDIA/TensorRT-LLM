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
  executorSamplingConfigTest
  OWNER
  executor
  SOURCES
  samplingConfigTest.cpp
  LINK_LIBRARIES
  tllm::executor_serialization
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  kvCacheConfigTest
  OWNER
  executor
  SOURCES
  kvCacheConfigTest.cpp
  LINK_LIBRARIES
  tllm::executor_types
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  requestTest
  OWNER
  executor
  SOURCES
  requestTest.cpp
  LINK_LIBRARIES
  tllm::executor_values
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  responseTest
  OWNER
  executor
  SOURCES
  responseTest.cpp
  LINK_LIBRARIES
  tllm::executor_values
  LABELS
  "unit;executor;host")

add_tllm_gtest(
  executorConfigTest
  OWNER
  executor
  SOURCES
  executorConfigTest.cpp
  LINK_LIBRARIES
  tllm::executor_types
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  executorTensorTest
  OWNER
  executor
  SOURCES
  tensorTest.cpp
  LINK_LIBRARIES
  tllm::executor_values
  LABELS
  "unit;executor;gpu")
add_tllm_gtest(
  serializeUtilsTest
  OWNER
  executor
  SOURCES
  serializeUtilsTest.cpp
  LINK_LIBRARIES
  tllm::executor_values
  tllm::executor_memory_desc
  tllm::batch_request_info
  tllm::runtime_distributed
  ${Python3_LIBRARIES}
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  loraConfigTest
  OWNER
  executor
  SOURCES
  loraConfigTest.cpp
  LINK_LIBRARIES
  tllm::executor_values
  LABELS
  "unit;executor;gpu")
add_tllm_gtest(
  coalesceTest
  OWNER
  executor
  SOURCES
  coalesceTest.cpp
  LINK_LIBRARIES
  tllm::executor_transfer_agent
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  genUniqueAgentNameTest
  OWNER
  executor
  SOURCES
  genUniqueAgentNameTest.cpp
  LINK_LIBRARIES
  tllm::executor_agent_name
  Threads::Threads
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  ucxCommTest
  SOURCES
  ucxCommTest.cpp
  FULL_STACK
  LABELS
  "unit;full-stack;gpu"
  TIMEOUT
  60)
target_link_libraries(ucxCommTest PRIVATE ${Python3_LIBRARIES})

# NIXL bounce-buffer v2 KV-cache transfer tests live in their own subdirectory
# (mirrors the source layout under nixl_utils/bounce/). It self-gates on zmq and
# NIXL_ROOT, so it is safe to descend unconditionally.
add_subdirectory(bounce)

# Skip MOONCAKE related tests on Rocky8
set(IS_ROCKY8 FALSE)
if(EXISTS "/etc/redhat-release")
  set(IS_ROCKY8 TRUE)
endif()

if(NIXL_ROOT OR (MOONCAKE_ROOT AND NOT IS_ROCKY8))
  add_tllm_gtest(agentCommTest SOURCES agentCommTest.cpp FULL_STACK LABELS
                 "unit;full-stack;gpu")
  add_tllm_gtest(transferAgentTest SOURCES transferAgentTest.cpp FULL_STACK
                 LABELS "unit;full-stack;gpu")

  if(NIXL_ROOT)
    target_link_libraries(transferAgentTest PRIVATE tensorrt_llm_nixl_wrapper)
    target_link_libraries(agentCommTest PRIVATE tensorrt_llm_nixl_wrapper
                                                ${Python3_LIBRARIES})
    target_compile_definitions(transferAgentTest PRIVATE TEST_NIXL_BACKEND=1)
    target_compile_definitions(agentCommTest PRIVATE TEST_NIXL_BACKEND=1)

    # NIXL bounce v2 tests (bounceTransportTest / bounceTransportFailureTest /
    # bounceAgentE2ETest) live in bounce/CMakeLists.txt.
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
