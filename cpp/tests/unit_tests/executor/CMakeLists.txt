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
add_gtest(genUniqueAgentNameTest genUniqueAgentNameTest.cpp)
target_link_libraries(genUniqueAgentNameTest PRIVATE ${Python3_LIBRARIES})
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
    # The P2P-specific tests rely on the NIXL wrapper, so they are gated on
    # NIXL_ROOT and live inside this branch alongside their link configuration.
    add_gtest(p2pMemInfoTest p2pMemInfoTest.cu)
    add_gtest(p2pTransferAgentTest p2pTransferAgentTest.cu)
    target_link_libraries(p2pMemInfoTest PRIVATE tensorrt_llm_nixl_wrapper)
    target_link_libraries(p2pTransferAgentTest PRIVATE tensorrt_llm_nixl_wrapper
                                                       CUDA::cudart)
    # End-to-end test: drives two NixlTransferAgents through all three P2P path
    # branches. Requires a P2P-capable GPU (POSIX-FD or FABRIC VMM handles) and
    # same-host UDS — not suitable for the default l0 stage. Link against the
    # NIXL wrapper so the test can call NixlTransferAgent::getPathCounters()
    # directly.
    #
    # The two EnvGated_* tests inside this binary are stateless about env: they
    # skip unless TRTLLM_KV_TRANSFER_P2P_DISABLE /
    # TRTLLM_KV_TRANSFER_P2P_MIXED_DISABLE was set at process launch (envUtils
    # caches the value in a `static bool const`, so it can't be flipped inside
    # the test). To exercise them, run the binary from CI / from a shell wrapper
    # with the env set, e.g.: TRTLLM_KV_TRANSFER_P2P_DISABLE=1 ./nixlP2pE2ETest
    # \ --gtest_filter=NixlP2pE2ETest.EnvGated_P2pDisabled
    # TRTLLM_KV_TRANSFER_P2P_MIXED_DISABLE=1 ./nixlP2pE2ETest \
    # --gtest_filter=NixlP2pE2ETest.EnvGated_MixedDisabledFallsBackFullNixl
    add_gtest(nixlP2pE2ETest nixlP2pE2ETest.cu)
    # nixl.h is transitively included via nixl_utils/transferAgent.h, so the
    # test needs NIXL::nixl's headers in addition to the wrapper library itself.
    target_link_libraries(
      nixlP2pE2ETest PRIVATE tensorrt_llm_nixl_wrapper NIXL::nixl CUDA::cudart
                             CUDA::cuda_driver)
    target_compile_definitions(nixlP2pE2ETest PRIVATE TEST_NIXL_BACKEND=1)
  endif()

  if(NIXL_ROOT)
    target_link_libraries(transferAgentTest PRIVATE tensorrt_llm_nixl_wrapper)
    target_link_libraries(agentCommTest PRIVATE tensorrt_llm_nixl_wrapper
                                                ${Python3_LIBRARIES})
    target_compile_definitions(transferAgentTest PRIVATE TEST_NIXL_BACKEND=1)
    target_compile_definitions(agentCommTest PRIVATE TEST_NIXL_BACKEND=1)

    add_gtest(coalesceTest coalesceTest.cpp)
    target_link_libraries(coalesceTest PRIVATE tensorrt_llm_nixl_wrapper
                                               NIXL::nixl)
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
