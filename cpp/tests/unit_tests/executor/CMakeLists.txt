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
add_gtest(genUniqueAgentNameTest genUniqueAgentNameTest.cpp)
target_link_libraries(genUniqueAgentNameTest PRIVATE ${Python3_LIBRARIES})
add_gtest(ucxCommTest ucxCommTest.cpp)
target_link_libraries(ucxCommTest PRIVATE ${Python3_LIBRARIES})
target_link_libraries(serializeUtilsTest PRIVATE ${Python3_LIBRARIES})

# Bounce-buffer v2 pure-logic unit tests (no GPU / NIXL / zmq / network). The
# bounce core .cpp files are compiled directly into each test; they will later
# also be built into the NIXL wrapper (see bounce/CMakeLists.txt when the
# IO/data plane lands).
set(BOUNCE_DIR
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce
)
add_gtest(creditSchedulerTest creditSchedulerTest.cpp)
target_sources(creditSchedulerTest PRIVATE ${BOUNCE_DIR}/CreditScheduler.cpp
                                           ${BOUNCE_DIR}/BuddyAllocator.cpp)
add_gtest(bounceTransferPlanTest bounceTransferPlanTest.cpp)
target_sources(bounceTransferPlanTest
               PRIVATE ${BOUNCE_DIR}/BounceTransferPlan.cpp)
add_gtest(bounceMessageCodecTest bounceMessageCodecTest.cpp)
target_sources(bounceMessageCodecTest PRIVATE ${BOUNCE_DIR}/BounceMessage.cpp)
add_gtest(buddyAllocatorTest buddyAllocatorTest.cpp)
target_sources(buddyAllocatorTest PRIVATE ${BOUNCE_DIR}/BuddyAllocator.cpp)
# GPU test: execution-context pool (stream/event/scratch/hostPinned)
# acquire/release.
add_gtest(execPoolTest execPoolTest.cpp)
target_sources(execPoolTest PRIVATE ${BOUNCE_DIR}/ExecPool.cpp)
# GPU test: gather/scatter CUDA kernel round-trip (needs a CUDA device to run).
add_gtest(gatherScatterKernelTest gatherScatterKernelTest.cpp)
target_sources(gatherScatterKernelTest
               PRIVATE ${BOUNCE_DIR}/GatherScatterKernel.cu)
# GPU test: shared data-arena alloc / address mapping / shutdown.
add_gtest(bounceArenaTest bounceArenaTest.cpp)
target_sources(bounceArenaTest PRIVATE ${BOUNCE_DIR}/BounceArena.cpp)

# zmq control-plane test — only when zmq/cppzmq are available (fetched under
# ENABLE_UCX).
if(ENABLE_UCX)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZMQ REQUIRED libzmq)
  add_gtest(zmqControlChannelTest zmqControlChannelTest.cpp)
  target_sources(
    zmqControlChannelTest PRIVATE ${BOUNCE_DIR}/ZmqControlChannel.cpp
                                  ${BOUNCE_DIR}/BounceMessage.cpp)
  target_include_directories(
    zmqControlChannelTest PRIVATE ${CMAKE_BINARY_DIR}/_deps/cppzmq-src
                                  ${ZMQ_INCLUDE_DIRS})
  target_link_libraries(zmqControlChannelTest PRIVATE ${ZMQ_LIBRARIES})

  # End-to-end loopback: full reactor (gather -> D2D "RDMA" -> scatter) on one
  # GPU.
  add_gtest(bounceTransportTest bounceTransportTest.cpp)
  target_sources(
    bounceTransportTest
    PRIVATE ${BOUNCE_DIR}/BounceTransport.cpp
            ${BOUNCE_DIR}/BounceTransferPlan.cpp
            ${BOUNCE_DIR}/BuddyAllocator.cpp
            ${BOUNCE_DIR}/CreditScheduler.cpp
            ${BOUNCE_DIR}/BounceMessage.cpp
            ${BOUNCE_DIR}/BounceArena.cpp
            ${BOUNCE_DIR}/ExecPool.cpp
            ${BOUNCE_DIR}/GatherScatterKernel.cu
            ${BOUNCE_DIR}/ZmqControlChannel.cpp
            ${BOUNCE_DIR}/LocalCopyTransferEngine.cpp)
  target_include_directories(
    bounceTransportTest PRIVATE ${CMAKE_BINARY_DIR}/_deps/cppzmq-src
                                ${ZMQ_INCLUDE_DIRS})
  target_link_libraries(bounceTransportTest PRIVATE ${ZMQ_LIBRARIES})

  # Boundary / failure-path + concurrency tests for the reactor.
  set(BOUNCE_REACTOR_SRCS
      ${BOUNCE_DIR}/BounceTransport.cpp
      ${BOUNCE_DIR}/BounceTransferPlan.cpp
      ${BOUNCE_DIR}/BuddyAllocator.cpp
      ${BOUNCE_DIR}/CreditScheduler.cpp
      ${BOUNCE_DIR}/BounceMessage.cpp
      ${BOUNCE_DIR}/BounceArena.cpp
      ${BOUNCE_DIR}/ExecPool.cpp
      ${BOUNCE_DIR}/GatherScatterKernel.cu
      ${BOUNCE_DIR}/ZmqControlChannel.cpp
      ${BOUNCE_DIR}/LocalCopyTransferEngine.cpp)
  add_gtest(bounceTransportFailureTest bounceTransportFailureTest.cpp)
  target_sources(bounceTransportFailureTest PRIVATE ${BOUNCE_REACTOR_SRCS})
  target_include_directories(
    bounceTransportFailureTest PRIVATE ${CMAKE_BINARY_DIR}/_deps/cppzmq-src
                                       ${ZMQ_INCLUDE_DIRS})
  target_link_libraries(bounceTransportFailureTest PRIVATE ${ZMQ_LIBRARIES})

  # Optional ThreadSanitizer variant of the reactor concurrency test. The bounce
  # sources + the test are instrumented; the shared libtensorrt_llm (linked for
  # the logger) and CUDA runtime are left uninstrumented (TSan
  # partial-instrumentation mode). Enable with -DBOUNCE_TSAN=ON and run with a
  # suppressions file that ignores CUDA/driver/libtllm frames.
  option(BOUNCE_TSAN
         "Build a ThreadSanitizer variant of the bounce reactor test" OFF)
  if(BOUNCE_TSAN)
    add_gtest(bounceTransportTsanTest bounceTransportFailureTest.cpp)
    target_sources(bounceTransportTsanTest PRIVATE ${BOUNCE_REACTOR_SRCS})
    target_include_directories(
      bounceTransportTsanTest PRIVATE ${CMAKE_BINARY_DIR}/_deps/cppzmq-src
                                      ${ZMQ_INCLUDE_DIRS})
    target_link_libraries(bounceTransportTsanTest PRIVATE ${ZMQ_LIBRARIES})
    target_compile_options(
      bounceTransportTsanTest
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread -g>
              $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=thread>)
    target_link_options(bounceTransportTsanTest PRIVATE -fsanitize=thread)
  endif()
endif()

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

    add_gtest(coalesceTest coalesceTest.cpp)
    target_link_libraries(coalesceTest PRIVATE tensorrt_llm_nixl_wrapper
                                               NIXL::nixl)

    # Bounce v2 NIXL TransferEngine: compile/link smoke test against real NIXL.
    add_gtest(nixlTransferEngineTest nixlTransferEngineTest.cpp)
    target_sources(nixlTransferEngineTest
                   PRIVATE ${BOUNCE_DIR}/NixlTransferEngine.cpp)
    # NixlTransferEngine.cpp calls nixlDescList::addDesc, which lives in
    # libnixl_build.
    target_link_libraries(
      nixlTransferEngineTest PRIVATE tensorrt_llm_nixl_wrapper NIXL::nixl
                                     ${NIXL_BUILD_LIBRARY})

    # Real-NIXL end-to-end loopback (needs both NIXL and zmq). Two real agents,
    # real RDMA.
    if(ENABLE_UCX)
      add_gtest(bounceNixlE2ETest bounceNixlE2ETest.cpp)
      target_sources(
        bounceNixlE2ETest
        PRIVATE ${BOUNCE_DIR}/BounceTransport.cpp
                ${BOUNCE_DIR}/BounceTransferPlan.cpp
                ${BOUNCE_DIR}/BuddyAllocator.cpp
                ${BOUNCE_DIR}/CreditScheduler.cpp
                ${BOUNCE_DIR}/BounceMessage.cpp
                ${BOUNCE_DIR}/BounceArena.cpp
                ${BOUNCE_DIR}/ExecPool.cpp
                ${BOUNCE_DIR}/GatherScatterKernel.cu
                ${BOUNCE_DIR}/ZmqControlChannel.cpp
                ${BOUNCE_DIR}/NixlTransferEngine.cpp)
      target_include_directories(
        bounceNixlE2ETest PRIVATE ${CMAKE_BINARY_DIR}/_deps/cppzmq-src
                                  ${ZMQ_INCLUDE_DIRS})
      target_link_libraries(
        bounceNixlE2ETest PRIVATE tensorrt_llm_nixl_wrapper NIXL::nixl
                                  ${NIXL_BUILD_LIBRARY} ${ZMQ_LIBRARIES})

      # Production-path e2e: bounce engaged via
      # NixlTransferAgent::submitTransferRequests. Bounce lives inside the
      # wrapper; this test only uses the public agent API.
      add_gtest(bounceAgentE2ETest bounceAgentE2ETest.cpp)
      target_link_libraries(
        bounceAgentE2ETest PRIVATE tensorrt_llm_nixl_wrapper NIXL::nixl
                                   ${NIXL_BUILD_LIBRARY})
    endif()
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
