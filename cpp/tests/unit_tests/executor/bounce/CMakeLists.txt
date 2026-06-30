# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
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

# Unit tests for the NIXL bounce-buffer v2 KV-cache transfer. The code under
# test lives in tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/; the
# bounce core .cpp files are compiled directly into each test (they are also
# built into the NIXL wrapper for production). Three tiers, each independently
# gated: - pure-logic / GPU tests  (no NIXL, no zmq, no network) - control-plane
# tests     (need zmq + the fetched cppzmq header) - real-NIXL e2e tests (need
# NIXL_ROOT + zmq; two real agents, real RDMA)
set(BOUNCE_DIR
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce
)

# ---- Pure-logic + GPU unit tests (no NIXL / zmq / network) ----
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
# acquire/release. ExecPool now sizes the optional cub batched-copy workspace
# via batchedCopyCubTempBytes(), so it links the kernel TU
# (GatherScatterKernel.cu) too.
add_gtest(execPoolTest execPoolTest.cpp)
target_sources(execPoolTest PRIVATE ${BOUNCE_DIR}/ExecPool.cpp
                                    ${BOUNCE_DIR}/GatherScatterKernel.cu)
# GPU test: gather/scatter CUDA kernel round-trip (needs a CUDA device to run).
add_gtest(gatherScatterKernelTest gatherScatterKernelTest.cpp)
target_sources(gatherScatterKernelTest
               PRIVATE ${BOUNCE_DIR}/GatherScatterKernel.cu)
# GPU test: shared data-arena alloc / address mapping / shutdown.
add_gtest(bounceArenaTest bounceArenaTest.cpp)
target_sources(bounceArenaTest PRIVATE ${BOUNCE_DIR}/BounceArena.cpp)

# ---- Control-plane tests — gated on zmq availability (+ the fetched cppzmq
# header), NOT UCX, mirroring the library (whose only extra dep beyond NIXL is
# zmq, orthogonal to UCX). ZMQ_FOUND is reused by the real-NIXL block below.
# ----
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(ZMQ libzmq)
endif()
if(ZMQ_FOUND AND EXISTS ${CMAKE_BINARY_DIR}/_deps/cppzmq-src)
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

# ---- Real-NIXL tests (need NIXL_ROOT). The bounce engine talks to real NIXL.
# ----
if(NIXL_ROOT)
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
  # real RDMA. Gated on zmq (ZMQ_FOUND set above), not UCX; cppzmq is fetched
  # when NIXL is on.
  if(ZMQ_FOUND)
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
