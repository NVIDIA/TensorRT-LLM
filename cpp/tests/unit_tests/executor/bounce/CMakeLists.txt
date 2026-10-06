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

# Bounce tests cover host logic, CUDA execution, ZeroMQ control channels and
# real NIXL transport. Transport tests require NIXL_ROOT and ZeroMQ.
set(BOUNCE_DIR
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce
)

# ---- Pure-logic + GPU unit tests (no NIXL / zmq / network) ----
add_tllm_gtest(
  creditSchedulerTest
  OWNER
  executor
  SOURCES
  creditSchedulerTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_core
  LABELS
  "unit;executor;bounce;host")
add_tllm_gtest(
  bounceTransferPlanTest
  OWNER
  executor
  SOURCES
  bounceTransferPlanTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_core
  LABELS
  "unit;executor;bounce;host")
add_tllm_gtest(
  bounceMessageCodecTest
  OWNER
  executor
  SOURCES
  bounceMessageCodecTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_core
  LABELS
  "unit;executor;bounce;host")
# Header-only: knob resolution (agent_bounce_params dict > TRTLLM_NIXL_BOUNCE_*
# env > default), byte-size suffixes, garbage fallback, derived-timeout
# invariants.
add_tllm_gtest(
  bounceConfigTest
  OWNER
  executor
  SOURCES
  bounceConfigTest.cpp
  LINK_LIBRARIES
  tllm::common_logging
  LABELS
  "unit;executor;host")
add_tllm_gtest(
  buddyAllocatorTest
  OWNER
  executor
  SOURCES
  buddyAllocatorTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_core
  LABELS
  "unit;executor;bounce;host")
# GPU test: execution-context pool (stream/event/scratch/hostPinned)
# acquire/release.
add_tllm_gtest(
  execPoolTest
  OWNER
  executor
  SOURCES
  execPoolTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_cuda
  LABELS
  "unit;executor;bounce;gpu")
# GPU test: gather/scatter CUDA kernel round-trip (needs a CUDA device to run).
add_tllm_gtest(
  gatherScatterKernelTest
  OWNER
  executor
  SOURCES
  gatherScatterKernelTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_cuda
  LABELS
  "unit;executor;bounce;gpu")
# GPU test: shared data-arena alloc / address mapping / shutdown.
add_tllm_gtest(
  bounceArenaTest
  OWNER
  executor
  SOURCES
  bounceArenaTest.cpp
  LINK_LIBRARIES
  tllm::executor_bounce_arena
  LABELS
  "unit;executor;bounce;gpu")

# ---- Control-plane unit test: ZeroMQ and cppzmq, without NIXL. ----
if(TARGET tllm_executor_bounce_control)
  set(BOUNCE_CPPZMQ_AVAILABLE TRUE)
  add_tllm_gtest(
    zmqControlChannelTest
    OWNER
    executor
    SOURCES
    zmqControlChannelTest.cpp
    LINK_LIBRARIES
    tllm::executor_bounce_control
    LABELS
    "unit;executor;bounce;host")
else()
  set(BOUNCE_CPPZMQ_AVAILABLE FALSE)
endif()

# ---- Real-NIXL tests (need NIXL_ROOT). The bounce DATA PLANE is always real
# NIXL — there is no loopback fake — so every test that drives the reactor
# (transport, failure-path, e2e) lives here, using the production components and
# bounceTestNixlNode.h helpers. ----
if(NIXL_ROOT)
  # The reactor tests also need zmq for the control plane (cppzmq is fetched
  # when NIXL is on); all of them share these include/link dependencies. Same
  # gate as the zmq-only tier above.
  if(BOUNCE_CPPZMQ_AVAILABLE)
    set(BOUNCE_TEST_INCS ${CMAKE_BINARY_DIR}/_deps/cppzmq-src)
    set(BOUNCE_TEST_LIBS tensorrt_llm_nixl_wrapper NIXL::nixl
                         ${NIXL_BUILD_LIBRARY} PkgConfig::TLLM_BOUNCE_ZMQ)

    # Reactor tests use the production component or its instrumented variant.
    function(add_bounce_reactor_test name src)
      cmake_parse_arguments(ARG "INSTRUMENTED" "" "" ${ARGN})
      if(ARG_INSTRUMENTED)
        set(reactor tllm::executor_bounce_reactor_tsan)
      else()
        set(reactor tllm::executor_bounce_reactor)
      endif()
      add_tllm_gtest(
        ${name}
        OWNER
        executor
        SOURCES
        ${src}
        LINK_LIBRARIES
        ${reactor}
        ${BOUNCE_TEST_LIBS}
        LABELS
        "unit;executor;bounce;integration;gpu")
      target_include_directories(${name} PRIVATE ${BOUNCE_TEST_INCS})
    endfunction()

    # End-to-end transport over real RDMA (gather -> RDMA write -> scatter +
    # credit recycling), byte-exact.
    add_bounce_reactor_test(bounceTransportTest bounceTransportTest.cpp)

    # Boundary, failure-path, and concurrency tests.
    add_bounce_reactor_test(bounceTransportFailureTest
                            bounceTransportFailureTest.cpp)

    # Optional ThreadSanitizer variant of the reactor concurrency test. The
    # bounce component and test are instrumented; the production NIXL wrapper,
    # common helpers, and CUDA runtime are left uninstrumented. Enable with
    # -DBOUNCE_TSAN=ON and use suppressions for CUDA/driver/runtime frames.
    option(BOUNCE_TSAN
           "Build a ThreadSanitizer variant of the bounce reactor test" OFF)
    if(BOUNCE_TSAN)
      add_library(
        tllm_executor_bounce_reactor_tsan STATIC EXCLUDE_FROM_ALL
        ${BOUNCE_DIR}/BounceTransport.cpp
        ${BOUNCE_DIR}/BounceTransferPlan.cpp
        ${BOUNCE_DIR}/BuddyAllocator.cpp
        ${BOUNCE_DIR}/CreditScheduler.cpp
        ${BOUNCE_DIR}/BounceMessage.cpp
        ${BOUNCE_DIR}/BounceArena.cpp
        ${BOUNCE_DIR}/ExecPool.cpp
        ${BOUNCE_DIR}/GatherScatterKernel.cu
        ${BOUNCE_DIR}/ZmqControlChannel.cpp)
      add_library(tllm::executor_bounce_reactor_tsan ALIAS
                  tllm_executor_bounce_reactor_tsan)
      target_link_libraries(
        tllm_executor_bounce_reactor_tsan
        PUBLIC tllm::common_cuda tllm::executor_transfer_agent NIXL::nixl
               PkgConfig::TLLM_BOUNCE_ZMQ Threads::Threads
        PRIVATE tllm::kv_cache_legacy)
      target_include_directories(tllm_executor_bounce_reactor_tsan SYSTEM
                                 PUBLIC ${BOUNCE_TEST_INCS})
      target_compile_features(tllm_executor_bounce_reactor_tsan
                              PUBLIC cxx_std_20)
      target_compile_options(
        tllm_executor_bounce_reactor_tsan
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-error -fsanitize=thread -g>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=thread>)
      target_link_options(tllm_executor_bounce_reactor_tsan INTERFACE
                          -fsanitize=thread)
      set_target_properties(
        tllm_executor_bounce_reactor_tsan
        PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

      add_bounce_reactor_test(bounceTransportTsanTest
                              bounceTransportFailureTest.cpp INSTRUMENTED)
      target_compile_options(bounceTransportTsanTest PRIVATE -fsanitize=thread
                                                             -g)
    endif()

    # Production-path e2e: bounce engaged via
    # NixlTransferAgent::submitTransferRequests. Bounce lives inside the
    # wrapper; this test only uses the public agent API (the zmq include is
    # needed only because the shared bounceTestNixlNode.h helpers pull in
    # ZmqControlChannel.h at compile time).
    add_tllm_gtest(
      bounceAgentE2ETest
      OWNER
      executor
      SOURCES
      bounceAgentE2ETest.cpp
      LINK_LIBRARIES
      tllm::executor_transfer_agent
      tllm::common_cuda
      tensorrt_llm_nixl_wrapper
      NIXL::nixl
      ${NIXL_BUILD_LIBRARY}
      Threads::Threads
      LABELS
      "unit;executor;bounce;integration;gpu")
    target_include_directories(bounceAgentE2ETest PRIVATE ${BOUNCE_TEST_INCS})
  endif()
endif()
