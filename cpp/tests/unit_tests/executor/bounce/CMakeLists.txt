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
  "unit;executor;bounce;gpu"
  REQUIRES_GPU)
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
  "unit;executor;bounce;gpu"
  REQUIRES_GPU)
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
  "unit;executor;bounce;gpu"
  REQUIRES_GPU)

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
# (transport, failure-path, e2e) lives here, sharing one source set + the
# bounceTestNixlNode.h helpers. ----
if(NIXL_ROOT)
  # The reactor tests also need zmq for the control plane (cppzmq is fetched
  # when NIXL is on); all of them share this source set + include/link sets.
  # Same gate as the zmq-only tier above.
  if(BOUNCE_CPPZMQ_AVAILABLE)
    set(BOUNCE_REACTOR_SRCS ${BOUNCE_DIR}/BounceTransport.cpp)
    set(BOUNCE_TSAN_SRCS
        ${BOUNCE_DIR}/BounceTransport.cpp
        ${BOUNCE_DIR}/BounceTransferPlan.cpp
        ${BOUNCE_DIR}/BuddyAllocator.cpp
        ${BOUNCE_DIR}/CreditScheduler.cpp
        ${BOUNCE_DIR}/BounceMessage.cpp
        ${BOUNCE_DIR}/BounceArena.cpp
        ${BOUNCE_DIR}/ExecPool.cpp
        ${BOUNCE_DIR}/GatherScatterKernel.cu
        ${BOUNCE_DIR}/ZmqControlChannel.cpp)
    set(BOUNCE_TEST_INCS ${CMAKE_BINARY_DIR}/_deps/cppzmq-src)
    set(BOUNCE_TEST_LIBS tensorrt_llm_nixl_wrapper NIXL::nixl
                         ${NIXL_BUILD_LIBRARY} PkgConfig::TLLM_BOUNCE_ZMQ)

    # One reactor test target: the bounce sources compiled in, plus the shared
    # include/link sets.
    function(add_bounce_reactor_test name src)
      cmake_parse_arguments(ARG "INSTRUMENTED" "" "" ${ARGN})
      add_tllm_gtest(${name} SOURCES ${src} FULL_STACK LABELS "unit;full-stack")
      if(ARG_INSTRUMENTED)
        target_sources(${name} PRIVATE ${BOUNCE_TSAN_SRCS})
      else()
        target_sources(${name} PRIVATE ${BOUNCE_REACTOR_SRCS})
        target_link_libraries(
          ${name} PRIVATE tllm::executor_bounce_core tllm::executor_bounce_cuda
                          tllm::executor_bounce_arena)
      endif()
      target_include_directories(${name} PRIVATE ${BOUNCE_TEST_INCS})
      target_link_libraries(${name} PRIVATE ${BOUNCE_TEST_LIBS})
    endfunction()

    # End-to-end transport over real RDMA (gather -> RDMA write -> scatter +
    # credit recycling), byte-exact.
    add_bounce_reactor_test(bounceTransportTest bounceTransportTest.cpp)

    # Boundary, failure-path, and concurrency tests.
    add_bounce_reactor_test(bounceTransportFailureTest
                            bounceTransportFailureTest.cpp)

    # Optional ThreadSanitizer variant of the reactor concurrency test. The
    # bounce sources + the test are instrumented; the shared libtensorrt_llm
    # (linked for the logger) and CUDA runtime are left uninstrumented (TSan
    # partial-instrumentation mode). Enable with -DBOUNCE_TSAN=ON and run with a
    # suppressions file that ignores CUDA/driver/libtllm frames.
    option(BOUNCE_TSAN
           "Build a ThreadSanitizer variant of the bounce reactor test" OFF)
    if(BOUNCE_TSAN)
      add_bounce_reactor_test(bounceTransportTsanTest
                              bounceTransportFailureTest.cpp INSTRUMENTED)
      target_compile_options(
        bounceTransportTsanTest
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread -g>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=thread>)
      target_link_options(bounceTransportTsanTest PRIVATE -fsanitize=thread)
    endif()

    # Production-path e2e: bounce engaged via
    # NixlTransferAgent::submitTransferRequests. Bounce lives inside the
    # wrapper; this test only uses the public agent API (the zmq include is
    # needed only because the shared bounceTestNixlNode.h helpers pull in
    # ZmqControlChannel.h at compile time).
    add_tllm_gtest(bounceAgentE2ETest SOURCES bounceAgentE2ETest.cpp FULL_STACK
                   LABELS "unit;full-stack")
    target_include_directories(bounceAgentE2ETest PRIVATE ${BOUNCE_TEST_INCS})
    target_link_libraries(
      bounceAgentE2ETest PRIVATE tensorrt_llm_nixl_wrapper NIXL::nixl
                                 ${NIXL_BUILD_LIBRARY})
  endif()
endif()
