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
add_gtest(bounceTransferPlanTest bounceTransferPlanTest.cpp)
target_sources(bounceTransferPlanTest
               PRIVATE ${BOUNCE_DIR}/BounceTransferPlan.cpp)
add_gtest(bounceMessageCodecTest bounceMessageCodecTest.cpp)
target_sources(bounceMessageCodecTest PRIVATE ${BOUNCE_DIR}/BounceMessage.cpp)
# Header-only: env-var parsing (byte-size suffixes, garbage fallback).
add_gtest(bounceConfigTest bounceConfigTest.cpp)
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

# ---- Control-plane unit test — needs only zmq (+ the fetched cppzmq header),
# no NIXL. ZMQ_FOUND is reused by the real-NIXL block below. ----
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
endif()
