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

# Sources for the C++ KVCacheManagerV2 implementation. These are added to the
# tensorrt_llm_batch_manager_static target by the parent CMakeLists.txt.

# Bitcoin Core SHA-256 (vendored, MIT; see 3rdparty/sha256/README.md) — needed
# by blockRadixTree.cpp. Slimmed to the single-block CSHA256 path: portable
# scalar core + runtime dispatch (sha256.cpp) plus a per-architecture hardware
# transform. Plain C++17; the include root 3rdparty/sha256 is added on the
# batch_manager target so "sha256.h" resolves.
set(SHA256_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/sha256)
set(SHA256_SRCS ${SHA256_DIR}/sha256.cpp)
# x86/x86-64: Intel SHA-NI transform.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  set(SHA256_X86_SHANI ${SHA256_DIR}/sha256_x86_shani.cpp)
  set_source_files_properties(
    ${SHA256_X86_SHANI} PROPERTIES COMPILE_OPTIONS "-msse4.1;-msse4;-msha")
  list(APPEND SHA256_SRCS ${SHA256_X86_SHANI})
endif()
# AArch64: ARMv8 crypto-extension (SHA-256) transform.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(SHA256_ARM_SHANI ${SHA256_DIR}/sha256_arm_shani.cpp)
  set_source_files_properties(
    ${SHA256_ARM_SHANI} PROPERTIES COMPILE_OPTIONS "-march=armv8-a+crypto")
  list(APPEND SHA256_SRCS ${SHA256_ARM_SHANI})
endif()

set(KV_CACHE_MANAGER_V2_SRCS
    kv_cache_manager_v2/common.cpp
    kv_cache_manager_v2/config.cpp
    kv_cache_manager_v2/lifeCycleRegistry.cpp
    kv_cache_manager_v2/utils/cudaEvent.cpp
    kv_cache_manager_v2/utils/hostMem.cpp
    kv_cache_manager_v2/cudaVirtMem.cpp
    kv_cache_manager_v2/storage/config.cpp
    kv_cache_manager_v2/storage/core.cpp
    kv_cache_manager_v2/evictionController.cpp
    kv_cache_manager_v2/copyEngine.cpp
    kv_cache_manager_v2/blockRadixTree.cpp
    kv_cache_manager_v2/eventManager.cpp
    kv_cache_manager_v2/page.cpp
    kv_cache_manager_v2/storageManager.cpp
    kv_cache_manager_v2/introspection.cpp
    kv_cache_manager_v2/kvCache.cpp
    kv_cache_manager_v2/kvCacheManager.cpp
    ${SHA256_SRCS})

# Headers and sources are co-located; the include root is batch_manager/ so that
# #include "kv_cache_manager_v2/common.h" resolves correctly. NOTE: this file is
# include()'d (not add_subdirectory()'d), so CMAKE_CURRENT_SOURCE_DIR is already
# batch_manager/ — no PARENT_SCOPE needed.
set(KV_CACHE_MANAGER_V2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
