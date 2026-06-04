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

# BLAKE3 C sources — needed by blockRadixTree.cpp. Always include the portable
# dispatch core.
set(BLAKE3_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_dispatch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_portable.c)
# x86/x86-64 SIMD back-ends
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  set(BLAKE3_SSE2
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_sse2.c)
  set(BLAKE3_SSE41
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_sse41.c)
  set(BLAKE3_AVX2
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_avx2.c)
  set(BLAKE3_AVX512
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_avx512.c)
  set_source_files_properties(${BLAKE3_SSE2} PROPERTIES COMPILE_FLAGS "-msse2")
  set_source_files_properties(${BLAKE3_SSE41} PROPERTIES COMPILE_FLAGS
                                                         "-msse4.1 -mssse3")
  set_source_files_properties(${BLAKE3_AVX2} PROPERTIES COMPILE_FLAGS "-mavx2")
  set_source_files_properties(${BLAKE3_AVX512}
                              PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl")
  list(APPEND BLAKE3_SRCS ${BLAKE3_SSE2} ${BLAKE3_SSE41} ${BLAKE3_AVX2}
       ${BLAKE3_AVX512})
endif()
# ARM NEON back-end (AArch64 and 32-bit ARM with NEON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|armv7|armv8")
  list(APPEND BLAKE3_SRCS
       ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/blake3/blake3_neon.c)
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
    kv_cache_manager_v2/page.cpp
    kv_cache_manager_v2/storageManager.cpp
    kv_cache_manager_v2/introspection.cpp
    kv_cache_manager_v2/kvCache.cpp
    kv_cache_manager_v2/kvCacheManager.cpp
    ${BLAKE3_SRCS})

# Headers and sources are co-located; the include root is batch_manager/ so that
# #include "kv_cache_manager_v2/common.h" resolves correctly. NOTE: this file is
# include()'d (not add_subdirectory()'d), so CMAKE_CURRENT_SOURCE_DIR is already
# batch_manager/ — no PARENT_SCOPE needed.
set(KV_CACHE_MANAGER_V2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
