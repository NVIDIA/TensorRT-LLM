#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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
#

# Directory holding the generated FMHA sources (fmha_v2_cu/*.cu and
# cubin/fmha_cubin.{h,cpp}). scripts/build_wheel.py points this at an
# out-of-tree location when --out-of-tree is set; by default the generated files
# live in this directory as before.
set(TRTLLM_FMHA_GEN_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE PATH "Directory containing generated FMHA sources")

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
list(FILTER SRC_CU EXCLUDE REGEX "fmha_v2_cu/.*")
# fmha_cubin.cpp is generated; always take it from TRTLLM_FMHA_GEN_DIR so an
# out-of-tree copy is used when configured and a stale in-tree copy is never
# compiled alongside it.
list(FILTER SRC_CPP EXCLUDE REGEX "cubin/fmha_cubin\\.cpp$")
if(EXISTS "${TRTLLM_FMHA_GEN_DIR}/cubin/fmha_cubin.cpp")
  list(APPEND SRC_CPP "${TRTLLM_FMHA_GEN_DIR}/cubin/fmha_cubin.cpp")
else()
  message(
    FATAL_ERROR
      "Generated FMHA source ${TRTLLM_FMHA_GEN_DIR}/cubin/fmha_cubin.cpp is "
      "missing. It is produced by the FMHA generation step "
      "(scripts/build_wheel.py runs it before CMake); a missing file means "
      "generation did not complete. Regenerate before configuring.")
endif()
# The skip_softmax sm_120/sm_121 warp-specialized FMHA is a hand-written TU that
# only compiles for the sm_120 family; it is added explicitly to the sm_120 arch
# target below, not to the all-arch source list.
list(FILTER SRC_CU EXCLUDE REGEX "skip_softmax_sm120/.*")

add_library(context_attention_src OBJECT)

filter_source_cuda_architectures(
  SOURCE_LIST SRC_CPP
  ARCHS 80 86 89 90 100 120
  IMPLICIT_FAMILY)

target_sources(context_attention_src PRIVATE ${SRC_CPP} ${SRC_CU})
# For #include "cubin/fmha_cubin.h" when the generated header is out of tree.
target_include_directories(context_attention_src
                           PRIVATE "${TRTLLM_FMHA_GEN_DIR}")
target_compile_definitions(context_attention_src PRIVATE USE_DEMO_BERT_PARAMS=1
                                                         GENERATE_CUBIN=1)
set_target_properties(
  context_attention_src PROPERTIES POSITION_INDEPENDENT_CODE ON
                                   CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Embed cubins via INCBIN. Replaces the bin2c-style hex-array .cpp stubs that
# previously lived under cubin/. Symbol convention is cubin_<stem>_cu_cubin
# because the generator (cpp/kernels/fmha_v2/setup.py) builds names from
# `<fname>.replace('.', '_')` where `<fname>` includes the trailing `.cu`.
include(tllm_cubin_archive)
tllm_add_cubin_archive_sources(
  context_attention_src "${CMAKE_CURRENT_SOURCE_DIR}/cubin"
  SYMBOL_PREFIX cubin_
  SYMBOL_SUFFIX _cu_cubin
  ARCHS 80 86 89 90 100 120
  IMPLICIT_FAMILY)

foreach(arch IN ITEMS 80 86 89 90 100 120)
  set(ARCH_ENABLED FALSE)
  set(TARGET_ARCH "")
  if(${arch} GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY})
    if("${arch}f" IN_LIST CMAKE_CUDA_ARCHITECTURES_FAMILIES)
      set(ARCH_ENABLED TRUE)
      set(TARGET_ARCH "${arch}f")
    endif()
  else()
    if("${arch}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      set(ARCH_ENABLED TRUE)
      set(TARGET_ARCH "${arch}")
    endif()
  endif()

  if(NOT ${ARCH_ENABLED})
    continue()
  endif()

  file(GLOB arch_files "${TRTLLM_FMHA_GEN_DIR}/fmha_v2_cu/*_sm${arch}.cu")
  # Compile the hand-written skip_softmax warp-specialized FMHA into the sm_120
  # family target. It uses sm_120-only TMA + sync-MMA, so it must never be
  # compiled for other architectures.
  if(${arch} EQUAL 120)
    list(
      APPEND
      arch_files
      "${CMAKE_CURRENT_SOURCE_DIR}/skip_softmax_sm120/fused_multihead_flash_attention_ws_sm120.cu"
    )
  endif()
  if(arch_files)
    set(TARGET_NAME _context_attention_kernels_${arch})
    add_library(${TARGET_NAME} OBJECT ${arch_files})
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_DEMO_BERT_PARAMS=1
                                                      GENERATE_CUBIN=1)
    # Let the all-arch dispatch TU (fused_multihead_attention_v2.cpp) know the
    # skip_softmax bridge symbols are available so it can reference them without
    # an undefined-symbol error on builds that exclude sm_120.
    if(${arch} EQUAL 120)
      target_compile_definitions(context_attention_src
                                 PRIVATE TLLM_ENABLE_SKIP_SOFTMAX_SM120=1)
    endif()
    set_target_properties(
      ${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
                                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    # CMAKE_CURRENT_SOURCE_DIR resolves the generated kernels' #include
    # "fused_multihead_attention_common.h" when they live out of tree (and is
    # where includer-relative lookup lands in-tree).
    target_include_directories(
      ${TARGET_NAME}
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
              ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/src/
              ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/generated/)
    set_cuda_architectures(${TARGET_NAME} ${TARGET_ARCH})
    target_sources(context_attention_src
                   PUBLIC $<TARGET_OBJECTS:${TARGET_NAME}>)
  else()
    # arch is enabled (disabled archs already `continue`d above) but the
    # generation step produced no *_sm${arch}.cu kernels for it. Fail rather
    # than silently building without this architecture's FMHA kernels, which
    # would surface only as a missing-kernel failure at runtime. (arch 120
    # always has the hand-written skip_softmax TU appended above, so this branch
    # covers the generated-only architectures.)
    message(
      FATAL_ERROR
        "No generated FMHA kernels (*_sm${arch}.cu) found in "
        "${TRTLLM_FMHA_GEN_DIR}/fmha_v2_cu for enabled architecture ${arch}. "
        "The FMHA generation step did not produce sources for this arch.")
  endif()
endforeach()
