#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
list(FILTER SRC_CU EXCLUDE REGEX "fmha_v2_cu/.*")

add_library(context_attention_src OBJECT)

filter_source_cuda_architectures(
  SOURCE_LIST SRC_CPP
  ARCHS 80 86 89 90 100 120
  TARGET context_attention_src
  IMPLICIT_FAMILY)

target_sources(context_attention_src PRIVATE ${SRC_CPP} ${SRC_CU})
target_compile_definitions(context_attention_src PRIVATE USE_DEMO_BERT_PARAMS=1
                                                         GENERATE_CUBIN=1)
set_target_properties(
  context_attention_src PROPERTIES POSITION_INDEPENDENT_CODE ON
                                   CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(context_attention_src PUBLIC trtllm_gen_fmha_interface)

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

  file(GLOB arch_files "fmha_v2_cu/*_sm${arch}.cu")
  if(arch_files)
    set(TARGET_NAME _context_attention_kernels_${arch})
    add_library(${TARGET_NAME} OBJECT ${arch_files})
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_DEMO_BERT_PARAMS=1
                                                      GENERATE_CUBIN=1)
    set_target_properties(
      ${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
                                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(
      ${TARGET_NAME}
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/src/
              ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/generated/)
    set_cuda_architectures(${TARGET_NAME} ${TARGET_ARCH})
    target_sources(context_attention_src
                   PUBLIC $<TARGET_OBJECTS:${TARGET_NAME}>)
  endif()
endforeach()
