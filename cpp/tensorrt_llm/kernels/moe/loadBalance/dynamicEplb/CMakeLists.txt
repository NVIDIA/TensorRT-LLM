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

set(DYNAMIC_EPLB_ARCHS)
foreach(ARCH 100 103 107)
  if("${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    list(APPEND DYNAMIC_EPLB_ARCHS ${ARCH})
  endif()
endforeach()

set(DYNAMIC_EPLB_SOURCES)
set(DYNAMIC_EPLB_DEFINITIONS)

if(DYNAMIC_EPLB_ARCHS)
  list(APPEND DYNAMIC_EPLB_SOURCES moeRebalanceHaloQ.cu)
else()
  list(APPEND DYNAMIC_EPLB_SOURCES moeRebalancePlaceholder.cpp)
  list(APPEND DYNAMIC_EPLB_DEFINITIONS
       TRTLLM_DYNAMIC_EPLB_PLACEHOLDER_HALO_Q)
  message(
    STATUS
      "Dynamic EPLB HALO-Q disabled: none of SM100, SM103, or SM107 was requested")
endif()

if(DYNAMIC_EPLB_ARCHS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
  list(APPEND DYNAMIC_EPLB_SOURCES moeRebalanceTma.cu)
else()
  list(APPEND DYNAMIC_EPLB_SOURCES moeRebalancePlaceholder.cpp)
  list(APPEND DYNAMIC_EPLB_DEFINITIONS TRTLLM_DYNAMIC_EPLB_PLACEHOLDER_TMA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
    message(
      STATUS
        "Dynamic EPLB TMA disabled: CUDA 13.1 or newer is required (found ${CMAKE_CUDA_COMPILER_VERSION})")
  else()
    message(
      STATUS
        "Dynamic EPLB TMA disabled: none of SM100, SM103, or SM107 was requested")
  endif()
endif()

list(REMOVE_DUPLICATES DYNAMIC_EPLB_SOURCES)
add_library(moe_dynamic_eplb_src OBJECT ${DYNAMIC_EPLB_SOURCES})

if(DYNAMIC_EPLB_ARCHS)
  # These kernels use SM100-family PTX. The family target avoids
  # inheriting the generic kernels_src SM80/SM90 compilation passes.
  set_cuda_architectures(moe_dynamic_eplb_src 100f)
endif()

if(DYNAMIC_EPLB_DEFINITIONS)
  target_compile_definitions(moe_dynamic_eplb_src
                             PRIVATE ${DYNAMIC_EPLB_DEFINITIONS})
endif()

set_property(TARGET moe_dynamic_eplb_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET moe_dynamic_eplb_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
