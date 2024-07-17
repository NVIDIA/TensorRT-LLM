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

# Exclude files in the cutlass_kernels and decoderMaskedMultiheadAttention
# folder
list(FILTER SRC_CPP EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")

function(filter_cuda_archs ARCH SOURCES_VAR)
  if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     AND NOT "${ARCH}-real" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     AND NOT "all" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     AND NOT "all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    set(FILTER_REGEX ".*_sm(_)?${ARCH}[.]cubin[.]cpp")
    list(APPEND SOURCES ${${SOURCES_VAR}})
    list(APPEND SOURCES_FILTERED ${SOURCES})
    list(FILTER SOURCES_FILTERED INCLUDE REGEX "${FILTER_REGEX}")
    list(LENGTH SOURCES_FILTERED SOURCES_FILTERED_LEN)
    message(
      STATUS
        "Excluding ${SOURCES_FILTERED_LEN} cubins for SM ${ARCH} from ${CMAKE_CURRENT_SOURCE_DIR}"
    )
    foreach(filtered_item ${SOURCES_FILTERED})
      message(VERBOSE "- ${filtered_item}")
    endforeach()
    list(FILTER SOURCES EXCLUDE REGEX "${FILTER_REGEX}")
    set(${SOURCES_VAR}
        "${SOURCES}"
        PARENT_SCOPE)
    if(SOURCES_FILTERED_LEN GREATER 0)
      add_compile_definitions("EXCLUDE_SM_${ARCH}")
    endif()
  endif()
endfunction()

filter_cuda_archs("70" SRC_CPP)
filter_cuda_archs("80" SRC_CPP)
filter_cuda_archs("86" SRC_CPP)
filter_cuda_archs("89" SRC_CPP)
filter_cuda_archs("90" SRC_CPP)

if(ENABLE_MULTI_DEVICE EQUAL 0)
  list(FILTER SRC_CU EXCLUDE REGEX "customAllReduceKernels*.*cu$")
endif()

add_library(kernels_src STATIC ${SRC_CPP} ${SRC_CU})
set_property(TARGET kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_subdirectory(cutlass_kernels)
add_subdirectory(decoderMaskedMultiheadAttention)
