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

# Exclude files in the cutlass_kernels, decoderMaskedMultiheadAttention and
# selectiveScan trtllmGenKernels folder
list(FILTER SRC_CPP EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CU EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CU EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "fusedLayernormKernels/.*")

function(filter_cuda_archs ARCH SOURCES_VAR)
  if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    set(FILTER_REGEX
        ".*_sm(_)?${ARCH}[.]cubin[.]cpp|^.*Sm(_)?${ARCH}.*cubin.cpp$|.*_sm(_)?${ARCH}[.]cu|^.*Sm(_)?${ARCH}.*cu$"
    )
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
    add_compile_definitions("EXCLUDE_SM_${ARCH}")
  endif()
endfunction()

if(NOT ENABLE_MULTI_DEVICE)
  list(FILTER SRC_CU EXCLUDE REGEX "customAllReduceKernels*.*cu$")
endif()

add_library(kernels_src STATIC ${SRC_CPP} ${SRC_CU})
set_property(TARGET kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(
  kernels_src
  PUBLIC
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)
add_cuda_architectures(kernels_src 89)

add_subdirectory(cutlass_kernels)
add_subdirectory(flashMLA)
add_subdirectory(contextFusedMultiHeadAttention)
add_subdirectory(decoderMaskedMultiheadAttention)
add_subdirectory(selectiveScan)
add_subdirectory(userbuffers)
add_subdirectory(trtllmGenKernels)
add_subdirectory(fusedLayernormKernels)
add_subdirectory(groupRmsNormKernels)
add_subdirectory(llama4MinLatencyKernels)
add_subdirectory(dsv3MinLatencyKernels)
add_subdirectory(causalConv1d)
