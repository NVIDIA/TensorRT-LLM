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

add_subdirectory(cutlass_kernels)
add_subdirectory(cuteDslKernels)
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

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

# Exclude files in the cutlass_kernels, decoderMaskedMultiheadAttention and
# selectiveScan trtllmGenKernels folder
list(FILTER SRC_CPP EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "cuteDslKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cuteDslKernels/.*")
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
target_link_libraries(kernels_src PUBLIC trtllm_gen_fmha_interface)
add_cuda_architectures(kernels_src 89)
