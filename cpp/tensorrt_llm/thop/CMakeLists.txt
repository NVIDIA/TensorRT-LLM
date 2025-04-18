# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
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

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(th_utils STATIC thUtils.cpp)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(th_utils PUBLIC ${TORCH_LIBRARIES} ${CUBLAS_LIB}
                                      ${CURAND_LIB})

add_library(
  th_common SHARED
  allgatherOp.cpp
  allreduceOp.cpp
  attentionOp.cpp
  convertSpecDecodingMaskToPackedMaskOp.cpp
  cutlassScaledMM.cpp
  cublasScaledMM.cpp
  deepseekAllreduceFusionOp.cpp
  dynamicDecodeOp.cpp
  fmhaPackMaskOp.cpp
  fp8Op.cpp
  fp4Op.cpp
  fp4Gemm.cpp
  fp4GemmTrtllmGen.cpp
  fp8BatchedGemmTrtllmGen.cpp
  fp4Quantize.cpp
  fp4BatchedQuantize.cpp
  fp8BlockScalingGemm.cpp
  fp8Quantize.cpp
  fusedTopkSoftmax.cpp
  gatherTreeOp.cpp
  logitsBitmaskOp.cpp
  mambaConv1dOp.cpp
  moeOp.cpp
  moeCommOp.cpp
  fp8BlockScaleMoe.cpp
  noAuxTcOp.cpp
  ncclCommunicatorOp.cpp
  parallelDecodeKVCacheUpdateOp.cpp
  redrafterCurandOp.cpp
  reducescatterOp.cpp
  relativeAttentionBiasOp.cpp
  selectiveScanOp.cpp
  userbuffersFinalizeOp.cpp
  userbuffersTensor.cpp
  weightOnlyQuantOp.cpp
  mtpOp.cpp
  loraOp.cpp)
set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(th_common PRIVATE ${TORCH_LIBRARIES} th_utils
                                        ${Python3_LIBRARIES} ${SHARED_TARGET})

if(ENABLE_MULTI_DEVICE)
  target_include_directories(th_common PUBLIC ${MPI_C_INCLUDE_DIRS})
  target_link_libraries(th_common PRIVATE ${MPI_C_LIBRARIES} ${NCCL_LIB}
                                          CUDA::nvml)
endif()

if(NOT WIN32)
  set_target_properties(
    th_common
    PROPERTIES LINK_FLAGS
               "-Wl,-rpath='$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
else()
  target_link_libraries(th_common PRIVATE context_attention_src)
endif()
