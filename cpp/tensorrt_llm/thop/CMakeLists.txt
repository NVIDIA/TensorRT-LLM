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
add_library(th_utils STATIC thUtils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(th_utils PUBLIC ${TORCH_LIBRARIES} ${CUBLAS_LIB}
                                      ${CURAND_LIB})

add_library(
  th_common SHARED
  convertSpecDecodingMaskToPackedMaskOp.cpp
  dynamicDecodeOp.cpp
  fmhaPackMaskOp.cpp
  fp8Op.cpp
  gatherTreeOp.cpp
  ncclCommunicatorOp.cpp
  parallelDecodeKVCacheUpdateOp.cpp
  redrafterCurandOp.cpp
  relativeAttentionBiasOp.cpp
  weightOnlyQuantOp.cpp)
set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(th_common PRIVATE ${TORCH_LIBRARIES} th_utils
                                        ${Python3_LIBRARIES} ${SHARED_TARGET})

if(NOT WIN32)
  set_target_properties(
    th_common
    PROPERTIES LINK_FLAGS
               "-Wl,-rpath='$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()
