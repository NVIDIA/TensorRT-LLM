#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
file(GLOB SRCS *.cpp)
file(GLOB CU_SRCS *.cu)

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

add_library(common_src OBJECT ${SRCS} ${CU_SRCS})
add_cuda_architectures(common_src 89)
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(common_src PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()
