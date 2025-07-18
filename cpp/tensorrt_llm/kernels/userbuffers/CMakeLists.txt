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
if(NOT ENABLE_MULTI_DEVICE)
  add_library(userbuffers_src OBJECT ub_interface.cpp userbuffersManager.cpp)
else()
  file(GLOB_RECURSE SRC_CPP *.cpp)
  file(GLOB_RECURSE SRC_CU *.cu)
  add_library(userbuffers_src OBJECT ${SRC_CPP} ${SRC_CU})
endif()

target_compile_options(userbuffers_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>)

set_property(TARGET userbuffers_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET userbuffers_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(userbuffers_src 89)

message(STATUS "UserBuffer ENABLE_MULTI_DEVICE is ${ENABLE_MULTI_DEVICE}")
