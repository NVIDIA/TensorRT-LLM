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
set(USERBUFFERS_SRCS ub_interface.cpp userbuffersManager.cpp)
if(ENABLE_MULTI_DEVICE)
  list(APPEND USERBUFFERS_SRCS ipcsocket.cpp ub_allocator.cpp
       userbuffers-host.cpp userbuffers.cu)
endif()

add_library(userbuffers_src STATIC ${USERBUFFERS_SRCS})
add_library(tllm::kernel_userbuffers ALIAS userbuffers_src)
target_link_libraries(
  userbuffers_src
  PUBLIC tllm::common_cuda tllm::runtime_distributed CUDA::cuda_driver
  PRIVATE tllm::common_environment Threads::Threads)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(userbuffers_src PUBLIC ${NCCL_LIB})
endif()
target_compile_features(userbuffers_src PUBLIC cxx_std_20)
target_compile_options(userbuffers_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>)
set_target_properties(
  userbuffers_src
  PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(userbuffers_src 89)

message(STATUS "UserBuffer ENABLE_MULTI_DEVICE is ${ENABLE_MULTI_DEVICE}")
