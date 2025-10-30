# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

if(NIXL_ROOT)
  find_package(NIXL REQUIRED)
  # Check if all required packages were found
  if(NOT NIXL_FOUND)
    message(
      FATAL_ERROR "NIXL not found. Please install NIXL or set `NIXL_ROOT`.")
  endif()

  # Define the NIXL wrapper target
  set(NIXL_WRAPPER_TARGET "tensorrt_llm_nixl_wrapper")

  # Add the NIXL wrapper target
  add_library(${NIXL_WRAPPER_TARGET} SHARED transferAgent.cpp)
  target_compile_options(${NIXL_WRAPPER_TARGET} PRIVATE -Wno-error)

  # Add include directories
  target_include_directories(${NIXL_WRAPPER_TARGET} PRIVATE NIXL::nixl)

  # Link against all NIXL libraries
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE NIXL::nixl)

  # Link against CUDA
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE CUDA::cudart)

endif()
