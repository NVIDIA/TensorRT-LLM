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

add_library(flash_mla_src OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET flash_mla_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET flash_mla_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Get all targeted architectures from parent CMake
if(CMAKE_CUDA_ARCHITECTURES)
  # Create a list for FlashMLA's supported architectures
  set(FLASH_MLA_CUDA_ARCHS "")

  # Only include architectures 90 and above
  foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
    # Extract the numeric part before any dash
    string(REGEX MATCH "^([0-9]+)" arch_num "${arch}")

    # Only add architectures 90 and above
    if(arch_num GREATER_EQUAL 90)
      list(APPEND FLASH_MLA_CUDA_ARCHS "${arch}")
    endif()
  endforeach()

  # Set specific architectures for this target only
  set_property(TARGET flash_mla_src PROPERTY CUDA_ARCHITECTURES
                                             ${FLASH_MLA_CUDA_ARCHS})

  # Log which architectures we're building for
  message(STATUS "Building FlashMLA for architectures: ${FLASH_MLA_CUDA_ARCHS}")
endif()
