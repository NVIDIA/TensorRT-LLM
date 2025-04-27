#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
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

set(NIXL_ROOT "/opt/nvidia/nvda_nixl")

# calculate TARGET_ARCH
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(NIXL_TARGET_ARCH "x86_64-linux-gnu")
else()
  message(FATAL_ERROR "Unsupported system with NIXL: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_path(NIXL_INCLUDE_DIR nixl.h HINTS ${NIXL_ROOT}/include)

find_library(NIXL_LIBRARY nixl HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH})
find_library(NIXL_BUILD_LIBRARY nixl_build
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH})
find_library(SERDES_LIBRARY serdes HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH})
find_library(UCX_BACKEND_LIBRARY plugin_UCX
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}/plugins)
find_library(UCX_UTILS_LIBRARY ucx_utils
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH})
find_library(GDS_BACKEND_LIBRARY plugin_GDS
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}/plugins)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NIXL
  FOUND_VAR NIXL_FOUND
  REQUIRED_VARS
    NIXL_INCLUDE_DIR
    NIXL_LIBRARY
    NIXL_BUILD_LIBRARY
    SERDES_LIBRARY
    UCX_BACKEND_LIBRARY
    UCX_UTILS_LIBRARY
    GDS_BACKEND_LIBRARY)

if(NIXL_FOUND)
  if(NOT TARGET NIXL::nixl)
    add_library(NIXL::nixl SHARED IMPORTED)
    set_target_properties(
      NIXL::nixl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NIXL_INCLUDE_DIR}"
                            IMPORTED_LOCATION "${NIXL_LIBRARY}")
  endif()
else()
  message(STATUS "NIXL_INCLUDE_DIR: ${NIXL_INCLUDE_DIR}")
  message(STATUS "NIXL_LIBRARY: ${NIXL_LIBRARY}")
  message(STATUS "NIXL_BUILD_LIBRARY: ${NIXL_BUILD_LIBRARY}")
  message(STATUS "SERDES_LIBRARY: ${SERDES_LIBRARY}")
  message(STATUS "UCX_BACKEND_LIBRARY: ${UCX_BACKEND_LIBRARY}")
  message(STATUS "UCX_UTILS_LIBRARY: ${UCX_UTILS_LIBRARY}")
  message(STATUS "GDS_BACKEND_LIBRARY: ${GDS_BACKEND_LIBRARY}")
  message(FATAL_ERROR "NIXL not found. Please install NIXL or set NIXL_ROOT")
endif()
