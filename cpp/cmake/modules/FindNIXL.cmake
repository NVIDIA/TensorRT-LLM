#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

# Check if NIXL is already found
if(NIXL_FOUND)
  # If NIXL is already found, exit the script early
  return()
endif()

find_package(ucx REQUIRED)

# Set default NIXL_ROOT if not provided
if(NOT NIXL_ROOT)
  set(NIXL_ROOT
      "/opt/nvidia/nvda_nixl"
      CACHE PATH "NIXL installation directory" FORCE)
  message(STATUS "NIXL_ROOT not set, using default: ${NIXL_ROOT}")
else()
  message(STATUS "Using provided NIXL_ROOT: ${NIXL_ROOT}")
endif()

find_path(NIXL_INCLUDE_DIR nixl.h HINTS ${NIXL_ROOT}/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(NIXL_TARGET_ARCH "x86_64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(NIXL_TARGET_ARCH "aarch64-linux-gnu")
endif()

find_library(NIXL_LIBRARY nixl HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}
                                     ${NIXL_ROOT}/lib64)
find_library(NIXL_BUILD_LIBRARY nixl_build
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH} ${NIXL_ROOT}/lib64)
find_library(SERDES_LIBRARY serdes HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}
                                         ${NIXL_ROOT}/lib64)
find_library(
  UCX_BACKEND_LIBRARY plugin_UCX
  HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH}/plugins ${NIXL_ROOT}/lib64/plugins)
find_library(UCX_UTILS_LIBRARY ucx_utils
             HINTS ${NIXL_ROOT}/lib/${NIXL_TARGET_ARCH} ${NIXL_ROOT}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NIXL
  FOUND_VAR NIXL_FOUND
  REQUIRED_VARS NIXL_INCLUDE_DIR NIXL_LIBRARY NIXL_BUILD_LIBRARY SERDES_LIBRARY)

# Re-attempt to find NIXL after installation
find_package_handle_standard_args(
  NIXL
  FOUND_VAR NIXL_FOUND
  REQUIRED_VARS NIXL_INCLUDE_DIR NIXL_LIBRARY NIXL_BUILD_LIBRARY SERDES_LIBRARY)

# Set up the NIXL target if found
if(NIXL_FOUND)
  if(NOT TARGET NIXL::nixl)
    add_library(NIXL::nixl SHARED IMPORTED)
    set_target_properties(
      NIXL::nixl
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${NIXL_INCLUDE_DIR}
                 IMPORTED_LOCATION ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY}
                                                   ${SERDES_LIBRARY})
  endif()
else()
  message(STATUS "NIXL_LIBRARY: ${NIXL_LIBRARY}")
  message(STATUS "NIXL_BUILD_LIBRARY: ${NIXL_BUILD_LIBRARY}")
  message(STATUS "SERDES_LIBRARY: ${SERDES_LIBRARY}")
  unset(NIXL_ROOT CACHE)
endif()
