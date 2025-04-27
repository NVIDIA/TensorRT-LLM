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

# Check if NIXL is already found
if(NIXL_FOUND)
  # If NIXL is already found, exit the script early
  return()
endif()

set(NIXL_ROOT "/opt/nvidia/nvda_nixl")

# calculate TARGET_ARCH
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(NIXL_TARGET_ARCH "x86_64-linux-gnu")
else()
  message(FATAL_ERROR "Unsupported system with NIXL: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# NIXL not found, attempt to build and install
message(STATUS "NIXL not found. Attempting to build and install NIXL.")

# Define the build directory for NIXL
set(NIXL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/nixl_build)

# Create the build directory
file(MAKE_DIRECTORY ${NIXL_BUILD_DIR})
message(STATUS "NIXL_BUILD_DIR: ${NIXL_BUILD_DIR}")

# Check if the UCX path exists
if(NOT EXISTS "${UCX_PATH}")
  message(FATAL_ERROR "UCX path does not exist: ${UCX_PATH}")
endif()
message(STATUS "UCX path exists: ${UCX_PATH}")

# Run the Meson setup and build commands
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
execute_process(
  COMMAND ${MESON_EXECUTABLE} setup ${NIXL_BUILD_DIR} -Ducx_path=${UCX_PATH}
  WORKING_DIRECTORY ${NIXL_SOURCE_DIR}
  RESULT_VARIABLE MESON_SETUP_RESULT
  OUTPUT_VARIABLE MESON_SETUP_OUTPUT
  ERROR_VARIABLE MESON_SETUP_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
message(STATUS "Meson setup output: ${MESON_SETUP_OUTPUT}")
message(STATUS "Meson setup error: ${MESON_SETUP_ERROR}")
if(NOT MESON_SETUP_RESULT EQUAL 0)
  message(
    FATAL_ERROR "Meson setup failed with error code ${MESON_SETUP_RESULT}")
endif()

# Build and install NIXL
execute_process(COMMAND ${NINJA_EXECUTABLE} WORKING_DIRECTORY ${NIXL_BUILD_DIR})

execute_process(COMMAND ${NINJA_EXECUTABLE} install
                WORKING_DIRECTORY ${NIXL_BUILD_DIR})

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

# Re-attempt to find NIXL after installation
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

# Set up the NIXL target if found
if(NIXL_FOUND)
  if(NOT TARGET NIXL::nixl)
    add_library(NIXL::nixl SHARED IMPORTED)
    set_target_properties(
      NIXL::nixl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NIXL_INCLUDE_DIR}"
                            IMPORTED_LOCATION "${NIXL_LIBRARY}")
  endif()
else()
  message(FATAL_ERROR "NIXL not found after installation attempt.")
endif()
