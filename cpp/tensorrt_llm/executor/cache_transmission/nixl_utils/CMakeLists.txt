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

# ============================================================================
# NIXL Wrapper Library
# ============================================================================
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
  target_include_directories(${NIXL_WRAPPER_TARGET}
                             PRIVATE ${PROJECT_SOURCE_DIR}/include)

  # Link against all NIXL libraries
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE NIXL::nixl)

  # Link against CUDA
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE CUDA::cudart)

  set(NIXL_ENABLED TRUE)
else()
  set(NIXL_ENABLED FALSE)
endif()

# ============================================================================
# Check if Mooncake wrapper is available (built in mooncake_utils)
# ============================================================================
if(MOONCAKE_ROOT AND TARGET tensorrt_llm_mooncake_wrapper)
  set(MOONCAKE_ENABLED TRUE)
else()
  set(MOONCAKE_ENABLED FALSE)
endif()

# ============================================================================
# TensorRT-LLM Transfer Agent Binding Python Module Build if either NIXL or
# Mooncake is enabled
# ============================================================================
if(NIXL_ENABLED OR MOONCAKE_ENABLED)
  set(TRANSFER_AGENT_BINDING_TARGET "tensorrt_llm_transfer_agent_binding")

  # Collect binding source files
  set(AGENT_BINDING_SOURCES "")
  list(APPEND AGENT_BINDING_SOURCES agentBindings.cpp)

  nanobind_add_module(${TRANSFER_AGENT_BINDING_TARGET} ${AGENT_BINDING_SOURCES})
  message(STATUS "Building tensorrt_llm_transfer_agent_binding with nanobind")

  target_compile_options(${TRANSFER_AGENT_BINDING_TARGET} PRIVATE -Wno-error)

  # Add common include directories
  target_include_directories(${TRANSFER_AGENT_BINDING_TARGET}
                             PRIVATE ${PROJECT_SOURCE_DIR}/include)

  # Conditionally add NIXL support
  if(NIXL_ENABLED)
    target_compile_definitions(${TRANSFER_AGENT_BINDING_TARGET}
                               PRIVATE ENABLE_NIXL)
    target_include_directories(${TRANSFER_AGENT_BINDING_TARGET}
                               PRIVATE NIXL::nixl)
    target_link_libraries(${TRANSFER_AGENT_BINDING_TARGET}
                          PRIVATE ${NIXL_WRAPPER_TARGET})
    target_link_libraries(${TRANSFER_AGENT_BINDING_TARGET} PRIVATE NIXL::nixl)
    message(STATUS "Transfer agent binding: NIXL support enabled")
  endif()

  # Conditionally add Mooncake support
  if(MOONCAKE_ENABLED)
    target_compile_definitions(${TRANSFER_AGENT_BINDING_TARGET}
                               PRIVATE ENABLE_MOONCAKE)
    target_include_directories(${TRANSFER_AGENT_BINDING_TARGET}
                               PRIVATE ${TRANSFER_ENGINE_INCLUDE_DIR})
    target_link_libraries(${TRANSFER_AGENT_BINDING_TARGET}
                          PRIVATE tensorrt_llm_mooncake_wrapper)
    message(STATUS "Transfer agent binding: Mooncake support enabled")
  endif()

  # Common dependencies
  target_link_libraries(${TRANSFER_AGENT_BINDING_TARGET} PRIVATE CUDA::cudart)
  target_link_libraries(${TRANSFER_AGENT_BINDING_TARGET}
                        PRIVATE ${SHARED_TARGET})

  # Set RPATH for the module to find wrapper libraries
  set_target_properties(
    ${TRANSFER_AGENT_BINDING_TARGET}
    PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/libs;$ORIGIN/libs/nixl"
               INSTALL_RPATH "$ORIGIN;$ORIGIN/libs;$ORIGIN/libs/nixl")

endif()
