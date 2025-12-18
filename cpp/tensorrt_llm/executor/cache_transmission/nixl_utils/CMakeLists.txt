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
  target_include_directories(${NIXL_WRAPPER_TARGET}
                             PRIVATE ${PROJECT_SOURCE_DIR}/include)

  # Link against all NIXL libraries
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE NIXL::nixl)

  # Link against CUDA
  target_link_libraries(${NIXL_WRAPPER_TARGET} PRIVATE CUDA::cudart)

  # ============================================================================
  # TensorRT-LLM Transfer Agent Binding Python Module (standalone, independent
  # of main trtllm bindings) Uses BINDING_TYPE to choose between pybind11 and
  # nanobind
  # ============================================================================
  set(AGENT_BINDING_TARGET "tensorrt_llm_transfer_agent_binding")

  if(BINDING_TYPE STREQUAL "pybind")
    # Use pybind11 (already fetched via FetchContent)
    pybind11_add_module(${AGENT_BINDING_TARGET} agentBindingsPybind.cpp)
    message(STATUS "Building tensorrt_llm_transfer_agent_binding with pybind11")
  else()
    # Default to nanobind (already fetched via FetchContent)
    nanobind_add_module(${AGENT_BINDING_TARGET} agentBindingsNanobind.cpp)
    message(STATUS "Building tensorrt_llm_transfer_agent_binding with nanobind")
  endif()

  target_compile_options(${AGENT_BINDING_TARGET} PRIVATE -Wno-error)

  # Add include directories
  target_include_directories(${AGENT_BINDING_TARGET} PRIVATE NIXL::nixl)
  target_include_directories(${AGENT_BINDING_TARGET}
                             PRIVATE ${PROJECT_SOURCE_DIR}/include)

  # Link against NIXL wrapper and NIXL libraries
  target_link_libraries(${AGENT_BINDING_TARGET} PRIVATE ${NIXL_WRAPPER_TARGET})
  target_link_libraries(${AGENT_BINDING_TARGET} PRIVATE NIXL::nixl)
  target_link_libraries(${AGENT_BINDING_TARGET} PRIVATE CUDA::cudart)

  # Set RPATH for the module to find libtensorrt_llm_nixl_wrapper.so
  set_target_properties(
    ${AGENT_BINDING_TARGET} PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/libs"
                                       INSTALL_RPATH "$ORIGIN;$ORIGIN/libs")

  # Export the agent binding target name to parent scope so it can be added as
  # dependency to tensorrt_llm main library
  set(AGENT_BINDING_TARGET
      ${AGENT_BINDING_TARGET}
      PARENT_SCOPE)

endif()
