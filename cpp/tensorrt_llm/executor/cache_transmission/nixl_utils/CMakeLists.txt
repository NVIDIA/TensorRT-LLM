# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

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
