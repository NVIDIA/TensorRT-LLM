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

if(MOONCAKE_ROOT)
  find_library(TRANSFER_ENGINE_LIB transfer_engine ${MOONCAKE_ROOT}/lib)
  find_path(TRANSFER_ENGINE_INCLUDE_DIR transfer_engine_c.h
            ${MOONCAKE_ROOT}/include)

  message(STATUS "Find transfer engine results:")
  message(STATUS "  TRANSFER_ENGINE_LIB = ${TRANSFER_ENGINE_LIB}")
  message(
    STATUS "  TRANSFER_ENGINE_INCLUDE_DIR = ${TRANSFER_ENGINE_INCLUDE_DIR}")

  if(TRANSFER_ENGINE_LIB AND TRANSFER_ENGINE_INCLUDE_DIR)
    set(MOONCAKE_WRAPPER_TARGET "tensorrt_llm_mooncake_wrapper")

    add_library(${MOONCAKE_WRAPPER_TARGET} SHARED transferAgent.cpp)
    target_compile_options(${MOONCAKE_WRAPPER_TARGET} PRIVATE -Wno-error)

    target_include_directories(${MOONCAKE_WRAPPER_TARGET}
                               PRIVATE ${TRANSFER_ENGINE_INCLUDE_DIR})

    target_link_libraries(${MOONCAKE_WRAPPER_TARGET}
                          PRIVATE ${TRANSFER_ENGINE_LIB} CUDA::cudart)
  endif()
endif()
