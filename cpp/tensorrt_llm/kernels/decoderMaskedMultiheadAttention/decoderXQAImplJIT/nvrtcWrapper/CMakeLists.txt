# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# Add xqa subdirectory for xqa_sources_h target.
add_subdirectory(${TRT_LLM_ROOT_DIR}/cpp/kernels/xqa xqa_build)

add_library(nvrtc_wrapper_src STATIC include/nvrtcWrapper.h
                                     src/nvrtcWrapper.cpp)
add_dependencies(nvrtc_wrapper_src xqa_sources_h)
target_include_directories(nvrtc_wrapper_src
                           PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/xqa_build)
target_link_libraries(
  nvrtc_wrapper_src PUBLIC ${NVPTX_LIB} ${NVRTC_LIB} ${NVRTC_BUILTINS_LIB}
                           ${CUDA_DRV_LIB} ${CUDA_RT_LIB})
set_property(TARGET nvrtc_wrapper_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET nvrtc_wrapper_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
