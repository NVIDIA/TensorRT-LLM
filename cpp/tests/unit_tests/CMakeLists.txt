# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batch_manager)
  add_subdirectory(batch_manager)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/executor)
  add_subdirectory(executor)
endif()

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(multi_gpu)
add_subdirectory(layers)
add_subdirectory(runtime)
add_subdirectory(thop)
add_subdirectory(utils)
