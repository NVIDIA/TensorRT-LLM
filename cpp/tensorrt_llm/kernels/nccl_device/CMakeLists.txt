#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
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

# CMakeLists.txt for nccl_device This directory contains CUDA kernels and host
# launcher code

# Enable CUDA
enable_language(CUDA)

# Create CUDA library
add_library(tensorrt_llm_nccl_device config.cu)

# Set properties for the CUDA library
set_target_properties(
  tensorrt_llm_nccl_device
  PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON
             POSITION_INDEPENDENT_CODE ON)

# Include directories
target_include_directories(
  tensorrt_llm_nccl_device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Link libraries
target_link_libraries(tensorrt_llm_nccl_device tensorrt_llm_common)
