# ~~~
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ~~~

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
