#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

# Native code of modeling_v2 targets: one torch-op library per model family,
# loaded only when routing picks one of its targets. A family keeps its native
# code with its Python code, in
# tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/csrc/ (declared
# by the CMakeLists.txt there) and <family>/<target>/csrc/; this directory holds
# only what every family shares. See cpp/cmake/modules/modeling_v2.cmake.

include(modeling_v2)

add_custom_target(modeling_v2_libs)

get_filename_component(
  modeling_v2_models_dir
  "${PROJECT_SOURCE_DIR}/../tensorrt_llm/_torch/_experimental/modeling_v2/models"
  ABSOLUTE)
file(
  GLOB family_lists
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS "${modeling_v2_models_dir}/*/csrc/CMakeLists.txt")
list(SORT family_lists)

if(WIN32 OR NOT BUILD_MODELING_V2)
  # As deep_ep does on a build without a matching architecture: an empty
  # cuda_architectures.txt per family, so build_wheel.py installs nothing,
  # including a library left from an earlier configuration.
  foreach(family_list IN LISTS family_lists)
    get_filename_component(csrc_dir "${family_list}" DIRECTORY)
    get_filename_component(family_dir "${csrc_dir}" DIRECTORY)
    get_filename_component(family "${family_dir}" NAME)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${family}/cuda_architectures.txt"
         "")
  endforeach()
  return()
endif()

foreach(family_list IN LISTS family_lists)
  get_filename_component(csrc_dir "${family_list}" DIRECTORY)
  get_filename_component(family_dir "${csrc_dir}" DIRECTORY)
  get_filename_component(family "${family_dir}" NAME)
  add_subdirectory("${csrc_dir}" "${CMAKE_CURRENT_BINARY_DIR}/${family}")
endforeach()
add_modeling_v2_family_libraries()
