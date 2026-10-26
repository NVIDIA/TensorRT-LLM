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
# loaded only when routing picks one of its targets. Every subdirectory with a
# CMakeLists.txt is a family; see README.md and
# cpp/cmake/modules/modeling_v2.cmake.

include(modeling_v2)

add_custom_target(modeling_v2_libs)

file(
  GLOB family_lists
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
list(SORT family_lists)

if(WIN32 OR NOT BUILD_MODELING_V2)
  # As deep_ep does on a build without a matching architecture: an empty
  # cuda_architectures.txt per family, so build_wheel.py installs nothing,
  # including a library left from an earlier configuration.
  foreach(family_list IN LISTS family_lists)
    get_filename_component(family_dir "${family_list}" DIRECTORY)
    get_filename_component(family_dir_name "${family_dir}" NAME)
    file(WRITE
         "${CMAKE_CURRENT_BINARY_DIR}/${family_dir_name}/cuda_architectures.txt"
         "")
  endforeach()
  return()
endif()

foreach(family_list IN LISTS family_lists)
  get_filename_component(family_dir "${family_list}" DIRECTORY)
  add_subdirectory("${family_dir}")
endforeach()
add_modeling_v2_family_libraries()
