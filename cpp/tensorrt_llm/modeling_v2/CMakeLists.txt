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

# Native code of modeling_v2 targets: one library per model family, loaded only
# when routing picks one of its targets. Every subdirectory with a
# CMakeLists.txt is a family; see README.md and
# cpp/cmake/modules/modelingV2.cmake.

add_custom_target(modeling_v2_libs)

if(WIN32)
  trtllm_mv2_write_disabled_manifest(
    "modeling_v2 libraries are not built on Windows")
elseif(NOT BUILD_MODELING_V2)
  trtllm_mv2_write_disabled_manifest("built with BUILD_MODELING_V2=OFF")
else()
  file(
    GLOB family_lists
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
  list(SORT family_lists)
  foreach(family_list IN LISTS family_lists)
    get_filename_component(family_dir "${family_list}" DIRECTORY)
    add_subdirectory("${family_dir}")
  endforeach()
  trtllm_mv2_finalize()
endif()
