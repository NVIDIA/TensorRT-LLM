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

include(visual_gen)

add_custom_target(visual_gen_libs)

get_filename_component(
  visual_gen_source_dir
  "${PROJECT_SOURCE_DIR}/../tensorrt_llm/_torch/visual_gen/modeling_v2"
  ABSOLUTE)
set(visual_gen_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

file(
  GLOB declarations
  RELATIVE "${visual_gen_source_dir}"
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS "${visual_gen_source_dir}/catalog/*/csrc/CMakeLists.txt"
  "${visual_gen_source_dir}/models/*/csrc/CMakeLists.txt"
  "${visual_gen_source_dir}/models/*/*/csrc/CMakeLists.txt")
list(SORT declarations)
foreach(declaration IN LISTS declarations)
  get_filename_component(directory "${declaration}" DIRECTORY)
  add_subdirectory("${visual_gen_source_dir}/${directory}"
                   "${visual_gen_binary_dir}/${directory}")
endforeach()

get_property(entries GLOBAL PROPERTY VISUAL_GEN_NATIVE_ENTRIES)
list(JOIN entries ",\n" manifest)
file(
  GENERATE
  OUTPUT "${visual_gen_binary_dir}/visual_gen_native.json"
  CONTENT "{\n${manifest}\n}\n")
