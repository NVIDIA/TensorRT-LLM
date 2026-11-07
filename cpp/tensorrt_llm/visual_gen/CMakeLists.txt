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

# Native code of VisualGen models: one torch-op library per model, built from
# tensorrt_llm/_torch/visual_gen/models/<model>/csrc/. See
# cpp/cmake/modules/visual_gen.cmake.

include(visual_gen)

add_custom_target(visual_gen_libs)

get_filename_component(
  visual_gen_models_dir
  "${PROJECT_SOURCE_DIR}/../tensorrt_llm/_torch/visual_gen/models" ABSOLUTE)
file(
  GLOB model_lists
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS "${visual_gen_models_dir}/*/csrc/CMakeLists.txt")
list(SORT model_lists)
foreach(model_list IN LISTS model_lists)
  get_filename_component(csrc_dir "${model_list}" DIRECTORY)
  get_filename_component(model_dir "${csrc_dir}" DIRECTORY)
  get_filename_component(model "${model_dir}" NAME)
  add_subdirectory("${csrc_dir}" "${CMAKE_CURRENT_BINARY_DIR}/${model}")
endforeach()

# scripts/build_wheel.py installs the libraries of the models listed here.
get_property(models GLOBAL PROPERTY VISUAL_GEN_MODELS)
list(JOIN models "\n" manifest)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/models.txt" "${manifest}")
