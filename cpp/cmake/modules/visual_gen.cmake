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

#[[
Native (C++/CUDA) code of VisualGen models.

A model keeps its native code with its Python code, in
tensorrt_llm/_torch/visual_gen/models/<model>/csrc/, whose CMakeLists.txt calls

  add_visual_gen_model_library(<model> SOURCES ...)

SOURCES   Paths or globs relative to that csrc/ directory. .cu files are the
          kernels, compiled for the build's CUDA architectures without torch;
          .cpp files are the torch bindings, which register their ops with
          TORCH_LIBRARY_FRAGMENT(vg_<model>, m).

Each model builds into one torch-op library, libth_visual_gen_<model>.so,
installed next to libth_common.so; nothing of it is linked into the stock
libraries. tensorrt_llm/_torch/visual_gen/_native.py opens it when the model
first needs it. The library exports no symbols: its ops register from static
initializers when it is loaded.
#]]

include_guard(GLOBAL)

function(add_visual_gen_model_library model)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
  get_filename_component(package_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(package_name "${package_dir}" NAME)
  if(NOT package_name STREQUAL model)
    message(
      FATAL_ERROR
        "visual_gen: model '${model}' is declared in models/${model}/csrc/CMakeLists.txt, "
        "not in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  set(cu_sources "")
  set(cpp_sources "")
  foreach(item IN LISTS arg_SOURCES)
    file(GLOB matched CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${item}")
    if(NOT matched)
      message(
        FATAL_ERROR
          "visual_gen: '${item}' matches no file in ${CMAKE_CURRENT_SOURCE_DIR}"
      )
    endif()
    list(SORT matched)
    foreach(source IN LISTS matched)
      if(source MATCHES "\\.cu$")
        list(APPEND cu_sources "${source}")
      elseif(source MATCHES "\\.cpp$")
        list(APPEND cpp_sources "${source}")
      endif()
    endforeach()
  endforeach()

  set(library th_visual_gen_${model})
  set(objects "")
  if(cu_sources)
    add_library(${library}_kernels OBJECT ${cu_sources})
    set_target_properties(${library}_kernels
                          PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND objects $<TARGET_OBJECTS:${library}_kernels>)
  endif()
  add_library(${library} MODULE ${cpp_sources} ${objects})
  target_link_libraries(${library} PRIVATE ${TORCH_LIBRARIES} ${CUDA_RT_LIB})
  set(version_script
      "${PROJECT_SOURCE_DIR}/tensorrt_llm/visual_gen/modelLibrary.version")
  set_target_properties(
    ${library}
    PROPERTIES PREFIX "lib" LIBRARY_OUTPUT_DIRECTORY
                            "${CMAKE_CURRENT_BINARY_DIR}" LINKER_LANGUAGE CXX
               LINK_DEPENDS "${version_script}")
  target_link_options(
    ${library} PRIVATE "-Wl,--version-script=${version_script}"
    "-Wl,--no-undefined")
  add_dependencies(visual_gen_libs ${library})
  set_property(GLOBAL APPEND PROPERTY VISUAL_GEN_MODELS "${model}")
endfunction()
