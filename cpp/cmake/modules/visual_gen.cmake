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

include_guard(GLOBAL)
include(modeling_v2)

# Each catalog category, model family or target declares its native translation
# units in its own csrc/CMakeLists.txt. Scope determines the library and
# namespace.
function(add_visual_gen_native)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "" "ARCHS;SOURCES")
  if(arg_UNPARSED_ARGUMENTS
     OR arg_KEYWORDS_MISSING_VALUES
     OR NOT arg_ARCHS
     OR NOT arg_SOURCES)
    message(
      FATAL_ERROR
        "add_visual_gen_native requires ARCHS and SOURCES in ${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
  file(RELATIVE_PATH directory "${visual_gen_source_dir}"
       "${CMAKE_CURRENT_SOURCE_DIR}")
  set(identifier "[A-Za-z][A-Za-z0-9_]*")
  if(directory MATCHES "^catalog/(${identifier})/csrc$")
    set(owner catalog)
    set(torch_ns vg_catalog)
    set(cxx_scope "catalog::category_${CMAKE_MATCH_1}")
  elseif(directory MATCHES "^models/(${identifier})(/(${identifier}))?/csrc$")
    set(owner "${CMAKE_MATCH_1}")
    set(target_name "${CMAKE_MATCH_3}")
    if(owner STREQUAL "catalog")
      message(FATAL_ERROR "visual_gen: catalog is reserved for shared ops")
    endif()
    set(torch_ns "vg_${owner}")
    set(cxx_scope "models::family_${owner}")
    if(NOT target_name STREQUAL "")
      string(APPEND torch_ns "_${target_name}")
      string(APPEND cxx_scope "::target_${target_name}")
    else()
      string(APPEND cxx_scope "::shared")
    endif()
  else()
    message(
      FATAL_ERROR "visual_gen: invalid native source scope '${directory}'")
  endif()
  get_filename_component(scope "${directory}" DIRECTORY)
  _modeling_v2_ident(torch_ns "${torch_ns}")
  _modeling_v2_ident(cxx_scope "${cxx_scope}")
  # Catalog categories share a Torch namespace and have distinct C++ namespaces.
  if(owner STREQUAL "catalog")
    set(namespace_key "${cxx_scope}")
  else()
    set(namespace_key "${torch_ns}")
  endif()
  get_property(previous GLOBAL PROPERTY "VISUAL_GEN_NAMESPACE_${namespace_key}")
  if(previous STREQUAL scope)
    message(FATAL_ERROR "visual_gen: declare native scope '${scope}' only once")
  elseif(previous)
    message(
      FATAL_ERROR
        "visual_gen: '${scope}' and '${previous}' both register '${namespace_key}'"
    )
  endif()
  set_property(GLOBAL PROPERTY "VISUAL_GEN_NAMESPACE_${namespace_key}"
                               "${scope}")

  _modeling_v2_resolve_archs(archs DECLARED ${arg_ARCHS})
  if(NOT archs)
    message(
      STATUS
        "visual_gen: skipping ${scope} (requires CUDA architectures ${arg_ARCHS})"
    )
    return()
  endif()
  _modeling_v2_expand_sources(sources "${CMAKE_CURRENT_SOURCE_DIR}"
                              ${arg_SOURCES})
  set(library "th_visual_gen_${owner}")
  if(NOT TARGET ${library})
    add_library(${library} MODULE)
    set_target_properties(
      ${library}
      PROPERTIES PREFIX "lib"
                 LIBRARY_OUTPUT_DIRECTORY "$<1:${visual_gen_binary_dir}>"
                 RUNTIME_OUTPUT_DIRECTORY "$<1:${visual_gen_binary_dir}>"
                 POSITION_INDEPENDENT_CODE ON
                 LINKER_LANGUAGE CXX
                 BUILD_RPATH "$ORIGIN")
    target_link_libraries(${library} PRIVATE ${TORCH_LIBRARIES}
                                             ${SHARED_TARGET} ${CUDA_RT_LIB})
    if(UNIX AND NOT APPLE)
      set(exports
          "${PROJECT_SOURCE_DIR}/tensorrt_llm/modeling_v2/common/familyLibrary.version"
      )
      set_property(TARGET ${library} APPEND PROPERTY LINK_DEPENDS "${exports}")
      target_link_options(
        ${library} PRIVATE "LINKER:--version-script=${exports}"
        "LINKER:--no-undefined")
    endif()
    add_dependencies(visual_gen_libs ${library})
  endif()

  set(kernels "")
  set(bindings "")
  foreach(source IN LISTS sources)
    if(source MATCHES "\\.cu$")
      list(APPEND kernels "${source}")
    elseif(source MATCHES "\\.cpp$")
      list(APPEND bindings "${source}")
    else()
      message(
        FATAL_ERROR
          "visual_gen: expected a .cu kernel or .cpp binding, got '${source}'")
    endif()
  endforeach()
  # CUDA objects use the selected architecture and only bindings need the Torch
  # API.
  string(REPLACE "/" "." unit "${scope}")
  foreach(kind kernels bindings)
    if(${kind})
      set(objects "visual_gen.${unit}.${kind}")
      add_library(${objects} OBJECT ${${kind}})
      set_property(TARGET ${objects} PROPERTY POSITION_INDEPENDENT_CODE ON)
      target_include_directories(${objects} PRIVATE "${visual_gen_source_dir}")
      target_compile_definitions(
        ${objects} PRIVATE TRTLLM_VG_TORCH_NS=${torch_ns}
                           TRTLLM_VG_SCOPE=${cxx_scope})
      if(kind STREQUAL "kernels")
        set_property(TARGET ${objects} PROPERTY CUDA_ARCHITECTURES "${archs}")
      else()
        target_link_libraries(${objects} PRIVATE ${TORCH_LIBRARIES})
      endif()
      target_sources(${library} PRIVATE $<TARGET_OBJECTS:${objects}>)
    endif()
  endforeach()

  set_property(
    GLOBAL APPEND
    PROPERTY
      VISUAL_GEN_NATIVE_ENTRIES
      "  \"${scope}\": {\"library\": \"$<TARGET_FILE_NAME:${library}>\", \"namespace\": \"${torch_ns}\"}"
  )
endfunction()
