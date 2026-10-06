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

function(tllm_check_kernel_source_ownership)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "" "TARGETS;DELEGATED_DIRECTORIES")
  if(ARG_UNPARSED_ARGUMENTS
     OR ARG_KEYWORDS_MISSING_VALUES
     OR NOT ARG_TARGETS)
    message(FATAL_ERROR "Kernel ownership audit requires explicit targets")
  endif()

  # This inventory is used only for validation, never as a compilation input.
  # CONFIGURE_DEPENDS makes adding an undeclared source trigger the audit.
  file(
    GLOB_RECURSE inventory CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cpp" "*.cu")
  foreach(source IN LISTS inventory)
    cmake_path(IS_PREFIX CMAKE_CURRENT_SOURCE_DIR "${CMAKE_BINARY_DIR}"
               NORMALIZE nested_build)
    cmake_path(IS_PREFIX CMAKE_BINARY_DIR
               "${CMAKE_CURRENT_SOURCE_DIR}/${source}" NORMALIZE build_source)
    if(nested_build
       AND build_source
       AND NOT CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
      list(REMOVE_ITEM inventory "${source}")
    endif()
  endforeach()
  foreach(delegation IN LISTS ARG_DELEGATED_DIRECTORIES)
    string(REPLACE "=" ";" parts "${delegation}")
    list(LENGTH parts count)
    if(count GREATER 2)
      message(FATAL_ERROR "Invalid kernel delegation: ${delegation}")
    endif()
    list(GET parts 0 directory)
    set(build_directory "${directory}")
    if(count EQUAL 2)
      list(GET parts 1 build_directory)
    endif()
    if(NOT EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/${build_directory}/CMakeLists.txt")
      message(FATAL_ERROR "Missing delegated kernel build: ${directory}")
    endif()
    foreach(source IN LISTS inventory)
      string(FIND "${source}" "${directory}/" prefix)
      if(prefix EQUAL 0)
        list(REMOVE_ITEM inventory "${source}")
      endif()
    endforeach()
  endforeach()

  foreach(target IN LISTS ARG_TARGETS)
    get_target_property(source_directory ${target} SOURCE_DIR)
    get_target_property(sources ${target} SOURCES)
    foreach(source IN LISTS sources)
      if(source MATCHES "\\$<" OR NOT source MATCHES "\\.(cpp|cu)$")
        continue()
      endif()
      get_filename_component(absolute_source "${source}" ABSOLUTE BASE_DIR
                             "${source_directory}")
      file(RELATIVE_PATH relative_source "${CMAKE_CURRENT_SOURCE_DIR}"
           "${absolute_source}")
      if(NOT relative_source IN_LIST inventory)
        continue()
      endif()
      string(SHA256 key "${relative_source}")
      if(DEFINED owner_${key} AND NOT owner_${key} STREQUAL target)
        message(
          FATAL_ERROR
            "Kernel source has multiple owners: ${relative_source}: ${owner_${key}}, ${target}"
        )
      endif()
      set(owner_${key} "${target}")
    endforeach()
  endforeach()

  foreach(source IN LISTS inventory)
    string(SHA256 key "${source}")
    if(NOT DEFINED owner_${key})
      message(
        FATAL_ERROR
          "Kernel source has no explicit owner: ${source}. Add it to the appropriate component SOURCES list."
      )
    endif()
  endforeach()
endfunction()
