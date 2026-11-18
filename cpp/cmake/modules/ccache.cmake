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

# This function sets up:
#
# * ccache `base_dir` to be the git checkout root, which lets ccache see paths
#   inside the checkout as relative paths to the build working directory.
#
# * Compiler flag `-fdebug-prefix-map` for debug builds, which sets paths in the
#   debug info to be relative as well.
#
# As a result, cache from one worktree can be used on another. This helps when a
# developer works in several worktrees at the same time, or when coding agents
# work in separate worktrees.
#
# The setup only runs if all following conditions are met:
#
# * At least one of the launchers, CMAKE_C_COMPILER_LAUNCHER,
#   CMAKE_CXX_COMPILER_LAUNCHER or CMAKE_CUDA_COMPILER_LAUNCHER must be a single
#   item whose file name is ccache.
#
# * ccache is 4.8 or newer. We use ccache base_dir=... which is only added in
#   4.8.
#
# * No ccache base_dir is configured yet. ccache --get-config base_dir must
#   print nothing.
function(setup_ccache_base_dir root_dir)
  set(ccache_program "")
  foreach(lang C CXX CUDA)
    set(launcher "${CMAKE_${lang}_COMPILER_LAUNCHER}")
    list(LENGTH launcher launcher_length)
    get_filename_component(launcher_name "${launcher}" NAME)
    if(launcher_length EQUAL 1 AND launcher_name STREQUAL "ccache")
      set(ccache_program "${launcher}")
    endif()
  endforeach()
  if(ccache_program STREQUAL "")
    return()
  endif()

  execute_process(
    COMMAND "${ccache_program}" --version
    OUTPUT_VARIABLE version_output
    RESULT_VARIABLE version_result
    ERROR_QUIET)
  if(NOT version_result EQUAL 0)
    message(STATUS "ccache not found: not sharing its cache across checkouts")
    return()
  endif()
  set(ccache_version "")
  if(version_output MATCHES "version ([0-9]+\\.[0-9]+)")
    set(ccache_version "${CMAKE_MATCH_1}")
  endif()
  if(ccache_version VERSION_LESS 4.8)
    message(
      STATUS "ccache older than 4.8: not sharing its cache across checkouts")
    return()
  endif()

  execute_process(
    COMMAND "${ccache_program}" --get-config base_dir
    OUTPUT_VARIABLE configured_base_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
  if(NOT configured_base_dir STREQUAL "")
    message(STATUS "ccache base_dir already set to ${configured_base_dir}")
    return()
  endif()

  message(STATUS "ccache base_dir: ${root_dir}")
  foreach(lang C CXX CUDA)
    if("${CMAKE_${lang}_COMPILER_LAUNCHER}" STREQUAL "${ccache_program}")
      set(CMAKE_${lang}_COMPILER_LAUNCHER
          "${ccache_program};base_dir=${root_dir}"
          PARENT_SCOPE)
    endif()
  endforeach()

  # Use compiler flag `-fdebug-prefix-map` for debug builds, which sets paths in
  # the debug info to be relative.
  add_compile_options(
    "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:C,CXX,CUDA>>:-fdebug-prefix-map=${root_dir}=.>"
  )
endfunction()
