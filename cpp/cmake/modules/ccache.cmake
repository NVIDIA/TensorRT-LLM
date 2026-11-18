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

# Lets checkouts at different paths, such as git worktrees, share one ccache.
#
# ccache rewrites absolute paths under its base_dir as paths relative to the
# compiler's working directory. With base_dir at the checkout root and the build
# folder inside the checkout, a source file gets the same compile command in
# every worktree, so a fresh worktree reuses the objects another one compiled.
# ccache 4.8 and newer read settings placed before the compiler on its command
# line, so the setting goes into the compiler launchers and needs no ccache
# configuration from the user.
#
# A base_dir that the user already configured (ccache.conf or CCACHE_BASEDIR) is
# kept, and a launcher other than a plain ccache is left alone.
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
  set(ccache_version "")
  if(version_output MATCHES "version ([0-9]+\\.[0-9]+)")
    set(ccache_version "${CMAKE_MATCH_1}")
  endif()
  if(NOT version_result EQUAL 0 OR ccache_version VERSION_LESS 4.8)
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

  # Debug info records absolute paths, and ccache then hashes the working
  # directory too. Mapping the checkout root to "." keeps both the same in every
  # worktree.
  set(debug_configs "$<CONFIG:Debug,RelWithDebInfo>")
  add_compile_options(
    "$<$<AND:${debug_configs},$<COMPILE_LANGUAGE:C,CXX>>:-fdebug-prefix-map=${root_dir}=.>"
    "$<$<AND:${debug_configs},$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fdebug-prefix-map=${root_dir}=.>"
  )
endfunction()
