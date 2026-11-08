#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

macro(setup_sanitizers)
  if(SANITIZE)
    if(WIN32)
      message(FATAL_ERROR "Sanitizer support is unimplemented on Windows.")
    endif()

    macro(add_clang_rt_lib lib_name)
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        execute_process(
          COMMAND
            ${CMAKE_CXX_COMPILER}
            "-print-file-name=libclang_rt.${lib_name}-${CMAKE_SYSTEM_PROCESSOR}.so"
          OUTPUT_VARIABLE CLANG_SAN_LIBRARY_PATH
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        link_libraries(${CLANG_SAN_LIBRARY_PATH})
      endif()
    endmacro()

    string(TOLOWER ${SANITIZE} SANITIZE)

    if("undefined" IN_LIST SANITIZE)
      message(STATUS "Enabling extra sub-sanitizers for UBSan")
      list(APPEND SANITIZE "float-divide-by-zero")

      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND SANITIZE "unsigned-integer-overflow" "implicit-conversion"
             "local-bounds")
      endif()
      add_clang_rt_lib("ubsan_standalone")
      add_compile_definitions("SANITIZE_UNDEFINED")
    endif()

    if("address" IN_LIST SANITIZE)
      message(STATUS "Enabling extra sub-sanitizers for ASan")
      list(APPEND SANITIZE "pointer-compare" "pointer-subtract")
      add_compile_options("-fno-omit-frame-pointer;-fno-optimize-sibling-calls")

      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options("-fsanitize-address-use-after-return=always")
        add_link_options("-fsanitize-address-use-after-return=always")
      endif()
      add_clang_rt_lib("asan")
    endif()

    if("thread" IN_LIST SANITIZE)
      add_compile_options("-ftls-model=local-dynamic")
      add_clang_rt_lib("tsan")
    endif()

    list(REMOVE_DUPLICATES SANITIZE)
    message(STATUS "Enabled sanitizers: ${SANITIZE}")

    foreach(SANITIZER IN LISTS SANITIZE)
      add_compile_options("-fsanitize=${SANITIZER}")
      add_link_options("-fsanitize=${SANITIZER}")
    endforeach()
  endif()
endmacro()
