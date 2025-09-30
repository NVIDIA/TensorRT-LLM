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

macro(setup_cuda_compiler)
  # Determine CUDA version before enabling the language extension
  # check_language(CUDA) clears CMAKE_CUDA_HOST_COMPILER if CMAKE_CUDA_COMPILER
  # is not set
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_COMPILER AND CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER_BACKUP ${CMAKE_CUDA_HOST_COMPILER})
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_HOST_COMPILER_BACKUP)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER_BACKUP})
    check_language(CUDA)
  endif()
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    if(NOT WIN32) # Linux
      execute_process(
        COMMAND
          "bash" "-c"
          "${CMAKE_CUDA_COMPILER} --version | egrep -o 'V[0-9]+.[0-9]+.[0-9]+' | cut -c2-"
        RESULT_VARIABLE _BASH_SUCCESS
        OUTPUT_VARIABLE CMAKE_CUDA_COMPILER_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(NOT _BASH_SUCCESS EQUAL 0)
        message(FATAL_ERROR "Failed to determine CUDA version")
      endif()

    else() # Windows
      execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        OUTPUT_VARIABLE versionString
        RESULT_VARIABLE versionResult)

      if(versionResult EQUAL 0 AND versionString MATCHES
                                   "V[0-9]+\\.[0-9]+\\.[0-9]+")
        string(REGEX REPLACE "V" "" version ${CMAKE_MATCH_0})
        set(CMAKE_CUDA_COMPILER_VERSION "${version}")
      else()
        message(FATAL_ERROR "Failed to determine CUDA version")
      endif()
    endif()
  else()
    message(FATAL_ERROR "No CUDA compiler found")
  endif()

  set(CUDA_REQUIRED_VERSION "11.2")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS CUDA_REQUIRED_VERSION)
    message(
      FATAL_ERROR
        "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} must be at least ${CUDA_REQUIRED_VERSION}"
    )
  endif()
endmacro()

function(setup_cuda_architectures)
  # cmake-format: off
  # Initialize and normalize CMAKE_CUDA_ARCHITECTURES.
  # Special values:
  # * `native` is resolved to HIGHEST available architecture.
  #   * Fallback to `all` if detection failed.
  # * `all`/unset is resolved to a set of architectures we optimized for and compiler supports.
  # * `all-major` is unsupported.
  # Numerical architectures:
  # * PTX is never included in result binary.
  #   * `*-virtual` architectures are therefore rejected.
  #   * `-real` suffix is automatically added to exclude PTX.
  # * Always use accelerated (`-a` suffix) target for supported architectures.
  # * On CUDA 12.9 or newer, family (`-f` suffix) target will be used for supported architectures to reduce number of
  #   targets to compile for.
  #   * Extra architectures can be requested via add_cuda_architectures
  #     for kernels that benefit from arch specific features.
  # cmake-format: on

  set(CMAKE_CUDA_ARCHITECTURES_RAW ${CMAKE_CUDA_ARCHITECTURES})
  if(CMAKE_CUDA_ARCHITECTURES_RAW STREQUAL "native")
    # Detect highest available compute capability
    set(OUTPUTFILE ${PROJECT_BINARY_DIR}/detect_cuda_arch)
    set(CUDAFILE ${CMAKE_SOURCE_DIR}/cmake/utils/detect_cuda_arch.cu)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o
                            ${OUTPUTFILE})
    message(VERBOSE "Detecting native CUDA compute capability")
    execute_process(
      COMMAND ${OUTPUTFILE}
      RESULT_VARIABLE CUDA_RETURN_CODE
      OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)
    if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
      message(WARNING "Detecting native CUDA compute capability - fail")
      message(
        WARNING
          "CUDA compute capability detection failed, compiling for all optimized architectures"
      )
      unset(CMAKE_CUDA_ARCHITECTURES_RAW)
    else()
      message(STATUS "Detecting native CUDA compute capability - done")
      set(CMAKE_CUDA_ARCHITECTURES_RAW "${CUDA_ARCH_OUTPUT}")
    endif()
  elseif(CMAKE_CUDA_ARCHITECTURES_RAW STREQUAL "all")
    unset(CMAKE_CUDA_ARCHITECTURES_RAW)
    message(
      STATUS
        "Setting CMAKE_CUDA_ARCHITECTURES to all enables all architectures TensorRT LLM optimized for, "
        "not all architectures CUDA compiler supports.")
  elseif(CMAKE_CUDA_ARCHITECTURES_RAW STREQUAL "all-major")
    message(
      FATAL_ERROR
        "Setting CMAKE_CUDA_ARCHITECTURES to all-major does not make sense for TensorRT-LLM. "
        "Please enable all architectures you intend to run on, so we can enable optimized kernels for them."
    )
  else()
    foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_RAW)
      if(CUDA_ARCH STREQUAL "")
        continue()
      endif()

      if(CUDA_ARCH MATCHES "^([1-9])([0-9])+a?-virtual$")
        message(FATAL_ERROR "Including PTX in compiled binary is unsupported.")
      elseif(CUDA_ARCH MATCHES "^(([1-9])([0-9])+)a?(-real)?$")
        list(APPEND CMAKE_CUDA_ARCHITECTURES_CLEAN ${CMAKE_MATCH_1})
      else()
        message(FATAL_ERROR "Unrecognized CUDA architecture: ${CUDA_ARCH}")
      endif()
    endforeach()
    if("103" IN_LIST CMAKE_CUDA_ARCHITECTURES_CLEAN)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_CLEAN "100")
    endif()
    list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES_CLEAN)
    set(CMAKE_CUDA_ARCHITECTURES_RAW ${CMAKE_CUDA_ARCHITECTURES_CLEAN})
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES_RAW)
    set(CMAKE_CUDA_ARCHITECTURES_RAW 80 86)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_RAW 89 90)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.7")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_RAW 100 120)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_RAW 103)
    endif()
  endif()

  # CMAKE_CUDA_ARCHITECTURES_ORIG contains all architectures enabled, without
  # automatically added -real or -a suffix.
  set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CMAKE_CUDA_ARCHITECTURES_RAW}")
  message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES_ORIG}")
  set(CMAKE_CUDA_ARCHITECTURES_ORIG
      ${CMAKE_CUDA_ARCHITECTURES_ORIG}
      PARENT_SCOPE)

  set(ARCHITECTURES_WITH_KERNELS
      80
      86
      89
      90
      100
      103
      120)
  foreach(CUDA_ARCH IN LISTS ARCHITECTURES_WITH_KERNELS)
    if(NOT ${CUDA_ARCH} IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      add_definitions("-DEXCLUDE_SM_${CUDA_ARCH}")
      message(STATUS "Excluding SM ${CUDA_ARCH}")
    endif()
  endforeach()

  # -a suffix supported from Hopper (90)
  set(MIN_ARCHITECTURE_HAS_ACCEL 90)
  # -f suffix supported from Blackwell (100) starting from CUDA 12.9.
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    set(MIN_ARCHITECTURE_HAS_FAMILY 100)
    set(CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES
        ON
        PARENT_SCOPE)
  else()
    # -a provides no cross architecture compatibility, but luckily until CUDA
    # 12.8 We have only one architecture within each family >= 9.
    set(MIN_ARCHITECTURE_HAS_FAMILY 9999) # Effectively exclude all
                                          # architectures
    set(CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES
        OFF
        PARENT_SCOPE)
  endif()
  # Compatibility low bounds: Always compile kernels for these architectures. 86
  # is enabled to avoid perf regression when using 80 kernels.
  set(ARCHITECTURES_COMPATIBILITY_BASE 80 86 90 100 120)
  # Exclude Tegra architectures
  set(ARCHITECTURES_NO_COMPATIBILITY 87 101)

  # Generate CMAKE_CUDA_ARCHITECTURES_NORMALIZED from
  # CMAKE_CUDA_ARCHITECTURES_ORIG
  set(CMAKE_CUDA_ARCHITECTURES_NORMALIZED_LIST)

  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_ORIG)
    # If ARCH is in ARCHITECTURES_NO_COMPATIBILITY or
    # ARCHITECTURES_COMPATIBILITY_BASE, add it directly
    if(${CUDA_ARCH} IN_LIST ARCHITECTURES_NO_COMPATIBILITY
       OR ${CUDA_ARCH} IN_LIST ARCHITECTURES_COMPATIBILITY_BASE)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED_LIST ${CUDA_ARCH})
    else()
      # Find the largest BASE_ARCH in ARCHITECTURES_COMPATIBILITY_BASE less than
      # ARCH
      set(BEST_BASE_ARCH "")
      set(ARCH_MAJOR "")
      math(EXPR ARCH_MAJOR "${CUDA_ARCH} / 10")

      foreach(BASE_ARCH IN LISTS ARCHITECTURES_COMPATIBILITY_BASE)
        if(BASE_ARCH LESS ${CUDA_ARCH})
          set(BASE_MAJOR "")
          math(EXPR BASE_MAJOR "${BASE_ARCH} / 10")

          # Check if major version matches
          if(BASE_MAJOR EQUAL ARCH_MAJOR)
            if(NOT "${BEST_BASE_ARCH}" OR ${BASE_ARCH} GREATER
                                          "${BEST_BASE_ARCH}")
              set(BEST_BASE_ARCH ${BASE_ARCH})
            endif()
          endif()
        endif()
      endforeach()

      if("${BEST_BASE_ARCH}")
        if(NOT ${BEST_BASE_ARCH} IN_LIST
           CMAKE_CUDA_ARCHITECTURES_NORMALIZED_LIST)
          list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED_LIST
               ${BEST_BASE_ARCH})
        endif()
      else()
        message(FATAL_ERROR "Unsupported CUDA architecture: ${CUDA_ARCH}.")
      endif()
    endif()
  endforeach()

  # Apply suffixes based on architecture capabilities
  set(CMAKE_CUDA_ARCHITECTURES_NORMALIZED)
  set(CMAKE_CUDA_ARCHITECTURES_FAMILIES)
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_NORMALIZED_LIST)
    if(CUDA_ARCH GREATER_EQUAL ${MIN_ARCHITECTURE_HAS_FAMILY}
       AND NOT CUDA_ARCH IN_LIST ARCHITECTURES_NO_COMPATIBILITY)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}f-real")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_FAMILIES "${CUDA_ARCH}f")
    elseif(CUDA_ARCH GREATER_EQUAL ${MIN_ARCHITECTURE_HAS_ACCEL})
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}a-real")
    else()
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}-real")
    endif()
  endforeach()

  set(CMAKE_CUDA_ARCHITECTURES
      ${CMAKE_CUDA_ARCHITECTURES_NORMALIZED}
      PARENT_SCOPE)
  set(CMAKE_CUDA_ARCHITECTURES_FAMILIES
      ${CMAKE_CUDA_ARCHITECTURES_FAMILIES}
      PARENT_SCOPE)
endfunction()

function(add_cuda_architectures target)
  # cmake-format: off
  # Add CUDA architectures to target.
  # -a suffix is added automatically for supported architectures.
  # Architectures are added only if user explicitly requested support for that architecture.
  # cmake-format: on
  set(MIN_ARCHITECTURE_HAS_ACCEL 90)

  foreach(CUDA_ARCH IN LISTS ARGN)
    if(${CUDA_ARCH} IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      if(${CUDA_ARCH} GREATER_EQUAL ${MIN_ARCHITECTURE_HAS_ACCEL})
        set(REAL_CUDA_ARCH "${CUDA_ARCH}a-real")
      else()
        set(REAL_CUDA_ARCH "${CUDA_ARCH}-real")
      endif()
      set_property(
        TARGET ${target}
        APPEND
        PROPERTY CUDA_ARCHITECTURES ${REAL_CUDA_ARCH})
    endif()
  endforeach()
endfunction()

function(set_cuda_architectures target)
  # cmake-format: off
  # Set CUDA architectures for a target.
  # -a suffix is added automatically for supported architectures.
  # Architectures passed in may be specified with -f suffix to build family conditional version of the kernel.
  # Non-family architectures are added only if user explicitly requested support for that architecture.
  # Family conditional architectures are only added if user requested architectures would enable compilation for it.
  # If user requested no architectures set on the target,
  # the target will be compiled with `PLACEHOLDER_KERNELS` macro defined.
  # cmake-format: on
  set(MIN_ARCHITECTURE_HAS_ACCEL 90)

  set(CUDA_ARCHITECTURES "")
  foreach(CUDA_ARCH IN LISTS ARGN)
    if(${CUDA_ARCH} MATCHES "[0-9]+f")
      if(CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES)
        if(${CUDA_ARCH} IN_LIST CMAKE_CUDA_ARCHITECTURES_FAMILIES)
          list(APPEND CUDA_ARCHITECTURES "${CUDA_ARCH}-real")
        endif()
      else()
        # Fallback for compiler without -f support: Enable all architectures in
        # the family and requested
        string(REGEX REPLACE "f$" "" CUDA_ARCH_NUMERIC "${CUDA_ARCH}")
        math(EXPR ARCH_MAJOR "${CUDA_ARCH_NUMERIC} / 10")
        foreach(ORIG_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_ORIG)
          math(EXPR ORIG_MAJOR "${ORIG_ARCH} / 10")
          if(ORIG_MAJOR EQUAL ARCH_MAJOR)
            list(APPEND CUDA_ARCHITECTURES "${ORIG_ARCH}a-real")
          endif()
        endforeach()
      endif()
    elseif(${CUDA_ARCH} IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      if(${CUDA_ARCH} GREATER_EQUAL ${MIN_ARCHITECTURE_HAS_ACCEL})
        list(APPEND CUDA_ARCHITECTURES "${CUDA_ARCH}a-real")
      else()
        list(APPEND CUDA_ARCHITECTURES "${CUDA_ARCH}-real")
      endif()
    endif()
  endforeach()
  if("${CUDA_ARCHITECTURES}" STREQUAL "")
    # We have to at least build for some architectures.
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES "80-real")
    target_compile_definitions(${target} PRIVATE PLACEHOLDER_KERNELS)
  else()
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES
                                           ${CUDA_ARCHITECTURES})
  endif()
endfunction()
