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

#[=======================================================================[.rst:
CudaConfiguration
-----------------

CUDA compiler and architecture configuration for TensorRT-LLM.

This module provides functions and macros to configure the CUDA compiler,
manage CUDA architectures, and filter source files based on target
architectures. It is tailored to meet TensorRT-LLM's specific requirements
for optimized kernel compilation across multiple GPU generations.

Macros
^^^^^^

.. command:: setup_cuda_compiler

  Detects and validates the CUDA compiler::

    setup_cuda_compiler()

  This macro determines the CUDA compiler version before enabling the CUDA
  language extension. It requires CUDA version 11.2 or later.

  The macro sets ``CMAKE_CUDA_COMPILER_VERSION`` upon successful detection.

Functions
^^^^^^^^^

.. command:: setup_cuda_architectures

  Initializes and normalizes ``CMAKE_CUDA_ARCHITECTURES``::

    setup_cuda_architectures()

  This function processes the ``CMAKE_CUDA_ARCHITECTURES`` variable and
  configures architecture-specific compilation settings. This function should
  be called after enabling the CUDA language extension.

  **Special Values for CMAKE_CUDA_ARCHITECTURES:**

  ``native``
    Resolves to the highest available architecture on the system.
    Falls back to ``all`` if detection fails.

  ``all`` or unset
    Resolves to architectures TensorRT-LLM is optimized for and the
    compiler supports (80, 86, 89, 90, 100, 103, 120 depending on CUDA version).

  ``all-major``
    Unsupported. Results in a fatal error.

  **Architecture Processing:**

  * PTX is never included in the result binary (``-virtual`` rejected).
  * The ``-real`` suffix is automatically added to exclude PTX.
  * Accelerated targets (``-a`` suffix) are used for SM 90+.
  * On CUDA 12.9+, family targets (``-f`` suffix) are used for SM 100+.

  **Output Variables (set in parent scope):**

  ``CMAKE_CUDA_ARCHITECTURES``
    Normalized list with appropriate suffixes (e.g., ``80-real``, ``90a-real``,
    ``100f-real``).

  ``CMAKE_CUDA_ARCHITECTURES_ORIG``
    Original list of enabled architectures without suffixes.

  ``CMAKE_CUDA_ARCHITECTURES_FAMILIES``
    List of family architectures (e.g., ``100f``, ``120f``).

  ``CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES``
    Boolean indicating if family targets are supported.

  ``CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL``
    Minimum architecture supporting accelerated (``-a``) suffix.

  ``CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY``
    Minimum architecture supporting family (``-f``) suffix.

.. command:: add_cuda_architectures

  Appends CUDA architectures to an existing target::

    add_cuda_architectures(<target> <arch1> [<arch2> ...])

  Adds the specified architectures to ``<target>``'s ``CUDA_ARCHITECTURES``
  property. The ``-a`` suffix is automatically added for supported
  architectures. Architectures are only added if they were explicitly
  requested by the user in ``CMAKE_CUDA_ARCHITECTURES_ORIG``.

.. command:: set_cuda_architectures

  Sets CUDA architectures for a target::

    set_cuda_architectures(<target> <arch1> [<arch2> ...])

  Replaces the ``CUDA_ARCHITECTURES`` property of ``<target>`` with the
  specified architectures.

  **Architecture Specification:**

  * Architectures may include the ``f`` suffix for family-conditional
    compilation (e.g., ``100f``).
  * Non-family architectures are only added if explicitly requested.
  * Family architectures are only added if requested architectures would
    enable compilation for that family.

  If no architectures are enabled for the target, it compiles with
  ``PLACEHOLDER_KERNELS`` macro defined. The kernel source shall compile
  with any architecture if ``PLACEHOLDER_KERNELS`` macro is defined.

.. command:: filter_source_cuda_architectures

  Filters source files based on enabled CUDA architectures::

    filter_source_cuda_architectures(
      SOURCE_LIST <variable>
      TARGET <target>
      ARCHS <arch1> [<arch2> ...]
      [IMPLICIT_FAMILY]
    )

  Removes source files targeting disabled CUDA architectures from the
  source list. Files are matched by patterns like ``sm80``, ``sm_80``,
  ``SM80``, etc. in their filenames (for ``.cu`` and ``cubin.cpp`` files).

  ``SOURCE_LIST <variable>``
    Name of the variable containing the list of source files.
    Modified in place to remove filtered files.

  ``TARGET <target>``
    Target to add compile definitions to. If the target does not exist,
    an INTERFACE library will be created.

  ``ARCHS <arch1> [<arch2> ...]``
    List of architectures to check. May include ``f`` suffix.

  ``IMPLICIT_FAMILY``
    When set, treats architectures >= ``CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY``
    as implicitly family-enabled.

  **Defined Macros:**

  For each filtered architecture, a compile definition ``EXCLUDE_SM_<ARCH>``
  (or ``EXCLUDE_SM_<ARCH>F`` for family architectures) is added to ``<target>``.

Example
^^^^^^^

.. code-block:: cmake

  include(cuda_configuration)

  # Setup compiler and detect version
  setup_cuda_compiler()

  # enable_language, or project(project_name LANGUAGES CUDA)
  # must be called after setup_cuda_compiler() and before
  # setup_cuda_architectures()
  enable_language(CUDA)

  # Configure architectures (uses CMAKE_CUDA_ARCHITECTURES if set)
  setup_cuda_architectures()

  # Add additional architecture to compile for, if it is beneficial.
  # e.g. Utilizing native FP8 support available in sm89 (Ada)
  # but not in sm86 (Ampere)
  # Note: The kernel source must still compiles for all the architectures,
  # by using less performant implementation.
  add_library(my_kernels_fp8 STATIC kernels.cu)
  add_cuda_architectures(my_kernels_fp8 89)

  # Set specific architecture this source should compile for.
  # e.g. Kernels using WGMMA instructions
  # Note: The kernel source must still compiles for other architectures when
  # ``PLACEHOLDER_KERNELS`` macro is defined.
  add_library(my_kernels_sm90_only STATIC kernels.cu)
  set_cuda_architectures(my_kernels_sm90_only 90)

  # Filter sources for disabled architectures
  set(KERNEL_SOURCES
    kernel_sm80.cubin.cpp
    kernel_sm90.cubin.cpp
    kernel_sm100.cubin.cpp
  )
  filter_source_cuda_architectures(
    SOURCE_LIST KERNEL_SOURCES
    TARGET my_kernel_interface
    ARCHS 80 90 100
  )
  # ``my_kernel_interface`` target is created with definitions to exclude
  # disabled architectures.

#]=======================================================================]

#[[
Determine CUDA version before enabling the language extension
check_language(CUDA) clears CMAKE_CUDA_HOST_COMPILER if CMAKE_CUDA_COMPILER
is not set
#]]
macro(setup_cuda_compiler)
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

#[[
Initialize and normalize CMAKE_CUDA_ARCHITECTURES.

Special values:

* `native` is resolved to HIGHEST available architecture.
  * Fallback to `all` if detection failed.
* `all`/unset is resolved to a set of architectures we optimized for and compiler supports.
* `all-major` is unsupported.

Numerical architectures:

* PTX is never included in result binary.
  * `*-virtual` architectures are therefore rejected.
  * `-real` suffix is automatically added to exclude PTX.
* Always use accelerated (`-a` suffix) target for supported architectures.
* On CUDA 12.9 or newer, family (`-f` suffix) target will be used for supported architectures to reduce number of
  targets to compile for.
  * Extra architectures can be requested via add_cuda_architectures
    for kernels that benefit from arch specific features.
#]]
function(setup_cuda_architectures)
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
  set(CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL 90)
  set(CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL
      ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL}
      PARENT_SCOPE)
  # -f suffix supported from Blackwell (100) starting from CUDA 12.9.
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    set(CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY 100)
    set(CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES
        ON
        PARENT_SCOPE)
  else()
    # -a provides no cross architecture compatibility, but luckily until CUDA
    # 12.8 We have only one architecture within each family >= 9.
    set(CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY 9999) # Effectively exclude all
                                                     # architectures
    set(CMAKE_CUDA_ARCHITECTURES_HAS_FAMILIES
        OFF
        PARENT_SCOPE)
  endif()
  set(CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY
      ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY}
      PARENT_SCOPE)

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
    if(CUDA_ARCH GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY}
       AND NOT CUDA_ARCH IN_LIST ARCHITECTURES_NO_COMPATIBILITY)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}f-real")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_FAMILIES "${CUDA_ARCH}f")
    elseif(CUDA_ARCH GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL})
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

#[[
Add CUDA architectures to target.
-a suffix is added automatically for supported architectures.
Architectures are added only if user explicitly requested support for that architecture.
#]]
function(add_cuda_architectures target)
  foreach(CUDA_ARCH IN LISTS ARGN)
    if(${CUDA_ARCH} IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      if(${CUDA_ARCH} GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL})
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

#[[
Set CUDA architectures for a target.

-a suffix is added automatically for supported architectures.
Architectures passed in may be specified with -f suffix to build family conditional version of the kernel.

Non-family architectures are added only if user explicitly requested support for that architecture.
Family conditional architectures are only added if user requested architectures would enable compilation for it.

If user requested no architectures set on the target,
the target will be compiled with `PLACEHOLDER_KERNELS` macro defined.
#]]
function(set_cuda_architectures target)
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
      if(${CUDA_ARCH} GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_ACCEL})
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

#[[
Filter out source files targeting CUDA architectures not enabled.

Arguments:
  SOURCE_LIST - Name of the variable containing the list of source files to filter
  TARGET      - Target to add compile definitions to. If the target does not exist,
                an INTERFACE library will be created.
  ARCHS       - List of architectures to check and potentially filter
  IMPLICIT_FAMILY - Optional flag to enable implicit family mode

For each ARCH passed in:

- if IMPLICIT_FAMILY is not set:
  - if ARCH is not suffixed by f:
    if ARCH is not in CMAKE_CUDA_ARCHITECTURES_ORIG, source files containing "sm${ARCH}"
    but not "sm${ARCH}f" (case insensitive) will be excluded
    Macro "EXCLUDE_SM_${ARCH}" will be defined on TARGET
  - if ARCH is suffixed by f, NARCH is ARCH without f suffix:
    if ARCH is not in CMAKE_CUDA_ARCHITECTURES_FAMILIES, source files containing
    "sm${NARCH}f" (case insensitive) will be excluded
    Macro "EXCLUDE_SM_${NARCH}F" will be defined on TARGET

- if IMPLICIT_FAMILY is set:
  ARCH shall not suffixed by f.
  - if ARCH >= CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY:
    if "${ARCH}f" is not in CMAKE_CUDA_ARCHITECTURES_FAMILIES,
    source files containing "sm${ARCH}" but not "sm${ARCH}a" (case insensitive) will be excluded
    Macro "EXCLUDE_SM_${ARCH}" (no F) will be defined on TARGET
  - else:
    if "${ARCH}" is not in CMAKE_CUDA_ARCHITECTURES_ORIG,
    source files containing "sm${ARCH}" (case insensitive) will be excluded
    Macro "EXCLUDE_SM_${ARCH}" will be defined on TARGET
#]]
function(filter_source_cuda_architectures)
  set(options IMPLICIT_FAMILY)
  set(oneValueArgs SOURCE_LIST TARGET)
  set(multiValueArgs ARCHS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")
  set(SOURCES "${${arg_SOURCE_LIST}}")

  if(NOT TARGET ${arg_TARGET})
    add_library(${arg_TARGET} INTERFACE)
  endif()

  # Determine if target is INTERFACE library to use correct visibility
  get_target_property(_target_type ${arg_TARGET} TYPE)
  if(_target_type STREQUAL "INTERFACE_LIBRARY")
    set(_compile_def_visibility INTERFACE)
  else()
    set(_compile_def_visibility PUBLIC)
  endif()

  foreach(ARCH IN LISTS arg_ARCHS)
    set(SHOULD_FILTER FALSE)
    set(MATCH_PATTERN "")
    set(EXCLUDE_PATTERN "")
    set(ARCH_FOR_DEFINE "")

    if(NOT arg_IMPLICIT_FAMILY)
      # Check if ARCH ends with 'f'
      string(REGEX MATCH "^(.+)f$" _has_f_suffix "${ARCH}")

      if(_has_f_suffix)
        # ARCH is suffixed by 'f' (e.g., "100f")
        set(BASE_ARCH "${CMAKE_MATCH_1}")
        if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_FAMILIES)
          set(SHOULD_FILTER TRUE)
          set(ARCH_FOR_DEFINE "${BASE_ARCH}F")
          # Match "sm${BASE_ARCH}f" - straightforward match, no exclusion
          # pattern needed
          set(MATCH_PATTERN ".*[Ss][Mm]_?${BASE_ARCH}f.*(cubin\.cpp|\.cu)$")
        endif()
      else()
        # ARCH is NOT suffixed by 'f' (e.g., "80")
        if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
          set(SHOULD_FILTER TRUE)
          set(ARCH_FOR_DEFINE "${ARCH}")
          # Match "sm${ARCH}" but NOT "sm${ARCH}f"
          set(MATCH_PATTERN ".*[Ss][Mm]_?${ARCH}.*(cubin\.cpp|\.cu)$")
          set(EXCLUDE_PATTERN ".*[Ss][Mm]_?${ARCH}f.*(cubin\.cpp|\.cu)$")
        endif()
      endif()
    else()
      # IMPLICIT_FAMILY is set - ARCH shall not be suffixed by 'f'
      if(${ARCH} GREATER_EQUAL ${CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY})
        # ARCH >= CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY
        if(NOT "${ARCH}f" IN_LIST CMAKE_CUDA_ARCHITECTURES_FAMILIES)
          set(SHOULD_FILTER TRUE)
          set(ARCH_FOR_DEFINE "${ARCH}")
          # Match "sm${ARCH}" but NOT "sm${ARCH}a"
          set(MATCH_PATTERN ".*[Ss][Mm]_?${ARCH}.*(cubin\.cpp|\.cu)$")
          set(EXCLUDE_PATTERN ".*[Ss][Mm]_?${ARCH}a.*(cubin\.cpp|\.cu)$")
        endif()
      else()
        # ARCH < CMAKE_CUDA_MIN_ARCHITECTURE_HAS_FAMILY
        if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
          set(SHOULD_FILTER TRUE)
          set(ARCH_FOR_DEFINE "${ARCH}")
          # Match "sm${ARCH}" - no exclusion pattern needed
          set(MATCH_PATTERN ".*[Ss][Mm]_?${ARCH}.*(cubin\.cpp|\.cu)$")
        endif()
      endif()
    endif()

    if(SHOULD_FILTER)
      # Get files matching the main pattern
      set(SOURCES_TO_CHECK "${SOURCES}")
      list(FILTER SOURCES_TO_CHECK INCLUDE REGEX "${MATCH_PATTERN}")

      if(NOT "${EXCLUDE_PATTERN}" STREQUAL "")
        # Find files matching the exclusion pattern (these should be kept)
        set(SOURCES_TO_KEEP "${SOURCES_TO_CHECK}")
        list(FILTER SOURCES_TO_KEEP INCLUDE REGEX "${EXCLUDE_PATTERN}")
        # Remove the files we want to keep from the check list
        if(SOURCES_TO_KEEP)
          list(REMOVE_ITEM SOURCES_TO_CHECK ${SOURCES_TO_KEEP})
        endif()
      endif()

      set(SOURCES_FILTERED "${SOURCES_TO_CHECK}")

      list(LENGTH SOURCES_FILTERED SOURCES_FILTERED_LEN)
      message(
        STATUS
          "Excluding ${SOURCES_FILTERED_LEN} cubins for SM ${ARCH} from ${CMAKE_CURRENT_SOURCE_DIR}"
      )
      foreach(filtered_item IN LISTS SOURCES_FILTERED)
        message(VERBOSE "- ${filtered_item}")
      endforeach()

      # Remove filtered files from sources
      if(SOURCES_FILTERED)
        list(REMOVE_ITEM SOURCES ${SOURCES_FILTERED})
      endif()

      # Add compile definition to target
      target_compile_definitions(
        ${arg_TARGET}
        ${_compile_def_visibility}
        "EXCLUDE_SM_${ARCH_FOR_DEFINE}")
    endif()
  endforeach()

  set(${arg_SOURCE_LIST}
      "${SOURCES}"
      PARENT_SCOPE)
endfunction()
