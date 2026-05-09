# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Pull cubin payloads into a target via INCBIN. Cubins are committed as per-file
# `<stem>.cubin.tar.zst` git-LFS objects; this module decompresses them at build
# time and arranges for the bytes to be linked in.
#
# Usage (from a per-directory CMakeLists.txt):
#
# include(tllm_cubin_archive) tllm_add_cubin_archive_sources(<target>
# <archive_dir> [SYMBOL_PREFIX <prefix>]   # default: "" (xqa, trtllmGen)
# [SYMBOL_SUFFIX <suffix>]   # default: "_cubin" [NAMESPACE <ns> ...]       #
# default: tensorrt_llm _v1 kernels [ARCHS <a> [<b> ...]]      # forwarded to
# filter_source_cuda_architectures [IMPLICIT_FAMILY])         # forwarded to
# filter_source_cuda_architectures
#
# Symbol resolution: For each `<archive_dir>/<stem>.cubin.tar.zst`, the *C++*
# symbol embedded by INCBIN is
# `<NAMESPACE>::${SYMBOL_PREFIX}${stem}${SYMBOL_SUFFIX}`. The contextFMHA
# convention (`tensorrt_llm::_v1::kernels::cubin_<X>_cu_cubin`) is reproduced
# with PREFIX=`cubin_`, SUFFIX=`_cu_cubin`, and the default NAMESPACE.
#
# The aggregator's asm block emits the *Itanium-mangled* linker name for the
# namespaced symbol (e.g., `_ZN12tensorrt_llm3_v17kernels<len><sym>E`), so
# consumer translation units that declare the same name inside the same
# namespace -- without any extra annotation -- have the compiler mangle their
# references to the same linker symbol. This protects against multi-definition
# errors when two packages, or two TRT-LLM ABI versions, end up in the same
# final link.
#
# Architecture filtering: When ARCHS is supplied, the tarball list is run
# through filter_source_cuda_architectures (the same helper used to drop
# disabled-arch .cu / .cubin.cpp files), so cubins matching SMs that the build
# does not enable are dropped at configure time. The `EXCLUDE_SM_<arch>`
# definitions that helper adds to ${TARGET} stay in sync with what the consumer
# expects.
#
# What this hands to the linker: * <mangled symbol>      - aligned(64) byte
# array, the raw cubin * <mangled symbol>_end  - one-past-the-end pointer
# (mangled too) * <mangled symbol>_len  - 32-bit byte count        (mangled too)

if(DEFINED _TLLM_CUBIN_ARCHIVE_INCLUDED)
  return()
endif()
set(_TLLM_CUBIN_ARCHIVE_INCLUDED TRUE)

get_filename_component(_TLLM_CUBIN_REPO_ROOT
                       "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# `TRTLLM_ABI_NAMESPACE` is the project-wide cache variable defined in
# cpp/CMakeLists.txt. Both the C++ macro (via config.h) and the cubin helper's
# mangled prefix derive from it; this module just reads the value.
if(NOT DEFINED TRTLLM_ABI_NAMESPACE)
  message(
    FATAL_ERROR
      "tllm_cubin_archive: TRTLLM_ABI_NAMESPACE is not set. The top-level "
      "cpp/CMakeLists.txt should define it before any cubin helper is used.")
endif()

function(tllm_add_cubin_archive_sources TARGET ARCHIVE_DIR)
  cmake_parse_arguments(_ARG "IMPLICIT_FAMILY" "SYMBOL_PREFIX;SYMBOL_SUFFIX"
                        "ARCHS;NAMESPACE" ${ARGN})
  if(NOT DEFINED _ARG_SYMBOL_PREFIX)
    set(_ARG_SYMBOL_PREFIX "")
  endif()
  if(NOT DEFINED _ARG_SYMBOL_SUFFIX)
    set(_ARG_SYMBOL_SUFFIX "_cubin")
  endif()
  if(NOT _ARG_NAMESPACE)
    set(_ARG_NAMESPACE tensorrt_llm ${TRTLLM_ABI_NAMESPACE} kernels)
  endif()

  # Itanium ABI mangling for `ns1::ns2::...::<sym>` is
  # `_ZN<len(ns1)><ns1><len(ns2)><ns2>...<len(sym)><sym>E` Inline namespaces are
  # not specially encoded; they look exactly like regular ones at the linker
  # level. Build the namespace prefix once; we'll append `<sym_len><sym>E` per
  # cubin below.
  set(_MANGLED_NS_PREFIX "_ZN")
  foreach(_NS ${_ARG_NAMESPACE})
    string(LENGTH "${_NS}" _NS_LEN)
    string(APPEND _MANGLED_NS_PREFIX "${_NS_LEN}${_NS}")
  endforeach()
  if(NOT TARGET ${TARGET})
    message(
      FATAL_ERROR "tllm_add_cubin_archive_sources: '${TARGET}' is not a target")
  endif()
  get_filename_component(ARCHIVE_DIR "${ARCHIVE_DIR}" ABSOLUTE)

  file(
    GLOB _ARCHIVES CONFIGURE_DEPENDS
    RELATIVE "${ARCHIVE_DIR}"
    "${ARCHIVE_DIR}/*.cubin.tar.zst")
  list(SORT _ARCHIVES)
  if(NOT _ARCHIVES)
    message(
      FATAL_ERROR
        "tllm_add_cubin_archive_sources: no *.cubin.tar.zst tarballs found in "
        "${ARCHIVE_DIR}. The producer (extract_cubins.py / fmha_v2 Makefile / "
        "xqa gen_cubins.py) needs to have run.")
  endif()

  # Selective build by SM. filter_source_cuda_architectures matches the
  # `sm_?<arch>` substring in each filename and removes cubins for archs that
  # the current build doesn't enable. It also stamps EXCLUDE_SM_<arch> on
  # ${TARGET}, which keeps the consumer .cpp's `#ifndef EXCLUDE_SM_<n>` gates
  # honest. Skipped entirely when ARCHS is empty -- in that case all globbed
  # cubins are embedded (existing trtllmGen behavior).
  if(_ARG_ARCHS)
    include(cuda_configuration)
    if(_ARG_IMPLICIT_FAMILY)
      filter_source_cuda_architectures(
        SOURCE_LIST _ARCHIVES
        ARCHS ${_ARG_ARCHS}
        IMPLICIT_FAMILY)
    else()
      filter_source_cuda_architectures(SOURCE_LIST _ARCHIVES
                                       ARCHS ${_ARG_ARCHS})
    endif()
    if(NOT _ARCHIVES)
      message(
        STATUS "tllm_add_cubin_archive_sources: no cubins remain for ${TARGET} "
               "after architecture filtering -- emitting an empty aggregator.")
    endif()
  endif()

  # Build-tree extraction directory. CMAKE_CURRENT_BINARY_DIR is already
  # per-CMakeLists, so a flat `cubins/` suffix is enough -- no need to mirror
  # the source-tree path here.
  set(EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cubins")
  file(MAKE_DIRECTORY "${EXTRACT_DIR}")

  # Generate the aggregator .cpp at configure time. Its only contents are
  # `#include "tensorrt_llm/common/cubinIncbin.h"` plus one TLLM_INCBIN per
  # tarball. We use `configure_file` so the file is regenerated whenever any
  # tarball name changes (the `_aggregator_lines` string baked into it changes),
  # but not on every reconfigure if the set is unchanged.
  set(AGGREGATOR_CPP
      "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_cubin_aggregator.cpp")

  set(_EXTRACTED_CUBINS "")
  set(_INCBIN_LINES "")
  foreach(_ARCHIVE ${_ARCHIVES})
    string(REGEX REPLACE "\\.cubin\\.tar\\.zst$" "" _STEM "${_ARCHIVE}")
    set(_CUBIN_FILE "${_STEM}.cubin")
    set(_EXTRACTED "${EXTRACT_DIR}/${_CUBIN_FILE}")
    set(_ARCHIVE_PATH "${ARCHIVE_DIR}/${_ARCHIVE}")
    set(_SYMBOL "${_ARG_SYMBOL_PREFIX}${_STEM}${_ARG_SYMBOL_SUFFIX}")

    # `cmake -E tar` honors the entry mtimes inside the archive. Producers pin
    # those to 1970-01-01 (so unchanged cubin bytes yield byte-identical
    # tarballs that git/LFS can dedupe), which would otherwise leave every
    # extracted cubin stuck at the epoch and break ninja's mtime tracking.
    # `touch -r <tarball> <cubin>` propagates the tarball file's filesystem
    # mtime onto the extracted cubin -- the tarball's mtime *does* advance
    # whenever the archive was rewritten, so this is the correct dependency
    # signal. ninja's `restat` then sees the cubin mtime change exactly when the
    # tarball was actually replaced, neither over- nor under- firing the
    # downstream .o rebuild. POSIX `touch -r` is universal on Linux, which is
    # the only platform cubinIncbin.h supports anyway.
    add_custom_command(
      OUTPUT "${_EXTRACTED}"
      COMMAND ${CMAKE_COMMAND} -E tar xf "${_ARCHIVE_PATH}"
      COMMAND touch -r "${_ARCHIVE_PATH}" "${_EXTRACTED}"
      WORKING_DIRECTORY "${EXTRACT_DIR}"
      DEPENDS "${_ARCHIVE_PATH}"
      COMMENT "Extracting cubin: ${_CUBIN_FILE}"
      VERBATIM)
    list(APPEND _EXTRACTED_CUBINS "${_EXTRACTED}")

    # Compute the three mangled linker names.
    string(LENGTH "${_SYMBOL}" _SYMLEN)
    set(_SYM_END "${_SYMBOL}_end")
    set(_SYM_LEN "${_SYMBOL}_len")
    string(LENGTH "${_SYM_END}" _SYM_END_LEN)
    string(LENGTH "${_SYM_LEN}" _SYM_LEN_LEN)
    set(_ASM_DATA "${_MANGLED_NS_PREFIX}${_SYMLEN}${_SYMBOL}E")
    set(_ASM_END "${_MANGLED_NS_PREFIX}${_SYM_END_LEN}${_SYM_END}E")
    set(_ASM_LEN "${_MANGLED_NS_PREFIX}${_SYM_LEN_LEN}${_SYM_LEN}E")

    string(
      APPEND
      _INCBIN_LINES
      "TLLM_INCBIN_NS(${_SYMBOL}, \"${_ASM_DATA}\", \"${_ASM_END}\", \"${_ASM_LEN}\", \"${_CUBIN_FILE}\");\n"
    )
  endforeach()

  # Build the C++ namespace open/close lines. We emit plain `namespace` for
  # every segment -- inline-namespace semantics affect name lookup, not
  # mangling, so the resulting linker symbols match TRTLLM_NAMESPACE_BEGIN
  # consumers (which DO use `inline namespace _v1`). Plain `namespace` here is
  # also the safe choice because it doesn't invent inline-ness for segments that
  # aren't actually inline in the consumer (e.g. `gemm`, `batchedGemm`, ...).
  set(_NS_OPEN "")
  set(_NS_CLOSE "")
  foreach(_NS ${_ARG_NAMESPACE})
    string(APPEND _NS_OPEN "namespace ${_NS} { ")
    string(APPEND _NS_CLOSE "} ")
  endforeach()
  string(REPLACE ";" "::" _NS_DOC "${_ARG_NAMESPACE}")

  # Build the aggregator body as one string. `file(GENERATE)` flattens lists by
  # joining with `;`, so we must keep the arguments as a single string rather
  # than a list of lines.
  set(_AGG_CONTENT
      "// SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES.\n"
  )
  string(APPEND _AGG_CONTENT "// SPDX-License-Identifier: Apache-2.0\n")
  string(APPEND _AGG_CONTENT "//\n")
  string(
    APPEND
    _AGG_CONTENT
    "// AUTO-GENERATED by tllm_cubin_archive.cmake from the *.cubin.tar.zst set in\n"
  )
  string(APPEND _AGG_CONTENT "//   ${ARCHIVE_DIR}\n")
  string(APPEND _AGG_CONTENT "// Symbols land in C++ namespace: ${_NS_DOC}\n")
  string(APPEND _AGG_CONTENT
         "// Do not edit; regenerate by re-running cmake.\n\n")
  string(APPEND _AGG_CONTENT
         "#include \"tensorrt_llm/common/cubinIncbin.h\"\n\n")
  string(APPEND _AGG_CONTENT "${_NS_OPEN}\n")
  string(APPEND _AGG_CONTENT "${_INCBIN_LINES}")
  string(APPEND _AGG_CONTENT "${_NS_CLOSE}// namespace ${_NS_DOC}\n")
  file(
    GENERATE
    OUTPUT "${AGGREGATOR_CPP}"
    CONTENT "${_AGG_CONTENT}")

  # Tie the aggregator .cpp's compilation to the extracted cubin files: the
  # assembler runs `.incbin "<stem>.cubin"` at compile time and resolves the
  # filename via the COMPILE_OPTIONS `-Wa,-I${EXTRACT_DIR}` we set below.
  # OBJECT_DEPENDS makes ninja re-run the .o build whenever any extracted
  # cubin's mtime changes (which only happens when its tarball changed content,
  # since `cmake -E tar` preserves entry mtimes).
  set_source_files_properties(
    "${AGGREGATOR_CPP}"
    PROPERTIES OBJECT_DEPENDS "${_EXTRACTED_CUBINS}" GENERATED TRUE
               COMPILE_OPTIONS "-Wa,-I${EXTRACT_DIR}")

  target_sources(${TARGET} PRIVATE "${AGGREGATOR_CPP}")
  target_include_directories(${TARGET}
                             PRIVATE "${_TLLM_CUBIN_REPO_ROOT}/cpp/include")
endfunction()
