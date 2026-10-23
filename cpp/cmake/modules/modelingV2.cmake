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
Native (C++/CUDA) code of modeling_v2 targets.

Each model family under cpp/tensorrt_llm/modeling_v2/<familyCamel>/ builds into
one MODULE library, libtrtllm_mv2_<family>.so. Nothing in it is linked into
libtensorrt_llm.so or libth_common.so; the Python loader
(tensorrt_llm/_torch/_experimental/modeling_v2/_native.py) opens it only when
routing picks one of the family's targets. cpp/tensorrt_llm/modeling_v2/README.md
is the guide for authors.

A family's CMakeLists.txt uses three functions. Names are the Python package
names (snake_case); directories are their camelCase form:

  trtllm_mv2_family(<family>)
  trtllm_mv2_shared(<family> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])
  trtllm_mv2_target(<family> <target> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])

ARCHS     NN, NNa or NNf. Each is intersected with the build's requested
          architectures; a target with an empty intersection is not built and
          the manifest says why. No placeholder kernels are generated.
SOURCES   Paths or globs relative to the target directory (shared/ for
          trtllm_mv2_shared). .cu files are compiled for ARCHS without torch;
          .cpp files are the torch bindings.
DEPS      From a closed list: cublas, cublaslt, cuda_driver, internal_cutlass,
          nccl, trtllm_gen_fmha_interface.
REQUIRES  Build features: MULTI_DEVICE, NVSHMEM. Unmet requirements skip the
          target.

The stock side of the build calls trtllm_mv2_define_contract_fingerprint()
before thop, then adds cpp/tensorrt_llm/modeling_v2, which calls
trtllm_mv2_finalize() after the families.
#]]

include_guard(GLOBAL)

set(_TRTLLM_MV2_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(_TRTLLM_MV2_KNOWN_DEPS cublas cublaslt cuda_driver internal_cutlass nccl
                           trtllm_gen_fmha_interface)
set(_TRTLLM_MV2_KNOWN_REQUIRES MULTI_DEVICE NVSHMEM)
set(_TRTLLM_MV2_MANIFEST_SCHEMA 1)

# --- Naming ------------------------------------------------------------------

# camelCase form of a snake_case name: minimax_m3_nvfp4__sm_107__tp2 ->
# minimaxM3Nvfp4Sm107Tp2. The same mapping as camel() in the Python checks.
function(_trtllm_mv2_camel out name)
  string(REPLACE "_" ";" parts "${name}")
  set(result "")
  foreach(part IN LISTS parts)
    if(part STREQUAL "")
      continue()
    endif()
    if(result STREQUAL "")
      set(result "${part}")
    else()
      string(SUBSTRING "${part}" 0 1 head)
      string(SUBSTRING "${part}" 1 -1 tail)
      string(TOUPPER "${head}" head)
      string(APPEND result "${head}${tail}")
    endif()
  endforeach()
  set(${out}
      "${result}"
      PARENT_SCOPE)
endfunction()

# A name usable as a C++ or torch identifier: "__" collapsed to "_", since C++
# reserves identifiers that contain a double underscore.
function(_trtllm_mv2_ident out name)
  set(result "${name}")
  while(result MATCHES "__")
    string(REPLACE "__" "_" result "${result}")
  endwhile()
  set(${out}
      "${result}"
      PARENT_SCOPE)
endfunction()

function(_trtllm_mv2_check_family_name family)
  if(NOT family MATCHES "^[a-z][a-z0-9_]*[a-z0-9]$" OR family MATCHES "__")
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' is not a snake_case package name")
  endif()
endfunction()

# <checkpoint>__<sm>__<parallel>, the identity the Python target directory
# carries (see tensorrt_llm/_torch/_experimental/modeling_v2/README.md).
function(_trtllm_mv2_check_target_name target)
  string(REPLACE "__" ";" segments "${target}")
  list(LENGTH segments count)
  if(NOT count EQUAL 3 OR NOT target MATCHES "^[a-z][a-z0-9_]*$")
    message(
      FATAL_ERROR
        "modeling_v2: target '${target}' is not a <checkpoint>__<sm>__<parallel> name"
    )
  endif()
  list(GET segments 1 sm)
  if(NOT sm MATCHES "^sm_[0-9]+[a-z0-9]*$")
    message(
      FATAL_ERROR
        "modeling_v2: target '${target}' has no sm_<arch> segment (got '${sm}')"
    )
  endif()
endfunction()

# --- Selection, requirements, architectures -----------------------------------

# Whether TRTLLM_MODELING_V2_TARGETS selects <family>/<target>.
function(_trtllm_mv2_selected out family target)
  set(filter "${TRTLLM_MODELING_V2_TARGETS}")
  if(filter STREQUAL "" OR filter STREQUAL "all")
    set(${out}
        TRUE
        PARENT_SCOPE)
    return()
  endif()
  if(filter STREQUAL "none")
    set(${out}
        FALSE
        PARENT_SCOPE)
    return()
  endif()
  set(selected FALSE)
  foreach(pattern IN LISTS filter)
    # Glob to regex: escape the dots, then * -> [^/]*, ? -> [^/].
    string(REPLACE "." "\\." regex "${pattern}")
    string(REPLACE "*" "[^/]*" regex "${regex}")
    string(REPLACE "?" "[^/]" regex "${regex}")
    if("${family}/${target}" MATCHES "^${regex}$")
      set(selected TRUE)
    endif()
  endforeach()
  set(${out}
      ${selected}
      PARENT_SCOPE)
endfunction()

# Empty when every requirement holds, otherwise the reason.
function(_trtllm_mv2_unmet_requirements out)
  set(reasons "")
  foreach(requirement IN LISTS ARGN)
    if(NOT requirement IN_LIST _TRTLLM_MV2_KNOWN_REQUIRES)
      message(
        FATAL_ERROR
          "modeling_v2: unknown REQUIRES '${requirement}' (known: ${_TRTLLM_MV2_KNOWN_REQUIRES})"
      )
    endif()
    if(requirement STREQUAL "MULTI_DEVICE" AND NOT ENABLE_MULTI_DEVICE)
      list(APPEND reasons "built with ENABLE_MULTI_DEVICE=OFF")
    elseif(requirement STREQUAL "NVSHMEM" AND NOT ENABLE_NVSHMEM)
      list(APPEND reasons "built with ENABLE_NVSHMEM=OFF")
    endif()
  endforeach()
  list(JOIN reasons ", " reasons)
  set(${out}
      "${reasons}"
      PARENT_SCOPE)
endfunction()

#[[
Intersect declared architectures with the requested ones.

  NN   -> NN-real   if NN is requested
  NNa  -> NNa-real  if NN is requested
  NNf  -> NNf-real  if a requested architecture is in family NN: the same
                    major version and a minor version >= NN's

`requested` defaults to CMAKE_CUDA_ARCHITECTURES_ORIG, the plain numbers the
build was asked for. This deliberately does not reuse set_cuda_architectures(),
which drops NNa silently and falls back to placeholder kernels.
#]]
function(_trtllm_mv2_resolve_archs out)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "DECLARED;REQUESTED")
  if(NOT DEFINED arg_REQUESTED)
    set(arg_REQUESTED ${CMAKE_CUDA_ARCHITECTURES_ORIG})
  endif()
  set(resolved "")
  foreach(arch IN LISTS arg_DECLARED)
    if(NOT arch MATCHES "^([0-9]+)([af]?)$")
      message(
        FATAL_ERROR "modeling_v2: ARCHS entry '${arch}' is not NN, NNa or NNf")
    endif()
    set(number "${CMAKE_MATCH_1}")
    set(suffix "${CMAKE_MATCH_2}")
    if(suffix STREQUAL "f")
      math(EXPR family_major "${number} / 10")
      set(in_family FALSE)
      foreach(requested IN LISTS arg_REQUESTED)
        math(EXPR requested_major "${requested} / 10")
        if(requested_major EQUAL family_major AND requested GREATER_EQUAL
                                                  number)
          set(in_family TRUE)
        endif()
      endforeach()
      if(in_family)
        list(APPEND resolved "${number}f-real")
      endif()
    elseif(number IN_LIST arg_REQUESTED)
      list(APPEND resolved "${number}${suffix}-real")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES resolved)
  set(${out}
      "${resolved}"
      PARENT_SCOPE)
endfunction()

# --- Contract fingerprint
# ------------------------------------------------------

#[[
Define the target `trtllm_mv2_contract_fingerprint`, which generates
${TRTLLM_MV2_GENERATED_DIR}/mv2ContractFingerprint.h.

The fingerprint hashes what a family library is compiled against on the stock
side: the stock headers listed in cpp/tensorrt_llm/modeling_v2/contractSurface.txt,
the global compile definitions, and the torch and CUDA versions. th_common and
every family library report it, and the loader refuses a family library whose
fingerprint differs from the loaded th_common's.

Call from cpp/tensorrt_llm/CMakeLists.txt before thop; it sets
TRTLLM_MV2_GENERATED_DIR in the caller's scope.
#]]
function(trtllm_mv2_define_contract_fingerprint)
  set(generated_dir "${CMAKE_BINARY_DIR}/tensorrt_llm/modeling_v2/generated")
  set(surface_file
      "${PROJECT_SOURCE_DIR}/tensorrt_llm/modeling_v2/contractSurface.txt")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${surface_file}")

  file(STRINGS "${surface_file}" patterns REGEX "^[^#]")
  set(headers "")
  foreach(pattern IN LISTS patterns)
    string(STRIP "${pattern}" pattern)
    if(pattern STREQUAL "")
      continue()
    endif()
    set(matched_any FALSE)
    foreach(root "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}")
      file(GLOB matched CONFIGURE_DEPENDS "${root}/${pattern}")
      if(matched)
        set(matched_any TRUE)
        list(APPEND headers ${matched})
      endif()
    endforeach()
    if(NOT matched_any)
      message(
        FATAL_ERROR
          "modeling_v2: contract surface pattern '${pattern}' matches no header"
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES headers)
  list(SORT headers)

  get_directory_property(definitions COMPILE_DEFINITIONS)
  set(inputs "torch:${TORCH_VERSION}\ncuda:${CUDAToolkit_VERSION_MAJOR}\n")
  foreach(definition IN LISTS definitions)
    string(APPEND inputs "definition:${definition}\n")
  endforeach()
  foreach(header IN LISTS headers)
    string(APPEND inputs "file:${header}\n")
  endforeach()
  file(WRITE "${generated_dir}/contractInputs.txt.tmp" "${inputs}")
  configure_file("${generated_dir}/contractInputs.txt.tmp"
                 "${generated_dir}/contractInputs.txt" COPYONLY)

  set(header "${generated_dir}/mv2ContractFingerprint.h")
  set(stamp "${generated_dir}/mv2ContractFingerprint.stamp")
  add_custom_command(
    OUTPUT "${stamp}"
    BYPRODUCTS "${header}"
    COMMAND
      ${CMAKE_COMMAND} "-DINPUTS=${generated_dir}/contractInputs.txt"
      "-DROOT=${PROJECT_SOURCE_DIR}" "-DOUTPUT=${header}" -P
      "${_TRTLLM_MV2_MODULE_DIR}/modelingV2Fingerprint.cmake"
    COMMAND ${CMAKE_COMMAND} -E touch "${stamp}"
    DEPENDS ${headers} "${generated_dir}/contractInputs.txt"
            "${_TRTLLM_MV2_MODULE_DIR}/modelingV2Fingerprint.cmake"
    COMMENT "Computing the modeling_v2 contract fingerprint"
    VERBATIM)
  add_custom_target(trtllm_mv2_contract_fingerprint DEPENDS "${stamp}")

  set(TRTLLM_MV2_GENERATED_DIR
      "${generated_dir}"
      PARENT_SCOPE)
endfunction()

# --- Families, shared code and targets ----------------------------------------

function(trtllm_mv2_family family)
  _trtllm_mv2_check_family_name("${family}")
  _trtllm_mv2_camel(camel "${family}")
  get_filename_component(dir_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  if(NOT dir_name STREQUAL camel)
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' must live in directory '${camel}', not '${dir_name}'"
    )
  endif()
  get_property(families GLOBAL PROPERTY TRTLLM_MV2_FAMILIES)
  if(family IN_LIST families)
    message(FATAL_ERROR "modeling_v2: family '${family}' is declared twice")
  endif()
  set_property(GLOBAL APPEND PROPERTY TRTLLM_MV2_FAMILIES "${family}")
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_DIR
                               "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

function(_trtllm_mv2_require_family family)
  get_property(families GLOBAL PROPERTY TRTLLM_MV2_FAMILIES)
  if(NOT family IN_LIST families)
    message(
      FATAL_ERROR
        "modeling_v2: call trtllm_mv2_family(${family}) before declaring its code"
    )
  endif()
  get_property(family_dir GLOBAL PROPERTY TRTLLM_MV2_${family}_DIR)
  if(NOT family_dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(
      FATAL_ERROR
        "modeling_v2: declare family '${family}' code in ${family_dir}/CMakeLists.txt"
    )
  endif()
endfunction()

# Expand SOURCES (paths or globs) relative to `base`.
function(_trtllm_mv2_expand_sources out base)
  set(sources "")
  foreach(item IN LISTS ARGN)
    if(IS_ABSOLUTE "${item}")
      message(
        FATAL_ERROR
          "modeling_v2: SOURCES must be relative to ${base}, got '${item}'")
    endif()
    if(item MATCHES "[*?]")
      file(GLOB matched CONFIGURE_DEPENDS "${base}/${item}")
      if(NOT matched)
        message(FATAL_ERROR "modeling_v2: '${base}/${item}' matches no file")
      endif()
      list(SORT matched)
      list(APPEND sources ${matched})
    else()
      if(NOT EXISTS "${base}/${item}")
        message(FATAL_ERROR "modeling_v2: '${base}/${item}' does not exist")
      endif()
      list(APPEND sources "${base}/${item}")
    endif()
  endforeach()
  set(${out}
      "${sources}"
      PARENT_SCOPE)
endfunction()

function(_trtllm_mv2_check_deps)
  foreach(dep IN LISTS ARGN)
    if(NOT dep IN_LIST _TRTLLM_MV2_KNOWN_DEPS)
      message(
        FATAL_ERROR
          "modeling_v2: unknown DEPS '${dep}' (known: ${_TRTLLM_MV2_KNOWN_DEPS})"
      )
    endif()
  endforeach()
endfunction()

# Compile-time usage requirements of DEPS on one OBJECT library.
function(_trtllm_mv2_apply_compile_deps object_lib)
  foreach(dep IN LISTS ARGN)
    if(dep STREQUAL "internal_cutlass")
      target_include_directories(
        ${object_lib}
        PRIVATE
          $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
      )
    elseif(dep STREQUAL "trtllm_gen_fmha_interface")
      target_link_libraries(${object_lib} PRIVATE trtllm_gen_fmha_interface)
    elseif(dep STREQUAL "nccl" AND ENABLE_MULTI_DEVICE)
      target_include_directories(${object_lib} PRIVATE ${MPI_C_INCLUDE_DIRS})
    endif()
  endforeach()
endfunction()

# Link-time libraries of DEPS, for the family library.
function(_trtllm_mv2_link_libraries out)
  set(libraries "")
  foreach(dep IN LISTS ARGN)
    if(dep STREQUAL "cublas")
      list(APPEND libraries ${CUBLAS_LIB})
    elseif(dep STREQUAL "cublaslt")
      list(APPEND libraries ${CUBLASLT_LIB})
    elseif(dep STREQUAL "cuda_driver")
      list(APPEND libraries ${CUDA_DRV_LIB})
    elseif(dep STREQUAL "nccl" AND ENABLE_MULTI_DEVICE)
      list(APPEND libraries ${NCCL_LIB} ${MPI_C_LIBRARIES})
    endif()
  endforeach()
  set(${out}
      "${libraries}"
      PARENT_SCOPE)
endfunction()

#[[
Create the OBJECT libraries of one unit (a target, or a family's shared code):
`<name>_kernels` for the .cu sources, compiled for `archs` without torch, and
`<name>_ops` for the .cpp torch bindings. Both get the unit's namespaces as
compile definitions (see common/torchLibrary.h and common/namespace.h).
#]]
function(
  _trtllm_mv2_add_objects
  name
  family
  torch_ns
  scope
  archs
  sources
  deps)
  set(cu_sources "")
  set(cpp_sources "")
  foreach(source IN LISTS sources)
    if(source MATCHES "\\.cu$")
      list(APPEND cu_sources "${source}")
    elseif(source MATCHES "\\.cpp$")
      list(APPEND cpp_sources "${source}")
    elseif(NOT source MATCHES "\\.(h|cuh|hpp|inl)$")
      message(
        FATAL_ERROR
          "modeling_v2: '${source}' is neither a .cu kernel source nor a .cpp binding"
      )
    endif()
  endforeach()

  set(definitions
      TRTLLM_MV2_TORCH_NS=${torch_ns} TRTLLM_MV2_FAMILY_TORCH_NS=mv2_${family}
      TRTLLM_MV2_SCOPE=${scope})
  set(objects "")
  if(cu_sources)
    add_library(${name}_kernels OBJECT ${cu_sources})
    set_target_properties(
      ${name}_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES
                                                              "${archs}")
    target_compile_definitions(${name}_kernels PRIVATE ${definitions})
    _trtllm_mv2_apply_compile_deps(${name}_kernels ${deps})
    list(APPEND objects ${name}_kernels)
  endif()
  if(cpp_sources)
    add_library(${name}_ops OBJECT ${cpp_sources})
    set_target_properties(${name}_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${name}_ops PRIVATE ${definitions})
    target_link_libraries(${name}_ops PRIVATE ${TORCH_LIBRARIES})
    _trtllm_mv2_apply_compile_deps(${name}_ops ${deps})
    list(APPEND objects ${name}_ops)
  endif()
  set_property(GLOBAL APPEND PROPERTY TRTLLM_MV2_${family}_OBJECTS ${objects})
endfunction()

# Record a unit's manifest entry and, when it is built, its link libraries.
function(
  _trtllm_mv2_record
  family
  key
  status
  declared
  archs
  reason
  torch_ns
  deps)
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_STATUS "${status}")
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_DECLARED
                               "${declared}")
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_ARCHS "${archs}")
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_REASON "${reason}")
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_NAMESPACE
                               "${torch_ns}")
  if(status STREQUAL "built")
    set_property(GLOBAL APPEND PROPERTY TRTLLM_MV2_${family}_DEPS ${deps})
  endif()
endfunction()

function(trtllm_mv2_shared family)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "ARCHS;SOURCES;DEPS;REQUIRES")
  _trtllm_mv2_require_family("${family}")
  get_property(already GLOBAL PROPERTY TRTLLM_MV2_${family}_HAS_SHARED)
  if(already)
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' declares trtllm_mv2_shared twice")
  endif()
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_HAS_SHARED TRUE)
  if(NOT arg_ARCHS OR NOT arg_SOURCES)
    message(
      FATAL_ERROR
        "modeling_v2: trtllm_mv2_shared(${family}) needs ARCHS and SOURCES")
  endif()
  _trtllm_mv2_check_deps(${arg_DEPS})
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shared")
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' shared code lives in ${CMAKE_CURRENT_SOURCE_DIR}/shared"
    )
  endif()

  # Shared code is built when a target of the family is; trtllm_mv2_finalize
  # drops it otherwise. Selection and requirements still apply to it here.
  _trtllm_mv2_unmet_requirements(unmet ${arg_REQUIRES})
  _trtllm_mv2_resolve_archs(archs DECLARED ${arg_ARCHS})
  if(NOT unmet STREQUAL "")
    set(status "skipped")
    set(reason "${unmet}")
    set(archs "")
  elseif(NOT archs)
    set(status "skipped")
    set(reason
        "declares ${arg_ARCHS}; this build requests ${CMAKE_CUDA_ARCHITECTURES_ORIG}"
    )
  else()
    set(status "built")
    set(reason "")
    _trtllm_mv2_expand_sources(sources "${CMAKE_CURRENT_SOURCE_DIR}/shared"
                               ${arg_SOURCES})
    _trtllm_mv2_add_objects(
      trtllm_mv2_${family}_shared
      "${family}"
      "mv2_${family}"
      "${family}::shared"
      "${archs}"
      "${sources}"
      "${arg_DEPS}")
  endif()
  _trtllm_mv2_record(
    "${family}"
    "SHARED"
    "${status}"
    "${arg_ARCHS}"
    "${archs}"
    "${reason}"
    "mv2_${family}"
    "${arg_DEPS}")
endfunction()

function(trtllm_mv2_target family target)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "ARCHS;SOURCES;DEPS;REQUIRES")
  _trtllm_mv2_require_family("${family}")
  _trtllm_mv2_check_target_name("${target}")
  if(NOT arg_ARCHS OR NOT arg_SOURCES)
    message(
      FATAL_ERROR
        "modeling_v2: trtllm_mv2_target(${family} ${target}) needs ARCHS and SOURCES"
    )
  endif()
  _trtllm_mv2_check_deps(${arg_DEPS})
  _trtllm_mv2_camel(target_dir_name "${target}")
  set(target_dir "${CMAKE_CURRENT_SOURCE_DIR}/${target_dir_name}")
  if(NOT IS_DIRECTORY "${target_dir}")
    message(
      FATAL_ERROR
        "modeling_v2: target '${target}' must live in directory ${target_dir}")
  endif()

  get_property(targets GLOBAL PROPERTY TRTLLM_MV2_${family}_TARGETS)
  if(target IN_LIST targets)
    message(
      FATAL_ERROR "modeling_v2: target '${family}/${target}' is declared twice")
  endif()
  set_property(GLOBAL APPEND PROPERTY TRTLLM_MV2_${family}_TARGETS "${target}")

  _trtllm_mv2_ident(target_ident "${target}")
  set(torch_ns "mv2_${family}_${target_ident}")
  get_property(namespaces GLOBAL PROPERTY TRTLLM_MV2_NAMESPACES)
  if(torch_ns IN_LIST namespaces)
    message(
      FATAL_ERROR
        "modeling_v2: torch namespace '${torch_ns}' of ${family}/${target} is already taken"
    )
  endif()
  set_property(GLOBAL APPEND PROPERTY TRTLLM_MV2_NAMESPACES "${torch_ns}")

  _trtllm_mv2_selected(selected "${family}" "${target}")
  _trtllm_mv2_unmet_requirements(unmet ${arg_REQUIRES})
  _trtllm_mv2_resolve_archs(archs DECLARED ${arg_ARCHS})
  if(NOT selected)
    set(status "excluded")
    set(reason
        "not selected by TRTLLM_MODELING_V2_TARGETS=${TRTLLM_MODELING_V2_TARGETS}"
    )
    set(archs "")
  elseif(NOT unmet STREQUAL "")
    set(status "skipped")
    set(reason "${unmet}")
    set(archs "")
  elseif(NOT archs)
    set(status "skipped")
    set(reason
        "declares ${arg_ARCHS}; this build requests ${CMAKE_CUDA_ARCHITECTURES_ORIG}"
    )
  else()
    set(status "built")
    set(reason "")
    _trtllm_mv2_expand_sources(sources "${target_dir}" ${arg_SOURCES})
    _trtllm_mv2_add_objects(
      trtllm_mv2_${family}_${target_ident}
      "${family}"
      "${torch_ns}"
      "${family}::${target_ident}"
      "${archs}"
      "${sources}"
      "${arg_DEPS}")
  endif()
  _trtllm_mv2_record(
    "${family}"
    "TARGET_${target}"
    "${status}"
    "${arg_ARCHS}"
    "${archs}"
    "${reason}"
    "${torch_ns}"
    "${arg_DEPS}")
endfunction()

# --- Family libraries and the manifest ---------------------------------------

function(_trtllm_mv2_json_list out)
  set(json "[]")
  set(index 0)
  foreach(item IN LISTS ARGN)
    string(JSON json SET "${json}" ${index} "\"${item}\"")
    math(EXPR index "${index} + 1")
  endforeach()
  set(${out}
      "${json}"
      PARENT_SCOPE)
endfunction()

function(_trtllm_mv2_json_unit out family key)
  get_property(status GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_STATUS)
  get_property(declared GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_DECLARED)
  get_property(archs GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_ARCHS)
  get_property(reason GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_REASON)
  get_property(torch_ns GLOBAL PROPERTY TRTLLM_MV2_${family}_${key}_NAMESPACE)
  _trtllm_mv2_json_list(declared_json ${declared})
  _trtllm_mv2_json_list(archs_json ${archs})
  set(json "{}")
  string(JSON json SET "${json}" status "\"${status}\"")
  string(JSON json SET "${json}" namespace "\"${torch_ns}\"")
  string(JSON json SET "${json}" declared_archs "${declared_json}")
  string(JSON json SET "${json}" archs "${archs_json}")
  string(JSON json SET "${json}" reason "\"${reason}\"")
  set(${out}
      "${json}"
      PARENT_SCOPE)
endfunction()

function(_trtllm_mv2_write_manifest json)
  file(
    GENERATE
    OUTPUT
      "${CMAKE_BINARY_DIR}/tensorrt_llm/modeling_v2/modeling_v2_manifest.json"
    CONTENT "${json}\n")
endfunction()

# The manifest of a build without modeling_v2 libraries.
function(trtllm_mv2_write_disabled_manifest reason)
  set(json "{}")
  string(JSON json SET "${json}" schema ${_TRTLLM_MV2_MANIFEST_SCHEMA})
  string(JSON json SET "${json}" enabled false)
  string(JSON json SET "${json}" reason "\"${reason}\"")
  string(JSON json SET "${json}" families "{}")
  _trtllm_mv2_write_manifest("${json}")
endfunction()

# The registration unit of a family library: it claims the family's torch
# namespace (with contract_fingerprint) and every built target's namespace, so a
# second library claiming any of them fails when it is loaded.
function(_trtllm_mv2_registration_source out family)
  set(source
      "// Generated by cpp/cmake/modules/modelingV2.cmake for family ${family}. Do not edit.\n"
  )
  string(APPEND source "#include \"mv2ContractFingerprint.h\"\n\n")
  string(APPEND source "#include <string>\n#include <torch/library.h>\n\n")
  string(APPEND source "namespace\n{\nstd::string contractFingerprint()\n{\n")
  string(APPEND source
         "    return TRTLLM_MV2_CONTRACT_FINGERPRINT;\n}\n} // namespace\n\n")
  string(APPEND source "TORCH_LIBRARY(mv2_${family}, m)\n{\n")
  string(APPEND source "    m.def(\"contract_fingerprint() -> str\");\n}\n\n")
  string(APPEND source
         "TORCH_LIBRARY_IMPL(mv2_${family}, CompositeExplicitAutograd, m)\n{\n")
  string(APPEND source
         "    m.impl(\"contract_fingerprint\", &contractFingerprint);\n}\n")
  foreach(torch_ns IN LISTS ARGN)
    string(APPEND source
           "\nTORCH_LIBRARY(${torch_ns}, m)\n{\n    (void) m;\n}\n")
  endforeach()
  set(path "${CMAKE_CURRENT_BINARY_DIR}/${family}/registration.cpp")
  file(
    GENERATE
    OUTPUT "${path}"
    CONTENT "${source}")
  set_source_files_properties("${path}" PROPERTIES GENERATED TRUE)
  set(${out}
      "${path}"
      PARENT_SCOPE)
endfunction()

function(_trtllm_mv2_add_family_library family)
  get_property(objects GLOBAL PROPERTY TRTLLM_MV2_${family}_OBJECTS)
  get_property(targets GLOBAL PROPERTY TRTLLM_MV2_${family}_TARGETS)
  get_property(deps GLOBAL PROPERTY TRTLLM_MV2_${family}_DEPS)

  set(built_namespaces "")
  foreach(target IN LISTS targets)
    get_property(status GLOBAL
                 PROPERTY TRTLLM_MV2_${family}_TARGET_${target}_STATUS)
    if(status STREQUAL "built")
      get_property(torch_ns GLOBAL
                   PROPERTY TRTLLM_MV2_${family}_TARGET_${target}_NAMESPACE)
      list(APPEND built_namespaces "${torch_ns}")
    endif()
  endforeach()

  # Shared code without a built target has nothing to serve.
  get_property(shared_status GLOBAL PROPERTY TRTLLM_MV2_${family}_SHARED_STATUS)
  if(NOT built_namespaces)
    if(shared_status STREQUAL "built")
      set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_SHARED_STATUS "skipped")
      set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_SHARED_REASON
                                   "no target of the family is built")
      set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_SHARED_ARCHS "")
    endif()
    set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_LIBRARY "")
    return()
  endif()

  set(library trtllm_mv2_${family})
  _trtllm_mv2_registration_source(registration "${family}" ${built_namespaces})
  set(object_sources "")
  foreach(object IN LISTS objects)
    list(APPEND object_sources $<TARGET_OBJECTS:${object}>)
  endforeach()
  add_library(${library} MODULE "${registration}" ${object_sources})
  add_dependencies(${library} trtllm_mv2_contract_fingerprint)
  target_include_directories(${library} PRIVATE "${TRTLLM_MV2_GENERATED_DIR}")
  list(REMOVE_DUPLICATES deps)
  _trtllm_mv2_link_libraries(dep_libraries ${deps})
  target_link_libraries(
    ${library} PRIVATE ${TORCH_LIBRARIES} th_utils ${SHARED_TARGET}
                       ${CUDA_RT_LIB} ${CUDA_DRV_LIB} ${dep_libraries})
  set(version_script
      "${PROJECT_SOURCE_DIR}/tensorrt_llm/modeling_v2/common/familyLibrary.version"
  )
  set_target_properties(
    ${library}
    PROPERTIES OUTPUT_NAME ${library}
               PREFIX "lib"
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/tensorrt_llm/modeling_v2"
               POSITION_INDEPENDENT_CODE ON
               LINKER_LANGUAGE CXX
               LINK_DEPENDS "${version_script}"
               BUILD_RPATH "$ORIGIN;$ORIGIN/../../nvidia/nccl/lib")
  target_link_options(
    ${library} PRIVATE "-Wl,--version-script=${version_script}"
    "-Wl,--no-undefined")
  add_dependencies(modeling_v2_libs ${library})
  set_property(GLOBAL PROPERTY TRTLLM_MV2_${family}_LIBRARY "lib${library}.so")
endfunction()

# Build every declared family's library and write the manifest. Called by
# cpp/tensorrt_llm/modeling_v2/CMakeLists.txt after it added the families.
function(trtllm_mv2_finalize)
  get_property(families GLOBAL PROPERTY TRTLLM_MV2_FAMILIES)
  _trtllm_mv2_json_list(requested_json ${CMAKE_CUDA_ARCHITECTURES_ORIG})
  set(json "{}")
  string(JSON json SET "${json}" schema ${_TRTLLM_MV2_MANIFEST_SCHEMA})
  string(JSON json SET "${json}" enabled true)
  string(JSON json SET "${json}" requested_archs "${requested_json}")
  string(JSON json SET "${json}" families "{}")

  foreach(family IN LISTS families)
    _trtllm_mv2_add_family_library("${family}")
    get_property(library GLOBAL PROPERTY TRTLLM_MV2_${family}_LIBRARY)
    get_property(targets GLOBAL PROPERTY TRTLLM_MV2_${family}_TARGETS)
    get_property(has_shared GLOBAL PROPERTY TRTLLM_MV2_${family}_HAS_SHARED)

    set(family_json "{}")
    if(library STREQUAL "")
      string(JSON family_json SET "${family_json}" library null)
    else()
      string(JSON family_json SET "${family_json}" library "\"${library}\"")
    endif()
    string(JSON family_json SET "${family_json}" namespace "\"mv2_${family}\"")
    if(has_shared)
      _trtllm_mv2_json_unit(shared_json "${family}" "SHARED")
      string(JSON family_json SET "${family_json}" shared "${shared_json}")
    else()
      string(JSON family_json SET "${family_json}" shared null)
    endif()
    set(targets_json "{}")
    foreach(target IN LISTS targets)
      _trtllm_mv2_json_unit(target_json "${family}" "TARGET_${target}")
      string(JSON targets_json SET "${targets_json}" "${target}"
             "${target_json}")
    endforeach()
    string(JSON family_json SET "${family_json}" targets "${targets_json}")
    string(
      JSON
      json
      SET
      "${json}"
      families
      "${family}"
      "${family_json}")
  endforeach()
  _trtllm_mv2_write_manifest("${json}")
endfunction()
