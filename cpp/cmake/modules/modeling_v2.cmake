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

A model family's native code lives with its Python code, under
tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/:

  csrc/CMakeLists.txt       declares the family, its shared code and its targets
  csrc/*.cu|*.h|*.cpp       code several targets of the family share
  <target>/csrc/...         one target's own kernels (.cu) and torch bindings (.cpp)

Each family builds into one torch-op library, libth_modeling_v2_<family>.so,
installed next to libth_common.so. Nothing in it is linked into
libtensorrt_llm.so or libth_common.so; the Python loader
(tensorrt_llm/_torch/_experimental/modeling_v2/_native.py) opens it only when
routing picks one of the family's targets. The "Native code" section of the
modeling_v2 README is the guide for authors; cpp/tensorrt_llm/modeling_v2/
holds only what every family shares (this discovery, the common/ macros).

A family's csrc/CMakeLists.txt uses three functions; names are the Python
package names:

  add_modeling_v2_family(<family>)
  add_modeling_v2_shared(<family> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])
  add_modeling_v2_target(<family> <target> ARCHS ... [SOURCES ...] [DEPS ...] [REQUIRES ...])

ARCHS     NN, NNa or NNf. Each is intersected with the build's requested
          architectures, the way flash_mla and deep_ep choose theirs; a target
          with an empty intersection is not built and its family library
          does not carry it.
SOURCES   Paths or globs relative to the unit's csrc/ directory: the
          family's for add_modeling_v2_shared, <target>/csrc/ for a target.
          .cu files are compiled for ARCHS without torch; .cpp files are the
          torch bindings. Includes of the family's own code are written from
          the models/ directory: "<family>/csrc/x.h",
          "<family>/<target>/csrc/x.h". A target without SOURCES has no csrc/
          and no code of its own: it runs its family's shared code, and
          declaring it is what makes the library carry it.
          Source properties the family's CMakeLists.txt sets with
          set_source_files_properties (e.g. COMPILE_OPTIONS) apply to shared
          and target sources alike.
DEPS      From a closed list: cublas, cublaslt, cuda_driver, internal_cutlass,
          nccl, trtllm_gen_fmha_interface.
REQUIRES  Build features: MULTI_DEVICE, NVSHMEM. Unmet requirements skip the
          target.

Like deep_ep and flash_mla, each family writes cuda_architectures.txt into its
build directory: the architectures its library was built for, empty when it
has none. scripts/build_wheel.py installs the library only when that list is
not empty.
#]]

include_guard(GLOBAL)

set(_MODELING_V2_KNOWN_DEPS cublas cublaslt cuda_driver internal_cutlass nccl
                            trtllm_gen_fmha_interface)
set(_MODELING_V2_KNOWN_REQUIRES MULTI_DEVICE NVSHMEM)

# --- Naming ------------------------------------------------------------------

# A name usable as a C++ or torch identifier: "__" collapsed to "_", since C++
# reserves identifiers that contain a double underscore.
function(_modeling_v2_ident out name)
  set(result "${name}")
  while(result MATCHES "__")
    string(REPLACE "__" "_" result "${result}")
  endwhile()
  set(${out}
      "${result}"
      PARENT_SCOPE)
endfunction()

function(_modeling_v2_check_family_name family)
  if(NOT family MATCHES "^[a-z][a-z0-9_]*[a-z0-9]$" OR family MATCHES "__")
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' is not a snake_case package name")
  endif()
endfunction()

# <checkpoint>__<sm>__<parallel>, the identity the Python target directory
# carries (see tensorrt_llm/_torch/_experimental/modeling_v2/README.md).
function(_modeling_v2_check_target_name target)
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
function(_modeling_v2_selected out family target)
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
function(_modeling_v2_unmet_requirements out)
  set(reasons "")
  foreach(requirement IN LISTS ARGN)
    if(NOT requirement IN_LIST _MODELING_V2_KNOWN_REQUIRES)
      message(
        FATAL_ERROR
          "modeling_v2: unknown REQUIRES '${requirement}' (known: ${_MODELING_V2_KNOWN_REQUIRES})"
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
build was asked for -- the list flash_mla intersects with too. This does not
reuse set_cuda_architectures(), which drops NNa silently and falls back to
placeholder kernels.
#]]
function(_modeling_v2_resolve_archs out)
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

# --- Families, shared code and targets ----------------------------------------

function(add_modeling_v2_family family)
  _modeling_v2_check_family_name("${family}")
  get_filename_component(dir_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  get_filename_component(package_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(package_name "${package_dir}" NAME)
  if(NOT dir_name STREQUAL "csrc" OR NOT package_name STREQUAL family)
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' is declared in models/${family}/csrc/CMakeLists.txt, "
        "not in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  get_property(families GLOBAL PROPERTY MODELING_V2_FAMILIES)
  if(family IN_LIST families)
    message(FATAL_ERROR "modeling_v2: family '${family}' is declared twice")
  endif()
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_FAMILIES "${family}")
  set_property(GLOBAL PROPERTY MODELING_V2_${family}_SOURCE_DIR
                               "${CMAKE_CURRENT_SOURCE_DIR}")
  set_property(GLOBAL PROPERTY MODELING_V2_${family}_PACKAGE_DIR
                               "${package_dir}")
  set_property(GLOBAL PROPERTY MODELING_V2_${family}_BINARY_DIR
                               "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

function(_modeling_v2_require_family family)
  get_property(families GLOBAL PROPERTY MODELING_V2_FAMILIES)
  if(NOT family IN_LIST families)
    message(
      FATAL_ERROR
        "modeling_v2: call add_modeling_v2_family(${family}) before declaring its code"
    )
  endif()
  get_property(family_dir GLOBAL PROPERTY MODELING_V2_${family}_SOURCE_DIR)
  if(NOT family_dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(
      FATAL_ERROR
        "modeling_v2: declare family '${family}' code in ${family_dir}/CMakeLists.txt"
    )
  endif()
endfunction()

# Expand SOURCES (paths or globs) relative to `base`.
function(_modeling_v2_expand_sources out base)
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

function(_modeling_v2_check_deps)
  foreach(dep IN LISTS ARGN)
    if(NOT dep IN_LIST _MODELING_V2_KNOWN_DEPS)
      message(
        FATAL_ERROR
          "modeling_v2: unknown DEPS '${dep}' (known: ${_MODELING_V2_KNOWN_DEPS})"
      )
    endif()
  endforeach()
endfunction()

# Compile-time usage requirements of DEPS on one OBJECT library.
function(_modeling_v2_apply_compile_deps object_lib)
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
function(_modeling_v2_link_libraries out)
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
  _modeling_v2_add_objects
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

  # Shared code is assembled from the modeling_v2 directory, where source
  # properties the family set in its own directory are not visible.
  get_property(family_dir GLOBAL PROPERTY MODELING_V2_${family}_SOURCE_DIR)
  foreach(source IN LISTS cu_sources cpp_sources)
    foreach(property COMPILE_OPTIONS COMPILE_DEFINITIONS)
      get_source_file_property(value "${source}" DIRECTORY "${family_dir}"
                                                           ${property})
      if(value)
        set_source_files_properties("${source}" PROPERTIES ${property}
                                                           "${value}")
      endif()
    endforeach()
  endforeach()

  set(definitions
      TRTLLM_MV2_TORCH_NS=${torch_ns} TRTLLM_MV2_FAMILY_TORCH_NS=mv2_${family}
      TRTLLM_MV2_SCOPE=${scope})
  # Family code includes its own headers from the models/ directory.
  get_filename_component(models_dir "${family_dir}" DIRECTORY)
  get_filename_component(models_dir "${models_dir}" DIRECTORY)

  set(objects "")
  if(cu_sources)
    add_library(${name}_kernels OBJECT ${cu_sources})
    set_target_properties(
      ${name}_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES
                                                              "${archs}")
    target_compile_definitions(${name}_kernels PRIVATE ${definitions})
    target_include_directories(${name}_kernels PRIVATE "${models_dir}")
    _modeling_v2_apply_compile_deps(${name}_kernels ${deps})
    list(APPEND objects ${name}_kernels)
  endif()
  if(cpp_sources)
    add_library(${name}_ops OBJECT ${cpp_sources})
    set_target_properties(${name}_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${name}_ops PRIVATE ${definitions})
    target_include_directories(${name}_ops PRIVATE "${models_dir}")
    target_link_libraries(${name}_ops PRIVATE ${TORCH_LIBRARIES})
    _modeling_v2_apply_compile_deps(${name}_ops ${deps})
    list(APPEND objects ${name}_ops)
  endif()
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_OBJECTS ${objects})
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_DEPS ${deps})
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_ARCHS ${archs})
endfunction()

# Decide whether a unit builds: "" when it does, otherwise why not.
function(_modeling_v2_skip_reason out family target declared requires)
  _modeling_v2_selected(selected "${family}" "${target}")
  _modeling_v2_unmet_requirements(unmet ${requires})
  _modeling_v2_resolve_archs(archs DECLARED ${declared})
  if(NOT selected)
    set(reason
        "not selected by TRTLLM_MODELING_V2_TARGETS=${TRTLLM_MODELING_V2_TARGETS}"
    )
  elseif(NOT unmet STREQUAL "")
    set(reason "${unmet}")
  elseif(NOT archs)
    set(reason
        "declares ${declared}; this build requests ${CMAKE_CUDA_ARCHITECTURES_ORIG}"
    )
  else()
    set(reason "")
  endif()
  set(${out}
      "${reason}"
      PARENT_SCOPE)
endfunction()

function(add_modeling_v2_shared family)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "ARCHS;SOURCES;DEPS;REQUIRES")
  _modeling_v2_require_family("${family}")
  get_property(already GLOBAL PROPERTY MODELING_V2_${family}_HAS_SHARED)
  if(already)
    message(
      FATAL_ERROR
        "modeling_v2: family '${family}' declares add_modeling_v2_shared twice")
  endif()
  set_property(GLOBAL PROPERTY MODELING_V2_${family}_HAS_SHARED TRUE)
  if(NOT arg_ARCHS OR NOT arg_SOURCES)
    message(
      FATAL_ERROR
        "modeling_v2: add_modeling_v2_shared(${family}) needs ARCHS and SOURCES"
    )
  endif()
  _modeling_v2_check_deps(${arg_DEPS})
  # Recorded now, built by _modeling_v2_add_family_library only when a target of
  # the family builds: shared code with no target has nothing to serve.
  foreach(field ARCHS SOURCES DEPS REQUIRES)
    set_property(GLOBAL PROPERTY MODELING_V2_${family}_SHARED_${field}
                                 "${arg_${field}}")
  endforeach()
endfunction()

function(add_modeling_v2_target family target)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "ARCHS;SOURCES;DEPS;REQUIRES")
  _modeling_v2_require_family("${family}")
  _modeling_v2_check_target_name("${target}")
  if(NOT arg_ARCHS)
    message(
      FATAL_ERROR
        "modeling_v2: add_modeling_v2_target(${family} ${target}) needs ARCHS")
  endif()
  _modeling_v2_check_deps(${arg_DEPS})
  get_property(package_dir GLOBAL PROPERTY MODELING_V2_${family}_PACKAGE_DIR)
  set(target_dir "${package_dir}/${target}/csrc")
  if(arg_SOURCES AND NOT IS_DIRECTORY "${target_dir}")
    message(
      FATAL_ERROR
        "modeling_v2: target '${target}' keeps its code in ${target_dir}")
  endif()
  if(NOT arg_SOURCES)
    get_property(has_shared GLOBAL PROPERTY MODELING_V2_${family}_HAS_SHARED)
    if(NOT has_shared)
      message(
        FATAL_ERROR
          "modeling_v2: target '${family}/${target}' has no SOURCES, so it runs "
          "its family's shared code; declare add_modeling_v2_shared(${family} ...) "
          "before it")
    endif()
  endif()

  get_property(targets GLOBAL PROPERTY MODELING_V2_${family}_TARGETS)
  if(target IN_LIST targets)
    message(
      FATAL_ERROR "modeling_v2: target '${family}/${target}' is declared twice")
  endif()
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_TARGETS "${target}")

  _modeling_v2_ident(target_ident "${target}")
  set(torch_ns "mv2_${family}_${target_ident}")
  get_property(namespaces GLOBAL PROPERTY MODELING_V2_NAMESPACES)
  if(torch_ns IN_LIST namespaces)
    message(
      FATAL_ERROR
        "modeling_v2: torch namespace '${torch_ns}' of ${family}/${target} is already taken"
    )
  endif()
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_NAMESPACES "${torch_ns}")

  _modeling_v2_skip_reason(reason "${family}" "${target}" "${arg_ARCHS}"
                           "${arg_REQUIRES}")
  if(reason STREQUAL "" AND NOT arg_SOURCES)
    # A target without code of its own runs only where its shared code builds.
    get_property(shared_archs_declared GLOBAL
                 PROPERTY MODELING_V2_${family}_SHARED_ARCHS)
    get_property(shared_requires GLOBAL
                 PROPERTY MODELING_V2_${family}_SHARED_REQUIRES)
    _modeling_v2_unmet_requirements(unmet ${shared_requires})
    _modeling_v2_resolve_archs(shared_archs DECLARED ${shared_archs_declared})
    if(NOT unmet STREQUAL "")
      set(reason "its family's shared code is not built: ${unmet}")
    elseif(NOT shared_archs)
      set(reason "its family's shared code declares ${shared_archs_declared}")
    endif()
  endif()
  if(NOT reason STREQUAL "")
    message(STATUS "modeling_v2: ${family}/${target} not built: ${reason}")
    return()
  endif()
  _modeling_v2_resolve_archs(archs DECLARED ${arg_ARCHS})
  if(arg_SOURCES)
    message(STATUS "modeling_v2: ${family}/${target} built for ${archs}")
    _modeling_v2_expand_sources(sources "${target_dir}" ${arg_SOURCES})
    _modeling_v2_add_objects(
      th_modeling_v2_${family}_${target_ident}
      "${family}"
      "${torch_ns}"
      "${family}::${target_ident}"
      "${archs}"
      "${sources}"
      "${arg_DEPS}")
  else()
    message(
      STATUS
        "modeling_v2: ${family}/${target} carried for ${archs} (family shared code only)"
    )
    set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_ARCHS ${archs})
  endif()
  set_property(GLOBAL APPEND PROPERTY MODELING_V2_${family}_BUILT "${target}")
endfunction()

# --- Family libraries ---------------------------------------------------------

# The registration unit of a family library: it claims the family's torch
# namespace and every built target's, so a second library claiming any of them
# fails when it is loaded, and it registers built_targets() -- which targets
# this library carries -- the way th_common answers feature queries such as
# mhc_fused_hc_mma_enabled().
function(_modeling_v2_registration_source out family binary_dir)
  get_property(built GLOBAL PROPERTY MODELING_V2_${family}_BUILT)
  list(JOIN built "," built_csv)
  set(source
      "// Generated by cpp/cmake/modules/modeling_v2.cmake for family ${family}. Do not edit.\n"
  )
  string(APPEND source "#include <string>\n#include <torch/library.h>\n\n")
  string(APPEND source "namespace\n{\n")
  string(APPEND source
         "//! The targets of this family compiled into this library.\n")
  string(APPEND source "std::string builtTargets()\n{\n")
  string(APPEND source "    return \"${built_csv}\";\n}\n} // namespace\n\n")
  string(APPEND source "TORCH_LIBRARY(mv2_${family}, m)\n{\n")
  string(APPEND source "    m.def(\"built_targets() -> str\");\n}\n\n")
  string(APPEND source
         "TORCH_LIBRARY_IMPL(mv2_${family}, CompositeExplicitAutograd, m)\n{\n")
  string(APPEND source "    m.impl(\"built_targets\", &builtTargets);\n}\n")
  foreach(target IN LISTS built)
    _modeling_v2_ident(target_ident "${target}")
    string(
      APPEND source
      "\nTORCH_LIBRARY(mv2_${family}_${target_ident}, m)\n{\n    (void) m;\n}\n"
    )
  endforeach()
  set(path "${binary_dir}/registration.cpp")
  file(
    GENERATE
    OUTPUT "${path}"
    CONTENT "${source}")
  set_source_files_properties("${path}" PROPERTIES GENERATED TRUE)
  set(${out}
      "${path}"
      PARENT_SCOPE)
endfunction()

function(_modeling_v2_add_family_library family)
  get_property(binary_dir GLOBAL PROPERTY MODELING_V2_${family}_BINARY_DIR)
  get_property(built GLOBAL PROPERTY MODELING_V2_${family}_BUILT)
  if(NOT built)
    # Like deep_ep and flash_mla without a matching architecture: an empty
    # cuda_architectures.txt, and nothing for build_wheel.py to install.
    file(WRITE "${binary_dir}/cuda_architectures.txt" "")
    return()
  endif()

  get_property(has_shared GLOBAL PROPERTY MODELING_V2_${family}_HAS_SHARED)
  if(has_shared)
    foreach(field ARCHS SOURCES DEPS REQUIRES)
      get_property(shared_${field} GLOBAL
                   PROPERTY MODELING_V2_${family}_SHARED_${field})
    endforeach()
    # The target filter does not apply: a built target needs its shared code.
    _modeling_v2_unmet_requirements(unmet ${shared_REQUIRES})
    _modeling_v2_resolve_archs(shared_archs DECLARED ${shared_ARCHS})
    if(NOT unmet STREQUAL "")
      message(STATUS "modeling_v2: ${family}/shared not built: ${unmet}")
    elseif(NOT shared_archs)
      message(
        STATUS
          "modeling_v2: ${family}/shared not built: declares ${shared_ARCHS}; "
          "this build requests ${CMAKE_CUDA_ARCHITECTURES_ORIG}")
    else()
      get_property(source_dir GLOBAL PROPERTY MODELING_V2_${family}_SOURCE_DIR)
      _modeling_v2_expand_sources(sources "${source_dir}" ${shared_SOURCES})
      message(STATUS "modeling_v2: ${family}/shared built for ${shared_archs}")
      _modeling_v2_add_objects(
        th_modeling_v2_${family}_shared
        "${family}"
        "mv2_${family}"
        "${family}::shared"
        "${shared_archs}"
        "${sources}"
        "${shared_DEPS}")
    endif()
  endif()

  get_property(objects GLOBAL PROPERTY MODELING_V2_${family}_OBJECTS)
  get_property(deps GLOBAL PROPERTY MODELING_V2_${family}_DEPS)
  get_property(archs GLOBAL PROPERTY MODELING_V2_${family}_ARCHS)
  list(REMOVE_DUPLICATES deps)
  list(REMOVE_DUPLICATES archs)

  set(library th_modeling_v2_${family})
  _modeling_v2_registration_source(registration "${family}" "${binary_dir}")
  set(object_sources "")
  foreach(object IN LISTS objects)
    list(APPEND object_sources $<TARGET_OBJECTS:${object}>)
  endforeach()
  add_library(${library} MODULE "${registration}" ${object_sources})
  _modeling_v2_link_libraries(dep_libraries ${deps})
  # Linked the way th_common is: torch, th_utils, Python (which
  # <torch/extension.h>, included by stock headers such as thop/thUtils.h,
  # references) and libtensorrt_llm.so, whose singletons (logger, mcast
  # registry, driver wrapper) stay single.
  target_link_libraries(
    ${library}
    PRIVATE ${TORCH_LIBRARIES}
            th_utils
            ${Python3_LIBRARIES}
            ${SHARED_TARGET}
            ${CUDA_RT_LIB}
            ${CUDA_DRV_LIB}
            ${dep_libraries})
  set(version_script
      "${PROJECT_SOURCE_DIR}/tensorrt_llm/modeling_v2/common/familyLibrary.version"
  )
  set_target_properties(
    ${library}
    PROPERTIES PREFIX "lib"
               LIBRARY_OUTPUT_DIRECTORY "${binary_dir}"
               POSITION_INDEPENDENT_CODE ON
               LINKER_LANGUAGE CXX
               LINK_DEPENDS "${version_script}"
               BUILD_RPATH "$ORIGIN;$ORIGIN/../../nvidia/nccl/lib")
  target_link_options(
    ${library} PRIVATE "-Wl,--version-script=${version_script}"
    "-Wl,--no-undefined")
  add_dependencies(modeling_v2_libs ${library})
  file(WRITE "${binary_dir}/cuda_architectures.txt" "${archs}")
endfunction()

# Build every declared family's library. Called by
# cpp/tensorrt_llm/modeling_v2/CMakeLists.txt after it added the families.
function(add_modeling_v2_family_libraries)
  get_property(families GLOBAL PROPERTY MODELING_V2_FAMILIES)
  foreach(family IN LISTS families)
    _modeling_v2_add_family_library("${family}")
  endforeach()
endfunction()
