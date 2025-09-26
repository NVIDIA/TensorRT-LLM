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

# TensorRT install path in docker image
set(TensorRT_WELL_KNOWN_ROOT /usr/local/tensorrt)

find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_WELL_KNOWN_ROOT}/include)

function(_tensorrt_get_version)
  unset(TensorRT_VERSION_STRING PARENT_SCOPE)
  set(_hdr_file "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")

  if(NOT EXISTS "${_hdr_file}")
    return()
  endif()

  file(STRINGS "${_hdr_file}" IS_10_11_NEW_MACRO REGEX "TRT_MAJOR_ENTERPRISE")
  if(IS_10_11_NEW_MACRO)
    file(STRINGS "${_hdr_file}" VERSION_STRINGS
         REGEX "#define TRT_.+_ENTERPRISE.*")
    foreach(TYPE MAJOR MINOR PATCH BUILD)
      string(REGEX MATCH "TRT_${TYPE}_ENTERPRISE [0-9]+" TRT_TYPE_STRING
                   ${VERSION_STRINGS})
      string(REGEX MATCH "[0-9]+" TensorRT_VERSION_${TYPE} ${TRT_TYPE_STRING})
    endforeach(TYPE)
  else()
    file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")
    foreach(TYPE MAJOR MINOR PATCH BUILD)
      string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING
                   ${VERSION_STRINGS})
      string(REGEX MATCH "[0-9]+" TensorRT_VERSION_${TYPE} ${TRT_TYPE_STRING})
    endforeach(TYPE)
  endif()

  set(TensorRT_VERSION_MAJOR
      ${TensorRT_VERSION_MAJOR}
      PARENT_SCOPE)
  set(TensorRT_VERSION_STRING
      "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}"
      PARENT_SCOPE)
endfunction(_tensorrt_get_version)

_tensorrt_get_version()

macro(_tensorrt_find_dll VAR)
  find_file(
    ${VAR}
    NAMES ${ARGN}
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES bin)
endmacro(_tensorrt_find_dll)

find_library(
  TensorRT_LIBRARY
  NAMES "nvinfer_${TensorRT_VERSION_MAJOR}" nvinfer
  PATHS ${TensorRT_WELL_KNOWN_ROOT}/lib)

if(WIN32)
  _tensorrt_find_dll(TensorRT_DLL "nvinfer_${TensorRT_VERSION_MAJOR}.dll"
                     nvinfer.dll)
endif()

if(TensorRT_LIBRARY)
  set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_LIBRARY})
endif(TensorRT_LIBRARY)

if(TensorRT_FIND_COMPONENTS)
  list(REMOVE_ITEM TensorRT_FIND_COMPONENTS "nvinfer")

  if("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
    find_path(
      TensorRT_OnnxParser_INCLUDE_DIR
      NAMES NvOnnxParser.h
      PATHS ${TensorRT_WELL_KNOWN_ROOT}/include)

    find_library(
      TensorRT_OnnxParser_LIBRARY
      NAMES "nvonnxparser_${TensorRT_VERSION_MAJOR}" nvonnxparser
      PATHS ${TensorRT_WELL_KNOWN_ROOT}/lib)
    if(TensorRT_OnnxParser_LIBRARY AND TensorRT_LIBRARIES)
      set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES}
                             ${TensorRT_OnnxParser_LIBRARY})
      set(TensorRT_OnnxParser_FOUND TRUE)
    endif()

    if(WIN32)
      _tensorrt_find_dll(
        TensorRT_OnnxParser_DLL "nvonnxparser_${TensorRT_VERSION_MAJOR}.dll"
        nvonnxparser.dll)
    endif()
  endif()

  if("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
    find_path(
      TensorRT_Plugin_INCLUDE_DIR
      NAMES NvInferPlugin.h
      PATHS ${TensorRT_WELL_KNOWN_ROOT}/include)

    find_library(
      TensorRT_Plugin_LIBRARY
      NAMES "nvinfer_plugin_${TensorRT_VERSION_MAJOR}" nvinfer_plugin
      PATHS ${TensorRT_WELL_KNOWN_ROOT}/lib)

    if(TensorRT_Plugin_LIBRARY AND TensorRT_LIBRARIES)
      set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_Plugin_LIBRARY})
      set(TensorRT_Plugin_FOUND TRUE)
    endif()

    if(WIN32)
      _tensorrt_find_dll(
        TensorRT_Plugin_DLL "nvinfer_plugin_${TensorRT_VERSION_MAJOR}.dll"
        nvinfer_plugin.dll)
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  FOUND_VAR TensorRT_FOUND
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_LIBRARIES TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION_STRING
  HANDLE_COMPONENTS)

if(NOT TARGET TensorRT::NvInfer)
  add_library(TensorRT::NvInfer SHARED IMPORTED)
  target_include_directories(TensorRT::NvInfer SYSTEM
                             INTERFACE "${TensorRT_INCLUDE_DIR}")
  if(WIN32)
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION
                                                   "${TensorRT_DLL}")
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_IMPLIB
                                                   "${TensorRT_LIBRARY}")
  else()
    set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION
                                                   "${TensorRT_LIBRARY}")
  endif()
endif()

if(NOT TARGET TensorRT::OnnxParser AND "OnnxParser" IN_LIST
                                       TensorRT_FIND_COMPONENTS)
  add_library(TensorRT::OnnxParser SHARED IMPORTED)
  target_include_directories(TensorRT::OnnxParser SYSTEM
                             INTERFACE "${TensorRT_OnnxParser_INCLUDE_DIR}")
  target_link_libraries(TensorRT::OnnxParser INTERFACE TensorRT::NvInfer)
  if(WIN32)
    set_property(TARGET TensorRT::OnnxParser
                 PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_DLL}")
    set_property(TARGET TensorRT::OnnxParser
                 PROPERTY IMPORTED_IMPLIB "${TensorRT_OnnxParser_LIBRARY}")
  else()
    set_property(TARGET TensorRT::OnnxParser
                 PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_LIBRARY}")
  endif()
endif()

if(NOT TARGET TensorRT::Plugin AND "Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
  add_library(TensorRT::Plugin SHARED IMPORTED)
  target_include_directories(TensorRT::Plugin SYSTEM
                             INTERFACE "${TensorRT_Plugin_INCLUDE_DIR}")
  target_link_libraries(TensorRT::Plugin INTERFACE TensorRT::NvInfer)
  if(WIN32)
    set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION
                                                  "${TensorRT_Plugin_DLL}")
    set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_IMPLIB
                                                  "${TensorRT_Plugin_LIBRARY}")
  else()
    set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION
                                                  "${TensorRT_Plugin_LIBRARY}")
  endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_LIBRARIES)
