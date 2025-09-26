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

find_library(NCCL_LIBRARY NAMES nccl)

if(NCCL_LIBRARY)
  set(NCCL_LIBRARIES ${NCCL_LIBRARIES} ${NCCL_LIBRARY})
endif()

find_library(NCCL_STATIC_LIBRARY NAMES nccl_static)

if(NCCL_STATIC_LIBRARY)
  set(NCCL_LIBRARIES ${NCCL_LIBRARIES} ${NCCL_STATIC_LIBRARY})
endif()

find_path(NCCL_INCLUDE_DIR NAMES nccl.h)

function(_nccl_get_version)
  unset(NCCL_VERSION_STRING PARENT_SCOPE)
  set(_hdr_file "${NCCL_INCLUDE_DIR}/nccl.h")

  if(NOT EXISTS "${_hdr_file}")
    return()
  endif()

  file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define NCCL_.*")

  foreach(TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NCCL_${TYPE} [0-9]+" NCCL_TYPE_STRING
                 ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]+" NCCL_VERSION_${TYPE} ${NCCL_TYPE_STRING})
  endforeach(TYPE)

  set(NCCL_VERSION_STRING
      "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}"
      PARENT_SCOPE)
endfunction()

_nccl_get_version()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL
  FOUND_VAR NCCL_FOUND
  REQUIRED_VARS NCCL_LIBRARIES NCCL_INCLUDE_DIR
  VERSION_VAR NCCL_VERSION_STRING)

if(NCCL_LIBRARY)
  add_library(NCCL::nccl SHARED IMPORTED)
  target_include_directories(NCCL::nccl SYSTEM INTERFACE "${NCCL_INCLUDE_DIR}")
  set_property(TARGET NCCL::nccl PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARY}")
endif()

if(NCCL_STATIC_LIBRARY)
  add_library(NCCL::nccl_static STATIC IMPORTED)
  target_include_directories(NCCL::nccl_static SYSTEM
                             INTERFACE "${NCCL_INCLUDE_DIR}")
  set_property(TARGET NCCL::nccl_static PROPERTY IMPORTED_LOCATION
                                                 "${NCCL_STATIC_LIBRARY}")
endif()
