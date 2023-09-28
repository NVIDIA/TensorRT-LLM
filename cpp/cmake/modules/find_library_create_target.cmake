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

macro(find_library_create_target target_name lib libtype hints)
  message(
    STATUS
      "========================= Importing and creating target ${target_name} =========================="
  )
  message(STATUS "Looking for library ${lib}")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_library(
      ${lib}_LIB_PATH ${lib}${TRT_DEBUG_POSTFIX}
      HINTS ${hints}
      NO_DEFAULT_PATH)
  endif()
  find_library(${lib}_LIB_PATH ${lib} HINTS ${hints} NO_DEFAULT_PATH)
  find_library(${lib}_LIB_PATH ${lib})
  message(STATUS "Library that was found ${${lib}_LIB_PATH}")
  add_library(${target_name} ${libtype} IMPORTED)
  set_target_properties(
    ${target_name} PROPERTIES IMPORTED_LOCATION ${${lib}_LIB_PATH}
                              IMPORTED_IMPLIB ${${lib}_LIB_PATH})
  message(
    STATUS
      "=========================================================================================="
  )
endmacro()
