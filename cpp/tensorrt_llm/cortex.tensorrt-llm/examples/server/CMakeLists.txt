# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
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
# C++17
# cortex.tensorrt-llm init
cmake_minimum_required(VERSION 3.5)
project(server)
find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
  set(LINKER_FLAGS -ldl)
endif()

include(CheckIncludeFileCXX)
# CPP version
check_include_file_cxx(any HAS_ANY)
check_include_file_cxx(string_view HAS_STRING_VIEW)
check_include_file_cxx(coroutine HAS_COROUTINE)
if(HAS_ANY
  AND HAS_STRING_VIEW
  AND HAS_COROUTINE)
  set(CMAKE_CXX_STANDARD 20)
elseif(HAS_ANY AND HAS_STRING_VIEW)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME}
    server.cc
    dylib.h
    httplib.h
)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build_deps/_install)
set(CORTEX_COMMON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../base/)

find_library(JSONCPP
    NAMES jsoncpp
    HINTS "${THIRD_PARTY_PATH}/lib"
)

find_library(TRANTOR
    NAMES trantor
    HINTS "${THIRD_PARTY_PATH}/lib"
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${JSONCPP} ${TRANTOR} ${LINKER_FLAGS}
                                              ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${PROJECT_NAME} PRIVATE 
                            ${CORTEX_COMMON_PATH}
                            ${THIRD_PARTY_PATH}/include)
