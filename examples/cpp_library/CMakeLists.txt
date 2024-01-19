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
# the License. cmake needs this line
cmake_minimum_required(VERSION 3.1)
# cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME trt_llm_plugins_cpp_load_example)
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)
# set(CMAKE_BUILD_TYPE debug)

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(${TARGET_NAME} main.cpp)

# Link your application with CUDA libraries
target_link_libraries(${TARGET_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${TARGET_NAME} LINK_PRIVATE cudnn)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer_plugin_tensorrt_llm)

target_include_directories(${TARGET_NAME} PUBLIC /usr/local/cuda/include)
