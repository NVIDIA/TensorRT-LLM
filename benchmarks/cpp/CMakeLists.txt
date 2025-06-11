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

include_directories(${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(benchmarks)

if(NOT TARGET cxxopts::cxxopts)
  set(CXXOPTS_SRC_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/cxxopts)
  add_subdirectory(${CXXOPTS_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxopts)
endif()

function(add_benchmark test_name test_src)
  add_executable(${test_name} ${test_src} utils/utils.cpp)

  target_link_libraries(
    ${test_name} PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm
                        cxxopts::cxxopts)

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  add_dependencies(benchmarks ${test_name})
endfunction()

add_benchmark(bertBenchmark bertBenchmark.cpp)
add_benchmark(gptManagerBenchmark gptManagerBenchmark.cpp)
add_benchmark(disaggServerBenchmark disaggServerBenchmark.cpp)
