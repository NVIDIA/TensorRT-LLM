# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

# Google Benchmark Preparation - Same as google test ../tests/CMakeLists.txt
# Google Benchmark is provided under Apache-2.0 license
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "Disable google-benchmark tests")
set(BENCHMARK_ENABLE_INSTALL
    OFF
    CACHE INTERNAL "Disable google-benchmark install")
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3)
FetchContent_MakeAvailable(googlebenchmark)

add_custom_target(micro_benchmarks)

include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

function(add_benchmark test_name test_src)
  add_executable(${test_name} ${test_src})

  message("Linking with ${SHARED_TARGET}")
  target_link_libraries(${test_name} PUBLIC ${SHARED_TARGET}
                                            benchmark::benchmark)

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(
    ${test_name} PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}"
                        USING_OSS_CUTLASS_MOE_GEMM)

  add_dependencies(micro_benchmarks ${test_name})
endfunction()

add_benchmark(mixtureOfExpertsBackendBenchmark
              mixtureOfExpertsBackendBenchmarkLauncher.cu)
