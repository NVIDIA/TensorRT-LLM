# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of NVIDIA CORPORATION nor the names
# of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
# EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

include_directories(
  ${TRTLLM_DIR}/cpp ${TRTLLM_DIR}/cpp/include ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CUDA_INCLUDE_DIRS})

set(TOP_LEVEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

function(add_gtest test_name test_src)
  add_executable(${test_name} ${test_src})

  target_link_libraries(
    ${test_name} PRIVATE gtest_main gmock_main triton-tensorrt-llm-backend
                         ${tensorrt_llm} ${CUDA_LIBRARIES})

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  target_compile_options(${test_name} PRIVATE -Wno-deprecated-declarations)

  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 30 DISCOVERY_MODE
                       PRE_TEST)
endfunction()

add_gtest(modelInstanceStateTest modelInstanceStateTest.cpp)
add_gtest(modelStateTest modelState.cpp)
add_gtest(utilsTest utilsTest.cpp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/first.json
               ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/second.json
               ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/third.json
               ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
