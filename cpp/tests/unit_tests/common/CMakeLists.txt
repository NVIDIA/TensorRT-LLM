# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
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

add_gtest(cudaProfilerUtilsTest cudaProfilerUtilsTest.cpp)
add_gtest(cudaUtilsTest cudaUtilsTest.cpp)
add_gtest(attentionWorkspaceTest attentionWorkspaceTest.cpp)
add_gtest(memoryUtilsTest memoryUtilsTest.cu)
add_tllm_gtest(
  optionalRefTest
  OWNER
  common
  SOURCES
  optionalRefTest.cpp
  LINK_LIBRARIES
  tllm::common_headers
  LABELS
  "unit;common;host")
add_tllm_gtest(
  quantizationTest
  OWNER
  common
  SOURCES
  quantizationTest.cpp
  LINK_LIBRARIES
  tllm::common_headers
  LABELS
  "unit;common;host")
add_tllm_gtest(
  stlUtilsTest
  OWNER
  common
  SOURCES
  stlUtilsTest.cpp
  LINK_LIBRARIES
  tllm::common_headers
  LABELS
  "unit;common;host")
add_tllm_gtest(
  stringUtilsTest
  OWNER
  common
  SOURCES
  stringUtilsTest.cpp
  LINK_LIBRARIES
  tllm::common_error
  LABELS
  "unit;common;host")
add_tllm_gtest(
  timestampUtilsTest
  OWNER
  common
  SOURCES
  timestampUtilsTest.cpp
  LINK_LIBRARIES
  tllm::common_logging
  Threads::Threads
  LABELS
  "unit;common;host")
add_tllm_gtest(
  tllmExceptionTest
  OWNER
  common
  SOURCES
  tllmExceptionTest.cpp
  LINK_LIBRARIES
  tllm::common_logging
  LABELS
  "unit;common;host")
add_gtest(cudaDriverWrapperTest cudaDriverWrapperTest.cpp)
add_tllm_gtest(
  loggerTest
  OWNER
  common
  SOURCES
  loggerTest.cpp
  LINK_LIBRARIES
  tllm::common_logging
  Threads::Threads
  LABELS
  "unit;common;host")
