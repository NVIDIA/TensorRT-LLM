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

if(${BUILD_PYT})
  add_tllm_gtest(
    allreduceTacticCacheTest
    OWNER
    common
    SOURCES
    allreduceTacticCacheTest.cpp
    LINK_LIBRARIES
    tllm::common_headers
    ${TORCH_LIBRARIES}
    CUDA::cudart
    LABELS
    "unit;common;host")
  add_tllm_gtest(
    thUtilsTest
    SOURCES
    thUtilsTest.cpp
    eagerMoeWorkspaceTest.cpp
    FULL_STACK
    LABELS
    "unit;full-stack;gpu")
  target_link_libraries(thUtilsTest PUBLIC th_utils ${Python3_LIBRARIES}
                                           ${TORCH_LIBRARIES})
endif()
