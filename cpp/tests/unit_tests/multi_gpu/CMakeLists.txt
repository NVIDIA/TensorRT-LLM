# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

add_subdirectory(kernels)

add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)
target_link_libraries(cacheTransceiverTest PRIVATE ${Python3_LIBRARIES})

add_gtest(
  kvCacheManagerV2HostMemNumaTest
  "kvCacheManagerV2HostMemNumaTest.cpp;../batch_manager/kvCacheManagerV2PoisonCheck.cpp"
)
target_include_directories(
  kvCacheManagerV2HostMemNumaTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
# Queries libnuma and get_mempolicy directly to see where pages landed.
target_link_libraries(kvCacheManagerV2HostMemNumaTest PRIVATE numa)

add_gtest(mpiUtilsTest mpiUtilsTest.cpp)
add_gtest(userBufferTest userBufferTest.cpp)
add_gtest(ncclUtilsTest ncclUtilsTest.cpp)
target_link_libraries(ncclUtilsTest PRIVATE ${Python3_LIBRARIES})
if(BUILD_PYT)
  target_compile_definitions(cacheTransceiverTest PUBLIC BUILD_PYT)
  target_link_libraries(cacheTransceiverTest PUBLIC ${TORCH_LIBRARIES})
endif()
if(BUILD_PYT)
  target_compile_definitions(ncclUtilsTest PUBLIC BUILD_PYT)
  target_link_libraries(ncclUtilsTest PUBLIC ${TORCH_LIBRARIES})
endif()
