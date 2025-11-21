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

add_gtest(blockKeyTest blockKeyTest.cpp)
add_gtest(cacheTransBufferTest cacheTransBufferTest.cpp)
add_gtest(capacitySchedulerTest capacitySchedulerTest.cpp)
add_gtest(contextProgressTest contextProgressTest.cu)
add_gtest(evictionPolicyTest evictionPolicyTest.cpp)
add_gtest(kvCacheManagerTest kvCacheManagerTest.cpp)
add_gtest(kvCacheUtilsTest kvCacheUtilsTest.cpp)
add_gtest(llmRequestTest llmRequestTest.cpp)
add_gtest(microBatchSchedulerTest microBatchSchedulerTest.cpp)
add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(staticThreadPoolTest staticThreadPoolTest.cpp)
