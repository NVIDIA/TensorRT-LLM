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

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batch_manager)
  add_subdirectory(batch_manager)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/executor)
  add_subdirectory(executor)
endif()

add_subdirectory(common)
add_subdirectory(kernels)
add_subdirectory(multi_gpu)
add_subdirectory(layers)
add_subdirectory(runtime)
add_subdirectory(thop)
add_subdirectory(utils)
