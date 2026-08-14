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

# Create the pg_utils shared library
add_library(pg_utils SHARED pgUtils.cpp)

set_property(TARGET pg_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

# Include directories
target_include_directories(pg_utils PUBLIC ${PROJECT_SOURCE_DIR}/include
                                           ${TORCH_INCLUDE_DIRS})

target_link_libraries(pg_utils PUBLIC ${TORCH_LIBRARIES})

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)
