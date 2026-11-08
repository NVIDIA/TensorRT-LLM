#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#

find_path(NUMA_INCLUDE_DIR NAMES numa.h)
find_library(NUMA_LIBRARY NAMES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA REQUIRED_VARS NUMA_LIBRARY
                                                     NUMA_INCLUDE_DIR)

if(NUMA_FOUND AND NOT TARGET NUMA::NUMA)
  add_library(NUMA::NUMA UNKNOWN IMPORTED)
  set_target_properties(
    NUMA::NUMA PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
endif()

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)
