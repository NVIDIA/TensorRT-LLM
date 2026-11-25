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

function(tllm_localize_cutlass_runtime_helpers target)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Cluster launch helpers must use the CUDA runtime that registered the
    # kernel.
    set(symbols_map
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../templates/cutlass_runtime_symbols.map"
    )
    target_link_options(${target} PRIVATE
                        "LINKER:--version-script,${symbols_map}")
    set_property(
      TARGET ${target}
      APPEND
      PROPERTY LINK_DEPENDS "${symbols_map}")
  endif()
endfunction()
