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

# Same-node direct peer copy for the NIXL transfer agent (opt-in at runtime via
# TRTLLM_NIXL_DIRECT_COPY=1). Linked whole-archive into the NIXL wrapper.
add_library(tllm_executor_direct_copy STATIC EXCLUDE_FROM_ALL
            DirectPeerCopy.cpp DirectCopyKernel.cu)
add_library(tllm::executor_direct_copy ALIAS tllm_executor_direct_copy)
target_include_directories(tllm_executor_direct_copy
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(
  tllm_executor_direct_copy PUBLIC tllm::common_logging CUDA::cudart
                                   CUDA::cuda_driver)
target_compile_features(tllm_executor_direct_copy PUBLIC cxx_std_20)
set_target_properties(
  tllm_executor_direct_copy
  PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
