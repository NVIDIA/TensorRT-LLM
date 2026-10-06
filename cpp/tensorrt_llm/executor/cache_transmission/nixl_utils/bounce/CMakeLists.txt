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

add_library(
  tllm_executor_bounce_core STATIC EXCLUDE_FROM_ALL
  BuddyAllocator.cpp CreditScheduler.cpp BounceTransferPlan.cpp
  BounceMessage.cpp)
add_library(tllm::executor_bounce_core ALIAS tllm_executor_bounce_core)
target_link_libraries(tllm_executor_bounce_core PUBLIC tllm::common_logging
                                                       Threads::Threads)
target_compile_features(tllm_executor_bounce_core PUBLIC cxx_std_20)
target_compile_options(tllm_executor_bounce_core PRIVATE -Wno-error)
set_target_properties(tllm_executor_bounce_core
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_executor_bounce_cuda STATIC EXCLUDE_FROM_ALL
            ExecPool.cpp GatherScatterKernel.cu)
add_library(tllm::executor_bounce_cuda ALIAS tllm_executor_bounce_cuda)
target_link_libraries(tllm_executor_bounce_cuda
                      PUBLIC tllm::executor_bounce_core tllm::common_cuda)
target_compile_features(tllm_executor_bounce_cuda PUBLIC cxx_std_20)
target_compile_options(tllm_executor_bounce_cuda
                       PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>)
set_target_properties(
  tllm_executor_bounce_cuda
  PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(tllm_executor_bounce_arena STATIC EXCLUDE_FROM_ALL BounceArena.cpp)
add_library(tllm::executor_bounce_arena ALIAS tllm_executor_bounce_arena)
target_link_libraries(
  tllm_executor_bounce_arena
  PUBLIC tllm::common_cuda
  PRIVATE tllm::kv_cache_legacy)
target_compile_features(tllm_executor_bounce_arena PUBLIC cxx_std_20)
target_compile_options(tllm_executor_bounce_arena PRIVATE -Wno-error)
set_target_properties(tllm_executor_bounce_arena
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)
