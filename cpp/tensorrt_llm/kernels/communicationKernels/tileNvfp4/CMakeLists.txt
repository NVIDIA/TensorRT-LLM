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

# SM107 Push-Combine MoE all-to-all combine (tile-wise packed NVFP4).
#
# The kernels use sm_107a-only PTX (system-scope cp.async.bulk S2G and u64
# release/acquire flags), so they are not part of the multi-arch
# libtensorrt_llm/libth_common. They are built into a separate Torch extension,
# tile_push_comm_ops.so, that registers the trtllm::moe_a2a_push_* ops and is
# loaded on demand by TRTLLM_FORCE_COMM_METHOD=PUSH_COMBINE
# (tensorrt_llm/_torch/modules/fused_moe/communication/tile_tma_forced_loader.py).
# The library is only built when SM107 is targeted.

if(NOT "107" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
  message(
    STATUS "Skipping Push-Combine (tile_push_comm_ops): SM107 is not targeted")
  return()
endif()

file(GLOB TILE_PUSH_COMM_SRC moeAlltoAllTileNvfp4V2Push*.cu
     moeAlltoAllTileNvfp4V2Push*.cpp)

add_library(tile_push_comm_ops SHARED ${TILE_PUSH_COMM_SRC})
set_target_properties(
  tile_push_comm_ops
  PROPERTIES PREFIX ""
             OUTPUT_NAME tile_push_comm_ops
             POSITION_INDEPENDENT_CODE ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON)
set_cuda_architectures(tile_push_comm_ops 107)
target_include_directories(tile_push_comm_ops
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  tile_push_comm_ops PRIVATE ${TORCH_LIBRARIES} ${Python3_LIBRARIES}
                             ${SHARED_TARGET})

if(NOT WIN32)
  # Installed next to libtensorrt_llm.so in tensorrt_llm/libs.
  set_target_properties(
    tile_push_comm_ops
    PROPERTIES BUILD_RPATH "$ORIGIN" LINK_FLAGS
                                     "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()
