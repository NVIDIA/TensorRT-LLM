# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

add_library(modelSpec SHARED modelSpec.cpp modelSpecBinding.cpp)
set_target_properties(modelSpec PROPERTIES PREFIX "")
set_target_properties(modelSpec PROPERTIES OUTPUT_NAME "model_spec")
target_link_libraries(modelSpec PRIVATE tensorrt_llm)

# Static lib for other test to link
add_library(modelSpecStatic STATIC modelSpec.cpp)

add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)
target_include_directories(cacheTransceiverTest PRIVATE ${NIXL_INCLUDE_DIR})
target_link_libraries(
  cacheTransceiverTest
  PRIVATE ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY} ${SERDES_LIBRARY}
          ${UCX_BACKEND_LIBRARY} ${UCX_UTILS_LIBRARY} ${GDS_BACKEND_LIBRARY})

# Add rpath for runtime library loading
if(NOT WIN32)
  set_target_properties(
    cacheTransceiverTest
    PROPERTIES BUILD_RPATH "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
               INSTALL_RPATH "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
               INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

add_gtest(trtGptModelTest trtGptModelTest.cpp)
target_link_libraries(trtGptModelTest PRIVATE modelSpecStatic)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
target_link_libraries(trtGptModelRealDecoderTest PRIVATE modelSpecStatic
                                                         testingUtils)

add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
add_gtest(guidedDecoderTest guidedDecoderTest.cpp)
add_gtest(blockKeyTest blockKeyTest.cpp)
