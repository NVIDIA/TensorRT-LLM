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

add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)

add_gtest(trtGptModelTest trtGptModelTest.cpp)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
target_link_libraries(trtGptModelRealDecoderTest PRIVATE testingUtils)

add_gtest(peftCacheManagerTest peftCacheManagerTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
add_gtest(guidedDecoderTest guidedDecoderTest.cpp)
add_gtest(blockKeyTest blockKeyTest.cpp)

add_gtest(metaTransceiverTest metaTransceiverTest.cpp)
target_link_libraries(metaTransceiverTest PRIVATE ${ZeroMQ_LIBRARIES})
