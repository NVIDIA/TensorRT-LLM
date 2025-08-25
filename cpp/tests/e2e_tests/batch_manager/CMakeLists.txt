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

# guidedDecoderTest requires model tokenizer info, so it's easier to run it with
# e2e tests instead of unit tests.
add_gtest(guidedDecoderTest guidedDecoderTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
add_gtest(trtGptModelTest trtGptModelTest.cpp)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
target_link_libraries(trtGptModelRealDecoderTest PRIVATE testingUtils)
