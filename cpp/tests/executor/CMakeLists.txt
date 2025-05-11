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

add_gtest(executorMockTest executorMockTest.cpp)
add_gtest(executorTest executorTest.cpp)
target_link_libraries(executorTest PRIVATE testingUtils)
add_gtest(encDecTest encDecTest.cpp)
target_link_libraries(encDecTest PRIVATE testingUtils)
add_gtest(disaggExecutorTest disaggExecutorTest.cpp)
target_link_libraries(disaggExecutorTest PRIVATE testingUtils)
