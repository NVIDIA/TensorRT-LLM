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

add_subdirectory(kernels)

add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)
target_link_libraries(cacheTransceiverTest PRIVATE ${Python3_LIBRARIES})

add_gtest(mpiUtilsTest mpiUtilsTest.cpp)
add_gtest(userBufferTest userBufferTest.cpp)
