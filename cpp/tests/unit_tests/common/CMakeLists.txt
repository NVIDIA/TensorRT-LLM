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

add_gtest(cudaProfilerUtilsTest cudaProfilerUtilsTest.cpp)
add_gtest(cudaUtilsTest cudaUtilsTest.cpp)
add_gtest(memoryUtilsTest memoryUtilsTest.cu)
add_gtest(optionalRefTest optionalRefTest.cpp)
add_gtest(quantizationTest quantizationTest.cpp)
add_gtest(stlUtilsTest stlUtilsTest.cpp)
add_gtest(stringUtilsTest stringUtilsTest.cpp)
add_gtest(timestampUtilsTest timestampUtilsTest.cpp)
add_gtest(tllmExceptionTest tllmExceptionTest.cpp)
add_gtest(cudaDriverWrapperTest cudaDriverWrapperTest.cpp)
