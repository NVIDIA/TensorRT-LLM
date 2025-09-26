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

if(${BUILD_PYT})
  add_gtest(thUtilsTest thUtilsTest.cpp)
  target_link_libraries(thUtilsTest PUBLIC th_utils ${Python3_LIBRARIES}
                                           ${TORCH_LIBRARIES})
endif()
