/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include "platform.h"

#if IS_IN_IDE_PARSER

#ifndef __CUDACC__
#define __CUDACC__ 1
#endif

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 900
#endif

#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 12
#endif
#ifndef __CUDACC_VER_MINOR__
#define __CUDACC_VER_MINOR__ 9
#endif

#if __CUDA_ARCH__ == 900
#ifndef __CUDA_ARCH_FEAT_SM90_ALL
#define __CUDA_ARCH_FEAT_SM90_ALL
#endif
#endif

#endif
