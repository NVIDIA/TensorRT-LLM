/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// for IDE parser
#if defined(Q_CREATOR_RUN) || defined(__CLION_IDE__) || defined(__INTELLISENSE__) || defined(IN_KDEVELOP_PARSER)       \
    || defined(__JETBRAINS_IDE__) || defined(__CLANGD__)
#define IS_IN_IDE_PARSER 1
#else
#define IS_IN_IDE_PARSER 0
#endif
