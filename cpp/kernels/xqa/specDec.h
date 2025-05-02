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

#include "defines.h"
#if SPEC_DEC

struct SpecDecParams
{
    uint32_t qSeqLen;
    uint32_t const* qCuSeqLens; // [nbReq + 1]
    MaskType const* mask;       // [nbReq][qSeqLen][divUp(qSeqLen, 32)] or [qCuSeqLen[nbReq]][divUp(qSeqLen, 32)]
};

#endif
