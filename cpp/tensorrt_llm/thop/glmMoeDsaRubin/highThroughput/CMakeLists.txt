# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

target_sources(th_common_attention PRIVATE glmNvfp4OwnerWorklistOp.cpp)
target_sources(th_common_moe PRIVATE glmNoAuxTcFp4QuantOp.cpp)

target_sources(th_common_distributed PRIVATE glmMtpDraftSampleGlueOp.cpp
                                             glmMtpHeadGatherOp.cpp)
