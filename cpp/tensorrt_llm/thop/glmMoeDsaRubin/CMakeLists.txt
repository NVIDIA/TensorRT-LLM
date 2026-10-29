# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

# Preserve each binding's Torch target and its compile definitions.

target_sources(th_common_attention PRIVATE glmMlaKvHalfAppendOp.cpp
                                           glmSparseMlaFlashDecodeOp.cpp)

target_sources(
  th_common_gemm_quant PRIVATE glmDenseMlpSmallMOp.cpp glmFusedAGemmOp.cpp
                               glmFusedEmbeddingOp.cpp)

target_sources(th_common_moe PRIVATE glmRouterGemmOp.cpp glmLowMBf16MoeOp.cpp
                                     glmRoutedExpertsOp.cpp)

target_sources(
  th_common_distributed
  PRIVATE glmDsaHistTopKOp.cpp glmMnnvlAllgatherOp.cpp glmMnnvlHopOp.cpp
          glmDraftSampleGlueOp.cpp glmIntraNodeReduceOp.cpp)

target_sources(th_common_decoding PRIVATE glmVerifyHeadAcceptOp.cpp
                                          glmVerifyHeadSampleOp.cpp)
