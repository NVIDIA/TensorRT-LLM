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
  th_common_distributed PRIVATE glmDsaHistTopKOp.cpp glmMnnvlAllgatherOp.cpp
                                glmMnnvlHopOp.cpp glmDraftSampleGlueOp.cpp)

target_sources(th_common_decoding PRIVATE glmVerifyHeadAcceptOp.cpp)

# torch.ops.trtllm.glm_moe_dsa_* ops of the GLM-5.2 GB300 target, one file per
# kernel of cpp/tensorrt_llm/kernels/glmMoeDsa/'s own library. The parent
# defines th_common_glm_moe_dsa.
target_sources(
  th_common_glm_moe_dsa
  PRIVATE crossRankTopkClusterMergeOp.cpp
          crossRankTopkGlobalEmitOp.cpp
          crossRankTopkOp.cpp
          denseMlpOp.cpp
          draftLiveRowHopOp.cpp
          draftRoutedMoeOp.cpp
          draftTailOp.cpp
          indexerLayerRopeKvAppendOp.cpp
          postAttentionHopFromPartialsOp.cpp
          postAttentionHopOp.cpp
          postAttentionHopRoutingOp.cpp
          ropeKvAppendOp.cpp
          sparseMlaDecodeOp.cpp)
