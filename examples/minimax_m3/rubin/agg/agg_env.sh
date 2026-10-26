# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Environment for MiniMax-M3 aggregated TP2 serving on Rubin (SM107), shared by
# agg_c14.yaml / agg_c20.yaml / agg_c46.yaml / agg_c56.yaml. Source it in every
# worker: `source agg_env.sh` (README.md).

# Prefix token cache for multi-turn traffic: each turn resends the previous turn's prompt.
export TLLM_PREFIX_TOKEN_CACHE_ENTRIES=2048
export TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS=1073741824

# Context-path items that the aggregated deployment keeps off: the dense-MLP SwiGLU + MXFP8 producer, the CLC GEMM,
# the shared-expert fold into the MoE finalize, the drafter's last-row context attention and the fused embedding
# gather.
export TRTLLM_CTX_SWIGLU_MXFP8=0
export TRTLLM_CTX_CLC_GEMM=0
export TLLM_M3_MOE_FOLD_SHARED=0
export TRTLLM_CTX_DRAFTER_LAST_ROW=0
export TRTLLM_CTX_DRAFTER_LAST_ROW_DECODE=0
export TRTLLM_CTX_EMBEDDING_GATHER=0
# Legacy-model items the aggregated deployment keeps off: the drafter's dead-row skip and the generation router_tc.
export TRTLLM_CTX_DRAFTER_DEAD_ROWS=0
export TRTLLM_M3_ROUTER_TC_GEN=0

# NCCL all-reduce with 64 CTAs.
export NCCL_MIN_CTAS=64
export NCCL_MAX_CTAS=64

# The all-reduce autotuner leaves NCCL_SYMMETRIC out, so no symmetric window is pre-allocated (more KV cache).
export TRTLLM_AR_TUNE_NCCL_SYMMETRIC=0
