# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Environment for MiniMax-M3 aggregated TP2 serving on Rubin (SM107), shared by
# agg_c14.yaml / agg_c20.yaml / agg_c46.yaml / agg_c56.yaml. Source it in every
# worker: `source agg_env.sh` (README.md).

# Prefix token cache for multi-turn traffic: each turn resends the previous turn's prompt.
export TLLM_PREFIX_TOKEN_CACHE_ENTRIES=2048
export TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS=1073741824

# NCCL all-reduce with 64 CTAs.
export NCCL_MIN_CTAS=64
export NCCL_MAX_CTAS=64

# The all-reduce autotuner leaves NCCL_SYMMETRIC out, so no symmetric window is pre-allocated (more KV cache).
export TRTLLM_AR_TUNE_NCCL_SYMMETRIC=0
