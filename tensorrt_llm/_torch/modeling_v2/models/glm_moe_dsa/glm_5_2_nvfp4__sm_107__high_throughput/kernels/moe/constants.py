# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation buckets shared by MegaMoE workspace sizing and kernel dispatch."""

GEN_TOKEN_BUCKETS = (32, 64, 128, 192)
# Deferred top-k output is supported only for these generation buckets.
DEFERRED_TOPK_REDUCE_BUCKETS = (64, 128, 192)
