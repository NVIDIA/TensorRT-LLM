# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supported Beam model configuration and numerical constants shared across components."""

ROUTER_EPS = 1e-20

SM = (10, 3)

WORLD_SIZE = 4

#: The configuration this component computes, as the checkpoint's config.json states it.
SUPPORTED_CONFIG = {
    "model_type": "beam",
    "hidden_size": 5120,
    "intermediate_size": 20480,
    "moe_intermediate_size": 2560,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "vocab_size": 151936,
    "num_hidden_layers": 52,
    "rms_norm_eps": 1e-06,
    "routed_scaling_factor": 2.5,
    "final_logit_softcapping": 30.0,
    "sliding_window": 512,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
}
SLIDING_ROPE_THETA = 10000.0
