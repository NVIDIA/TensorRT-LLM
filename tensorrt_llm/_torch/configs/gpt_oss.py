# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import GptOssConfig


def normalize_gpt_oss_config(config: GptOssConfig) -> None:
    """Translate legacy checkpoint fields while preserving native HF settings."""
    if hasattr(config, "experts_per_token"):
        config.num_local_experts = config.num_experts
        config.num_experts_per_tok = config.experts_per_token

    # Native configs also expose num_experts as an alias for num_local_experts.
    # Only the legacy rotary fields require conversion to HF's rope parameters.
    if hasattr(config, "rope_scaling_factor"):
        config.rope_scaling = {
            "factor": config.rope_scaling_factor,
            "beta_fast": config.rope_ntk_beta,
            "beta_slow": config.rope_ntk_alpha,
            "original_max_position_embeddings": config.initial_context_length,
        }
