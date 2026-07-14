# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal example showing Soft-FSD params with string divergence names.
# This script is illustrative; adapt runners and inputs to your environment.

import torch

from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp


def example_usage(target_runner: ModelRunnerCpp, draft_tokens_list, draft_logits_list=None):
    # Example prompt (batch size 1)
    batch_input_ids = [torch.tensor([1, 2, 3], dtype=torch.int32)]

    kwargs = {
        "batch_input_ids": batch_input_ids,
        "draft_tokens_list": draft_tokens_list,
        "draft_logits_list": draft_logits_list,
        "max_new_tokens": 16,
        # Soft-FSD controls:
        "fsd_threshold": 0.05,
        "fsd_divergence_type": "js",  # also supports "kl", "tv", "reverse_kl"
    }

    return target_runner.generate(**kwargs)
