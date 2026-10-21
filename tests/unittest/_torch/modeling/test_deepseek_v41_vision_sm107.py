# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41VisionConfig
from tensorrt_llm._torch.models.deepseek_v41_vision import DeepseekV41VisionModel


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM 10.7 only",
)
def test_vision_attention_runs_on_sm107() -> None:
    torch.manual_seed(0)
    config = DeepseekV41VisionConfig(
        num_hidden_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=64,
        patch_size=2,
        downsample_ratio=3,
    )
    model = DeepseekV41VisionModel(config).cuda().bfloat16().eval()
    patches = torch.randn(32 * 32, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = model(patches, 32, 32)
        with sdpa_kernel([SDPBackend.MATH]):
            expected = model(patches, 32, 32)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
