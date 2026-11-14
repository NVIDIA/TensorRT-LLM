# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_LOGITS_WIDEN's switch line, and the widening declines CPU tensors. The install into the logits processor is
``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        logits_widen,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(logits_widen, "DEP_LOGITS_WIDEN", False)
    assert "TRTLLM_M31_DISABLE_DEP_LOGITS_WIDEN=1" in logits_widen.engagement()
    monkeypatch.setattr(logits_widen, "DEP_LOGITS_WIDEN", True)
    assert logits_widen.engagement().startswith("engaged")


def test_cpu_logits_are_declined():
    assert logits_widen.widen(torch.zeros(2, 8, dtype=torch.bfloat16)) is None
