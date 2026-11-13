# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PAD_FILL's switch line, and the fill declines CPU tensors (the stock fills then run). The install into the
sampler is ``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        pad_fill,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(pad_fill, "DEP_PAD_FILL", False)
    assert "TRTLLM_M31_DISABLE_DEP_PAD_FILL=1" in pad_fill.engagement()
    monkeypatch.setattr(pad_fill, "DEP_PAD_FILL", True)
    assert pad_fill.engagement().startswith("engaged")


def test_cpu_rows_are_declined():
    assert pad_fill.pad_rows_fill(torch.zeros(4, 8), torch.zeros(4, dtype=torch.bool), 0.0) is None
