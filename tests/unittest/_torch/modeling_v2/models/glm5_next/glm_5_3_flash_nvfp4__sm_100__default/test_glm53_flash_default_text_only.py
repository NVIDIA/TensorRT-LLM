# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission for the text-only GLM target."""

import pytest
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling import (  # noqa: E501
    ModelingV2Glm53FlashNvfp4Sm100Default as Target,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import _validate_multimodal_input

pytestmark = pytest.mark.cpu_only


def test_media_is_refused_at_admission():
    model = Target.__new__(Target)
    nn.Module.__init__(model)
    with pytest.raises(ValueError, match="no local multimodal encoder"):
        _validate_multimodal_input(model, {"image": {"pixel_values": object()}})
