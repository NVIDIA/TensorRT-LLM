# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP-fused target serves text only: its multimodal admission hook refuses a request that carries image or video
input, which the forward would otherwise serve as its text alone."""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default_mtp_fused.modeling import (  # noqa: E501
    ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused as Target,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin

pytestmark = pytest.mark.cpu_only

_HOOK = "validate_multimodal_request_data"


def test_the_target_defines_the_admission_hook():
    """``PyExecutor._validate_request`` calls the hook by the name ``MultimodalModelMixin`` declares."""
    assert callable(getattr(MultimodalModelMixin, _HOOK, None))
    assert _HOOK in vars(Target)


@pytest.mark.parametrize(
    "mm_data",
    [
        {"image": {"pixel_values": None, "image_grid_thw": None}},
        {"video": {"pixel_values_videos": None, "video_grid_thw": None}},
    ],
    ids=["image", "video"],
)
def test_multimodal_input_is_refused_at_admission(mm_data):
    # The hook reads no model state, so it runs without building the model.
    with pytest.raises(ValueError, match="serves text only"):
        getattr(Target, _HOOK)(types.SimpleNamespace(), mm_data)
