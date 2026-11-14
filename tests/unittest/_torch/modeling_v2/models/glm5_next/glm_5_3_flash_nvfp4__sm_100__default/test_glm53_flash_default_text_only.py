# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The split target serves text only: its multimodal admission hook refuses a request that carries media, which the
forward would otherwise serve as its text alone, and accepts a payload without media."""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling import (  # noqa: E501
    ModelingV2Glm53FlashNvfp4Sm100Default as Target,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin

pytestmark = pytest.mark.cpu_only

_HOOK = "validate_multimodal_request_data"


def test_the_target_defines_the_admission_hook():
    """``PyExecutor._validate_request`` calls the hook by the name ``MultimodalModelMixin`` declares."""
    assert callable(getattr(MultimodalModelMixin, _HOOK, None))
    assert _HOOK in vars(Target)


@pytest.mark.parametrize(
    "mm_data, carried",
    [
        ({"image": {"pixel_values": None, "image_grid_thw": None}}, "image"),
        ({"video": {"pixel_values_videos": None, "video_grid_thw": None}}, "video"),
        ({"audio": object()}, "audio"),
        ({"multimodal_embedding": object()}, "multimodal_embedding"),
        ({"image": object(), "video": object()}, "image, video"),
    ],
    ids=["image", "video", "audio", "embedding", "image-and-video"],
)
def test_media_is_refused_at_admission(mm_data, carried):
    # The hook reads no model state, so it runs without building the model.
    with pytest.raises(
        ValueError, match=f"does not accept multimodal input; the request carries {carried}$"
    ):
        getattr(Target, _HOOK)(types.SimpleNamespace(), mm_data)


@pytest.mark.parametrize(
    "mm_data",
    [{"image": None}, {"mrope_config": {"mrope_position_deltas": None}}],
    ids=["empty-media-entry", "no-media"],
)
def test_a_payload_without_media_is_accepted(mm_data):
    assert getattr(Target, _HOOK)(types.SimpleNamespace(), mm_data) is None
