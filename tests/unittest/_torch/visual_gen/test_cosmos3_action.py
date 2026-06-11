# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 action sizing helpers (no checkpoint / GPU required).

Run:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_action.py -v
"""

import PIL.Image
import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    VIDEO_RES_SIZE_INFO,
    find_closest_target_size,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline

pytestmark = pytest.mark.cosmos3


class TestFindClosestTargetSize:
    @pytest.mark.parametrize(
        "input_h,input_w,action_resolution,expected",
        [
            (480, 832, 480, (832, 480)),
            (832, 480, 480, (480, 832)),
            (512, 512, 480, (640, 640)),
            (704, 1280, 704, (1280, 704)),
            (256, 256, 256, (256, 256)),
            (720, 1280, 720, (1280, 720)),
        ],
    )
    def test_picks_closest_aspect_bucket(self, input_h, input_w, action_resolution, expected):
        assert find_closest_target_size(input_h, input_w, action_resolution) == expected

    def test_accepts_string_and_int_resolution_keys(self):
        ref_h, ref_w = 480, 832
        assert find_closest_target_size(ref_h, ref_w, 480) == find_closest_target_size(
            ref_h, ref_w, "480"
        )

    def test_unknown_resolution_raises(self):
        with pytest.raises(ValueError, match="Unknown Cosmos3 action resolution"):
            find_closest_target_size(480, 832, 1080)

    @pytest.mark.parametrize("action_resolution", sorted(VIDEO_RES_SIZE_INFO))
    def test_all_buckets_have_aspect_entries(self, action_resolution):
        assert VIDEO_RES_SIZE_INFO[action_resolution]


class TestResolveActionSize:
    @staticmethod
    def _ref_image(width: int, height: int) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (width, height))

    def test_explicit_height_and_width_are_unchanged(self):
        ref = self._ref_image(832, 480)
        assert Cosmos3OmniMoTPipeline._resolve_action_size(400, 600, ref, 480) == (400, 600)

    def test_unset_height_and_width_use_action_resolution_bucket(self):
        ref = self._ref_image(832, 480)
        assert Cosmos3OmniMoTPipeline._resolve_action_size(None, None, ref, 480) == (480, 832)

    def test_partial_height_fills_width_from_bucket(self):
        ref = self._ref_image(832, 480)
        assert Cosmos3OmniMoTPipeline._resolve_action_size(400, None, ref, 480) == (400, 832)

    def test_partial_width_fills_height_from_bucket(self):
        ref = self._ref_image(832, 480)
        assert Cosmos3OmniMoTPipeline._resolve_action_size(None, 600, ref, 480) == (480, 600)


class TestActionResolutionExtraParam:
    def test_extra_param_spec_uses_action_resolution_key(self):
        spec = COSMOS3_EXTRA_SPECS["action_resolution"]
        assert spec.type == "int"
        assert spec.default == 480
