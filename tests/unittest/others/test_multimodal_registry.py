# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from tensorrt_llm.inputs.registry import (MULTIMODAL_PLACEHOLDER_REGISTRY,
                                          BaseMultimodalInputProcessor,
                                          MultimodalPlaceholderMetadata,
                                          MultimodalPlaceholderPlacement)


class TestMultimodalPlaceholderRegistry(unittest.TestCase):

    def setUp(self):
        self.model_type = "test_model_type"
        self.placeholder_metadata = MultimodalPlaceholderMetadata(
            placeholder_map={
                "image": "IMAGE_PLACEHOLDER",
                "video": "VIDEO_PLACEHOLDER"
            },
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            placeholders_separator="\n")

    def test_new_registration(self):
        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            self.model_type, self.placeholder_metadata)
        self.assertEqual(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata(
                self.model_type), self.placeholder_metadata)
        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(
            self.model_type)

    def test_registered_model_types(self):
        pre_reg_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types())

        # register the model type
        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            self.model_type, self.placeholder_metadata)

        post_reg_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types())
        self.assertEqual(
            len(pre_reg_model_types) + 1, len(post_reg_model_types))
        self.assertIn(self.model_type, post_reg_model_types)

        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(
            self.model_type)

    def test_validity(self):
        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            self.model_type, self.placeholder_metadata)
        self.assertTrue(
            MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(self.model_type, "image"))
        self.assertTrue(
            MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(self.model_type, "video"))
        self.assertFalse(
            MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(self.model_type, "audio"))

        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(
            self.model_type)

    def test_model_types_per_modality(self):
        pre_reg_image_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_image_model_types())
        pre_reg_video_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_video_model_types())
        pre_reg_audio_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_audio_model_types())

        # register the model type for image and video
        MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
            self.model_type, self.placeholder_metadata)

        post_reg_image_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_image_model_types())
        post_reg_video_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_video_model_types())
        post_reg_audio_model_types = list(
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_audio_model_types())
        self.assertEqual(
            len(pre_reg_image_model_types) + 1, len(post_reg_image_model_types))
        self.assertEqual(
            len(pre_reg_video_model_types) + 1, len(post_reg_video_model_types))
        self.assertEqual(len(pre_reg_audio_model_types),
                         len(post_reg_audio_model_types))
        self.assertIn(self.model_type, post_reg_image_model_types)
        self.assertIn(self.model_type, post_reg_video_model_types)
        self.assertNotIn(self.model_type, post_reg_audio_model_types)

        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(
            self.model_type)


class TestDeriveMmItemOrder(unittest.TestCase):
    """Tests for ``BaseMultimodalInputProcessor.derive_mm_item_order``.

    The method walks pre-expansion prompt text and returns a prompt-order
    manifest ``[{"modality": ..., "index": ...}, ...]``. It is a pure function
    of (text, placeholder strings) and does not touch ``self``, so tests call
    it as an unbound method with ``None`` self.
    """

    _IMAGE_PH = "<|image_pad|>"
    _VIDEO_PH = "<|video_pad|>"

    def _call(self,
              text,
              image_placeholder=_IMAGE_PH,
              video_placeholder=_VIDEO_PH):
        return BaseMultimodalInputProcessor.derive_mm_item_order(
            None,
            text,
            image_placeholder=image_placeholder,
            video_placeholder=video_placeholder,
        )

    def test_pure_image_two_items(self):
        # Two same-modality items → per-modality index increments from 0.
        text = f"{self._IMAGE_PH} vs {self._IMAGE_PH}"
        self.assertEqual(self._call(text), [
            {
                "modality": "image",
                "index": 0
            },
            {
                "modality": "image",
                "index": 1
            },
        ])

    def test_video_then_image_preserves_prompt_order(self):
        # Guards against any implementation that sorts by modality — output
        # must reflect the text order, not group images before videos.
        text = f"{self._VIDEO_PH} first, then {self._IMAGE_PH}"
        self.assertEqual(self._call(text), [
            {
                "modality": "video",
                "index": 0
            },
            {
                "modality": "image",
                "index": 0
            },
        ])

    def test_image_video_image_prompt_order(self):
        # Mixed sequence with a same-modality repeat: index for images
        # advances (0, 1) while video's own counter stays at 0.
        text = f"{self._IMAGE_PH} a {self._VIDEO_PH} b {self._IMAGE_PH}"
        self.assertEqual(self._call(text), [
            {
                "modality": "image",
                "index": 0
            },
            {
                "modality": "video",
                "index": 0
            },
            {
                "modality": "image",
                "index": 1
            },
        ])

    def test_empty_or_no_placeholders_returns_empty(self):
        # Covers three separate early-return paths: empty text, no
        # placeholders in text, and both placeholder strings being ``None``.
        self.assertEqual(self._call(""), [])
        self.assertEqual(self._call("hello world with no media"), [])
        self.assertEqual(
            self._call(f"{self._IMAGE_PH} + {self._VIDEO_PH}",
                       image_placeholder=None,
                       video_placeholder=None), [])

    def test_missing_video_placeholder_ignores_video_matches(self):
        # Partial ``None`` contract: only the modality with a placeholder
        # string is recognized; the other modality's occurrences are silently
        # skipped rather than raising.
        text = f"{self._IMAGE_PH} + {self._VIDEO_PH}"
        self.assertEqual(
            self._call(text, video_placeholder=None),
            [{
                "modality": "image",
                "index": 0
            }],
        )

    def test_regex_special_chars_in_placeholder_are_literal(self):
        # Placeholder strings contain ``<|...|>`` — the ``|`` is a regex
        # metacharacter that would alternate if not escaped. This asserts
        # the implementation runs the placeholders through ``re.escape``.
        text = "before <|image_pad|> after"
        self.assertEqual(self._call(text), [{"modality": "image", "index": 0}])


if __name__ == "__main__":
    unittest.main()
