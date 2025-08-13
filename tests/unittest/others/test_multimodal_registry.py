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


if __name__ == "__main__":
    unittest.main()
